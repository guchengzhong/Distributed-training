import os  # parameter registration
import time  # record time
import torch
import argparse  # setting parameters
import torch.nn as nn
from model import ConvNet  # loading model
import torch.distributed as dist
from torch.cuda.amp import GradScaler  # mixed precision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from dl import load_mnist_data, split_data, EarlyStop  # loading data
from torch.utils.data.distributed import DistributedSampler  # distributed data sampler

def prepare():
    parser= argparse.ArgumentParser()
    parser.add_argument('--gpu', default= '2,3')
    parser.add_argument('--epochs', type= int, default= 10000, help= 'epoch number')
    parser.add_argument('--lr', type= float, default= 1e-4, help= 'learning rate')
    parser.add_argument('--weight_decay', type= float, default= 5e-5, help= 'weight decay')
    parser.add_argument('--batch_size', type= int, default= 256, help= 'batch size number')
    parser.add_argument('--mnist_file_path', type= str, default= '../data/mnist_test.csv')
    parser.add_argument('--seed', type= int, default= 1206)
    parser.add_argument('--master_addr', type= str, default= 'localhost')
    parser.add_argument('--port', type= str, default= '1206')
    parser.add_argument('--pt_path', type= str, default= '../pt/')
    parser.add_argument('--patience', type= int, default= 5)
    args= parser.parse_args()
    # setting environment variables to enable DDP
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu  # visible gpu devices
    return args

def init_ddp(local_rank):
    # after this setup, tensors can be moved to GPU via 'a= a.cuda()' rather than 'a= a.to(local_rank)'
    torch.cuda.set_device(local_rank)
    os.environ['RANK']= str(local_rank)
    dist.init_process_group(backend= 'nccl', init_method= 'env://')  # init communication mode

# generate a torch generator, which is used for randomness operations during the sampling process, such
# as randomly selecting data samples.
def get_ddp_generator(seed= 1206):
    local_rank= dist.get_rank()
    g= torch.Generator()
    g.manual_seed(seed+ local_rank)
    return g

def train(epoch, net, train_loader, loss_fn, opt, scaler):
    '''
    parameters:
        scaler, Part of the Automatic Mixed Precision (AMP) library. AMP allows you to store tensors with lower precision (
                such as half precision floating-point numbers, i.e. torch. float 16) when training deep learning models, while
                avoiding numerical underflow or overflow issues with higher precision (such as full precision floating-point 
                numbers, i.e. torch. float 32) when calculating gradients.
    '''
    net.train()
    for i, data in enumerate(train_loader):
        imgs= data[0].cuda()
        lbls= data[1].cuda()
        outputs= net(imgs)
        loss= loss_fn(outputs, lbls)
        opt.zero_grad()
        scaler.scale(loss).backward()  # avoiding the problem of overflow caused by numbers that are difficult to represent with float16
        scaler.step(opt)
        # The main purpose of the 'scaler. update()' operation is to adjust the threshold of the scale factor based on
        # the most recent gradient update, in order to ensure more effective management of the gradient range in subse
        # quent iterations and avoid the problem of gradient vanishing or exploding.
        scaler.update()
        print(f'epoch: {epoch+ 1}, step: {i+ 1}, loss: {loss.item()}, device: {dist.get_rank()}', flush= True)

def valid(net, valid_loader):
    net.eval()
    num= torch.tensor(0.0).cuda()
    correct= torch.tensor(0.0).cuda()
    for imgs, lbls in valid_loader:
        imgs= imgs.cuda()
        lbls= lbls.cuda()
        with torch.no_grad():
            outputs= net(imgs)
            num+= outputs.shape[0]
        correct+= (outputs.argmax(1)== lbls).type(torch.float).sum()
    dist.reduce(num, dst= 0, op= dist.ReduceOp.SUM)  # All reduce on num
    dist.reduce(correct, dst= 0, op= dist.ReduceOp.SUM) # All reduce
    return correct* 1.0/ num

def main(args):

    local_rank= int(os.environ['LOCAL_RANK'])
    # init ddp
    init_ddp(local_rank)
    # load data
    lbls, imgs= load_mnist_data(args.mnist_file_path)
    lbls= lbls.long()
    imgs= imgs.float().view(-1, 1, 28, 28)
    lbl_tr, lbl_val, lbl_te, img_tr, img_val, img_te= split_data(args.seed, lbls, imgs, 0.8, 0.1)
    dataset_tr, dataset_val, dataset_te= TensorDataset(img_tr, lbl_tr), TensorDataset(img_val, lbl_val), TensorDataset(img_te, lbl_te)
    train_sampler, val_sampler, test_sampler= DistributedSampler(dataset_tr), DistributedSampler(dataset_val), DistributedSampler(dataset_te)
    # generator control random behavior (such as data shuffling) to ensure that the order of data on each GPU is different but controllable.
    train_loader= DataLoader(dataset= dataset_tr, batch_size= args.batch_size, shuffle= False, sampler= train_sampler, generator= get_ddp_generator())
    valid_loader= DataLoader(dataset= dataset_val, batch_size= args.batch_size, shuffle= False, sampler= val_sampler)
    test_loader= DataLoader(dataset= dataset_te, batch_size= args.batch_size, shuffle= False, sampler= test_sampler)
    # init net
    net= ConvNet().cuda()
    net= nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net= nn.parallel.DistributedDataParallel(net, device_ids= [local_rank])
    loss_fn= nn.CrossEntropyLoss().cuda()
    opt= torch.optim.SGD(net.parameters(), args.lr)
    scaler= GradScaler()  # mixed precision training
    earlystopping= EarlyStop(args.pt_path, 'cnn.pt', args.patience)
    stop_flag_tensor= torch.tensor([0], dtype= torch.long).cuda()
    # train
    for epoch in range(args.epochs):
        # changing the sampling starting point to ensure that the order of data in each round is different, the model's generalization ability can be improved.        
        train_loader.sampler.set_epoch(epoch)
        train(epoch, net, train_loader, loss_fn, opt, scaler)
        # valid
        acc= valid(net, valid_loader)
        if local_rank== 0:
            earlystopping.check_early_stopping(-acc, net, opt, scaler)
            print(f'epoch: {epoch}, early_stopping.time: {earlystopping.time}, acc: {acc}')
            stop_flag_tensor.fill_(int(earlystopping.flag))
        dist.all_reduce(stop_flag_tensor, op= dist.ReduceOp.SUM)
        if stop_flag_tensor> 0:break
    # destory the process group
    dist.destroy_process_group()

if __name__== '__main__':
    '''
    start mode:
        torchrun --standalone --nproc_per_node=2 /data_new/gyh/240817ParallelTrain/code/main4ddp_torchrun.py 1> result.txt 2>err.txt
    parameters:
        standalone,
            Standalone means running distributed training using single node mode. In this mode, communication between nodes will use
            an automatic configuration method without the need for users to set masteraddr and master_port. In multi node training, 
            it is usually necessary to specify the IP address and port number of the primary node, but in single node mode, these 
            configurations are automatically processed. Therefore, standalone is suitable for running distributed training on a single
            machine (single node).
        nproc_per_node,
            Specify the number of processes to be started for each node. Usually, this value is equal to the number of GPUs available 
            on each node. Each process is typically bound to a GPU for parallel training.
    '''
    args= prepare()
    time_start= time.time()
    main(args)
    time_end= time.time()
    cost_time= time_end- time_start
    local_rank= int(os.environ['LOCAL_RANK'])
    if local_rank== 0:
        print(f'cost time: {cost_time:.2f} seconds')