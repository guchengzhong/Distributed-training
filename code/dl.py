import torch
import numpy as np
import matplotlib.pyplot as plt

# load mnist
def load_mnist_data(path):
    df= np.asfarray([line.split(',') for line in np.loadtxt(path, dtype= str)])
    lbl, df= df[:, 0], df[:, 1:]
    return torch.from_numpy(lbl), torch.from_numpy(df/ 255.0)

# split data into train, valid, and test data
def split_data(seed, lbl, data, train_ratio, valid_ratio):
    torch.manual_seed(seed)
    sd= torch.rand((len(lbl), ))
    train_flag, valid_flag, test_flag= sd< train_ratio, (sd>= train_ratio)* (sd< (train_ratio+ valid_ratio)), sd>= train_ratio+ valid_ratio
    return lbl[train_flag], lbl[valid_flag], lbl[test_flag], data[train_flag], data[valid_flag], data[test_flag]

class EarlyStop():
    def __init__(self, pt_path, model_name, patience= 10, init_val= 0):
        self.save_pt_path= pt_path
        self.record_val= init_val
        self.patience= patience
        self.model_name= model_name
        self.time= 0
        self.flag= False

    def check_early_stopping(self, val, model, opt, scaler= None, milestone= 0):
        if val< self.record_val:
            self.time= 0
            self.record_val= val
            self.save(model, opt, scaler, milestone)
        else:
            self.time+= 1
            if self.time>= self.patience:
                self.flag= True

    def save(self, model, opt, scaler, milestone):
        data= {}
        if scaler== None:
            data= {
                'model': model.state_dict(),
                'opt': opt.state_dict()
            }
        else:
            data= {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scaler': scaler.state_dict()
            }
        if milestone== '':
            torch.save(data, f'{self.save_pt_path}/{self.model_name}.pt')
        else:
            torch.save(data, f'{self.save_pt_path}/{self.model_name}-{milestone}.pt')

def show_sample(img_hats, imgs, save_path= '../result/result.pdf', show_flag= False):
    fig, axes= plt.subplots(nrows= 2, ncols= img_hats.shape[0], figsize=(10, 5))
    for i in range(img_hats.shape[0]):
        axes[0, i].imshow(img_hats[i, :, :], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(imgs[i, :, :], cmap='gray')
        axes[1, i].axis('off')
    plt.subplots_adjust(wspace= 0.05, hspace= -0.75)
    plt.savefig(save_path)
    if show_flag:plt.show()