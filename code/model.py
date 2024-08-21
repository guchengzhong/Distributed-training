import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1= nn.Sequential(
            nn.Conv2d(1, 32, kernel_size= 3, padding= 1, stride= 2),  # (b, 1, 28, 28)>> (b, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride= 2)  # (b, 32, 14, 14)>> (b, 32, 7, 7)
        )
        self.layer2= nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= 3, padding= 1, stride= 2),  # (b, 32, 7, 7)>> (b, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride= 2)  # (b, 64, 4, 4)>> (b, 64, 2, 2)
        )
        self.fc= nn.Sequential(
            nn.Linear(64* 2* 2, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(self.layer2(self.layer1(x)).view(x.shape[0], -1))