import torch.nn as nn
import torchvision.datasets as dataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input size 1475
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 737
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 368
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 184
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 92
            nn.Conv1d(512, 1, kernel_size=92, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # output size 1
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.main = nn.Sequential(
            # input size 100
            nn.ConvTranspose1d(nz, 512, kernel_size=92, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # state size 92
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # state size 184
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # state size 368
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # state size 737
            nn.ConvTranspose1d(64, 1, kernel_size=5, stride=2, padding=1, bias=False),
            nn.Tanh()
            # output size 1475
        )

    def forward(self, x):
        x = self.main(x)
        return x