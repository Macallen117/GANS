import torch.nn as nn


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
            self._block(1, 64, 5, 2, 1),
            # state size 737
            self._block(64, 128, 5, 2, 1),
            # state size 368
            self._block(128, 256, 4, 2, 1),
            # state size 184
            self._block(256, 512, 4, 2, 1),
            # state size 92
            nn.Conv1d(512, 1, kernel_size=92, stride=1, padding=0, bias=False)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x, y=None):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.main = nn.Sequential(
            # input size 100
            self._block(nz, 512, 92, 1, 0),
            # state size 92
            self._block(512, 256, 4, 2, 1),
            # state size 184
            self._block(256, 128, 4, 2, 1),
            # state size 368
            self._block(128, 64, 5, 2, 1),
            # state size 737
            nn.ConvTranspose1d(64, 1, kernel_size=5, stride=2, padding=1, bias=False),
            nn.Tanh()
            # output size 1475
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.main(x)
        return x