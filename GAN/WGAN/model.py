import torch
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
        self.conv_1 = self._block(1, 32, 5, 2, 1)
        self.conv_2 = self._block(32, 64, 5, 2, 1)
        self.conv_3 = self._block(64, 128, 4, 2, 1)
        self.conv_4 = self._block(128, 256, 4, 2, 1)
        self.conv_5 = self._block(256, 512, 4, 2, 1)
        self.conv_6 = nn.Conv1d(512, 1, kernel_size=10, stride=1, padding=0, bias=False)

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
        output = x.permute(0, 2, 1)
        output = self.conv_1(output)
        # print(output.shape)
        output = self.conv_2(output)
        # print(output.shape)
        output = self.conv_3(output)
        # print(output.shape)
        output = self.conv_4(output)
        # print(output.shape)
        output = self.conv_5(output)
        # print(output.shape)
        output = self.conv_6(output)
        # print(output.shape)
        return output

class Generator(nn.Module):
    def __init__(self, nz=100):
        self.nz = nz
        super().__init__()
        self.conv_1 = self._block(self.nz, 512, 10, 1, 0)
        self.conv_2 = self._block(512, 256, 5, 2, 1)
        self.conv_3 = self._block(256, 128, 5, 2, 1)
        self.conv_4 = self._block(128, 64, 5, 2, 1)
        self.conv_5 = self._block(64, 32, 5, 2, 1)
        self.conv_6 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.sigmoid =  nn.Sigmoid()

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
        output = self.conv_1(x)
        # print(output.shape)
        output = self.conv_2(output)
        # print(output.shape)
        output = self.conv_3(output)
        # print(output.shape)
        output = self.conv_4(output)
        # print(output.shape)
        output = self.conv_5(output)
        # print(output.shape)
        output = self.conv_6(output)
        # print(output.shape)
        output = self.sigmoid(output)
        # print(output.shape)
        output = output.permute(0, 2, 1)
        # print(output.shape)
        return output


class DiscriminatorTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = self._block(1, 32, 5, 2, 1)
        self.conv_1 = self._block(32, 64, 5, 2, 1)
        self.conv_2 = self._block(64, 128, 4, 2, 1)
        self.conv_3 = self._block(128, 256, 4, 2, 1)
        self.conv_4 = self._block(256, 512, 4, 2, 1)
        self.conv_5 = self._block(512, 1024, 4, 2, 1)
        self.conv_6 = nn.Conv1d(1024, 1, kernel_size=5, stride=2, padding=0, bias=False)

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
        output = x.permute(0, 2, 1)
        output = self.conv_0(output)
        # print(output.shape)
        output = self.conv_1(output)
        # print(output.shape)
        output = self.conv_2(output)
        # print(output.shape)
        output = self.conv_3(output)
        # print(output.shape)
        output = self.conv_4(output)
        # print(output.shape)
        output = self.conv_5(output)
        # print(output.shape)
        output = self.conv_6(output)
        # print(output.shape)
        return output

class GeneratorTest(nn.Module):
    def __init__(self, nz=100):
        self.nz = nz
        super().__init__()
        self.conv_0 = self._block(self.nz, 1024, 5, 2, 0)
        self.conv_1 = self._block(1024, 512, 4, 2, 1)
        self.conv_2 = self._block(512, 256, 5, 2, 1)
        self.conv_3 = self._block(256, 128, 5, 2, 1)
        self.conv_4 = self._block(128, 64, 5, 2, 1)
        self.conv_5 = self._block(64, 32, 5, 2, 1)
        self.conv_6 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.sigmoid =  nn.Sigmoid()

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
        # print(x.shape)
        output = self.conv_0(x)
        # print(output.shape)
        output = self.conv_1(output)
        # print(output.shape)
        output = self.conv_2(output)
        # print(output.shape)
        output = self.conv_3(output)
        # print(output.shape)
        output = self.conv_4(output)
        # print(output.shape)
        output = self.conv_5(output)
        # print(output.shape)
        output = self.sigmoid(self.conv_6(output))
        # print(output.shape)
        output = output.permute(0, 2, 1)
        # print(output.shape)
        return output

if __name__ == '__main__':
    noise = torch.randn(12, 100, 1)
    g = GeneratorTest(100)
    d = DiscriminatorTest()
    gen = g(noise)
    d(gen)