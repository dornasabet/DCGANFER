import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            self.block(features, 2 * features, 4, 2, 1),
            self.block(2 * features, 4 * features, 4, 2, 1),
            self.block(4 * features, 8 * features, 4, 2, 1),
            nn.Conv2d(8 * features, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        blck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        return blck

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, noise_channel, img_channel, features=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.block(noise_channel, features * 8, 4, 1, 0),
            self.block(features * 8, features * 4, 4, 2, 1),
            self.block(features * 4, features * 2, 4, 2, 1),
            self.block(features * 2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, img_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def block(self, in_channel, out_channel, kernel_size, stride, padding):
        blck = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        return blck

    def forward(self, x):
        return self.gen(x)


def weight_initialize(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


v = 10
