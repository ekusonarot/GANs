import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(25, 128, 3, 1, 3//2),
            nn.Conv2d(128, 256, 3, 1, 3//2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(64, 128, 3, 1, 3//2),
                nn.Conv2d(128, 256, 3, 1, 3//2),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            for _ in range(4)
        ])
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 16, 3, 1, 0),
            nn.Conv2d(16, 1, 3, 1, 0),
            nn.Tanh()
        )
    
    def forward(self, num):
        noise = torch.randn(size=(num, 100, 1, 1)).to(device)
        x = self.conv1(noise)
        for conv in self.conv2:
            x = conv(x)
        x = self.conv3(x)
        return torch.clamp(x, 0., 1.)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, 5, 1, 5//2),
            nn.Conv2d(256, 128, 3, 1, 3//2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 3//2),
                nn.Conv2d(64, 128, 3, 2, 3//2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU()
            )
            for _ in range(3)
        ])
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 16, 3, 1, 3//2),
            nn.Conv2d(16, 1, 3, 1, 3//2),
            nn.AdaptiveAvgPool2d((1,1))
        )
    
    def forward(self, x):
        x = self.conv1(x)
        for conv in self.conv2:
            x = conv(x)
        x = self.conv3(x)
        return x