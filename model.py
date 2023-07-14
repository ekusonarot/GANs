import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import normal

generaters = {
    "normal": normal.Generator
}
discriminators = {
    "normal": normal.Discriminator
}
class GAN(nn.Module):
    def __init__(self, device):
        super(GAN, self).__init__()
        self.device = device
    
    def load(self, name, lr, dis_iter):
        self.gen = generaters[name]()
        self.dis = discriminators[name]()
        self.gen_opt = Adam(self.gen.parameters(), lr=lr)
        self.dis_opt = Adam(self.dis.parameters(), lr=lr)
        self.dis_iter = dis_iter
    
    def names(self):
        return list(generaters.keys())

    def forward(self, noise):
        return self.gen(noise)
    
    def one_epoch(self, x):
        # discriminator update
        for _ in range(self.dis_iter):
            self.dis_opt.zero_grad()
            self.gen_opt.zero_grad()
            noise = torch.randn(size=(x.shape[0], 100, 1, 1)).to(self.device)
            real = self.dis(x)
            fake = self.dis(self.gen(noise))
            dis_loss = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
            dis_loss.backward()
            self.dis_opt.step()
        # generator update
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()
        noise = torch.randn(size=(x.shape[0], 100, 1, 1)).to(self.device)
        fake = self.dis(self.gen(noise))
        gen_loss = -fake.mean()
        gen_loss.backward()
        self.gen_opt.step()
        