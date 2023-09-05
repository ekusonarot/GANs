import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random

import normal
import sngan

generaters = {
    "normal": normal.Generator,
    "sngan": sngan.Generator
}
discriminators = {
    "normal": normal.Discriminator,
    "sngan": sngan.Discriminator
}
class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
    
    def load(self, name, lr, dis_iter, gradient_penalty):
        self.gen = generaters[name]()
        self.dis = discriminators[name]()
        self.gen_opt = Adam(self.gen.parameters(), lr=lr)
        self.dis_opt = Adam(self.dis.parameters(), lr=lr)
        self.dis_iter = dis_iter
        self.gradient_penalty = gradient_penalty
    
    def names(self):
        return list(generaters.keys())

    def forward(self, num):
        return self.gen(num)
    
    def one_epoch(self, x):
        # discriminator update
        for _ in range(self.dis_iter):
            self.dis_opt.zero_grad()
            self.gen_opt.zero_grad()
            real = self.dis(x)
            x_ = self.gen(x.shape[0])
            fake = self.dis(x_)
            dis_loss = F.relu(1 - real).mean() + F.relu(1 + fake).mean()
            if self.gradient_penalty:
                r = random.uniform(0, 1)
                x = (x*r+x_*(1-r)).clone().detach().requires_grad_(True)
                out = self.dis(x)
                out = torch.norm(out)
                gp = torch.autograd.grad(out, x, create_graph=True)
                gp = gp[0].mean()
                gp = (gp-1)**2
                dis_loss += 10.*gp
            dis_loss.backward()
            self.dis_opt.step()
        # generator update
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()
        fake = self.dis(self.gen(x.shape[0]))
        gen_loss = -fake.mean()
        gen_loss.backward()
        self.gen_opt.step()
        