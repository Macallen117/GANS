import os
import time

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam

from gan import Generator, Discriminator
from dataset import Dataset, get_dataloader

class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        batch_size,
        num_epochs,
        lr,
        label
    ):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.netG = generator.to(self.device)
        self.netD = discriminator.to(self.device)
        self.lr = lr
        self.optimizerD = Adam(self.netD.parameters(), lr=self.lr)
        self.optimizerG = Adam(self.netG.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()
        
        self.batch_size = batch_size
        self.signal_dim = [self.batch_size, 1, 1475]
        self.num_epochs = num_epochs
        self.dataloader = get_dataloader(
            label_name=label, batch_size=self.batch_size
        )
        self.fixed_noise = torch.randn(self.batch_size, 1, 1475,
                                       device=self.device)
        self.lossG = []
        self.lossD = []
        
    def _one_epoch(self):
        real_label = 1
        fake_label = 0
        
        for i, data in enumerate(self.dataloader, 0):
            ##### Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) #####
            ## train with real data
            self.netD.zero_grad()
            real_data = data[0].to(self.device)
            # dim for noise
            batch_size = real_data.size(0)
            self.signal_dim[0] = batch_size
            
            label = torch.full((batch_size,), real_label,
                           dtype=real_data.dtype, device=self.device)

            output = self.netD(real_data)
            output = output.view(-1)

            loss_disc_real = self.criterion(output, label)
            loss_disc_real.backward()
            D_x = output.mean().item()
            
            ## train with fake data
            noise = torch.randn(self.signal_dim, device=self.device)
            fake = self.netG(noise)
            label.fill_(fake_label)
            
            output = self.netD(fake.detach())
            output = output.view(-1)
            
            loss_disc_fake = self.criterion(output, label)
            loss_disc_fake.backward()
            D_G_z1 = output.mean().item()
            loss_disc = loss_disc_real + loss_disc_fake 
            self.optimizerD.step()
            
            ##### Update Generator: maximaze log(D(G(z)))  
            self.netG.zero_grad()
            label.fill_(real_label) 
            output = self.netD(fake)
            output = output.view(-1)
            
            loss_gen = self.criterion(output, label)
            loss_gen.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()
            
        return loss_disc.item(), loss_gen.item()
        
    def run(self):
        for epoch in range(self.num_epochs):
            loss_disc, loss_gen = self._one_epoch()
            self.lossD.append(loss_disc)
            self.lossG.append(loss_gen)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss_D: {loss_disc} | Loss_G: {loss_gen} | Time: {time.strftime('%H:%M:%S')}")
   
                fake = self.netG(self.fixed_noise)
                real = next(iter(self.dataloader))[0]
                plt.plot(fake.detach().cpu().squeeze(1).numpy()[0].transpose())
                plt.plot(real[0].view(-1))
                plt.show()
            
        # torch.save(self.netG.state_dict(), f"generator.pth")
        # torch.save(self.netG.state_dict(), f"discriminator.pth")
               
if __name__ == '__main__':
    g = Generator()
    d = Discriminator()
                      
    trainer = Trainer(
      generator=g,
      discriminator=d,
      batch_size=3,
      num_epochs=30000,
      lr = 5e-5,
      label='L005'
    )
    trainer.run()
