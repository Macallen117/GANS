from __future__ import print_function
import os
import csv
import numpy as np
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib import ticker
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import matplotlib.animation as animation
from IPython.display import HTML

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            # nn.Linear(in_features, 512),
            # nn.LeakyReLU(0.2),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 1),

            nn.Linear(in_features, 25),
            nn.LeakyReLU(0.2),
            nn.Linear(25, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim):
        super().__init__()
        self.gen = nn.Sequential(
            # nn.Linear(z_dim, 32),
            # nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(32, momentum= 0.8),
            # nn.Linear(32, 64),
            # nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(64, momentum=0.8),
            # nn.Linear(64, output_dim),

            nn.Linear(z_dim, 15),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(15),
            nn.Linear(15, output_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

z_dim = 5
input_dim = 2
output_dim = 2
num_epochs = 500

#initialize disc and gen
disc = Discriminator(input_dim).to(device)
gen = Generator(z_dim, output_dim).to(device)

# set optimizer to adam
lr = 3e-4 # learning rate works well for adam

optimizerD = optim.Adam(disc.parameters(), lr=lr)
optimizerG = optim.Adam(gen.parameters(), lr=lr)

# use BCE loss
criterion = nn.BCELoss()

# Lists to keep track of progress
G_losses = []
D_losses = []

# directory to load real data
os.chdir('../dataset/headerRemoved')

for epoch in range(num_epochs):
    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files
        print('open ' + csvFilename + '...')

        # Read the CSV file
        real = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)
        for row in readerObj:
            x = float(row[0])
            y = float(row[1])
            real.append([x, y])
        csvFileObj.close()

        real = torch.Tensor(real).to(device)
        batch_size = real.shape[0]

        # ###########################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # ##########################

        # loss of batch_size of real data
        output = disc(real).view(-1)                                    # batch_size * [0,1]
        D_x = output.mean().item()                                      # D(x)
        lossD_real = criterion(output, torch.ones_like(output))         # 1 because its real

        # loss of batch_size of fake data
        noise = torch.randn(batch_size, z_dim).to(device)               # batch_size * latent_size
        fake = gen(noise)                                               # batch_size * 2
        output = disc(fake).view(-1)                                    # batch_size * [0,1]
        D_G_z1 = output.mean().item()                                   # D(G(z1))
        lossD_fake = criterion(output, torch.zeros_like(output))        # 0 because its fake

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)                   # Calculate gradients for D
        optimizerD.step()                                   # update D

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from saturating gradients
        ###########################

        output = disc(fake).view(-1)                        # batch_size * [0,1]
        D_G_z2 = output.mean().item()                       # D(G(z2))
        lossG = criterion(output, torch.ones_like(output))  # 1 try to let generator think its true
        gen.zero_grad()
        lossG.backward()                                    # Calculate gradients for G
        optimizerG.step()                                   # update G

        # Save Losses for plotting later
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f'
                  % (epoch, num_epochs, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

        with torch.no_grad():
            fixed_noise = torch.randn((batch_size, z_dim)).to(device)
            fake = gen(fixed_noise)

            # Plot the real images
            ax1 = plt.subplot(1, 2, 1)
            x_real, y_real =[row[0] for row in real.cpu()], [row[1] for row in real.cpu()]
            ax1.plot(x_real, y_real)
            plt.title("Real Images")
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))

            # Plot the fake images
            ax2 = plt.subplot(1, 2, 2)
            x_fake, y_fake = [row[0] for row in fake.cpu()], [row[1] for row in fake.cpu()]
            ax2.plot(x_fake, y_fake)
            plt.title("Fake Images")
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.show()

    # plot the loss
    if(epoch % 5 == 0):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
