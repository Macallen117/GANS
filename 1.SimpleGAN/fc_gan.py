from __future__ import print_function
import numpy as np
import argparse
import os
import random
import torch
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
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum= 0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 500
save_interval = 10

#initialize disc and gen
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# normalize the initial MNIST dataset
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# set optimizer to adam
lr = 3e-4 # learning rate works well for adam
optimizerD = optim.Adam(disc.parameters(), lr=lr)
optimizerG = optim.Adam(gen.parameters(), lr=lr)

# use BCE loss
criterion = nn.BCELoss()

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        real = real.view(-1, 784).to(device)                            # batch_size * 784
        batch_size = real.shape[0]

        # loss of batch_size of real data
        output = disc(real).view(-1)                                    # batch_size * [0,1]
        D_x = output.mean().item()                                      # D(x)
        lossD_real = criterion(output, torch.ones_like(output))         # 1 because its real

        # loss of batch_size of fake data
        noise = torch.randn(batch_size, z_dim).to(device)               # batch_size * 100
        fake = gen(noise)                                               # batch_size * 784
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

        # Output training stats
        if batch_idx % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f'
                  % (epoch, num_epochs, batch_idx, len(dataloader),
                     lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        # Save Losses for plotting later
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (batch_idx == len(dataloader) - 1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
                fake = fake.view(32, 1, 28, 28)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # Plot the real images
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=5, normalize=True).cpu(),
                                    (1, 2, 0)))

            # Plot the fake images from the last epoch
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            vutils.save_image(vutils.make_grid(fake.to(device)[:32]),  '%s/fake_samples_epoch_%03d.png' %
            ('./dataset/result', epoch),
                normalize=True)
            plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:32], padding=5, normalize=True).cpu(),
                                    (1, 2, 0)))
            plt.show()
        iters += 1
