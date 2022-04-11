import os
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from wgan import Discriminator, Generator, weights_init
from preprocessing import Dataset

n_critic = 5
clip_value = 0.01
lr = 5e-5
epoch_num = 4001
batch_size = 7
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # load training data
    os.chdir('../dataset/headerRemoved')
    trainset = Dataset('../dataset/headerRemoved')

    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # init critic and gen
    critic = Discriminator().to(device)
    critic.apply(weights_init)

    gen = Generator(nz).to(device)
    gen.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(4, nz, 1, device=device)

    # optimizers
    optimizer_critic = optim.RMSprop(critic.parameters(), lr=lr)
    optimizer_gen = optim.RMSprop(gen.parameters(), lr=lr)

    lossD = []
    lossG = []

    for epoch in range(epoch_num):
        for step, (data, _) in enumerate(dataloader):
            # training critic
            real = data.to(device)
            b_size = real.size(0)
            critic.zero_grad()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = gen(noise)

            loss_critic = -torch.mean(critic(real)) + torch.mean(critic(fake))
            loss_critic.backward()
            optimizer_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if step % n_critic == 0:
                # training gen
                noise = torch.randn(b_size, nz, 1, device=device)
                gen.zero_grad()
                fake = gen(noise)
                loss_gen = -torch.mean(critic(fake))

                critic.zero_grad()
                gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()

            print('[%d/%d][%d/%d]\tloss_critic: %.4f\tloss_gen: %.4f'
                  % (epoch, epoch_num, step, len(dataloader), loss_critic.item(), loss_gen.item()))

            # Save Losses for plotting later
            lossG.append(loss_gen.item())
            lossD.append(loss_critic.item())

            # save training process every 20 epochs
            if epoch % 50 == 0 and step == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                    real = next(iter(dataloader))[0]
                    f, a = plt.subplots(2, 2, figsize=(16, 16))

                    for j in range(2):
                        a[0][j].set_title('Fake Image', fontsize=20, fontweight='bold',
                                          color='#30302f', loc='center')
                        a[0][j].plot(fake[j].view(-1))
                        a[0][j].set_xticks(())
                        a[0][j].set_yticks(())

                    for j in range(2):
                        a[1][j].set_title('Real Image', fontsize=20, fontweight='bold',
                                          color='#30302f', loc='center')
                        a[1][j].plot(real[j].view(-1))
                        a[1][j].set_xticks(())
                        a[1][j].set_yticks(())
                    plt.show()

            # plot the loss every 50 epochs
            if epoch % 50 == 0 and step == 0:
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(lossG, label="G")
                plt.plot(lossD, label="D")
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.show()
if __name__ == '__main__':
    main()