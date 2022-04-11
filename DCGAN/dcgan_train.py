import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dcgan import Discriminator, Generator, weights_init
from preprocessing import Dataset


lr = 1e-4
beta1 = 0.5
epoch_num = 500
batch_size = 11
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # load training data
    os.chdir('../dataset/headerRemoved')
    trainset = Dataset('../dataset/headerRemoved')

    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )

    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)


    criterion = nn.BCELoss()

    # used for visualzing training process
    fixed_noise = torch.randn(4, nz, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # optimizerD = optim.Adam(netD.parameters(), lr=lr)
    # optimizerG = optim.Adam(netG.parameters(), lr=lr)
    
    lossD = []
    lossG = []
    
    for epoch in range(epoch_num):
        for step, (data, _) in enumerate(dataloader):

            real = data.to(device)  # torch.Size([batch_size, 1, 1475])
            b_size = real.size(0)   # batch_size

            # train netD
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)
            netD.zero_grad()
            output = netD(real).view(-1)
            loss_D_real = criterion(output, label)
            loss_D_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            loss_D_fake = criterion(output, label)
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()
            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()

            # train netG
            netG.zero_grad()
            label.fill_(real_label)
            fake = netG(noise)
            output = netD(fake).view(-1)
            loss_G = criterion(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epoch_num, step, len(dataloader),
                     loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            lossG.append(loss_G.item())
            lossD.append(loss_D.item())

            # save training process every 5 epochs
            if epoch % 20 == 0 and step == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
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
            # plt.savefig('../images/result/dcgan_epoch_%d.png' % epoch)
            #plt.close()

if __name__ == '__main__':
    main()
