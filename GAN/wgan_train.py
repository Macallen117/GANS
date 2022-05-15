import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from WGAN.wgan import *
from GAN.dataset import *

class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        batch_size,
        num_epochs,
        n_critic,
        clip_value,
        nz,
        lr,
        label
    ):
        self.n_critic = n_critic
        self.clip_value = clip_value
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.nz = nz
        self.dataset = 'real'
        self.label = label
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.netD = discriminator.to(self.device)
        self.netD.apply(weights_init)
        self.netG = generator.to(self.device)
        self.netG.apply(weights_init)

        self.fixed_noise = torch.randn(self.batch_size, self.nz, 1, device=self.device)
        self.dataloader = get_dataloader(dataset=self.dataset,label_name=self.label,batch_size=self.batch_size)
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.lr)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.lr)
    
        self.lossD = []
        self.lossG = []
    
    def _one_epoch(self):
        for step, (data, labels) in enumerate(self.dataloader):
            # training netD
            real = data.to(self.device)
            b_size = real.size(0)
            self.netD.zero_grad()

            noise = torch.randn(b_size, self.nz, 1, device=self.device)
            fake = self.netG(noise)

            loss_disc = -torch.mean(self.netD(real)) + torch.mean(self.netD(fake))
            loss_disc.backward()
            self.optimizerD.step()

            for p in self.netD.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

            if step % self.n_critic == 0:
                # training netG
                noise = torch.randn(b_size, self.nz, 1, device=self.device)
                self.netG.zero_grad()
                fake = self.netG(noise)
                loss_gen = -torch.mean(self.netD(fake))

                self.netD.zero_grad()
                self.netG.zero_grad()
                loss_gen.backward()
                self.optimizerG.step()
        return step, loss_disc.item(), loss_gen.item()

    def run(self):
        for epoch in range(self.num_epochs):
            step, loss_disc, loss_gen = self._one_epoch()
            self.lossG.append(loss_gen)
            self.lossD.append(loss_disc)

            fake = self.netG(self.fixed_noise).detach().cpu()
            real = next(iter(self.dataloader))[0]
            diff = np.array(fake.view(-1)) - np.array(real.view(-1))
            diff = np.sum(diff ** 2) / self.batch_size
            print('[%d/%d][%d/%d]\tloss_netD: %.4f\tloss_gen: %.4f\tdiff: %.4f'
                  % (epoch, self.num_epochs, step, len(self.dataloader), loss_disc, loss_gen, diff))

            # plot training process
            if epoch % 100 == 0:
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise).detach().cpu()
                    real = next(iter(self.dataloader))[0]
                    fig, ax = plt.subplots()
                    line_width = 0.5
                    ax.plot(fake[0].view(-1), label='fake', c='blue', linewidth=line_width)
                    ax.plot(real[0].view(-1), label='real', c='red', linewidth=line_width)
                    ax.legend()
                    plt.show()
            # plot the loss
            if epoch % 500 == 0:
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(self.lossG, label="G")
                plt.plot(self.lossD, label="D")
                plt.xlabel("iterations")
                plt.ylabel("Loss")
                plt.legend()
                plt.show()
            # save model
            if epoch % 500 == 0:
                netD_path = 'WGAN/netD/'
                netG_path = 'WGAN/netG/'
                netD_path = os.path.join(netD_path, 'netD_{}_{}.pt'.format(self.label,epoch))
                netG_path = os.path.join(netG_path, 'netG_{}_{}.pt'.format(self.label,epoch))
                torch.save(self.netD.state_dict(), netD_path)
                torch.save(self.netG.state_dict(), netG_path)

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    seed = 2021
    seed_everything(seed)
    
    g = Generator()
    d = Discriminator()

    trainer = Trainer(
        generator=g,
        discriminator=d,
        batch_size=3,
        num_epochs=30000,
        n_critic=5,
        clip_value=0.005,
        nz = 100,
        lr=1e-4,
        label='L005'
    )
    trainer.run()