import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
import sklearn.datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import optuna
from optuna.trial import TrialState

from WGAN.model import *
from GAN.dataset import *
from config import *

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
        self.best_diff = 10000.0

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
        for epoch in range(self.num_epochs + 1):
            step, loss_disc, loss_gen = self._one_epoch()
            self.lossG.append(loss_gen)
            self.lossD.append(loss_disc)

            fake = self.netG(self.fixed_noise).detach().cpu()
            real = next(iter(self.dataloader))[0]

            diff = np.array(fake.view(-1)) - np.array(real.view(-1))
            diff = np.sum(diff ** 2) / self.batch_size
            print('[%d/%d][%d/%d]\tloss_netD: %.4f\tloss_netG: %.4f\tdiff: %.4f'
                  % (epoch, self.num_epochs, step, len(self.dataloader), loss_disc, loss_gen, diff))

            # plot training process and save model
            if diff < 1 and diff <= self.best_diff:
                with torch.no_grad():
                    fake = self.netG(self.fixed_noise).detach().cpu()
                    real = next(iter(self.dataloader))[0]
                    fig, ax = plt.subplots()
                    ax.set_title(f'{self.label}{" mean square error"} (loss: {np.around(diff, 2)})')
                    line_width = 0.5
                    ax.plot(fake[0].view(-1), label='fake', c='blue', linewidth=line_width)
                    ax.plot(real[0].view(-1), label='real', c='red', linewidth=line_width)
                    ax.legend()
                    plt.show()

                    self.best_diff = diff
                    netD_path = 'WGAN/netD/'
                    netG_path = 'WGAN/netG/'
                    netD_path = os.path.join(netD_path, 'netD_{}.pt'.format(self.label))
                    netG_path = os.path.join(netG_path, 'netG_{}.pt'.format(self.label))
                    torch.save(self.netD.state_dict(), netD_path)
                    torch.save(self.netG.state_dict(), netG_path)


        print(self.best_diff)
        # plot the loss
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.lossG, label="G")
        plt.plot(self.lossD, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    config = Config()
    seed_everything(config.seed)
    nz = 100

    g = Generator(nz)
    d = Discriminator()

    trainer = Trainer(
        generator=g,
        discriminator=d,
        batch_size=12,
        num_epochs=6000,
        n_critic=5,
        clip_value=0.005,
        nz = nz,
        lr=5e-4,
        label='N'
    )
    trainer.run()