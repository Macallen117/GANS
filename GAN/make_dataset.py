import os
import random

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from WGAN.model import Generator
from GAN.dataset import get_dataloader
from GAN.config import Config


def showGeneratedFrame(label_name, ax, line_width, color):
    dataloader = get_dataloader(dataset='fake', label_name=label_name, batch_size=32)
    data, labels = next(iter(dataloader))
    rgb = (random.random(), random.random(), random.random())
    ax.plot(data[5].view(-1), label=label_name, c=color, linewidth=line_width)


def showOriginalFrame(label_name, ax, line_width, color):
    dataloader = get_dataloader(dataset='real', label_name=label_name, batch_size=32)
    data, labels = next(iter(dataloader))
    rgb = (random.random(), random.random(), random.random())
    ax.plot(data[5].view(-1), label=label_name, c=color, linewidth=line_width)

def generateFakeImg(label_name, batch_size):
    dataloader = get_dataloader(dataset='fake', label_name=label_name, batch_size=batch_size)
    fake, label = next(iter(dataloader))
    fake = fake.squeeze()
    global num
    train_num = 0.8 * batch_size

    if label_name.startswith('L'):
        train_img_path = config.train_img_path+'/L'
        val_img_path = config.val_img_path+'/L'
    elif label_name.startswith('N'):
        train_img_path = config.train_img_path+'/N'
        val_img_path = config.val_img_path+'/N'
    elif label_name.startswith('R'):
        train_img_path = config.train_img_path+'/R'
        val_img_path = config.val_img_path+'/R'

    for count, i in enumerate(fake):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        if count < train_num:
            filename = train_img_path + '/' + str(num) + '.png'
        else:
            filename = val_img_path + '/' + str(num) + '.png'
        num = num + 1
        print(filename)
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)

def generateRealImg(label_name, batch_size):
    dataloader = get_dataloader(dataset='real', label_name=label_name, batch_size=batch_size)
    real, labels = next(iter(dataloader))
    real = real.squeeze()
    global num

    if label_name.startswith('L'):
        test_img_path = config.test_img_path+'/L'
    elif label_name.startswith('N'):
        test_img_path = config.test_img_path+'/N'
    elif label_name.startswith('R'):
        test_img_path = config.test_img_path+'/R'

    for count, i in enumerate(real):
        fig = plt.figure(frameon=False)
        plt.plot(i)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        filename = test_img_path + '/' + str(num) + '.png'
        num = num + 1
        print(filename)
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)

def generateData(label_name, batch_size, path):
    netG_path = 'WGAN/netG/'
    netG_path = os.path.join(netG_path, 'netG_{}.pt'.format(label_name))
    g.load_state_dict(torch.load(netG_path))

    dataloader = get_dataloader(dataset='real', label_name=label_name, batch_size=32)
    real = next(iter(dataloader))[0]

    count = 0
    fig, (ax1, ax2) = plt.subplots(2, sharey=True, sharex=True)
    while count < batch_size:
        fixed_noise = torch.randn(1, nz, 1)
        fake = g(fixed_noise).to(config.device)

        diff_sum = 0
        for i in range(real.shape[0]):
            diff = np.array(fake.detach().cpu().view(-1)) - np.array(real[i].detach().cpu().view(-1))
            diff = diff ** 2
            diff_sum += diff
        diff_sum = np.sum(diff_sum) / real.shape[0]
        if(diff_sum < 0.7):
            print(diff_sum)
            count += 1

            ax1.plot(fake.detach().cpu().view(-1), label=label_name, c = 'r', linewidth=0.5)

            target_id = config.class_to_id[label_name]
            target_id = torch.tensor(target_id, device=config.device)
            fake = torch.cat((fake, target_id.repeat(1, 1, 1)), 1)
            fake = fake.squeeze(2)
            fake_np = fake.detach().cpu().numpy()
            fake_df = pd.DataFrame(fake_np)
            fake_df.to_csv(path, index=False, header=False, mode='a')

    for i in range(real.shape[0]):
        ax2.plot(real[i].detach().cpu().view(-1), label=label_name, c='b', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    config = Config()
    g = Generator()
    batch_size = 30
    nz = 100

    # generateData('005', batch_size, config.fake_005_cycle_path)
    # generateData('010', batch_size, config.fake_010_cycle_path)
    # generateData('015', batch_size, config.fake_015_cycle_path)
    # generateData('020', batch_size, config.fake_020_cycle_path)
    # generateData('025', batch_size, config.fake_025_cycle_path)
    # generateData('N', batch_size * 5, config.fake_N_cycle_path)

    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, sharex= True)
    # line_width = 0.5
    # ## display all kinds of synthetic signals
    # showGeneratedFrame('005', ax1, line_width, color = 'r')
    # showGeneratedFrame('010', ax1, line_width, color = 'b')
    # showGeneratedFrame('015', ax1, line_width, color = 'g')
    # showGeneratedFrame('020', ax1, line_width, color = 'c')
    # showGeneratedFrame('025', ax1, line_width, color = 'm')
    # showGeneratedFrame('N', ax1, line_width, color = 'y')
    # ax1.set_title('synthetic signal')
    # ax1.legend()

    fig, ax2 = plt.subplots()
    line_width = 1
    ## display all kinds of original signals
    # showOriginalFrame('005', ax2, line_width, color = 'r')
    # showOriginalFrame('010', ax2, line_width, color = 'b')
    # showOriginalFrame('015', ax2, line_width, color = 'g')
    # showOriginalFrame('020', ax2, line_width, color = 'c')
    showOriginalFrame('025', ax2, line_width, color = 'm')
    showOriginalFrame('N', ax2, line_width, color = 'y')
    ax2.set_title('original signal')
    ax2.legend()
    plt.show()

