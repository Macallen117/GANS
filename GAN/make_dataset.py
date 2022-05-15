import os
import random

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from WGAN.wgan import Generator
from GAN.dataset import get_dataloader
from GAN.config import Config


def showGeneratedFrame(label_name):
    dataloader = get_dataloader(dataset='fake', label_name=label_name, batch_size=3)
    data, labels, classes = next(iter(dataloader))
    rgb = (random.random(), random.random(), random.random())
    ax.plot(data[0].view(-1), label=label_name, c=rgb, linewidth=line_width)

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

def generateData(label_name, batch_size):
    netG_path = 'WGAN/netG/'
    netG_path = os.path.join(netG_path, 'netG_{}.pt'.format(label_name))

    g.load_state_dict(torch.load(netG_path))
    fixed_noise = torch.randn(batch_size, nz, 1)
    fake = g(fixed_noise).to(config.device)

    label_id = config.label_to_id[label_name]
    label_id = torch.tensor(label_id, device=config.device)
    fake = torch.cat((fake, label_id.repeat(batch_size, 1, 1)), 2)
    class_id = config.class_to_id[label_name]
    class_id = torch.tensor(class_id, device=config.device)
    fake = torch.cat((fake, class_id.repeat(batch_size, 1, 1)), 2)
    fake = fake.squeeze()

    fake_np = fake.detach().cpu().numpy()
    fake_df = pd.DataFrame(fake_np)
    fake_df.to_csv(config.fake_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    config = Config()
    g = Generator()
    batch_size = 30
    nz = 100
    num = 0
    sample_size = 3

    # generateData('L005', batch_size)
    # generateData('L010', batch_size)
    # generateData('L015', batch_size)
    # generateData('L020', batch_size)
    # generateData('L025', batch_size)
    # generateData('N', batch_size * 5)
    # generateData('R005', batch_size)
    # generateData('R010', batch_size)
    # generateData('R015', batch_size)
    # generateData('R020', batch_size)
    # generateData('R025', batch_size)

    # generateFakeImg('L005', batch_size)
    # generateFakeImg('L010', batch_size)
    # generateFakeImg('L015', batch_size)
    # generateFakeImg('L020', batch_size)
    # generateFakeImg('L025', batch_size)
    # generateFakeImg('N', batch_size * 5)
    # generateFakeImg('R005', batch_size)
    # generateFakeImg('R010', batch_size)
    # generateFakeImg('R015', batch_size)
    # generateFakeImg('R020', batch_size)
    # generateFakeImg('R025', batch_size)

    generateRealImg('L005', sample_size)
    generateRealImg('L010', sample_size)
    generateRealImg('L015', sample_size)
    generateRealImg('L020', sample_size)
    generateRealImg('L025', sample_size)
    generateRealImg('N', sample_size)
    generateRealImg('R005', sample_size)
    generateRealImg('R010', sample_size)
    generateRealImg('R015', sample_size)
    generateRealImg('R020', sample_size)
    generateRealImg('R025', sample_size)

    fig, ax = plt.subplots()
    line_width = 0.75

    # showGeneratedFrame('L005')
    # showGeneratedFrame('L010')
    # showGeneratedFrame('L015')
    # showGeneratedFrame('L020')
    # showGeneratedFrame('L025')
    # showGeneratedFrame('N')
    # showGeneratedFrame('R005')
    # showGeneratedFrame('R010')
    # showGeneratedFrame('R015')
    # showGeneratedFrame('R020')
    # showGeneratedFrame('R025')

    ax.set_title('fake signal')
    ax.legend()
    # plt.show()
