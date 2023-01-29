import os
import random

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from WGAN.model import *
from GAN.dataset import get_dataloader
from GAN.config import Config


def getRealData(label_name, batch_size):
    dataloader = get_dataloader(dataset='real', label_name=label_name, batch_size=batch_size)
    real = []
    for step, (data, labels) in enumerate(dataloader):
        real.append(data)
    real = torch.stack(real, dim=0)
    real = real.squeeze(0)
    return real

def getFakeData(label_name, batch_size):
    g = Generator()
    netG_path = 'WGAN/netG/'
    netG_path = os.path.join(netG_path, 'netG_{}.pt'.format(label_name))

    g.load_state_dict(torch.load(netG_path))
    fixed_noise = torch.randn(batch_size, 100, 1)
    fake = g(fixed_noise).to(config.device)
    return fake


def visualization_compare_real_fake(label_name, analysis_type, visualization_type):
    fig, ax = plt.subplots()
    line_width = 1

    real = getRealData(label_name, batch_size)
    rgb = (random.random(), random.random(), random.random())
    ax.plot(real[0].view(-1), label="Original", c=rgb, linewidth=line_width)

    fake = getFakeData(label_name, batch_size)
    rgb = (random.random(), random.random(), random.random())
    ax.plot(fake[0].view(-1).detach().cpu(), label="Synthetic", c=rgb, linewidth=line_width)
    plt.show()

    ori_data = real.detach().cpu()
    generated_data = fake.detach().cpu()

    real_no = min([1000, len(ori_data)])   # real的个数
    fake_no = min([1000, len(generated_data)])   # fake的个数

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    no, seq_len, dim = ori_data.shape

    prep_data = np.reshape(ori_data[:, :, :], [real_no, seq_len])
    prep_data_hat = np.reshape(generated_data[:, :, :], [fake_no, seq_len])

    colors = ["r" for i in range(real_no)] + ["g" for i in range(fake_no)]

    if visualization_type == '3d':
        ax = plt.axes(projection='3d')
    elif visualization_type == '2d':
        fig, ax = plt.subplots()
    if analysis_type == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # ax = plt.figure().gca(projection='3d')
        ax.scatter(pca_results[:, 0],
                   pca_results[:, 1],
                   c=colors[:real_no],
                   label="Original")
        ax.scatter(pca_hat_results[:, 0],
                   pca_hat_results[:, 1],
                   c=colors[real_no:],
                   label="Synthetic")

        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.legend()
        plt.show()

    elif analysis_type == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        tsne = TSNE(n_components=2, random_state=config.seed, n_jobs=-1,
                        init='random', learning_rate='auto')
        tsne_results = tsne.fit_transform(prep_data_final)

        ax.scatter(tsne_results[:real_no, 0],
                   tsne_results[:real_no, 1],
                   c=colors[:real_no],
                   label="Original")
        ax.scatter(tsne_results[real_no:, 0],
                   tsne_results[real_no:, 1],
                   c=colors[real_no:],
                   label="Synthetic")

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.legend()
        plt.show()


def visualization_one_kind(label_name, batch_size, analysis_type, ax):
    if label_name == 'N':
        label_show = 'Normal'
    elif label_name == '005':
        label_show = 'left 0.05mm'
    elif label_name == '010':
        label_show = 'left 0.10mm'
    elif label_name == '015':
        label_show = 'left 0.15mm'
    elif label_name == '020':
        label_show = 'left 0.20mm'
    elif label_name == '025':
        label_show = 'left 0.25mm'
    real = getRealData(label_name, batch_size).detach().cpu()
    fake = getFakeData(label_name, batch_size).detach().cpu()

    real = np.asarray(real)
    fake = np.asarray(fake)

    no, seq_len, dim = real.shape

    real = np.reshape(real[:, :, :], [-1, seq_len])
    fake = np.reshape(fake[:, :, :], [-1, seq_len])

    data = np.concatenate((real, fake), axis=0)
    print(data.shape)

    rgb = (random.random(), random.random(), random.random())
    rgb = np.array([rgb])

    if analysis_type == 'pca':
        pca = PCA(n_components=2, random_state=config.seed)
        pca_results = pca.fit_transform(data)

        ax.scatter(pca_results[:, 0],
                   pca_results[:, 1],
                   c=rgb,
                   label=label_show)

    elif analysis_type == 'tsne':
        tsne = TSNE(n_components=2, random_state=config.seed, n_jobs=-1,
                    init='random', learning_rate='auto')
        tsne_results = tsne.fit_transform(data)

        ax.scatter(tsne_results[:, 0],
                   tsne_results[:, 1],
                   c=rgb,
                   label=label_show)

def visualization_compare_all_kinds(analysis_type, visualization_type):
    batch_size = 30
    if visualization_type == '3d':
        ax = plt.axes(projection='3d')
    elif visualization_type == '2d':
        fig, ax = plt.subplots()
    visualization_one_kind('005', batch_size, analysis_type, ax)
    visualization_one_kind('010', batch_size, analysis_type, ax)
    visualization_one_kind('015', batch_size, analysis_type, ax)
    visualization_one_kind('020', batch_size, analysis_type, ax)
    visualization_one_kind('025', batch_size, analysis_type, ax)
    visualization_one_kind('N', batch_size, analysis_type, ax)

    plt.title(analysis_type)
    # plt.axis('off')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    config = Config()
    batch_size = 32

    # visualization_compare_real_fake('005', 'pca', '3d')
    # visualization_compare_real_fake('005', 'tsne', '3d')

    # visualization_compare_real_fake('005', 'pca', '2d')
    # visualization_compare_real_fake('005', 'tsne', '2d')

    visualization_compare_all_kinds('pca', '2d')
    visualization_compare_all_kinds('pca', '3d')

    # visualization_compare_all_kinds('tsne', '3d')
    # visualization_compare_all_kinds('tsne', '2d')
