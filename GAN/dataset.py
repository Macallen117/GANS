import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from GAN.config import Config

# class Dataset():
#     def __init__(self, df):
#         self.df = df
#         self.data_columns = self.df.columns[:-1].tolist()
#         self.dataLength = len(self.data_columns)  # 1475
#         self.length = len(df) # num of data which belongs to same label
#         self.minmax_normalize()
#
#
#     def __getitem__(self, idx):
#         signal = self.df.loc[idx, self.data_columns].astype('float32')
#         signal = torch.FloatTensor(np.array([signal.values]))
#         # label_num = torch.LongTensor(np.array([self.df.loc[idx, 'label']]))
#         class_num = torch.LongTensor(np.array([self.df.loc[idx, 'target']]))
#         return signal, class_num
#
#     def __len__(self):
#         return len(self.df)
#
#     def minmax_normalize(self):
#         for index in range(self.length):
#             max = self.df.loc[index, :self.dataLength-1].max()
#             min = self.df.loc[index, :self.dataLength-1].min()
#             self.df.loc[index, :self.dataLength-1] = \
#                 2 * (self.df.loc[index, :self.dataLength-1] - min) / (max - min) - 1

def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.125
    )
    ax.set_title(class_name,fontsize=20)

def create_cycle_dataset(df):
    x_values = []
    y_values = []
    for i in range(len(df)):
        x = df.iloc[i,:-1].values
        y = df.iloc[i,-1]
        x_values.append(x)
        y_values.append(y)

    x = [torch.tensor(x).unsqueeze(1).float() for x in x_values]
    y = [torch.tensor(y).unsqueeze(0).float() for y in y_values]
    x = torch.stack(x)
    y = torch.stack(y)

    n_seq, seq_len, n_features = x.shape
    return x, y, seq_len, n_features

def get_dataloader(dataset, label_name, batch_size):
    config = Config()
    if dataset == 'fake':
        df_path = config.fake_cycle_path + label_name + '.csv'
    elif dataset == 'real':
        df_path = config.real_cycle_path + label_name + '.csv'

    df = pd.read_csv(df_path, header=None)
    df.rename(columns={350: 'target'}, inplace=True)
    # print(df)

    # fig, axs = plt.subplots(
    #     nrows=1,
    #     ncols=1,
    #     sharey=True,
    #     figsize=(12, 8),
    # )

    # data = df.drop(labels='target', axis=1).mean(axis=0).to_numpy()
    # plot_time_series_class(data, label_name, axs)
    # plt.show()

    X, Y, seq_len, n_features = create_cycle_dataset(df)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader(dataset = 'real', label_name='010',batch_size=32)

    print(len(dataloader))
    x, y = next(iter(dataloader))
    print(x.shape, y.shape)


