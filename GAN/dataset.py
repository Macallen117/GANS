import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from GAN.config import Config

class Dataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()
        self.dataLength = len(self.data_columns)  # 1475
        self.length = len(df) # num of data which belongs to same label
        self.minmax_normalize()


    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.data_columns].astype('float32')
        signal = torch.FloatTensor(np.array([signal.values]))
        label_num = torch.LongTensor(np.array([self.df.loc[idx, 'label']]))
        class_num = torch.LongTensor(np.array([self.df.loc[idx, 'class']]))
        return signal, label_num

    def __len__(self):
        return len(self.df)

    def minmax_normalize(self):
        for index in range(self.length):
            max = self.df.loc[index, :self.dataLength-1].max()
            min = self.df.loc[index, :self.dataLength-1].min()
            self.df.loc[index, :self.dataLength-1] = \
                2 * (self.df.loc[index, :self.dataLength-1] - min) / (max - min) - 1

def get_dataloader(dataset, label_name, batch_size):
    config = Config()
    if dataset == 'fake':
        df_path = config.fake_path
    elif dataset == 'real':
        df_path = config.real_path

    df = pd.read_csv(df_path, header=None)
    df.rename(columns={1475: 'label'}, inplace=True)
    df.rename(columns={1476: 'class'}, inplace=True)
    # print(df['label'].value_counts())
    # print(df['class'].value_counts())
    # df['label'] = df.iloc[:, -1].map(config.id_to_label)
    df = df.loc[df['label'] == config.label_to_id[label_name]]
    df.reset_index(drop=True, inplace=True)

    dataset = Dataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    return dataloader


if __name__ == '__main__':

    dataloader = get_dataloader(dataset = 'fake', label_name='R005',batch_size=3)

    print(len(dataloader))
    x, y = next(iter(dataloader))
    print(x.shape, y.shape)

