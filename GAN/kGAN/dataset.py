import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader


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
        target = torch.LongTensor(np.array([self.df.loc[idx, 'class']]))
        return signal, target

    def __len__(self):
        return len(self.df)

    def minmax_normalize(self):
        for index in range(self.length):
            max = self.df.loc[index, :self.dataLength-1].max()
            min = self.df.loc[index, :self.dataLength-1].min()
            self.df.loc[index, :self.dataLength-1] = \
                2 * (self.df.loc[index, :self.dataLength-1] - min) / (max - min) - 1

def get_dataloader(label_name, batch_size):
    csv_path = '../../data/MA_1D/dataset_real.csv'
    id_to_label = {
        0: "L005",
        1: "L010",
        2: "L015",
        3: "L020",
        4: "L025",
        5: "N",
        6: "R005",
        7: "R010",
        8: "R015",
        9: "R020",
        10: "R025",
    }

    df = pd.read_csv(csv_path, header=None)
    df.rename(columns={1475: 'class'}, inplace=True)
    df['label'] = df.iloc[:, -1].map(id_to_label)

    df = df.loc[df['label'] == label_name]
    df.reset_index(drop=True, inplace=True)

    dataset = Dataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    return dataloader


if __name__ == '__main__':

    dataloader = get_dataloader(label_name='L005',batch_size=3)

    # print(len(dataloader))
    x, y = next(iter(dataloader))
    print(x.shape, y.shape)

