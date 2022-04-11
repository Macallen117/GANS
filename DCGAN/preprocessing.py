import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self, root):
        self.root = root
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)

        target = 1  # only one class: real data
        return step, target

    def build_dataset(self):
        dataset = []
        for csvFilename in os.listdir('.'):
            csvFileObj = open(csvFilename)
            readerObj = csv.reader(csvFileObj, delimiter=',')
            sample = []
            for row in readerObj:
                if row:
                    sample.append(float(row[1]))
            csvFileObj.close()
            dataset.append(sample)

        dataset = np.vstack(dataset).T
        dataset = torch.from_numpy(dataset).float()
        return dataset

    def minmax_normalize(self):
        for index in range(self.length):
            self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                self.dataset[:, index].max() - self.dataset[:, index].min())


if __name__ == '__main__':
    os.chdir('../dataset/headerRemoved')
    dataset = Dataset('../dataset/headerRemoved')  #torch.Size([1475, 33])
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()