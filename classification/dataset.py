import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from sklearn.model_selection import train_test_split
from classification.config import Config

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
        target = torch.zeros([1, 3])
        target[0, int(self.df.loc[idx, 'class'])] = 1
        # target = torch.LongTensor(np.array([self.df.loc[idx, 'class']]))
        return signal, target.squeeze()

    def __len__(self):
        return len(self.df)

    def minmax_normalize(self):
        for index in range(self.length):
            max = self.df.loc[index, :self.dataLength-1].max()
            min = self.df.loc[index, :self.dataLength-1].min()
            self.df.loc[index, :self.dataLength-1] = \
                2 * (self.df.loc[index, :self.dataLength-1] - min) / (max - min) - 1

def get_dataloader(phase, batch_size):
    config = Config()
    if phase == 'train' or phase == 'val':
        df_path = config.fake_path
    elif phase == 'test':
        df_path = config.real_path

    df = pd.read_csv(df_path, header=None)
    df.rename(columns={1475: 'label'}, inplace=True)
    df.rename(columns={1476: 'class'}, inplace=True)
    # df['label'] = df.iloc[:, -1].map(config.id_to_label)
    # print(df['label'].value_counts())
    # print(df['class'].value_counts())

    if phase == 'train' or phase == 'val':
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=config.seed, stratify=df['class']
        )
        train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
        df = train_df if phase == 'train' else val_df
    dataset = Dataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    return dataloader

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.Grayscale(),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize((0.5, ), (0.5, ))  # 标准化至[-1,1]
])

def get_imagedataloader(phase, batch_size):
    config = Config()
    if phase == 'train':
        df_path = config.train_img_path
    elif phase == 'val':
        df_path = config.val_img_path
    elif phase == 'test':
        df_path = config.test_img_path
    dataset = datasets.ImageFolder(root=df_path, transform=data_transform)
    dataloader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=3)
    return dataloader

def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]
    # print(images_batch.shape)

    for i in range(len(labels_batch)):
        label_ = labels_batch[i].item()
        # print(images_batch[i].shape)
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 3, i + 1)
        ax.imshow(image_)
        # ax.plot(image_[0], image_[2])
        ax.set_title(str(label_))
        ax.axis('off')

if __name__ == '__main__':
    # # dataloader = get_dataloader(phase='train', batch_size=32)
    # dataloader = get_dataloader(phase='test', batch_size=33)
    # print(len(dataloader))
    # x, y = next(iter(dataloader))
    # print(x.shape, y.shape)

    # dataloader = get_imagedataloader(phase='train', batch_size=3)
    # dataloader = get_imagedataloader(phase='val', batch_size=3)
    dataloader = get_imagedataloader(phase='test', batch_size=3)
    print(len(dataloader))
    x, y = next(iter(dataloader))
    print(x.shape, y.shape)

    plt.figure()
    for i_batch, sample_batch in enumerate(dataloader):
        show_batch_images(sample_batch)
        plt.show()

