import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.rnn_layer = nn.LSTM(
            input_size=1475,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.gen = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Linear(4096, 1475),
            nn.Tanh()
        )

    def forward(self, x):
        x,_ = self.rnn_layer(x)
        x = x.view(-1,1024)
        x = self.gen(x)
        return x.unsqueeze(1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.rnn_layer = nn.LSTM(
                input_size=1475,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        self.disc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x,_ = self.rnn_layer(x)
        x = x.view(-1, 1024)
        x = self.disc(x)
        return x
