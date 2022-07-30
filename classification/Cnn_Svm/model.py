import torch
from torch import nn

class CNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        num_kernal=8,
        kernel_size=3,
        num_classes=3,
        out_features = 512,
        dropout = 0.2
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_kernal,
            kernel_size=kernel_size,
        )

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv_2 = nn.Conv1d(
            in_channels=num_kernal,
            out_channels=num_kernal * 2,
            kernel_size=kernel_size
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.fc = nn.Linear(out_features = out_features)

    def flatten(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

    def forward(self, input):
        print(input.shape)
        x = self.conv1(input)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        # print(x.shape) # num_features * num_channels
        # x = x.view(-1, x.size(1) * x.size(2))
        # x = F.softmax(self.fc(x), dim=1)
        return x

if __name__ == '__main__':
    cnn = CNN(num_classes=3, num_kernal=8, out_features = 512)


