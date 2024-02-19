import torch
from torch import nn
from torch import flatten


class LeNet(nn.Module):
    """(CONV => RELU => POOL) * 2 => FC => RELU => FC => SOFTMAX"""

    def __init__(self, classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(in_features=6 * 16 * 5, out_features=120)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=120, out_features=classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        output = self.softmax(x)

        return output


if __name__ == "__main__":
    _ = LeNet()
