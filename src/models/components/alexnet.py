import torch
from torch import nn


class AlexNet(nn.Module):
    """
    Paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=55, kernel_size=(11, 11), stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5)),
            #
            nn.Conv2d(in_channels=55, out_channels=27, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.Conv2d(in_channels=55, out_channels=27, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.Conv2d(in_channels=55, out_channels=27, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.Conv2d(in_channels=55, out_channels=27, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            #
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=192),
        )
