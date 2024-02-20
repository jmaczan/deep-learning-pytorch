import torch
from torch import nn


class AlexNet(nn.Module):
    """
    Paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=(11, 11),
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 2nd conv layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 3rd conv layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3)),
            nn.ReLU(),
            # 4th conv layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3)),
            nn.ReLU(),
            # 5th conv layer
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            nn.ReLU(),
            # 1st fc layer with dropout
            nn.Linear(in_features=256, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # 2nd fc layer with dropout
            nn.Linear(in_features=4096, out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # 3rd fc layer
            nn.Linear(in_features=2048, out_features=1000),
        )
