import torch
from torch import nn


class VGG(nn.Module):
    """https://arxiv.org/pdf/1409.1556"""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # max pooling doesn't change number of channels, but the size of spatial dimensions
            # output dimension (for instance, height): (height - kernel_size + 2*padding)/stride + 1
            # so dimensions now are 112x112
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # dimensions halved again - 56x56 now

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # dimensions halved - 28x28

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # dimensions halved - 14x14

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # final dimensions - 7x7
            
            nn.Flatten(),
            
            nn.Linear(in_features=512*7*7, out_features=4096), #in_features use computed value of dimensions after 5 max poolings
            nn.Dropout(p=0.5),
            nn.ReLU(), 

            nn.Linear(in_features=4096, out_features=4096), 
            nn.Dropout(p=0.5),
            nn.ReLU(), 

            nn.Linear(in_features=4096, out_features=1000), 
            nn.LogSoftmax(dim=1),

            # missing other regularizations, ReLUs might be incorrectly placed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    _ = VGG()
