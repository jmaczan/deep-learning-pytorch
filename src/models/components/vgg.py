import torch
from torch import nn


class VGG(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 224*224,
        output_size: int = 1000,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(input_size, 64, stride=3),
            nn.ReLU(),
            nn.Conv3d(64, 64, stride=3),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(32, 128, stride=3),
            nn.ReLU(),
            nn.Conv3d(128, 128, stride=3),
            nn.ReLU(),
            
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(64, 256, stride=3),
            nn.ReLU(),
            nn.Conv3d(256, 256, stride=3),
            nn.ReLU(),
            nn.Conv3d(256, 256, stride=3),
            nn.ReLU(),
            nn.Conv3d(256, 256, stride=3),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(128, 512, stride=3),
            nn.ReLU(),
            nn.Conv3d(512, 512, stride=3),
            nn.ReLU(),
            nn.Conv3d(512, 512, stride=3),
            nn.ReLU(),
            nn.Conv3d(512, 512, stride=3),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(256, 512, stride=3),
            nn.ReLU(),
            nn.Conv3d(512, 512, stride=3),
            nn.ReLU(),
            nn.Conv3d(512, 512, stride=3),
            nn.ReLU(),
            nn.Conv3d(512, 512, stride=3),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2),
            nn.Flatten(), # likely it's neded    
            nn.Linear(in_features=512, out_features=4096), #in_features likely wrong
            nn.Dropout(p=0.5),
            nn.ReLU(), 

            nn.Linear(in_features=4096, out_features=4096), 
            nn.Dropout(p=0.5),
            nn.ReLU(), 

            nn.Linear(in_features=4096, out_features=1000), 
            nn.ReLU(), 
            nn.LogSoftmax(dim=1),

            # missing dropouts and other regularizations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
