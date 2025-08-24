import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseEstimator(nn.Module):
    def __init__(self):
        super(NoiseEstimator , self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1 , out_channels=32 , kernel_size=3,  padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32 , out_channels=64 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64 , out_channels=32 , kernel_size=3 , padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32 , out_channels=1 , kernel_size=3 , padding=1)
        )

    def forward(self , lr):
        return self.layers(lr)


