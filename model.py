import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, represent_dim = 128):
        super(ResNetSimCLR, self).__init__()

        self.f = models.resnet50()
        self.f.fc = nn.Identity()
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Linear(512, represent_dim, bias = True)
        )

    def forward(self, x):
        x = self.f(x)  # [b, 2048, 1, 1]
        feature = torch.flatten(x, 1)  # [b, 2048]
        out = self.g(feature)  # ]b, 128
        return F.normalize(feature, dim = -1), F.normalize(out, dim = -1)