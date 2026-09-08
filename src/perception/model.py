import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PoseRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 3)

    def forward(self, x):
        return self.backbone(x)