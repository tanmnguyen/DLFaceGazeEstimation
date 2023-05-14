import sys
sys.path.append("../")

import os
import math
import torch
import traceback
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
from utils.general import pitchyaw2xyz
from utils.runtime import available_device
from models.GazeEstimation import GazeEstimationModel

class ResNet18ConvModel(nn.Module):
    def __init__(self):
        super(ResNet18ConvModel, self).__init__()
        self.model = models.resnet18()
        # Remove the last fully connected layer
        self.model.fc = torch.nn.Identity()
                
    def forward(self, x):
        x = self.model(x)
        return x

class ResNet18RegrModel(nn.Module):
    def __init__(self):
        super(ResNet18RegrModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class EyeGazeEstimationModelResNet18(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(EyeGazeEstimationModelResNet18, self).__init__(device)
        self.name = "EyeGazeEstimationModel-ResNet18.pt"

        self.ResNet18ConvModel = ResNet18ConvModel() 
        self.ResNet18RegrModel = ResNet18RegrModel()

        self.device = device
        self.to(device)
        
    def forward(self, data): 
        # left and right eye images 
        left, right = data[0].to(self.device), data[1].to(self.device)
        # forward left eye
        xEyeL = self.ResNet18ConvModel(left)
        xEyeL = xEyeL.view(xEyeL.size(0), -1)
        # forward right eye
        xEyeR = self.ResNet18ConvModel(right)
        xEyeR = xEyeR.view(xEyeR.size(0), -1)
        # concat both eye images for regression
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.ResNet18RegrModel(xEyes)
        # result
        return xEyes


    