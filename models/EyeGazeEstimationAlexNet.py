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

class AlexNetConvModel(nn.Module):
    def __init__(self):
        super(AlexNetConvModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(384),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class AlexNetRegrModel(nn.Module):
    def __init__(self):
        super(AlexNetRegrModel, self).__init__()
        self.features = nn.Sequential(
            # nn.Dropout(),

            nn.Linear(in_features=1024, out_features=4096),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(4096),

            # nn.Dropout(),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(4096),
            
            nn.Linear(in_features=4096, out_features=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class EyeGazeEstimationModelAlexNet(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(EyeGazeEstimationModel, self).__init__(device)
        self.name = "EyeGazeEstimationModel-AlexNet.pt"

        self.AlexNetConvModel = AlexNetConvModel() 
        self.AlexNetRegrModel = AlexNetRegrModel()

        self.device = device
        self.to(device)
        
    def forward(self, data): 
        # left and right eye images 
        left, right = data[0].to(self.device), data[1].to(self.device)
        # forward left eye
        xEyeL = self.AlexNetConvModel(left)
        xEyeL = xEyeL.view(xEyeL.size(0), -1)
        # forward right eye
        xEyeR = self.AlexNetConvModel(right)
        xEyeR = xEyeR.view(xEyeR.size(0), -1)
        # concat both eye images for regression
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.AlexNetRegrModel(xEyes)
        # result
        return xEyes


    