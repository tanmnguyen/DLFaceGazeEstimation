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
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class AlexNetRegrModel(nn.Module):
    def __init__(self):
        super(AlexNetRegrModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=12544, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.85),

            # nn.Linear(in_features=4096, out_features=4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.85),
            
            nn.Linear(in_features=2048, out_features=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class FaceGazeEstimationModelAlexNet(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(FaceGazeEstimationModelAlexNet, self).__init__(device)
        self.name = "FaceGazeEstimationModel-AlexNet.pt"

        self.AlexNetConvModel = AlexNetConvModel() 
        self.AlexNetRegrModel = AlexNetRegrModel()

        self.device = device
        self.to(device)
        
    def forward(self, data): 
        # facial image 
        data = data.to(self.device)
        # forward face image 
        xFace = self.AlexNetConvModel(data)
        xFace = xFace.view(xFace.size(0), -1)
        # regression face image 
        xFace = self.AlexNetRegrModel(xFace)
        # result
        return xFace


    