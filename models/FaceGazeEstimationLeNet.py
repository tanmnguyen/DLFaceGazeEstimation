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

class LeNetConvModel(nn.Module):
    def __init__(self):
        super(LeNetConvModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class LeNetRegrModel(nn.Module):
    def __init__(self):
        super(LeNetRegrModel, self).__init__()
        self.features = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(32448, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class FaceGazeEstimationModelLeNet(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(FaceGazeEstimationModelLeNet, self).__init__(device)
        self.name = "FaceGazeEstimationModel-LeNet.pt"

        self.LeNetConvModel = LeNetConvModel() 
        self.LeNetRegrModel = LeNetRegrModel()

        self.device = device
        self.to(device)
        
    def forward(self, data): 
        # facial image 
        data = data.to(self.device)
        # forward face image 
        xFace = self.LeNetConvModel(data)
        xFace = xFace.view(xFace.size(0), -1)
        # regression face image 
        xFace = self.LeNetRegrModel(xFace)
        # result
        return xFace