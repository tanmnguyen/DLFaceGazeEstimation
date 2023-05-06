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
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 12, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(12, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class LeNetRegrModel(nn.Module):
    def __init__(self):
        super(LeNetRegrModel, self).__init__()
        self.features = nn.Sequential(

            nn.Linear(9216, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.85),

            nn.Linear(84, 2),
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


    