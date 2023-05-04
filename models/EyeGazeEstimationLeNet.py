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
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class LeNetRegrModel(nn.Module):
    def __init__(self):
        super(LeNetRegrModel, self).__init__()
        self.features = nn.Sequential(

            nn.Linear(11520, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),

            nn.Linear(84, 2),
            nn.Tanh() # tanh activation function maps to (-1,1)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class EyeGazeEstimationModelLeNet(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(EyeGazeEstimationModelLeNet, self).__init__(device)
        self.name = "EyeGazeEstimationModel-LeNet.pt"

        self.LeNetConvModel = LeNetConvModel() 
        self.LeNetRegrModel = LeNetRegrModel()

        self.device = device
        self.to(device)
        
    def forward(self, data): 
        # left and right eye images 
        left, right = data[0].to(self.device), data[1].to(self.device)
        # forward left eye
        xEyeL = self.LeNetConvModel(left)
        xEyeL = xEyeL.view(xEyeL.size(0), -1)
        # forward right eye
        xEyeR = self.LeNetConvModel(right)
        xEyeR = xEyeR.view(xEyeR.size(0), -1)
        # concat both eye images for regression
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.LeNetRegrModel(xEyes)
        # result
        return xEyes


    