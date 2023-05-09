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
        self.features = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
        # Adjust subsequent layer configurations
        self.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.features(x)
        return x

class AlexNetRegrModel(nn.Module):
    def __init__(self):
        super(AlexNetRegrModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=1536, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=4096, out_features=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class EyeGazeEstimationModelAlexNet(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(EyeGazeEstimationModelAlexNet, self).__init__(device)
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


    