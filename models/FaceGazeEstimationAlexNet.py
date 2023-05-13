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
        self.features = models.alexnet().features

    def forward(self, x):
        x = self.features(x)
        return x

class AlexNetRegrModel(nn.Module):
    def __init__(self):
        super(AlexNetRegrModel, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm1d(9216),
            nn.Dropout(0.5),
            nn.Linear(in_features=9216, out_features=2),
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


    