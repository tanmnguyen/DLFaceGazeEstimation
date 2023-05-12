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

class FaceGazeEstimationModelResNet18(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(FaceGazeEstimationModelResNet18, self).__init__(device)
        self.name = "FaceGazeEstimationModel-ResNet18.pt"
        # init resnet 18 with pre-trained weight 
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # modify output layer 
        self.model.fc = nn.Linear(512, 2)
        # device configuration
        self.device = device

    def tune_config(self):
        # Freeze all Batch Normalization (BN) layers
        for name, param in self.model.named_parameters():
            if 'bn' in name:
                param.requires_grad = False

    def forward(self, data): 
        # facial image 
        data = data.to(self.device)
        # forward face image 
        result = self.model(data)
        # result
        return result