import sys
sys.path.append("../")

import os
import math
import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from utils.general import pitchyaw2xyz
from utils.runtime import available_device
from models.GazeEstimation import GazeEstimationModel

class FaceGazeEstimationModelAlexNet(GazeEstimationModel):
    def __init__(self, device=available_device()):
        super(FaceGazeEstimationModelAlexNet, self).__init__(device)
        self.name = "FaceGazeEstimationModel-AlexNet.pt"

        self.features = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features

        # RGB order to BGR 
        module = getattr(self.features, '0')
        module.weight.data = module.weight.data[:, [2, 1, 0]]

        self.fc1 = nn.Linear(256 * 13**2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

        self.device = device

    def forward(self, data): 
        x = data.to(self.device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = self.fc3(x)
        return x


    