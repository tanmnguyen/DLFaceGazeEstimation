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
from models.GazeEstimation import GazeEstimationModel

class ItrackerImageModel(nn.Module):
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class ITrackerModel(GazeEstimationModel):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.name = "ITrackerModel.pt"
        self.eyeModel = ItrackerImageModel() # both eyes images 
        self.faceModel = FaceImageModel()    # face image 
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*12*12*64, 128),
            nn.ReLU(inplace=True),
        )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, left, right, face):
        # Eye nets
        xEyeL = self.eyeModel(left)
        xEyeR = self.eyeModel(right)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)
        # Face net
        xFace = self.faceModel(face)
        # Cat all
        x = torch.cat((xEyes, xFace), 1)
        x = self.fc(x)
        # result
        return x

    # override 
    def _learn(self, epoch, l1_criterion, cs_criterion, optimizer):
        self.train()

        train_l1_loss, train_cs_loss = 0, 0
        for l_eye, r_eye, face, target in tqdm(self.train_loader, desc=f"(Training) Epoch {epoch}"):
            try:
                optimizer.zero_grad()
                output = self.forward(l_eye, r_eye, face)
                l1_loss = l1_criterion(output, target)
                cs_loss = torch.abs(cs_criterion(pitchyaw2xyz(output), pitchyaw2xyz(target)))
                # update based on l1 loss
                l1_loss.backward()
                optimizer.step()

                train_l1_loss += l1_loss.item()
                train_cs_loss += cs_loss.mean()
            except:
                traceback.print_exc()
                break

        train_l1_loss /= len(self.train_loader)
        train_cs_loss /= len(self.train_loader)

        return train_l1_loss, train_cs_loss

    # override
    def _eval(self, epoch, l1_criterion, cs_criterion, optimizer):
        self.eval()

        val_cs_loss, val_l1_loss = 0, 0
        with torch.no_grad():
            for l_eye, r_eye, face, target in tqdm(self.val_loader, desc=f"(Validating) Epoch {epoch}"):
                output = self.forward(l_eye, r_eye, face)
                l1_loss = l1_criterion(output, target)
                cs_loss = torch.abs(cs_criterion(pitchyaw2xyz(output), pitchyaw2xyz(target)))
                val_l1_loss += l1_loss.item()
                val_cs_loss += cs_loss.mean()

        val_l1_loss /= len(self.val_loader)
        val_cs_loss /= len(self.val_loader)
        
        return val_l1_loss, val_cs_loss

    