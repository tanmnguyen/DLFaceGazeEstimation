import sys
sys.path.append("../")

import cv2 
import torch
import traceback
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from utils.general import yawpitch2xyz
from utils.general import letterbox_resize
from models.GazeEstimation import GazeEstimationModel

class EyeGazeEstimationModel(GazeEstimationModel):
    def __init__(self):
        super(EyeGazeEstimationModel, self).__init__()

        # specify model name 
        self.set_name("eyeGazeEstimationModel.pt")

        # define model 
        eyes_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 64, kernel_size=1),
            nn.ReLU()
        )

        self.right_eye_model = eyes_model
        self.left_eye_model  = eyes_model

        self.face_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 64, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(9216, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.concat_eyes = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18432, 128),
            nn.ReLU()
        )

        self.concat_face_eye = nn.Sequential(
            nn.Flatten(),

            nn.Linear(192, 128),
            nn.ReLU(),

            nn.Linear(128, 2),
        )

    def forward(self, l_eye, r_eye, face):
        output_1 = self.left_eye_model(l_eye)
        output_2 = self.right_eye_model(r_eye)
        output_3 = self.face_model(face)
        output   = torch.cat([output_1, output_2], dim=1)  
        output   = self.concat_eyes(output)
        output   = torch.cat([output, output_3], dim=1)
        output   = self.concat_face_eye(output)
        return output

    # override 
    def fit(self, epoch, l1_criterion, cs_criterion, optimizer):
        self.train()

        train_l1_loss, train_cs_loss = 0, 0
        for batch_idx, (l_eye, r_eye, face, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            try:
                optimizer.zero_grad()
                output = self.forward(l_eye, r_eye, face)
                l1_loss = l1_criterion(output, target)
                with torch.no_grad():
                    cs_loss = cs_criterion(
                        torch.tensor(yawpitch2xyz(output.numpy())).float(), 
                        torch.tensor(yawpitch2xyz(target.numpy())).float()
                    )
                # update based on l1 loss
                l1_loss.backward()
                optimizer.step()

                train_l1_loss += l1_loss.item()
                train_cs_loss += torch.sum(cs_loss)
            except:
                traceback.print_exc()
                break

        train_l1_loss /= len(self.train_loader)
        train_cs_loss /= len(self.train_loader)

        return train_l1_loss, train_cs_loss

    # override 
    def evaluate(self, epoch, l1_criterion, cs_criterion, optimizer):
        self.eval()

        val_cs_loss, val_l1_loss = 0, 0
        with torch.no_grad():
            for batch_idx, (l_eye, r_eye, face, target) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch}")):
                output = self.forward(l_eye, r_eye, face)
                l1_loss = l1_criterion(output, target)
                cs_loss = cs_criterion(
                    torch.tensor(yawpitch2xyz(output.numpy())).float(), 
                    torch.tensor(yawpitch2xyz(target.numpy())).float()
                )
                val_l1_loss += l1_loss.item()
                val_cs_loss += torch.sum(cs_loss)

        val_l1_loss /= len(self.val_loader)
        val_cs_loss /= len(self.val_loader)
        
        return val_l1_loss, val_cs_loss