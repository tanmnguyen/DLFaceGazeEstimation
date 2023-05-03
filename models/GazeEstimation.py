import os
import yaml
import torch
import datetime
import traceback
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.runtime import available_device
from utils.general import pitchyaw2xyz, mean_abs_angle_loss

class GazeEstimationModel(nn.Module):
    def __init__(self, device=available_device()):
        super(GazeEstimationModel, self).__init__()
        self.name  = "gaze-estimation-model.pt"
        self.device = device 
        self.to(device)

    def _save_fig(self, dst_dir: str):
        # Plot the training and validation losses
        train_l1_loss = [h["l1_loss"] for h in self.train_step_history]
        train_mal_loss = [h["mal_loss"] for h in self.train_step_history]
        val_l1_loss = [h["l1_loss"] for h in self.val_step_history]
        val_mal_loss = [h["mal_loss"] for h in self.val_step_history]

        # Training L1 Loss
        plt.figure()
        plt.plot(train_l1_loss)
        plt.title("Training L1 Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(dst_dir, "training_l1_loss.png"))

        # Training Mal Loss
        plt.figure()
        plt.plot(train_mal_loss)
        plt.title("Training Mean Absolute Angle Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(dst_dir, "training_mal_loss.png"))

        # Validation L1 Loss
        plt.figure()
        plt.plot(val_l1_loss)
        plt.title("Validation L1 Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(dst_dir, "validation_l1_loss.png"))

        # Validation Mal Loss
        plt.figure()
        plt.plot(val_mal_loss)
        plt.title("Validation Mean Absolute Angle Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(dst_dir, "validation_mal_loss.png"))

    def _learn(self, epoch, l1_criterion, mal_criterion, optimizer):
        self.train() 

        train_l1_loss, train_mal_loss = 0, 0
        for data, target in tqdm(self.train_loader, desc=f"(Train) Epoch {epoch}"):
            try:
                optimizer.zero_grad()
                target      = target.to(self.device)
                output      = self.forward(data)
                l1_loss     = l1_criterion(output, target)
                mal_loss    = mal_criterion(pitchyaw2xyz(output), pitchyaw2xyz(target))
                # update based on l1 loss
                l1_loss.backward()
                optimizer.step()

                train_l1_loss  += l1_loss.item()
                train_mal_loss += mal_loss.item()

                self.train_step_history.append({
                    "l1_loss":  l1_loss.item(),
                    "mal_loss": mal_loss.item()
                })

            except:
                traceback.print_exc()
                break

        train_l1_loss  /= len(self.train_loader)
        train_mal_loss /= len(self.train_loader)

        return train_l1_loss, train_mal_loss

    def _eval(self, epoch, l1_criterion, mal_criterion, optimizer):
        self.eval()

        val_l1_loss, val_mal_loss = 0, 0
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=f"(Valid) Epoch {epoch}"):
                target      = target.to(self.device)
                output      = self.forward(data)
                l1_loss     = l1_criterion(output, target)
                mal_loss    = mal_criterion(pitchyaw2xyz(output), pitchyaw2xyz(target))
                val_l1_loss  += l1_loss.item()
                val_mal_loss += mal_loss.item()

                self.val_step_history.append({
                    "l1_loss":  l1_loss.item(),
                    "mal_loss": mal_loss.item()
                })

        val_l1_loss  /= len(self.val_loader)
        val_mal_loss /= len(self.val_loader)
        
        return val_l1_loss, val_mal_loss

    def fit(self, train_loader, val_loader, epochs, lr=0.001):
        self.train_loader, self.val_loader = train_loader, val_loader 

        dst_dir = f"trains/train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"
        os.makedirs(dst_dir, exist_ok=True)

        # optimizer, l1 loss, mean absolute angle loss criterions 
        optimizer, l1_criterion, mal_criterion = optim.Adam(self.parameters(), lr=lr), F.l1_loss, mean_abs_angle_loss

        optimal_loss = None
        self.train_step_history = []
        self.val_step_history   = []
        for epoch in range(epochs):
            train_l1_loss, train_mal_loss = self._learn(epoch + 1, l1_criterion, mal_criterion, optimizer)
            val_l1_loss, val_mal_loss     = self._eval(epoch + 1, l1_criterion, mal_criterion, optimizer)

            # update and save the best model per epoch using mean absolute angle loss criteria
            if optimal_loss is None or optimal_loss > val_mal_loss:
                optimal_loss = val_mal_loss 
                torch.save(self.state_dict(), os.path.join(dst_dir, self.name))

            # log info
            print(f"Train Loss (L1): {train_l1_loss:.4f}, Train Loss (Mean Absolute Angle Loss): {train_mal_loss:4f}")
            print(f"Val Loss (L1):   {val_l1_loss:.4f}, Val Loss (Mean Absolute Angle Loss):   {val_mal_loss:.4f}")
            print()
        
        self._save_fig(dst_dir)
