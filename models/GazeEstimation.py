import os
import yaml
import torch
import datetime
import traceback
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from utils.runtime import available_device
from utils.general import pitchyaw2xyz, angular_loss
from utils.plot import save_step_history, save_epoc_history

class GazeEstimationModel(nn.Module):
    def __init__(self, device=available_device()):
        super(GazeEstimationModel, self).__init__()
        self.name       = "gaze-estimation-model.pt"
        self.device     = device 

    def tune_config(self):
        num_bn = 0
        # Freeze all Batch Normalization (BN) layers
        for module in self.modules():
           if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad   = False
                num_bn += 1

    def _save_fig(self, dst_dir: str):
        save_step_history(self.train_step_history, self.val_step_history, dst_dir)
        save_epoc_history(self.epoch_history, dst_dir)

    def _train_net(self, epoch: int):
        self.train() 

        train_l1_loss, train_ma_loss = 0, 0
        for data, target in tqdm(self.train_loader, desc=f"(Train) Epoch {epoch} [{self.device}]"):
            target  = target.to(self.device)

            # train per step 
            self.optimizer.zero_grad()
            output  = self.forward(data)
            l1_loss = self.l1_loss(output, target)
            l1_loss.backward()
            self.optimizer.step()

            ma_loss = self.ma_loss(
                pitchyaw2xyz(output), 
                pitchyaw2xyz(target)
            )

            train_l1_loss += l1_loss.item()
            train_ma_loss += ma_loss.item()

            self.train_step_history.append({
                "l1_loss": l1_loss.item(),
                "ma_loss": ma_loss.item()
            })

        train_l1_loss /= len(self.train_loader)
        train_ma_loss /= len(self.train_loader)

        return train_l1_loss, train_ma_loss

    def _eval_net(self, epoch: int):
        self.eval()

        val_l1_loss, val_ma_loss = 0, 0
        with torch.no_grad():
            for data, target in tqdm(self.valid_loader, desc=f"(Valid) Epoch {epoch} [{self.device}]"):
                target  = target.to(self.device)
                output  = self.forward(data)
                l1_loss = self.l1_loss(output, target)
                ma_loss = self.ma_loss(
                    pitchyaw2xyz(output), 
                    pitchyaw2xyz(target)
                )

                val_l1_loss += l1_loss.item()
                val_ma_loss += ma_loss.item()

                self.val_step_history.append({
                    "l1_loss": l1_loss.item(),
                    "ma_loss": ma_loss.item()
                })

            val_l1_loss /= len(self.valid_loader)
            val_ma_loss /= len(self.valid_loader)
        
        return val_l1_loss, val_ma_loss

    def _config(self, train_loader, valid_loader, lr: float, dst_dir: str):
        self.l1_loss      = F.l1_loss      # mean absolute loss 
        self.ma_loss      = angular_loss   # mean angular  loss 
        self.train_loader = train_loader   
        self.valid_loader = valid_loader 
        self.optimizer    = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        os.makedirs(dst_dir, exist_ok=True)

    def fit(self, train_loader, valid_loader, epochs: int, lr: float, dst_dir: str):
        self._config(train_loader, valid_loader, lr, dst_dir)
        self.train_step_history = []
        self.val_step_history   = []
        self.epoch_history      = []
        optimal_loss            = None
        out_path                = os.path.join(dst_dir, f"output.txt")

        for epoch in range(epochs):
            train_l1_loss, train_ma_loss = self._train_net(epoch + 1)
            valid_l1_loss, valid_ma_loss = self._eval_net (epoch + 1)

            # save training history per epoch
            self.epoch_history.append({
                "train_l1_loss": train_l1_loss,
                "train_ma_loss": train_ma_loss,
                "valid_l1_loss": valid_l1_loss,
                "valid_ma_loss": valid_ma_loss
            })

            with open(out_path, "a") as file:
                file.write(f"Epoch {epoch + 1}:\n")
                file.write(f"train_l1_loss {train_l1_loss:.4f}, train_ma_loss {train_ma_loss:.4f}\n")
                file.write(f"valid_l1_loss {valid_l1_loss:.4f}, valid_ma_loss {valid_ma_loss:.4f}\n")
                file.write("\n")
 
            # update and save the best model per epoch using mean angular loss 
            if optimal_loss is None or optimal_loss > valid_ma_loss:
                optimal_loss = valid_ma_loss
                torch.save(
                    self.state_dict(), 
                    os.path.join(dst_dir, self.name)
                )

            # log info
            print(f"Train Loss (L1): {train_l1_loss:.4f}, Train Loss (Mean Angular Loss): {train_ma_loss:.4f}")
            print(f"Valid Loss (L1): {valid_l1_loss:.4f}, Valid Loss (Mean Angular Loss): {valid_ma_loss:.4f}")
            print()
        
        self._save_fig(dst_dir)
