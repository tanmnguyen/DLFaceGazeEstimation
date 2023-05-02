import os
import yaml
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.general import pitchyaw2xyz

class GazeEstimationModel(nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        self.name  = "gaze-estimation-model.pt"

    def _save_fig(self, history, dst_dir: str):
        # Extract loss values from history
        train_l1_loss = [h["train_l1_loss"] for h in history]
        train_cs_loss = [h["train_cs_loss"].numpy() for h in history]
        val_l1_loss = [h["val_l1_loss"] for h in history]
        val_cs_loss = [h["val_cs_loss"].numpy() for h in history]

        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot training and validation l1 loss curves
        ax1.plot(train_l1_loss, label="Train L1 Loss")
        ax1.plot(val_l1_loss, label="Val L1 Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("L1 Loss")
        ax1.legend()

        # Plot training and validation cosine similarity loss curves
        ax2.plot(train_cs_loss, label="Train Cosine Similarity Loss")
        ax2.plot(val_cs_loss, label="Val Cosine Similarity Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Cosine Similarity Loss")
        ax2.legend()

        # Save figure
        fig.savefig(os.path.join(dst_dir, "loss_curves.png"))

    def fit(self, train_loader, val_loader, epochs, lr=0.001):
        self.train_loader,  self.val_loader = train_loader, val_loader 

        dst_dir = f"trains/train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"
        os.makedirs(dst_dir, exist_ok=True)

        # optimizer, l1 loss, cosine similarity loss 
        optimizer, l1_criterion, cs_criterion = optim.Adam(self.parameters(), lr=lr), F.l1_loss, F.cosine_similarity

        history, optimal_loss = [], None
        for epoch in range(epochs):
            train_l1_loss, train_cs_loss = self._learn(epoch + 1, l1_criterion, cs_criterion, optimizer)
            val_l1_loss, val_cs_loss     = self._eval(epoch + 1, l1_criterion, cs_criterion, optimizer)

            history.append({
                "train_l1_loss": train_l1_loss,
                "train_cs_loss": train_cs_loss,
                "val_l1_loss": val_l1_loss,
                "val_cs_loss": val_cs_loss
            })

            # update and save the best model per epoch using cosine similarity
            if optimal_loss is None or optimal_loss > val_cs_loss:
                optimal_loss = val_cs_loss 
                torch.save(self.state_dict(), os.path.join(dst_dir, self.name))

            # log info
            print(f"Train Loss (L1): {train_l1_loss:.4f}, Train Loss (Cosine Similarity): {train_cs_loss:4f}")
            print(f"Val Loss (L1):   {val_l1_loss:.4f},   Val Loss (Cosine Similarity):   {val_cs_loss:.4f}")
            print()
        
        with torch.no_grad():
            self._save_fig(history, dst_dir)
