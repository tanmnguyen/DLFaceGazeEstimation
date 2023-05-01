import os
import yaml
import torch
import datetime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from utils.general import yawpitch2xyz

class GazeEstimationModel(nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        self.name  = "gaze-estimation-model.pt"

    def fit(self, train_loader, val_loader, epochs, lr=0.001):
        self.train_loader = train_loader
        self.val_loader   = val_loader 

        dst_dir = f"trains/train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"
        os.makedirs(dst_dir, exist_ok=True)

        # optimizer, l1 loss, cosine similarity loss 
        optimizer, l1_criterion, cs_criterion = optim.Adam(self.parameters(), lr=lr), F.l1_loss, F.cosine_similarity

        optimal_loss = None
        for epoch in range(epochs):
            train_l1_loss, train_cs_loss = self._learn(epoch + 1, l1_criterion, cs_criterion, optimizer)
            val_l1_loss, val_cs_loss     = self._eval(epoch + 1, l1_criterion, cs_criterion, optimizer)

            # update and save the best model per epoch using cosine similarity
            if optimal_loss is None or optimal_loss > val_cs_loss:
                optimal_loss = val_cs_loss 
                torch.save(self.state_dict(), os.path.join(dst_dir, self.name))

            # log info
            print(f"Train Loss (L1): {train_l1_loss:.4f}, Train Loss (Cosine Similarity): {train_cs_loss:4f}")
            print(f"Val Loss (L1):   {val_l1_loss:.4f},   Val Loss (Cosine Similarity):   {val_cs_loss:.4f}")
            print()