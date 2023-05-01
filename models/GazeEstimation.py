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

    def set_name(self, name):
        self.name = name

    def set_train_loader(self, train_loader):
        self.train_loader = train_loader
    
    def set_val_loader(self, val_loader):
        self.val_loader = val_loader

    def fit(self, epoch, l1_criterion, cs_criterion, optimizer):
        self.train()

        train_l1_loss, train_cs_loss = 0, 0
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            try:
                optimizer.zero_grad()
                output = self.forward(data)
                l1_loss = l1_criterion(output, target)
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
                pass

        train_l1_loss /= len(self.train_loader)
        train_cs_loss /= len(self.train_loader)

        return train_l1_loss, train_cs_loss

    def evaluate(self, epoch, l1_criterion, cs_criterion, optimizer):
        self.eval()

        val_cs_loss, val_l1_loss = 0, 0
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=f"Epoch {epoch}"):
                output = self.forward(data)
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

    def learn(self, num_epochs: int, lr: float):
        dst_dir = f"trains/train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"
        os.makedirs(dst_dir, exist_ok=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        l1_criterion, cs_criterion = F.l1_loss, F.cosine_similarity

        optimal_loss = None
        for epoch in range(num_epochs):
            # train 
            train_l1_loss, train_cs_loss = self.fit(epoch + 1, l1_criterion, cs_criterion, optimizer)
            # evaluate
            val_l1_loss, val_cs_loss     = self.evaluate(epoch + 1, l1_criterion, cs_criterion, optimizer)
            # update and save the best model per epoch using cosine similarity
            if optimal_loss is None or optimal_loss > val_cs_loss:
                optimal_loss = val_cs_loss 
                torch.save(self.state_dict(), os.path.join(dst_dir, self.name))
            # log info
            print(f"Train Loss (L1): {train_l1_loss:.4f}, Train Loss (Cosine Similarity): {train_cs_loss:4f}")
            print(f"Val Loss (L1): {val_l1_loss:.4f}, Val Loss (Cosine Similarity): {val_cs_loss:.4f}")