import os
import yaml
import torch
import datetime

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from BuildDataset import BuildDataset
from utils.general import yawpitch2xyz
from torch.utils.data import Dataset, DataLoader

class GazeEstimationModel(nn.Module):
    def __init__(self, config_path: str):
        super(GazeEstimationModel, self).__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        layers = []
        for i, layer_cfg in enumerate(config['backbone']):
            module_name = layer_cfg[2]
            module_args = layer_cfg[3]
            module = getattr(nn, module_name)(*module_args)
            layers.append(module)

        self.model_name = os.path.basename(config_path).split('.')[0] + ".pt"
        self.backbone   = nn.Sequential(*layers)
        
    def _build_dataloader(self, images, labels):
        # convert images to tensor
        images_tensor = torch.from_numpy(images).float()
        # convert labels to tensor
        labels_tensor = torch.from_numpy(labels).float()
        # build train data loader
        return DataLoader(BuildDataset(images_tensor, labels_tensor), batch_size=32, shuffle=True)

    def _read_and_process(self, data_dict, indices): 
        images, labels = [], []
        for idx in indices:
            images.extend(data_dict[str(idx)]['images'])
            labels.extend(data_dict[str(idx)]['labels'])
        # to numpy 
        images, labels = np.array(images), np.array(labels)
        # re-order shape
        images = images.transpose((0, 3, 1, 2))
        # result
        return images, labels

    def load_data(self, data_dict, train_indices=np.arange(0,14), val_indices=np.arange(14, 15)):
        # training data 
        train_images, train_labels = self._read_and_process(data_dict, train_indices)
        # validation data 
        val_images, val_labels     = self._read_and_process(data_dict, val_indices)
        # build training data loader
        self.train_loader          = self._build_dataloader(train_images, train_labels)
        # build validation data loader
        self.val_loader            = self._build_dataloader(val_images, val_labels)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def train(self, num_epochs: int, lr: float):
        dst_dir = f"trains/train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"
        os.makedirs(dst_dir, exist_ok=True)

        optimizer = optim.Adam(self.backbone.parameters(), lr=lr)
        # L1 loss for training
        l1_criterion = F.l1_loss
        # cosine similarity loss for validation
        cs_criterion = F.cosine_similarity
        # start training 
        opt_loss = None
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch+1}")

            # train mode
            self.backbone.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                try:
                    optimizer.zero_grad()
                    output = self.backbone(data)
                    loss = l1_criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                except:
                    pass

            train_loss /= len(self.train_loader)

            # Validation mode
            self.backbone.eval()
            avg_cs_loss, avg_l1_loss = 0, 0
            with torch.no_grad():
                for data, target in tqdm(self.val_loader):
                    output = self.backbone(data)
                    # compute l1 loss
                    l1_loss = l1_criterion(output, target)

                    # compute cosine similarity loss
                    cs_loss = cs_criterion(
                        torch.tensor(yawpitch2xyz(output.numpy())).float(), 
                        torch.tensor(yawpitch2xyz(target.numpy())).float()
                    )

                    avg_l1_loss += l1_loss.item()
                    avg_cs_loss += torch.sum(cs_loss)

            avg_cs_loss /= len(self.val_loader)
            avg_l1_loss /= len(self.val_loader)

            # update and save the best model per epoch using cosine similarity
            if opt_loss is None or opt_loss > avg_cs_loss:
                opt_loss = avg_cs_loss 
                torch.save(self.backbone.state_dict(), os.path.join(dst_dir, self.model_name))

            print(f'Train Loss (L1): {train_loss:.4f}, Val Loss (Cosine Similarity): {avg_cs_loss:.4f}, Val Loss (L1): {avg_l1_loss:.4f}\n')