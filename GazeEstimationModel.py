import yaml
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

        self.backbone = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

    def train(self, train_loader, val_loader, num_epochs: int, lr: float):
        optimizer = optim.Adam(self.backbone.parameters(), lr=lr)
        criterion = F.mse_loss

        for epoch in range(num_epochs):
            self.backbone.train()
            train_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                print(batch_idx)
                optimizer.zero_grad()
                output = self.backbone(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.backbone.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.backbone(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')