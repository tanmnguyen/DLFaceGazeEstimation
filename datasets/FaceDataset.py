import os
import torch
import numpy as np

from typing import List
from torch.utils.data import Dataset
from utils.general import letterbox_resize

class FaceDataset(Dataset):
    def __init__(self, data_dir: str, id_list: List[str], lw_bound: int, up_bound: int):
        assert 0 <= up_bound <= 3000
        assert 0 <= lw_bound <= 3000
        assert lw_bound <= up_bound

        self.face = [] # face image 
        self.targ = [] # gaze direction

        for pid in id_list:
            images = np.load(os.path.join(data_dir, pid, "images.npy"))[lw_bound:up_bound]
            gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"))[lw_bound:up_bound]

            self.face.extend([letterbox_resize(img, (224, 224)) for img in images])
            self.targ.extend(gazes)

        # to tensor
        self.face, self.targ = np.array(self.face), np.array(self.targ)

        self.face = torch.Tensor(self.face).float()
        self.targ = torch.Tensor(self.targ).float()

        # reshape
        self.face = self.face.permute(0, 3, 1, 2)

        # size check
        assert len(self.face) == len(self.targ)

    def __len__(self):
        return len(self.targ)

    def __getitem__(self, idx):
        return self.face[idx], self.targ[idx]


