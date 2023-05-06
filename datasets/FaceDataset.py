import os
import cv2 
import torch
import numpy as np

from typing import List
from torch.utils.data import Dataset
from utils.general import letterbox_resize

class FaceDataset(Dataset):
    def __init__(self, data_dir: str, id_list: List[str], lw_bound: int, up_bound: int):
        self.face = [] # face image 
        self.targ = [] # gaze direction

        for pid in id_list:
            self.add(data_dir, pid, lw_bound, up_bound)

    def add(self, data_dir, pid, lw_bound, up_bound):
        assert 0 <= up_bound <= 3000
        assert 0 <= lw_bound <= 3000
        assert lw_bound <= up_bound

        images = np.load(os.path.join(data_dir, pid, "images.npy"))[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"))[lw_bound:up_bound]
        
        face = np.array([letterbox_resize(img, (224, 224)) for img in images])
        targ = np.array(gazes)

        face = torch.Tensor(face).float()
        targ = torch.Tensor(targ).float()
        face = face.permute(0, 3, 1, 2)

        if self.__len__() == 0:
            self.face = face 
            self.targ = targ 
        else:
            self.face = torch.cat([self.face, face], dim=0)
            self.targ = torch.cat([self.targ, targ], dim=0)
        
        # size check
        assert len(self.face) == len(self.targ)

    def __len__(self):
        return len(self.targ)

    def __getitem__(self, idx):
        return self.face[idx], self.targ[idx]


