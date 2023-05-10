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

    def _load_data(self, data_dir: str, pid: str, lw_bound: int, up_bound: int):
        images = np.load(os.path.join(data_dir, pid, "images.npy"))[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"))[lw_bound:up_bound]

        return images, gazes 

    def add(self, data_dir, pid, lw_bound, up_bound):
        images, gazes = self._load_data(data_dir, pid, lw_bound, up_bound)

        self.face.extend([letterbox_resize(img, (224, 224)) for img in images])
        self.targ.extend(gazes)
        
    def __len__(self):
        # size check
        assert len(self.face) == len(self.targ)
        # get length
        return len(self.targ)

    def __getitem__(self, idx):
        data = torch.Tensor(self.face[idx]).float().permute(2,0,1)
        targ = torch.Tensor(self.targ[idx]).float()
        return data, targ


