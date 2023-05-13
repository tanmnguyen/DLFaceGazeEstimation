import os
import cv2 
import torch
import torchvision
import numpy as np

from typing import List
from utils.plot import draw_gaze
from torch.utils.data import Dataset
from utils.general import letterbox_resize

class FaceDataset(Dataset):
    def __init__(self, data_dir: str, id_list: List[str], lw_bound: int, up_bound: int):
        self.face = [] # face image 
        self.targ = [] # gaze direction
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: letterbox_resize(x, (448, 448))), # resize 
            torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)), # re-order dimension
            torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255), # scale 
            torch.from_numpy,   # to tensor 
            torchvision.transforms.Normalize(
                mean = [0.406, 0.456, 0.485],
                std  = [0.225, 0.224, 0.229]
            ), 
        ])

        for pid in id_list:
            self.add(data_dir, pid, lw_bound, up_bound)

    def _load_data(self, data_dir: str, pid: str, lw_bound: int, up_bound: int):
        images = np.load(os.path.join(data_dir, pid, "images.npy"), mmap_mode='c')[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"),  mmap_mode='c')[lw_bound:up_bound]

        return images, gazes 

    def add(self, data_dir, pid, lw_bound, up_bound):
        images, gazes = self._load_data(data_dir, pid, lw_bound, up_bound)

        self.face.extend(images)
        self.targ.extend(gazes)
        
    def __len__(self):
        # size check
        assert len(self.face) == len(self.targ)
        # get length
        return len(self.targ)

    def __getitem__(self, idx):
        data = self.transform(self.face[idx])
        targ = torch.Tensor(self.targ[idx])
        return data, targ


