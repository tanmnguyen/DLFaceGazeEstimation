import os
import cv2 
import torch
import numpy as np

from typing import List
from torch.utils.data import Dataset

from utils.plot import show_image
from utils.general import letterbox_resize

class FaceDataset(Dataset):
    def __init__(self, data_dir: str, id_list: List[str], lw_bound: int, up_bound: int):
        self.face = [] # face image 
        self.targ = [] # gaze direction

        for pid in id_list:
            self.add(data_dir, pid, lw_bound, up_bound)

    def _load_data(self, data_dir: str, pid: str, lw_bound: int, up_bound: int):
        images = np.load(os.path.join(data_dir, pid, "images.npy"), mmap_mode='c')[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"),  mmap_mode='c')[lw_bound:up_bound]

        return images, gazes 

    def set_mask(self, mask: np.ndarray, id: int, save_path: str = None):
        self.mask = mask

        # save mask example
        if mask is not None and save_path is not None:
            show_image(self._apply_mask(self.face[0]), f"Region Mask {id}", save_path)


    def add(self, data_dir, pid, lw_bound, up_bound):
        images, gazes = self._load_data(data_dir, pid, lw_bound, up_bound)

        self.face.extend([letterbox_resize(img, (224, 224)) for img in images])
        self.targ.extend(gazes)
        
    def __len__(self):
        # size check
        assert len(self.face) == len(self.targ)
        # get length
        return len(self.targ)

    def _apply_mask(self, img: np.ndarray):
        # original image 
        if self.mask is None:
            return img 

        # apply mask 
        _img = np.copy(img)
        _img[self.mask[1]: self.mask[3], self.mask[0] : self.mask[2]] = 255 / 2

        return _img
    def __getitem__(self, idx):
        # build data 
        data = torch.Tensor(self._apply_mask(self.face[idx])).float().permute(2,0,1)
        targ = torch.Tensor(self.targ[idx]).float()
        return data, targ


