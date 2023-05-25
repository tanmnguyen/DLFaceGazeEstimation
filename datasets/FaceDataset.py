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
        self.masks    = []
        self.mask_ids = []
        for pid in id_list:
            self.add(data_dir, pid, lw_bound, up_bound)

    def _load_data(self, data_dir: str, pid: str, lw_bound: int, up_bound: int):
        images = np.load(os.path.join(data_dir, pid, "images.npy"), mmap_mode='c')[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"),  mmap_mode='c')[lw_bound:up_bound]

        return images, gazes 

    def set_mask(self, masks, mode: str, save_path: str = None):
        for mask_id, mask in masks:
            self.masks.append(mask)
            self.mask_ids.append(mask_id)

        self.mask_mode = mode 

        # save example masked image 
        if len(self.masks) > 0 and save_path is not None:
            _img = self._apply_mask(self.face[0]) 
            show_image(_img, caption=f"Region Mask {mode} {self.mask_ids[0]}", save_path=save_path)

    def get_mask_name(self):
        # no mask 
        if len(self.masks) == 0:
            return ""
        
        # positive mask 
        if self.mode == "positive":
            return "-positive"

        # negative mask 
        return f"-negative-{self.masks_ids[0]}"

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
        # negative mask 
        if self.mask_mode == "negative":
            _img = np.copy(img)
            for mask in self.masks:
                _img[mask[1]: mask[3], mask[0]: mask[2]] = 255 // 2

            return _img 
        
        # positive mask 
        if self.mask_mode == "positive":
            _img = np.full_like(img, 255 // 2)
            for mask in self.masks:
                _img[mask[1]: mask[3], mask[0]: mask[2]] = img[mask[1]: mask[3], mask[0]: mask[2]]

        return _img 

    def __getitem__(self, idx):
        # build data 
        data = torch.Tensor(self._apply_mask(self.face[idx])).float().permute(2,0,1)
        targ = torch.Tensor(self.targ[idx]).float()
        return data, targ


