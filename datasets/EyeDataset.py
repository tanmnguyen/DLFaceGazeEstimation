import os
import cv2 
import torch
import traceback
import numpy as np

from typing import List
from torch.utils.data import Dataset
from utils.detect import extract_bbox_img
from utils.general import letterbox_resize
from utils.plot import draw_bboxes, draw_gaze

class EyeDataset(Dataset):
    def __init__(self, data_dir: str, id_list: List[str], lw_bound: int, up_bound: int):
        self.leye = [] # left eye image 
        self.reye = [] # righ eye image 
        self.targ = [] # gaze direction 

        for pid in id_list:
            self.add(data_dir, pid, lw_bound, up_bound)

    def _load_data(self, data_dir: str, pid: str, lw_bound: int, up_bound: int):
        images = np.load(os.path.join(data_dir, pid, "images.npy"))[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"))[lw_bound:up_bound]
        lboxes = np.load(os.path.join(data_dir, pid, "l_eye_bboxes.npy"))[lw_bound:up_bound]
        rboxes = np.load(os.path.join(data_dir, pid, "r_eye_bboxes.npy"))[lw_bound:up_bound]
        lcorns = np.load(os.path.join(data_dir, pid, "l_eye_corners.npy"))[lw_bound:up_bound]
        rcorns = np.load(os.path.join(data_dir, pid, "r_eye_corners.npy"))[lw_bound:up_bound]

        return images, gazes, lboxes, rboxes, lcorns, rcorns
    
    def add(self, data_dir, pid: str, lw_bound: int, up_bound: int):
        images, gazes, lboxes, rboxes, lcorns, rcorns = self._load_data(data_dir, pid, lw_bound, up_bound)
        for idx, img in enumerate(images):
            try:
                leye_img = extract_bbox_img(img, lboxes[idx])
                reye_img = extract_bbox_img(img, rboxes[idx])
                # resize 
                leye_img = letterbox_resize(leye_img, (43, 73))
                reye_img = letterbox_resize(reye_img, (43, 73))
                # save 
                self.leye.append(leye_img)
                self.reye.append(reye_img)
                self.targ.append(gazes[idx])
            except:
                pass 

    def __len__(self):
        # size check
        assert len(self.targ) == len(self.leye) == len(self.reye)
        # get length
        return len(self.targ)

    def __getitem__(self, idx):
        data = [
            torch.Tensor(self.leye[idx]).float().permute(2,0,1),
            torch.Tensor(self.reye[idx]).float().permute(2,0,1),
        ]
        targ = torch.Tensor(self.targ[idx]).float()
        return data, targ


