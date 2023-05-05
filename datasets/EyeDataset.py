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

    def add(self, data_dir: str, pid: str, lw_bound: int, up_bound: int):
        assert 0 <= up_bound <= 3000
        assert 0 <= lw_bound <= 3000
        assert lw_bound <= up_bound

        images = np.load(os.path.join(data_dir, pid, "images.npy"))[lw_bound:up_bound]
        gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"))[lw_bound:up_bound]
        lboxes = np.load(os.path.join(data_dir, pid, "l_eye_bboxes.npy"))[lw_bound:up_bound]
        rboxes = np.load(os.path.join(data_dir, pid, "r_eye_bboxes.npy"))[lw_bound:up_bound]
        lcorns = np.load(os.path.join(data_dir, pid, "l_eye_corners.npy"))[lw_bound:up_bound]
        rcorns = np.load(os.path.join(data_dir, pid, "r_eye_corners.npy"))[lw_bound:up_bound]

        leye, reye, targ = [], [], []

        # extract eye images 
        for idx, img in enumerate(images):
            # check normalized property
            assert -1 <= gazes[idx][0] <= 1 and -1 <= gazes[idx][1] <= 1
            try:
                leye_img = extract_bbox_img(img, lboxes[idx])
                reye_img = extract_bbox_img(img, rboxes[idx])
                # resize 
                leye_img = letterbox_resize(leye_img, (43, 73))
                reye_img = letterbox_resize(reye_img, (43, 73))
                # save 
                leye.append(leye_img)
                reye.append(reye_img)
                targ.append(gazes[idx])
            except:
                pass

        leye, reye, targ = np.array(leye), np.array(reye), np.array(targ)

        leye = torch.Tensor(leye).float()
        reye = torch.Tensor(reye).float()
        targ = torch.Tensor(targ).float()

        leye = leye.permute(0, 3, 1, 2)
        reye = reye.permute(0, 3, 1, 2)
        
        if self.__len__() == 0:
            self.leye = leye
            self.reye = reye
            self.targ = targ
        else:
            self.leye = torch.cat([self.leye, leye], dim=0)
            self.reye = torch.cat([self.reye, reye], dim=0)
            self.targ = torch.cat([self.targ, targ], dim=0)

        # size check
        assert len(self.leye) == len(self.reye)
        assert len(self.leye) == len(self.targ)
        
    def __len__(self):
        return len(self.targ)

    def __getitem__(self, idx):
        return [self.leye[idx], self.reye[idx]], self.targ[idx]


