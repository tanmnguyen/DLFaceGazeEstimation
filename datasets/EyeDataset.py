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
        assert 0 <= up_bound <= 3000
        assert 0 <= lw_bound <= 3000
        assert lw_bound <= up_bound

        self.leye = [] # left eye image 
        self.reye = [] # righ eye image 
        self.targ = [] # gaze direction 

        for pid in id_list:
            images = np.load(os.path.join(data_dir, pid, "images.npy"))[lw_bound:up_bound]
            gazes  = np.load(os.path.join(data_dir, pid, "gazes.npy"))[lw_bound:up_bound]
            lboxes = np.load(os.path.join(data_dir, pid, "l_eye_bboxes.npy"))[lw_bound:up_bound]
            rboxes = np.load(os.path.join(data_dir, pid, "r_eye_bboxes.npy"))[lw_bound:up_bound]
            lcorns = np.load(os.path.join(data_dir, pid, "l_eye_corners.npy"))[lw_bound:up_bound]
            rcorns = np.load(os.path.join(data_dir, pid, "r_eye_corners.npy"))[lw_bound:up_bound]

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
                    self.leye.append(leye_img)
                    self.reye.append(reye_img)
                    self.targ.append(gazes[idx])
                except:
                    pass
        # to tensor
        self.leye, self.reye, self.targ = np.array(self.leye), np.array(self.reye), np.array(self.targ)

        self.leye = torch.Tensor(self.leye).float()
        self.reye = torch.Tensor(self.reye).float()
        self.targ = torch.Tensor(self.targ).float()

        # reshape 
        self.leye = self.leye.permute(0, 3, 1, 2)
        self.reye = self.reye.permute(0, 3, 1, 2)
        
        # size check
        assert len(self.leye) == len(self.reye)
        assert len(self.leye) == len(self.targ)

    def __len__(self):
        return len(self.targ)

    def __getitem__(self, idx):
        return [self.leye[idx], self.reye[idx]], self.targ[idx]


