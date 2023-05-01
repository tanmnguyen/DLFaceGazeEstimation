import torch
import numpy as np
from utils.general import letterbox_resize
from torch.utils.data import Dataset, DataLoader

class DefaultDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

class EyeDataset(Dataset):
    def __init__(self, images, labels, llmarks, rlmarks):
        self.images = images 
        self.labels = labels

        self.l_eye_imgs = []
        self.r_eye_imgs = []

        for i, img in enumerate(images):
            self.l_eye_imgs.append(img[
                llmarks[i][1]: llmarks[i][3],
                llmarks[i][0]: llmarks[i][2],
            ])
            self.l_eye_imgs[-1] = letterbox_resize(self.l_eye_imgs[-1], (224, 224))

            self.r_eye_imgs.append(img[
                rlmarks[i][1]: rlmarks[i][3],
                rlmarks[i][0]: rlmarks[i][2],
            ])
            self.r_eye_imgs[-1] = letterbox_resize(self.r_eye_imgs[-1], (224, 224))

        self.images = torch.from_numpy(np.array(self.images)).float()
        self.labels = torch.from_numpy(np.array(self.labels)).float()
        self.r_eye_imgs = torch.from_numpy(np.array(self.r_eye_imgs)).float()
        self.l_eye_imgs = torch.from_numpy(np.array(self.l_eye_imgs)).float()

        self.images = self.images.permute(0, 3, 1, 2)
        self.r_eye_imgs = self.r_eye_imgs.permute(0, 3, 1, 2)
        self.l_eye_imgs = self.l_eye_imgs.permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        l_eye = self.l_eye_imgs[idx]
        r_eye = self.r_eye_imgs[idx]
        image = self.images[idx]
        label = self.labels[idx]

        return l_eye, r_eye, image, label