import os
import cv2 
import torch 
import random
import datetime
import argparse 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from utils.file import get_model
from utils.runtime import available_device
from torchvision import models, transforms
from utils.plot import show_overlay_heat_map
from datasets.FaceDataset import FaceDataset
from utils.general import pitchyaw2xyz, angular_loss

from pytorch_grad_cam import GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget

dst_name = f"interpret-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"

def _normalize(arr: np.ndarray):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def _mean_max(heat_map: np.ndarray, batch_size: int = 85):
    n       = len(heat_map)
    batches = [heat_map[i:min(n, i + batch_size)] for i in range(0, n, batch_size)]  # Divide into batches

    for i in range(len(batches)):
        batches[i] = np.max(batches[i], axis=0)

    return np.mean(batches, axis=0)

# generate Region Importance Map based on the paper: 
# It’s Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation
def _RIM(model: torch.nn.Module, gaze: torch.Tensor, img: np.ndarray, kernel_size=32, stride=16):
    h, w = img.shape[0], img.shape[1]

    imgs, indices = [], []
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # mask border 
            y = [max(0, i - kernel_size // 2), min(h, i + kernel_size // 2)]
            x = [max(0, j - kernel_size // 2), min(w, j + kernel_size // 2)]

            # apply box filter 
            _img = np.copy(img)
            _img[y[0]: y[1], x[0]: x[1]] = 255 // 2

            # save 
            imgs.append(_img)
            indices.append((i,j))


    selected = random.choices(np.arange(0, len(imgs)), k=5)
    for idx in selected:
        cv2.imwrite(f"Index: {idx} Pos {indices[idx][0]}, {indices[idx][1]}.png", imgs[idx])

    # convert to tensor 
    imgs = np.array(imgs)
    imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
    
    # predict 
    gazes = model(imgs)

    # region importance heat map 
    heat_map = np.zeros((h,w), dtype=np.float32)
    for idx, pred in enumerate(gazes):
        i, j = indices[idx][0], indices[idx][1]

        # compute angular loss 
        al = angular_loss(pitchyaw2xyz(pred.unsqueeze(0)), pitchyaw2xyz(gaze)).sum()

        # fill indices 
        idx_i = [max(0, i - stride // 2), min(h, i + stride // 2)]
        idx_j = [max(0, j - stride // 2), min(w, j + stride // 2)]

        # edge case
        if i + stride >= h:
            idx_i[1] = h
        
        if j + stride >= w:
            idx_j[1] = w

        # assign heat map value 
        heat_map[idx_i[0]:idx_i[1],idx_j[0]:idx_j[1]] = al.cpu().detach().numpy()    

    # smooth error distribution 
    heat_map = cv2.blur(heat_map, (32, 32))

    # normalize the heat map
    heat_map = _normalize(heat_map)

    # result
    return heat_map

def _interpret_model(model: torch.nn.Module, input_tensor: torch.Tensor, img: np.ndarray):
    # construct cam object 
    cam = FullGrad(model=model, target_layers=[model.model.layer4[-1]], use_cuda=False)

    # compute heat map 
    heat_map = cam(input_tensor=input_tensor)

    # reshape
    heat_map = heat_map.reshape((heat_map.shape[1], heat_map.shape[2], heat_map.shape[0]))

    # result
    return heat_map 

def main(args):
    # load pretrained model
    _type, model = get_model(args.model)
    model = model.to(available_device()).eval() 

    # load dataset 
    dataset = FaceDataset if _type == "face" else EyeDataset
    dataset = dataset(args.data, [f"p{args.testid:02}"], 0, 3000)

    indices = np.arange(0, min(args.upperbound, len(dataset))) if args.idx is None else [args.idx]

    # save destination 
    save_path = f"interprets/{dst_name}-testid-{args.testid}"
    os.makedirs(save_path, exist_ok=True)

    rim_heat_map, fullgrad_heat_map = [], []
    for enum, idx in enumerate(indices):
        print(f"Processing Image {idx}")
        # load image 
        input_tensor, gaze = dataset[idx]
        img = input_tensor.permute(1,2,0).numpy().astype(np.uint8)
        
        # show Region Importance Map (RIM)
        rim_heat_map.append(_RIM(
            model=model, 
            gaze=gaze.unsqueeze(0).to(available_device()),
            img=img,
        ))
        
        # interpret the deep cnn
        fullgrad_heat_map.append(_interpret_model(
            model=model, 
            input_tensor=input_tensor.unsqueeze(0),
            img=img,
        ))

        if enum % 10 == 0 or enum == len(indices) - 1:            
            # average region importance map
            avg_rim_hm = _normalize(_mean_max(rim_heat_map))
            
            # average fullgrad heat map
            avg_fullgrad_hm = _normalize(_mean_max(fullgrad_heat_map))
            
            # save heat maps
            np.save(os.path.join(save_path, "region_importance_map.npy"), avg_rim_hm)
            np.save(os.path.join(save_path, "full_grad_heat_map.npy"), avg_fullgrad_hm)

            # plot and save result
            show_overlay_heat_map(img, avg_rim_hm, "Region Importance Map", save_path=save_path)
            show_overlay_heat_map(img, avg_fullgrad_hm, "FullGrad", save_path=save_path)

            print("checkpoint saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to a pretrained model .pt file")

    parser.add_argument('-data',
                        '--data',
                        type=str,
                        required=True,
                        help="path to an data folder")

    parser.add_argument('-testid',
                        '--testid',
                        type=int,
                        required=True,
                        help="id of the test subject")

    parser.add_argument('-idx',
                        '--idx',
                        type=int,
                        required=False,
                        help="image index. Average all images if not specified")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=3000,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")


    args = parser.parse_args()
    main(args)