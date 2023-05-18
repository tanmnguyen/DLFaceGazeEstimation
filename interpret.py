import cv2 
import torch 
import datetime
import argparse 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from utils.file import load_model
from utils.runtime import available_device
from torchvision import models, transforms
from utils.plot import show_overlay_heat_map
from datasets.FaceDataset import FaceDataset
from utils.general import pitchyaw2xyz, angular_loss

from pytorch_grad_cam import GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget

from tqdm import tqdm

dst_name = f"interpret-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"

class RegressionOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.sum(model_output)

def _normalize(arr: np.ndarray):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# generate Region Importance Map based on the paper: 
# It’s Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation
def _RIM(model: torch.nn.Module, gaze: torch.Tensor, img: np.ndarray, kernel_size=64, stride=8):
    h, w = img.shape[0], img.shape[1]
    # region importance heat map 
    heat_map = np.zeros((h,w), dtype=np.float32)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # mask border 
            y = [max(0, i - kernel_size // 2), min(h, i + kernel_size // 2)]
            x = [max(0, j - kernel_size // 2), min(w, j + kernel_size // 2)]
            # apply box filter 
            _img = np.copy(img)
            _img[y[0]: y[1], x[0]: x[1]] = 255 // 2
            # prediction
            pred = model(torch.Tensor(_img).permute(2,0,1).unsqueeze(0))
            # compute angular loss 
            al = angular_loss(
                pitchyaw2xyz(pred),
                pitchyaw2xyz(gaze)
            ).sum()
            # build heat map
            idx_i = [max(0, i - stride // 2), min(h, i + stride // 2)]
            idx_j = [max(0, j - stride // 2), min(h, j + stride // 2)]
            heat_map[idx_i[0]:idx_i[1],idx_j[0]:idx_j[1]] = al.cpu().detach().numpy()

    # smooth error distribution 
    heat_map = cv2.blur(heat_map, (48, 48))

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
    _type, model = load_model(args.model)
    model = model.to(available_device()).eval() 

    # load dataset 
    dataset = FaceDataset if _type == "face" else EyeDataset
    dataset = dataset(args.data, [f"p{args.testid:02}"], 0, 3000)

    indices = np.arange(0, min(args.upperbound, len(dataset))) if args.idx is None else [args.idx]

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
            # average region importance map heat map
            avg_rim_hm = _normalize(np.max(rim_heat_map, axis=0))

            # average fullgrad heat map
            avg_fullgrad_hm = _normalize(np.max(fullgrad_heat_map, axis=0))

            print("max val", np.max(avg_rim_hm))
            # plot result
            save_path = f"interprets/{dst_name}-testid-{args.testid}"
            show_overlay_heat_map(img, avg_rim_hm, "Region Importance Map", save_path=save_path)
            show_overlay_heat_map(img, avg_fullgrad_hm, "FullGrad", save_path=save_path)

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