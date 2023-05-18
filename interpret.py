import cv2 
import torch 
import argparse 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from utils.file import load_model
from utils.plot import show_heat_map
from utils.runtime import available_device
from torchvision import models, transforms
from datasets.FaceDataset import FaceDataset
from utils.general import pitchyaw2xyz, angular_loss

from pytorch_grad_cam import GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget

class RegressionOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.sum(model_output)

def _RIM(model: torch.nn.Module, gaze: torch.Tensor, img: np.ndarray, kernel_size=32, stride=8):
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
            # pass through model
            pred = model(torch.Tensor(_img).permute(2,0,1).unsqueeze(0))
            heat_map[i,j] = angular_loss(
                pitchyaw2xyz(pred),
                pitchyaw2xyz(gaze)
            ).sum()

    # smooth error distribution 
    heat_map = cv2.blur(heat_map, (32, 32))
    
    # normalize the heat map
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))

    # plot result
    show_heat_map(img, heat_map, "Region Importance Map")


def _interpret_model(model: torch.nn.Module, input_tensor: torch.Tensor, img: np.ndarray):
    # construct cam object 
    cam = FullGrad(model=model, target_layers=[model.model.layer4[-1]], use_cuda=False)

    # compute heat map 
    heat_map = cam(input_tensor=input_tensor, targets=None)

    # reshape
    heat_map = heat_map.reshape((heat_map.shape[1], heat_map.shape[2], heat_map.shape[0]))

    # plot result
    show_heat_map(img, heat_map, "FullGrad")

def main(args):
    # load pretrained model
    _type, model = load_model(args.model)
    model = model.to(available_device()).eval()

    # load dataset 
    dataset = FaceDataset if _type == "face" else EyeDataset
    dataset = dataset(args.data, [f"p{args.testid:02}"], 0, 3000)

    # load image 
    input_tensor, gaze = dataset[args.idx]
    
    # show Region Importance Map (RIM)
    _RIM(
        model=model, 
        gaze=gaze.unsqueeze(0).to(available_device()),
        img=input_tensor.permute(1,2,0).numpy().astype(np.uint8), 
    )
    
    # interpret the deep cnn
    _interpret_model(
        model=model, 
        input_tensor=input_tensor.unsqueeze(0),
        img=input_tensor.permute(1,2,0).numpy().astype(np.uint8), 
    )

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
                        default=0,
                        required=False,
                        help="image index")

    args = parser.parse_args()
    main(args)