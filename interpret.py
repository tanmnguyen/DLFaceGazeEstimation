import cv2 
import torch 
import argparse 
import numpy as np
from utils.file import load_model
from utils.runtime import available_device

from torchvision import models, transforms
from datasets.FaceDataset import FaceDataset

from pytorch_grad_cam import GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, RawScoresOutputTarget

class RegressionOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.sum(model_output)

def main(args):
    # load pretrained model
    _type, model = load_model(args.model)
    model = model.to(available_device()).eval()

    # read image 
    rgb_img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img.astype(np.float32) / 255.0

    # convert to tensor 
    input_tensor = torch.Tensor(rgb_img).float().permute(2,0,1)
    input_tensor = input_tensor.unsqueeze(0)

    # Construct the CAM object once, and then re-use it on many images:
    cam = FullGrad(model=model, target_layers=[model.model.layer4[-1]], use_cuda=False)

    targets = [RegressionOutputTarget()]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # grayscale_cam = np.transpose(grayscale_cam, (1, 2, 0))[:, :, 0]
    # cv2.imshow("GradCam", grayscale_cam)
    # cv2.waitKey(0)
    
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    cv2.imshow("GradCam", visualization)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to a pretrained model .pt file")

    parser.add_argument('-image',
                        '--image',
                        type=str,
                        required=True,
                        help="path to an image file")

    args = parser.parse_args()
    main(args)