import cv2 
import torch 
import argparse 
import numpy as np

from utils.file import get_model
from utils.plot import draw_gaze
from utils.runtime import available_device
from datasets.FaceDataset import FaceDataset
from utils.general import pitchyaw2xyz, letterbox_resize

device = available_device()
def main(args):
    # load image 
    img = cv2.imread(args.image)
    inp = letterbox_resize(np.copy(img), (224, 224))
    inp = torch.Tensor(inp).float().permute(2,0,1)

    # load model 
    _, model = get_model(args.model)
    model.load_state_dict(torch.load(args.model))
    model = model.to(device)

    # inference 
    gaze = model(inp.unsqueeze(0))

    # show result 
    img = draw_gaze(img=img, pitchyaw=gaze)
    cv2.imshow("inference result", img)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-image',
                        '--image',
                        type=str,
                        required=True,
                        help="path to image file")

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to model weight")

    args = parser.parse_args()

    with torch.no_grad():
        main(args)