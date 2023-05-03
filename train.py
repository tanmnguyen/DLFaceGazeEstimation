import sys
import torch
import argparse
import numpy as np

from utils.file import build_loader
from models.EyeGazeEstimationAlexNet import EyeGazeEstimationModelAlexNet
from models.EyeGazeEstimationLeNet   import EyeGazeEstimationModelLeNet
from models.FaceGazeEstimationLeNet  import FaceGazeEstimationModelLeNet

def main(args):
    # load data 
    train_loader, val_loader = build_loader(
        args.path, 
        args.lowerbound, 
        args.upperbound,
        batch_size=128,
        loader_type=args.type,
    )

    # eye-gaze model
    if args.type == "eye":
        # init eye-based model
        model = EyeGazeEstimationModelLeNet()
        # start training
        model.fit(train_loader, val_loader, args.epochs)

    # face-gaze model
    if args.type == "face":
        # init face-based model
        model = FaceGazeEstimationModelLeNet()
        # start training
        model.fit(train_loader, val_loader, args.epochs)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-path',
                        '--path',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=50,
                        type=int,
                        required=False,
                        help="path to config model backbone (yaml file)")

    parser.add_argument('-type',
                        '--type',
                        type=str,
                        required=True,
                        help="model type: eye/face for baseline/full model")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=sys.maxsize,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")

    parser.add_argument('-lowerbound',
                        '--lowerbound',
                        default=-sys.maxsize - 1,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")


    args = parser.parse_args()
    main(args)