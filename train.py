import sys
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datasets.EyeDataset import EyeDataset
from datasets.FaceDataset import FaceDataset
from models.EyeGazeEstimationAlexNet  import EyeGazeEstimationModelAlexNet
from models.EyeGazeEstimationLeNet    import EyeGazeEstimationModelLeNet
from models.FaceGazeEstimationLeNet   import FaceGazeEstimationModelLeNet
from models.FaceGazeEstimationAlexNet import FaceGazeEstimationModelAlexNet

train_list = [f"p{id:02}" for id in range(00, 14)]
valid_list = [f"p{id:02}" for id in range(14, 15)]

def main(args):
    lw_bound, up_bound = int(0.2 * args.upperbound), args.upperbound

    # eye-gaze model
    if args.type == "eye":
        train_dataset = EyeDataset(args.data, train_list, lw_bound=0, up_bound=up_bound)

        # simulate calibration process
        for pid in valid_list:
            train_dataset.add(args.data, pid, lw_bound=0, up_bound=lw_bound)

        valid_dataset = EyeDataset(args.data, valid_list, lw_bound=lw_bound, up_bound=up_bound)
        model         = EyeGazeEstimationModelLeNet()

    # face-gaze model
    if args.type == "face":
        train_dataset = FaceDataset(args.data, train_list, lw_bound=0, up_bound=up_bound)
        
        # simulate calibration process 
        for pid in valid_list:
            train_dataset.add(args.data, pid, lw_bound=0, up_bound=lw_bound)

        valid_dataset = FaceDataset(args.data, valid_list, lw_bound=lw_bound, up_bound=up_bound)
        model         = FaceGazeEstimationModelAlexNet()

    # build data loader 
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=64,
        shuffle=False
    )

    # train model
    model.fit(
        train_loader,
        valid_loader,
        args.epochs,
        lr=0.0005,
        decay_step_size=20,
        decay_gamma=0.1,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
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
                        default=3000 ,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")


    args = parser.parse_args()
    main(args)