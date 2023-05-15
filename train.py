import sys
import torch
import argparse
import datetime
import numpy as np

from utils.general import split_data

from torch.utils.data import DataLoader
from datasets.EyeDataset  import EyeDataset
from datasets.FaceDataset import FaceDataset

from models.EyeGazeEstimationResNet18  import EyeGazeEstimationModelResNet18
from models.FaceGazeEstimationResNet18 import FaceGazeEstimationModelResNet18

dst_name = f"train-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"

def main(args):
    train_list = [f"p{id:02}" for id in range(00, 14) if id != args.testid]

    # eye-gaze model
    if args.type == "eye":
        dataset = EyeDataset
        model   = EyeGazeEstimationModelResNet18()

    # face-gaze model
    if args.type == "face":
        dataset = FaceDataset
        model   = FaceGazeEstimationModelResNet18()

    # pre tune data
    pre_tune = dataset(args.data, train_list, 0, up_bound=args.upperbound) 

    # train and validation set 
    train_dataset, valid_dataset = split_data(pre_tune, 0.9, 0.1)

    # pre-tune (train) model
    model.fit(
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        DataLoader(valid_dataset, batch_size=32, shuffle=False),
        epochs=args.epochs,
        lr=0.0001,
        dst_dir=f"trains/{dst_name}-testid-{args.testid}"
    )   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=20,
                        type=int,
                        required=False,
                        help="number of epochs to train the model")

    parser.add_argument('-type',
                        '--type',
                        type=str,
                        required=True,
                        help="model type: [eye/face] models")

    parser.add_argument('-testid',
                        '--testid',
                        type=int,
                        required=True,
                        help="test id")

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=3000 ,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")


    args = parser.parse_args()
    main(args)