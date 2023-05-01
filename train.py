import sys
import torch
import argparse
import numpy as np

from dataset import EyeDataset
from utils.file import read_images_and_labels
from torch.utils.data import Dataset, DataLoader
from models.EyeGazeEstimation import EyeGazeEstimationModel

def train_eye_model(data_dict, epochs: int):
    # train data
    train_images, train_labels, train_llmarks, train_rlmarks = [], [], [], []
    for i in range(0, 14):
        train_images.extend(data_dict[str(i)]['images'])
        train_labels.extend(data_dict[str(i)]['labels'])
        train_llmarks.extend(data_dict[str(i)]['left_landmarks'])
        train_rlmarks.extend(data_dict[str(i)]['right_landmarks'])

    # validation data 
    val_images, val_labels, val_llmarks, val_rlmarks = [], [], [], []
    for i in range(14, 15):
        val_images.extend(data_dict[str(i)]['images'])
        val_labels.extend(data_dict[str(i)]['labels'])
        val_llmarks.extend(data_dict[str(i)]['left_landmarks'])
        val_rlmarks.extend(data_dict[str(i)]['right_landmarks'])

    eyeGazeEstimationModel = EyeGazeEstimationModel()

    eyeGazeEstimationModel.set_train_loader(
        DataLoader(EyeDataset(train_images, train_labels, train_llmarks, train_rlmarks), batch_size=2, shuffle=True)
    )

    eyeGazeEstimationModel.set_val_loader(
        DataLoader(EyeDataset(val_images, val_labels, val_llmarks, val_rlmarks), batch_size=2, shuffle=True)
    )

    eyeGazeEstimationModel.learn(epochs, lr=0.001)

def main(args):
    # load data 
    data_dict = read_images_and_labels(args.path, upper_bound=args.upperbound)
    # eye-gaze model
    if args.type == "eye":
        train_eye_model(data_dict, epochs=args.epochs)
        
    # face-gaze model
    if args.type == "face":
        # TODO
        pass
    
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