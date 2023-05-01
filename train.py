import sys
import torch
import argparse
import numpy as np

from utils.file import read_images_and_labels
from EyeGazeEstimation import EyeGazeEstimationModel

def main(args):
    # load data 
    data_dict = read_images_and_labels(args.path, upper_bound=args.upperbound)
    # eye-gaze model
    if args.type == "eye":
        # init model
        eye_gaze_model = EyeGazeEstimationModel(config_path=args.config)
        # build training data
        eye_gaze_model.load_data(data_dict)
        # train 
        eye_gaze_model.train(num_epochs=args.epochs, lr=0.001)
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

    parser.add_argument('-config',
                        '--config',
                        required=True,
                        help="path to config model backbone (yaml file)")

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