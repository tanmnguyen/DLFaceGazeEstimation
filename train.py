import sys
import torch
import argparse
import numpy as np

from BuildDataset import BuildDataset
from utils.file import read_images_and_labels
from torch.utils.data import Dataset, DataLoader
from GazeEstimationModel import GazeEstimationModel

def main(args):
    # load data 
    data_dict = read_images_and_labels(args.path, upper_bound=args.upperbound)
    # build model 
    gaze_est_model = GazeEstimationModel(config_path=args.config) 
    # dummy data for testing purpose
    images = torch.from_numpy(np.random.rand(3, 3, 224, 224)).float()
    labels = torch.from_numpy(np.array([[1, 2], [3, 4], [5, 6]])).float()
    # build data set for training
    dataset    = BuildDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # train model
    gaze_est_model.train(dataloader, dataloader, num_epochs=args.epochs, lr=0.001)

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