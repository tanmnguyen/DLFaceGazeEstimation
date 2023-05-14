import sys
import torch
import argparse
import datetime
import numpy as np

from utils.file import load_model
from utils.runtime import available_device
from torch.utils.data import DataLoader, random_split

from datasets.EyeDataset  import EyeDataset
from datasets.FaceDataset import FaceDataset

from models.EyeGazeEstimationAlexNet import EyeGazeEstimationModelAlexNet
from models.EyeGazeEstimationLeNet import EyeGazeEstimationModelLeNet
from models.EyeGazeEstimationResNet18 import EyeGazeEstimationModelResNet18
from models.FaceGazeEstimationLeNet import FaceGazeEstimationModelLeNet
from models.FaceGazeEstimationAlexNet import FaceGazeEstimationModelAlexNet
from models.FaceGazeEstimationResNet18 import FaceGazeEstimationModelResNet18

dst_name   = f"train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"

def _split_data(dataset, train_ratio: float, valid_ratio: float):
    n = dataset.__len__()
    train_sz = int(n * train_ratio)
    valid_sz = n - train_sz 

    return random_split(dataset, [train_sz, valid_sz])

def main(args):
    lw_bound, up_bound = int(0.1 * args.upperbound), args.upperbound

    train_list = [f"p{id:02}" for id in range(00, 14) if id != args.testid]
    test_list  = [f"p{args.testid:02}"]

    # eye-gaze model
    if args.type == "eye":
        dataset = EyeDataset
        model   = EyeGazeEstimationModelResNet18()

    # face-gaze model
    if args.type == "face":
        dataset = FaceDataset
        model   = FaceGazeEstimationModelResNet18()

    pre_tune = dataset(args.data, train_list, 0, up_bound) # pre tune data
    fne_tune = dataset(args.data, test_list,  0, up_bound) # fine tune data

    train_dataset, valid_dataset = _split_data(pre_tune, 0.9, 0.1)
    tune_dataset,  test_dataset  = _split_data(fne_tune, 0.1, 0.9)

    print(f"Pre-Tune : train_dataset {len(train_dataset):05} valid_dataset {len(valid_dataset):05}")
    print(f"Fine-Tune: tune_dataset  {len(tune_dataset):05} test_dataset  {len(test_dataset):05}")

    if args.model is not None:
        model = load_model(args.model)
    else:
        # pre-tune (train) model
        model.fit(
            DataLoader(train_dataset, batch_size=32, shuffle=True),
            DataLoader(valid_dataset, batch_size=32, shuffle=False),
            epochs=args.epochs,
            lr=0.0001,
            dst_dir=f"trains/{dst_name}/train"
        )

    # tuning configuration
    model.tune_config()

    # fine tune model
    model.fit(
        DataLoader(tune_dataset, batch_size=32 , shuffle=True),
        DataLoader(test_dataset, batch_size=32 , shuffle=False),
        epochs=30,
        lr=0.0001,
        dst_dir=f"trains/{dst_name}/tune-testid-{args.testid}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-model',
                        '--model',
                        required=False,
                        help="path to model .pt file. Only do calibration tuning if this argument is provided")

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