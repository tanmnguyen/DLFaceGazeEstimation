import sys
import torch
import argparse
import datetime
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
dst_name   = f"train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"

def main(args):
    lw_bound, up_bound = int(0.1 * args.upperbound), args.upperbound

    # eye-gaze model
    if args.type == "eye":
        train_dataset = EyeDataset(args.data, train_list, lw_bound=0, up_bound=up_bound)
        tuner_dataset = EyeDataset(args.data, valid_list, lw_bound=0, up_bound=lw_bound)
        valid_dataset = EyeDataset(args.data, valid_list, lw_bound=lw_bound, up_bound=up_bound)
        model         = EyeGazeEstimationModelLeNet()

    # face-gaze model
    if args.type == "face":
        train_dataset = FaceDataset(args.data, train_list, lw_bound=0, up_bound=up_bound)
        tuner_dataset = FaceDataset(args.data, valid_list, lw_bound=0, up_bound=lw_bound)
        valid_dataset = FaceDataset(args.data, valid_list, lw_bound=lw_bound, up_bound=up_bound)
        model         = FaceGazeEstimationModelLeNet()

    # build data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    tuner_loader = DataLoader(tuner_dataset, batch_size=32, shuffle=True) 
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    if args.model is not None:
        model = torch.load(args.model)
    else:
        # train model 
        model.fit(
            train_loader,
            valid_loader, 
            epochs=args.epochs,
            lr=0.0005,
            dst_dir=f"trains/{dst_name}/train"
        )

    # freeze conv layer(s)
    model.set_require_grad_conv(False)

    # calibration (tune) model
    model.fit(
        tuner_loader,
        valid_loader,
        epochs=50,
        lr=0.0005,
        dst_dir=f"trains/{dst_name}/calibration"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=False,
                        help="path to model .pt file, do calibration only if this argument is provided")

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