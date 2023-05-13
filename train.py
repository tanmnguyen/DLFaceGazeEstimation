import sys
import torch
import argparse
import datetime
import numpy as np

from utils.file           import load_model
from torch.utils.data     import DataLoader
from datasets.EyeDataset  import EyeDataset
from datasets.FaceDataset import FaceDataset
from utils.runtime        import available_device
from models.EyeGazeEstimationAlexNet   import EyeGazeEstimationModelAlexNet
from models.EyeGazeEstimationLeNet     import EyeGazeEstimationModelLeNet
from models.FaceGazeEstimationLeNet    import FaceGazeEstimationModelLeNet
from models.FaceGazeEstimationAlexNet  import FaceGazeEstimationModelAlexNet
from models.FaceGazeEstimationResNet18 import FaceGazeEstimationModelResNet18

train_list = [f"p{id:02}" for id in range(00, 14)]
valid_list = [f"p{id:02}" for id in range(14, 15)]
dst_name   = f"train-{datetime.datetime.now().strftime('%Y%m%d %H%M%S')}"

def main(args):
    lw_bound, up_bound = min(100, int(0.1 * args.upperbound)), args.upperbound

    # eye-gaze model
    if args.type == "eye":
        train_dataset = EyeDataset(args.data, train_list, 0, up_bound)
        tuner_dataset = EyeDataset(args.data, valid_list, 0, lw_bound)
        valid_dataset = EyeDataset(args.data, valid_list, lw_bound, up_bound)
        model         = EyeGazeEstimationModelLeNet()

    # face-gaze model
    if args.type == "face":
        train_dataset = FaceDataset(args.data, train_list, 0, up_bound)
        tuner_dataset = FaceDataset(args.data, valid_list, 0, lw_bound)
        valid_dataset = FaceDataset(args.data, valid_list, lw_bound, up_bound)
        model         = FaceGazeEstimationModelAlexNet()
        
    model = model.to(available_device())

    print("Train Dataset:", train_dataset.__len__())
    print("Tuner Dataset:", tuner_dataset.__len__())
    print("Valid Dataset:", valid_dataset.__len__())

    # pre-tune data 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # fine-tune data 
    tuner_loader = DataLoader(tuner_dataset, batch_size=32, shuffle=True)
    # validation data 
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    if args.model is not None:
        model = load_model(args.model)
    else:
        # train model 
        model.fit(
            train_loader,
            valid_loader, 
            epochs=args.epochs,
            lr=0.0001,
            dst_dir=f"trains/{dst_name}/train"
        )

    # tuning configuration
    model.tune_config()

    # tune model
    model.fit(
        tuner_loader, 
        valid_loader, 
        epochs=100,
        lr=0.0001,
        dst_dir=f"trains/{dst_name}/tune(calibration)"
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

    parser.add_argument('-upperbound',
                        '--upperbound',
                        default=3000 ,
                        type=int,
                        required=False,
                        help="upper bound for image per directory")


    args = parser.parse_args()
    main(args)