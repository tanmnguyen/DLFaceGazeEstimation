import os 
import torch
import argparse 
import datetime 
import numpy as np

from utils.file import get_model
from utils.general import split_data
from torch.utils.data import DataLoader

from datasets.EyeDataset  import EyeDataset
from datasets.FaceDataset import FaceDataset

dst_name = f"tests/test-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"

def main(args):
    test_list  = [f"p{args.testid:02}"]

    # get model 
    _type, model = get_model(args.model)

    # load dataset type 
    dataset = EyeDataset if _type == "eye" else FaceDataset

    # load data 
    fine_tune = dataset(args.data, test_list,  0, up_bound=args.upperbound) 

    # load mask regions
    regions = [None] if args.regions is None else np.load(args.regions)

    # define save_path
    save_path=f"{dst_name}-testid-{args.testid}"
    os.makedirs(save_path, exist_ok=True)

    for region_id, region in enumerate(regions):
        # load weight
        model.load_state_dict(torch.load(args.model))

        # tuning configuration
        model.tune_config()

        # set mask information 
        fine_tune.set_mask(region, region_id, save_path=save_path)

        # tune and test set
        tune_dataset, test_dataset  = split_data(fine_tune, 0.1, 0.9)

        # fine tune model
        model.fit(
            DataLoader(tune_dataset, batch_size=32 , shuffle=True),
            DataLoader(test_dataset, batch_size=32 , shuffle=False),
            epochs=args.epochs,
            lr=0.0001,
            dst_dir=save_path
        )

        # average angular error for the last 10 epochs to obtain final test error value
        valid_angular_err = np.mean(list(map(lambda his: his["valid_ma_loss"], model.epoch_history[-10:])))

        # define result save path
        result_path = f"{dst_name}-testid-{args.testid}"
        result_name = f"result.txt" if args.regions is None else f"result-region-{region_id}.txt"

        # save result 
        with open(os.path.join(result_path, result_name), 'w') as file:
            content = f"Validation Angular Error: {valid_angular_err}"
            print(content)
            file.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data file")

    parser.add_argument('-model',
                        '--model',
                        type=str,
                        required=True,
                        help="path to a pretrained model .pt file")

    parser.add_argument('-epochs',
                        '--epochs',
                        default=30,
                        type=int,
                        required=False,
                        help="number of epochs to calibrate the model on the test data")

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

    parser.add_argument('-regions',
                        '--regions',
                        type=str,
                        required=False,
                        help="path to heat map mask regions (.npy file)")

    args = parser.parse_args()
    main(args)