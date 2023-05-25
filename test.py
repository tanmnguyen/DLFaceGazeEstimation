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


def fine_tune(args, model, fine_tune_dataset, save_path):
    # load weight
    model.load_state_dict(torch.load(args.model))

    # tuning configuration
    model.tune_config()

    # tune and test set
    tune_dataset, test_dataset  = split_data(fine_tune_dataset, 0.1, 0.9)

    # fine tune model
    model.fit(
        DataLoader(tune_dataset, batch_size=32 , shuffle=True),
        DataLoader(test_dataset, batch_size=32 , shuffle=False),
        epochs=args.epochs,
        lr=0.0001,
        dst_dir=save_path
    )

def eval(args, model, fine_tune_dataset):
    # average angular error for the last 10 epochs to obtain final test error value
    valid_angular_err = np.mean(
        list(map(lambda his: his["valid_ma_loss"], model.epoch_history[-10:]))
    )

    # define result save path
    result_path = f"{dst_name}-testid-{args.testid}"

    # define result file
    result_name = f"result{fine_tune_dataset.get_mask_name()}.txt" 

    # save result 
    with open(os.path.join(result_path, result_name), 'w') as file:
        content = f"Validation Angular Error: {valid_angular_err}"
        file.write(content)
    

def process_negative_mask(args, model, fine_tune_dataset, mask_regions, save_path):
    for region_id, region in enumerate(mask_regions):
        # set single mask 
        fine_tune_dataset.set_mask([(region_id, region)], mode=args.mode, save_path=save_path)
        # fine tune (calibration)
        fine_tune(args, model, fine_tune_dataset, save_path)
        # evaluate model 
        eval(args, model, fine_tune_dataset)


def process_positive_mask(args, model, fine_tune_dataset, mask_regions, save_path):
    # set multiple masks
    fine_tune_dataset.set_mask(
        [(region_id, region) for region_id, region in enumerate(mask_regions)],
        mode=args.mode, 
        save_path=save_path,    
    )
    # fine tune (calibration)
    fine_tune(args, model, fine_tune_dataset, save_path)
    # evaluate model 
    eval(args, model, fine_tune_dataset)

def main(args):
    # load test id
    test_list  = [f"p{args.testid:02}"]

    # load model type
    _type, model = get_model(args.model)

    # load dataset type 
    dataset = EyeDataset if _type == "eye" else FaceDataset

    # load data 
    fine_tune_dataset = dataset(args.data, test_list,  0, up_bound=args.upperbound) 

    # load mask regions 
    mask_regions = [] if args.mask is None else np.load(args.mask) 

    # define save_path
    save_path=f"{dst_name}-testid-{args.testid}"
    os.makedirs(save_path, exist_ok=True)

    # exclude mask region
    if args.mode == "negative":
        process_negative_mask(args, model, fine_tune_dataset, mask_regions, save_path)

    # include mask region 
    if args.mode == "positive":
        process_positive_mask(args, model, fine_tune_dataset, mask_regions, save_path)


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

    parser.add_argument('-mask',
                        '--mask',
                        type=str,
                        required=False,
                        help="path to .npy region mask file")

    parser.add_argument('-mode',
                        '--mode',
                        default='negative',
                        type=str,
                        required=False,
                        help="negative/positive: exclude masked regions or include masked regions")

    args = parser.parse_args()
    main(args)