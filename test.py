import argparse 
import datetime 
import numpy as np

from utils.file import load_model
from utils.general import split_data
from torch.utils.data import DataLoader

from datasets.EyeDataset  import EyeDataset
from datasets.FaceDataset import FaceDataset

dst_name = f"test-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"

def main(args):
    test_list  = [f"p{args.testid:02}"]

    # load pre-trained model 
    _type, model = load_model(args.model)

    # load dataset type 
    dataset = EyeDataset if _type == "eye" else FaceDataset

    # fine tune data
    fne_tune = dataset(args.data, test_list,  0, up_bound=args.upperbound) 

    # tune and test set
    tune_dataset, test_dataset  = split_data(fne_tune, 0.1, 0.9)

    # tuning configuration
    model.tune_config()

    # fine tune model
    model.fit(
        DataLoader(tune_dataset, batch_size=32 , shuffle=True),
        DataLoader(test_dataset, batch_size=32 , shuffle=False),
        epochs=30,
        lr=0.0001,
        dst_dir=f"tests/{dst_name}-testid-{args.testid}"
    )

    # average angular error for the last 10 epochs to obtain final test error value
    valid_angular_err = np.mean(list(map(lambda his: his["valid_ma_loss"], model.epoch_history[-10:])))

    with open(f"tests/{dst_name}-testid-{args.testid}/result.txt", 'w') as file:
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

    args = parser.parse_args()
    main(args)