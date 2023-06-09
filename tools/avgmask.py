# This file computes the average of the heat map generated by the interpretation. 
import sys 
sys.path.append('../')

import os
import datetime
import argparse 
import numpy as np

from utils.plot import show_heat_map
from utils.file import extract_test_id

dst_name = f"masks/masks-{datetime.datetime.now().strftime('%Y.%m.%d. %H.%M.%S')}"
os.makedirs(dst_name, exist_ok=True)

def main(args):
    folder_names = os.listdir(args.path)
    folder_names.sort()

    rims, ids = [], []
    for folder_name in folder_names:
        folder_path = os.path.join(args.path, folder_name)
        if os.path.isdir(folder_path):
            test_id = extract_test_id(folder_name)
            if test_id is not None:
                rims.append(np.load(os.path.join(folder_path, "region_importance_map.npy")))
                ids.append(test_id)

    # hard coded regions (research purpose only) based on the average importance map
    # represented in the paper.
    regions = [
        [0,0,56,50],
        [160,0,224,55],
        [13,50, 107,104],
        [116,50,213,102],
        [75,118,143,183],
        [0,105, 57,187],
        [160,112, 217,186]
    ]

    # compute average map 
    rim = np.mean(rims, axis=0)
    
    # show (save) heat map
    show_heat_map(rim, "Average Region Importance Map", dst_name)

    # save regions 
    np.save(os.path.join(dst_name, "regions.npy"), regions)

    # show different regions 
    for idx, region in enumerate(regions):
        _rim = np.copy(rim)
        _rim[region[1]:region[3], region[0]:region[2]] = 0
        # show (save)
        show_heat_map(_rim, f"Avg Region Importance Map Region {idx}", dst_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-path',
                        '--path',
                        required=True,
                        help="path to the intepretation folder")

    args = parser.parse_args()
    main(args)