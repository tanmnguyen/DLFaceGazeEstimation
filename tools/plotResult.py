import sys 
sys.path.append("../")

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.file import extract_test_id

def read_result_file(folder_path: str, result_file: str):
    file_path = os.path.join(folder_path, result_file)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("Validation Angular Error:"):
                    try:
                        result = float(line.split(":")[1].strip())
                        return result
                    except ValueError:
                        pass
    return None

def plot_histogram(test_ids, results, title):
    plt.figure(figsize=(9, 7))

    num_bins = len(set(test_ids))
    _, bins, _ = plt.hist(test_ids, bins=num_bins, weights=results, alpha=1, width=0.95, color='cyan')
    plt.xlabel('Test Subject ID')
    plt.ylabel('Error in Degree')
    plt.title(title)

    # Adjust the x-axis tick positions and labels
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, sorted(set(test_ids)))

    # Assign a different color to each bin
    color_map = plt.get_cmap('Spectral')
    bin_colors = [color_map(i / num_bins) for i in range(num_bins)]
    for patch, color in zip(plt.gca().patches, bin_colors):
        patch.set_facecolor(color)

    # Calculate and display the average line
    average_result = np.mean(results)
    std_result = np.std(results)
    plt.axhline(y=average_result, color='blue', linestyle='--', label='Average')

    print("Average Result:", average_result)
    print("Standard Deviation:", std_result)

    # Show results inside each histogram bin
    for _id, result, x in zip(test_ids, results, bin_centers):
        plt.text(x, result, f'{result:.3f}          ', ha='center', va='top', rotation='vertical', fontsize=12)

    # Set y-axis limits to include the average value and standard deviation
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, max(ymax, average_result + 0.1 * ymax + std_result))

    # Show average and standard deviation text outside histogram space
    plt.text(
        max(bin_centers) + 1.2, 
        average_result, f'{average_result:.3f} ± {std_result:.3f} ', ha='left', va='center', 
        rotation='vertical',
        fontsize=12,
    )

    plt.legend()
    plt.show()
    plt.close()

def process_folders(path: str, result_file: str):
    test_ids = []
    results = []

    folder_names = os.listdir(path)
    folder_names.sort()

    for folder_name in folder_names:
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            test_id = extract_test_id(folder_name)
            if test_id is not None:
                result = read_result_file(folder_path, result_file)
                if result is not None:
                    test_ids.append(test_id)
                    results.append(result)

    if len(results) > 0:
        plot_histogram(test_ids, results, title=f"Angular Error Histogram {os.path.basename(result_file).split('.')[0]}")

def main(args):
    result_files = ["result.txt"]
    result_files.extend([f"result-region-{id}.txt" for id in range(7)])

    for result_file in result_files:
        process_folders(args.path, result_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-path',
                        '--path',
                        required=True,
                        help="path to the test result folder")

    args = parser.parse_args()
    main(args)