import sys
sys.path.append('../')

import os, cv2
import numpy as np

from dataset import EyeDataset

from utils.general import split_train_val_indices
from torch.utils.data import Dataset, DataLoader

def read_images_and_labels(path: str, upper_bound: int):
    """
    Goes one level under the given path and checks if each subfolder contains `labels.npy`,
    `left_landmarks.npy`, and `right_landmarks.npy`. If all three files are present, the function reads all images
    in the subfolder and returns a dictionary where the keys are the folder names and the values are another
    dictionary containing the image array, left eye landmarks, and right eye landmarks.

    Parameters:
    path (str): The path to the directory containing the subfolders to be searched.
    upper_bound (int | None): The upper bound image number to be read per directory.
    Returns:
    Dict: A dictionary where the keys are the folder names and the values are another dictionary with keys:
        "images" (numpy.ndarray): An array containing all the images found in the subfolder.
        "labels" (numpy.ndarray): An array containing the normalized direction label.
        "left_landmarks" (numpy.ndarray): An array containing the left eye landmarks for the images.
        "right_landmarks" (numpy.ndarray): An array containing the right eye landmarks for the images.
    """
    data_dict = {}

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        images, left_landmarks, right_landmarks = [], [], []
        if ('labels.npy' in os.listdir(folder_path)) and ('left_landmarks.npy' in os.listdir(folder_path)) \
                and ('right_landmarks.npy' in os.listdir(folder_path)):

            labels_file = os.path.join(folder_path, 'labels.npy')
            left_landmarks_file = os.path.join(folder_path, 'left_landmarks.npy')
            right_landmarks_file = os.path.join(folder_path, 'right_landmarks.npy')

            labels = np.load(labels_file)
            left_landmarks = np.load(left_landmarks_file)
            right_landmarks = np.load(right_landmarks_file)

            n_files = min(len(os.listdir(folder_path)), upper_bound)
            for file_idx in range(n_files):
                image_file = os.path.join(folder_path, str(file_idx) + ".png")
                if os.path.basename(image_file) in os.listdir(folder_path):
                    images.append(cv2.imread(image_file))
                    if images[-1] is None:
                        print(f"Image Not Found or Deprecated: {image_file}")
                        images.pop()

            data_dict[folder] = { 'images': np.array(images),
                                    'labels': np.array(labels),
                                    'left_landmarks': left_landmarks,
                                    'right_landmarks': right_landmarks
                                }

    return data_dict

def build_loader(path: str, lower_bound: int, upper_bound: int, batch_size: int):
    data_dict = read_images_and_labels(path, upper_bound=upper_bound)

    # training data
    train_images, train_labels, train_llmarks, train_rlmarks = [], [], [], []
    for i in range(0, 14):
        train_images.extend(data_dict[str(i)]['images'])
        train_labels.extend(data_dict[str(i)]['labels'])
        train_llmarks.extend(data_dict[str(i)]['left_landmarks'])
        train_rlmarks.extend(data_dict[str(i)]['right_landmarks'])

    # validation data 
    val_images, val_labels, val_llmarks, val_rlmarks = [], [], [], []
    for i in range(14, 15):
        # simulate gaze calibration process
        train_indices, val_indices = split_train_val_indices(
            array_len=len(data_dict[str(i)]['images']), 
            train_percentage=0.1
        )
        print(f"Calibration Simulation - Train: {len(train_indices)} - Val: {len(val_indices)}")

        train_images.extend(data_dict[str(i)]['images'][train_indices])
        train_labels.extend(data_dict[str(i)]['labels'][train_indices])
        train_llmarks.extend(data_dict[str(i)]['left_landmarks'][train_indices])
        train_rlmarks.extend(data_dict[str(i)]['right_landmarks'][train_indices])
        # use the rest for validation
        val_images.extend(data_dict[str(i)]['images'][val_indices])
        val_labels.extend(data_dict[str(i)]['labels'][val_indices])
        val_llmarks.extend(data_dict[str(i)]['left_landmarks'][val_indices])
        val_rlmarks.extend(data_dict[str(i)]['right_landmarks'][val_indices])

    train_loader = DataLoader(
        EyeDataset(train_images, train_labels, train_llmarks, train_rlmarks), 
        batch_size=batch_size, 
        shuffle=True
    )

    val_loader   = DataLoader(
        EyeDataset(val_images, val_labels, val_llmarks, val_rlmarks), 
        batch_size=batch_size, 
        shuffle=True
    )

    return train_loader, val_loader