import os, cv2
import numpy as np

import os
import numpy as np
import cv2

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
            for file in os.listdir(folder_path)[:n_files]:
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    image_file = os.path.join(folder_path, file)
                    images.append(cv2.imread(image_file))

            data_dict[folder] = { 'images': np.array(images),
                                    'labels': np.array(labels),
                                    'left_landmarks': left_landmarks,
                                    'right_landmarks': right_landmarks
                                }

    return data_dict
