import cv2
import math
import torch
import random
import numpy as np

from torch.utils.data import DataLoader, random_split

def split_data(dataset, train_ratio: float, valid_ratio: float):
    n = len(dataset)
    train_sz = int(n * train_ratio)

    return random_split(dataset, [train_sz, n - train_sz])

def pitchyaw2xyz(pitchyaw: torch.Tensor) -> torch.Tensor:
    pitches, yaws = pitchyaw[:, 0], pitchyaw[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    result = torch.stack((x, y, z), dim=1)
    return result

def angular_loss(vectors1: torch.Tensor, vectors2: torch.Tensor) -> torch.Tensor:
    cos_sims = torch.sum(vectors1 * vectors2, dim=1) / (torch.norm(vectors1, dim=1) * torch.norm(vectors2, dim=1))
    cos_sims = torch.clamp(cos_sims, -1.0, 1.0)
    angle_diffs = torch.acos(cos_sims).mean() 
    angle_diffs = angle_diffs * (180 / torch.tensor(np.pi))
    return angle_diffs

def letterbox_resize(image, target_size):
    """
    Resizes the input image to the target size with letterboxing.

    Args:
    image: NumPy array of shape (height, width, channels)
    target_size: Tuple of (height, width) specifying the desired output size

    Returns:
    NumPy array of shape (target_height, target_width, channels) with letterboxing
    """

    # Compute the new size while maintaining aspect ratio
    height, width = image.shape[:2]
    target_height, target_width = target_size
    scale = min(target_height/height, target_width/width)
    new_height, new_width = int(height * scale), int(width * scale)

    # Resize the image using OpenCV
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a blank image with the target size and fill it with gray color
    blank_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    blank_image[:, :] = (0, 0, 0)

    # Compute the x, y offsets to center the resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image in the center of the blank image
    blank_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return np.array(blank_image)

def split_train_val_indices(array_len, train_percentage):
    """
    Randomly splits an array of images into training and validation sets.
    Returns the training and validation sets as two separate numpy arrays.

    Args:
        images (numpy array): An array of images to be split.

    Returns:
        A tuple containing two numpy arrays: the training set and the validation set.
    """
    # Calculate the split point
    split_point = int(train_percentage * array_len)
    # Get the training and validation indices
    train_indices = random.sample(range(array_len), split_point)
    val_indices   = [i for i in range(array_len) if i not in train_indices]
    # result
    return train_indices, val_indices