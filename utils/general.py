import cv2
import math
import numpy as np

def yawpitch2xyz(yawpitch_list):
    yawpitch_arr = np.array(yawpitch_list)
    cos_yaw = np.cos(yawpitch_arr[:, 0])
    sin_yaw = np.sin(yawpitch_arr[:, 0])
    cos_pitch = np.cos(yawpitch_arr[:, 1])
    sin_pitch = np.sin(yawpitch_arr[:, 1])

    x = cos_yaw * cos_pitch
    y = sin_yaw * cos_pitch
    z = sin_pitch

    return np.column_stack((x, y, z))

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
