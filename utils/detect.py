import cv2

def extract_bbox_img(img, bbox):
    return img[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

def get_eye_bbox_with_corners(corners):
    # Convert the image to grayscale
    return [corners[0] - 35, corners[1] - 35, corners[2] + 35, corners[3] + 35]
