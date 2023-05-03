import cv2
import torch 
from utils.general import pitchyaw2xyz

def draw_gaze(img, pitchyaw: torch.Tensor, scale: int = 200):
    gaze_vecs = pitchyaw2xyz(pitchyaw)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    for xyz in gaze_vecs:
        dx, dy, dz = xyz[0].item(), xyz[1].item(), xyz[2].item()
        img = cv2.arrowedLine(img, center, (int(center[0] + dx * scale), int(center[1] + dy * scale)), [0,255,0], 1) 
    return img 