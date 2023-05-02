import torch 

def available_device():
    # use cuda gpu 
    if torch.cuda.is_available():
        return torch.device("cuda")
    # use apple mps (unsupported for segmentation task by ultralytics yolov8)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # use cpu 
    return torch.device("cpu")