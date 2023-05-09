import os
import cv2
import torch 
import matplotlib.pyplot as plt

from utils.general import pitchyaw2xyz

def draw_bboxes(img, bboxes, color=[0,255,0]):
    for box in bboxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return img 

def draw_gaze(img, pitchyaw: torch.Tensor, scale: int = 200):
    gaze_vecs = pitchyaw2xyz(pitchyaw)
    center = (img.shape[1] // 2, img.shape[0] // 2)
    for xyz in gaze_vecs:
        dx, dy, dz = xyz[0].item(), xyz[1].item(), xyz[2].item()
        img = cv2.arrowedLine(img, center, (int(center[0] + dx * scale), int(center[1] + dy * scale)), [0,255,0], 1) 
    return img 

def save_epoc_history(epoch_history, dst_dir):
    # Extract the data from epoch_history
    train_l1_losses = [epoch["train_l1_loss"]   for epoch in epoch_history]
    train_mal_losses = [epoch["train_mal_loss"] for epoch in epoch_history]
    val_l1_losses = [epoch["val_l1_loss"]       for epoch in epoch_history]
    val_mal_losses = [epoch["val_mal_loss"]     for epoch in epoch_history]

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot the L1 losses in the first subplot
    ax1.plot(train_l1_losses, label="Train L1 Loss")
    ax1.plot(val_l1_losses, label="Val L1 Loss")
    ax1.legend()

    # Plot the Mal losses in the second subplot
    ax2.plot(train_mal_losses, label="Train Mal Loss")
    ax2.plot(val_mal_losses, label="Val Mal Loss")
    ax2.legend()
    # save both plots in one figure as a PNG file
    plt.savefig(os.path.join(dst_dir, 'epoch_losses.png'))

def save_step_history(train_step_history, val_step_history, dst_dir):
    # Plot the training and validation losses
    train_l1_loss = [h["l1_loss"] for h in train_step_history]
    train_mal_loss = [h["mal_loss"] for h in train_step_history]
    val_l1_loss = [h["l1_loss"] for h in val_step_history]
    val_mal_loss = [h["mal_loss"] for h in val_step_history]

    # Training L1 Loss
    plt.figure()
    plt.plot(train_l1_loss)
    plt.title("Training L1 Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "training_l1_loss.png"))

    # Training Mal Loss
    plt.figure()
    plt.plot(train_mal_loss)
    plt.title("Training Mean Angular Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "training_mal_loss.png"))

    # Validation L1 Loss
    plt.figure()
    plt.plot(val_l1_loss)
    plt.title("Validation L1 Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "validation_l1_loss.png"))

    # Validation Mal Loss
    plt.figure()
    plt.plot(val_mal_loss)
    plt.title("Validation Mean Angular Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "validation_mal_loss.png"))