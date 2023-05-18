import os
import cv2
import torch 
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from utils.general import pitchyaw2xyz

def show_overlay_heat_map(img: np.ndarray, heat_map: np.ndarray, caption: str, save_path: str = None):
    os.makedirs(save_path, exist_ok=True)

    # Apply a color map to the normalized heat map
    heat_map_color = cv2.applyColorMap(np.uint8(heat_map * 255), cv2.COLORMAP_JET)

    # Convert color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heat_map_color = cv2.cvtColor(heat_map_color, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the original image
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Plot the heat map
    axs[1].imshow(img)
    axs[1].imshow(heat_map_color, alpha=0.6)
    axs[1].axis('off')
    axs[1].set_title('Heat Map')

    # Create a ScalarMappable with the jet colormap
    scalar_mappable = cm.ScalarMappable(cmap='jet')
    scalar_mappable.set_array(heat_map)

    # Add a colorbar legend for the heat map (outside the images)
    cbar = plt.colorbar(scalar_mappable, ax=axs[1], fraction=0.05, pad=0.04, orientation='vertical')
    cbar.ax.set_ylabel('Intensity')

    # Add caption
    fig.suptitle(caption, fontsize=12)

    # Adjust the layout and display the figure
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, f"{caption}.svg"))

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
    train_l1_losses = [epoch["train_l1_loss"] for epoch in epoch_history]
    train_ma_losses = [epoch["train_ma_loss"] for epoch in epoch_history]
    valid_l1_losses = [epoch["valid_l1_loss"] for epoch in epoch_history]
    valid_ma_losses = [epoch["valid_ma_loss"] for epoch in epoch_history]

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot the L1 losses in the first subplot
    ax1.plot(train_l1_losses, label="Train L1 Loss")
    ax1.plot(valid_l1_losses, label="Valid L1 Loss")
    ax1.legend()

    # Plot the Mal losses in the second subplot
    ax2.plot(train_ma_losses, label="Train Ma Loss")
    ax2.plot(valid_ma_losses, label="Valid Ma Loss")
    ax2.legend()
    # save both plots in one figure as a PNG file
    plt.savefig(os.path.join(dst_dir, 'epoch_losses.png'))

def save_step_history(train_step_history, valid_step_history, dst_dir):
    # Plot the training and validation losses
    train_l1_loss = [h["l1_loss"] for h in train_step_history]
    train_ma_loss = [h["ma_loss"] for h in train_step_history]
    valid_l1_loss = [h["l1_loss"] for h in valid_step_history]
    valid_ma_loss = [h["ma_loss"] for h in valid_step_history]

    # Training L1 Loss
    plt.figure()
    plt.plot(train_l1_loss)
    plt.title("Training L1 Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "training_l1_loss.png"))

    # Training Mal Loss
    plt.figure()
    plt.plot(train_ma_loss)
    plt.title("Training Mean Angular Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "training_ma_loss.png"))

    # Validation L1 Loss
    plt.figure()
    plt.plot(valid_l1_loss)
    plt.title("Validation L1 Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "validation_l1_loss.png"))

    # Validation Mal Loss
    plt.figure()
    plt.plot(valid_ma_loss)
    plt.title("Validation Mean Angular Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(dst_dir, "validation_ma_loss.png"))