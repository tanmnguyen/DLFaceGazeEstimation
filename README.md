# Deep Learning Face Gaze Estimation
This implementation, as part of the research thesis "Deep Learning-Based Gaze Estimation", aims to investigate the significance of facial features in the eye-gaze direction estimation task. The research uses the normalized MPIIFaceGaze Dataset, found here https://perceptualui.org/research/datasets/MPIIFaceGaze/, and establishes deep CNN(s) to study which and how to incorporate facial features beyond eye regions to improve the performance of gaze direction regression task. 

# Installation
Clone the repository
```bash
git@github.com:tanmnguyen/DLFaceGazeEstimation.git
```
Change to directory path and install the required packages (virtual environment recommended).
```bash
cd DLFaceGazeEstimation/
pip install -r requirements.txt
```
# Training
```bash
python train.py --data path/to/training/data --type [eye/face] --epochs 20
```

**Note:** It should be noted that the present work constitutes ongoing research, and as such, modifications are expected to occur on a regular basis.