# Deep Learning Face Gaze Estimation
Most existing implementations for eye-gaze direction estimation focus solely on the eye image, ignoring the potential benefits of incorporating information from other facial features. This program, as part of the research thesis "Deep Learning-Based Gaze Estimation", takes a novel approach by utilizing Deep Convolutional Neural Network on both eye and facial features to improve the accuracy of gaze direction estimation in 3D space.

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
PyTorch framework is used in this project with different network architecture backbones provided in the `configs/` directory. To train a model:
```bash
python train.py --path path/to/train/data --config path/to/backbone.yaml --type [eye/face] --epochs 10 
```
The program accepts `--upperbound` and `--lowerbound` arguments to set limits on the amount of training data to be read. 

**Note:** The project is ongoing and changes are expected to be made regularly.