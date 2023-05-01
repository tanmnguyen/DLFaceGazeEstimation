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
To train the baseline model (using eye images and full face image). The model implements itracker architecture [@cvpr2016_gazecapture]
```bash
python train.py --path path/to/train/data --type eye --epochs 10 
```
The program accepts `--upperbound` and `--lowerbound` arguments to set limits on the amount of training data to be read. 

**Note:** The project is ongoing and changes are expected to be made regularly.


[@inproceedings{cvpr2016_gazecapture,
  Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
  Title = {Eye Tracking for Everyone},
  Year = {2016},
  Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}]