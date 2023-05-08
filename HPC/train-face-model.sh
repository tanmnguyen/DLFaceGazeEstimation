#!/bin/sh
#
#SBATCH --job-name="train_face_gaze_model_job"
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load miniconda3
conda activate gaze 

srun ./train.py --data ../../../../scratch/mnguyen1/MPIFaceGazeData --type face --epochs 20