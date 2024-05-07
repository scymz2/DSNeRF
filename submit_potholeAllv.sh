#!/bin/bash --login
#$ -cwd
#$ -l v100           # A 1-GPU request (v100 is just a shorter name for nvidia_v100)
                     # Can instead use 'a100' for the A100 GPUs (if permitted!)
#$ -N pAllv_new       # job name
#$ -pe smp.pe 4      # 4 CPU cores available to the host code
                     # Can use up to 12 CPUs with an A100 GPU.
#$ -o ~/scratch/DSNeRF/logs/output_Allv_new.txt
#$ -e ~/scratch/DSNeRF/logs/error_Allv_new.txt

module load libs/nvidia-hpc-sdk/23.9
module load apps/binapps/anaconda3/2022.10
source activate colmap

python imgs2poses.py data/potholeAllView/
python run_nerf.py --config configs/pothole_Allv_new.txt
