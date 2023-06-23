#!/bin/bash
#SBATCH -o ../slurm_outputs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<rjb19>

export PATH=/vol/bitbucket/${USER}/venv/bin/:$PATH
source activate
. /vol/cuda/11.2.1-cudnn8.1.0.77/setup.shTERM=vt100
/usr/bin/nvidia-smi
uptime
cd ..
python experiments/dgp_vs_single_layer_gp.py $1 > python_outputs/dgp_vs_single_layer_$1.txt