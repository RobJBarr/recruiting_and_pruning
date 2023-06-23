#!/bin/bash
#SBATCH -o ../slurm_outputs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<rjb19>

export PATH=/vol/bitbucket/${USER}/venv/bin/:$PATH
source activate
. /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime
cd ..
python experiments/train_sparse_gp.py bike > python_outputs/python_out_bike.txt
