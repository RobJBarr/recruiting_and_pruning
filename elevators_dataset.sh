#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<rjb19>
export PATH=/vol/bitbucket/${USER}/venv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime
python -u elevators_sparse_gp.py > python_out.txt