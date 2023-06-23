#!/bin/bash
#SBATCH -o ../slurm_outputs/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<rjb19>

export PATH=/vol/bitbucket/${USER}/venv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime
cd ..
python experiments/test_kmeans_vs_condvar.py > python_outputs/python_out.txt
