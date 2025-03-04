#!/bin/bash
#SBATCH --job-name=st_mae                 # the title of the job
#SBATCH --output=slurm_out.log            # file to which logs are saved
#SBATCH --time=1:00:00                    # job time limit - full format is D-H:M:S
#SBATCH --nodes=1                         # number of nodes
#SBATCH --gres=gpu:1                      # number of gpus
#SBATCH --ntasks=1                        # number of tasks
#SBATCH --mem-per-gpu=81G                 # memory allocation
#SBATCH --partition=gpu                   # partition to run on nodes that contain gpus

source /d/hpc/projects/FRI/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /d/hpc/projects/FRI/$USER/miniconda3/envs/thesis_env

export DATA_PATH="/d/hpc/projects/FRI/mm1706/data/"
srun --nodes=1 --exclusive --ntasks=1 python -m pytest --html=report.html