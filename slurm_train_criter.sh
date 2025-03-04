#!/bin/sh
#SBATCH --job-name=train_criter             # the title of the job
#SBATCH --output=slurm_out.log            # file to which logs are saved
#SBATCH --time=2-00:00:00                 # job time limit - full format is D-H:M:S
#SBATCH --nodes=1                         # number of nodes
#SBATCH --gres=gpu:1                      # number of gpus
#SBATCH --ntasks=1                        # number of tasks
#SBATCH --mem-per-gpu=82G                 # memory allocation
#SBATCH --partition=gpu                   # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=12                # number of allocated cores

# activate conda environment
source /d/hpc/projects/FRI/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /d/hpc/projects/FRI/$USER/miniconda3/envs/thesis_env
export WANDB_API_KEY=c4520e6867f0975e7b1ac5de49de1bc539c69309

# Mediterranean
DATA_PATH="/d/hpc/projects/FRI/mm1706/data/SST_L3_CMEMS_2006-2021_Mediterranean.nc"
CLOUD_COVERAGE_THRESHOLD=1.0
S_PATCH_SIZE=8

# Adriatic
# DATA_PATH="/d/hpc/projects/FRI/mm1706/data/CHEMS_L3_SST_Adriatic.nc"
# CLOUD_COVERAGE_THRESHOLD=0.60
# S_PATCH_SIZE=6

# Atlantic
# DATA_PATH="/d/hpc/projects/FRI/mm1706/data/CHEMS_SST_Atlantic.nc"
# CLOUD_COVERAGE_THRESHOLD=0.75
# S_PATCH_SIZE=8

# train Refinement Module
NUM_REFINEMENT_STEPS=3
OUT_PATH="./output/train_criter"
MODEL_PATH="./output/train_crm/CRM.pt"
srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 python3 train_criter.py \
    --data_path=$DATA_PATH \
    --model_path=$MODEL_PATH \
    --cloud_coverage_threshold=$CLOUD_COVERAGE_THRESHOLD \
    --s_patch_size=$S_PATCH_SIZE \
    --num_refinement_steps=$NUM_REFINEMENT_STEPS \
    --lr=3e-4 \
    --batch_size=8 \
    --step_size=50 \
    --n_epochs=300 \
    --plot_period=150 \
    --out_path=$OUT_PATH

cp slurm_out.log $OUT_PATH/train_criter.log
