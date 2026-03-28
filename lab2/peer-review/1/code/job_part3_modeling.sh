#!/bin/bash
#SBATCH --job-name=part3_models
#SBATCH --output=../results/part3_models_%j.out
#SBATCH --error=../results/part3_models_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G

module load anaconda3
conda activate stat214

cd "$SLURM_SUBMIT_DIR"
srun python part3_model_tuning.py