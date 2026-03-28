#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 08:00:00
#SBATCH -J ae_debug
#SBATCH -o ../results/ae_debug_%j.out
#SBATCH -e ../results/ae_debug_%j.err

module load anaconda3
conda activate stat214

cd "$SLURM_SUBMIT_DIR"
srun python run_autoencoder.py configs/debug.yaml
