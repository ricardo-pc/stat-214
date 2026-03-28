#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 04:00:00
#SBATCH -J ae_finetune
#SBATCH -o ../results/ae_finetune_%j.out
#SBATCH -e ../results/ae_finetune_%j.err

module load anaconda3
conda activate stat214

cd "$SLURM_SUBMIT_DIR"
srun python fine_tune_autoencoder.py configs/fine_tune.yaml ../results/ae_pretrained_best.ckpt