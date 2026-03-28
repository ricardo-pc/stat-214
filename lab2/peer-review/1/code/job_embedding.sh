#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 08:00:00
#SBATCH -J ae_embed
#SBATCH -o ../results/ae_embed_%j.out
#SBATCH -e ../results/ae_embed_%j.err

module load anaconda3
conda activate stat214

cd "$SLURM_SUBMIT_DIR"
srun python get_embedding.py configs/fine_tune.yaml ../results/ae_finetuned_best.ckpt


