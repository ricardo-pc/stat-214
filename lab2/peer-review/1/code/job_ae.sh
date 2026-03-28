#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 16:00:00
#SBATCH -J ae_pipeline
#SBATCH -o ../results/ae_pipeline_%j.out
#SBATCH -e ../results/ae_pipeline_%j.err

set -e

module load anaconda3
conda activate stat214

cd "$SLURM_SUBMIT_DIR"

echo "Step 1: pretrain autoencoder"
srun python run_autoencoder.py configs/debug.yaml

echo "Step 2: fine-tune autoencoder"
srun python fine_tune_autoencoder.py configs/fine_tune.yaml ../results/ae_pretrained_best.ckpt
# or srun python fine_tune_autoencoder.py configs/fine_tune.yaml ../results/ae_pretrained.pt

echo "Step 3: extract embeddings"
srun python get_embedding.py configs/fine_tune.yaml ../results/ae_finetuned_best.ckpt
# or srun python get_embedding.py configs/fine_tune.yaml ../results/ae_finetuned.pt

echo "Done"