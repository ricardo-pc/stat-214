#!/bin/bash
#SBATCH --account=mth250011p
#SBATCH --job-name=ae_analyze
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH -o ../results/analyze.out
#SBATCH -e ../results/analyze.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --mem=22G

set -e

module load anaconda3
conda activate stat214

cd "$SLURM_SUBMIT_DIR"

python analyze_autoencoder.py configs/fine_tune.yaml \
  --checkpoint ../results/ae_finetuned_best.ckpt
