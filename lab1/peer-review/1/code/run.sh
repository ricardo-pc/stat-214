#!/bin/bash

set -e
ENV_NAME="stat214"

echo "Activating conda environment: $ENV_NAME"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Running lab1.ipynb..."

#creates a fully executed notebook
jupyter nbconvert \
    --to notebook \
    --execute lab1.ipynb \
    --output lab1_executed.ipynb


echo "Running clean.py..."
python clean.py

#we can pass argument perturb to run modeling for perturbed data
#python models.py perturb
#Note: running models.py will save result plots in outputs subfolder
echo "Running models.py..."
python models.py


echo "Deactivating environment..."
conda deactivate

echo "Pipeline completed successfully!"