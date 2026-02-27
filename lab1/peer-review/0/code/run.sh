#!/bin/bash

# Activate environment

conda activate stat214    

# make sure we're in the code directory
cd "$(dirname "$0")"

# Run data cleaning
echo "Running data cleaning..."
python clean.py


# Run modeling pipeline
echo "Running modeling pipeline..."
python models.py

echo "Running stability analysis..."

# Run stability analysis
python models_stability.py


# Execute notebooks (EDA + comparison)
echo "Executing notebooks for EDA and prediction comparison..."
jupyter nbconvert --to notebook --execute --inplace 02_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_prediction_comparison.ipynb

# Deactivate environment

conda deactivate