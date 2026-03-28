# STAT 214 Lab 2 — Cloud Detection in Polar Regions
**Spring 2025**

## Overview

This repository contains all code, figures, and report materials for Lab 2 of STAT 214.
The goal is to build a cloud detection classifier for MISR polar satellite imagery using
radiance data from 164 images, 3 of which carry expert labels.

The pipeline proceeds in three parts:
1. **EDA** — data cleaning, visualization, and train/val/test splitting
2. **Feature Engineering** — feature importance analysis, patch-based features, and CNN autoencoder transfer learning
3. **Modeling** — classifier training, evaluation, post-hoc EDA, and stability analysis

---

## Repository Structure

```
stat-214-lab2/
│
├── code/                         # All scripts for the full pipeline
│   ├── eda.py                    # Part 1: data cleaning, EDA, train/val/test split
│   ├── feature_engineering.py    # Part 2: feature importance + patch feature engineering
│   │
│   ├── autoencoder.py            # CNN autoencoder architecture
│   ├── data.py                   # Data loading utilities (lazy patch dataset)
│   ├── lazy_patch_dataset.py     # Memory-efficient patch loader for pretraining
│   ├── run_autoencoder.py        # Run autoencoder pretraining
│   ├── fine_tune_autoencoder.py  # Fine-tune pretrained autoencoder on labeled images
│   ├── get_embedding.py          # Extract latent embeddings for all images
│   ├── analyze_autoencoder.py    # Latent space analysis (PCA, t-SNE, dim selection)
│   │
│   ├── part3_prepare_data.py     # Merge features + embeddings into model-ready CSVs
│   ├── part3_modeling.py         # Train and evaluate all candidate classifiers
│   ├── part3_model_tuning.py     # Hyperparameter tuning for candidate models
│   ├── part3_final_model.py      # Refit final model on train+val, evaluate on test
│   ├── part3_posthoc.py          # Post-hoc EDA on final model errors
│   ├── part3_stability.py        # Stability analysis (noise perturbation, bootstrap)
│   ├── plot_roc_val.py           # ROC curve plots for all candidate classifiers
│   │
│   ├── assumptions_qda.py        # QDA assumption diagnostics
│   ├── assumptions_logistic.py   # Logistic regression assumption diagnostics
│   ├── assumptions_rf.py         # Random forest diagnostics
│   ├── assumptions_gb.py         # Gradient boosting diagnostics
│   │
│   ├── job_ae.sh                 # SLURM job script for the entire autoencoder/transfer learning pipeline
│   ├── job_ae_debug.sh           # SLURM job script for pretraining
│   ├── job_finetune.sh           # SLURM job script for fine-tuning
│   ├── job_embedding.sh          # SLURM job script for embedding extraction
│   ├── job_part3_modeling.sh     # SLURM job script for Part 3 modeling
│   ├── job_analyze.sh            # SLURM job script for autoencoder analysis/plots
│   │
│   ├── environment.yaml          # Conda environment specification
│   └── configs/                  # Configuration YAML files for autoencoder training
│
├── data/                         # Data directory (NOT committed — see below)
│   ├── image_data/               # Raw .npz image files (164 images)
│   ├── train.csv                 # Labeled pixels from O013257 (intermediate)
│   ├── val.csv                   # Labeled pixels from O012791 (intermediate)
│   ├── test.csv                  # Labeled pixels from O013490 (intermediate)
│   ├── train_features.csv        # Train + patch features
│   ├── val_features.csv          # Val + patch features
│   ├── test_features.csv         # Test + patch features
│   ├── train_model.csv           # Train + patch features + AE embeddings
│   ├── val_model.csv             # Val + patch features + AE embeddings
│   ├── test_model.csv            # Test + patch features + AE embeddings
│   └── ae_embeddings/            # Per-image autoencoder embedding CSVs
│
├── figs/                         # All figures produced by the pipeline
├── results/                      # Model outputs, metrics, and prediction files
│   ├── ae_pretrained_best.ckpt
│   ├── ae_pretrained.pt
│   ├── ae_finetuned_best.ckpt
│   └── ae_finetuned.pt
├── report/                       # LaTeX source and compiled PDF
├── documents/                    # Lab instructions and reference papers
└── instructions/                 # Teammate setup and pipeline instructions
```

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:sharyali05/stat-214-lab2.git
cd stat-214-lab2
```

**2. Create the conda environment**
```bash
conda env create -f code/environment.yaml
conda activate env_214
```

**3. Download the data**

Download `image_data.zip` from bCourses under Files > Labs > lab2 and unzip into `data/`:
```bash
mkdir -p data
unzip image_data.zip -d data/
```

---

## Running the Pipeline

All scripts should be run from the `code/` directory.

### Part 1: EDA
```bash
cd code
python eda.py
```
Outputs: `data/train.csv`, `data/val.csv`, `data/test.csv`, figures in `figs/`

### Part 2a: Feature Engineering
```bash
python feature_engineering.py
```
Outputs: `data/train_features.csv`, `data/val_features.csv`, `data/test_features.csv`

### Part 2b: Autoencoder (run on Bridges-2)

#### Option 1: run each stage separately
```bash
# Pretrain on all 164 images
sbatch job_ae_debug.sh

# After pretraining finishes, fine-tune on labeled images
sbatch job_finetune.sh

# After fine-tuning finishes, extract embeddings
sbatch job_embedding.sh

# After embeddings are created, analyze latent space
python analyze_autoencoder.py
```

#### Option 2: run the full pipeline in one job
```bash
# This script runs pretraining, fine-tuning, and embedding extraction in sequence
sbatch job_ae.sh

# After the job finishes, analyze latent space
python analyze_autoencoder.py
```
Outputs: 
- embeddings in `data/ae_embeddings/`
- figures in `figs/`
- best checkpoints in `results/` as .pt and .ckpt files 

### Part 3: Modeling
```bash
# Merge features and embeddings
python part3_prepare_data.py

# Train and tune candidate classifiers
# On Bridges-2, submit:
sbatch job_part3_modeling.sh
# Or run locally:
python part3_model_tuning.py

# Fit models with selected parameters and evaluate performance
python part3_modeling.py

# Refit and evaluate the final model
python part3_final_model.py

# Run post-hoc error analysis and stability checks
python part3_posthoc.py
python part3_stability.py
```
Outputs: `results/`, figures in `figs/`

---

## Data

The `data/` directory is excluded from version control via `.gitignore` because the
raw image files are too large to store in Git. All data files are reproducible by
running the pipeline scripts in order after downloading the raw images from bCourses.

---

## Dependencies

Key packages (see `code/environment.yaml` for full list):
- Python 3.10+
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- scipy
- PyTorch (for autoencoder)
- CUDA (required for autoencoder training on Bridges-2)

---

## Troubleshooting

**Import errors when running scripts:**
Make sure you have activated the environment with `conda activate env_214`
and that all packages are installed. Key packages needed:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

**File not found errors:**
Make sure you are running scripts from the `code/` directory, not from
the repo root or data directory.

**Scripts must be run in order:**
The pipeline has strict dependencies. Running scripts out of order will
cause file not found errors. The required order is:
1. `eda.py`
2. `feature_engineering.py`
3. Autoencoder scripts (`run_autoencoder.py` → `fine_tune_autoencoder.py` → `get_embedding.py`)
4. `part3_prepare_data.py`
5. `part3_modeling.py` / `part3_model_tuning.py`
6. `part3_final_model.py`
7. `part3_posthoc.py`, `part3_stability.py`

**Autoencoder scripts require Bridges-2:**
The autoencoder pretraining and fine-tuning steps require a GPU and must
be run on Bridges-2 via the provided SLURM job scripts. Do not attempt
to run `run_autoencoder.py` or `fine_tune_autoencoder.py` locally.
See `instructions/` for Bridges-2 setup guidance.

**part3_prepare_data.py fails with missing embedding files:**
Make sure the autoencoder embedding CSVs exist at `data/ae_embeddings/`
before running `part3_prepare_data.py`. These are produced by `get_embedding.py`
on Bridges-2 and must be copied to your local `data/` directory first.

**CUDA out of memory on Bridges-2:**
If the autoencoder job fails with a CUDA memory error, reduce the batch
size in the config file under `configs/` and resubmit the job.

**Conda environment already exists:**
If `conda env create` fails because `env_214` already exists, update it instead:
```bash
conda env update -f code/environment.yaml
```
