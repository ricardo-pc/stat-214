#!/usr/bin/env bash
# run.sh — Unified pipeline script (feature engineering + logreg/svm + ensemble + analysis)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_NAME="${ENV_NAME:-env_214}"
SKIP_FEATURE_PIPELINE="${SKIP_FEATURE_PIPELINE:-0}"
SKIP_ENSEMBLE_TRAIN="${SKIP_ENSEMBLE_TRAIN:-0}"
RUN_LOGREG_SVM_STABILITY="${RUN_LOGREG_SVM_STABILITY:-1}"
RUN_ENSEMBLE_ANALYSIS="${RUN_ENSEMBLE_ANALYSIS:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither python nor python3 is available in PATH." >&2
    exit 127
fi

CONFIG="$SCRIPT_DIR/configs/default.yaml"
CKPT="$PROJECT_ROOT/results/checkpoints/ae_opt-best.ckpt"
TRAIN_DATA="$PROJECT_ROOT/feature_eng_dataset/train_features_opt.csv"
TEST_DATA="$PROJECT_ROOT/feature_eng_dataset/test_features_opt.csv"
RF_MODEL="$PROJECT_ROOT/results/rf/best_rf_model.pkl"

echo "========================================"
echo "Unified pipeline start"
echo "Project root: $PROJECT_ROOT"
echo "Conda env:    $ENV_NAME"
echo "Skip stage 1 (feature): $SKIP_FEATURE_PIPELINE"
echo "Skip stage 3 (ensemble): $SKIP_ENSEMBLE_TRAIN"
echo "Run logreg/svm stability: $RUN_LOGREG_SVM_STABILITY"
echo "Run ensemble analysis: $RUN_ENSEMBLE_ANALYSIS"
echo "========================================"

if [[ "$SKIP_FEATURE_PIPELINE" == "1" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "1) Autoencoder + feature pipeline"
    echo "Skipped (SKIP_FEATURE_PIPELINE=1)"
    echo "----------------------------------------"
else
    echo ""
    echo "----------------------------------------"
    echo "1) Autoencoder + feature pipeline"
    echo "----------------------------------------"
    "$PYTHON_BIN" "$SCRIPT_DIR/run_autoencoder.py" "$CONFIG"
    "$PYTHON_BIN" "$SCRIPT_DIR/get_embedding.py" "$CONFIG" "$CKPT"
    "$PYTHON_BIN" "$SCRIPT_DIR/feature_engineering_autoencoder.py" "$CONFIG"

    PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

project_root = os.environ["PROJECT_ROOT"]
df_train = pd.read_csv(f"{project_root}/feature_eng_dataset/train_features_opt.csv")
df_test = pd.read_csv(f"{project_root}/feature_eng_dataset/test_features_opt.csv")

ae_cols = [c for c in df_train.columns if c.startswith("ae")]
if len(ae_cols) == 0:
    raise ValueError("No AE columns found in train_features_opt.csv")

print("\nTrain shape:", df_train.shape)
print("Test shape :", df_test.shape)
print("AE dim     :", len(ae_cols))
print("Train null :", int(df_train.isnull().sum().sum()))
print("Test null  :", int(df_test.isnull().sum().sum()))

print("\nTrain image counts:")
print(df_train["image"].value_counts(dropna=False))

print("\nTest image counts:")
print(df_test["image"].value_counts(dropna=False))
PY
fi

echo ""
echo "----------------------------------------"
echo "2) Logistic Regression + SVM"
echo "----------------------------------------"
for f in "$TRAIN_DATA" "$TEST_DATA"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required data not found at $f" >&2
        exit 1
    fi
done
"$PYTHON_BIN" "$SCRIPT_DIR/models/logreg_svm.py"

if [[ "$RUN_LOGREG_SVM_STABILITY" == "1" ]]; then
    echo ""
    echo "2b) Running label-flip stability ..."
    "$PYTHON_BIN" "$SCRIPT_DIR/models/logreg_svm_stability.py"
fi

if [[ "$SKIP_ENSEMBLE_TRAIN" == "1" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "3) Ensemble training"
    echo "Skipped (SKIP_ENSEMBLE_TRAIN=1)"
    echo "----------------------------------------"
else
    echo ""
    echo "----------------------------------------"
    echo "3) Ensemble training (ensemble.py)"
    echo "----------------------------------------"
    "$PYTHON_BIN" "$SCRIPT_DIR/models/ensemble.py"
fi

if [[ "$RUN_ENSEMBLE_ANALYSIS" == "1" ]]; then
    echo ""
    echo "----------------------------------------"
    echo "4) Ensemble analysis"
    echo "----------------------------------------"
    for f in "$TRAIN_DATA" "$TEST_DATA"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: Required data not found at $f" >&2
            exit 1
        fi
    done
    if [[ ! -f "$RF_MODEL" ]]; then
        echo "ERROR: RF model not found at $RF_MODEL" >&2
        echo "       Provide the fixed model at this path or set SKIP_ENSEMBLE_TRAIN=0." >&2
        exit 1
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/models/ensemble_analysis.py"
fi

echo ""
echo "========================================"
echo "Unified pipeline finished successfully."
echo "========================================"