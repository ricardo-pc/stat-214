#!/usr/bin/env bash
# run_ensemble_analysis.sh — Run ensemble model analysis (HGB + RF diagnostics)

set -euo pipefail

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_214

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither python nor python3 is available in PATH." >&2
    exit 127
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Ensemble Model Analysis ==="
echo "Project root: $PROJECT_ROOT"

# Verify required input data
TRAIN_DATA="$PROJECT_ROOT/feature_eng_dataset/train_features_opt.csv"
TEST_DATA="$PROJECT_ROOT/feature_eng_dataset/test_features_opt.csv"

for f in "$TRAIN_DATA" "$TEST_DATA"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required data not found at $f" >&2
        exit 1
    fi
done

echo "Train data: $TRAIN_DATA"
echo "Test data:  $TEST_DATA"
echo ""

# Verify trained model exists (produced by run_ensemble_train.sh)
RF_MODEL="$PROJECT_ROOT/results/rf/best_rf_model.pkl"
if [[ ! -f "$RF_MODEL" ]]; then
    echo "ERROR: RF model not found at $RF_MODEL" >&2
    echo "       Run run_ensemble_train.sh first." >&2
    exit 1
fi

echo "RF model:   $RF_MODEL"
echo ""

# Run analysis
"$PYTHON_BIN" "$PROJECT_ROOT/code/models/ensemble_analysis.py"

echo ""
echo "=== Done. ==="
echo "  HGB outputs → $PROJECT_ROOT/results/hgb/"
echo "  RF  outputs → $PROJECT_ROOT/results/rf/"
