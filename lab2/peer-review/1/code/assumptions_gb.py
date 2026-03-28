"""Gradient Boosting diagnostic plots and metrics.

This script fits a Gradient Boosting classifier with the chosen final
hyperparameters on the training set and evaluates it on a held-out
validation set. It saves a small collection of artifacts to
`../results/gb_assumptions/` to support assumption checks and reporting:

- `gb_metrics.csv` : summary metrics for train and validation splits
- `gb_fit_info.json` : hyperparameters used for the fit
- `gb_calibration.png` : calibration curve (predicted vs observed probabilities)
- `gb_feature_importance.csv` : feature importances from the trained model
- `gb_top_importance.png` : bar plot of the top features

Notes:
 - No train/val split is performed here: the script expects prepared CSVs
   `../data/train_model.csv` and `../data/val_model.csv`.
 - The script uses `predict_proba` and a fixed threshold (0.5) to derive
   class predictions when computing discrete metrics.
"""

import os
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


def splitFeaturesTarget(df):
    """Return feature matrix X and binary target y from a DataFrame.

    This helper drops spatial columns 'x' and 'y' and converts the label
    column to a binary 0/1 vector where 1 indicates the positive class.
    """
    targetCol = "label"
    dropCols = [targetCol, "x", "y"]
    featureCols = [col for col in df.columns if col not in dropCols]
    X = df[featureCols].copy()
    y = (df[targetCol] == 1).astype(int)
    return X, y


def computeMetrics(yTrue, yPred, yProb):
    """Compute a small set of metrics for binary classification.

    `yProb` is the predicted probability for the positive class and is used
    to compute ROC-AUC; `yPred` are discrete predictions (0/1).
    """
    return {
        "accuracy": accuracy_score(yTrue, yPred),
        "balanced_accuracy": balanced_accuracy_score(yTrue, yPred),
        "precision": precision_score(yTrue, yPred, zero_division=0),
        "recall": recall_score(yTrue, yPred, zero_division=0),
        "f1": f1_score(yTrue, yPred, zero_division=0),
        "roc_auc": roc_auc_score(yTrue, yProb),
    }


os.makedirs("../results/gb_assumptions", exist_ok=True)

# ----------------------------
# Load prepared train/validation CSVs and split X / y
# ----------------------------
train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")

XTrain, yTrain = splitFeaturesTarget(train)
XVal, yVal = splitFeaturesTarget(val)

# ----------------------------
# Fit the Gradient Boosting model with final/tuned hyperparameters
# ----------------------------
gbModel = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=3
)

gbModel.fit(XTrain, yTrain)

# ----------------------------
# Compute predicted probabilities and discrete predictions
# ----------------------------
trainProb = gbModel.predict_proba(XTrain)[:, 1]
valProb = gbModel.predict_proba(XVal)[:, 1]

# Use a fixed threshold (0.5) to create class predictions for discrete metrics
threshold = 0.5
trainPred = (trainProb >= threshold).astype(int)
valPred = (valProb >= threshold).astype(int)

# ----------------------------
# Compute and save metrics
# ----------------------------
trainMetrics = computeMetrics(yTrain, trainPred, trainProb)
valMetrics = computeMetrics(yVal, valPred, valProb)

metricsDf = pd.DataFrame([
    {"split": "train", **trainMetrics},
    {"split": "validation", **valMetrics},
])
metricsDf.to_csv("../results/gb_assumptions/gb_metrics.csv", index=False)

with open("../results/gb_assumptions/gb_fit_info.json", "w") as f:
    json.dump(
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "subsample": 0.8,
            "threshold": 0.5,
        },
        f,
        indent=2,
    )

# ----------------------------
# Calibration curve: compare predicted probability quantiles to observed frequency
# ----------------------------
probTrue, probPred = calibration_curve(yVal, valProb, n_bins=10, strategy="quantile")

plt.figure(figsize=(5, 5))
plt.plot(probPred, probTrue, marker="o", color="#7a3b69", linewidth=2, label="Validation")
plt.plot([0, 1], [0, 1], linestyle="--", color="#c55a11", linewidth=2, label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Gradient Boosting Calibration")
plt.legend()
plt.tight_layout()
plt.savefig("../results/gb_assumptions/gb_calibration.png", dpi=200)
plt.close()

# ----------------------------
# Feature importance table and plot
# ----------------------------
importanceDf = pd.DataFrame({
    "feature": XTrain.columns,
    "importance": gbModel.feature_importances_
}).sort_values("importance", ascending=False)

importanceDf.to_csv("../results/gb_assumptions/gb_feature_importance.csv", index=False)

topImp = importanceDf.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(7, 6))
plt.barh(topImp["feature"], topImp["importance"], color="#7a9e7e")
plt.xlabel("Feature importance")
plt.title("Top Gradient Boosting Feature Importances")
plt.tight_layout()
plt.savefig("../results/gb_assumptions/gb_top_importance.png", dpi=200)
plt.close()

print(metricsDf)
print("\nSaved files in ../results/gb_assumptions/")