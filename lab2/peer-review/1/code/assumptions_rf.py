"""Random Forest diagnostic plots and metrics.

Fits a Random Forest with final hyperparameters on the training set and
evaluates on a held-out validation set. Saves metrics, calibration plots,
probability distributions by class, feature importances, and the confusion
matrix for later inspection.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


def splitFeaturesTarget(df):
    """Extract features and binary target from the prepared DataFrame.

    Drops spatial columns 'x','y' and converts label to 0/1.
    """
    targetCol = "label"
    dropCols = [targetCol, "x", "y"]
    featureCols = [col for col in df.columns if col not in dropCols]
    X = df[featureCols].copy()
    y = (df[targetCol] == 1).astype(int)
    return X, y


def computeMetrics(yTrue, yPred, yProb):
    """Compute common binary-classification metrics used in reports."""
    return {
        "accuracy": accuracy_score(yTrue, yPred),
        "balanced_accuracy": balanced_accuracy_score(yTrue, yPred),
        "precision": precision_score(yTrue, yPred, zero_division=0),
        "recall": recall_score(yTrue, yPred, zero_division=0),
        "f1": f1_score(yTrue, yPred, zero_division=0),
        "roc_auc": roc_auc_score(yTrue, yProb),
    }


os.makedirs("../results/rf_assumptions", exist_ok=True)

train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")

XTrain, yTrain = splitFeaturesTarget(train)
XVal, yVal = splitFeaturesTarget(val)

rfModel = RandomForestClassifier(
    n_estimators=250,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=3,
    n_jobs=1
)

rfModel.fit(XTrain, yTrain)

trainProb = rfModel.predict_proba(XTrain)[:, 1]
valProb = rfModel.predict_proba(XVal)[:, 1]

# Threshold for discrete predictions when computing accuracy, precision, etc.
threshold = 0.5
trainPred = (trainProb >= threshold).astype(int)
valPred = (valProb >= threshold).astype(int)

trainMetrics = computeMetrics(yTrain, trainPred, trainProb)
valMetrics = computeMetrics(yVal, valPred, valProb)

metricsDf = pd.DataFrame([
    {"split": "train", **trainMetrics},
    {"split": "validation", **valMetrics},
])
metricsDf.to_csv("../results/rf_assumptions/rf_metrics.csv", index=False)

with open("../results/rf_assumptions/rf_fit_info.json", "w") as f:
    json.dump(
        {
            "n_estimators": 250,
            "max_depth": 15,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "threshold": 0.5,
        },
        f,
        indent=2,
    )

# ----------------------------
# Calibration plot
# ----------------------------
probTrue, probPred = calibration_curve(yVal, valProb, n_bins=10, strategy="quantile")

plt.figure(figsize=(5, 5))
plt.plot(probPred, probTrue, marker="o", color="#1f4e79", linewidth=2, label="Validation")
plt.plot([0, 1], [0, 1], linestyle="--", color="#c55a11", linewidth=2, label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Random Forest Calibration")
plt.legend()
plt.tight_layout()
plt.savefig("../results/rf_assumptions/rf_calibration.png", dpi=200)
plt.close()

# ----------------------------
# Predicted probability by class
# ----------------------------
plt.figure(figsize=(7, 5))
plt.hist(valProb[yVal == 0], bins=30, alpha=0.65, density=True, color="#4f81bd", label="True class 0")
plt.hist(valProb[yVal == 1], bins=30, alpha=0.65, density=True, color="#f79646", label="True class 1")
plt.xlabel("Predicted probability of cloud")
plt.ylabel("Density")
plt.title("Validation Predicted Probabilities by Class")
plt.legend()
plt.tight_layout()
plt.savefig("../results/rf_assumptions/rf_prob_by_class.png", dpi=200)
plt.close()

# ----------------------------
# Built-in feature importance
# ----------------------------
importanceDf = pd.DataFrame({
    "feature": XTrain.columns,
    "importance": rfModel.feature_importances_
}).sort_values("importance", ascending=False)

importanceDf.to_csv("../results/rf_assumptions/rf_feature_importance.csv", index=False)

topImp = importanceDf.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(7, 6))
plt.barh(topImp["feature"], topImp["importance"], color="#2e8b57")
plt.xlabel("Feature importance")
plt.title("Top Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("../results/rf_assumptions/rf_top_importance.png", dpi=200)
plt.close()

# ----------------------------
# Confusion matrix
# ----------------------------
cm = confusion_matrix(yVal, valPred)
cmDf = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
cmDf.to_csv("../results/rf_assumptions/rf_confusion_matrix.csv")

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Random Forest Confusion Matrix")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("../results/rf_assumptions/rf_confusion_matrix.png", dpi=200)
plt.close()

print(metricsDf)
print("\nSaved files in ../results/rf_assumptions/")