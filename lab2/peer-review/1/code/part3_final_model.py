"""Final model training and test evaluation for Part 3.

This script takes the selected final model (Random Forest on the `all`
feature set), refits it on the combined training + validation data, and
evaluates performance on the held-out test set. It saves a small suite of
artifacts under `../results/final_model/` for reporting and error analysis.

Outputs written:
 - `final_test_metrics.csv` : summary metrics (accuracy, balanced accuracy, F1, ROC-AUC)
 - `final_model_info.json` : meta information about the model and dataset sizes
 - `final_test_predictions.csv` : per-patch coordinates, true label, probability, predicted label, error type
 - `final_feature_importance.csv` and `final_feature_importance.png` : RF feature importances
 - `final_confusion_matrix.csv` and `final_confusion_matrix.png` : confusion matrix on test set

Usage (from repository root):
        python code/part3_final_model.py

Notes:
 - The script assumes preprocessing has been run and expects `train_model.csv`,
     `val_model.csv`, and `test_model.csv` to exist in `../data/`.
 - The selected Random Forest hyperparameters were discovered during tuning
     (see `part3_model_tuning.py` / `part3_modeling.py`).
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def split_features_target(df):
    """Return (X, y) where X are the features and y is a binary label vector.

    This function drops spatial coordinate columns ('x','y') and converts the
    label column into a 0/1 vector where 1 denotes the positive class.
    """
    target_col = "label"
    drop_cols = [target_col, "x", "y"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = (df[target_col] == 1).astype(int)
    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    """Compute a small dictionary of evaluation metrics for reporting.

    `y_prob` should be the predicted probability for the positive class.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


os.makedirs("../results/final_model", exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
# Expect that preprocessing / feature engineering has generated these files
train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")
test = pd.read_csv("../data/test_model.csv")

# Combine training and validation to produce the dataset used to fit the final model
train_val = pd.concat([train, val], axis=0, ignore_index=True)

# Split into feature matrices and binary labels
X_train_val, y_train_val = split_features_target(train_val)
X_test, y_test = split_features_target(test)

# Preserve spatial coordinates (x,y) for downstream error analysis / plotting
test_coords = test[["x", "y"]].copy()

# ----------------------------
# Fit final random forest
# The hyperparameters below are the chosen final configuration from tuning
# ----------------------------
rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=3,
    n_jobs=1,
)

rf_model.fit(X_train_val, y_train_val)

# Use a fixed decision threshold to convert probabilities into class labels
threshold = 0.5
test_prob = rf_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

# ----------------------------
# Final test metrics
# ----------------------------
test_metrics = compute_metrics(y_test, test_pred, test_prob)
metrics_df = pd.DataFrame([test_metrics])
metrics_df.to_csv("../results/final_model/final_test_metrics.csv", index=False)

# Save a small JSON describing the model and dataset sizes (useful for reproducibility)
with open("../results/final_model/final_model_info.json", "w") as f:
    json.dump(
        {
            "model": "Random Forest",
            "feature_set": "all",
            "n_estimators": 250,
            "max_depth": 15,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "threshold": 0.5,
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "train_val_rows": int(len(train_val)),
            "test_rows": int(len(test)),
        },
        f,
        indent=2,
    )

# ----------------------------
# Save test predictions (with coordinates and an error-type label)
# ----------------------------
pred_df = test_coords.copy()
pred_df["true_label"] = y_test.values
pred_df["pred_prob"] = test_prob
pred_df["pred_label"] = test_pred

def label_error_type(row):
    """Categorize each prediction as TP/TN/FP/FN for quick error analysis."""
    if row["true_label"] == 1 and row["pred_label"] == 1:
        return "TP"
    elif row["true_label"] == 0 and row["pred_label"] == 0:
        return "TN"
    elif row["true_label"] == 0 and row["pred_label"] == 1:
        return "FP"
    else:
        return "FN"

pred_df["error_type"] = pred_df.apply(label_error_type, axis=1)
pred_df.to_csv("../results/final_model/final_test_predictions.csv", index=False)

# ----------------------------
# Feature importance
# ----------------------------
importance_df = pd.DataFrame({
    "feature": X_train_val.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv("../results/final_model/final_feature_importance.csv", index=False)

# Plot the top 15 features for a compact visualization
top_imp = importance_df.head(15).sort_values("importance", ascending=True)

plt.figure(figsize=(7.5, 6))
plt.barh(top_imp["feature"], top_imp["importance"], color="#2e8b57")
plt.xlabel("Feature importance")
plt.title("Final random forest feature importance")
plt.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("../results/final_model/final_feature_importance.png", dpi=250, bbox_inches="tight")
plt.close()

# ----------------------------
# Confusion matrix
# ----------------------------
cm = confusion_matrix(y_test, test_pred)
cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
cm_df.to_csv("../results/final_model/final_confusion_matrix.csv")

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
plt.title("Final test confusion matrix")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("../results/final_model/final_confusion_matrix.png", dpi=250, bbox_inches="tight")
plt.close()

print("Final test metrics:")
print(metrics_df.to_string(index=False))
print("\nSaved outputs to ../results/final_model/")