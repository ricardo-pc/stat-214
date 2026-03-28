"""
Plot ROC curves for the validation set predictions of the selected models. 
This script re-builds the selected models (from the hyperparameter search in `part3_modeling.py`) 
using their chosen hyperparameters, fits them on the training data, and evaluates on the held-out validation set. 
It plots the ROC curves for each model and saves the figure under `../figs/roc_curve_val.png`.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score


def split_features_target(df):
    """Return (X, y) where X are the features and y is a binary label vector.
    """
    target_col = "label"
    drop_cols = [target_col, "x", "y"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = (df[target_col] == 1).astype(int)
    return X, y


os.makedirs("../results", exist_ok=True)

train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")

X_train, y_train = split_features_target(train)
X_val, y_val = split_features_target(val)

base_features = [
    "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN",
    "NDAI_local_mean", "NDAI_local_std", "NDAI_local_min", "NDAI_local_max",
    "SD_local_mean", "SD_local_std", "SD_local_min", "SD_local_max",
    "CORR_local_mean", "CORR_local_std", "CORR_local_min", "CORR_local_max",
    "AN_local_mean", "AN_local_std", "AN_local_min", "AN_local_max",
    "AF_local_mean", "AF_local_std", "AF_local_min", "AF_local_max",
]

ae_features = [col for col in X_train.columns if col.startswith("ae")]
all_features = base_features + ae_features

X_train_base = X_train[base_features].copy()
X_val_base = X_val[base_features].copy()
X_train_all = X_train[all_features].copy()
X_val_all = X_val[all_features].copy()

# Fit selected models
qda_base = QuadraticDiscriminantAnalysis(reg_param=0.005)
qda_base.fit(X_train_base, y_train)

qda_all = QuadraticDiscriminantAnalysis(reg_param=0.005)
qda_all.fit(X_train_all, y_train)

rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=3,
    n_jobs=1,
)
rf.fit(X_train_all, y_train)

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        C=0.005,
        class_weight="balanced",
        solver="liblinear",
        max_iter=20000,
        random_state=3,
    ))
])
lr.fit(X_train_all, y_train)

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=3,
)
gb.fit(X_train_all, y_train)

# Validation probabilities
model_probs = {
    "Random Forest": rf.predict_proba(X_val_all)[:, 1],
    "Gradient Boosting": gb.predict_proba(X_val_all)[:, 1],
    "QDA (base features)": qda_base.predict_proba(X_val_base)[:, 1],
    "QDA (all features)": qda_all.predict_proba(X_val_all)[:, 1],
    "Logistic Regression": lr.predict_proba(X_val_all)[:, 1],
}

colors = {
    "Random Forest": "#1f4e79",
    "Gradient Boosting": "#c55a11",
    "QDA (base features)": "#2e8b57",
    "QDA (all features)": "#7a3b69",
    "Logistic Regression": "#3b7a78",
}

plt.figure(figsize=(7, 6))

for model_name, probs in model_probs.items():
    fpr, tpr, _ = roc_curve(y_val, probs)
    auc = roc_auc_score(y_val, probs)
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        color=colors[model_name],
        label=f"{model_name} (AUC = {auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#808080")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Validation ROC curves")
plt.legend(loc="lower right", frameon=True)
plt.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("../figs/roc_curve_val.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: ../figs/roc_curve_val.png")