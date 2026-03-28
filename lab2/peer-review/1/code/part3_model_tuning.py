"""Hyperparameter tuning and model comparison for Part 3.

This script performs grid search hyperparameter tuning for several classical
classification models and compares their validation performance.

Models tuned:
 - Quadratic Discriminant Analysis (QDA)
 - Random Forest
 - Logistic Regression
 - Gradient Boosting

Main steps:
 1) Load train/validation feature CSVs from ../data/*.csv
 2) Prepare two feature sets:
    - `base_features`: hand-crafted features
    - `all_features`: base features + autoencoder embedding features (columns starting with 'ae')
 3) For each model type, run GridSearchCV (scoring='roc_auc') and pick the best estimator.
 4) Evaluate the best estimator on train and validation sets and collect summary metrics.
 5) Save a CSV summary (`../results/validation_model_comparison.csv`) and a JSON of best params
    (`../results/best_params_part3.json`).

Usage (run from repository root):
    python code/part3_model_tuning.py

Notes:
 - The downstream script `part3_modeling.py` reconstructs selected models for final evaluation.
"""

import os
import json
import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ----------------------------
# helpers
# ----------------------------
def split_features_target(df):
    """Split a DataFrame into feature matrix X and target vector y.
    """
    target_col = "label"
    drop_cols = [target_col, "x", "y"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

def convert_labels(y):
    """Convert original labels to binary (1 for positive class, 0 otherwise).

    The dataset uses numeric labels; this ensures we have a 0/1 vector for scikit-learn.
    """
    return (y == 1).astype(int)

def summarize_results(model_name, y_true, y_pred, y_score):
    """Compute a compact set of evaluation metrics for reporting.

    Returns a dict with accuracy, balanced accuracy, precision/recall/F1 for the positive class,
    and ROC AUC computed from the probability scores `y_score`.
    """
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }

def print_model_report(model_name, y_train_true, y_train_pred, y_train_score,
                       y_val_true, y_val_pred, y_val_score):
    """Prints classification reports and ROC-AUC scores for both training and validation sets."""
    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    print("\nTrain classification report:")
    print(classification_report(y_train_true, y_train_pred))

    print("Validation classification report:")
    print(classification_report(y_val_true, y_val_pred))

    print("Train ROC-AUC:", roc_auc_score(y_train_true, y_train_score))
    print("Val ROC-AUC:", roc_auc_score(y_val_true, y_val_score))

def save_best_params(best_params_dict, out_path="../results/best_params_part3.json"):
    """Saves the best hyperparameters for each model to a JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_params_dict, f, indent=2)

# Load training and validation feature CSVs produced by the preprocessing pipeline
train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")

# Split into X (features) and y (target) and convert labels to binary
X_train, y_train = split_features_target(train)
X_val, y_val = split_features_target(val)

y_train_bin = convert_labels(y_train)
y_val_bin = convert_labels(y_val)

base_features = [
    "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN",
    "NDAI_local_mean", "NDAI_local_std", "NDAI_local_min", "NDAI_local_max",
    "SD_local_mean", "SD_local_std", "SD_local_min", "SD_local_max",
    "CORR_local_mean", "CORR_local_std", "CORR_local_min", "CORR_local_max",
    "AN_local_mean", "AN_local_std", "AN_local_min", "AN_local_max",
    "AF_local_mean", "AF_local_std", "AF_local_min", "AF_local_max",
]

# Autoencoder-derived features are expected to start with the prefix 'ae'
ae_features = [col for col in X_train.columns if col.startswith("ae")]
all_features = base_features + ae_features

# Create copies of the subsets we'll use for model training
X_train_base = X_train[base_features].copy()
X_val_base = X_val[base_features].copy()

X_train_all = X_train[all_features].copy()
X_val_all = X_val[all_features].copy()

print("Train:", X_train_all.shape, y_train_bin.shape)
print("Val:", X_val_all.shape, y_val_bin.shape)

# containers for storing per-model validation summaries and selected parameters
results_rows = []
best_params_out = {}

# QDA
print("\nTraining QDA (base features)...")
param_grid_qda = {
    "reg_param": [0.001, 0.005, 0.01]
}

grid_search_qda_base = GridSearchCV(
    estimator=QuadraticDiscriminantAnalysis(),
    param_grid=param_grid_qda,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
    verbose=2,
)

grid_search_qda_base.fit(X_train_base, y_train_bin)
qda_base = grid_search_qda_base.best_estimator_
best_params_out["QDA_base"] = grid_search_qda_base.best_params_
print("Best QDA_base Parameters:", grid_search_qda_base.best_params_)

y_train_qda_base_pred = qda_base.predict(X_train_base)
y_val_qda_base_pred = qda_base.predict(X_val_base)
y_train_qda_base_prob = qda_base.predict_proba(X_train_base)[:, 1]
y_val_qda_base_prob = qda_base.predict_proba(X_val_base)[:, 1]

print_model_report(
    "QDA_base",
    y_train_bin, y_train_qda_base_pred, y_train_qda_base_prob,
    y_val_bin, y_val_qda_base_pred, y_val_qda_base_prob
)

results_rows.append(
    summarize_results("QDA_base", y_val_bin, y_val_qda_base_pred, y_val_qda_base_prob)
)

print("\nTraining QDA (all features)...")
grid_search_qda_all = GridSearchCV(
    estimator=QuadraticDiscriminantAnalysis(),
    param_grid=param_grid_qda,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
    verbose=2,
)

grid_search_qda_all.fit(X_train_all, y_train_bin)
qda_all = grid_search_qda_all.best_estimator_
best_params_out["QDA_all"] = grid_search_qda_all.best_params_
print("Best QDA_all Parameters:", grid_search_qda_all.best_params_)

y_train_qda_all_pred = qda_all.predict(X_train_all)
y_val_qda_all_pred = qda_all.predict(X_val_all)
y_train_qda_all_prob = qda_all.predict_proba(X_train_all)[:, 1]
y_val_qda_all_prob = qda_all.predict_proba(X_val_all)[:, 1]

print_model_report(
    "QDA_all",
    y_train_bin, y_train_qda_all_pred, y_train_qda_all_prob,
    y_val_bin, y_val_qda_all_pred, y_val_qda_all_prob
)

results_rows.append(
    summarize_results("QDA_all", y_val_bin, y_val_qda_all_pred, y_val_qda_all_prob)
)

# Random Forest
print("\nTraining Random Forest...")
param_grid_rf = {
    "n_estimators": [200, 250],
    "max_depth": [15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"],
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=3, n_jobs=1),
    param_grid=param_grid_rf,
    cv=3,
    scoring="roc_auc",
    n_jobs=4,
    verbose=2,
)

grid_search_rf.fit(X_train_all, y_train_bin)
rf = grid_search_rf.best_estimator_
best_params_out["Random Forest"] = grid_search_rf.best_params_
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

rf_train_prob = rf.predict_proba(X_train_all)[:, 1]
rf_val_prob = rf.predict_proba(X_val_all)[:, 1]

rf_threshold = 0.5
y_train_rf_pred = (rf_train_prob >= rf_threshold).astype(int)
y_val_rf_pred = (rf_val_prob >= rf_threshold).astype(int)

print_model_report(
    "Random Forest",
    y_train_bin, y_train_rf_pred, rf_train_prob,
    y_val_bin, y_val_rf_pred, rf_val_prob
)

results_rows.append(
    summarize_results("Random Forest", y_val_bin, y_val_rf_pred, rf_val_prob)
)

# Logistic Regression
print("\nTraining Logistic Regression...")
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(random_state=3, max_iter=20000))
])

param_grid_lr = {
    "lr__C": [0.005, 0.01, 0.05],
    "lr__class_weight": ["balanced"],
    "lr__solver": ["liblinear"],
}

grid_search_lr = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid_lr,
    cv=3,
    scoring="roc_auc",
    n_jobs=2,
    verbose=2,
)

grid_search_lr.fit(X_train_all, y_train_bin)
lr = grid_search_lr.best_estimator_
best_params_out["Logistic Regression"] = grid_search_lr.best_params_
print("Best Logistic Regression Parameters:", grid_search_lr.best_params_)

y_train_lr_pred = lr.predict(X_train_all)
y_val_lr_pred = lr.predict(X_val_all)
y_train_lr_prob = lr.predict_proba(X_train_all)[:, 1]
y_val_lr_prob = lr.predict_proba(X_val_all)[:, 1]

print_model_report(
    "Logistic Regression",
    y_train_bin, y_train_lr_pred, y_train_lr_prob,
    y_val_bin, y_val_lr_pred, y_val_lr_prob
)

results_rows.append(
    summarize_results("Logistic Regression", y_val_bin, y_val_lr_pred, y_val_lr_prob)
)

# Gradient Boosting
print("\nTraining Gradient Boosting...")
param_grid_gb = {
    "n_estimators": [150, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "min_samples_leaf": [1, 2],
    "subsample": [0.8, 1.0],
}

grid_search_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=3),
    param_grid=param_grid_gb,
    cv=3,
    scoring="roc_auc",
    n_jobs=4,
    verbose=2,
)

grid_search_gb.fit(X_train_all, y_train_bin)
gb = grid_search_gb.best_estimator_
best_params_out["Gradient Boosting"] = grid_search_gb.best_params_
print("Best Gradient Boosting Parameters:", grid_search_gb.best_params_)

y_train_gb_pred = gb.predict(X_train_all)
y_val_gb_pred = gb.predict(X_val_all)
y_train_gb_prob = gb.predict_proba(X_train_all)[:, 1]
y_val_gb_prob = gb.predict_proba(X_val_all)[:, 1]

print_model_report(
    "Gradient Boosting",
    y_train_bin, y_train_gb_pred, y_train_gb_prob,
    y_val_bin, y_val_gb_pred, y_val_gb_prob
)

results_rows.append(
    summarize_results("Gradient Boosting", y_val_bin, y_val_gb_pred, y_val_gb_prob)
)

# save summary
results_df = pd.DataFrame(results_rows).sort_values("roc_auc", ascending=False)
print("\nValidation summary:")
print(results_df.to_string(index=False))

os.makedirs("../results", exist_ok=True)
results_df.to_csv("../results/validation_model_comparison.csv", index=False)
save_best_params(best_params_out, "../results/best_params_part3.json")

print("\nSaved:")
print("../results/validation_model_comparison.csv")
print("../results/best_params_part3.json")