"""Part 3 predictive modeling confirmation script.

This script re-builds the selected models (from the hyperparameter search in
`part3_model_tuning.py`) using their chosen hyperparameters, fits them on the
training set, and evaluates on the held-out validation set.

We evaluate each model class on two feature sets:
    - base features: hand-engineered predictors
    - all features: base features augmented with autoencoder-derived embeddings

The goal is to confirm whether adding the AE embeddings improves validation
performance for each model family. Final test-set evaluation is performed in
`part3_final_model.py`.

Usage (from repository root):
        python code/part3_modeling.py

Outputs:
    - ../results/validation_model_comparison_base_vs_all.csv : DataFrame with per-model metrics
"""

import os
import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def split_features_target(df):
    """Return (X, y) given a DataFrame `df`.

    Assumes the label column is named 'label' and drops spatial columns 'x' and 'y'.
    Returns copies of the slices to avoid accidental mutation of original data.
    """
    target_col = "label"
    drop_cols = [target_col, "x", "y"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def convert_labels(y):
    """Convert dataset labels to binary 0/1 vector.

    The dataset encodes the positive class as `1`; this helper ensures the
    downstream classifiers get consistent binary labels.
    """
    return (y == 1).astype(int)


def summarize_results(model_name, y_true, y_pred, y_score):
    """Return a compact dictionary of evaluation metrics for reporting.

    `y_score` should be the predicted probability for the positive class.
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
    # Nicely formatted console report for quick inspection
    print(f"\n{'='*60}")
    print(model_name)
    print(f"{'='*60}")

    print("\nTrain classification report:")
    print(classification_report(y_train_true, y_train_pred, zero_division=0))

    print("Validation classification report:")
    print(classification_report(y_val_true, y_val_pred, zero_division=0))

    print("Train ROC-AUC:", roc_auc_score(y_train_true, y_train_score))
    print("Val ROC-AUC:", roc_auc_score(y_val_true, y_val_score))


def fit_qda(X, y, reg_param=0.005):
    """Instantiate and fit a QDA model with the chosen regularization.

    reg_param is a small positive value that regularizes the covariance estimates
    and helps stability when features are correlated or sample sizes are small.
    """
    model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    model.fit(X, y)
    return model


def fit_rf(X, y, n_estimators=250, max_depth=15,
           min_samples_leaf=2, min_samples_split=2,
           class_weight="balanced"):
    """Build and fit a Random Forest classifier with tuned hyperparameters.

    The defaults here reflect the best params discovered during grid search.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=3,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def fit_logistic_regression(X, y, C=0.005,
                            class_weight="balanced",
                            solver="liblinear"):
    """Build a logistic regression pipeline with standard scaling.

    The scaler is required because LR is sensitive to feature scaling.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=20000,
            random_state=3,
        ))
    ])
    model.fit(X, y)
    return model


def fit_gradient_boosting(X, y, n_estimators=150,
                          learning_rate=0.05, max_depth=3,
                          min_samples_leaf=2, subsample=0.8):
    """Instantiate and fit a Gradient Boosting classifier with chosen hyperparams."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=3,
    )
    model.fit(X, y)
    return model


def evaluate_model(model_name, model, X_train, X_val, y_train_bin, y_val_bin):
    # Predict class labels and probabilities for train/validation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # We assume the model exposes predict_proba; this holds for all classifiers used here
    y_train_score = model.predict_proba(X_train)[:, 1]
    y_val_score = model.predict_proba(X_val)[:, 1]

    # Print a human-readable report and return a compact summary dict
    print_model_report(
        model_name,
        y_train_bin, y_train_pred, y_train_score,
        y_val_bin, y_val_pred, y_val_score
    )

    return summarize_results(model_name, y_val_bin, y_val_pred, y_val_score)


def main():
    train = pd.read_csv("../data/train_model.csv")
    val = pd.read_csv("../data/val_model.csv")

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

    ae_features = [col for col in X_train.columns if col.startswith("ae")]
    all_features = base_features + ae_features

    X_train_base = X_train[base_features].copy()
    X_val_base = X_val[base_features].copy()

    X_train_all = X_train[all_features].copy()
    X_val_all = X_val[all_features].copy()

    print("Train base:", X_train_base.shape, y_train_bin.shape)
    print("Val base:", X_val_base.shape, y_val_bin.shape)
    print("Train all:", X_train_all.shape, y_train_bin.shape)
    print("Val all:", X_val_all.shape, y_val_bin.shape)

    # Collect per-model evaluation dicts here
    results = []

    # QDA
    qda_base = fit_qda(X_train_base, y_train_bin, reg_param=0.005)
    qda_all = fit_qda(X_train_all, y_train_bin, reg_param=0.005)

    results.append(evaluate_model("QDA_base", qda_base, X_train_base, X_val_base, y_train_bin, y_val_bin))
    results.append(evaluate_model("QDA_all", qda_all, X_train_all, X_val_all, y_train_bin, y_val_bin))

    # Random Forest
    rf_base = fit_rf(X_train_base, y_train_bin)
    rf_all = fit_rf(X_train_all, y_train_bin)

    results.append(evaluate_model("Random Forest (base features)", rf_base, X_train_base, X_val_base, y_train_bin, y_val_bin))
    results.append(evaluate_model("Random Forest (all features)", rf_all, X_train_all, X_val_all, y_train_bin, y_val_bin))

    # Logistic Regression
    lr_base = fit_logistic_regression(X_train_base, y_train_bin)
    lr_all = fit_logistic_regression(X_train_all, y_train_bin)

    results.append(evaluate_model("Logistic Regression (base features)", lr_base, X_train_base, X_val_base, y_train_bin, y_val_bin))
    results.append(evaluate_model("Logistic Regression (all features)", lr_all, X_train_all, X_val_all, y_train_bin, y_val_bin))

    # Gradient Boosting
    gb_base = fit_gradient_boosting(X_train_base, y_train_bin)
    gb_all = fit_gradient_boosting(X_train_all, y_train_bin)

    results.append(evaluate_model("Gradient Boosting (base features)", gb_base, X_train_base, X_val_base, y_train_bin, y_val_bin))
    results.append(evaluate_model("Gradient Boosting (all features)", gb_all, X_train_all, X_val_all, y_train_bin, y_val_bin))

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    print("\nValidation summary:")
    print(results_df.to_string(index=False))

    os.makedirs("../results", exist_ok=True)
    results_df.to_csv("../results/validation_model_comparison_base_vs_all.csv", index=False)

    print("\nSaved:")
    print("../results/validation_model_comparison_base_vs_all.csv")


if __name__ == "__main__":
    main()

