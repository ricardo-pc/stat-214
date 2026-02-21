"""
models.py - Classification models for PECARN TBI prediction.

Implements three models:
1. Kuppermann Clinical Decision Rule (age-stratified binary OR rule)
2. Logistic Regression (L2-regularized, balanced class weights, threshold-tuned)
3. Random Forest (hyperparameter-tuned ensemble)
"""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Features used for logistic regression and random forest ──
FEATURES_FOR_MODELING = [
    # Binary clinical signs (objective, observed by physician)
    "altered_mental_status", "skull_fx_palpable", "basilar_skull_fx",
    "fontanelle_bulging", "scalp_hematoma", "neuro_deficit",
    # Binary symptoms (reported by patient/parent)
    "loc", "amnesia", "seizure", "vomiting", "headache",
    "acting_normal",
    # Injury characteristics
    "injury_severity", "other_injuries", "drug_intoxication",
    # Demographics and vitals
    "age_years", "gcs_total", "gender",
    # Multi-category (will be one-hot encoded)
    "injury_mechanism",
]


# Model 1: Kuppermann Clinical Decision Rule

def kuppermann_predict(df):
    """Apply the Kuppermann clinical decision rule to each patient.

    Implements the age-stratified binary OR rule from Kuppermann et al. (2009).
    For children < 2 years, flags patients with: altered mental status,
    palpable skull fracture, non-frontal scalp hematoma, LOC >= 5 seconds,
    severe injury mechanism, or not acting normally.
    For children >= 2 years, flags patients with: altered mental status,
    basilar skull fracture signs, any LOC, vomiting, severe injury mechanism,
    or severe headache.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PECARN dataset with integer-coded features.

    Returns
    -------
    pd.Series
        Binary predictions (1 = recommend CT, 0 = no CT).
    """
    predictions = pd.Series(0, index=df.index, dtype="Int64")

    # Children < 2 years (age_group == 1)
    under2 = (df["age_group"] == 1).to_numpy(dtype=bool, na_value=False)
    flag_under2 = (
        (df.loc[under2, "altered_mental_status"] == 1)
        | (df.loc[under2, "skull_fx_palpable"] == 1)
        | (
            (df.loc[under2, "hematoma_location"] != 92)
            & (df.loc[under2, "hematoma_location"] != 1)
        )
        | (df.loc[under2, "loc_duration"] >= 2)
        | (df.loc[under2, "injury_severity"] == 1)
        | (df.loc[under2, "acting_normal"] == 0)
    ).fillna(False).astype(int)
    predictions.loc[under2] = flag_under2

    # Children >= 2 years (age_group == 2)
    over2 = (df["age_group"] == 2).to_numpy(dtype=bool, na_value=False)
    flag_over2 = (
        (df.loc[over2, "altered_mental_status"] == 1)
        | (df.loc[over2, "basilar_skull_fx"] == 1)
        | (df.loc[over2, "loc"] == 1)
        | (df.loc[over2, "vomiting"] == 1)
        | (df.loc[over2, "injury_severity"] == 1)
        | (df.loc[over2, "headache_severity"] == 3)
    ).fillna(False).astype(int)
    predictions.loc[over2] = flag_over2

    return predictions.astype(int)


# Shared utilities

def prepare_features(df, features=None):
    """Prepare feature matrix with one-hot encoding and missing value handling.

    Selects modeling features, one-hot encodes injury_mechanism,
    converts to float, and fills NaN with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned PECARN dataset.
    features : list, optional
        Feature column names. Defaults to FEATURES_FOR_MODELING.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for modeling.
    pd.Series
        Binary outcome vector.
    """
    if features is None:
        features = FEATURES_FOR_MODELING

    X = df[features].copy()
    y = df["clinically_important_tbi"].copy()

    # One-hot encode injury_mechanism (drop first to avoid multicollinearity)
    X = pd.get_dummies(X, columns=["injury_mechanism"], drop_first=True, dtype=int)

    # Convert nullable Int64 to float, fill NaN with 0
    X = X.astype(float).fillna(0)

    return X, y


def split_data(X, y, random_state=214):
    """Split into train (70%), validation (15%), test (15%) with stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary outcome.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_best_threshold(y_true, y_proba, min_sensitivity=90.0):
    """Find the highest threshold achieving >= min_sensitivity on a dataset.

    Sweeps thresholds from 0.01 to 0.99 and selects the one that
    maximizes specificity subject to sensitivity >= min_sensitivity.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    min_sensitivity : float
        Minimum required sensitivity (in percent).

    Returns
    -------
    float
        Optimal threshold.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_spec = -1
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

        if sens >= min_sensitivity and spec > best_spec:
            best_spec = spec
            best_t = t

    return best_t


# Model 2: Logistic Regression

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train an L2-regularized logistic regression with threshold tuning.

    Uses balanced class weights (~112x upweighting for TBI cases),
    L2 regularization (C=1.0), and tunes the decision threshold on
    the validation set to achieve >= 90% sensitivity.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation labels.

    Returns
    -------
    LogisticRegression
        Fitted model.
    StandardScaler
        Fitted scaler (needed to transform test data).
    float
        Optimal decision threshold.
    """
    # Standardize features (fit on train only to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Fit logistic regression
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        penalty="l2",
        random_state=214,
    )
    model.fit(X_train_scaled, y_train)

    # Tune threshold on validation set
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    threshold = find_best_threshold(y_val, y_val_proba, min_sensitivity=90.0)

    return model, scaler, threshold


# Model 3: Random Forest

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train a random forest with hyperparameter tuning and threshold selection.

    Evaluates 18 hyperparameter combinations (n_estimators x max_depth x
    min_samples_leaf) on the validation set. Selects the configuration that
    maximizes specificity subject to >= 90% sensitivity, then tunes the
    decision threshold.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (unscaled).
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation features (unscaled).
    y_val : pd.Series
        Validation labels.

    Returns
    -------
    RandomForestClassifier
        Best fitted model.
    float
        Optimal decision threshold.
    """
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10, 20],
    }

    combos = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_leaf"],
    ))

    best_spec = -1
    best_model = None

    for n_est, max_d, min_leaf in combos:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=min_leaf,
            max_features="sqrt",
            class_weight="balanced",
            random_state=214,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_val_proba = rf.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, y_val_proba, min_sensitivity=90.0)
        y_val_pred = (y_val_proba >= threshold).astype(int)

        tp = ((y_val_pred == 1) & (y_val == 1)).sum()
        fn = ((y_val_pred == 0) & (y_val == 1)).sum()
        tn = ((y_val_pred == 0) & (y_val == 0)).sum()
        fp = ((y_val_pred == 1) & (y_val == 0)).sum()

        sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

        if sens >= 90.0 and spec > best_spec:
            best_spec = spec
            best_model = rf

    # Final threshold tuning with the best model
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, y_val_proba, min_sensitivity=90.0)

    return best_model, best_threshold


# Evaluation utilities

def evaluate_model(y_true, y_pred, y_proba=None):
    """Compute classification metrics for a binary prediction.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_proba : array-like, optional
        Predicted probabilities (for AUC computation).

    Returns
    -------
    dict
        Dictionary with sensitivity, specificity, precision, missed_tbi,
        unnecessary_cts, and optionally auc_roc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    results = {
        "sensitivity": tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) * 100 if (tn + fp) > 0 else 0,
        "precision": tp / (tp + fp) * 100 if (tp + fp) > 0 else 0,
        "missed_tbi": int(fn),
        "unnecessary_cts": int(fp),
    }

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        results["auc_roc"] = auc(fpr, tpr)

    return results


if __name__ == "__main__":
    from clean import clean_data

    # Load and clean data
    raw_df = pd.read_csv("../data/TBI PUD 10-08-2013.csv")
    df_prep, _ = clean_data(raw_df)

    # ── Model 1: Kuppermann ──
    print("=" * 60)
    print("Model 1: Kuppermann Clinical Decision Rule")
    print("=" * 60)
    y_true = df_prep["clinically_important_tbi"]
    y_pred_k = kuppermann_predict(df_prep)
    metrics_k = evaluate_model(y_true, y_pred_k)
    for k, v in metrics_k.items():
        print(f"  {k}: {v}")

    # ── Prepare features and split ──
    X, y = prepare_features(df_prep)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Model 2: Logistic Regression ──
    print("\n" + "=" * 60)
    print("Model 2: Logistic Regression")
    print("=" * 60)
    model_lr, scaler, threshold_lr = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    print(f"  Optimal threshold: {threshold_lr:.2f}")

    X_test_scaled = scaler.transform(X_test)
    y_test_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_lr = (y_test_proba_lr >= threshold_lr).astype(int)
    metrics_lr = evaluate_model(y_test, y_test_pred_lr, y_test_proba_lr)
    for k, v in metrics_lr.items():
        print(f"  {k}: {v}")

    # ── Model 3: Random Forest ──
    print("\n" + "=" * 60)
    print("Model 3: Random Forest")
    print("=" * 60)
    model_rf, threshold_rf = train_random_forest(X_train, y_train, X_val, y_val)
    print(f"  Optimal threshold: {threshold_rf:.2f}")
    print(f"  Best params: n_estimators={model_rf.n_estimators}, "
          f"max_depth={model_rf.max_depth}, "
          f"min_samples_leaf={model_rf.min_samples_leaf}")

    y_test_proba_rf = model_rf.predict_proba(X_test)[:, 1]
    y_test_pred_rf = (y_test_proba_rf >= threshold_rf).astype(int)
    metrics_rf = evaluate_model(y_test, y_test_pred_rf, y_test_proba_rf)
    for k, v in metrics_rf.items():
        print(f"  {k}: {v}")
