"""Logistic regression diagnostic plots and multicollinearity checks.

This script fits a tuned logistic regression (with standard scaling) and
produces diagnostics useful for checking model assumptions:

- Convergence and coef summaries
- Binned logit plots to check linearity of the log-odds vs predictors
- VIF (variance inflation factor) to detect multicollinearity
- Calibration plot and Brier score

Outputs are saved under `../results/logistic_diagnostics/`.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------- colors ----------
navy = "#1f4e79"
orange = "#c55a11"
green = "#2e8b57"
teal = "#3b7a78"
crimson = "#8c2d2d"
gray = "#666666"


def split_features_target(df):
    """Split a DataFrame into feature matrix X and binary target y.

    Drops spatial coordinates 'x' and 'y' and returns copies to avoid
    accidental modification of the original DataFrame.
    """
    target_col = "label"
    drop_cols = [target_col, "x", "y"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = (df[target_col] == 1).astype(int)
    return X, y


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_logit(p, eps=1e-6):
    """Compute logit(p) with clipping to avoid infinite values for p near 0/1."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def compute_vif(X):
    """Compute variance inflation factors for features in X.

    Drops constant columns before computing VIF. Returns a DataFrame
    with features sorted by VIF (descending).
    """
    X_num = X.copy()
    nunique = X_num.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X_num = X_num.drop(columns=constant_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    rows = []
    for i, col in enumerate(X_num.columns):
        vif_val = variance_inflation_factor(X_scaled, i)
        rows.append({"feature": col, "VIF": float(vif_val)})

    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def make_binned_logit_plot(df, feature, out_path, n_bins=20):
    """Create a binned logit plot to inspect linearity in the log-odds.

    The function bins the feature (quantiles), computes event rate per bin,
    and plots the empirical log-odds vs the bin mean. A linear fit is
    overplotted when there are at least two bins.
    """
    temp = df[[feature, "y"]].dropna().copy()
    temp["bin"] = pd.qcut(temp[feature], q=n_bins, duplicates="drop")

    grouped = temp.groupby("bin", observed=False).agg(
        feature_mean=(feature, "mean"),
        event_rate=("y", "mean"),
        count=("y", "size")
    ).reset_index(drop=True)

    grouped = grouped[(grouped["event_rate"] > 0) & (grouped["event_rate"] < 1)].copy()
    grouped["emp_logit"] = safe_logit(grouped["event_rate"])

    plt.figure(figsize=(6, 4))
    plt.scatter(
        grouped["feature_mean"],
        grouped["emp_logit"],
        s=38,
        color=navy,
        edgecolor="white",
        linewidth=0.5,
        zorder=3
    )

    if len(grouped) >= 2:
        z = np.polyfit(grouped["feature_mean"], grouped["emp_logit"], deg=1)
        xline = np.linspace(grouped["feature_mean"].min(), grouped["feature_mean"].max(), 200)
        yline = np.polyval(z, xline)
        plt.plot(xline, yline, color=orange, linewidth=2.2)

    plt.xlabel(feature)
    plt.ylabel("Empirical log-odds")
    plt.title(f"Binned logit plot: {feature}")
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


# ----------------------------
# Load data
# ----------------------------
train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")

X_train_all, y_train = split_features_target(train)
X_val_all, y_val = split_features_target(val)

base_features = [
    "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN",
    "NDAI_local_mean", "NDAI_local_std", "NDAI_local_min", "NDAI_local_max",
    "SD_local_mean", "SD_local_std", "SD_local_min", "SD_local_max",
    "CORR_local_mean", "CORR_local_std", "CORR_local_min", "CORR_local_max",
    "AN_local_mean", "AN_local_std", "AN_local_min", "AN_local_max",
    "AF_local_mean", "AF_local_std", "AF_local_min", "AF_local_max",
]

ae_features = [c for c in X_train_all.columns if c.startswith("ae")]
all_features = [c for c in base_features + ae_features if c in X_train_all.columns]

X_train = X_train_all[all_features].copy()
X_val = X_val_all[all_features].copy()

out_dir = "../results/logistic_diagnostics"
ensure_dir(out_dir)

# ----------------------------
# Fit tuned logistic regression
# ----------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        C=0.005,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=3
    ))
])

pipe.fit(X_train, y_train)

lr = pipe.named_steps["lr"]

train_prob = pipe.predict_proba(X_train)[:, 1]
val_prob = pipe.predict_proba(X_val)[:, 1]
train_pred = pipe.predict(X_train)
val_pred = pipe.predict(X_val)

metrics = pd.DataFrame([
    {
        "split": "train",
        "accuracy": accuracy_score(y_train, train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
        "roc_auc": roc_auc_score(y_train, train_prob),
        "brier_score": brier_score_loss(y_train, train_prob),
    },
    {
        "split": "validation",
        "accuracy": accuracy_score(y_val, val_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
        "roc_auc": roc_auc_score(y_val, val_prob),
        "brier_score": brier_score_loss(y_val, val_prob),
    },
])
metrics.to_csv("../results/logistic_diagnostics/logistic_metrics.csv", index=False)

# ----------------------------
# Convergence info
# ----------------------------
convergence_info = {
    "solver": lr.solver,
    "n_iter": lr.n_iter_.tolist(),
    "max_iter": lr.max_iter,
    "intercept": lr.intercept_.tolist(),
}
with open("../results/logistic_diagnostics/logistic_convergence.json", "w") as f:
    json.dump(convergence_info, f, indent=2)

# ----------------------------
# Coefficients
# ----------------------------
coef = lr.coef_.ravel()
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coef": coef,
    "abs_coef": np.abs(coef),
}).sort_values("abs_coef", ascending=False)
coef_df.to_csv("../results/logistic_diagnostics/logistic_coefficients.csv", index=False)

top_coef = coef_df.head(15).sort_values("coef")
coef_colors = [orange if x < 0 else navy for x in top_coef["coef"]]

plt.figure(figsize=(7.5, 6))
plt.barh(top_coef["feature"], top_coef["coef"], color=coef_colors)
plt.axvline(0, color=gray, linewidth=1)
plt.xlabel("Standardized coefficient")
plt.title("Top logistic regression coefficients")
plt.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("../results/logistic_diagnostics/logistic_top_coefficients.png", dpi=250, bbox_inches="tight")
plt.close()

# ----------------------------
# VIF
# ----------------------------
vif_df = compute_vif(X_train)
vif_df.to_csv("../results/logistic_diagnostics/logistic_vif.csv", index=False)

top_vif = vif_df.head(15).sort_values("VIF", ascending=True)

plt.figure(figsize=(7.5, 6))
plt.barh(top_vif["feature"], top_vif["VIF"], color=teal)
plt.xlabel("VIF")
plt.title("Top VIF values")
plt.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("../results/logistic_diagnostics/logistic_top_vif.png", dpi=250, bbox_inches="tight")
plt.close()

# ----------------------------
# Calibration plot
# ----------------------------
prob_true, prob_pred = calibration_curve(y_val, val_prob, n_bins=10, strategy="quantile")

plt.figure(figsize=(5.5, 5.5))
plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    markersize=7,
    linewidth=2.2,
    color=navy,
    label="Validation"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    color=orange,
    label="Perfect calibration"
)
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Logistic regression calibration")
plt.legend(frameon=True)
plt.grid(alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("../results/logistic_diagnostics/logistic_calibration.png", dpi=250, bbox_inches="tight")
plt.close()

# ----------------------------
# Binned logit plots for representative features
# ----------------------------
plot_features = ["ae0", "ae1", "AF", "AN", "CORR", "SD"]
plot_features = [f for f in plot_features if f in X_train.columns]

plot_df = X_train.copy()
plot_df["y"] = y_train

for feature in plot_features:
    make_binned_logit_plot(
        plot_df,
        feature,
        f"../results/logistic_diagnostics/logit_linearity_{feature}.png",
        n_bins=20
    )

print("Saved outputs to:", out_dir)
print("Representative features for linearity checks:", plot_features)