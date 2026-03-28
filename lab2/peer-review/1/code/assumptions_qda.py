"""Diagnostics for QDA assumptions and visualizations.

This script fits a Quadratic Discriminant Analysis (QDA) model using the
final regularization parameter, computes basic metrics, and produces a set
of diagnostics useful for assessing QDA assumptions:

- Class-conditional histograms and QQ-plots to inspect marginal normality
- Covariance diagnostics (eigenvalues, condition number, determinant)
- 2D decision-boundary visualizations for selected feature pairs
- A PCA-based 2D decision-boundary visualization for a global view

Outputs are saved under `../results/qda_assumptions/`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import probplot
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score


# ----------------------------
# Helpers
# ----------------------------
def split_features_target(df):
    """Split a DataFrame into features X and binary target y.

    Drops coordinates 'x' and 'y' and converts the label column to 0/1.
    """
    target_col = "label"
    drop_cols = [target_col, "x", "y"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = (df[target_col] == 1).astype(int)
    return X, y


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
full_features = base_features + ae_features

X_train = X_train_all[full_features].copy()
X_val = X_val_all[full_features].copy()

out_dir = "../results/qda_assumptions"
ensure_dir(out_dir)


# ----------------------------
# Fit QDA
# ----------------------------
qda = QuadraticDiscriminantAnalysis(reg_param=0.005, store_covariance=True)
qda.fit(X_train, y_train)

train_prob = qda.predict_proba(X_train)[:, 1]
val_prob = qda.predict_proba(X_val)[:, 1]
train_pred = qda.predict(X_train)
val_pred = qda.predict(X_val)

metrics = pd.DataFrame([
    {
        "split": "train",
        "accuracy": accuracy_score(y_train, train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
        "roc_auc": roc_auc_score(y_train, train_prob),
    },
    {
        "split": "validation",
        "accuracy": accuracy_score(y_val, val_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
        "roc_auc": roc_auc_score(y_val, val_prob),
    },
])
metrics.to_csv(os.path.join(out_dir, "qda_metrics.csv"), index=False)


# ----------------------------
# 1. Covariance diagnostics
# ----------------------------
cov_rows = []
for cls, cov in zip(qda.classes_, qda.covariance_):
    eigvals = np.linalg.eigvalsh(cov)
    cov_rows.append({
        "class": int(cls),
        "n_features": cov.shape[0],
        "min_eigenvalue": float(np.min(eigvals)),
        "max_eigenvalue": float(np.max(eigvals)),
        "condition_number": float(np.linalg.cond(cov)),
        "determinant": float(np.linalg.det(cov)),
        "matrix_rank": int(np.linalg.matrix_rank(cov)),
    })

cov_df = pd.DataFrame(cov_rows)
cov_df.to_csv(os.path.join(out_dir, "qda_covariance_diagnostics.csv"), index=False)

print("\nQDA covariance diagnostics")
print(cov_df)


# ----------------------------
# 2. Class-conditional histograms
# ----------------------------
qq_features = ["NDAI", "SD", "CORR", "AF", "ae1", "ae2"]

for feature in qq_features:
    plt.figure(figsize=(7, 5))
    for cls in [0, 1]:
        vals = X_train.loc[y_train == cls, feature].dropna()
        plt.hist(vals, bins=40, density=True, alpha=0.5, label=f"class {cls}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(f"Class-conditional histogram: {feature}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_{feature}.png"), dpi=200)
    plt.close()


# ----------------------------
# 3. QQ plots for rough normality check
# ----------------------------
for feature in qq_features:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, cls in enumerate([0, 1]):
        vals = X_train.loc[y_train == cls, feature].dropna()
        probplot(vals, dist="norm", plot=axes[j])
        axes[j].set_title(f"{feature}, class {cls}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"qq_{feature}.png"), dpi=200)
    plt.close()


# ----------------------------
# 4. Pairwise 2D decision boundary plots
#    Good for visualization only
# ----------------------------
pair_list = [
    ("NDAI", "SD"),
    ("NDAI", "CORR"),
    ("SD", "CORR"),
    ("NDAI", ae_features[0]),
    ("SD", ae_features[0]),
    ("CORR", ae_features[0]),
]

for f1, f2 in pair_list:
    X_pair = X_train[[f1, f2]].copy()
    qda_pair = QuadraticDiscriminantAnalysis(reg_param=0.005)
    qda_pair.fit(X_pair, y_train)

    x_min, x_max = X_pair[f1].min(), X_pair[f1].max()
    y_min, y_max = X_pair[f2].min(), X_pair[f2].max()

    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 300),
        np.linspace(y_min - y_pad, y_max + y_pad, 300),
    )
    grid = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel()})
    zz = qda_pair.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, zz, levels=np.linspace(0, 1, 21), alpha=0.35)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)

    plt.scatter(
        X_pair.loc[y_train == 0, f1],
        X_pair.loc[y_train == 0, f2],
        s=10,
        alpha=0.5,
        label="class 0"
    )
    plt.scatter(
        X_pair.loc[y_train == 1, f1],
        X_pair.loc[y_train == 1, f2],
        s=10,
        alpha=0.5,
        label="class 1"
    )

    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f"QDA decision boundary: {f1} vs {f2}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"boundary_{f1}_{f2}.png"), dpi=200)
    plt.close()


# ----------------------------
# 5. PCA-based decision boundary visualization
#    Visual approximation in 2D
# ----------------------------
pca = PCA(n_components=2, random_state=3)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

qda_pca = QuadraticDiscriminantAnalysis(reg_param=0.005)
qda_pca.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min(), X_train_pca[:, 0].max()
y_min, y_max = X_train_pca[:, 1].min(), X_train_pca[:, 1].max()

x_pad = 0.05 * (x_max - x_min)
y_pad = 0.05 * (y_max - y_min)

xx, yy = np.meshgrid(
    np.linspace(x_min - x_pad, x_max + x_pad, 300),
    np.linspace(y_min - y_pad, y_max + y_pad, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]
zz = qda_pca.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, zz, levels=np.linspace(0, 1, 21), alpha=0.35)
plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)

plt.scatter(
    X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
    s=10, alpha=0.5, label="train class 0"
)
plt.scatter(
    X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
    s=10, alpha=0.5, label="train class 1"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("QDA decision boundary in PCA space")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "boundary_pca_qda.png"), dpi=200)
plt.close()

print("\nSaved outputs to:", out_dir)