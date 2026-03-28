"""
Part 3 – Stability Analysis + Unlabeled Sanity Check
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ----------------------------
# Load data
# ----------------------------
train = pd.read_csv("../data/train_model.csv")
val = pd.read_csv("../data/val_model.csv")
test = pd.read_csv("../data/test_model.csv")

def split_X_y(df):
    X = df.drop(columns=["label", "x", "y"])
    y = (df["label"] == 1).astype(int)
    return X, y

X_train, y_train = split_X_y(train)
X_val, y_val = split_X_y(val)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

val_prob = model.predict_proba(X_val)[:, 1]
baseline_auc = roc_auc_score(y_val, val_prob)
print("\nBaseline ROC-AUC:", baseline_auc)

# ============================================================
# 1. Noise Stability
# ============================================================

print("\n=== Noise Stability (with repeats) ===")

def add_noise(X, level):
    return X + np.random.normal(0, level, X.shape)

noise_levels = [0.0, 0.05, 0.1, 0.2]
mean_aucs, std_aucs = [], []

for nl in noise_levels:
    aucs = []
    for _ in range(5):  # repeat
        X_noisy = add_noise(X_val, nl)
        prob = model.predict_proba(X_noisy)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))

    mean_aucs.append(np.mean(aucs))
    std_aucs.append(np.std(aucs))

    print(f"Noise={nl:.2f} -> AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# plot with error bar
plt.figure()
plt.errorbar(noise_levels, mean_aucs, yerr=std_aucs, marker="o")
plt.xlabel("Noise Level")
plt.ylabel("ROC-AUC")
plt.title("Noise Stability (mean ± std)")
plt.savefig("../figs/stability_noise_enhanced.png")
plt.close()

# ============================================================
# 2. Bootstrap Stability
# ============================================================

print("\n=== Bootstrap Stability ===")

boot_aucs = []

for i in range(5):
    sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[sample_idx]
    y_boot = y_train.iloc[sample_idx]

    m = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=i
    )
    m.fit(X_boot, y_boot)

    prob = m.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob)
    boot_aucs.append(auc)

print("Bootstrap AUCs:", boot_aucs)

plt.figure()
plt.hist(boot_aucs, bins=10)
plt.title("Bootstrap AUC Distribution")
plt.savefig("../figs/bootstrap_auc.png")
plt.close()

# ============================================================
# 3. Prediction Consistency
# ============================================================

print("\n=== Prediction Consistency ===")

X_noisy = add_noise(X_val, 0.1)

prob_clean = model.predict_proba(X_val)[:, 1]
prob_noisy = model.predict_proba(X_noisy)[:, 1]

diff = np.abs(prob_clean - prob_noisy)

print("Mean prediction change:", diff.mean())

plt.figure()
plt.hist(diff, bins=50)
plt.title("Prediction Change under Noise")
plt.xlabel("|p_clean - p_noisy|")
plt.savefig("../figs/prediction_change.png")
plt.close()

# ============================================================
# 4. Feature Importance
# ============================================================

importances = model.feature_importances_
feat_names = X_train.columns

imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(6,4))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top Feature Importances")
plt.savefig("../figs/feature_importance_rf.png")
plt.close()

# ============================================================
# 5. Unlabeled Sanity Check
# ============================================================

print("\n=== Unlabeled Sanity Check ===")

X_test = test.drop(columns=["label", "x", "y"], errors="ignore")
probs = model.predict_proba(X_test)[:, 1]

print("Prediction summary:")
print("mean:", probs.mean())

# 1. Histogram
plt.figure()
plt.hist(probs, bins=50)
plt.title("Prediction Distribution")
plt.savefig("../figs/unlabeled_hist.png")
plt.close()

# 2. Spatial map
plt.figure(figsize=(6,5))
plt.scatter(test["x"], test["y"], c=probs, s=1, cmap="coolwarm")
plt.colorbar()
plt.title("Prediction Map")
plt.savefig("../figs/unlabeled_map.png")
plt.close()

# 3. Confidence map
confident = (probs > 0.9) | (probs < 0.1)

plt.figure(figsize=(6,5))
plt.scatter(
    test["x"][confident],
    test["y"][confident],
    c=probs[confident],
    s=1,
    cmap="coolwarm"
)
plt.title("High Confidence Predictions")
plt.savefig("../figs/unlabeled_confident.png")
plt.close()

print("\nDone.")