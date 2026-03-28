"""
Post-hoc EDA for the final selected classifier.

This script uses the saved outputs from final_model.py and focuses only on
misclassification patterns for the held-out test image.

Main questions:
1. Are false positives and false negatives concentrated in particular regions?
2. Do errors occur in particular ranges of important feature values?
3. Are errors associated with lower-confidence predictions?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
TEST_PATH = "../data/test_model.csv"
PRED_PATH = "../results/final_model/final_test_predictions.csv"
OUT_DIR = "../results/posthoc_eda_rf"

top_features = ["AF_local_std", "AN_local_std", "SD_local_mean", "SD_local_min", "SD_local_max"]

BINS = 30


# --------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def make_output_dir(path):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_inputs():
    """Load test data and predictions from CSV files."""
    test_df = pd.read_csv(TEST_PATH)
    pred_df = pd.read_csv(PRED_PATH)
    return test_df, pred_df


def attach_predictions(test_df, pred_df):
    """Merge test data with predictions and compute error types and related variables. """
    # Merge on spatial coordinates to align predictions with true labels
    merged = test_df.merge(
        pred_df,
        left_on=["x", "y"],
        right_on=["x", "y"],
        how="inner",
        validate="one_to_one",
    )
    # Compute error type: correct, false positive, or false negative
    merged["correct"] = (merged["true_label"] == merged["pred_label"]).astype(int)
    merged["is_error"] = 1 - merged["correct"]
    merged["pred_confidence"] = np.where(
        merged["pred_label"] == 1,
        merged["pred_prob"],
        1 - merged["pred_prob"],
    )
    merged["uncertainty"] = np.abs(merged["pred_prob"] - 0.5) # distance from 0.5 as a measure of uncertainty

    merged["outcome4"] = np.select(
        [
            (merged["true_label"] == 0) & (merged["pred_label"] == 0),
            (merged["true_label"] == 0) & (merged["pred_label"] == 1),
            (merged["true_label"] == 1) & (merged["pred_label"] == 0),
            (merged["true_label"] == 1) & (merged["pred_label"] == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="other",
    )

    return merged


# -------------------------------------------------------------
# Summary tables
# -------------------------------------------------------------
def error_summary_table(df):
    """Summary table of error counts and average predicted probabilities by error type."""
    summary = (
        df.groupby("error_type", observed=False)
        .agg(
            n=("error_type", "size"),
            mean_prob=("pred_prob", "mean"),
            mean_confidence=("pred_confidence", "mean"),
            mean_uncertainty=("uncertainty", "mean"),
        )
        .reset_index()
    )
    return summary


def feature_summary_by_error(df, features):
    """"Summary table of feature distributions by error type."""
    rows = []
    for col in features:
        for group in ["correct", "FP", "FN"]:
            vals = df.loc[df["error_type"] == group, col].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "feature": col,
                "group": group,
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
                "n": len(vals),
            })
    return pd.DataFrame(rows)


def region_error_summary(df, bins=10):
    """Summary table of error rates by spatial bins."""
    temp = df[["x", "y", "is_error"]].copy()
    temp["x_bin"] = pd.cut(temp["x"], bins=bins, duplicates="drop")
    temp["y_bin"] = pd.cut(temp["y"], bins=bins, duplicates="drop")

    summary = (
        temp.groupby(["y_bin", "x_bin"], observed=False)
        .agg(error_rate=("is_error", "mean"), n=("is_error", "size"))
        .reset_index()
        .sort_values(["error_rate", "n"], ascending=[False, False])
    )
    return summary


# -----------------------------------------------------------
# Plots
# -----------------------------------------------------------

def plot_error_map(df, out_path):
    """Scatter plot of spatial locations of false positives and false negatives."""
    fig, ax = plt.subplots(figsize=(8, 6))

    fp = df[df["error_type"] == "FP"]
    fn = df[df["error_type"] == "FN"]

    ax.scatter(fp["x"], fp["y"], s=10, alpha=0.75, label="False positive", color="#c55a11")
    ax.scatter(fn["x"], fn["y"], s=10, alpha=0.75, label="False negative", color="#1f4e79")

    ax.set_title("Spatial locations of misclassified pixels")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_rate_by_coordinate(df, coord_col, out_path, bins=30):
    """Line plot of misclassification rate by a single coordinate (x or y)."""
    temp = df[[coord_col, "is_error"]].copy()
    temp["bin"] = pd.cut(temp[coord_col], bins=bins, duplicates="drop")

    summary = (
        temp.groupby("bin", observed=False)
        .agg(error_rate=("is_error", "mean"), n=("is_error", "size"))
        .reset_index()
    )

    summary = summary[summary["n"] > 0].copy()
    summary["bin_mid"] = summary["bin"].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["bin_mid"], summary["error_rate"], marker="o", color="#3b7a78")
    ax.set_title(f"Misclassification rate by {coord_col}")
    ax.set_xlabel(coord_col)
    ax.set_ylabel("Misclassification rate")
    ax.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_error_rate_heatmap(df, out_path, bins=25):
    """"Heatmap of misclassification rate across spatial bins."""
    temp = df[["x", "y", "is_error"]].copy()
    temp["x_bin"] = pd.cut(temp["x"], bins=bins, duplicates="drop")
    temp["y_bin"] = pd.cut(temp["y"], bins=bins, duplicates="drop")

    heat = (
        temp.groupby(["y_bin", "x_bin"], observed=False)["is_error"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat.values, aspect="auto", origin="lower")
    ax.set_title("Spatial heatmap of misclassification rate")
    ax.set_xlabel("x bins")
    ax.set_ylabel("y bins")
    fig.colorbar(im, ax=ax, label="Error rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_feature_distributions_by_error(df, features, out_dir):
    """Histograms of feature distributions by prediction outcome."""
    for col in features:
        fig, ax = plt.subplots(figsize=(7, 5))

        correct = df.loc[df["error_type"] == "correct", col].dropna()
        fp = df.loc[df["error_type"] == "FP", col].dropna()
        fn = df.loc[df["error_type"] == "FN", col].dropna()

        ax.hist(correct, bins=BINS, alpha=0.35, density=True, label="Correct", color="#b0b0b0")
        if len(fp) > 0:
            ax.hist(fp, bins=BINS, alpha=0.55, density=True, label="False positive", color="#c55a11")
        if len(fn) > 0:
            ax.hist(fn, bins=BINS, alpha=0.55, density=True, label="False negative", color="#1f4e79")

        ax.set_title(f"{col} by prediction outcome")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dist_{col}.png"), dpi=300)
        plt.close()


def plot_boxplots_by_error(df, features, out_dir):
    """Box plots of feature distributions by prediction outcome."""
    for col in features:
        groups = [
            df.loc[df["error_type"] == "correct", col].dropna(),
            df.loc[df["error_type"] == "FP", col].dropna(),
            df.loc[df["error_type"] == "FN", col].dropna(),
        ]
        labels = ["Correct", "False positive", "False negative"]

        nonempty_groups = []
        nonempty_labels = []
        for g, lab in zip(groups, labels):
            if len(g) > 0:
                nonempty_groups.append(g)
                nonempty_labels.append(lab)

        if len(nonempty_groups) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        bp = ax.boxplot(nonempty_groups, tick_labels=nonempty_labels, showfliers=False, patch_artist=True)

        colors = ["#b0b0b0", "#c55a11", "#1f4e79"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(f"{col} by prediction outcome")
        ax.set_ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{col}.png"), dpi=300)
        plt.close()


def plot_probability_distributions(df, out_path):
    """Histograms of predicted probabilities conditional on correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, color in [("correct", "#b0b0b0"), ("FP", "#c55a11"), ("FN", "#1f4e79")]:
        vals = df.loc[df["error_type"] == label, "pred_prob"].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=BINS, alpha=0.55, density=True, label=label, color=color)

    ax.set_title("Predicted cloud probability by outcome")
    ax.set_xlabel("Predicted probability of class 1")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_probability_by_outcome4(df, out_path):
    """Histograms of predicted probabilities conditional on the 4 outcome types: TN, FP, TP, FN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    tn = df.loc[df["outcome4"] == "TN", "pred_prob"].dropna()
    fp = df.loc[df["outcome4"] == "FP", "pred_prob"].dropna()
    axes[0].hist(tn, bins=BINS, alpha=0.45, density=True, label="TN", color="#b0b0b0")
    axes[0].hist(fp, bins=BINS, alpha=0.55, density=True, label="FP", color="#c55a11")
    axes[0].set_title("Predicted cloud probability for true no-cloud")
    axes[0].set_xlabel("Predicted probability of class 1")
    axes[0].legend()

    tp = df.loc[df["outcome4"] == "TP", "pred_prob"].dropna()
    fn = df.loc[df["outcome4"] == "FN", "pred_prob"].dropna()
    axes[1].hist(tp, bins=BINS, alpha=0.45, density=True, label="TP", color="#b0b0b0")
    axes[1].hist(fn, bins=BINS, alpha=0.55, density=True, label="FN", color="#1f4e79")
    axes[1].set_title("Predicted cloud probability for true cloud")
    axes[1].set_xlabel("Predicted probability of class 1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_uncertainty_vs_error(df, out_path, bins=25):
    """Line plot of misclassification rate vs prediction uncertainty."""
    temp = df[["uncertainty", "is_error"]].copy()
    temp["bin"] = pd.cut(temp["uncertainty"], bins=bins, duplicates="drop")

    summary = (
        temp.groupby("bin", observed=False)
        .agg(error_rate=("is_error", "mean"), n=("is_error", "size"))
        .reset_index()
    )
    summary = summary[summary["n"] > 0].copy()
    summary["bin_mid"] = summary["bin"].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["bin_mid"], summary["error_rate"], marker="o", color="#3b7a78")
    ax.set_title("Misclassification rate vs prediction certainty")
    ax.set_xlabel("|predicted probability - 0.5|")
    ax.set_ylabel("Misclassification rate")
    ax.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_feature_distributions_conditional(df, features, out_dir):
    """Histograms of feature distributions conditional on the 4 outcome types: TN, FP, TP, FN."""
    colors = {
        "TN": "#b0b0b0",
        "FP": "#c55a11",
        "TP": "#b0b0b0",
        "FN": "#1f4e79",
    }

    for col in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # true no-cloud: TN vs FP
        tn = df.loc[df["outcome4"] == "TN", col].dropna()
        fp = df.loc[df["outcome4"] == "FP", col].dropna()

        axes[0].hist(tn, bins=BINS, alpha=0.45, density=True, label="TN", color=colors["TN"])
        if len(fp) > 0:
            axes[0].hist(fp, bins=BINS, alpha=0.55, density=True, label="FP", color=colors["FP"])
        axes[0].set_title(f"{col}: true no-cloud pixels")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # true cloud: TP vs FN
        tp = df.loc[df["outcome4"] == "TP", col].dropna()
        fn = df.loc[df["outcome4"] == "FN", col].dropna()

        axes[1].hist(tp, bins=BINS, alpha=0.45, density=True, label="TP", color=colors["TP"])
        if len(fn) > 0:
            axes[1].hist(fn, bins=BINS, alpha=0.55, density=True, label="FN", color=colors["FN"])
        axes[1].set_title(f"{col}: true cloud pixels")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Density")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"conditional_dist_{col}.png"), dpi=300)
        plt.close()

def plot_error_rate_by_feature_quantile(df, features, out_dir, q=20):
    """Line plots of misclassification rate by feature quantiles."""
    for col in features:
        temp = df[[col, "is_error"]].dropna().copy()
        try:
            temp["bin"] = pd.qcut(temp[col], q=q, duplicates="drop")
        except ValueError:
            continue

        summary = (
            temp.groupby("bin", observed=False)
            .agg(error_rate=("is_error", "mean"), n=("is_error", "size"))
            .reset_index()
        )
        summary["bin_mid"] = summary["bin"].apply(lambda x: x.mid)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(summary["bin_mid"], summary["error_rate"], marker="o")
        ax.set_title(f"Misclassification rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Misclassification rate")
        ax.grid(alpha=0.2, linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"error_rate_by_{col}.png"), dpi=300)
        plt.close()

def plot_top_feature_pairs(df, features, out_path, sample_n=8000):
    """Scatter plots of top feature pairs, colored by outcome type."""
    keep = df[df["outcome4"].isin(["TN", "FP", "FN", "TP"])].copy()
    if len(keep) > sample_n:
        keep = keep.sample(sample_n, random_state=214)

    colors = {
        "TN": "#b0b0b0",
        "TP": "#9ecae1",
        "FP": "#c55a11",
        "FN": "#1f4e79",
    }

    pairs = [(features[0], features[1]), (features[0], features[2]), (features[1], features[2])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (xcol, ycol) in zip(axes, pairs):
        for group in ["TN", "TP", "FP", "FN"]:
            sub = keep[keep["outcome4"] == group]
            ax.scatter(sub[xcol], sub[ycol], s=8, alpha=0.35, label=group, color=colors[group])
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{xcol} vs {ycol}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    make_output_dir(OUT_DIR)

    test_df, pred_df = load_inputs()
    posthoc_df = attach_predictions(test_df, pred_df)

    # summaries
    err_summary = error_summary_table(posthoc_df)
    err_summary.to_csv(os.path.join(OUT_DIR, "error_summary.csv"), index=False)

    feat_summary = feature_summary_by_error(posthoc_df, top_features)
    feat_summary.to_csv(os.path.join(OUT_DIR, "feature_summary_by_error.csv"), index=False)

    region_summary = region_error_summary(posthoc_df, bins=10)
    region_summary.to_csv(os.path.join(OUT_DIR, "region_error_summary.csv"), index=False)

    # plots
    plot_error_map(posthoc_df, os.path.join(OUT_DIR, "error_map.png"))
    plot_error_rate_by_coordinate(posthoc_df, "x", os.path.join(OUT_DIR, "error_rate_by_x.png"))
    plot_error_rate_by_coordinate(posthoc_df, "y", os.path.join(OUT_DIR, "error_rate_by_y.png"))
    plot_error_rate_heatmap(posthoc_df, os.path.join(OUT_DIR, "error_rate_heatmap.png"))
    plot_error_rate_by_feature_quantile(posthoc_df, top_features, OUT_DIR)

    plot_probability_distributions(posthoc_df, os.path.join(OUT_DIR, "probability_by_outcome.png"))
    plot_uncertainty_vs_error(posthoc_df, os.path.join(OUT_DIR, "uncertainty_vs_error.png"))

    plot_feature_distributions_by_error(posthoc_df, top_features, OUT_DIR)
    plot_boxplots_by_error(posthoc_df, top_features, OUT_DIR)
    plot_feature_distributions_conditional(posthoc_df, top_features, OUT_DIR)
    plot_top_feature_pairs(posthoc_df, top_features, os.path.join(OUT_DIR, "top_feature_pairs.png"))
    plot_probability_by_outcome4(posthoc_df, os.path.join(OUT_DIR, "probability_by_outcome4.png"))

    # short text summary
    with open(os.path.join(OUT_DIR, "posthoc_summary.txt"), "w") as f:
        f.write("Post-hoc EDA for final random forest model\n")
        f.write("=" * 50 + "\n\n")
        f.write("Error summary:\n")
        f.write(err_summary.to_string(index=False))

        f.write("\n\nFeature summary by outcome:\n")
        f.write(feat_summary.to_string(index=False))

        f.write("\n\nTop 10 highest-error regions:\n")
        f.write(region_summary.head(10).to_string(index=False))

    print("Post-hoc EDA finished.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()