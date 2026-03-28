"""
feature_engineering.py
-----------------------
Part 2 Feature Engineering for Lab 2: Cloud Detection in Polar Regions.

This script:
    1. Identifies the 3 most informative features using:
        - ANOVA F-scores
        - Mutual information scores
        - Logistic regression coefficients
    2. Engineers new patch-based features for each labeled pixel using a
       local neighborhood window, capturing spatial context that single-pixel
       features cannot.
    3. Saves the enriched feature sets as CSVs for use in Part 3 modeling.

Usage:
    Run from the code/ directory AFTER running eda.py:
        python feature_engineering.py

    eda.py must be run first to generate:
        ../data/train.csv
        ../data/val.csv
        ../data/test.csv

Outputs:
    - ../figs/feature_importance.png           bar chart of all 3 importance metrics
    - ../figs/top_features_boxplot.png         boxplot of top 3 features by class
    - ../figs/patch_feature_distributions.png  KDE plots of patch features by class
    - ../data/train_features.csv               train set with engineered patch features
    - ../data/val_features.csv                 val set with engineered patch features
    - ../data/test_features.csv                test set with engineered patch features

FOR TEAMMATES:
    Use train_features.csv, val_features.csv, test_features.csv for Part 3
    modeling. These contain all original features plus the new patch features.
    The autoencoder embeddings should be concatenated to these files when ready.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif, mutual_info_classif

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names from eda.py — must match exactly
COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
RADIANCE_COLS = ["DF", "CF", "BF", "AF", "AN"]
FEATURE_COLS = ["NDAI", "SD", "CORR"]

# All original features (excluding coordinates and label) used for importance analysis
ALL_FEATURE_COLS = FEATURE_COLS + RADIANCE_COLS

# Patch size for neighborhood features — using 5x5 window (radius=2).
# This is smaller than the autoencoder's 9x9 to be complementary, not redundant.
# Each pixel's patch features summarize the 5x5 neighborhood centered on it.
PATCH_RADIUS = 2

# Features to compute patch statistics for — subset of ALL_FEATURE_COLS.
# We focus on the engineered features and the most informative radiance angles
# rather than all 8 to avoid creating too many redundant columns.
PATCH_FEATURE_SOURCES = ["NDAI", "SD", "CORR", "AN", "AF"]

# All paths are relative to the code/ directory
DATA_DIR = "../data"
FIGS_DIR = "../figs"

# Labeled image files — needed to reconstruct spatial grids for patch features
LABELED_IMAGE_FILES = {
    "O013257": os.path.join(DATA_DIR, "image_data", "O013257.npz"),
    "O013490": os.path.join(DATA_DIR, "image_data", "O013490.npz"),
    "O012791": os.path.join(DATA_DIR, "image_data", "O012791.npz"),
}

# Which image corresponds to which split (must match eda.py split assignments)
SPLIT_IMAGE_MAP = {
    "train": "O013257",
    "val":   "O012791",
    "test":  "O013490",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create output directories if they do not already exist."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def load_split(split_name):
    """
    Load a train/val/test CSV produced by eda.py.

    Args:
        split_name: One of 'train', 'val', 'test'.

    Returns:
        df: DataFrame with labeled pixels for that split.
    """
    path = os.path.join(DATA_DIR, f"{split_name}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {split_name}: {df.shape[0]} rows")
    return df


def load_full_image(image_name):
    """
    Load a full .npz image (including unlabeled pixels) as a DataFrame.
    Needed for patch feature computation since patches may extend into
    unlabeled regions.

    Args:
        image_name: Key in LABELED_IMAGE_FILES (e.g. 'O013257').

    Returns:
        df: Full image DataFrame including unlabeled pixels.
    """
    path = LABELED_IMAGE_FILES[image_name]
    npz = np.load(path)
    key = list(npz.files)[0]
    data = npz[key]
    df = pd.DataFrame(data, columns=COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Task 1: Feature importance analysis
# ---------------------------------------------------------------------------

def compute_feature_importance(train_df):
    """
    Compute three feature importance metrics on the training set:
        - ANOVA F-score: tests whether class means differ significantly
        - Mutual information: measures non-linear dependence with the label
        - Logistic regression coefficients: model-based linear importance

    Feature selection is performed on the training set only to avoid
    data leakage from val/test into the selection process.

    Args:
        train_df: Training set DataFrame with labeled pixels.

    Returns:
        importance_df: DataFrame with normalized importance scores for each
                       feature under all three metrics, sorted by mutual info.
    """
    X = train_df[ALL_FEATURE_COLS].values
    # Convert labels from {-1, 1} to {0, 1} for sklearn compatibility
    y = (train_df["label"].values == 1).astype(int)

    # --- ANOVA F-score ---
    # Tests whether the mean of each feature differs between classes.
    # Higher F = stronger mean separation between cloud and no-cloud.
    f_scores, _ = f_classif(X, y)

    # --- Mutual information ---
    # Measures how much knowing a feature reduces uncertainty about the label.
    # Captures non-linear relationships that ANOVA misses.
    # random_state for reproducibility across runs.
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # --- Logistic regression coefficients ---
    # Fit a simple linear classifier and use absolute coefficient magnitudes.
    # Features must be standardized first so coefficients are on comparable scales.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    lr_coefs = np.abs(lr.coef_[0])

    # Combine all three metrics into a single DataFrame for comparison
    importance_df = pd.DataFrame({
        "feature":        ALL_FEATURE_COLS,
        "anova_f":        f_scores,
        "mutual_info":    mi_scores,
        "lr_coefficient": lr_coefs,
    })

    # Normalize each metric to [0, 1] so they can be compared on the same scale
    for col in ["anova_f", "mutual_info", "lr_coefficient"]:
        importance_df[col] = importance_df[col] / importance_df[col].max()

    # Sort by mutual information as primary ranking metric
    importance_df = importance_df.sort_values(
        "mutual_info", ascending=False
    ).reset_index(drop=True)

    print("\n[Task 1] Feature importance scores (normalized to [0, 1]):")
    print(importance_df.round(3).to_string(index=False))

    return importance_df


def plot_feature_importance(importance_df):
    """
    Plot a grouped bar chart showing all three importance metrics
    side by side for each feature.

    Args:
        importance_df: DataFrame from compute_feature_importance().
    """
    print("\n[Task 1] Plotting feature importance...")

    metrics = ["anova_f", "mutual_info", "lr_coefficient"]
    metric_labels = ["ANOVA F-score", "Mutual Information", "Logistic Regression |coef|"]

    n_features = len(importance_df)
    x = np.arange(n_features)
    bar_width = 0.25  # space bars so each feature has a group of 3 bars

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(
            x + i * bar_width,
            importance_df[metric],
            width=bar_width,
            label=label,
            alpha=0.85,
        )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(importance_df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Normalized Importance Score")
    ax.set_title(
        "Feature Importance: ANOVA, Mutual Information, Logistic Regression",
        fontsize=13
    )
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def identify_top_features(importance_df, n=3):
    """
    Identify the top N features by average normalized importance across
    all three metrics.

    Args:
        importance_df: DataFrame from compute_feature_importance().
        n: Number of top features to select (default 3 per lab instructions).

    Returns:
        top_features: List of the top N feature names.
    """
    # Average the three normalized scores to get a single consensus ranking
    importance_df["mean_importance"] = importance_df[
        ["anova_f", "mutual_info", "lr_coefficient"]
    ].mean(axis=1)

    top_features = importance_df.nlargest(n, "mean_importance")["feature"].tolist()

    print(f"\n[Task 1] Top {n} most informative features:")
    for i, feat in enumerate(top_features, 1):
        row = importance_df[importance_df["feature"] == feat].iloc[0]
        print(f"  {i}. {feat} (mean importance: {row['mean_importance']:.3f})")

    return top_features


def plot_top_features_boxplot(train_df, top_features):
    """
    For each of the top features, plot side-by-side boxplots for cloud
    vs. no-cloud to visually confirm class separation.

    Args:
        train_df: Training set DataFrame.
        top_features: List of top feature names from identify_top_features().
    """
    print("\n[Task 1] Plotting boxplots for top features...")

    fig, axes = plt.subplots(1, len(top_features), figsize=(12, 5))
    # Handle edge case where only 1 top feature is selected
    if len(top_features) == 1:
        axes = [axes]

    for ax, feat in zip(axes, top_features):
        data_to_plot = [
            train_df[train_df["label"] == label][feat].values
            for label in [-1, 1]
        ]
        bp = ax.boxplot(
            data_to_plot,
            labels=["No Cloud", "Cloud"],
            patch_artist=True,
            notch=True,     # notched boxplot — notch overlap indicates similar medians
        )
        # Color boxes to match the spatial plot color scheme from eda.py
        colors = ["sandybrown", "steelblue"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{feat}", fontsize=12)
        ax.set_ylabel("Value")

        # Use log scale for SD since it is heavily right-skewed
        if feat == "SD":
            ax.set_yscale("log")

    plt.suptitle("Top Features by Class (Training Set)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "top_features_boxplot.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Task 2: Patch-based feature engineering
# ---------------------------------------------------------------------------

def build_spatial_grid(df):
    """
    Reconstruct a 2D spatial grid from a flat list of (x, y, features) rows.

    Each pixel is placed at its (x, y) coordinate in a 2D array.
    Missing positions (unlabeled pixels that were dropped) are filled
    with NaN so patch extraction still works at boundaries.

    Args:
        df: Full image DataFrame (including unlabeled pixels).

    Returns:
        grid: numpy array of shape (n_channels, height, width).
        x_min: Minimum x value (used to compute relative indices).
        y_min: Minimum y value (used to compute relative indices).
        feature_names: List of column names corresponding to grid channels.
    """
    feature_names = ALL_FEATURE_COLS

    x = df["x"].values.astype(int)
    y = df["y"].values.astype(int)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    width  = x_max - x_min + 1
    height = y_max - y_min + 1

    # Initialize grid with NaN — positions not present in data stay NaN
    grid = np.full((len(feature_names), height, width), np.nan)

    # Place each pixel's feature values at its relative grid position
    x_rel = x - x_min
    y_rel = y - y_min
    for c, feat in enumerate(feature_names):
        grid[c, y_rel, x_rel] = df[feat].values

    return grid, x_min, y_min, feature_names


def extract_patch_features(df, grid, x_min, y_min, feature_names, radius):
    """
    For each labeled pixel, extract summary statistics from a square patch
    of neighboring pixels centered on that pixel.

    Patch features computed for each input feature channel:
        - local_mean:  average value in the patch neighborhood
        - local_std:   standard deviation in the patch neighborhood
        - local_min:   minimum value in the patch neighborhood
        - local_max:   maximum value in the patch neighborhood

    Fully vectorized using scipy.ndimage filters — computes stats across
    the entire grid at once rather than looping over individual pixels.
    This is orders of magnitude faster than a per-pixel loop on ~200k rows.

    NaN positions (missing neighbors) are filled with the channel mean
    before filtering so border pixels are handled gracefully without
    propagating NaNs through the neighborhood.

    Args:
        df: DataFrame of labeled pixels for one split.
        grid: Spatial grid from build_spatial_grid() for the full image.
        x_min: Minimum x coordinate used when building the grid.
        y_min: Minimum y coordinate used when building the grid.
        feature_names: List of feature column names (grid channels).
        radius: Half-width of the patch (e.g. radius=2 gives a 5x5 patch).

    Returns:
        patch_df: DataFrame with one column per patch feature,
                  indexed to match df.
    """
    patch_size = 2 * radius + 1

    # Pixel indices for all labeled pixels in this split
    x_idx = df["x"].values.astype(int) - x_min
    y_idx = df["y"].values.astype(int) - y_min

    result = {}

    for feat in PATCH_FEATURE_SOURCES:
        c = feature_names.index(feat)
        channel = grid[c].copy()    # shape: (height, width)

        # Fill NaN positions with the channel mean before filtering.
        # This prevents NaNs from spreading into patch statistics for
        # pixels that have complete neighborhoods.
        channel_mean = np.nanmean(channel)
        channel[np.isnan(channel)] = channel_mean

        # --- local mean ---
        # uniform_filter computes the sliding window mean over the entire
        # grid simultaneously — no Python loop required.
        # mode="nearest" pads borders by repeating edge values.
        mean_grid = uniform_filter(channel, size=patch_size, mode="nearest")
        result[f"{feat}_local_mean"] = mean_grid[y_idx, x_idx]

        # --- local std ---
        # Var(X) = E[X^2] - E[X]^2, computed via two uniform filters.
        # Much faster than computing std in a sliding window loop.
        mean_sq_grid = uniform_filter(channel ** 2, size=patch_size, mode="nearest")
        var_grid = mean_sq_grid - mean_grid ** 2
        # Clip small negatives caused by floating point errors before sqrt
        std_grid = np.sqrt(np.clip(var_grid, 0, None))
        result[f"{feat}_local_std"] = std_grid[y_idx, x_idx]

        # --- local min ---
        min_grid = minimum_filter(channel, size=patch_size, mode="nearest")
        result[f"{feat}_local_min"] = min_grid[y_idx, x_idx]

        # --- local max ---
        max_grid = maximum_filter(channel, size=patch_size, mode="nearest")
        result[f"{feat}_local_max"] = max_grid[y_idx, x_idx]

    patch_df = pd.DataFrame(result, index=df.index)
    return patch_df


def engineer_patch_features(splits, radius=PATCH_RADIUS):
    """
    For each split (train/val/test), reconstruct the full image spatial grid,
    extract patch features for each labeled pixel, and combine with the
    original features into an enriched DataFrame.

    We load the full image (including unlabeled pixels) to build the grid
    so that patches near labeled/unlabeled boundaries have complete
    neighborhood information.

    Args:
        splits: dict mapping split name -> labeled pixel DataFrame.
        radius: Patch radius for neighborhood window (default PATCH_RADIUS).

    Returns:
        enriched_splits: dict mapping split name -> enriched DataFrame.
    """
    print(f"\n[Task 2] Engineering patch features (radius={radius}, "
          f"window={2*radius+1}x{2*radius+1})...")

    enriched_splits = {}

    for split_name, split_df in splits.items():
        image_name = SPLIT_IMAGE_MAP[split_name]
        print(f"\n  Processing {split_name} ({image_name})...")

        # Load full image including unlabeled pixels for complete spatial grid
        full_df = load_full_image(image_name)
        grid, x_min, y_min, feature_names = build_spatial_grid(full_df)
        print(f"    Grid shape: {grid.shape} (channels x height x width)")

        # Extract patch features — vectorized, should be fast
        patch_df = extract_patch_features(
            split_df, grid, x_min, y_min, feature_names, radius
        )
        print(f"    Patch features shape: {patch_df.shape}")

        # Concatenate original features with new patch features
        enriched_df = pd.concat(
            [split_df.reset_index(drop=True), patch_df.reset_index(drop=True)],
            axis=1
        )
        enriched_splits[split_name] = enriched_df

        # Report the new columns added
        new_cols = patch_df.columns.tolist()
        print(f"    New patch feature columns ({len(new_cols)}): {new_cols}")

    return enriched_splits


def plot_patch_feature_distributions(enriched_splits, top_features):
    """
    For a selection of patch features derived from the top original features,
    plot KDE distributions by class to verify that the patch features
    add discriminative power beyond the original single-pixel features.

    Args:
        enriched_splits: dict from engineer_patch_features().
        top_features: List of top original feature names from Task 1.
    """
    print("\n[Task 2] Plotting patch feature distributions...")

    train_df = enriched_splits["train"]

    # Plot local_mean and local_std for the top features
    patch_cols_to_plot = []
    for feat in top_features:
        for stat in ["local_mean", "local_std"]:
            col = f"{feat}_{stat}"
            if col in train_df.columns:
                patch_cols_to_plot.append(col)

    if not patch_cols_to_plot:
        print("  No patch columns found to plot — skipping.")
        return

    n_cols = len(patch_cols_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, patch_cols_to_plot):
        for label_val, label_name in [(1, "Cloud"), (-1, "No Cloud")]:
            subset = train_df[train_df["label"] == label_val][col].dropna()
            subset.plot.kde(ax=ax, label=label_name)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Value")
        ax.legend(fontsize=7)
        # Apply same axis bounds as eda.py for consistency
        if "SD" in col:
            ax.set_xlim(left=0)
        if "NDAI" in col:
            ax.set_xlim(left=0)

    plt.suptitle(
        "Patch Feature Distributions by Class (Training Set)", fontsize=13, y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "patch_feature_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def save_enriched_splits(enriched_splits):
    """
    Save each enriched split (original + patch features) as a CSV.

    Args:
        enriched_splits: dict from engineer_patch_features().
    """
    print("\n[Task 2] Saving enriched feature sets...")
    for split_name, df in enriched_splits.items():
        save_path = os.path.join(DATA_DIR, f"{split_name}_features.csv")
        df.to_csv(save_path, index=False)
        print(
            f"  {split_name}: {df.shape[1]} columns, "
            f"{df.shape[0]} rows → {save_path}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full feature engineering pipeline."""
    ensure_dirs()

    # Load train/val/test splits produced by eda.py
    print("=" * 60)
    print("Loading splits from eda.py output...")
    print("=" * 60)
    splits = {name: load_split(name) for name in ["train", "val", "test"]}
    train_df = splits["train"]

    # ------------------------------------------------------------------
    # Task 1: Feature importance analysis
    # All importance metrics are computed on the training set only —
    # never val/test — to avoid data leakage into feature selection.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Task 1: Feature importance analysis")
    print("=" * 60)
    importance_df = compute_feature_importance(train_df)
    plot_feature_importance(importance_df)
    top_features = identify_top_features(importance_df, n=3)
    plot_top_features_boxplot(train_df, top_features)

    # ------------------------------------------------------------------
    # Task 2: Patch-based feature engineering
    # For each pixel, compute local neighborhood statistics using a
    # 5x5 window (radius=2). Uses vectorized scipy.ndimage filters
    # for speed — no per-pixel Python loop.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Task 2: Patch-based feature engineering")
    print("=" * 60)
    enriched_splits = engineer_patch_features(splits, radius=PATCH_RADIUS)
    plot_patch_feature_distributions(enriched_splits, top_features)
    save_enriched_splits(enriched_splits)

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print(f"  Top 3 features identified: {top_features}")
    print(f"  Enriched CSVs saved to {DATA_DIR}/")
    print(f"  Figures saved to {FIGS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()