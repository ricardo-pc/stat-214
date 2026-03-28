"""
eda.py
------
Exploratory Data Analysis for Lab 2: Cloud Detection in Polar Regions.

This script:
    1. Loads the 3 labeled MISR images.
    2. Cleans the data:
        - Drops NaN rows.
        - Checks for physically impossible values (negative radiances,
          CORR outside [-1, 1], negative NDAI/SD).
        - Checks for duplicate rows.
        - Clips extreme outliers (beyond 5 std devs) and visualizes them spatially.
        - Checks cross-image feature ranges for comparability.
    3. Plots expert cloud labels spatially (X, Y coordinates), including
       a separate plot showing where unlabeled pixels are.
    4. Explores feature relationships (radiances, NDAI, SD, CORR) by class.
    5. Splits data into train, validation, and test sets by image.
    6. Saves splits as CSVs for use in downstream modeling.

Usage:
    Run from the code/ directory:
        python eda.py

Outputs:
    - ../figs/          all exploratory and explanatory figures
    - ../data/train.csv labeled pixels from O013257 (training set)
    - ../data/val.csv   labeled pixels from O012791 (validation set)
    - ../data/test.csv  labeled pixels from O013490 (test set)

FOR MY TEAMMATES:
    The train/val/test CSVs saved at the end of this script are the
    canonical data splits for the whole project. Please use these files
    rather than re-splitting the data yourself, to ensure consistency
    across all parts of the pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names match the order described in the lab instructions.
# Columns 0-1: spatial coordinates; 2-4: engineered features;
# 5-9: radiance at 5 viewing angles; 10: expert label (+1=cloud, -1=no cloud, 0=unlabeled)
COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]

# Radiance angles from the MISR sensor (DF=downward-forward, AN=nadir, etc.)
# These capture how brightness changes with viewing angle — key for cloud detection
# because clouds are at a different altitude than the polar surface.
RADIANCE_COLS = ["DF", "CF", "BF", "AF", "AN"]

# Expert-engineered features derived from the radiance data (see yu2008.pdf)
FEATURE_COLS = ["NDAI", "SD", "CORR"]

# Colors and names used consistently across all spatial scatter plots
LABEL_COLORS = {1: "steelblue", -1: "sandybrown", 0: "lightgray"}
LABEL_NAMES = {1: "Cloud", -1: "No Cloud", 0: "Unlabeled"}

# All paths are relative to the code/ directory (i.e. run this script from code/)
DATA_DIR = "../data/image_data"
FIGS_DIR = "../figs"
OUTPUT_DATA_DIR = "../data"

# Only these 3 images have expert labels — all other images are unlabeled
# and are used by the autoencoder for transfer learning (see autoencoder.py)
LABELED_IMAGES = {
    "O013257": os.path.join(DATA_DIR, "O013257.npz"),
    "O013490": os.path.join(DATA_DIR, "O013490.npz"),
    "O012791": os.path.join(DATA_DIR, "O012791.npz"),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_labeled_image(filepath):
    """
    Load a labeled .npz image file and return it as a pandas DataFrame.

    Args:
        filepath: Path to the .npz file.

    Returns:
        df: DataFrame with columns for coordinates, features, radiances, and label.
    """
    npz = np.load(filepath)
    # Each .npz file stores data under a single key — we extract it dynamically
    # rather than hardcoding the key name, since it varies across files
    key = list(npz.files)[0]
    data = npz[key]
    # Assign human-readable column names (order matches lab instructions)
    df = pd.DataFrame(data, columns=COLUMNS)
    return df


def get_labeled_pixels(df):
    """
    Filter out unlabeled pixels (label == 0).

    Args:
        df: DataFrame with a 'label' column.

    Returns:
        DataFrame containing only rows where label is +1 or -1.
    """
    # label == 0 means the expert chose not to label that pixel (typically
    # at ambiguous cloud boundaries). We exclude these from supervised analyses
    # but keep them in spatial plots for context.
    return df[df["label"] != 0].copy()


def ensure_dirs():
    """Create output directories if they do not already exist."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_images():
    """
    Load all 3 labeled images, report basic info, and return as a dict.

    Returns:
        images: dict mapping image name -> raw DataFrame (with unlabeled pixels).
    """
    images = {}
    for name, path in LABELED_IMAGES.items():
        df = load_labeled_image(path)
        print(f"\n--- {name} ---")
        print(f"  Shape: {df.shape}")
        # Check class balance — important for understanding the difficulty of the task
        # and for justifying our train/val/test split choices
        print(f"  Label counts:\n{df['label'].value_counts().to_string()}")
        print(f"  Missing values:\n{df.isnull().sum().to_string()}")
        images[name] = df
    return images


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def check_impossible_values(df, name):
    """
    Check for physically impossible values based on domain knowledge:
        - Radiance values should be non-negative.
        - CORR should be in [-1, 1].
        - NDAI and SD should be non-negative.

    Prints a summary of any violations found. Does not modify the DataFrame.

    Args:
        df: DataFrame for one image.
        name: Image name string (used for logging).
    """
    print(f"\n  [{name}] Checking for physically impossible values...")

    # Radiance is a measure of light intensity — physically must be >= 0
    for col in RADIANCE_COLS:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            print(f"    WARNING: {n_neg} negative values in radiance column '{col}'.")
        else:
            print(f"    '{col}': all values non-negative. OK.")

    # CORR is a correlation coefficient — must be in [-1, 1] by definition
    n_corr = ((df["CORR"] < -1) | (df["CORR"] > 1)).sum()
    if n_corr > 0:
        print(f"    WARNING: {n_corr} values of CORR outside [-1, 1].")
    else:
        print(f"    'CORR': all values in [-1, 1]. OK.")

    # NDAI and SD are non-negative by construction (see yu2008.pdf for definitions)
    for col in ["NDAI", "SD"]:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            print(f"    WARNING: {n_neg} negative values in '{col}'.")
        else:
            print(f"    '{col}': all values non-negative. OK.")


def check_duplicates(df, name):
    """
    Check for duplicate rows in a DataFrame and report the count.

    Args:
        df: DataFrame for one image.
        name: Image name string (used for logging).
    """
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"  [{name}] WARNING: {n_dupes} duplicate rows found.")
    else:
        print(f"  [{name}] No duplicate rows found. OK.")


def check_cross_image_ranges(images):
    """
    Print min/max/mean/std for each feature and radiance column across all 3 images
    to check whether feature scales are comparable across train/val/test.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    # This check is important because our model trains on one image and predicts
    # on another. If feature scales differ dramatically, the model may struggle
    # to generalize. The output of this function should be reviewed carefully.
    print("\n[Cleaning] Cross-image feature range comparison:")
    check_cols = FEATURE_COLS + RADIANCE_COLS
    for col in check_cols:
        print(f"\n  {col}:")
        for name, df in images.items():
            col_data = df[col]
            print(
                f"    {name}: min={col_data.min():.3f}, "
                f"max={col_data.max():.3f}, "
                f"mean={col_data.mean():.3f}, "
                f"std={col_data.std():.3f}"
            )


def clean_image(df, name):
    """
    Clean a single image DataFrame by:
        - Dropping rows with NaN values.
        - Checking for physically impossible values (domain constraints).
        - Clipping negative NDAI values to zero.
        - Checking for duplicate rows.
        - Clipping extreme outliers (beyond 5 std devs).

    Args:
        df: Raw DataFrame for one image.
        name: Image name string (used for logging).

    Returns:
        df_clean: Cleaned DataFrame.
    """
    original_len = len(df)

    # 1. Drop NaN rows — real sensor data occasionally has missing readings
    df_clean = df.dropna()
    dropped_na = original_len - len(df_clean)
    if dropped_na > 0:
        print(f"  [{name}] Dropped {dropped_na} rows with NaN values.")
    else:
        print(f"  [{name}] No NaN rows found. OK.")

    # 2. Check for physically impossible values (does not modify data)
    # This is a read-only diagnostic — actual fixes happen in steps 3 and 5
    check_impossible_values(df_clean, name)

    # 3. Clip negative NDAI values to zero (physically impossible per domain knowledge)
    # We found 90, 23, and 961 negative NDAI values across the 3 images.
    # These are likely numerical artifacts. Clipping to 0 rather than dropping
    # to preserve sample size without introducing bias.
    n_neg_ndai = (df_clean["NDAI"] < 0).sum()
    if n_neg_ndai > 0:
        print(f"  [{name}] Clipping {n_neg_ndai} negative NDAI values to 0.")
        df_clean["NDAI"] = df_clean["NDAI"].clip(lower=0)

    # 4. Check for duplicate rows — would inflate training data if present
    check_duplicates(df_clean, name)

    # 5. Clip extreme outliers beyond 5 standard deviations
    # We use 5 std (rather than the more common 3) to be conservative —
    # satellite imagery can have legitimate high-variance regions.
    # Notable: O013490 had 631 and 662 outliers in DF and CF respectively,
    # likely corresponding to sensor artifacts in those viewing angles.
    check_cols = RADIANCE_COLS + FEATURE_COLS
    for col in check_cols:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        outlier_mask = z_scores > 5
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  [{name}] {n_outliers} extreme outliers in '{col}' (>5 std). Clipping.")
            lower = df_clean[col].mean() - 5 * df_clean[col].std()
            upper = df_clean[col].mean() + 5 * df_clean[col].std()
            df_clean[col] = df_clean[col].clip(lower, upper)

    return df_clean


# ---------------------------------------------------------------------------
# Task 1: Spatial plots of expert labels
# ---------------------------------------------------------------------------

def plot_expert_labels(images):
    """
    For each labeled image, plot expert cloud labels on the X, Y spatial grid.
    Points are colored by label: cloud (blue), no cloud (orange), unlabeled (gray).

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 1] Plotting expert labels spatially...")
    for name, df in images.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot each label class separately so we can assign colors and legend entries
        for label_val, group in df.groupby("label"):
            ax.scatter(
                group["x"],
                group["y"],
                c=LABEL_COLORS[label_val],
                s=0.2,              # small point size since we have ~115k pixels per image
                label=LABEL_NAMES[label_val],
                alpha=0.6,
                rasterized=True,    # rasterize for smaller file size with dense scatter plots
            )
        ax.set_title(f"Expert Cloud Labels — Image {name}", fontsize=13)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend(markerscale=6, loc="upper right")
        plt.tight_layout()
        save_path = os.path.join(FIGS_DIR, f"{name}_labels.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")


def plot_unlabeled_pixels(images):
    """
    For each image, plot only the unlabeled pixels (label == 0) spatially
    to understand whether unlabeled regions are random or spatially clustered
    (e.g. at image borders or in specific regions).

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 1] Plotting unlabeled pixel locations...")
    for name, df in images.items():
        unlabeled = df[df["label"] == 0]
        labeled = df[df["label"] != 0]

        fig, ax = plt.subplots(figsize=(8, 6))
        # Plot labeled pixels as background context in gray
        ax.scatter(
            labeled["x"], labeled["y"],
            c="lightgray", s=0.1, label="Labeled", rasterized=True
        )
        # Overlay unlabeled pixels in red to reveal their spatial pattern.
        # Key finding: unlabeled pixels cluster at cloud boundaries, not randomly —
        # suggesting the expert deliberately left ambiguous border regions unlabeled.
        ax.scatter(
            unlabeled["x"], unlabeled["y"],
            c="red", s=0.1, label="Unlabeled", alpha=0.5, rasterized=True
        )
        pct_unlabeled = 100 * len(unlabeled) / len(df)
        ax.set_title(
            f"Unlabeled Pixel Locations — {name} ({pct_unlabeled:.1f}% unlabeled)",
            fontsize=12
        )
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend(markerscale=6)
        plt.tight_layout()
        save_path = os.path.join(FIGS_DIR, f"{name}_unlabeled.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Task 2: Feature relationship exploration
# ---------------------------------------------------------------------------

def plot_radiance_distributions(images):
    """
    For each radiance angle, plot kernel density estimates by cloud/no-cloud class,
    combining data from all 3 labeled images.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 2] Plotting radiance distributions by class...")

    # Combine labeled pixels from all 3 images for a more robust distribution estimate
    combined = pd.concat(
        [get_labeled_pixels(df) for df in images.values()], ignore_index=True
    )

    fig, axes = plt.subplots(1, len(RADIANCE_COLS), figsize=(18, 4), sharey=False)
    for ax, col in zip(axes, RADIANCE_COLS):
        for label_val, label_name in [(1, "Cloud"), (-1, "No Cloud")]:
            subset = combined[combined["label"] == label_val][col]
            subset.plot.kde(ax=ax, label=label_name)
        ax.set_title(f"Radiance: {col}")
        ax.set_xlabel("Radiance Value")
        ax.legend()
    plt.suptitle("Radiance Distributions by Class (All 3 Images)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "radiance_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_distributions(images):
    """
    For NDAI, SD, and CORR, plot distributions by cloud/no-cloud class.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 2] Plotting NDAI/SD/CORR distributions by class...")

    combined = pd.concat(
        [get_labeled_pixels(df) for df in images.values()], ignore_index=True
    )

    fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(12, 4))
    for ax, col in zip(axes, FEATURE_COLS):
        for label_val, label_name in [(1, "Cloud"), (-1, "No Cloud")]:
            subset = combined[combined["label"] == label_val][col]
            subset.plot.kde(ax=ax, label=label_name)
        ax.set_title(f"Feature: {col}")
        ax.set_xlabel("Value")
        ax.legend()
        # Clip x-axis to physically meaningful ranges to avoid KDE smoothing artifacts:
        # NDAI and SD are bounded at 0; CORR is bounded at [-1, 1]
        if col in ["SD", "NDAI"]:
            ax.set_xlim(left=0)
        if col == "SD":
            ax.set_xscale("log")    # SD is heavily right-skewed
        if col == "CORR":
            ax.set_xlim(-1, 1)
    plt.suptitle("Engineered Feature Distributions by Class", fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "feature_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_matrix(images):
    """
    Plot a correlation matrix of all features and radiance columns
    using labeled pixels from all 3 images combined.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 2] Plotting correlation matrix...")

    combined = pd.concat(
        [get_labeled_pixels(df) for df in images.values()], ignore_index=True
    )
    corr_cols = FEATURE_COLS + RADIANCE_COLS
    corr_matrix = combined[corr_cols].corr()

    print("  Correlation matrix:")
    print(corr_matrix.round(3).to_string())

    # Key findings to note in the report:
    # - AF and AN are correlated at 0.99 — nearly redundant features
    # - BF/AF/AN form a tight cluster (0.94-0.99) — near-nadir angles are very similar
    # - DF is least correlated with the others (0.51-0.69) — most distinct viewing angle
    # - NDAI is strongly negatively correlated with AF (-0.70) and AN (-0.75)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
    )
    ax.set_title("Feature Correlation Matrix (Labeled Pixels)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(FIGS_DIR, "correlation_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pairwise_scatterplots(images):
    """
    Plot pairwise scatterplots for NDAI, SD, CORR colored by cloud label,
    using a random sample of labeled pixels for readability.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 2] Plotting pairwise scatterplots...")

    combined = pd.concat(
        [get_labeled_pixels(df) for df in images.values()], ignore_index=True
    )

    # Sample for readability — pairplot with all ~200k pixels is very slow.
    # random_state=42 ensures reproducibility across runs.
    sample = combined.sample(n=min(5000, len(combined)), random_state=42)
    sample["label_name"] = sample["label"].map(LABEL_NAMES)

    g = sns.pairplot(
        sample[FEATURE_COLS + ["label_name"]],
        hue="label_name",
        plot_kws={"alpha": 0.3, "s": 10},
        diag_kind="kde",    # KDE on diagonal gives cleaner view of each feature's distribution
    )
    g.figure.suptitle("Pairwise Feature Scatterplots by Class", y=1.02, fontsize=13)
    save_path = os.path.join(FIGS_DIR, "pairwise_scatterplots.png")
    g.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_class_statistics(images):
    """
    Print mean and standard deviation of each feature and radiance column,
    broken down by cloud vs. no-cloud class.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 2] Class-wise summary statistics:")
    combined = pd.concat(
        [get_labeled_pixels(df) for df in images.values()], ignore_index=True
    )
    # Key findings from this output:
    # - Cloud pixels have much higher SD (mean=723) vs no-cloud (mean=163)
    # - Cloud pixels have higher NDAI (mean=0.264) vs no-cloud (mean=0.143)
    # - CORR shows the weakest class separation (means of 0.413 vs 0.367)
    # These findings directly inform Part 2 feature importance analysis
    summary = combined.groupby("label")[FEATURE_COLS + RADIANCE_COLS].agg(["mean", "std"])
    print(summary.round(3).to_string())


# ---------------------------------------------------------------------------
# Task 3: Train / validation / test split
# ---------------------------------------------------------------------------

def split_and_save(images):
    """
    Split the 3 labeled images into train, validation, and test sets
    at the IMAGE level (not pixel level) to avoid spatial data leakage.

    Assignment:
        O013257 -> train:  most labeled pixels (70,826), most to learn from
        O012791 -> val:    fewest labeled pixels, suitable for hyperparameter tuning
        O013490 -> test:   most balanced class distribution (~50/50), most honest eval

    Only labeled pixels (label != 0) are included in the splits.
    Saves each split as a CSV for downstream use.

    Args:
        images: dict mapping image name -> DataFrame.
    """
    print("\n[Task 3] Splitting data by image...")

    # Split assignment is deliberate based on class balance and labeled pixel counts.
    # Splitting at the image level (not pixel level) avoids spatial data leakage —
    # pixels within the same image are spatially correlated, so a pixel-level
    # random split would inflate performance on test data by leaking spatial
    # information from training pixels to nearby test pixels.
    #
    # FOR TEAMMATES: Please use these CSVs as your data source for Parts 2 and 3.
    # Do not re-split the data yourself — consistent splits are essential for
    # comparing model performance across the team.
    splits = {
        "train": "O013257",   # 70,826 labeled pixels, most training data
        "val":   "O012791",   # 54,772 labeled pixels, used for hyperparameter tuning
        "test":  "O013490",   # 82,083 labeled pixels, ~50/50 class balance — most honest eval
    }

    for split_name, img_name in splits.items():
        df = get_labeled_pixels(images[img_name])
        save_path = os.path.join(OUTPUT_DATA_DIR, f"{split_name}.csv")
        df.to_csv(save_path, index=False)
        print(f"  {split_name}: {img_name} — {len(df)} labeled pixels → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full EDA pipeline."""
    ensure_dirs()

    # Load all 3 labeled images
    print("=" * 60)
    print("Loading images...")
    print("=" * 60)
    images = load_all_images()

    # Clean each image — all cleaning decisions are documented in clean_image()
    print("\n" + "=" * 60)
    print("Cleaning images...")
    print("=" * 60)
    images = {name: clean_image(df, name) for name, df in images.items()}

    # Check that feature scales are comparable across images
    # (important since we train on one image and predict on another)
    check_cross_image_ranges(images)

    # Task 1: Spatial label plots
    print("\n" + "=" * 60)
    print("Task 1: Spatial label plots")
    print("=" * 60)
    plot_expert_labels(images)
    plot_unlabeled_pixels(images)  # reveals that unlabeled pixels cluster at cloud boundaries

    # Task 2: Feature relationships
    # Key finding: NDAI and SD are the strongest discriminators between cloud/no-cloud.
    # CORR is the weakest. AF and AN are nearly redundant (corr=0.99).
    print("\n" + "=" * 60)
    print("Task 2: Feature relationships")
    print("=" * 60)
    plot_radiance_distributions(images)
    plot_feature_distributions(images)
    plot_correlation_matrix(images)
    plot_pairwise_scatterplots(images)
    print_class_statistics(images)

    # Task 3: Train/val/test split
    # Splits by image (not pixel) to avoid spatial data leakage.
    # Outputs train.csv, val.csv, test.csv to ../data/ for use by all teammates.
    print("\n" + "=" * 60)
    print("Task 3: Train/val/test split")
    print("=" * 60)
    split_and_save(images)

    print("\n" + "=" * 60)
    print("EDA complete! Figures saved to ../figs/")
    print("=" * 60)


if __name__ == "__main__":
    main()