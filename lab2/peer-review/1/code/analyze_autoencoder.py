"""
analyze_autoencoder.py
----------------------
Part 2 – Task 3 (Transfer Learning) analysis for Lab 2.

This script assumes the autoencoder has already been trained via:
    python run_autoencoder.py configs/default.yaml

And that embeddings have been extracted via:
    python get_embedding.py configs/default.yaml <checkpoint_path>

What this script produces
--------------------------
1.  Latent dimension selection plot
    Trains small autoencoders with latent_dim in {2,4,8,16,32} on a
    subsample of patches and plots val reconstruction loss vs. dim.
    Justifies the choice of latent_dim=8 for the report.

2.  Reconstruction quality examples
    Side-by-side: original patch | reconstructed patch for 8 random
    pixels from the labeled training image. Shows the autoencoder
    is learning meaningful structure (not just memorising noise).

3.  Latent space separation (PCA + t-SNE)
    Loads the saved embeddings (image1_ae.csv … imageN_ae.csv) for the
    3 labeled images and plots the 8-d latent space projected to 2-d
    with cloud / no-cloud labels coloured. This is the key figure
    showing that the CNN encoder has learned a representation that
    separates the two classes even though it was trained unsupervised.

4.  Per-dimension embedding distributions
    Box-plots of each of the 8 latent dimensions split by class,
    identifying which dimensions carry the most class signal.

Usage (from code/ directory, after training + embedding extraction):
    python analyze_autoencoder.py configs/default.yaml

If embeddings are not yet available, pass --skip-latent to generate
only the latent-dim selection and reconstruction plots (which do not
need pre-computed embeddings).

    python analyze_autoencoder.py configs/default.yaml --skip-latent
"""

import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe on compute clusters
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Paths (all relative to code/ directory)
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
FIGS_DIR = "../figs"

# ---------------------------------------------------------------------------
# Minimal inline autoencoder (mirrors autoencoder.py) so this script is
# self-contained and can be run without importing the Lightning module.
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, in_channels=8, latent_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=0),           nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, latent_dim),
        )
    def forward(self, x): return self.fc(self.conv(x))


class _Decoder(nn.Module):
    def __init__(self, in_channels=8, latent_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, 64 * 7 * 7), nn.LeakyReLU(0.1),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, in_channels, 3, padding=1),
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        return self.conv(x)


class _AE(nn.Module):
    def __init__(self, in_channels=8, latent_dim=8):
        super().__init__()
        self.encoder = _Encoder(in_channels, latent_dim)
        self.decoder = _Decoder(in_channels, latent_dim)
    def forward(self, x): return self.decoder(self.encoder(x))
    def embed(self, x):   return self.encoder(x)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    os.makedirs(FIGS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)


def load_patches_subsample(patch_size=9, n_images=10, seed=42):
    """
    Load a random subsample of patches from the unlabeled images.
    Used for the latent-dim sweep (no need to load all 164 images).
    """
    np.random.seed(seed)
    filepaths = glob.glob(os.path.join(DATA_DIR, "image_data", "*.npz"))
    if not filepaths:
        raise FileNotFoundError(
            f"No .npz files found in {DATA_DIR}/image_data/. "
            "Make sure image_data.zip has been extracted."
        )

    # Use only a subset for speed
    chosen = np.random.choice(filepaths, size=min(n_images, len(filepaths)), replace=False)

    all_patches = []
    pad = patch_size // 2

    for fp in chosen:
        npz = np.load(fp)
        key = list(npz.files)[0]
        data = npz[key]
        if data.shape[1] == 11:
            data = data[:, :-1]

        ys = data[:, 0].astype(int)
        xs = data[:, 1].astype(int)
        feats = data[:, 2:]                  # 8 channels
        n_ch = feats.shape[1]

        y_min, x_min = ys.min(), xs.min()
        h = ys.max() - y_min + 1
        w = xs.max() - x_min + 1

        grid = np.zeros((n_ch, h, w), dtype=np.float32)
        for c in range(n_ch):
            grid[c, ys - y_min, xs - x_min] = feats[:, c]

        # Global-ish normalisation per channel
        for c in range(n_ch):
            mu, sigma = grid[c].mean(), grid[c].std() + 1e-8
            grid[c] = (grid[c] - mu) / sigma

        img_mirror = np.pad(grid, ((0,0),(pad,pad),(pad,pad)), mode="reflect")

        for y, x in zip(ys - y_min, xs - x_min):
            patch = img_mirror[:, y:y+patch_size, x:x+patch_size]
            all_patches.append(patch.astype(np.float32))

    return all_patches


# ---------------------------------------------------------------------------
# Plot 1: Latent dimension selection
# ---------------------------------------------------------------------------

def plot_latent_dim_sweep(patch_size=9, dims=(2, 4, 8, 16, 32),
                           epochs=15, batch_size=256, subsample=30_000):
    """
    Train lightweight autoencoders with varying latent dimensions on a
    patch subsample. Plots val reconstruction MSE vs. latent_dim to
    justify the choice of 8.
    """
    print("\n[Analysis 1] Latent dimension sweep …")

    patches = load_patches_subsample(patch_size=patch_size)
    # subsample for speed
    idx = np.random.choice(len(patches), size=min(subsample, len(patches)), replace=False)
    patches = [patches[i] for i in idx]

    tensors = torch.tensor(np.stack(patches))         # (N, C, H, W)
    n_val   = int(0.15 * len(tensors))
    n_train = len(tensors) - n_val
    train_ds, val_ds = random_split(tensors, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for dim in dims:
        print(f"  latent_dim={dim} …", end=" ", flush=True)
        model = _AE(in_channels=patch_size > 0 and tensors.shape[1], latent_dim=dim).to(device)
        # fix: get actual channel count
        model = _AE(in_channels=tensors.shape[1], latent_dim=dim).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

        tr_loader  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train()
            for batch in tr_loader:
                x = batch.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), x)
                loss.backward()
                opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            tr_loss = np.mean([
                loss_fn(model(b.to(device)), b.to(device)).item()
                for b in tr_loader
            ])
            v_loss = np.mean([
                loss_fn(model(b.to(device)), b.to(device)).item()
                for b in val_loader
            ])
        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        print(f"val_MSE={v_loss:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dims, train_losses, "o-", label="Train MSE", color="steelblue")
    ax.plot(dims, val_losses,   "s--", label="Val MSE",   color="tomato")
    ax.axvline(8, color="green", linestyle=":", linewidth=1.5, label="Chosen dim=8")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Autoencoder Reconstruction Loss vs. Latent Dimension")
    ax.set_xticks(dims)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "ae_latent_dim_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Reconstruction quality examples
# ---------------------------------------------------------------------------

def plot_reconstruction_examples(checkpoint_path, config, n_examples=8):
    """
    Load the trained model, run a few labeled-image patches through it,
    and show original vs. reconstruction side by side.
    Visualises a single channel (AN, index 7) which is the most
    informative radiance angle.
    """
    print("\n[Analysis 2] Reconstruction quality …")

    from autoencoder import Autoencoder   # Lightning module

    patch_size = config["data"]["patch_size"]
    model = Autoencoder(patch_size=patch_size, **config["autoencoder"])
    map_loc = None if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=map_loc)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load a handful of patches from the training image
    patches = load_patches_subsample(patch_size=patch_size, n_images=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(patches), size=n_examples, replace=False)
    sample = torch.tensor(np.stack([patches[i] for i in idx]))   # (N,C,H,W)

    with torch.no_grad():
        recon = model(sample).numpy()
    sample = sample.numpy()

    channel = 7   # AN channel (most informative)
    fig, axes = plt.subplots(2, n_examples, figsize=(2 * n_examples, 5))
    for j in range(n_examples):
        vmin = sample[j, channel].min()
        vmax = sample[j, channel].max()
        axes[0, j].imshow(sample[j, channel], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, j].imshow(recon[j,  channel], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, j].axis("off")
        axes[1, j].axis("off")
        if j == 0:
            axes[0, j].set_ylabel("Original",      fontsize=9)
            axes[1, j].set_ylabel("Reconstructed", fontsize=9)

    fig.suptitle("CNN Autoencoder – Original vs. Reconstructed Patches (AN channel)",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "ae_reconstruction_examples.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3 + 4: Latent space analysis using saved embeddings
# ---------------------------------------------------------------------------

def load_labeled_embeddings():
    """
    Load embedding CSVs produced by get_embedding.py and join with
    expert labels from the original .npz files for the 3 labeled images.
    Returns a single DataFrame with columns ae0..ae7 + label.
    """
    LABELED = {
        "O013257": os.path.join(DATA_DIR, "image_data", "O013257.npz"),
        "O012791": os.path.join(DATA_DIR, "image_data", "O012791.npz"),
        "O013490": os.path.join(DATA_DIR, "image_data", "O013490.npz"),
    }

    # Map image name -> embedding CSV index
    # get_embedding.py saves image1_ae.csv, image2_ae.csv, … in order of glob
    ae_files = sorted(glob.glob(os.path.join(DATA_DIR, "ae_embeddings", "*_ae.csv")))
    if not ae_files:
        raise FileNotFoundError(
            "No image*_ae.csv files found. Run get_embedding.py first."
        )

    # Build a lookup: (y,x) -> label for each labeled image
    all_frames = []
    for ae_path in ae_files:
        ae_df = pd.read_csv(ae_path)
        ae_df["y"] = ae_df["y"].astype(int)
        ae_df["x"] = ae_df["x"].astype(int)

        # Try to match with a labeled image
        for name, npz_path in LABELED.items():
            npz = np.load(npz_path)
            key = list(npz.files)[0]
            data = npz[key]
            if data.shape[1] != 11:
                continue
            label_df = pd.DataFrame(
                data[:, [0, 1, 10]].astype(int),
                columns=["y", "x", "label"]
            )
            merged = ae_df.merge(label_df, on=["y", "x"], how="inner")
            labeled = merged[merged["label"] != 0]
            if len(labeled) > 100:
                all_frames.append(labeled)
                break   # found the matching image

    if not all_frames:
        raise RuntimeError(
            "Could not match any embedding CSV with a labeled image. "
            "Check that get_embedding.py processed the labeled images."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  Loaded {len(combined)} labeled embedded pixels "
          f"({(combined['label']==1).sum()} cloud, "
          f"{(combined['label']==-1).sum()} no-cloud)")
    return combined


def plot_latent_space(combined_df, max_points=8000):
    """
    PCA (fast, interpretable) + t-SNE (non-linear) projections of the
    latent space, coloured by cloud / no-cloud label.
    """
    print("\n[Analysis 3] Latent space visualisation …")

    ae_cols = [c for c in combined_df.columns if c.startswith("ae")]
    X = combined_df[ae_cols].values
    y = combined_df["label"].values       # +1 cloud, -1 no-cloud

    # Subsample for t-SNE speed
    rng = np.random.default_rng(42)
    if len(X) > max_points:
        idx = rng.choice(len(X), size=max_points, replace=False)
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y

    colors = {1: "steelblue", -1: "tomato"}
    labels = {1: "Cloud", -1: "No Cloud"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(X_s)
    var_exp = pca.explained_variance_ratio_ * 100
    for cls in [1, -1]:
        mask = y_s == cls
        axes[0].scatter(Z_pca[mask, 0], Z_pca[mask, 1],
                        c=colors[cls], label=labels[cls],
                        s=4, alpha=0.35, rasterized=True)
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    axes[0].set_title("PCA of CNN Autoencoder Latent Space")
    axes[0].legend(markerscale=3)

    # --- t-SNE ---
    print("  Running t-SNE (this may take ~1 min) …", flush=True)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1)
    Z_tsne = tsne.fit_transform(X_s)
    for cls in [1, -1]:
        mask = y_s == cls
        axes[1].scatter(Z_tsne[mask, 0], Z_tsne[mask, 1],
                        c=colors[cls], label=labels[cls],
                        s=4, alpha=0.35, rasterized=True)
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")
    axes[1].set_title("t-SNE of CNN Autoencoder Latent Space")
    axes[1].legend(markerscale=3)

    plt.suptitle("CNN Autoencoder Latent Space — Cloud vs. No-Cloud Separation",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "ae_latent_space.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_embedding_distributions(combined_df):
    """
    Box-plots for each of the 8 latent dimensions, split by class.
    Reveals which dimensions carry the most discriminative signal.
    """
    print("\n[Analysis 4] Per-dimension embedding distributions …")

    ae_cols = sorted([c for c in combined_df.columns if c.startswith("ae")])
    n = len(ae_cols)

    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6), sharey=False)
    axes = axes.flatten()

    for i, col in enumerate(ae_cols):
        cloud    = combined_df[combined_df["label"] ==  1][col].values
        no_cloud = combined_df[combined_df["label"] == -1][col].values
        bp = axes[i].boxplot(
            [no_cloud, cloud],
            labels=["No Cloud", "Cloud"],
            patch_artist=True,
            notch=False,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], ["tomato", "steelblue"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axes[i].set_title(f"Latent dim {i}", fontsize=10)
        axes[i].tick_params(axis="x", labelsize=8)

    plt.suptitle("Latent Embedding Distributions by Class", fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, "ae_embedding_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config (e.g. configs/default.yaml)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained .ckpt file for reconstruction plot. "
                             "If omitted, reconstruction plot is skipped.")
    parser.add_argument("--skip-latent", action="store_true",
                        help="Skip latent-space plots (use if embeddings not yet generated).")
    args = parser.parse_args()

    ensure_dirs()

    config = yaml.safe_load(open(args.config))
    patch_size = config["data"]["patch_size"]

    # --- Plot 1: latent dim sweep (always run) ---
    plot_latent_dim_sweep(patch_size=patch_size)

    # --- Plot 2: reconstruction examples (needs checkpoint) ---
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"  WARNING: checkpoint not found at {args.checkpoint}, skipping.")
        else:
            plot_reconstruction_examples(args.checkpoint, config)
    else:
        print("\n[Analysis 2] No --checkpoint provided, skipping reconstruction plot.")
        print("  Run again with: --checkpoint ../results/checkpoints/default-last.ckpt")

    # --- Plots 3 + 4: latent space (needs embedding CSVs) ---
    if not args.skip_latent:
        try:
            combined_df = load_labeled_embeddings()
            plot_latent_space(combined_df)
            plot_embedding_distributions(combined_df)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\n[Analysis 3/4] Skipping latent space plots: {e}")
            print("  Run get_embedding.py first, then re-run this script.")
    else:
        print("\n[Analysis 3/4] --skip-latent set, skipping latent space plots.")

    print("\n" + "=" * 55)
    print("Autoencoder analysis complete.")
    print(f"All figures saved to {FIGS_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
