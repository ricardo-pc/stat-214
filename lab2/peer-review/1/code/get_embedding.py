"""
get_embedding.py

Extracts autoencoder latent embeddings for every patch in each image and writes
them to CSV files under ../data/ae_embeddings.

Usage: python get_embedding.py <config.yaml> <checkpoint>

High level steps:
 - Load YAML config and a pretrained autoencoder checkpoint.
 - For each .npz image in ../data/image_data:
     - Build a LazyPatchDataset for that image and a PatchCollator to assemble minibatches.
     - Iterate over patch coordinates in batches, compute embeddings with model.embed().
     - Save a CSV with columns [y, x, ae0, ae1, ..., aeN] describing each patch's location
       and its latent vector components.

Notes:
 - The script tries to be memory-conscious by batching embeddings and using torch.no_grad().
 - It accepts either a raw state_dict checkpoint or a dict containing a "state_dict" key.
"""

import os
import sys
import glob
import yaml
import torch
import pandas as pd
from tqdm import tqdm

from autoencoder import Autoencoder
from lazy_patch_dataset import LazyPatchDataset, PatchCollator

def extract_image_id(filepath):
    """Return the image id (filename without extension) for a given path.

    e.g. ../data/image_data/O012791.npz -> 'O012791'
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def main():
    # Expect config yaml and a checkpoint path as positional args
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    assert os.path.exists(config_path), f"Config file {config_path} not found"
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found"

    # Load configuration (data shapes, patch size, model hyperparams, etc.)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint from {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model instance (same constructor used for training)
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    # Load checkpoint. Support both raw state_dicts and wrapped checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # set to eval mode (disables dropout/batch-norm updates)

    # Discover all image files we want to process
    print("Finding image files")
    all_files = sorted(glob.glob("../data/image_data/*.npz"))
    print(f"Found {len(all_files)} image files")

    # Ensure output directory exists
    os.makedirs("../data/ae_embeddings", exist_ok=True)

    # Number of patches to process per inference batch. Chosen to be large but not too large
    # (tuneable depending on GPU memory). We batch over patch coordinates, not images.
    batch_size = 4096

    print("Extracting embeddings")
    # Use the fine-tune training image to define a fixed normalization rule
    # for all embedding extraction, so preprocessing matches the final model.
    reference_ds = LazyPatchDataset(
        filepaths=["../data/image_data/O013257.npz"],
        patch_size=config["data"]["patch_size"],
    )
    fixed_means = reference_ds.means
    fixed_stds = reference_ds.stds
    with torch.no_grad():
        # Iterate images one-by-one to avoid building a dataset for all images at once
        for file_path in tqdm(all_files):
            img_id = extract_image_id(file_path)

            # Build a LazyPatchDataset for the single image. This object knows how to sample
            # patches and has fields like `images` and `images_long` used below.
            ds = LazyPatchDataset(
                filepaths=[file_path],
                patch_size=config["data"]["patch_size"],
                means=fixed_means,
                stds=fixed_stds,
            )

            # Collator turns a list of patch coordinate entries into a tensor batch of patches
            collator = PatchCollator(
                ds.images,
                patch_size=config["data"]["patch_size"],
            )

            # Collect all patch coordinate entries for this image (ds[i] returns a coordinate entry)
            coords = [ds[i] for i in range(len(ds))]

            embeddings_list = []

            # Process coordinates in batches to avoid blowing up memory
            for start in range(0, len(coords), batch_size):
                batch_coords = coords[start:start + batch_size]
                patches = collator(batch_coords).to(device)  # shape: [B, C, H, W]
                emb = model.embed(patches).detach().cpu()   # [B, latent_dim]
                embeddings_list.append(emb)

            # Concatenate all batches for the image and convert to numpy
            emb_all = torch.cat(embeddings_list, dim=0).numpy()

            # ds.images_long contains (y, x) coordinates for each patch; extract them for CSV
            img_long = ds.images_long[0]
            ys = img_long[:, 0].astype(int)
            xs = img_long[:, 1].astype(int)

            # Create a DataFrame with columns [y, x, ae0, ae1, ...]
            latent_dim = emb_all.shape[1]
            embedding_df = pd.DataFrame(
                emb_all,
                columns=[f"ae{j}" for j in range(latent_dim)]
            )
            embedding_df["y"] = ys
            embedding_df["x"] = xs

            # Reorder columns to put coordinates first
            cols = ["y", "x"] + [f"ae{j}" for j in range(latent_dim)]
            embedding_df = embedding_df[cols]

            # Write CSV for downstream analysis/visualization
            out_path = f"../data/ae_embeddings/{img_id}_ae.csv"
            embedding_df.to_csv(out_path, index=False)

    print("Done")


if __name__ == "__main__":
    main()