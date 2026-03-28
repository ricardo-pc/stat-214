"""
run_autoencoder.py

To train an autoencoder on image patches.

Usage: python run_autoencoder.py <config.yaml>

What it does:
 - Loads a YAML config describing data, model, optimizer, and trainer options.
 - Finds image files in ../data/image_data and excludes a small set of labeled images.
 - Splits remaining images into train / val sets and constructs LazyPatchDataset objects.
 - Wraps datasets with DataLoader and a PatchCollator (to sample/assemble patches into batches).
 - Builds an Autoencoder, sets up checkpointing and early stopping callbacks, and trains using Lightning.
 - After training, loads the best checkpoint and saves the model state dict to ../results/ae_pretrained.pt
"""

import glob
import sys
import os
import yaml
import shutil
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import Autoencoder
from lazy_patch_dataset import LazyPatchDataset, PatchCollator

def main():
    # ----------------------------
    # 1) Load configuration
    # ----------------------------
    print("loading config file")
    config_path = sys.argv[1]
    assert os.path.exists(config_path), f"Config file {config_path} not found"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Free up any leftover GPU memory from previous runs in this process
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------
    # 2) Locate image files and split into train / val
    # ----------------------------
    print("finding image files")
    # A small set of labeled files (we exclude these from training here)
    labeled_files = {"O013257", "O012791", "O013490"}

    # Find all .npz files in the image_data folder (each file corresponds to an image)
    all_files = sorted(glob.glob("../data/image_data/*.npz"))

    # Keep only files that are not in the labeled set
    unlabeled_files = [
        f for f in all_files if os.path.splitext(os.path.basename(f))[0] not in labeled_files
    ]
    print(f"found {len(unlabeled_files)} unlabeled image files")

    # Simple 80/20 split by file into training and validation images
    n_total = len(unlabeled_files)
    n_train_imgs = int(0.8 * n_total)
    train_files = unlabeled_files[:n_train_imgs]
    val_files = unlabeled_files[n_train_imgs:]

    print(f"train images: {len(train_files)}")
    print(f"val images: {len(val_files)}")

    # ----------------------------
    # 3) Build datasets (lazy-loading patches)
    # ----------------------------
    print("building train dataset")
    # LazyPatchDataset loads image files lazily and samples patches on demand.
    # max_patches is optional and may be used to limit dataset size for quick experiments.
    train_dataset = LazyPatchDataset(
        filepaths=train_files,
        patch_size=config["data"]["patch_size"],
        max_patches=config["data"].get("max_train_patches"),
    )

    print("building val dataset")
    # For validation we reuse the normalization (means/stds) estimated from the training dataset
    val_dataset = LazyPatchDataset(
        filepaths=val_files,
        patch_size=config["data"]["patch_size"],
        means=train_dataset.means,
        stds=train_dataset.stds,
        max_patches=config["data"].get("max_val_patches"),
    )

    print(f"train patches: {len(train_dataset)}")
    print(f"val patches: {len(val_dataset)}")

    # PatchCollator takes the in-memory representations and assembles them into minibatches.
    train_collate = PatchCollator(train_dataset.images, patch_size=config["data"]["patch_size"])
    val_collate = PatchCollator(val_dataset.images, patch_size=config["data"]["patch_size"])

    # ----------------------------
    # 4) DataLoader wrappers
    # ----------------------------
    # Use pin_memory and persistent_workers when num_workers > 0 for performance
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config["dataloader_train"]["batch_size"],
        shuffle=config["dataloader_train"]["shuffle"],
        num_workers=config["dataloader_train"].get("num_workers", 0),
        pin_memory=True,
        persistent_workers=config["dataloader_train"].get("num_workers", 0) > 0,
        collate_fn=train_collate,
    )

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=config["dataloader_val"]["batch_size"],
        shuffle=config["dataloader_val"]["shuffle"],
        num_workers=config["dataloader_val"].get("num_workers", 0),
        pin_memory=True,
        persistent_workers=config["dataloader_val"].get("num_workers", 0) > 0,
        collate_fn=val_collate,
    )

    # ----------------------------
    # 5) Model initialization
    # ----------------------------
    print("initializing model")
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )
    # Print the model architecture for quick verification
    print(model)

    # ----------------------------
    # 6) Lightning Trainer + Callbacks
    # ----------------------------
    print("preparing for training")
    checkpoint_callback = ModelCheckpoint(**config["checkpoint"])  # saves best checkpoints
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["early_stopping"]["patience"],
        mode="min",
    )

    trainer = L.Trainer(
        logger=False,
        callbacks=[checkpoint_callback, early_stopping_callback],
        **config["trainer"]
    )

    # ----------------------------
    # 7) Training
    # ----------------------------
    print("training")
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # Ensure results directory exists and save the best model's weights
    os.makedirs("../results", exist_ok=True)

    best_ckpt_path = checkpoint_callback.best_model_path
    assert best_ckpt_path, "No best checkpoint was saved."
    print(f"best checkpoint: {best_ckpt_path}")

    best_ckpt_out = "../results/ae_pretrained_best.ckpt"
    shutil.copy2(best_ckpt_path, best_ckpt_out)
    print(f"copied best checkpoint to {best_ckpt_out}")

    # Load the best checkpoint into a model instance and save its state_dict
    best_model = Autoencoder.load_from_checkpoint(
        best_ckpt_path,
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    torch.save(best_model.state_dict(), "../results/ae_pretrained.pt")
    print("saved pretrained model to ../results/ae_pretrained.pt")


if __name__ == "__main__":
    main()