"""
fine_tune_autoencoder.py

Script to fine-tune a pretrained autoencoder on one or two held-out images.

Usage: python fine_tune_autoencoder.py <config.yaml> <pretrained_checkpoint>

What it does:
 - Loads training configuration from YAML.
 - Loads a pretrained checkpoint (either a raw state_dict or a dict containing "state_dict").
 - Builds small LazyPatchDataset objects for one image used for fine-tuning and another for validation.
 - Fine-tunes the model with Lightning using checkpointing and early stopping.
 - Saves the best fine-tuned state_dict to ../results/ae_finetuned.pt

"""

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
    # 1) CLI args and config
    # ----------------------------
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    assert os.path.exists(config_path), f"Config file {config_path} not found"
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Free any leftover memory (helpful when running repeatedly in the same process)
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------
    # 2) Select images for fine-tuning
    # ----------------------------
    # We exclude O013490 here (it's the final held-out test image).
    # Fine-tuning will be performed on O013257 and validated on O012791.
    train_ids = ["O013257"]
    val_ids = ["O012791"]

    train_files = [f"../data/image_data/{img_id}.npz" for img_id in train_ids]
    val_files = [f"../data/image_data/{img_id}.npz" for img_id in val_ids]

    print("Fine-tune train files:")
    for f in train_files:
        print(f)

    print("Fine-tune validation files:")
    for f in val_files:
        print(f)

    # ----------------------------
    # 3) Build LazyPatchDataset objects
    # ----------------------------
    # LazyPatchDataset samples patches from the provided image files on demand. Use max_patches for
    # quicker experiments or to limit memory when debugging.
    train_dataset = LazyPatchDataset(
        filepaths=train_files,
        patch_size=config["data"]["patch_size"],
        max_patches=config["data"].get("max_train_patches"),
    )

    # For validation reuse normalization parameters from the training dataset (means/stds)
    val_dataset = LazyPatchDataset(
        filepaths=val_files,
        patch_size=config["data"]["patch_size"],
        means=train_dataset.means,
        stds=train_dataset.stds,
        max_patches=config["data"].get("max_val_patches"),
    )

    # PatchCollator converts the dataset's internal image/patch storage into minibatches for DataLoader
    train_collate = PatchCollator(
        train_dataset.images,
        patch_size=config["data"]["patch_size"],
    )

    val_collate = PatchCollator(
        val_dataset.images,
        patch_size=config["data"]["patch_size"],
    )

    # ----------------------------
    # 4) DataLoaders
    # ----------------------------
    # pin_memory=True can improve performance when using CUDA. num_workers can be set in the config.
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config["dataloader_train"]["batch_size"],
        shuffle=config["dataloader_train"]["shuffle"],
        num_workers=config["dataloader_train"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=train_collate,
    )

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=config["dataloader_val"]["batch_size"],
        shuffle=config["dataloader_val"]["shuffle"],
        num_workers=config["dataloader_val"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=val_collate,
    )

    # ----------------------------
    # 5) Load the pretrained model
    # ----------------------------
    print("Loading pretrained autoencoder")
    model = Autoencoder(
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Huggingface/Lightning-style checkpoint that wraps state_dict
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Raw state_dict was provided
        model.load_state_dict(checkpoint)

    # ----------------------------
    # 6) Trainer and callbacks
    # ----------------------------
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
    # 7) Fine-tuning
    # ----------------------------
    print("Fine-tuning autoencoder")
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # Save the best fine-tuned model state_dict for downstream use
    os.makedirs("../results", exist_ok=True)

    best_ckpt_path = checkpoint_callback.best_model_path
    assert best_ckpt_path, "No best checkpoint was saved."
    print(f"Best fine-tuned checkpoint: {best_ckpt_path}")

    best_ckpt_out = "../results/ae_finetuned_best.ckpt"
    shutil.copy2(best_ckpt_path, best_ckpt_out)
    print(f"Copied best fine-tuned checkpoint to {best_ckpt_out}")

    best_model = Autoencoder.load_from_checkpoint(
        best_ckpt_path,
        optimizer_config=config["optimizer"],
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )

    torch.save(best_model.state_dict(), "../results/ae_finetuned.pt")
    print("Saved clean fine-tuned model to ../results/ae_finetuned.pt")

    # Clean up GPU memory again
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()