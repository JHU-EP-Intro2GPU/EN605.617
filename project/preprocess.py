#!/usr/bin/env python3
"""
Preprocess — download MNIST and verify the dataset for MPNP training.

Downloads the MNIST dataset via torchvision (if not already cached),
prints summary statistics, and saves a few sample images for inspection.

Usage:
    uv run preprocess.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# ============================================================================
# Configuration (must match train_inpainting.py)
# ============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
SEED = 42

IMG_H, IMG_W = 28, 28
N_PIXELS = IMG_H * IMG_W  # 784

TRAIN_SIZE = 50_000
VAL_SIZE = 10_000


# ============================================================================
# Data loading
# ============================================================================

def download_mnist():
    """Download MNIST via torchvision and return train/test datasets."""
    print("\nDownloading MNIST (if needed) ...")
    train_ds = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True,
        transform=transforms.ToTensor(),
    )
    test_ds = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    assert len(train_ds) == 60_000, f"Expected 60,000 train, got {len(train_ds)}"
    assert len(test_ds) == 10_000, f"Expected 10,000 test, got {len(test_ds)}"
    return train_ds, test_ds


def split_indices():
    """Return train/val index split using the paper's 50K/10K scheme."""
    gen = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(60_000, generator=gen)
    return indices[:TRAIN_SIZE], indices[TRAIN_SIZE:]


# ============================================================================
# Summary statistics
# ============================================================================

def print_split_summary(train_ds, test_ds):
    print(f"\n  Total train images:  {len(train_ds):,}")
    print(f"  → Training split:   {TRAIN_SIZE:,}")
    print(f"  → Validation split: {VAL_SIZE:,}")
    print(f"  Test images:         {len(test_ds):,}")
    print(f"  Image size:          {IMG_H}×{IMG_W} ({N_PIXELS} pixels)")


def print_pixel_statistics(train_ds, train_idx):
    """Compute and print intensity statistics over a 5K sample."""
    print("\nComputing pixel statistics (training split) ...")
    pixels = []
    for i in train_idx[:5000]:
        img, _ = train_ds[int(i)]
        pixels.append(img.numpy().ravel())
    pixels = np.concatenate(pixels)
    print(f"  Mean intensity:  {pixels.mean():.4f}")
    print(f"  Std intensity:   {pixels.std():.4f}")
    print(f"  Min:             {pixels.min():.4f}")
    print(f"  Max:             {pixels.max():.4f}")
    print(f"  % zero pixels:   {(pixels == 0).mean() * 100:.1f}%")


def print_label_distribution(train_ds, train_idx):
    labels = [train_ds[int(i)][1] for i in train_idx]
    counts = np.bincount(labels, minlength=10)
    print("\n  Label distribution (train split):")
    for digit in range(10):
        print(f"    {digit}: {counts[digit]:,}")


# ============================================================================
# Visualisation
# ============================================================================

def save_sample_grid(train_ds, train_idx):
    """Save a 2x10 grid of sample images for visual inspection."""
    print("\nSaving sample grid → output/mnist_samples.png ...")
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for i in range(20):
        img, label = train_ds[int(train_idx[i])]
        r, c = divmod(i, 10)
        axes[r, c].imshow(img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_title(str(label), fontsize=9)
        axes[r, c].axis("off")
    plt.suptitle("MNIST Training Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mnist_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Done.")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    print("=" * 60)
    print("MPNP Preprocess — MNIST Setup")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = download_mnist()
    train_idx, _val_idx = split_indices()

    print_split_summary(train_ds, test_ds)
    print_pixel_statistics(train_ds, train_idx)
    print_label_distribution(train_ds, train_idx)
    save_sample_grid(train_ds, train_idx)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
