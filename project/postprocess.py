#!/usr/bin/env python3
"""
Postprocess: load a trained MPNP, run image completion on MNIST test
images, and generate comparison figures.

Reproduces the visualisation from the MPNP paper (ICLR 2023, Figure 6):
  columns = Image | Context | MPNP \mu | MPNP \sigma

For each test image a random subset of pixels is given as context and the
model predicts mean (\mu) and standard deviation (\sigma) for all 784 pixels.

Usage:
    uv run postprocess.py
    uv run postprocess.py --num_context 100 --num_samples 30
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from martingale_posterior_neural_process import MartingalePosteriorNeuralProcess

# ============================================================================
# Config (must match train_inpainting.py)
# ============================================================================

X_DIM = 2
Y_DIM = 1
R_DIM = 256
Z_DIM = 256
H_DIM = 256
NUM_PSEUDO_POINTS = 30
NUM_PSEUDO_SAMPLES = 5
LOSS_WEIGHTS = {"marg": 1.0, "amort": 1.0, "pseudo": 0.1}

NUM_IMAGES = 10                 # test images to visualise
NUM_CONTEXT_DEFAULT = 100       # context pixels per image
NUM_POSTERIOR_SAMPLES = 30      # draws for mean + uncertainty

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
MODEL_PATH = OUTPUT_DIR / "mpnp_mnist.pt"

IMG_H, IMG_W = 28, 28
N_PIXELS = IMG_H * IMG_W        # 784


# ============================================================================
# Helpers
# ============================================================================

def build_coord_grid():
    """Normalised (row, col) coordinate grid for 28x28."""
    rows = np.linspace(-1, 1, IMG_H)
    cols = np.linspace(-1, 1, IMG_W)
    cc, rr = np.meshgrid(cols, rows)
    return torch.tensor(
        np.stack([rr.ravel(), cc.ravel()], axis=1), dtype=torch.float32
    )  # (784, 2)


def predict_image(model, x_ctx, y_ctx, x_all, device, n_samples):
    """Run n_samples forward passes and return (mean, std) images."""
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p_y = model(x_ctx.to(device), y_ctx.to(device), x_all.to(device))
            preds.append(p_y.loc.cpu().squeeze().numpy())

    preds = np.array(preds)  # (S, 784)
    mean = np.clip(preds.mean(axis=0).reshape(IMG_H, IMG_W), 0, 1)
    std  = preds.std(axis=0).reshape(IMG_H, IMG_W)
    return mean, std


def build_context_image(coords, values, shape=(IMG_H, IMG_W)):
    """Scatter context pixels onto a black canvas for visualisation."""
    img = np.zeros(shape, dtype=np.float32)
    rows = ((coords[:, 0].numpy() + 1) / 2 * (shape[0] - 1)).round().astype(int)
    cols = ((coords[:, 1].numpy() + 1) / 2 * (shape[1] - 1)).round().astype(int)
    rows = np.clip(rows, 0, shape[0] - 1)
    cols = np.clip(cols, 0, shape[1] - 1)
    img[rows, cols] = values.squeeze().numpy()
    return img


# ============================================================================
# Visualisations
# ============================================================================

def make_comparison_figure(images, contexts, means, stds, output_path):
    """
    Grid figure: rows = test images, cols = Image | Context | MPNP μ | MPNP σ
    """
    n = len(images)
    col_titles = ["Image", "Context", "MPNP μ", "MPNP σ"]
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for r in range(n):
        axes[r, 0].imshow(images[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(contexts[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(means[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 3].imshow(stds[r], cmap="hot", vmin=0)
        for c in range(4):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_training_metrics(output_path):
    """Plot training loss curve from saved metrics."""
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(metrics["loss_history"], "b-", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    if metrics["eval_history"]:
        key = "val_nll" if "val_nll" in metrics["eval_history"][0] else "test_nll"
        epochs = [e["epoch"] for e in metrics["eval_history"]]
        nlls = [e[key] for e in metrics["eval_history"]]
        ax2.plot(epochs, nlls, "r-o", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("NLL")
        ax2.set_title("Validation Negative Log-Likelihood")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("MPNP Training Metrics (MNIST)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Model loading
# ============================================================================

def load_model(device: torch.device):
    """Load trained MPNP model from disk."""
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Run train_inpainting.py first.")
        return None

    model = MartingalePosteriorNeuralProcess(
        x_dim=X_DIM, y_dim=Y_DIM, r_dim=R_DIM, z_dim=Z_DIM, h_dim=H_DIM,
        num_pseudo_points=NUM_PSEUDO_POINTS,
        num_pseudo_samples=NUM_PSEUDO_SAMPLES,
        use_autoregressive=False,
        x_range=(-1, 1),
        loss_weights=LOSS_WEIGHTS,
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.\n")
    return model


# ============================================================================
# Image completion
# ============================================================================

def run_completions(model, device: torch.device, num_context: int,
                    num_samples: int, num_images: int):
    """Run image completion on MNIST test images, return visualisation data."""
    test_ds = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True,
        transform=transforms.ToTensor(),
    )
    coords = build_coord_grid()

    all_images, all_contexts, all_means, all_stds = [], [], [], []

    print(f"Running image completion ({num_context} context pixels, "
          f"{num_samples} posterior samples) ...\n")

    for i in range(min(num_images, len(test_ds))):
        img_tensor, label = test_ds[i]
        img = img_tensor.squeeze().numpy()
        values = img_tensor.squeeze().reshape(-1, 1)

        perm = torch.randperm(N_PIXELS)
        ctx_idx = perm[:num_context]

        x_ctx = coords[ctx_idx].unsqueeze(0)
        y_ctx = values[ctx_idx].unsqueeze(0)
        x_all = coords.unsqueeze(0)

        mean, std = predict_image(model, x_ctx, y_ctx, x_all, device, num_samples)
        ctx_img = build_context_image(coords[ctx_idx], values[ctx_idx])

        all_images.append(img)
        all_contexts.append(ctx_img)
        all_means.append(mean)
        all_stds.append(std)
        print(f"  [{i+1}] digit={label}  ctx={num_context}px")

    return all_images, all_contexts, all_means, all_stds


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MPNP MNIST postprocess")
    parser.add_argument("--num_context", type=int, default=NUM_CONTEXT_DEFAULT)
    parser.add_argument("--num_samples", type=int, default=NUM_POSTERIOR_SAMPLES)
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("MPNP Postprocess — MNIST Image Completion")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    if model is None:
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images, contexts, means, stds = run_completions(
        model, device, args.num_context, args.num_samples, args.num_images,
    )

    make_comparison_figure(images, contexts, means, stds,
                      OUTPUT_DIR / "mpnp_mnist_completion.png")
    plot_training_metrics(OUTPUT_DIR / "training_metrics.png")

    print("\n" + "=" * 60)
    print("Postprocessing complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
