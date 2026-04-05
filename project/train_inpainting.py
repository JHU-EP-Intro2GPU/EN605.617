#!/usr/bin/env python3
"""
Train the Martingale Posterior Neural Process (MPNP) on MNIST.

Follows the original MPNP paper (ICLR 2023) setup:
  - MNIST 28x28 grayscale images as pixel point clouds
  - 50,000 train / 10,000 validation / 10,000 test split
  - Random context pixels → predict all pixels (image completion)
  - Evaluate predicted mean and standard deviation

Usage:
    uv run train_inpainting.py                         # defaults
    uv run train_inpainting.py --epochs 50 --lr 5e-4   # custom
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from martingale_posterior_neural_process import MartingalePosteriorNeuralProcess

# ============================================================================
# Configuration
# ============================================================================

# Model architecture
X_DIM = 2          # (row, col)
Y_DIM = 1          # grayscale intensity
R_DIM = 256
Z_DIM = 256
H_DIM = 256

# MPNP pseudo-context config
NUM_PSEUDO_POINTS = 30
NUM_PSEUDO_SAMPLES = 5
LOSS_WEIGHTS = {"marg": 1.0, "amort": 1.0, "pseudo": 0.1}

# Training defaults (paper: 50K train, 10K val, 10K test)
EPOCHS = 100
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5
NUM_CONTEXT_RANGE = (10, 150)   # random number of context pixels per image
NUM_EXTRA_TARGET = 784          # predict all pixels per sample
SEED = 42

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")


# ============================================================================
# Dataset — MNIST as NP point clouds
# ============================================================================

class MNISTPointCloud(Dataset):
    """
    Wraps torchvision MNIST → pixel point cloud.
        x = normalised (row, col) ∈ [-1, 1]²
        y = pixel intensity ∈ [0, 1]
    """

    def __init__(self, root: Path, train: bool = True):
        self.ds = datasets.MNIST(
            root=str(root), train=train, download=True,
            transform=transforms.ToTensor(),
        )
        # Pre-compute normalised coordinate grid for 28×28
        h, w = 28, 28
        rows = np.linspace(-1, 1, h)
        cols = np.linspace(-1, 1, w)
        cc, rr = np.meshgrid(cols, rows)
        self.coords = torch.tensor(
            np.stack([rr.ravel(), cc.ravel()], axis=1), dtype=torch.float32
        )  # (784, 2)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]                  # (1, 28, 28)
        values = img.squeeze().reshape(-1, 1)  # (784, 1)
        return self.coords, values             # both float32


# ============================================================================
# Training helpers
# ============================================================================

def context_target_split(x, y, num_context, num_extra_target):
    """Random context+target split from a full pixel point cloud."""
    n = x.shape[1]  # (batch, N, dim)
    n_tgt = min(num_context + num_extra_target, n)
    n_ctx = min(num_context, n_tgt - 1)

    perm = torch.randperm(n, device=x.device)
    idx_tgt = perm[:n_tgt]
    idx_ctx = idx_tgt[:n_ctx]

    return x[:, idx_ctx], y[:, idx_ctx], x[:, idx_tgt], y[:, idx_tgt]


def train_epoch(model, loader, optimizer, device, ctx_range, n_extra_tgt):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_full, y_full in loader:
        x_full = x_full.to(device)  # (B, 784, 2)
        y_full = y_full.to(device)  # (B, 784, 1)

        num_ctx = np.random.randint(*ctx_range)
        x_ctx, y_ctx, x_tgt, y_tgt = context_target_split(
            x_full, y_full, num_ctx, n_extra_tgt
        )

        optimizer.zero_grad()
        loss = model.compute_loss(x_ctx, y_ctx, x_tgt, y_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, device, num_context=100):
    model.eval()
    total_nll = 0.0
    n_batches = 0
    with torch.no_grad():
        for x_full, y_full in loader:
            x_full = x_full.to(device)
            y_full = y_full.to(device)
            x_ctx, y_ctx, x_tgt, y_tgt = context_target_split(
                x_full, y_full, num_context, 300
            )
            pred = model(x_ctx, y_ctx, x_tgt)
            nll = -pred.log_prob(y_tgt).mean().item()
            total_nll += nll
            n_batches += 1
    return total_nll / n_batches


# ============================================================================
# Data loading
# ============================================================================

def create_dataloaders(seed: int):
    """Load MNIST and return train/val/test DataLoaders."""
    DATA_DIR.mkdir(exist_ok=True)
    full_train = MNISTPointCloud(DATA_DIR, train=True)
    test_ds    = MNISTPointCloud(DATA_DIR, train=False)

    indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed))
    train_ds = Subset(full_train, indices[:50000].tolist())
    val_ds   = Subset(full_train, indices[50000:].tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ============================================================================
# Model + optimiser setup
# ============================================================================

def create_model(device: torch.device, lr: float, epochs: int):
    """Instantiate MPNP model, optimiser, and scheduler."""
    model = MartingalePosteriorNeuralProcess(
        x_dim=X_DIM, y_dim=Y_DIM, r_dim=R_DIM, z_dim=Z_DIM, h_dim=H_DIM,
        num_pseudo_points=NUM_PSEUDO_POINTS,
        num_pseudo_samples=NUM_PSEUDO_SAMPLES,
        use_autoregressive=False,
        x_range=(-1, 1),
        loss_weights=LOSS_WEIGHTS,
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    return model, optimizer, scheduler


# ============================================================================
# Training loop
# ============================================================================

def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 device: torch.device, epochs: int):
    """Run the full training loop with periodic validation."""
    best_nll = float("inf")
    best_state = None
    loss_history = []
    eval_history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(
            model, train_loader, optimizer, device,
            NUM_CONTEXT_RANGE, NUM_EXTRA_TARGET,
        )
        scheduler.step()
        loss_history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            val_nll = evaluate(model, val_loader, device)
            eval_history.append({"epoch": epoch, "val_nll": val_nll})
            tag = ""
            if val_nll < best_nll:
                best_nll = val_nll
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                tag = " ★"
            print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  val_nll={val_nll:.4f}{tag}")

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Best val NLL: {best_nll:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return best_nll, elapsed, loss_history, eval_history


# ============================================================================
# Saving results
# ============================================================================

def save_results(model, best_nll: float, test_nll: float, elapsed: float,
                 epochs: int, lr: float, loss_history: list, eval_history: list):
    """Save model weights and metrics JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "mpnp_mnist.pt")

    metrics = {
        "best_val_nll": best_nll,
        "test_nll": test_nll,
        "elapsed_s": elapsed,
        "epochs": epochs,
        "loss_history": loss_history,
        "eval_history": eval_history,
        "config": {
            "dataset": "MNIST",
            "train_size": 50000, "val_size": 10000, "test_size": 10000,
            "r_dim": R_DIM, "z_dim": Z_DIM, "h_dim": H_DIM,
            "num_pseudo_points": NUM_PSEUDO_POINTS,
            "num_pseudo_samples": NUM_PSEUDO_SAMPLES,
            "lr": lr,
        },
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {OUTPUT_DIR / 'mpnp_mnist.pt'}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MPNP on MNIST")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("MPNP Training (MNIST image completion)")
    print("=" * 60)
    print(f"Device: {device}")

    print("\nLoading MNIST ...")
    train_loader, val_loader, test_loader = create_dataloaders(args.seed)

    model, optimizer, scheduler = create_model(device, args.lr, args.epochs)

    print(f"\nTraining for {args.epochs} epochs ...\n")
    best_nll, elapsed, loss_history, eval_history = run_training(
        model, train_loader, val_loader, optimizer, scheduler,
        device, args.epochs,
    )

    test_nll = evaluate(model, test_loader, device)
    print(f"Test NLL: {test_nll:.4f}")

    save_results(model, best_nll, test_nll, elapsed, args.epochs,
                 args.lr, loss_history, eval_history)
    print("=" * 60)


if __name__ == "__main__":
    main()
