"""Postprocessing for Martingale Posterior.

Reads CSV output from the CUDA program and generates:
  - charts/mean_posterior.png
  - charts/variance_posterior.png
  - charts/combined.png
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def parse_histogram(path):
    centers, densities = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            centers.append(float(row["bin_center"]))
            densities.append(float(row["density"]))
    return centers, densities


def parse_summary(path):
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            result[row["statistic"]] = {
                k: float(v) for k, v in row.items()
                if k != "statistic"
            }
    return result


def plot_posterior(centers, density, name,
                   true_val, q025, q975, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bw = (centers[1] - centers[0]) * 0.9

    ax.bar(centers, density, width=bw,
           alpha=0.7, color="#2196F3",
           label="Martingale posterior")
    ax.axvline(true_val, color="green", lw=2,
               label=f"True value = {true_val:.2f}")
    ax.axvspan(q025, q975, alpha=0.12, color="orange",
               label=f"95% CI [{q025:.3f}, {q975:.3f}]")
    ax.axvline(q025, color="orange", lw=1.5, ls="--")
    ax.axvline(q975, color="orange", lw=1.5, ls="--")

    ax.set_xlabel(name, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Martingale Posterior: {name}\n",
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def plot_combined(mc, md, ms, vc, vd, vs, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))

    bw_m = (mc[1] - mc[0]) * 0.9
    bw_v = (vc[1] - vc[0]) * 0.9

    # --- Mean posterior (left) ---
    a1.bar(mc, md, width=bw_m, alpha=0.6,
           color="#2196F3", label="Martingale posterior")

    # Overlay: frequentist normal approximation
    obs_mean = ms["mean"]
    obs_se = ms["std"]
    x_norm = np.linspace(mc[0], mc[-1], 300)
    y_norm = norm.pdf(x_norm, loc=obs_mean,
                      scale=obs_se)
    a1.plot(x_norm, y_norm, "k--", lw=1.5,
            label=f"Normal approx (frequentist)")

    a1.axvline(ms["true_value"], color="#4CAF50",
               lw=2.5, label=f'True μ = {ms["true_value"]:.1f}')
    a1.axvspan(ms["q025"], ms["q975"],
               alpha=0.12, color="orange")
    a1.axvline(ms["q025"], color="orange",
               lw=1.5, ls="--",
               label=(f'95% CI [{ms["q025"]:.2f},'
                      f' {ms["q975"]:.2f}]'))
    a1.axvline(ms["q975"], color="orange",
               lw=1.5, ls="--")

    a1.set_title("Mean Posterior", fontsize=13,
                 fontweight="bold")
    a1.set_xlabel("Mean", fontsize=11)
    a1.set_ylabel("Density", fontsize=11)
    a1.legend(fontsize=8, loc="upper left")
    a1.grid(True, alpha=0.3)

    # --- Variance posterior (right) ---
    a2.bar(vc, vd, width=bw_v, alpha=0.6,
           color="#E91E63",
           label="Martingale posterior")

    a2.axvline(vs["true_value"], color="#4CAF50",
               lw=2.5,
               label=f'True σ² = {vs["true_value"]:.1f}')
    a2.axvspan(vs["q025"], vs["q975"],
               alpha=0.12, color="orange")
    a2.axvline(vs["q025"], color="orange",
               lw=1.5, ls="--",
               label=(f'95% CI [{vs["q025"]:.2f},'
                      f' {vs["q975"]:.2f}]'))
    a2.axvline(vs["q975"], color="orange",
               lw=1.5, ls="--")

    a2.set_title("Variance Posterior", fontsize=13,
                 fontweight="bold")
    a2.set_xlabel("Variance", fontsize=11)
    a2.set_ylabel("Density", fontsize=11)
    a2.legend(fontsize=8, loc="upper right")
    a2.grid(True, alpha=0.3)

    fig.suptitle(
        "Bayesian posteriors from predictive resampling",
        fontsize=12, y=1.01, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main():
    data = Path("data")
    charts = Path("charts")
    charts.mkdir(exist_ok=True)

    for f in ["mean_posterior.csv",
              "variance_posterior.csv", "summary.csv"]:
        if not (data / f).exists():
            print(f"Error: data/{f} not found. "
                  "Run ./assignment.exe first.")
            sys.exit(1)

    print("Generating charts...")
    s = parse_summary(data / "summary.csv")

    mc, md = parse_histogram(data / "mean_posterior.csv")
    plot_posterior(mc, md, "Mean",
                   s["mean"]["true_value"],
                   s["mean"]["q025"],
                   s["mean"]["q975"],
                   charts / "mean_posterior.png")

    vc, vd = parse_histogram(data / "variance_posterior.csv")
    plot_posterior(vc, vd, "Variance",
                   s["variance"]["true_value"],
                   s["variance"]["q025"],
                   s["variance"]["q975"],
                   charts / "variance_posterior.png")

    plot_combined(mc, md, s["mean"],
                  vc, vd, s["variance"],
                  charts / "combined.png")

    print(f"\nAll outputs in {charts}/")


if __name__ == "__main__":
    main()
