#!/usr/bin/env python3
"""
Publication-quality graph utilities for ProXtal-LM.

Provides reusable plotting functions that produce journal-ready figures
with consistent styling. Each function supports two modes:

- ``mode='view'``:  Display the figure inline (e.g. in a Jupyter notebook).
- ``mode='save'``:  Save the figure to disk as a high-resolution PDF/PNG.

Usage (script)::

    python publication_graphs.py checkpoints/v8_esmc_0/training_metrics.csv

Usage (notebook)::

    from scripts.publication_graphs import plot_training_curves, plot_metric_comparison
    plot_training_curves("checkpoints/v8_esmc_0/training_metrics.csv", mode="view")
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global style setup
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def apply_publication_style() -> None:
    """Apply a clean, publication-ready matplotlib style (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    plt.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            # Axes
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "0.8",
        }
    )
    _STYLE_APPLIED = True


# Colour palette (colour-blind friendly)
COLOURS = {
    "train": "#0072B2",
    "val": "#D55E00",
    "best": "#009E73",
    "highlight": "#CC79A7",
    "grey": "#999999",
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "red": "#E69F00",
}


def _output(fig: plt.Figure, path: Optional[str], mode: str) -> None:
    """Handle view/save output."""
    if mode == "save" and path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path)
        print(f"Saved: {path}")
        plt.close(fig)
    elif mode == "view":
        plt.show()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Core plotting functions
# ---------------------------------------------------------------------------


def plot_training_curves(
    csv_path: str,
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        csv_path: Path to the training_metrics.csv file.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path (used when ``mode='save'``).
        figsize: Figure dimensions in inches.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Loss ---
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train", color=COLOURS["train"])
    ax.plot(df["epoch"], df["val_loss"], label="Validation", color=COLOURS["val"])
    if "best_val_loss" in df.columns:
        best_mask = df["best_val_loss"] != np.inf
        best_series = df.loc[best_mask, "best_val_loss"]
        if not best_series.empty:
            best_epoch = df.loc[best_series.idxmin(), "epoch"]
            best_loss = best_series.min()
            ax.axvline(best_epoch, ls="--", color=COLOURS["best"], alpha=0.6, lw=1)
            ax.annotate(
                f"Best: {best_loss:.3f}",
                xy=(best_epoch, best_loss),
                xytext=(10, 15),
                textcoords="offset points",
                fontsize=8,
                color=COLOURS["best"],
                arrowprops=dict(arrowstyle="->", color=COLOURS["best"], lw=0.8),
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()

    # --- Learning rate ---
    ax = axes[1]
    ax.plot(df["epoch"], df["lr"], color=COLOURS["cyan"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


def plot_precision_recall(
    csv_path: str,
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot precision at L, L/2, L/5 and recall/F1 for train and validation.

    Args:
        csv_path: Path to the training_metrics.csv file.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Precision ---
    ax = axes[0]
    for suffix, ls in [("_L", "-"), ("_L2", "--"), ("_L5", ":")]:
        for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
            col = f"{prefix}_precision{suffix}"
            if col in df.columns:
                label = f"{prefix.capitalize()} P@{suffix.replace('_', '')}"
                ax.plot(df["epoch"], df[col], ls=ls, color=colour, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Precision @ L / L/2 / L/5")
    ax.legend(fontsize=7, ncol=2)

    # --- Recall ---
    ax = axes[1]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_recall_L"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=f"{prefix.capitalize()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall @ L")
    ax.set_title("Recall @ L")
    ax.legend()

    # --- F1 ---
    ax = axes[2]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_f1_L"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=f"{prefix.capitalize()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 @ L")
    ax.set_title("F1 Score @ L")
    ax.legend()

    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


def plot_crystal_metrics(
    csv_path: str,
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot crystal-specific metrics (precision, accuracy, changed positions).

    Args:
        csv_path: Path to the training_metrics.csv file.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Crystal-only precision ---
    ax = axes[0]
    for suffix, ls in [("_L", "-"), ("_L2", "--"), ("_L5", ":")]:
        for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
            col = f"{prefix}_crystal_only_precision{suffix}"
            if col in df.columns:
                label = f"{prefix.capitalize()} P@{suffix.replace('_', '')}"
                ax.plot(df["epoch"], df[col], ls=ls, color=colour, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Crystal-Only Precision")
    ax.legend(fontsize=7, ncol=2)

    # --- Crystal-only accuracy ---
    ax = axes[1]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_crystal_only_accuracy"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=f"{prefix.capitalize()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Crystal-Only Accuracy")
    ax.legend()

    # --- Changed positions ---
    ax = axes[2]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_pct_changed_positions"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=f"{prefix.capitalize()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("% Changed Positions")
    ax.set_title("Changed Positions (%)")
    ax.legend()

    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


def plot_auprc(
    csv_path: str,
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    """Plot area under the precision-recall curve (AUPRC).

    Args:
        csv_path: Path to the training_metrics.csv file.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=figsize)
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_auprc"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=f"{prefix.capitalize()}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPRC")
    ax.set_title("Area Under Precision-Recall Curve")
    ax.legend()

    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


def plot_hypothesis_losses(
    csv_path: str,
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot per-hypothesis per-target validation losses.

    Args:
        csv_path: Path to the training_metrics.csv file.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    df = pd.read_csv(csv_path)

    # Detect hypothesis columns
    hyp_cols = [c for c in df.columns if c.startswith("val_hyp") and "loss" in c]
    if not hyp_cols:
        print("No hypothesis loss columns found; skipping.")
        return plt.figure()

    # Group by hypothesis
    best_cols = [c for c in hyp_cols if "best" in c]
    tgt_cols = [c for c in hyp_cols if "tgt" in c and "best" not in c]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Best loss per hypothesis ---
    ax = axes[0]
    palette = [COLOURS["blue"], COLOURS["orange"], COLOURS["green"]]
    for i, col in enumerate(sorted(best_cols)):
        colour = palette[i % len(palette)]
        label = col.replace("val_", "").replace("_", " ").title()
        ax.plot(df["epoch"], df[col], color=colour, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Best Hypothesis Loss")
    ax.legend(fontsize=8)

    # --- Per-target losses ---
    ax = axes[1]
    for i, col in enumerate(sorted(tgt_cols)):
        colour = palette[i % len(palette)]
        ls = ["-", "--", ":"][i % 3]
        label = col.replace("val_", "").replace("_", " ").title()
        ax.plot(df["epoch"], df[col], color=colour, ls=ls, label=label, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Per-Hypothesis Per-Target Loss")
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


def plot_summary_dashboard(
    csv_path: str,
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """Create a comprehensive 2x3 dashboard of all key metrics.

    Args:
        csv_path: Path to the training_metrics.csv file.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # (0,0) Loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="Train", color=COLOURS["train"])
    ax.plot(df["epoch"], df["val_loss"], label="Val", color=COLOURS["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # (0,1) Precision @ L
    ax = axes[0, 1]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_precision_L"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=prefix.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision @ L")
    ax.set_title("Precision @ L")
    ax.legend()

    # (0,2) AUPRC
    ax = axes[0, 2]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_auprc"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=prefix.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPRC")
    ax.set_title("AUPRC")
    ax.legend()

    # (1,0) F1 @ L
    ax = axes[1, 0]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_f1_L"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=prefix.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 @ L")
    ax.set_title("F1 Score @ L")
    ax.legend()

    # (1,1) Crystal-only precision @ L
    ax = axes[1, 1]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_crystal_only_precision_L"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=prefix.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Crystal Precision @ L")
    ax.set_title("Crystal-Only Precision @ L")
    ax.legend()

    # (1,2) Crystal-only accuracy
    ax = axes[1, 2]
    for prefix, colour in [("train", COLOURS["train"]), ("val", COLOURS["val"])]:
        col = f"{prefix}_crystal_only_accuracy"
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colour, label=prefix.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Crystal-Only Accuracy")
    ax.legend()

    fig.suptitle("ProXtal-LM Training Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


def plot_metric_comparison(
    csv_paths: Dict[str, str],
    metric: str = "val_loss",
    *,
    mode: str = "view",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Compare a single metric across multiple training runs.

    Args:
        csv_paths: Mapping of ``{run_name: csv_path}``.
        metric: Column name to plot.
        mode: ``'view'`` to display, ``'save'`` to write to disk.
        save_path: Output file path.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure object.
    """
    apply_publication_style()
    palette = list(COLOURS.values())

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, path) in enumerate(csv_paths.items()):
        df = pd.read_csv(path)
        if metric in df.columns:
            ax.plot(df["epoch"], df[metric], color=palette[i % len(palette)], label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Comparison: {metric.replace('_', ' ').title()}")
    ax.legend()

    fig.tight_layout()
    _output(fig, save_path, mode)
    return fig


# ---------------------------------------------------------------------------
# CLI for batch generation
# ---------------------------------------------------------------------------


def generate_all(csv_path: str, output_dir: str = "figures") -> None:
    """Generate all publication figures and save them as PDFs.

    Args:
        csv_path: Path to the training_metrics.csv file.
        output_dir: Directory to save figures into.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(csv_path, mode="save", save_path=f"{output_dir}/loss_curves.pdf")
    plot_precision_recall(csv_path, mode="save", save_path=f"{output_dir}/precision_recall.pdf")
    plot_crystal_metrics(csv_path, mode="save", save_path=f"{output_dir}/crystal_metrics.pdf")
    plot_auprc(csv_path, mode="save", save_path=f"{output_dir}/auprc.pdf")
    plot_hypothesis_losses(csv_path, mode="save", save_path=f"{output_dir}/hypothesis_losses.pdf")
    plot_summary_dashboard(csv_path, mode="save", save_path=f"{output_dir}/summary_dashboard.pdf")

    print(f"\nAll figures saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality ProXtal-LM figures")
    parser.add_argument("csv_path", help="Path to training_metrics.csv")
    parser.add_argument(
        "--output-dir", "-o", default="figures", help="Output directory (default: figures)"
    )
    args = parser.parse_args()
    generate_all(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()
