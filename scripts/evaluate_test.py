#!/usr/bin/env python3
"""
Test-set evaluation script for ProXtal-LM.

Evaluates a trained checkpoint on the test split and reports all metrics
from ``training_metrics.csv`` (precision, recall, F1, AUPRC,
crystal-specific metrics, and per-hypothesis losses).

Usage::

    python scripts/evaluate_test.py \\
        --checkpoint checkpoints/v8_esmc_0/best_checkpoint.pt \\
        --config configs/large.json

    # Save results to CSV
    python scripts/evaluate_test.py \\
        --checkpoint checkpoints/v8_esmc_0/best_checkpoint.pt \\
        --config configs/large.json \\
        --output results/test_metrics.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxtal_lm.data import CrystalContactsDataset, collate_pad
from proxtal_lm.training import validate_one_epoch
from scripts.inference import (
    build_model_from_config,
    load_checkpoint_weights,
    load_config_from_json,
)


def evaluate_test(
    checkpoint_path: str,
    config_path: str,
    data_path: str | None = None,
    batch_size: int | None = None,
    device_str: str = "cuda",
) -> Dict[str, float]:
    """Evaluate a checkpoint on the test set.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to JSON config file.
        data_path: Optional override for the test data path.
        batch_size: Optional batch-size override.
        device_str: ``'cuda'`` or ``'cpu'``.

    Returns:
        Dictionary of all evaluation metrics.
    """
    config = load_config_from_json(config_path)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Resolve data path
    test_path = data_path or config.data.test_path
    bs = batch_size or config.data.batch_size

    print(f"Device:     {device}")
    print(f"Test data:  {test_path}")
    print(f"Batch size: {bs}")

    # Build model and load weights
    model = build_model_from_config(config, device)
    ckpt = load_checkpoint_weights(model, checkpoint_path, device)
    epoch = ckpt.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:      {n_params:,} params (epoch {epoch})")

    # Dataset
    dataset = CrystalContactsDataset(test_path, B=config.data.num_bins)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=lambda batch: collate_pad(batch, pad_multiple=config.data.pad_multiple),
    )
    print(f"Samples:    {len(dataset)}")

    # Run evaluation (reuses the validation loop which computes all metrics)
    metrics = validate_one_epoch(model, loader, device)

    # Rename val_ prefix to test_ for clarity
    test_metrics = {}
    for key, value in metrics.items():
        new_key = key.replace("val_", "test_", 1) if key.startswith("val_") else key
        test_metrics[new_key] = value

    return test_metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics grouped by category."""
    print("\n" + "=" * 65)
    print("  TEST-SET EVALUATION RESULTS")
    print("=" * 65)

    # Group metrics
    groups = {
        "Loss": [],
        "Precision / Recall / F1": [],
        "Crystal-Only": [],
        "AUPRC": [],
        "Hypothesis": [],
        "Other": [],
    }

    for key, val in sorted(metrics.items()):
        if "loss" in key:
            groups["Loss"].append((key, val))
        elif any(k in key for k in ("precision", "recall", "f1")):
            if "crystal" in key:
                groups["Crystal-Only"].append((key, val))
            else:
                groups["Precision / Recall / F1"].append((key, val))
        elif "auprc" in key:
            groups["AUPRC"].append((key, val))
        elif "hyp" in key:
            groups["Hypothesis"].append((key, val))
        elif "crystal" in key or "changed" in key or "accuracy" in key:
            groups["Crystal-Only"].append((key, val))
        else:
            groups["Other"].append((key, val))

    for group_name, items in groups.items():
        if not items:
            continue
        print(f"\n  {group_name}")
        print("  " + "-" * 40)
        for key, val in items:
            print(f"    {key:40s}  {val:.6f}")

    print("\n" + "=" * 65)


def save_metrics_csv(metrics: Dict[str, float], output_path: str) -> None:
    """Save metrics to a CSV file.

    Args:
        metrics: Dictionary of metric name -> value.
        output_path: Output CSV file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, val in sorted(metrics.items()):
            writer.writerow([key, val])
    print(f"Metrics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProXtal-LM on test data")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--data", default=None, help="Override test data path"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Save metrics to CSV"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size override"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    metrics = evaluate_test(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_path=args.data,
        batch_size=args.batch_size,
        device_str=args.device,
    )

    print_metrics(metrics)

    if args.output:
        save_metrics_csv(metrics, args.output)


if __name__ == "__main__":
    main()
