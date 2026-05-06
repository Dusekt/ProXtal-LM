#!/usr/bin/env python3
"""
Inference script for ProXtal-LM models.

Loads a trained checkpoint and runs inference on HDF5 datasets or single
protein embeddings.  Configuration is read from a JSON file (see
``configs/`` directory).

Usage examples::

    # Run inference on test set using a JSON config
    python scripts/inference.py \\
        --checkpoint checkpoints/v8_esmc_0/best_checkpoint.pt \\
        --config configs/large.json

    # Override data path
    python scripts/inference.py \\
        --checkpoint checkpoints/v8_esmc_0/best_checkpoint.pt \\
        --config configs/large.json \\
        --data data/test_data_3d_esmc

    # Save raw predictions to disk
    python scripts/inference.py \\
        --checkpoint checkpoints/v8_esmc_0/best_checkpoint.pt \\
        --config configs/large.json \\
        --output predictions.h5
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxtal_lm.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from proxtal_lm.data import CrystalContactsDataset, collate_pad
from proxtal_lm.models import CrystalTriangularModel


# ================================================================
# Config loading from JSON
# ================================================================


def load_config_from_json(json_path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a JSON file.

    Args:
        json_path: Path to the JSON configuration file.

    Returns:
        A fully populated ExperimentConfig.
    """
    with open(json_path) as f:
        raw = json.load(f)

    model_cfg = ModelConfig(**raw.get("model", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))

    config = ExperimentConfig(
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
        name=raw.get("name", "inference"),
        description=raw.get("description", ""),
    )
    return config


def build_model_from_config(
    config: ExperimentConfig,
    device: torch.device,
) -> CrystalTriangularModel:
    """Instantiate a CrystalTriangularModel from config.

    Args:
        config: Experiment configuration.
        device: Target device.

    Returns:
        The model moved to the target device with eval mode set.
    """
    mc = config.model
    win = mc.attention_window_size if mc.attention_window_size > 0 else None
    model = CrystalTriangularModel(
        emb_dim=mc.emb_dim,
        d_model=mc.d_model,
        d_pair=mc.d_pair,
        n_seq_layers=mc.n_seq_layers,
        n_blocks=mc.n_blocks,
        tri_hidden=mc.tri_hidden,
        out_ch=mc.out_ch,
        use_checkpoint=False,
        n_recycles=mc.n_recycles,
        max_rel_pos=mc.max_rel_pos,
        num_space_groups=mc.num_space_groups,
        n_heads=mc.n_heads,
        attention_window_size=win,
        dropout=0.0,
        n_hypotheses=mc.n_hypotheses,
    ).to(device)
    model.eval()
    return model


def load_checkpoint_weights(
    model: CrystalTriangularModel,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Load model weights from a checkpoint file.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to the checkpoint file.
        device: Target device.

    Returns:
        The full checkpoint dictionary (for metadata access).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        cleaned = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned)

    return ckpt


# ================================================================
# Inference routines
# ================================================================


@torch.no_grad()
def predict_dataset(
    model: CrystalTriangularModel,
    dataloader: DataLoader,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    """Run inference on an entire dataset.

    Args:
        model: Trained model in eval mode.
        dataloader: DataLoader yielding batches.
        device: Target device.

    Returns:
        List of dicts, each containing ``'logits'``, ``'pred_bins'``,
        ``'contact_prob'``, and ``'lengths'``.
    """
    results = []

    for batch in dataloader:
        emb = batch["embedding"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"]

        chain_id = batch.get("chain_id")
        if chain_id is not None:
            chain_id = chain_id.to(device)

        space_group = batch.get("space_group")
        if space_group is not None:
            space_group = space_group.to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(emb, seq_mask=mask, chain_id=chain_id, space_group=space_group)

        # Handle multi-hypothesis: use first hypothesis
        if logits.dim() == 5:
            logits = logits[:, 0]

        probs = torch.softmax(logits, dim=-1)
        pred_bins = torch.argmax(logits, dim=-1)
        contact_prob = probs[..., :27].sum(dim=-1)

        for b in range(emb.size(0)):
            L = lengths[b].item()
            results.append({
                "logits": logits[b, :L, :L].cpu().numpy(),
                "pred_bins": pred_bins[b, :L, :L].cpu().numpy(),
                "contact_prob": contact_prob[b, :L, :L].cpu().numpy(),
                "length": L,
            })

    return results


@torch.no_grad()
def predict_single(
    model: CrystalTriangularModel,
    embedding: torch.Tensor,
    device: torch.device,
    chain_id: Optional[torch.Tensor] = None,
    space_group: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Run inference on a single protein embedding.

    Args:
        model: Trained model in eval mode.
        embedding: ``[L, D]`` ESM embedding for one protein.
        device: Target device.
        chain_id: ``[L]`` chain assignments (optional).
        space_group: scalar space-group ID (optional).

    Returns:
        Dict with ``'logits'``, ``'pred_bins'``, ``'contact_prob'``.
    """
    emb = embedding.unsqueeze(0).to(device)
    mask = torch.ones(1, emb.size(1), dtype=torch.bool, device=device)

    if chain_id is not None:
        chain_id = chain_id.unsqueeze(0).to(device)
    if space_group is not None:
        if not isinstance(space_group, torch.Tensor):
            space_group = torch.tensor([space_group], dtype=torch.long)
        space_group = space_group.to(device)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(emb, seq_mask=mask, chain_id=chain_id, space_group=space_group)

    if logits.dim() == 5:
        logits = logits[:, 0]

    probs = torch.softmax(logits, dim=-1)
    L = emb.size(1)
    return {
        "logits": logits[0, :L, :L].cpu().numpy(),
        "pred_bins": torch.argmax(logits, dim=-1)[0, :L, :L].cpu().numpy(),
        "contact_prob": probs[0, :L, :L, :27].sum(dim=-1).cpu().numpy(),
    }


def save_predictions_h5(results: List[Dict[str, np.ndarray]], output_path: str) -> None:
    """Save predictions to an HDF5 file.

    Args:
        results: List of prediction dicts from ``predict_dataset``.
        output_path: Path to the output HDF5 file.
    """
    with h5py.File(output_path, "w") as f:
        for i, res in enumerate(results):
            grp = f.create_group(str(i))
            for key, val in res.items():
                if isinstance(val, np.ndarray):
                    grp.create_dataset(key, data=val, compression="gzip")
                else:
                    grp.attrs[key] = val
    print(f"Saved {len(results)} predictions to {output_path}")


# ================================================================
# CLI
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="Run ProXtal-LM inference")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--data", default=None, help="Path to HDF5 data (overrides config)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Path to save predictions (HDF5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config_from_json(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine data path
    data_path = args.data or config.data.test_path
    batch_size = args.batch_size or config.data.batch_size
    print(f"Data:   {data_path}")

    # Build model
    model = build_model_from_config(config, device)
    ckpt = load_checkpoint_weights(model, args.checkpoint, device)
    epoch = ckpt.get("epoch", "unknown")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:  {n_params:,} params (epoch {epoch})")

    # Load dataset
    dataset = CrystalContactsDataset(data_path, B=config.data.num_bins)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=lambda batch: collate_pad(batch, pad_multiple=config.data.pad_multiple),
    )
    print(f"Samples: {len(dataset)}")

    # Run inference
    results = predict_dataset(model, dataloader, device)
    print(f"Predicted {len(results)} samples")

    # Save if requested
    if args.output:
        save_predictions_h5(results, args.output)

    return results


if __name__ == "__main__":
    main()
