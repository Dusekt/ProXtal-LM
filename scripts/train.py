#!/usr/bin/env python3
"""
Main training script for ProXtal-LM V2.

Usage examples:
    # Baseline (no recycling)
    python train.py

    # Optimised preset with recycling
    python train.py --config optimized --name optimized_v1

    # Quick test (3 epochs, small model)
    python train.py --config small --name test --max-epochs 3

    # With recycling enabled
    python train.py --config default --n-recycles 2 --name recycle_test

    # Resume from checkpoint
    python train.py --resume checkpoints/latest_checkpoint.pt
"""

import argparse
import csv
import os
import sys

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxtal_lm.config import (
    ExperimentConfig,
    get_default_config,
    get_large_config,
    get_optimized_config,
    get_small_config,
    get_small_config_og,
    get_cap_config,
)
from proxtal_lm.data import CrystalContactsDataset, collate_pad
from proxtal_lm.models import CrystalTriangularModel
from proxtal_lm.training import train_one_epoch, validate_one_epoch


# ================================================================
# Utilities
# ================================================================

class LoggerCsv:
    """Simple CSV logger for training metrics."""

    def __init__(self, filepath: str):
        self.log_header: list = []
        self.filepath = filepath
        self.header = not os.path.exists(self.filepath)

    def write(self, cont):
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(cont)

    def write_header(self):
        if self.header:
            self.write(self.log_header)
            self.header = False


def save_checkpoint(model, opt, scaler, scheduler, epoch, save_path, best_val_loss, config):
    """Save a training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(ckpt, save_path)
    print(f"  Checkpoint saved to {save_path}")


def load_checkpoint(model, opt, scaler, scheduler, load_path, device):
    """Load a training checkpoint."""
    print(f"  Loading checkpoint from {load_path}...")
    ckpt = torch.load(load_path, map_location=device, weights_only=False)

    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        print(f"  Warning: {e}\n  Trying to strip module/compile prefixes...")
        new_sd = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        model.load_state_dict(new_sd)

    opt.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed from epoch {start_epoch}, best loss {best_loss:.4f}")
    return start_epoch, best_loss


# ================================================================
# Main Training Function
# ================================================================

def train(config: ExperimentConfig):
    """Run the full training loop."""

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}")
    print(f"\n{config}\n")

    best_val_loss = float("inf")
    start_epoch = 0
    epochs_no_improve = 0

    logger = LoggerCsv(config.training.log_file)

    # --- Model ---
    win = config.model.attention_window_size
    model = CrystalTriangularModel(
        emb_dim=config.model.emb_dim,
        d_model=config.model.d_model,
        d_pair=config.model.d_pair,
        n_seq_layers=config.model.n_seq_layers,
        n_blocks=config.model.n_blocks,
        tri_hidden=config.model.tri_hidden,
        out_ch=config.model.out_ch,
        use_checkpoint=config.model.use_checkpoint,
        n_recycles=config.model.n_recycles,
        max_rel_pos=config.model.max_rel_pos,
        num_space_groups=config.model.num_space_groups,
        n_heads=config.model.n_heads,
        attention_window_size=win if win > 0 else None,
        dropout=config.model.dropout,
        n_hypotheses=config.model.n_hypotheses,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if config.training.use_compile:
        print(f"Compiling model (mode={config.training.compile_mode})...")
        model = torch.compile(model, mode=config.training.compile_mode)
        print("Model compiled!")

    # --- Data ---
    train_ds = CrystalContactsDataset(config.data.train_path, config.data.num_bins)
    val_ds   = CrystalContactsDataset(config.data.val_path,   config.data.num_bins)

    collate_fn = lambda batch: collate_pad(batch, pad_multiple=config.data.pad_multiple)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # --- Optimiser & Scheduler ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if config.training.use_amp else None

    scheduler = None
    if config.training.scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.scheduler_min_lr,
        )
    elif config.training.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=config.training.max_epochs)

    # --- Resume ---
    if config.resume_from and os.path.exists(config.resume_from):
        start_epoch, best_val_loss = load_checkpoint(
            model, opt, scaler, scheduler, config.resume_from, device
        )
        if config.reset_lr_on_resume:
            new_lr = config.training.learning_rate
            for pg in opt.param_groups:
                pg["lr"] = new_lr
            print(f"  LR reset to {new_lr} (overriding checkpoint)")

    # --- Training loop ---
    for epoch in range(start_epoch, config.training.max_epochs):
        torch.cuda.empty_cache()

        train_metrics = train_one_epoch(
            model, train_loader, opt, device,
            scaler=scaler,
            grad_clip=config.training.grad_clip,
            accum_steps=config.training.accum_steps,
            matching_mode=config.training.matching_mode,
            crystal_og_weight=config.training.crystal_og_weight,
            diversity_weight=config.training.diversity_weight,
            crystal_contact_weight=config.training.crystal_contact_weight,
            contact_threshold=config.training.contact_threshold,
            contact_emphasis=config.training.contact_emphasis,
        )

        val_metrics = validate_one_epoch(model, val_loader, device)

        val_loss = val_metrics.get("val_loss", val_metrics.get("loss"))
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        cur_epoch = epoch + 1

        if config.training.verbose:
            print(f"\n======== EPOCH {cur_epoch}/{config.training.max_epochs} ========")
            print(f"LR: {current_lr:.8f}")
            print(f"Train: {train_metrics}")
            print(f"Val:   {val_metrics}")

        # CSV logging
        if logger.header:
            hdr = ["epoch", "lr"]
            hdr.extend(train_metrics.keys())
            hdr.extend(val_metrics.keys())
            hdr.append("best_val_loss")
            logger.log_header = hdr
            logger.write_header()

        row = [cur_epoch, current_lr]
        row.extend(train_metrics.values())
        row.extend(val_metrics.values())
        row.append(best_val_loss)
        logger.write(row)

        # Checkpointing
        latest = os.path.join(config.training.checkpoint_dir, "latest_checkpoint.pt")
        save_checkpoint(model, opt, scaler, scheduler, cur_epoch, latest, best_val_loss, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best = os.path.join(config.training.checkpoint_dir, "best_checkpoint.pt")
            save_checkpoint(model, opt, scaler, scheduler, cur_epoch, best, best_val_loss, config)
            print(f"  New best model (val loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{config.training.early_stop_patience} epochs.")

        if config.training.save_every_n_epochs > 0 and cur_epoch % config.training.save_every_n_epochs == 0:
            p = os.path.join(config.training.checkpoint_dir, f"checkpoint_epoch_{cur_epoch}.pt")
            save_checkpoint(model, opt, scaler, scheduler, cur_epoch, p, best_val_loss, config)

        if epochs_no_improve >= config.training.early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ProXtal-LM V2 model")

    # Configuration preset
    parser.add_argument(
        "--config", type=str, default="default",
        choices=["default", "small", "large", "optimized", "cap","small_og"],
        help="Preset configuration",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")

    # Training overrides
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)

    # Model overrides
    parser.add_argument("--n-recycles", type=int, default=None, help="Number of recycling iterations")
    parser.add_argument("--window-size", type=int, default=None, help="Axial attention window size (0=full)")
    parser.add_argument("--n-hypotheses", type=int, default=None, help="Number of output hypotheses (1=single)")

    # Resumption
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset-lr", action="store_true",
                        help="Override saved LR with config LR on resume")

    # Compilation
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    # Matching
    parser.add_argument(
        "--matching-mode", type=str, default=None,
        choices=["none", "hungarian", "greedy"],
        help="Multi-target matching mode (default: none)",
    )
    parser.add_argument("--crystal-og-weight", type=float, default=None,
                        help="Weight for original-structure loss (unmatched hyps)")
    parser.add_argument("--diversity-weight", type=float, default=None,
                        help="Weight for diversity loss")
    parser.add_argument("--crystal-contact-weight", type=float, default=None,
                        help="Weight for universal crystal contact loss (ALL hyps)")
    parser.add_argument("--contact-threshold", type=float, default=None,
                        help="KL threshold to detect contact positions")
    parser.add_argument("--contact-emphasis", type=float, default=None,
                        help="Weight multiplier for contacts in matched loss (1.0=uniform)")

    # LR scheduler
    parser.add_argument("--scheduler-min-lr", type=float, default=None,
                        help="Minimum learning rate for plateau scheduler")
    parser.add_argument("--scheduler-factor", type=float, default=None,
                        help="Factor for plateau scheduler LR reduction")
    parser.add_argument("--scheduler-patience", type=int, default=None,
                        help="Patience for plateau scheduler")

    args = parser.parse_args()

    # Get preset
    configs = {
        "default": get_default_config,
        "small": get_small_config,
        "small_og": get_small_config_og,
        "large": get_large_config,
        "optimized": get_optimized_config,
        "cap": get_cap_config,
    }
    config = configs[args.config]()

    # Apply overrides
    if args.resume:
        config.resume_from = args.resume
    if args.reset_lr:
        config.reset_lr_on_resume = True
    if args.name:
        config.name = args.name
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.accum_steps is not None:
        config.training.accum_steps = args.accum_steps
    if args.n_recycles is not None:
        config.model.n_recycles = args.n_recycles
    if args.window_size is not None:
        config.model.attention_window_size = args.window_size
    if args.n_hypotheses is not None:
        config.model.n_hypotheses = args.n_hypotheses
    if args.no_compile:
        config.training.use_compile = False
    if args.matching_mode is not None:
        config.training.matching_mode = args.matching_mode
    if args.crystal_og_weight is not None:
        config.training.crystal_og_weight = args.crystal_og_weight
    if args.diversity_weight is not None:
        config.training.diversity_weight = args.diversity_weight
    if args.crystal_contact_weight is not None:
        config.training.crystal_contact_weight = args.crystal_contact_weight
    if args.contact_threshold is not None:
        config.training.contact_threshold = args.contact_threshold
    if args.contact_emphasis is not None:
        config.training.contact_emphasis = args.contact_emphasis
    if args.scheduler_min_lr is not None:
        config.training.scheduler_min_lr = args.scheduler_min_lr
    if args.scheduler_factor is not None:
        config.training.scheduler_factor = args.scheduler_factor
    if args.scheduler_patience is not None:
        config.training.scheduler_patience = args.scheduler_patience

    config.__post_init__()
    train(config)


if __name__ == "__main__":
    main()
