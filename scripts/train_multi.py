#!/usr/bin/env python3
"""
Multi-GPU training script for ProXtal-LM V2 using DistributedDataParallel.

Usage examples:
    # Train with 8 GPUs (default)
    torchrun --nproc_per_node=8 train_multi.py

    # Train with specific number of GPUs
    torchrun --nproc_per_node=4 train_multi.py

    # With configuration preset
    torchrun --nproc_per_node=8 train_multi.py --config optimized --name multi_gpu_run

    # Quick test
    torchrun --nproc_per_node=8 train_multi.py --config small --max-epochs 3

    # Resume from checkpoint
    torchrun --nproc_per_node=8 train_multi.py --resume checkpoints/latest_checkpoint.pt

Note:
    - Batch size specified is PER GPU. Total effective batch size = batch_size * num_gpus
    - Learning rate is automatically scaled by the number of GPUs (linear scaling rule)
    - Only rank 0 saves checkpoints and prints logs
"""

import argparse
import csv
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
# Distributed Training Utilities
# ================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    # Get rank and world_size from environment variables (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


def all_reduce_tensor(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Average a tensor across all processes."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


# ================================================================
# Utilities
# ================================================================

class LoggerCsv:
    """Simple CSV logger for training metrics (only writes on rank 0)."""

    def __init__(self, filepath: str, rank: int):
        self.log_header: list = []
        self.filepath = filepath
        self.header = not os.path.exists(self.filepath)
        self.rank = rank

    def write(self, cont):
        if not is_main_process(self.rank):
            return
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(cont)

    def write_header(self):
        if self.header:
            self.write(self.log_header)
            self.header = False


def save_checkpoint(model, opt, scaler, scheduler, epoch, save_path, best_val_loss, config, rank):
    """Save a training checkpoint (only on rank 0)."""
    if not is_main_process(rank):
        return

    # Get unwrapped model state dict
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": opt.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    torch.save(ckpt, save_path)
    print(f"  Checkpoint saved to {save_path}")


def load_checkpoint(model, opt, scaler, scheduler, load_path, device, rank):
    """Load a training checkpoint."""
    if is_main_process(rank):
        print(f"  Loading checkpoint from {load_path}...")

    # Load on all ranks (or just rank 0 and broadcast)
    ckpt = torch.load(load_path, map_location=device, weights_only=False)

    # Handle DDP-wrapped model
    try:
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        if is_main_process(rank):
            print(f"  Warning: {e}\n  Trying to strip module/compile prefixes...")
        new_sd = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        if isinstance(model, DDP):
            model.module.load_state_dict(new_sd)
        else:
            model.load_state_dict(new_sd)

    opt.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    best_loss = ckpt.get("best_val_loss", float("inf"))

    if is_main_process(rank):
        print(f"  Resumed from epoch {start_epoch}, best loss {best_loss:.4f}")

    return start_epoch, best_loss


# ================================================================
# Modified Training Functions for DDP
# ================================================================

def train_one_epoch_ddp(
    model, train_loader, optimizer, device, sampler, rank, **kwargs
):
    """Train one epoch with DDP - set epoch on sampler before training."""
    # Important: set epoch for proper shuffling in DistributedSampler
    sampler.set_epoch(kwargs.get("epoch", 0))

    # Call original training function
    metrics = train_one_epoch(
        model, train_loader, optimizer, device,
        scaler=kwargs.get("scaler"),
        grad_clip=kwargs.get("grad_clip", 1.0),
        accum_steps=kwargs.get("accum_steps", 1),
        matching_mode=kwargs.get("matching_mode", "none"),
        crystal_og_weight=kwargs.get("crystal_og_weight", 1.0),
        diversity_weight=kwargs.get("diversity_weight", 0.0),
        crystal_contact_weight=kwargs.get("crystal_contact_weight", 0.0),
        contact_threshold=kwargs.get("contact_threshold", 0.5),
        contact_emphasis=kwargs.get("contact_emphasis", 1.0),
    )

    return metrics


def validate_one_epoch_ddp(model, val_loader, device, rank, world_size):
    """Validate one epoch with DDP - aggregate metrics across all GPUs."""
    # Run validation
    metrics = validate_one_epoch(model, val_loader, device)

    # Aggregate metrics across all processes
    if dist.is_initialized():
        # Convert metrics to tensors for all_reduce
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                metrics[key] = (tensor / world_size).item()

    return metrics


# ================================================================
# Main Training Function
# ================================================================

def train(config: ExperimentConfig, rank: int, local_rank: int, world_size: int):
    """Run the full training loop with DDP."""

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process(rank):
        print(f"Starting distributed training on {world_size} GPUs")
        print(f"\n{config}\n")

    # Synchronize all processes at the start
    dist.barrier()

    best_val_loss = float("inf")
    start_epoch = 0
    epochs_no_improve = 0

    logger = LoggerCsv(config.training.log_file, rank)

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

    # Compile before wrapping in DDP
    if config.training.use_compile:
        if is_main_process(rank):
            print(f"Compiling model (mode={config.training.compile_mode})...")
        model = torch.compile(model, mode=config.training.compile_mode)
        if is_main_process(rank):
            print("Model compiled!")

    # Wrap model with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=config.training.find_unused_params,
    )

    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process(rank):
        print(f"Model parameters: {n_params:,}")
        print(f"Effective batch size: {config.data.batch_size * world_size}")

    # --- Data ---
    train_ds = CrystalContactsDataset(config.data.train_path, config.data.num_bins)
    val_ds   = CrystalContactsDataset(config.data.val_path,   config.data.num_bins)

    collate_fn = lambda batch: collate_pad(batch, pad_multiple=config.data.pad_multiple)

    # Use DistributedSampler for proper data distribution
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,  # Drop last batch for consistent batch sizes
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )

    if is_main_process(rank):
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        print(f"Samples per GPU - Train: {len(train_sampler)}, Val: {len(val_sampler)}")

    # --- Optimiser & Scheduler ---
    # Scale learning rate by world_size (linear scaling rule)
    base_lr = config.training.learning_rate
    scaled_lr = base_lr * world_size

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
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
            min_lr=config.training.scheduler_min_lr * world_size,  # Scale min_lr too
        )
    elif config.training.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=config.training.max_epochs)

    if is_main_process(rank):
        print(f"Learning rate scaled: {base_lr} -> {scaled_lr} (x{world_size})")

    # --- Resume ---
    if config.resume_from and os.path.exists(config.resume_from):
        start_epoch, best_val_loss = load_checkpoint(
            model, opt, scaler, scheduler, config.resume_from, device, rank
        )
        if config.reset_lr_on_resume:
            new_lr = config.training.learning_rate * world_size
            for pg in opt.param_groups:
                pg["lr"] = new_lr
            if is_main_process(rank):
                print(f"  LR reset to {new_lr} (overriding checkpoint)")

    # Sync after loading checkpoint
    dist.barrier()

    # --- Training loop ---
    for epoch in range(start_epoch, config.training.max_epochs):
        torch.cuda.empty_cache()

        train_metrics = train_one_epoch_ddp(
            model, train_loader, opt, device, train_sampler, rank,
            epoch=epoch,
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

        val_metrics = validate_one_epoch_ddp(model, val_loader, device, rank, world_size)

        val_loss = val_metrics.get("val_loss", val_metrics.get("loss"))
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        cur_epoch = epoch + 1

        if config.training.verbose and is_main_process(rank):
            print(f"\n======== EPOCH {cur_epoch}/{config.training.max_epochs} ========")
            print(f"LR: {current_lr:.8f}")
            print(f"Train: {train_metrics}")
            print(f"Val:   {val_metrics}")

        # CSV logging (only on rank 0)
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

        # Checkpointing (only on rank 0)
        latest = os.path.join(config.training.checkpoint_dir, "latest_checkpoint.pt")
        save_checkpoint(model, opt, scaler, scheduler, cur_epoch, latest, best_val_loss, config, rank)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best = os.path.join(config.training.checkpoint_dir, "best_checkpoint.pt")
            save_checkpoint(model, opt, scaler, scheduler, cur_epoch, best, best_val_loss, config, rank)
            if is_main_process(rank):
                print(f"  New best model (val loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if is_main_process(rank):
                print(f"No improvement for {epochs_no_improve}/{config.training.early_stop_patience} epochs.")

        if config.training.save_every_n_epochs > 0 and cur_epoch % config.training.save_every_n_epochs == 0:
            p = os.path.join(config.training.checkpoint_dir, f"checkpoint_epoch_{cur_epoch}.pt")
            save_checkpoint(model, opt, scaler, scheduler, cur_epoch, p, best_val_loss, config, rank)

        if epochs_no_improve >= config.training.early_stop_patience:
            if is_main_process(rank):
                print("Early stopping triggered.")
            break

        # Sync before next epoch
        dist.barrier()

    if is_main_process(rank):
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ProXtal-LM V2 model with multi-GPU DDP")

    # Configuration preset
    parser.add_argument(
        "--config", type=str, default="default",
        choices=["default", "small", "large", "optimized", "cap", "small_og"],
        help="Preset configuration",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")

    # Training overrides
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size PER GPU (total = batch_size * num_gpus)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Base learning rate (will be scaled by num_gpus)")
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

    # DDP-specific options
    parser.add_argument("--find-unused-params", action="store_true",
                        help="Set find_unused_parameters=True in DDP (slower but needed for some models)")

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

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    try:
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
        if args.find_unused_params:
            config.training.find_unused_params = True
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

        # Add find_unused_params to config if not present
        if not hasattr(config.training, 'find_unused_params'):
            config.training.find_unused_params = args.find_unused_params

        config.__post_init__()
        train(config, rank, local_rank, world_size)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()