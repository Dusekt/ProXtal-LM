"""
Training and validation loops for the crystal distogram model.

Key changes from V1:
- Single-target loss (polymorphs flattened by dataloader)
- Chain-ID and space-group forwarded to model and metrics
- Crystal-specific metrics computed when ``contact_og`` is present
- Multi-hypothesis support: model may return [B, N, L, L, S]
"""

import torch
from torch.optim.lr_scheduler import LambdaLR

from .utils import (
    Metric_Tracker,
    TargetSmoother,
    get_distogram_loss,
    get_multi_target_distogram_loss,
    make_pair_mask,
    calculate_metrics,
)
from .validation_metrics import (
    calculate_crystal_specific_metrics,
    calculate_per_hypothesis_metrics,
)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    scaler=None,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
    scheduler=None,
    matching_mode: str = "none",
    crystal_og_weight: float = 0.0,
    diversity_weight: float = 0.0,
    crystal_contact_weight: float = 0.0,
    contact_threshold: float = 0.1,
    contact_emphasis: float = 1.0,
):
    """
    Train the model for one epoch.

    Args:
        model:       Neural network
        dataloader:  Training DataLoader
        optimizer:   Optimiser (e.g. AdamW)
        device:      CUDA / CPU device
        scaler:      GradScaler for mixed-precision training (None to disable)
        grad_clip:   Gradient clipping threshold (None to disable)
        accum_steps: Number of gradient-accumulation steps
        scheduler:   Optional per-step LR scheduler (e.g. LambdaLR)
        matching_mode:    "none", "hungarian", or "greedy"
        crystal_og_weight: Weight for original-structure loss (unmatched hyps)
        diversity_weight:  Weight for diversity loss
        crystal_contact_weight: Weight for universal contact loss (ALL hyps)
        contact_threshold:  KL threshold to detect contact positions
        contact_emphasis:   Weight multiplier for contacts in matched loss

    Returns:
        Dict of averaged training metrics for the epoch
    """
    torch.cuda.empty_cache()
    tracker = Metric_Tracker(prefix="train")
    model.train()
    num_batches = 0

    target_gen = TargetSmoother(num_bins=64, sigma=0.8, ignore_index=-1).to(device)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        emb     = batch["embedding"].to(device)
        mask    = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        target  = batch["contact"].to(device)          # [B, L, L]

        chain_id = batch.get("chain_id")
        if chain_id is not None:
            chain_id = chain_id.to(device)

        space_group = batch.get("space_group")
        if space_group is not None:
            space_group = space_group.to(device)

        target_og = batch.get("contact_og")
        if target_og is not None:
            target_og = target_og.to(device)

        if mask.sum() == 0:
            continue

        pair_mask = make_pair_mask(mask)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(emb, seq_mask=mask, chain_id=chain_id, space_group=space_group)

            # Determine if model produces multi-hypothesis output
            is_multi_hyp = logits.dim() == 5   # [B, N, L, L, S]

            if matching_mode != "none" and "all_contacts" in batch:
                # --- multi-target matching loss ---
                all_targets = batch["all_contacts"].to(device)   # [B, C, L, L]
                t_mask = batch["target_mask"].to(device)         # [B, C]
                soft_targets = target_gen(all_targets)            # [B, C, L, L, 64]

                soft_og = None
                needs_og = (crystal_og_weight > 0.0 or crystal_contact_weight > 0.0)
                if needs_og and target_og is not None:
                    soft_og = target_gen(target_og)               # [B, L, L, 64]

                total_loss = get_multi_target_distogram_loss(
                    logits, soft_targets, pair_mask, t_mask,
                    mode=matching_mode,
                    soft_target_og=soft_og,
                    crystal_og_weight=crystal_og_weight,
                    diversity_weight=diversity_weight,
                    crystal_contact_weight=crystal_contact_weight,
                    contact_threshold=contact_threshold,
                    contact_emphasis=contact_emphasis,
                )
            else:
                # --- single-target loss (default) ---
                soft_target = target_gen(target)                  # [B, L, L, 64]
                if is_multi_hyp:
                    # Use first hypothesis for single-target loss
                    total_loss = get_distogram_loss(logits[:, 0], soft_target, pair_mask)
                else:
                    total_loss = get_distogram_loss(logits, soft_target, pair_mask)

            total_loss = total_loss / accum_steps

        if scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # --- metrics (use first hypothesis for standard metrics) ---
        metrics_logits = logits[:, 0] if is_multi_hyp else logits
        tracker.append_metrics("loss", total_loss.item() * accum_steps)

        with torch.no_grad():
            metrics = calculate_metrics(
                metrics_logits, target, pair_mask, lengths, chain_id=chain_id,
            )
            tracker.write_metrics(metrics)

            if target_og is not None:
                crystal_metrics = calculate_crystal_specific_metrics(
                    metrics_logits, target, target_og, pair_mask, lengths, chain_id=chain_id,
                )
                tracker.write_metrics(crystal_metrics)

        num_batches += 1

        # --- optimiser step (after accumulation) ---
        if (step + 1) % accum_steps == 0:
            if grad_clip is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and isinstance(scheduler, LambdaLR):
                scheduler.step()

    # Handle leftover partial accumulation batch
    if num_batches > 0 and (step + 1) % accum_steps != 0:
        if grad_clip is not None:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and isinstance(scheduler, LambdaLR):
            scheduler.step()

    tracker.collapse_metrics(max(num_batches, 1))
    return tracker.metrics


@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    """
    Validate the model for one epoch.

    When the model produces multiple hypotheses (``[B, N, L, L, S]``),
    per-hypothesis per-target screening metrics are computed alongside
    the standard metrics (which use the first hypothesis).

    Args:
        model:      Neural network
        dataloader: Validation DataLoader
        device:     CUDA / CPU device

    Returns:
        Dict of averaged validation metrics
    """
    tracker = Metric_Tracker(prefix="val")
    model.eval()
    num_batches = 0

    target_gen = TargetSmoother(num_bins=64, sigma=0.8, ignore_index=-1).to(device)

    for batch in dataloader:
        emb     = batch["embedding"].to(device)
        mask    = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        target  = batch["contact"].to(device)

        chain_id = batch.get("chain_id")
        if chain_id is not None:
            chain_id = chain_id.to(device)

        space_group = batch.get("space_group")
        if space_group is not None:
            space_group = space_group.to(device)

        target_og = batch.get("contact_og")
        if target_og is not None:
            target_og = target_og.to(device)

        if mask.sum() == 0:
            continue

        pair_mask = make_pair_mask(mask)

        with torch.amp.autocast("cuda"):
            logits = model(emb, seq_mask=mask, chain_id=chain_id, space_group=space_group)

            is_multi_hyp = logits.dim() == 5
            metrics_logits = logits[:, 0] if is_multi_hyp else logits

            soft_target = target_gen(target)
            loss = get_distogram_loss(metrics_logits, soft_target, pair_mask)

            metrics = calculate_metrics(
                metrics_logits, target, pair_mask, lengths, chain_id=chain_id,
            )

            if target_og is not None:
                crystal_metrics = calculate_crystal_specific_metrics(
                    metrics_logits, target, target_og, pair_mask, lengths, chain_id=chain_id,
                )
                metrics.update(crystal_metrics)

            # --- per-hypothesis screening ---
            if is_multi_hyp and "all_contacts" in batch:
                all_targets = batch["all_contacts"].to(device)
                t_mask = batch["target_mask"].to(device)
                hyp_metrics = calculate_per_hypothesis_metrics(
                    logits, all_targets, pair_mask, t_mask, target_gen,
                )
                metrics.update(hyp_metrics)

        tracker.write_metrics(metrics)
        tracker.append_metrics("loss", loss.item())
        num_batches += 1

    tracker.collapse_metrics(max(num_batches, 1))
    return tracker.metrics
