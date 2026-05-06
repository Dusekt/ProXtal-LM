"""
Crystal-specific validation metrics.

Evaluates model predictions **only** on residue pairs where the crystal
distogram differs from the original (PDB) distogram.  These "changed
positions" represent inter-molecular crystal-packing contacts.

Masking logic (chain-aware)
---------------------------
For crystal-specific metrics the sequence-separation filter is conditioned
on ``chain_id``:

- **Same chain:**      |i − j| ≥ 6  (filter trivial local contacts)
- **Different chain:** no filter     (all inter-chain packing contacts kept)
- **No chain_id:**     no filter     (legacy behaviour — changed positions
  are assumed to be inter-molecular by definition, so applying |i−j| ≥ 6
  would incorrectly mask valid contacts)
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .utils import make_seq_sep_mask


def calculate_crystal_specific_metrics(
    logits: torch.Tensor,
    target_crystal: torch.Tensor,
    target_original: torch.Tensor,
    pair_mask: torch.Tensor,
    lengths: torch.Tensor,
    chain_id: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Metrics focused on crystal-specific changes (positions where crystal ≠ original).

    Args:
        logits:          [B, L, L, S]  raw model logits
        target_crystal:  [B, L, L]     crystal distogram bin indices
        target_original: [B, L, L]     original / PDB distogram bin indices
        pair_mask:       [B, L, L]     padding mask
        lengths:         [B]           original sequence lengths
        chain_id:        [B, L]        chain assignments (optional)

    Returns:
        Dictionary with crystal_only_precision_L/L2/L5,
        crystal_only_recall_L, crystal_only_f1_L, crystal_only_accuracy,
        pct_changed_positions, num_changed_positions
    """
    B, L, _, S = logits.shape
    device = logits.device

    pred_bins = torch.argmax(logits, dim=-1)                          # [B, L, L]
    pred_probs = torch.softmax(logits, dim=-1)
    pred_contact_prob = pred_probs[..., :27].sum(dim=-1)              # [B, L, L]

    triu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    metrics: Dict[str, list] = {
        "crystal_only_precision_L":  [],
        "crystal_only_precision_L2": [],
        "crystal_only_precision_L5": [],
        "crystal_only_recall_L":     [],
        "crystal_only_f1_L":         [],
        "crystal_only_accuracy":     [],
        "pct_changed_positions":     [],
        "num_changed_positions":     [],
    }

    for b in range(B):
        curr_len = lengths[b].item()

        # --- build eval mask ------------------------------------------------
        # When chain_id is available, use chain-aware separation.
        # When absent, skip separation entirely (changed positions are
        # inter-molecular by definition).
        if chain_id is not None:
            cid = chain_id[b]
            sep = make_seq_sep_mask(L, cid, min_sep=6, device=device)
            valid = pair_mask[b] & sep & triu
        else:
            # No chain_id → no sequence separation filter on changed positions
            valid = pair_mask[b] & triu

        # Ensure valid targets
        valid_crystal  = valid & (target_crystal[b]  != -1)
        valid_original = valid & (target_original[b] != -1)
        valid_both     = valid_crystal & valid_original

        if valid_both.sum() == 0:
            continue

        # Extract flat vectors at valid positions
        pred_flat     = pred_bins[b][valid_both]
        pred_prob_flat = pred_contact_prob[b][valid_both]
        crystal_flat  = target_crystal[b][valid_both]
        original_flat = target_original[b][valid_both]

        # *** Changed positions: crystal ≠ original ***
        changed = crystal_flat != original_flat
        num_changed = int(changed.sum().item())
        num_total   = int(valid_both.sum().item())

        if num_changed == 0:
            continue

        pred_changed      = pred_flat[changed]
        pred_prob_changed = pred_prob_flat[changed]
        crystal_changed   = crystal_flat[changed]

        # 1. Bin accuracy
        acc = (pred_changed == crystal_changed).float().mean().item()
        metrics["crystal_only_accuracy"].append(acc)

        # 2. Contact classification on changed positions
        is_contact = crystal_changed < 27
        n_true = int(is_contact.sum().item())

        # Precision @ L, L/2, L/5
        sorted_idx = torch.argsort(pred_prob_changed, descending=True)
        sorted_contacts = is_contact[sorted_idx]

        for fac, name in [
            (1.0, "crystal_only_precision_L"),
            (0.5, "crystal_only_precision_L2"),
            (0.2, "crystal_only_precision_L5"),
        ]:
            k = max(1, min(int(curr_len * fac), len(sorted_contacts)))
            tp = sorted_contacts[:k].sum().item()
            metrics[name].append(tp / k)

        # Recall @ L  &  F1 @ L
        k_L = max(1, min(int(curr_len), len(sorted_contacts)))
        tp_L   = sorted_contacts[:k_L].sum().item()
        prec_L = tp_L / k_L
        rec_L  = tp_L / n_true if n_true > 0 else 0.0
        f1_L   = 2 * prec_L * rec_L / (prec_L + rec_L) if (prec_L + rec_L) > 0 else 0.0
        metrics["crystal_only_recall_L"].append(rec_L)
        metrics["crystal_only_f1_L"].append(f1_L)

        # 3. Statistics
        metrics["pct_changed_positions"].append(100.0 * num_changed / num_total)
        metrics["num_changed_positions"].append(num_changed)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def calculate_comprehensive_metrics(
    logits: torch.Tensor,
    target_crystal: torch.Tensor,
    target_original: torch.Tensor,
    pair_mask: torch.Tensor,
    lengths: torch.Tensor,
    chain_id: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Combined standard + crystal-specific metrics.

    Standard metrics evaluate all crystal contacts (with chain-aware |i−j| ≥ 6).
    Crystal-specific metrics evaluate only changed positions.

    Args:
        logits, target_crystal, target_original, pair_mask, lengths, chain_id:
            Same as ``calculate_crystal_specific_metrics``.

    Returns:
        Combined dictionary with ``all_crystal_*`` and ``crystal_only_*`` keys.
    """
    from .utils import calculate_metrics

    standard = calculate_metrics(logits, target_crystal, pair_mask, lengths, chain_id=chain_id)

    combined: Dict[str, float] = {}
    for k, v in standard.items():
        combined[f"all_crystal_{k}"] = v

    crystal = calculate_crystal_specific_metrics(
        logits, target_crystal, target_original, pair_mask, lengths, chain_id=chain_id,
    )
    combined.update(crystal)

    return combined


def calculate_per_hypothesis_metrics(
    logits: torch.Tensor,
    all_targets: torch.Tensor,
    pair_mask: torch.Tensor,
    target_mask: torch.Tensor,
    target_gen,
    max_hyp: int = 3,
    max_tgt: int = 3,
) -> Dict[str, float]:
    """
    Screen each hypothesis against each target for validation.

    Computes CE loss for hypothesis *n* vs target *k* (up to ``max_hyp``
    hypotheses and ``max_tgt`` targets), plus the best-target loss for
    each hypothesis.

    Args:
        logits:       [B, N, L, L, S]  multi-hypothesis logits
        all_targets:  [B, K, L, L]     integer target bin indices
        pair_mask:    [B, L, L]        boolean mask
        target_mask:  [B, K]           which targets are valid
        target_gen:   TargetSmoother instance
        max_hyp:      Maximum number of hypotheses to screen (default 3)
        max_tgt:      Maximum number of targets to screen (default 3)

    Returns:
        Dictionary with keys like ``hyp0_tgt0_loss``, ``hyp0_tgt1_loss``,
        ``hyp0_best_loss``, etc.
    """
    B, N = logits.shape[:2]
    K = all_targets.shape[1]
    N_use = min(N, max_hyp)
    K_use = min(K, max_tgt)

    soft_targets = target_gen(all_targets[:, :K_use])              # [B, K_use, L, L, S]
    mask_f = pair_mask.float()
    denom = mask_f.sum(dim=(1, 2)).clamp(min=1.0)                  # [B]

    metrics: Dict[str, float] = {}

    for n in range(N_use):
        log_p = F.log_softmax(logits[:, n], dim=-1)                # [B, L, L, S]
        best_loss = float("inf")

        for k in range(K_use):
            # Only count samples where target k is valid
            valid = target_mask[:, k]                              # [B]
            if valid.sum() == 0:
                continue

            ce = -(soft_targets[:, k] * log_p).sum(dim=-1)         # [B, L, L]
            ce = ce * mask_f
            per_sample = ce.sum(dim=(1, 2)) / denom                # [B]
            mean_loss = per_sample[valid].mean().item()

            metrics[f"hyp{n}_tgt{k}_loss"] = mean_loss
            best_loss = min(best_loss, mean_loss)

        if best_loss < float("inf"):
            metrics[f"hyp{n}_best_loss"] = best_loss

    return metrics
