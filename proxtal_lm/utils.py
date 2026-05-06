"""
Utility functions for training and evaluation.

This module provides:
- TargetSmoother:   GPU-based Gaussian smoothing for target distributions
- Metric_Tracker:   Simple accumulator for training metrics
- make_pair_mask:   Basic pairwise padding mask
- make_seq_sep_mask: Chain-aware sequence-separation mask
- get_distogram_loss: Single-target categorical cross-entropy loss
- get_multi_target_distogram_loss: Multi-target matching loss (hungarian/greedy)
- _crystal_contact_loss: Universal contact loss for ALL hypotheses
- calculate_metrics: Top-K contact-prediction metrics with chain-aware masking
"""

import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Target Smoother (GPU Based)
# ================================================================

class TargetSmoother(nn.Module):
    """
    Converts ground-truth integer bin indices → Gaussian-smoothed soft targets.

    The output is a valid probability distribution (sums to 1 over bins)
    suitable for use with categorical cross-entropy / KL divergence loss.

    Args:
        num_bins:     Number of distance bins (default 64)
        sigma:        Standard deviation of the Gaussian kernel (default 0.8)
        ignore_index: Index value indicating padding / invalid data (default -1)
    """

    def __init__(self, num_bins: int = 64, sigma: float = 0.8, ignore_index: int = -1):
        super().__init__()
        self.num_bins = num_bins
        self.ignore_index = ignore_index

        k_size = int(math.ceil(3 * sigma) * 2 + 1)
        k_range = torch.arange(k_size, dtype=torch.float32) - (k_size - 1) / 2
        kernel = torch.exp(-0.5 * (k_range / sigma) ** 2)
        kernel = kernel / kernel.sum()

        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.pad = k_size // 2

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: arbitrary-shape integer tensor (padded with -1)
                     e.g. [B, L, L] or [B, C, L, L]

        Returns:
            same shape + trailing num_bins dim, float tensor
        """
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1)

        mask = flat_idx != self.ignore_index
        safe_idx = flat_idx.clamp(0, self.num_bins - 1)

        one_hot = F.one_hot(safe_idx, num_classes=self.num_bins).float()
        one_hot = one_hot.unsqueeze(1)
        smoothed = F.conv1d(one_hot, self.kernel, padding=self.pad).squeeze(1)

        # Re-normalise to valid probability distribution
        smoothed_sum = smoothed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        smoothed = smoothed / smoothed_sum

        # Zero out padding positions
        smoothed = smoothed * mask.unsqueeze(-1)

        return smoothed.view(*orig_shape, self.num_bins)


# ================================================================
# Metric Tracker
# ================================================================

class Metric_Tracker:
    """Simple accumulator for training/validation metrics."""

    def __init__(self, prefix: str = "train"):
        self.metrics: Dict[str, float] = defaultdict(float)
        self.prefix = prefix

    def write_metrics(self, d: Dict[str, float]):
        for key, value in d.items():
            self.metrics[f"{self.prefix}_{key}"] += value

    def append_metrics(self, key: str, value: float):
        self.metrics[f"{self.prefix}_{key}"] += value

    def collapse_metrics(self, div: int):
        for key in self.metrics:
            self.metrics[key] /= div


# ================================================================
# Mask Helpers
# ================================================================

def make_pair_mask(seq_mask: torch.Tensor) -> torch.Tensor:
    """
    Create a pairwise padding mask from a 1D sequence mask.

    Args:
        seq_mask: [B, L] boolean (True = valid position)

    Returns:
        [B, L, L] boolean (True where both i and j are valid)
    """
    return seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)


def make_seq_sep_mask(
    L: int,
    chain_id: Optional[torch.Tensor] = None,
    min_sep: int = 6,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Chain-aware sequence-separation mask.

    - **Intra-chain** (same chain_id):  |i − j| ≥ min_sep  (filter trivial local contacts)
    - **Inter-chain** (different chain_id): always True   (all crystal-packing contacts kept)

    When ``chain_id`` is ``None`` all residues are treated as belonging to the
    same chain (standard monomeric behaviour).

    Args:
        L:        Sequence length
        chain_id: [L] integer tensor (optional)
        min_sep:  Minimum sequence separation for intra-chain pairs
        device:   Torch device

    Returns:
        [L, L] boolean mask (True = keep this pair)
    """
    idx = torch.arange(L, device=device)
    sep_mask = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)) >= min_sep  # [L, L]

    if chain_id is not None:
        diff_chain = chain_id.unsqueeze(1) != chain_id.unsqueeze(0)      # [L, L]
        sep_mask = sep_mask | diff_chain  # inter-chain pairs always pass

    return sep_mask


# ================================================================
# Loss Function — Single-Target Categorical Cross-Entropy
# ================================================================

def get_distogram_loss(
    logits: torch.Tensor,
    soft_target: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Categorical cross-entropy between predicted distogram and soft target.

    Applies ``log_softmax`` over the 64 distance bins to produce valid
    log-probability distributions, then computes  −Σ_s  target_s · log p_s.

    Unlike binary cross-entropy, this correctly models the **mutually exclusive**
    nature of distance bins: an atom pair exists at exactly one distance.

    Polymorphs are assumed to have been flattened by the dataloader into
    separate samples, so this function handles a single target per sample.

    Args:
        logits:      [B, L, L, S]  raw model logits
        soft_target: [B, L, L, S]  Gaussian-smoothed target distribution (sums to 1)
        pair_mask:   [B, L, L]     boolean mask for valid positions

    Returns:
        Scalar loss
    """
    log_probs = F.log_softmax(logits, dim=-1)                       # [B, L, L, S]
    ce = -(soft_target * log_probs).sum(dim=-1)                     # [B, L, L]
    ce = ce * pair_mask.float()
    return ce.sum() / pair_mask.float().sum().clamp(min=1.0)


# ================================================================
# Loss Function — Multi-Target Matching
# ================================================================

def _per_target_ce(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample, per-target cross-entropy losses.

    Args:
        log_probs:    [B, L, L, S]  log-softmax of model logits (single hypothesis)
        soft_targets: [B, C, L, L, S]  smoothed targets for C polymorphs
        pair_mask:    [B, L, L]     boolean mask

    Returns:
        [B, C]  mean CE loss per (sample, target)
    """
    B, C = soft_targets.shape[:2]
    mask_f = pair_mask.float()                                     # [B, L, L]
    denom = mask_f.sum(dim=(1, 2)).clamp(min=1.0)                  # [B]

    log_p = log_probs.unsqueeze(1).expand_as(soft_targets)         # [B, C, L, L, S]
    ce = -(soft_targets * log_p).sum(dim=-1)                       # [B, C, L, L]
    ce = ce * mask_f.unsqueeze(1)                                  # apply mask
    return ce.sum(dim=(2, 3)) / denom.unsqueeze(1)                 # [B, C]


def _per_target_ce_contact_weighted(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    soft_target_og: torch.Tensor,
    pair_mask: torch.Tensor,
    target_mask: torch.Tensor,
    contact_emphasis: float = 2.0,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Compute per-sample, per-target CE losses with contact position emphasis.

    Contact positions (where polymorph differs from original) are weighted
    more heavily, encouraging matched hypotheses to focus on crystal contacts.

    Args:
        log_probs:       [B, L, L, S]  log-softmax of model logits (single hypothesis)
        soft_targets:    [B, K, L, L, S]  smoothed targets for K polymorphs
        soft_target_og:  [B, L, L, S]  smoothed original/PDB target
        pair_mask:       [B, L, L]     boolean mask
        target_mask:     [B, K]        which targets are valid
        contact_emphasis: weight multiplier for contact positions (default 2.0)
        contact_threshold: KL threshold to detect contacts

    Returns:
        [B, K]  weighted mean CE loss per (sample, target)
    """
    B, K = soft_targets.shape[:2]
    L = log_probs.shape[1]
    device = log_probs.device

    mask_f = pair_mask.float()                                     # [B, L, L]

    # --- Detect contact positions for each polymorph ---
    eps = 1e-8
    og_expanded = soft_target_og.unsqueeze(1)                      # [B, 1, L, L, S]
    og_safe = og_expanded.expand(B, K, L, L, -1).clamp(min=eps)    # [B, K, L, L, S] (last dim handled separately)
    og_safe = soft_target_og.unsqueeze(1).expand(B, K, L, L, -1).clamp(min=eps)  # [B, K, L, L, S]
    targets_safe = soft_targets.clamp(min=eps)                     # [B, K, L, L, S]

    # KL divergence per position for each polymorph
    kl_per_polymorph = (soft_targets * (targets_safe.log() - og_safe.log())).sum(dim=-1)  # [B, K, L, L]

    # For each target, its contact positions are where it differs from original
    is_contact = (kl_per_polymorph > contact_threshold) & pair_mask.unsqueeze(1)  # [B, K, L, L]

    # --- Compute weighted CE ---
    log_p = log_probs.unsqueeze(1).expand_as(soft_targets)         # [B, K, L, L, S]
    ce = -(soft_targets * log_p).sum(dim=-1)                       # [B, K, L, L]

    # Weight: contact positions get higher weight
    weight = torch.ones_like(ce)                                   # [B, K, L, L]
    weight = torch.where(is_contact, weight * contact_emphasis, weight)

    # Apply mask and weights
    weighted_ce = ce * weight * mask_f.unsqueeze(1)                # [B, K, L, L]

    # Normalize by weighted count
    weight_sum = (weight * mask_f.unsqueeze(1)).sum(dim=(2, 3)).clamp(min=1.0)  # [B, K]
    weighted_loss = weighted_ce.sum(dim=(2, 3)) / weight_sum       # [B, K]

    # Mask out invalid targets
    weighted_loss = weighted_loss.masked_fill(~target_mask, float('inf'))

    return weighted_loss


def _per_hypothesis_target_ce(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample, per-hypothesis, per-target cross-entropy losses.

    Iterates over N hypotheses (small N, typically 1–5) to avoid the
    memory cost of a fully expanded ``[B, N, K, L, L, S]`` tensor.

    Args:
        log_probs:    [B, N, L, L, S]  log-softmax of N hypothesis logits
        soft_targets: [B, K, L, L, S]  smoothed targets for K polymorphs
        pair_mask:    [B, L, L]        boolean mask

    Returns:
        [B, N, K]  mean CE loss per (sample, hypothesis, target)
    """
    N = log_probs.shape[1]
    # For each hypothesis, reuse _per_target_ce which handles [B, L, L, S]
    slices = []
    for n in range(N):
        slices.append(_per_target_ce(log_probs[:, n], soft_targets, pair_mask))
    return torch.stack(slices, dim=1)                              # [B, N, K]


def _per_hypothesis_target_ce_weighted(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    soft_target_og: torch.Tensor,
    pair_mask: torch.Tensor,
    target_mask: torch.Tensor,
    contact_emphasis: float = 2.0,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Compute per-sample, per-hypothesis, per-target CE with contact emphasis.

    Contact positions (where target differs from original) are weighted
    more heavily, encouraging matched hypotheses to focus on crystal contacts.

    Args:
        log_probs:       [B, N, L, L, S]  log-softmax of N hypothesis logits
        soft_targets:    [B, K, L, L, S]  smoothed targets for K polymorphs
        soft_target_og:  [B, L, L, S]     smoothed original/PDB target
        pair_mask:       [B, L, L]        boolean mask
        target_mask:     [B, K]           which targets are valid
        contact_emphasis: weight multiplier for contact positions
        contact_threshold: KL threshold to detect contacts

    Returns:
        [B, N, K]  weighted mean CE loss per (sample, hypothesis, target)
    """
    N = log_probs.shape[1]
    # For each hypothesis, compute contact-weighted CE
    slices = []
    for n in range(N):
        weighted_ce = _per_target_ce_contact_weighted(
            log_probs[:, n], soft_targets, soft_target_og, pair_mask, target_mask,
            contact_emphasis=contact_emphasis,
            contact_threshold=contact_threshold,
        )
        slices.append(weighted_ce)
    return torch.stack(slices, dim=1)                              # [B, N, K]


def _diversity_loss(
    log_probs: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Negative-entropy diversity term for a single prediction.

    Encourages the predicted distribution to remain spread out (not
    collapsed into a single bin), acting as a regulariser.

    Args:
        log_probs: [B, L, L, S]
        pair_mask: [B, L, L]

    Returns:
        Scalar diversity loss (lower = more diverse)
    """
    probs = log_probs.exp()                                        # [B, L, L, S]
    entropy = -(probs * log_probs).sum(dim=-1)                     # [B, L, L]
    entropy = entropy * pair_mask.float()
    denom = pair_mask.float().sum().clamp(min=1.0)
    return -entropy.sum() / denom


def _multi_hypothesis_diversity_loss(
    log_probs: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage N hypotheses to produce diverse (different) predictions.

    Computes the negative mean pairwise L2 distance between hypothesis
    probability distributions.  Minimising this loss maximises inter-
    hypothesis distance → discourages mode collapse.

    Args:
        log_probs: [B, N, L, L, S]  log-softmax of N hypothesis logits
        pair_mask: [B, L, L]        boolean mask

    Returns:
        Scalar diversity loss (lower = more diverse)
    """
    B, N = log_probs.shape[:2]
    if N < 2:
        return torch.tensor(0.0, device=log_probs.device)

    all_probs = log_probs.exp()                                    # [B, N, L, L, S]
    mask_f = pair_mask.float()                                     # [B, L, L]
    denom = mask_f.sum(dim=(1, 2)).clamp(min=1.0)                  # [B]

    total_div = torch.tensor(0.0, device=log_probs.device)
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            # L2 distance between probability distributions
            diff = (all_probs[:, i] - all_probs[:, j]).pow(2).sum(dim=-1)  # [B, L, L]
            diff = (diff * mask_f).sum(dim=(1, 2)) / denom                 # [B]
            total_div = total_div + diff.mean()
            count += 1

    return -total_div / count                                      # negative → minimise


def get_multi_target_distogram_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    pair_mask: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str = "greedy",
    soft_target_og: Optional[torch.Tensor] = None,
    crystal_og_weight: float = 0.0,
    diversity_weight: float = 0.0,
    crystal_contact_weight: float = 0.0,
    contact_threshold: float = 0.1,
    contact_emphasis: float = 1.0,
) -> torch.Tensor:
    """
    Multi-target distogram loss with optional matching and contact emphasis.

    Supports both single-hypothesis ``[B, L, L, S]`` and multi-hypothesis
    ``[B, N, L, L, S]`` logits.

    Matching
    --------
    For each sample in the batch, N hypotheses are matched against K valid
    targets.  The cost matrix ``[N, K]`` is computed per sample and solved
    via greedy or Hungarian assignment.

    - **Matched** hypothesis–target pairs receive structural CE loss.
      When ``contact_emphasis > 1.0``, contact positions are weighted more
      heavily, encouraging focus on crystal-packing regions.
    - **Unmatched** hypotheses (N > K) receive ``crystal_og`` loss
      (CE against original PDB distogram) and ``diversity`` loss
      (encourage hypotheses to differ from each other).

    If there are more targets than hypotheses (K > N), N targets are
    randomly sampled per sample so all polymorphs receive training
    signal over time.

    When ``n_hypotheses == 1`` (single-hypothesis) the function degrades
    to the behaviour of the previous version.

    Matching modes
    --------------
    - **greedy**:    Each hypothesis picks its lowest-loss target
                     independently (fast, may double-assign).
    - **hungarian**: Optimal bipartite assignment via the Hungarian
                     algorithm (``scipy.optimize.linear_sum_assignment``).

    Args:
        logits:             [B, L, L, S] or [B, N, L, L, S]  model logits
        soft_targets:       [B, K, L, L, S]  smoothed targets (K polymorphs)
        pair_mask:          [B, L, L]        boolean validity mask
        target_mask:        [B, K]           boolean — True where polymorph k
                            is valid for sample b
        mode:               "greedy" or "hungarian"
        soft_target_og:     [B, L, L, S]     smoothed original/PDB target
                            (optional; used for unmatched hypotheses and contact detection)
        crystal_og_weight:  scalar weight for original-structure loss
        diversity_weight:   scalar weight for diversity loss
        crystal_contact_weight: scalar weight for universal crystal contact loss
                            (applied to ALL hypotheses, teaches about contact positions)
        contact_threshold:  minimum KL divergence to consider position as "contact"
        contact_emphasis:   weight multiplier for contact positions in matched loss
                            (default 1.0 = no emphasis; 2.0 = contacts count 2x)

    Returns:
        Scalar loss
    """
    # --- normalise logits shape ---
    if logits.dim() == 4:
        # Single-hypothesis: [B, L, L, S] → [B, 1, L, L, S]
        logits = logits.unsqueeze(1)

    B, N, L, _, S = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)                      # [B, N, L, L, S]

    # --- N×K cost matrix ---
    # Use contact-weighted CE if emphasis is enabled and we have original target
    if contact_emphasis > 1.0 and soft_target_og is not None:
        # Compute contact-weighted cost matrix for each hypothesis
        ce_matrix = _per_hypothesis_target_ce_weighted(
            log_probs, soft_targets, soft_target_og, pair_mask, target_mask,
            contact_emphasis=contact_emphasis,
            contact_threshold=contact_threshold,
        )
    else:
        ce_matrix = _per_hypothesis_target_ce(                     # [B, N, K]
            log_probs, soft_targets, pair_mask,
        )

    # Mask out invalid targets with a large sentinel
    # target_mask: [B, K] → [B, 1, K]
    ce_matrix = ce_matrix.masked_fill(~target_mask.unsqueeze(1), 1e9)

    # --- matching ---
    if mode == "greedy":
        matched_loss, unmatched_mask = _greedy_match_multi(
            ce_matrix, target_mask,
        )
    elif mode == "hungarian":
        matched_loss, unmatched_mask = _hungarian_match_multi(
            ce_matrix, target_mask,
        )
    else:
        raise ValueError(f"Unknown matching mode: {mode!r}")

    total = matched_loss.mean()

    # --- crystal_og loss on UNMATCHED hypotheses ---
    if crystal_og_weight > 0.0 and soft_target_og is not None and unmatched_mask.any():
        og_loss = _unmatched_og_loss(log_probs, soft_target_og, pair_mask, unmatched_mask)
        total = total + crystal_og_weight * og_loss

    # --- crystal contact loss on ALL hypotheses ---
    # Teaches all hypotheses about positions where crystal packing differs from monomer
    # Matched: focus on non-contact positions (contacts handled by structural loss)
    # Unmatched: explore at contact positions, match original elsewhere
    if crystal_contact_weight > 0.0 and soft_target_og is not None:
        contact_loss = _crystal_contact_loss(
            log_probs, soft_targets, soft_target_og, pair_mask, target_mask,
            unmatched_mask=unmatched_mask,
            contact_threshold=contact_threshold,
        )
        total = total + crystal_contact_weight * contact_loss

    # --- diversity loss between ALL hypotheses ---
    if diversity_weight > 0.0 and N > 1:
        div_loss = _multi_hypothesis_diversity_loss(log_probs, pair_mask)
        total = total + diversity_weight * div_loss

    return total


def _unmatched_og_loss(
    log_probs: torch.Tensor,
    soft_target_og: torch.Tensor,
    pair_mask: torch.Tensor,
    unmatched_mask: torch.Tensor,
) -> torch.Tensor:
    """
    CE loss between unmatched hypotheses and the original PDB distogram.

    Args:
        log_probs:      [B, N, L, L, S]
        soft_target_og: [B, L, L, S]
        pair_mask:      [B, L, L]
        unmatched_mask: [B, N]  True where hypothesis is unmatched

    Returns:
        Scalar loss
    """
    mask_f = pair_mask.float()
    denom = mask_f.sum(dim=(1, 2)).clamp(min=1.0)                  # [B]

    og_exp = soft_target_og.unsqueeze(1)                           # [B, 1, L, L, S]
    ce = -(og_exp * log_probs).sum(dim=-1)                         # [B, N, L, L]
    ce = ce * mask_f.unsqueeze(1)                                  # [B, N, L, L]
    per_hyp = ce.sum(dim=(2, 3)) / denom.unsqueeze(1)              # [B, N]

    # Average only over unmatched hypotheses
    per_hyp = per_hyp * unmatched_mask.float()
    n_unmatched = unmatched_mask.float().sum().clamp(min=1.0)
    return per_hyp.sum() / n_unmatched


def _crystal_contact_loss(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    soft_target_og: torch.Tensor,
    pair_mask: torch.Tensor,
    target_mask: torch.Tensor,
    unmatched_mask: torch.Tensor,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Universal crystal contact loss with different behavior for matched/unmatched.

    Detects "contact positions" where ANY polymorph differs from the original.
    Then applies different strategies:

    - **Matched hypotheses**: At non-contact positions, match original structure.
      (Contact positions are handled by structural loss against their polymorph)

    - **Unmatched hypotheses**: At contact positions, maximize entropy (explore).
      At non-contact positions, match original structure.

    This ensures:
    - Matched hypotheses focus on their specific polymorph (via structural loss)
    - Unmatched hypotheses learn WHERE contacts occur without being forced to
      predict a specific packing (via entropy bonus)

    Args:
        log_probs:       [B, N, L, L, S]  log-softmax of N hypothesis logits
        soft_targets:    [B, K, L, L, S]  smoothed targets for K polymorphs
        soft_target_og:  [B, L, L, S]     smoothed original/PDB target
        pair_mask:       [B, L, L]        boolean validity mask
        target_mask:     [B, K]           boolean — which polymorphs are valid
        unmatched_mask:  [B, N]           boolean — True where hypothesis is unmatched
        contact_threshold: minimum KL divergence to count as "contact position"

    Returns:
        Scalar loss
    """
    B, N, L, _, S = log_probs.shape
    K = soft_targets.shape[1]
    device = log_probs.device

    mask_f = pair_mask.float()                                     # [B, L, L]

    # --- Detect contact positions ---
    # A position is a "crystal contact" if ANY valid polymorph differs from original
    # Use KL divergence: D_KL(polymorph || original) at each position

    og_expanded = soft_target_og.unsqueeze(1)                      # [B, 1, L, L, S]

    # Compute KL divergence for each polymorph vs original
    # KL(P||Q) = sum_s P_s * log(P_s / Q_s)
    # Add small epsilon for numerical stability
    eps = 1e-8
    og_safe = og_expanded.expand(B, K, L, L, S).clamp(min=eps)     # [B, K, L, L, S]
    targets_safe = soft_targets.clamp(min=eps)                     # [B, K, L, L, S]

    # KL divergence per position: sum over bins
    kl_per_polymorph = (soft_targets * (targets_safe.log() - og_safe.log())).sum(dim=-1)  # [B, K, L, L]

    # Mask out invalid polymorphs
    target_mask_exp = target_mask.unsqueeze(-1).unsqueeze(-1)      # [B, K, 1, 1]
    kl_per_polymorph = kl_per_polymorph * target_mask_exp.float()

    # Contact position: ANY valid polymorph has KL > threshold
    max_kl, _ = kl_per_polymorph.max(dim=1)                        # [B, L, L]
    is_contact = (max_kl > contact_threshold) & pair_mask          # [B, L, L]

    if is_contact.sum() == 0:
        return torch.tensor(0.0, device=device)

    # --- Prepare masks ---
    contact_mask_f = is_contact.float()                            # [B, L, L]
    non_contact_mask_f = (~is_contact).float() * mask_f            # [B, L, L]

    # Matched vs unmatched hypothesis masks
    matched_mask_f = (~unmatched_mask).float()                     # [B, N]
    unmatched_mask_f = unmatched_mask.float()                      # [B, N]

    # --- CE to original (used for non-contact positions) ---
    og_ce = -(soft_target_og.unsqueeze(1) * log_probs).sum(dim=-1)  # [B, N, L, L]

    # --- Loss for MATCHED hypotheses ---
    # At non-contact positions: match original (contact positions handled by structural loss)
    if matched_mask_f.sum() > 0:
        og_ce_matched = (og_ce * non_contact_mask_f.unsqueeze(1)).sum(dim=(2, 3))  # [B, N]
        n_non_contact = non_contact_mask_f.sum(dim=(1, 2)).clamp(min=1.0)  # [B]
        og_ce_matched = og_ce_matched / n_non_contact.unsqueeze(1)          # [B, N]
        # Weight by matched status
        matched_loss = (og_ce_matched * matched_mask_f).sum() / matched_mask_f.sum().clamp(min=1.0)
    else:
        matched_loss = torch.tensor(0.0, device=device)

    # --- Loss for UNMATCHED hypotheses ---
    # At contact positions: maximize entropy (explore alternatives)
    # At non-contact positions: match original
    if unmatched_mask_f.sum() > 0:
        # Entropy at contact positions
        probs = log_probs.exp()                                    # [B, N, L, L, S]
        entropy = -(probs * log_probs).sum(dim=-1)                 # [B, N, L, L]

        contact_entropy = (entropy * contact_mask_f.unsqueeze(1)).sum(dim=(2, 3))  # [B, N]
        n_contact = contact_mask_f.sum(dim=(1, 2)).clamp(min=1.0)  # [B]
        contact_entropy = contact_entropy / n_contact.unsqueeze(1) # [B, N]

        # CE to original at non-contact positions
        og_ce_unmatched = (og_ce * non_contact_mask_f.unsqueeze(1)).sum(dim=(2, 3))  # [B, N]
        n_non_contact = non_contact_mask_f.sum(dim=(1, 2)).clamp(min=1.0)  # [B]
        og_ce_unmatched = og_ce_unmatched / n_non_contact.unsqueeze(1)  # [B, N]

        # Combined: minimize (-entropy at contacts + CE at non-contacts)
        # Negative entropy because we want HIGH entropy at contacts
        unmatched_loss_val = (-contact_entropy + og_ce_unmatched)  # [B, N]
        # Weight by unmatched status
        unmatched_loss = (unmatched_loss_val * unmatched_mask_f).sum() / unmatched_mask_f.sum().clamp(min=1.0)
    else:
        unmatched_loss = torch.tensor(0.0, device=device)

    # --- Combine matched and unmatched losses ---
    n_matched = matched_mask_f.sum()
    n_unmatched = unmatched_mask_f.sum()
    total_weight = (n_matched + n_unmatched).clamp(min=1.0)

    total = (matched_loss * n_matched + unmatched_loss * n_unmatched) / total_weight

    return total

def _DISCONTINUED_greedy_match_multi(
    cost_matrix: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple:
    """
    Greedy matching: each hypothesis picks its lowest-cost valid target.

    Args:
        cost_matrix: [B, N, K]
        target_mask: [B, K]

    Returns:
        matched_loss:  [B]    mean matched loss per sample
        unmatched_mask: [B, N] True where hypothesis has no valid match
    """
    B, N, K = cost_matrix.shape
    device = cost_matrix.device

    # For each hypothesis, pick the valid target with lowest cost
    best_per_hyp = cost_matrix.min(dim=2).values                   # [B, N]

    # A hypothesis is "unmatched" if all targets are invalid (all 1e9)
    any_valid = target_mask.any(dim=1)                             # [B]
    unmatched_mask = ~any_valid.unsqueeze(1).expand_as(best_per_hyp)

    # Count valid targets per sample to determine truly unmatched hyps
    n_valid = target_mask.float().sum(dim=1)                       # [B]
    for b in range(B):
        nv = int(n_valid[b].item())
        if nv < N:
            # Mark the (N - nv) hypotheses with highest cost as unmatched
            _, worst_idx = best_per_hyp[b].topk(N - nv, largest=True)
            unmatched_mask[b, worst_idx] = True

    matched_mask = ~unmatched_mask
    # Compute mean matched loss per sample
    matched_vals = best_per_hyp * matched_mask.float()
    n_matched = matched_mask.float().sum(dim=1).clamp(min=1.0)     # [B]
    matched_loss = matched_vals.sum(dim=1) / n_matched             # [B]

    return matched_loss, unmatched_mask

def _greedy_match_multi(
    cost_matrix: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple:
    """
    True Greedy Bipartite Matching.
    Iteratively finds the lowest global cost pair (hypothesis, target), 
    assigns them, and removes them from the available pool.

    Args:
        cost_matrix: [B, N, K]
        target_mask: [B, K]

    Returns:
        matched_loss:  [B]    mean matched loss per sample
        unmatched_mask: [B, N] True where hypothesis has no valid match
    """
    B, N, K = cost_matrix.shape
    device = cost_matrix.device

    matched_losses = []
    unmatched_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    # Clone the cost matrix for searching so we can safely overwrite values 
    # with infinity without breaking the differentiable computation graph
    search_cost = cost_matrix.detach().clone()

    for b in range(B):
        # Identify valid targets
        valid_cols = torch.where(target_mask[b])[0]
        
        if len(valid_cols) == 0:
            matched_losses.append(
                torch.tensor(0.0, device=device, dtype=cost_matrix.dtype)
            )
            continue

        # Mask out invalid targets immediately so they are never picked
        invalid_cols = torch.where(~target_mask[b])[0]
        search_cost[b, :, invalid_cols] = float('inf')

        # We can only make as many matches as the limiting dimension
        n_matches = min(N, len(valid_cols))
        sample_loss = 0.0

        for _ in range(n_matches):
            # Find the flat index of the absolute minimum cost in the remaining matrix
            flat_idx = torch.argmin(search_cost[b])
            
            # Convert the flat index back to 2D coordinates (row, col)
            row = flat_idx // K
            col = flat_idx % K

            # 1. Add to the differentiable loss using the ORIGINAL cost_matrix
            sample_loss += cost_matrix[b, row, col]

            # 2. Mark this hypothesis as successfully matched
            unmatched_mask[b, row] = False

            # 3. Mask out the claimed hypothesis (row) and target (col) 
            # so they cannot be selected in the next iterations
            search_cost[b, row, :] = float('inf')
            search_cost[b, :, col] = float('inf')

        # Calculate the mean loss for this sample
        matched_losses.append(sample_loss / max(n_matches, 1))

    return torch.stack(matched_losses), unmatched_mask

def _hungarian_match_multi(
    cost_matrix: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple:
    """
    Optimal bipartite matching using the Hungarian algorithm.

    Matches N hypotheses to K valid targets.  When K < N the extra
    hypotheses are flagged as unmatched.  When K > N, N targets are
    randomly sampled so all polymorphs receive training signal.

    Gradients flow through the selected cost entries (the assignment
    indices are computed in NumPy and used to gather from the
    differentiable cost tensor).

    Args:
        cost_matrix: [B, N, K]
        target_mask: [B, K]

    Returns:
        matched_loss:  [B]    mean matched loss per sample
        unmatched_mask: [B, N] True where hypothesis is unmatched
    """
    from scipy.optimize import linear_sum_assignment

    B, N, K = cost_matrix.shape
    device = cost_matrix.device
    matched_losses = []
    unmatched_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    cost_np = cost_matrix.detach().cpu().numpy()
    mask_np = target_mask.cpu().numpy()

    for b in range(B):
        valid_cols = np.where(mask_np[b])[0]
        if len(valid_cols) == 0:
            matched_losses.append(
                torch.tensor(0.0, device=device, dtype=cost_matrix.dtype)
            )
            continue

        # If more targets than hypotheses, randomly sample N targets
        if len(valid_cols) > N:
            selected = np.random.choice(valid_cols, size=N, replace=False)
        else:
            selected = valid_cols

        sub_cost = cost_np[b][:, selected]                         # [N, n_sel]
        row_idx, col_idx = linear_sum_assignment(sub_cost)

        # Gather the matched costs from the differentiable tensor
        parts = []
        for r, c in zip(row_idx, col_idx):
            parts.append(cost_matrix[b, r, selected[c]])
            unmatched_mask[b, r] = False

        n_matched = len(parts)
        matched_losses.append(torch.stack(parts).sum() / max(n_matched, 1))

    return torch.stack(matched_losses), unmatched_mask


# Alias for backward-compat with test_installation.py
get_loss_from_multiple_targets = get_multi_target_distogram_loss


# ================================================================
# Evaluation Metrics
# ================================================================

def calculate_metrics(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    pair_mask: torch.Tensor,
    lengths: torch.Tensor,
    chain_id: Optional[torch.Tensor] = None,
    min_seq_sep: int = 6,
) -> Dict[str, float]:
    """
    Top-K contact-prediction metrics with chain-aware sequence separation.

    Contacts: bins < 27 (~8 Å).

    Masking logic (conditioned on ``chain_id``):
      - Same chain:      |i−j| ≥ min_seq_sep  (ignore trivial local contacts)
      - Different chain:  all pairs kept       (valid crystal-packing contacts)
      - No chain_id:     |i−j| ≥ min_seq_sep everywhere (standard monomeric behaviour)

    Args:
        logits:         [B, L, L, S]  model raw logits
        target_indices: [B, L, L]     ground-truth bin indices (−1 = padding)
        pair_mask:      [B, L, L]     boolean padding mask
        lengths:        [B]           original sequence lengths
        chain_id:       [B, L]        chain assignments (optional)
        min_seq_sep:    minimum intra-chain sequence separation

    Returns:
        Dictionary with precision_L, precision_L2, precision_L5,
        recall_L, f1_L, auprc
    """
    B, L, _, S = logits.shape
    device = logits.device

    # Ground truth contacts
    gt_contact = (target_indices >= 0) & (target_indices < 27)       # [B, L, L]

    # Predicted contact probability
    pred_probs = torch.softmax(logits, dim=-1)
    pred_contact = pred_probs[..., :27].sum(dim=-1)                  # [B, L, L]

    # Upper-triangle mask
    triu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    metrics: Dict[str, list] = {
        "precision_L": [], "precision_L2": [], "precision_L5": [],
        "recall_L": [], "f1_L": [], "auprc": [],
    }

    for b in range(B):
        curr_len = lengths[b].item()

        # Chain-aware sequence separation
        cid = chain_id[b] if chain_id is not None else None
        sep_mask = make_seq_sep_mask(L, cid, min_seq_sep, device)

        valid = pair_mask[b] & sep_mask & triu

        flat_pred = pred_contact[b][valid]
        flat_gt   = gt_contact[b][valid].float()

        if flat_gt.numel() == 0 or flat_gt.sum() == 0:
            continue

        sorted_idx = torch.argsort(flat_pred, descending=True)
        sorted_gt  = flat_gt[sorted_idx]
        n_true = int(flat_gt.sum().item())

        # Precision @ L, L/2, L/5
        for fac, name in [(1.0, "precision_L"), (0.5, "precision_L2"), (0.2, "precision_L5")]:
            k = max(1, min(int(curr_len * fac), len(sorted_gt)))
            tp = sorted_gt[:k].sum().item()
            metrics[name].append(tp / k)

        # Recall @ L  &  F1 @ L
        k_L = max(1, min(int(curr_len), len(sorted_gt)))
        tp_L   = sorted_gt[:k_L].sum().item()
        prec_L = tp_L / k_L
        rec_L  = tp_L / n_true if n_true > 0 else 0.0
        f1_L   = 2 * prec_L * rec_L / (prec_L + rec_L) if (prec_L + rec_L) > 0 else 0.0
        metrics["recall_L"].append(rec_L)
        metrics["f1_L"].append(f1_L)

        # AUPRC
        cum_tp = torch.cumsum(sorted_gt, dim=0)
        cum_k  = torch.arange(1, len(sorted_gt) + 1, dtype=torch.float32, device=device)
        precs  = cum_tp / cum_k
        recs   = cum_tp / n_true
        if len(recs) > 1:
            rdiff = recs[1:] - recs[:-1]
            avg_p = (precs[1:] + precs[:-1]) / 2
            auprc = (rdiff * avg_p).sum().item()
        else:
            auprc = precs[0].item() if len(precs) > 0 else 0.0
        metrics["auprc"].append(auprc)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
