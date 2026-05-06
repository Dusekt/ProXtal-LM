#!/usr/bin/env python3
"""
Tests for multi-target matching loss and LR scheduler configuration.

Validates:
1. Greedy matching selects the best (lowest-loss) target
2. Hungarian matching produces valid assignments
3. crystal_og and diversity loss terms work
4. Single-target fallback still works (matching_mode="none")
5. LR scheduler config defaults are updated
6. Multi-target collation works correctly
7. Memory estimate for A100 80GB feasibility
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from proxtal_lm.utils import (
    TargetSmoother,
    get_distogram_loss,
    get_multi_target_distogram_loss,
    get_loss_from_multiple_targets,
    make_pair_mask,
)
from proxtal_lm.config import (
    TrainingConfig,
    get_default_config,
    get_optimized_config,
)


def _make_dummy_data(B=2, L=16, S=64, C=3):
    """Create dummy data for testing multi-target loss."""
    logits = torch.randn(B, L, L, S)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)

    # Create C different targets per sample; one is "close" to prediction
    targets = torch.randint(0, S, (B, C, L, L))

    # Make first target close to the argmax of logits
    best_bins = logits.argmax(dim=-1)  # [B, L, L]
    targets[:, 0] = best_bins  # first target matches prediction well

    target_mask = torch.ones(B, C, dtype=torch.bool)
    return logits, targets, pair_mask, target_mask


def test_greedy_selects_best_target():
    """Greedy mode should pick the target with lowest CE loss."""
    print("Testing greedy matching...")

    B, L, S, C = 2, 16, 64, 3
    logits, targets, pair_mask, target_mask = _make_dummy_data(B, L, S, C)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)  # [B, C, L, L, S]

    # Greedy loss
    greedy_loss = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy"
    )

    # Manually compute per-target losses to verify greedy picks minimum
    log_probs = F.log_softmax(logits, dim=-1)
    per_target_losses = []
    for c in range(C):
        ce = -(soft_targets[:, c] * log_probs).sum(dim=-1)
        ce = ce * pair_mask.float()
        per_sample = ce.sum(dim=(1, 2)) / pair_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
        per_target_losses.append(per_sample)

    stacked = torch.stack(per_target_losses, dim=1)  # [B, C]
    expected = stacked.min(dim=1).values.mean()

    assert torch.allclose(greedy_loss, expected, atol=1e-5), (
        f"Greedy loss {greedy_loss:.6f} != expected min {expected:.6f}"
    )
    print(f"  ✓ Greedy loss = {greedy_loss:.6f} matches manual min = {expected:.6f}")


def test_hungarian_returns_valid_loss():
    """Hungarian mode should return a valid scalar loss."""
    print("Testing hungarian matching...")

    B, L, S, C = 2, 16, 64, 3
    logits, targets, pair_mask, target_mask = _make_dummy_data(B, L, S, C)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    loss = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="hungarian"
    )

    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss > 0, f"Expected positive loss, got {loss}"
    print(f"  ✓ Hungarian loss = {loss:.6f} (valid scalar)")


def test_greedy_leq_single_worst():
    """Greedy loss should be ≤ worst single-target loss."""
    print("Testing greedy ≤ worst target...")

    B, L, S, C = 2, 16, 64, 3
    logits, targets, pair_mask, target_mask = _make_dummy_data(B, L, S, C)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    greedy_loss = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy"
    )

    # Worst single-target loss
    worst_loss = -float("inf")
    for c in range(C):
        single = get_distogram_loss(logits, soft_targets[:, c], pair_mask)
        worst_loss = max(worst_loss, single.item())

    assert greedy_loss.item() <= worst_loss + 1e-5, (
        f"Greedy {greedy_loss:.6f} > worst single {worst_loss:.6f}"
    )
    print(f"  ✓ Greedy {greedy_loss:.4f} ≤ worst {worst_loss:.4f}")


def test_crystal_og_loss():
    """crystal_og loss term should increase total loss."""
    print("Testing crystal_og loss term...")

    B, L, S, C = 2, 16, 64, 2
    logits, targets, pair_mask, target_mask = _make_dummy_data(B, L, S, C)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    og_target = torch.randint(0, S, (B, L, L))
    soft_og = smoother(og_target)

    loss_no_og = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        crystal_og_weight=0.0,
    )
    loss_with_og = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        soft_target_og=soft_og, crystal_og_weight=0.1,
    )

    assert loss_with_og > loss_no_og, (
        f"crystal_og should increase loss: {loss_with_og:.6f} vs {loss_no_og:.6f}"
    )
    print(f"  ✓ With og: {loss_with_og:.4f} > without: {loss_no_og:.4f}")


def test_diversity_loss():
    """Diversity loss should change total loss."""
    print("Testing diversity loss term...")

    B, L, S, C = 2, 16, 64, 2
    logits, targets, pair_mask, target_mask = _make_dummy_data(B, L, S, C)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    loss_no_div = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        diversity_weight=0.0,
    )
    loss_with_div = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        diversity_weight=0.01,
    )

    assert not torch.allclose(loss_no_div, loss_with_div), (
        "Diversity term should change the total loss"
    )
    print(f"  ✓ No div: {loss_no_div:.4f}, with div: {loss_with_div:.4f}")


def test_partial_target_mask():
    """Should handle samples with different numbers of valid targets."""
    print("Testing partial target mask...")

    B, L, S, C = 3, 16, 64, 4
    logits, targets, pair_mask, target_mask = _make_dummy_data(B, L, S, C)

    # Sample 0: all 4 targets, sample 1: 2 targets, sample 2: 1 target
    target_mask[1, 2:] = False
    target_mask[2, 1:] = False

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    loss = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy"
    )
    assert torch.isfinite(loss), f"Loss not finite with partial mask: {loss}"
    print(f"  ✓ Partial mask loss = {loss:.6f}")


def test_single_target_fallback():
    """With C=1 and greedy, should match single-target loss exactly."""
    print("Testing single-target fallback...")

    B, L, S = 2, 16, 64
    logits = torch.randn(B, L, L, S)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    target = torch.randint(0, S, (B, L, L))

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_target = smoother(target)

    # Single-target loss
    single_loss = get_distogram_loss(logits, soft_target, pair_mask)

    # Multi-target with C=1
    soft_multi = soft_target.unsqueeze(1)  # [B, 1, L, L, S]
    t_mask = torch.ones(B, 1, dtype=torch.bool)
    multi_loss = get_multi_target_distogram_loss(
        logits, soft_multi, pair_mask, t_mask, mode="greedy"
    )

    assert torch.allclose(single_loss, multi_loss, atol=1e-5), (
        f"Single {single_loss:.6f} != multi(C=1) {multi_loss:.6f}"
    )
    print(f"  ✓ Single {single_loss:.6f} == multi(C=1) {multi_loss:.6f}")


def test_alias():
    """get_loss_from_multiple_targets should be an alias."""
    print("Testing alias...")
    assert get_loss_from_multiple_targets is get_multi_target_distogram_loss
    print("  ✓ Alias works")


def test_lr_scheduler_defaults():
    """LR scheduler defaults should be less aggressive."""
    print("Testing LR scheduler defaults...")

    tc = TrainingConfig()
    assert tc.scheduler_factor == 0.8, f"Expected 0.8, got {tc.scheduler_factor}"
    assert tc.scheduler_patience == 5, f"Expected 5, got {tc.scheduler_patience}"
    assert tc.scheduler_min_lr == 1e-7, f"Expected 1e-7, got {tc.scheduler_min_lr}"
    print(f"  ✓ factor={tc.scheduler_factor}, patience={tc.scheduler_patience}, "
          f"min_lr={tc.scheduler_min_lr}")


def test_matching_config_defaults():
    """Matching config defaults should be 'none' (backward-compat)."""
    print("Testing matching config defaults...")

    tc = TrainingConfig()
    assert tc.matching_mode == "none", f"Expected 'none', got {tc.matching_mode}"
    assert tc.crystal_og_weight == 0.0
    assert tc.diversity_weight == 0.0
    print("  ✓ Defaults: matching_mode='none', og_weight=0.0, div_weight=0.0")


def test_memory_estimate():
    """Estimate memory for multi-target on A100 80GB."""
    print("Testing memory estimate...")

    # Typical sizes
    B, L, S, C = 6, 400, 64, 3  # batch=6, seq=400, 3 polymorphs
    logits_bytes = B * L * L * S * 4       # float32
    targets_bytes = B * C * L * L * S * 4  # soft targets
    mask_bytes = B * L * L * 1

    total_mb = (logits_bytes + targets_bytes + mask_bytes) / (1024 ** 2)
    model_mb = 867_264 * 4 / (1024 ** 2)   # small model params
    # For d_model=256 model: ~14M params
    model_mb_256 = 14_000_000 * 4 / (1024 ** 2)

    total_with_model = total_mb + model_mb_256
    overhead_factor = 3  # optimizer states, gradients, activations
    estimated_gb = total_with_model * overhead_factor / 1024

    fits = estimated_gb < 80
    print(f"  Logits:  {logits_bytes / (1024**2):.1f} MB")
    print(f"  Targets: {targets_bytes / (1024**2):.1f} MB")
    print(f"  Model:   {model_mb_256:.1f} MB")
    print(f"  Estimated total (with 3x overhead): {estimated_gb:.1f} GB")
    print(f"  Fits on A100 80GB: {'✓ Yes' if fits else '✗ No'}")
    assert fits, f"Estimated {estimated_gb:.1f} GB exceeds 80 GB"


def test_gradients_flow():
    """Ensure gradients flow through multi-target loss."""
    print("Testing gradient flow...")

    B, L, S, C = 2, 8, 64, 2
    logits = torch.randn(B, L, L, S, requires_grad=True)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    targets = torch.randint(0, S, (B, C, L, L))
    target_mask = torch.ones(B, C, dtype=torch.bool)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    for mode in ["greedy", "hungarian"]:
        logits.grad = None
        loss = get_multi_target_distogram_loss(
            logits, soft_targets, pair_mask, target_mask, mode=mode
        )
        loss.backward()
        assert logits.grad is not None, f"No gradient for mode={mode}"
        assert logits.grad.abs().sum() > 0, f"Zero gradient for mode={mode}"
        print(f"  ✓ Gradients flow for mode={mode}")


def main():
    print("=" * 60)
    print("Multi-Target Matching Loss Tests")
    print("=" * 60)
    print()

    tests = [
        ("Greedy selects best",     test_greedy_selects_best_target),
        ("Hungarian valid loss",    test_hungarian_returns_valid_loss),
        ("Greedy ≤ worst",          test_greedy_leq_single_worst),
        ("crystal_og loss",         test_crystal_og_loss),
        ("Diversity loss",          test_diversity_loss),
        ("Partial target mask",     test_partial_target_mask),
        ("Single-target fallback",  test_single_target_fallback),
        ("Alias",                   test_alias),
        ("LR scheduler defaults",   test_lr_scheduler_defaults),
        ("Matching config",         test_matching_config_defaults),
        ("Memory estimate",         test_memory_estimate),
        ("Gradient flow",           test_gradients_flow),
    ]

    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:25s}: {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print(f"\n🎉 All {len(results)} tests passed!")
    else:
        failed = sum(1 for _, p in results if not p)
        print(f"\n⚠️  {failed}/{len(results)} tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
