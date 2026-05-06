#!/usr/bin/env python3
"""
Tests for multi-target matching loss, multi-hypothesis model, and
LR scheduler configuration.

Validates:
1. Greedy matching selects the best (lowest-loss) target
2. Hungarian matching produces valid assignments
3. crystal_og and diversity loss terms work with multi-hypothesis
4. Single-target fallback still works (matching_mode="none")
5. LR scheduler config defaults are updated
6. Multi-target collation works correctly
7. Memory estimate for A100 80GB feasibility
8. Multi-hypothesis model forward pass
9. N hypotheses × K targets matching
10. Per-hypothesis validation metrics
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
    ModelConfig,
    get_default_config,
    get_optimized_config,
    get_small_config,
    get_large_config,
    get_cap_config,
)
from proxtal_lm.models import CrystalTriangularModel
from proxtal_lm.validation_metrics import calculate_per_hypothesis_metrics


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


def test_crystal_og_loss_multi_hypothesis():
    """crystal_og loss on unmatched hypotheses should increase total loss."""
    print("Testing crystal_og loss (multi-hypothesis)...")

    B, L, S, C, N = 2, 16, 64, 1, 3  # 3 hyp, 1 target → 2 unmatched
    logits = torch.randn(B, N, L, L, S)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    targets = torch.randint(0, S, (B, C, L, L))
    target_mask = torch.ones(B, C, dtype=torch.bool)

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


def test_diversity_loss_multi_hypothesis():
    """Diversity loss between hypotheses should change total loss."""
    print("Testing diversity loss (multi-hypothesis)...")

    B, L, S, C, N = 2, 16, 64, 1, 3
    logits = torch.randn(B, N, L, L, S)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    targets = torch.randint(0, S, (B, C, L, L))
    target_mask = torch.ones(B, C, dtype=torch.bool)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    loss_no_div = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        diversity_weight=0.0,
    )
    loss_with_div = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        diversity_weight=0.1,
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
    """Estimate memory for multi-hypothesis multi-target on A100 80GB."""
    print("Testing memory estimate...")

    # Typical sizes with multi-hypothesis
    B, L, S, C, N = 6, 400, 64, 3, 3  # 3 hypotheses, 3 polymorphs
    logits_bytes = B * N * L * L * S * 4       # float32
    targets_bytes = B * C * L * L * S * 4      # soft targets
    mask_bytes = B * L * L * 1

    total_mb = (logits_bytes + targets_bytes + mask_bytes) / (1024 ** 2)
    # For d_model=256 model: ~14M params + 3 output heads (~24K extra)
    model_mb_256 = 14_100_000 * 4 / (1024 ** 2)

    total_with_model = total_mb + model_mb_256
    overhead_factor = 3  # optimizer states, gradients, activations
    estimated_gb = total_with_model * overhead_factor / 1024

    fits = estimated_gb < 80
    print(f"  Logits (N={N}):  {logits_bytes / (1024**2):.1f} MB")
    print(f"  Targets (C={C}): {targets_bytes / (1024**2):.1f} MB")
    print(f"  Model:           {model_mb_256:.1f} MB")
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


def test_multi_hypothesis_forward():
    """Model with n_hypotheses>1 should return [B, N, L, L, out_ch]."""
    print("Testing multi-hypothesis forward pass...")

    N = 3
    model = CrystalTriangularModel(
        emb_dim=1280, d_model=128, d_pair=32, n_blocks=2,
        tri_hidden=16, out_ch=64, use_checkpoint=False,
        n_hypotheses=N,
    )

    B, L, D = 2, 50, 1280
    emb = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        output = model(emb, seq_mask=mask)

    assert output.shape == (B, N, L, L, 64), (
        f"Expected {(B, N, L, L, 64)}, got {output.shape}"
    )
    print(f"  ✓ Output shape: {output.shape}")


def test_single_hypothesis_backward_compat():
    """Model with n_hypotheses=1 should return [B, L, L, out_ch]."""
    print("Testing single-hypothesis backward compat...")

    model = CrystalTriangularModel(
        emb_dim=1280, d_model=128, d_pair=32, n_blocks=2,
        tri_hidden=16, out_ch=64, use_checkpoint=False,
        n_hypotheses=1,
    )

    B, L, D = 2, 50, 1280
    emb = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        output = model(emb, seq_mask=mask)

    assert output.shape == (B, L, L, 64), (
        f"Expected {(B, L, L, 64)}, got {output.shape}"
    )
    print(f"  ✓ Output shape: {output.shape}")


def test_multi_hypothesis_matching():
    """N hypotheses × K targets matching should produce valid losses."""
    print("Testing N×K matching...")

    B, L, S = 2, 12, 64
    N, K = 3, 2
    logits = torch.randn(B, N, L, L, S, requires_grad=True)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    targets = torch.randint(0, S, (B, K, L, L))
    target_mask = torch.ones(B, K, dtype=torch.bool)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    for mode in ["greedy", "hungarian"]:
        logits.grad = None
        loss = get_multi_target_distogram_loss(
            logits, soft_targets, pair_mask, target_mask, mode=mode,
            diversity_weight=0.1,
        )
        assert loss.dim() == 0, f"Expected scalar for {mode}"
        assert torch.isfinite(loss), f"Non-finite loss for {mode}"
        loss.backward()
        assert logits.grad is not None and logits.grad.abs().sum() > 0, (
            f"No gradient flow for {mode}"
        )
        print(f"  ✓ {mode}: loss={loss:.4f}, grad flows")


def test_multi_hyp_with_recycling():
    """Multi-hypothesis should work with recycling."""
    print("Testing multi-hypothesis + recycling...")

    model = CrystalTriangularModel(
        emb_dim=1280, d_model=128, d_pair=32, n_blocks=2,
        tri_hidden=16, out_ch=64, use_checkpoint=False,
        n_recycles=1, n_hypotheses=3,
    )

    B, L = 2, 30
    emb = torch.randn(B, L, 1280)
    mask = torch.ones(B, L, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        output = model(emb, seq_mask=mask)

    assert output.shape == (B, 3, L, L, 64), (
        f"Expected {(B, 3, L, L, 64)}, got {output.shape}"
    )
    print(f"  ✓ Shape with recycling: {output.shape}")


def test_per_hypothesis_metrics():
    """Per-hypothesis validation metrics should work."""
    print("Testing per-hypothesis metrics...")

    B, N, L, S, K = 2, 3, 16, 64, 2
    logits = torch.randn(B, N, L, L, S)
    targets = torch.randint(0, S, (B, K, L, L))
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    target_mask = torch.ones(B, K, dtype=torch.bool)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)

    metrics = calculate_per_hypothesis_metrics(
        logits, targets, pair_mask, target_mask, smoother,
    )

    # Should have keys like hyp0_tgt0_loss, hyp0_tgt1_loss, hyp0_best_loss, etc.
    for n in range(min(N, 3)):
        for k in range(min(K, 3)):
            key = f"hyp{n}_tgt{k}_loss"
            assert key in metrics, f"Missing metric: {key}"
            assert metrics[key] > 0, f"Non-positive loss: {key}={metrics[key]}"
        best_key = f"hyp{n}_best_loss"
        assert best_key in metrics, f"Missing metric: {best_key}"

    print(f"  ✓ Metrics: {list(metrics.keys())}")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")


def test_all_configs_support_n_hypotheses():
    """All config presets should work with n_hypotheses."""
    print("Testing all configs support n_hypotheses...")

    configs = {
        "default": get_default_config(),
        "small": get_small_config(),
        "large": get_large_config(),
        "optimized": get_optimized_config(),
        "cap": get_cap_config(),
    }

    for name, config in configs.items():
        assert hasattr(config.model, "n_hypotheses"), (
            f"Config {name} missing n_hypotheses"
        )
        assert config.model.n_hypotheses == 1, (
            f"Config {name} default n_hypotheses should be 1"
        )
        # Verify we can set it
        config.model.n_hypotheses = 3
        assert config.model.n_hypotheses == 3
        print(f"  ✓ {name}: n_hypotheses configurable")


def test_n_greater_than_k():
    """When N > K, unmatched hypotheses should exist."""
    print("Testing N > K matching...")

    B, L, S = 2, 12, 64
    N, K = 4, 2  # 4 hyp, 2 targets → 2 unmatched
    logits = torch.randn(B, N, L, L, S)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    targets = torch.randint(0, S, (B, K, L, L))
    target_mask = torch.ones(B, K, dtype=torch.bool)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)
    og_target = torch.randint(0, S, (B, L, L))
    soft_og = smoother(og_target)

    # With OG weight, unmatched hypotheses should add to loss
    loss_no_og = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        crystal_og_weight=0.0,
    )
    loss_with_og = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="greedy",
        soft_target_og=soft_og, crystal_og_weight=0.1,
    )
    assert loss_with_og > loss_no_og, (
        f"OG loss on unmatched should increase total: {loss_with_og:.4f} vs {loss_no_og:.4f}"
    )
    print(f"  ✓ N>K: with OG {loss_with_og:.4f} > without {loss_no_og:.4f}")


def test_k_greater_than_n():
    """When K > N, random targets should be sampled (no crash)."""
    print("Testing K > N matching (hungarian)...")

    B, L, S = 2, 12, 64
    N, K = 2, 5  # 2 hyp, 5 targets
    logits = torch.randn(B, N, L, L, S)
    pair_mask = torch.ones(B, L, L, dtype=torch.bool)
    targets = torch.randint(0, S, (B, K, L, L))
    target_mask = torch.ones(B, K, dtype=torch.bool)

    smoother = TargetSmoother(num_bins=S, sigma=0.8, ignore_index=-1)
    soft_targets = smoother(targets)

    loss = get_multi_target_distogram_loss(
        logits, soft_targets, pair_mask, target_mask, mode="hungarian"
    )
    assert torch.isfinite(loss), f"Non-finite loss with K>N: {loss}"
    assert loss > 0
    print(f"  ✓ K>N loss = {loss:.4f}")


def main():
    print("=" * 60)
    print("Multi-Hypothesis Multi-Target Tests")
    print("=" * 60)
    print()

    tests = [
        # --- existing tests (backward compat) ---
        ("Greedy selects best",        test_greedy_selects_best_target),
        ("Hungarian valid loss",       test_hungarian_returns_valid_loss),
        ("Greedy ≤ worst",             test_greedy_leq_single_worst),
        ("Partial target mask",        test_partial_target_mask),
        ("Single-target fallback",     test_single_target_fallback),
        ("Alias",                      test_alias),
        ("LR scheduler defaults",      test_lr_scheduler_defaults),
        ("Matching config",            test_matching_config_defaults),
        ("Memory estimate",            test_memory_estimate),
        ("Gradient flow",              test_gradients_flow),
        # --- new multi-hypothesis tests ---
        ("Multi-hyp forward",          test_multi_hypothesis_forward),
        ("Single-hyp compat",          test_single_hypothesis_backward_compat),
        ("N×K matching",               test_multi_hypothesis_matching),
        ("Multi-hyp + recycling",      test_multi_hyp_with_recycling),
        ("Per-hyp metrics",            test_per_hypothesis_metrics),
        ("All configs n_hypotheses",   test_all_configs_support_n_hypotheses),
        ("OG loss multi-hyp",          test_crystal_og_loss_multi_hypothesis),
        ("Diversity multi-hyp",        test_diversity_loss_multi_hypothesis),
        ("N > K matching",             test_n_greater_than_k),
        ("K > N matching",             test_k_greater_than_n),
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
        print(f"  {name:30s}: {status}")

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
