#!/usr/bin/env python3
"""
Quick test to verify ProXtal-LM installation and imports.

This script checks that all modules can be imported correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all module imports."""
    print("Testing ProXtal-LM imports...")
    
    try:
        print("  ✓ Importing proxtal_lm...")
        import proxtal_lm
        
        print("  ✓ Importing models...")
        from proxtal_lm.models import CrystalTriangularModel
        
        print("  ✓ Importing data...")
        from proxtal_lm.data import CrystalContactsDataset, collate_pad
        
        print("  ✓ Importing training...")
        from proxtal_lm.training import train_one_epoch, validate_one_epoch
        
        print("  ✓ Importing utils...")
        from proxtal_lm.utils import (
            TargetSmoother, Metric_Tracker, make_pair_mask,
            get_loss_from_multiple_targets, calculate_metrics
        )
        
        print("  ✓ Importing config...")
        from proxtal_lm.config import (
            ExperimentConfig, ModelConfig, DataConfig, TrainingConfig,
            get_default_config, get_small_config, get_large_config
        )
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_config():
    """Test configuration creation."""
    print("\nTesting configurations...")
    
    try:
        from proxtal_lm.config import get_default_config, get_small_config, get_large_config
        
        configs = {
            'default': get_default_config(),
            'small': get_small_config(),
            'large': get_large_config()
        }
        
        for name, config in configs.items():
            print(f"  ✓ {name:8s}: d_model={config.model.d_model}, "
                  f"d_pair={config.model.d_pair}, blocks={config.model.n_blocks}")
        
        print("\n✅ Configurations working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Config error: {e}")
        return False


def test_model_creation():
    """Test model instantiation."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from proxtal_lm.models import CrystalTriangularModel
        
        model = CrystalTriangularModel(
            emb_dim=1280,
            d_model=128,
            d_pair=32,
            n_blocks=2,
            tri_hidden=16,
            out_ch=64,
            use_checkpoint=False
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created with {num_params:,} parameters")
        
        # Test forward pass
        B, L, D = 2, 50, 1280
        emb = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)
        
        model.eval()
        with torch.no_grad():
            output = model(emb, seq_mask=mask)
        
        print(f"  ✓ Forward pass successful: {emb.shape} -> {output.shape}")
        print("\n✅ Model working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ProXtal-LM Installation Test")
    print("="*60)
    print()
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Model", test_model_creation()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:12s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("="*60)
    if all_passed:
        print("\n🎉 All tests passed! ProXtal-LM is ready to use.")
        print("\nNext steps:")
        print("  1. Set up data paths in proxtal_lm/config.py")
        print("  2. Run: python scripts/train.py --config small")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
