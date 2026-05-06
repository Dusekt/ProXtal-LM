#!/usr/bin/env python3
"""
Validation script for ProXtal-LM checkpoints with crystal vs original comparison.

Usage:
    python scripts/validate_checkpoint.py checkpoints/best_checkpoint.pt
    python scripts/validate_checkpoint.py checkpoints/best_checkpoint.pt --data path/to/val.h5
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proxtal_lm.models import CrystalTriangularModel
from proxtal_lm.data import CrystalContactsDataset, collate_pad
from proxtal_lm.utils import (
    TargetSmoother, make_pair_mask, get_loss_from_multiple_targets,
    calculate_metrics
)
from proxtal_lm.validation_metrics import calculate_comprehensive_metrics


@torch.no_grad()
def validate_with_metrics(model, dataloader, device):
    """
    Validate model with comprehensive metrics including crystal vs original comparison.
    
    Args:
        model: The neural network model
        dataloader: Validation data loader
        device: Device to run on
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize target generator
    target_generator = TargetSmoother(num_bins=64, sigma=0.8, ignore_index=-1).to(device)
    
    all_metrics = {
        'loss': [],
        'all_crystal_precision_L': [],
        'all_crystal_precision_L2': [],
        'all_crystal_precision_L5': [],
        'all_crystal_recall': [],
        'all_crystal_f1': [],
        'crystal_only_precision_L': [],
        'crystal_only_precision_L2': [],
        'crystal_only_precision_L5': [],
        'crystal_only_recall': [],
        'crystal_only_f1': [],
        'crystal_only_accuracy': [],
        'pct_changed_positions': [],
        'num_changed_positions': [],
    }
    
    num_batches = 0
    num_batches_with_og = 0
    
    print("\n" + "="*70)
    print("Running validation...")
    print("="*70)
    
    for batch_idx, batch in enumerate(dataloader):
        emb = batch["embedding"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)
        target_crystal = batch["contact"].to(device)
        
        if mask.sum() == 0:
            continue
        
        pair_mask = make_pair_mask(mask)
        target_valid_mask = (target_crystal[:, :, 0, 0] != -1)
        
        # Forward pass
        with torch.amp.autocast('cuda'):
            logits = model(emb, seq_mask=mask)
            
            # Calculate loss
            soft_targets = target_generator(target_crystal)
            loss = get_loss_from_multiple_targets(
                logits, soft_targets, target_valid_mask, pair_mask, mode='mean'
            )
        
        all_metrics['loss'].append(loss.item())
        
        # Standard metrics (on crystal contacts only)
        standard_metrics = calculate_metrics(logits, target_crystal, pair_mask, lengths)
        for key, value in standard_metrics.items():
            metric_key = f'crystal_{key}'
            if metric_key in all_metrics:
                all_metrics[metric_key].append(value)
        
        # Crystal vs Original metrics (if available)
        if "contact_og" in batch:
            target_original = batch["contact_og"].to(device)
            
            if target_original.shape[1] > 0:  # Has original contacts
                crystal_metrics = calculate_comprehensive_metrics(
                    logits, target_crystal, target_original, pair_mask, lengths
                )
                
                for key, value in crystal_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
                
                num_batches_with_og += 1
        
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...", end='\r')
    
    print(f"\n  Completed {num_batches} batches ({num_batches_with_og} with original contacts)")
    
    # Average all metrics
    final_metrics = {}
    for key, values in all_metrics.items():
        if len(values) > 0:
            final_metrics[key] = sum(values) / len(values)
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Validate ProXtal-LM checkpoint with crystal vs original comparison"
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to validation data file or directory (default: use config)'
    )
    parser.add_argument(
        '--val-file',
        type=str,
        default='valid_data_3d_cln',
        help='Name of validation file if --data is a directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for validation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load checkpoint
    print("="*70)
    print("LOADING CHECKPOINT")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config')
    if config is None:
        print("⚠️  No config found in checkpoint, using default")
        from proxtal_lm.config import get_default_config
        config = get_default_config()
    
    print(f"Device: {device}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # Create model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    model = CrystalTriangularModel(
        emb_dim=config.model.emb_dim,
        d_model=config.model.d_model,
        d_pair=config.model.d_pair,
        n_seq_layers=config.model.n_seq_layers,
        n_blocks=config.model.n_blocks,
        tri_hidden=config.model.tri_hidden,
        out_ch=config.model.out_ch,
        use_checkpoint=False  # Disable for inference
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        # Handle module prefix issues
        state_dict = {
            k.replace('module.', '').replace('_orig_mod.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
        }
        model.load_state_dict(state_dict)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Determine data path
    if args.data is None:
        data_path = config.data.val_path
        print(f"\nUsing validation data from config: {data_path}")
    else:
        # Handle both directory and file paths
        if os.path.isdir(args.data):
            # It's a directory, construct path to validation file
            data_path = os.path.join(args.data, args.val_file)
            print(f"\nUsing data directory: {args.data}")
            print(f"Validation file: {args.val_file}")
        else:
            # Assume it's a file path
            data_path = args.data
            print(f"\nUsing specified data file: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        if args.data and os.path.isdir(args.data):
            print(f"   Looking for '{args.val_file}' in {args.data}")
            print(f"   Use --val-file to specify a different filename")
        sys.exit(1)
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    dataset = CrystalContactsDataset(data_path, B=config.data.num_bins)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: collate_pad(batch, pad_multiple=config.data.pad_multiple)
    )
    
    print(f"Validation samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Run validation
    metrics = validate_with_metrics(model, dataloader, device)
    
    # Print results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    # Group metrics
    print("\n📊 LOSS:")
    print(f"  {'Validation Loss':<30} {metrics.get('loss', 0):.4f}")
    
    print("\n🎯 ALL CRYSTAL CONTACTS (Standard Metrics):")
    print(f"  {'Precision@L':<30} {metrics.get('all_crystal_precision_L', 0):.4f}")
    print(f"  {'Precision@L/2':<30} {metrics.get('all_crystal_precision_L2', 0):.4f}")
    print(f"  {'Precision@L/5':<30} {metrics.get('all_crystal_precision_L5', 0):.4f}")
    print(f"  {'Recall':<30} {metrics.get('all_crystal_recall', 0):.4f}")
    print(f"  {'F1 Score':<30} {metrics.get('all_crystal_f1', 0):.4f}")
    
    if metrics.get('crystal_only_accuracy', 0) > 0:
        print("\n🔬 CRYSTAL-SPECIFIC CONTACTS (Changed Positions Only):")
        print(f"  {'Avg Changed Positions':<30} {metrics.get('num_changed_positions', 0):.1f}")
        print(f"  {'% Changed of Total':<30} {metrics.get('pct_changed_positions', 0):.2f}%")
        print("")
        print(f"  {'Bin Accuracy':<30} {metrics.get('crystal_only_accuracy', 0):.4f}")
        print(f"  {'Precision@L':<30} {metrics.get('crystal_only_precision_L', 0):.4f}")
        print(f"  {'Precision@L/2':<30} {metrics.get('crystal_only_precision_L2', 0):.4f}")
        print(f"  {'Precision@L/5':<30} {metrics.get('crystal_only_precision_L5', 0):.4f}")
        print(f"  {'Recall':<30} {metrics.get('crystal_only_recall', 0):.4f}")
        print(f"  {'F1 Score':<30} {metrics.get('crystal_only_f1', 0):.4f}")
        
        print("\n📈 COMPARISON:")
        all_prec = metrics.get('all_crystal_precision_L', 0)
        crystal_prec = metrics.get('crystal_only_precision_L', 0)
        print(f"  {'All Contacts Precision@L':<30} {all_prec:.4f}")
        print(f"  {'Crystal-Only Precision@L':<30} {crystal_prec:.4f}")
        print(f"  {'Difference':<30} {crystal_prec - all_prec:+.4f}")
        print("")
        print("  Note: Crystal-only metrics are harder (only changed positions)")
        print("        and show model's ability to predict crystal-specific contacts.")
    else:
        print("\n⚠️  No original contacts found in validation data")
        print("   Only showing standard metrics on all crystal contacts.")
    
    print("\n" + "="*70)
    print("✅ Validation complete!")
    print("="*70)
    
    return metrics


if __name__ == "__main__":
    main()
