#!/usr/bin/env python3
"""
Evaluation script for trained ProXtal-LM models.

Usage:
    python evaluate.py checkpoints/best_checkpoint.pt --data test
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
from proxtal_lm.training import validate_one_epoch


def evaluate(checkpoint_path, data_path=None, batch_size=4, device='cuda'):
    """
    Evaluate a trained model on a dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to h5 data file (if None, use config from checkpoint)
        batch_size: Batch size for evaluation
        device: Device to run on
    """
    # Load checkpoint
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    
    if config is None:
        print("⚠️  No config found in checkpoint, using default")
        from proxtal_lm.config import get_default_config
        config = get_default_config()
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Create model
    print("🏗️  Building model...")
    model = CrystalTriangularModel(
        emb_dim=config.model.emb_dim,
        d_model=config.model.d_model,
        d_pair=config.model.d_pair,
        n_seq_layers=config.model.n_seq_layers,
        n_blocks=config.model.n_blocks,
        tri_hidden=config.model.tri_hidden,
        out_ch=config.model.out_ch,
        use_checkpoint=False  # Disable checkpointing for inference
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
    
    model.eval()
    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Determine data path
    if data_path is None:
        data_path = config.data.test_path
        print(f"📊 Using test data from config: {data_path}")
    else:
        print(f"📊 Using specified data: {data_path}")
    
    # Load dataset
    dataset = CrystalContactsDataset(data_path, B=config.data.num_bins)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: collate_pad(batch, pad_multiple=config.data.pad_multiple)
    )
    
    print(f"📈 Evaluating on {len(dataset)} samples...")
    
    # Run evaluation
    metrics = validate_one_epoch(model, dataloader, device, mode='mean')
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProXtal-LM model")
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data file (default: use test set from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation'
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
    
    evaluate(args.checkpoint, args.data, args.batch_size, args.device)


if __name__ == "__main__":
    main()
