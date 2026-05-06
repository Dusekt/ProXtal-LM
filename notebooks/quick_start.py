"""
Quick guide for using ProXtal-LM.

This notebook demonstrates:
1. Loading and inspecting the data
2. Creating a model
3. Basic inference
4. Visualizing results
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

from proxtal_lm.models import CrystalTriangularModel
from proxtal_lm.data import CrystalContactsDataset, collate_pad
from proxtal_lm.config import get_default_config

# %% Load configuration
config = get_default_config()
print(config)

# %% Inspect dataset
dataset = CrystalContactsDataset(config.data.train_path, B=64)
print(f"Dataset size: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"\nSample 0:")
print(f"  Embedding shape: {sample['embedding'].shape}")
print(f"  Sequence length: {sample['length']}")
print(f"  Number of targets: {len(sample['contact'])}")
print(f"  Contact map shape: {sample['contact'][0].shape}")

# %% Visualize contact map
contact_map = sample['contact'][0].numpy()
L = sample['length'].item()

plt.figure(figsize=(10, 10))
plt.imshow(contact_map[:L, :L], cmap='viridis', origin='lower')
plt.colorbar(label='Distance bin')
plt.title(f'Contact Map (L={L})')
plt.xlabel('Residue j')
plt.ylabel('Residue i')
plt.tight_layout()
plt.savefig('example_contact_map.png', dpi=150)
plt.show()

# %% Create model
model = CrystalTriangularModel(
    emb_dim=1280,
    d_model=256,
    d_pair=64,
    n_blocks=4,
    tri_hidden=32,
    out_ch=64,
    use_checkpoint=False  # Disable for inference
)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% Forward pass example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Prepare input
emb = sample['embedding'].unsqueeze(0).to(device)  # Add batch dim
mask = sample['mask'].unsqueeze(0).to(device)

print(f"\nInput shape: {emb.shape}")

# Run inference
with torch.no_grad():
    logits = model(emb, seq_mask=mask)
    probs = torch.softmax(logits, dim=-1)

print(f"Output shape: {logits.shape}")

# %% Visualize predictions
# Get predicted contact probabilities (sum of first 27 bins)
pred_contacts = probs[0, :L, :L, :27].sum(dim=-1).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Ground truth
im1 = axes[0].imshow((contact_map[:L, :L] < 27).astype(float), 
                     cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
axes[0].set_title('Ground Truth Contacts')
axes[0].set_xlabel('Residue j')
axes[0].set_ylabel('Residue i')
plt.colorbar(im1, ax=axes[0])

# Predictions
im2 = axes[1].imshow(pred_contacts, cmap='RdBu_r', origin='lower', vmin=0, vmax=1)
axes[1].set_title('Predicted Contact Probability')
axes[1].set_xlabel('Residue j')
axes[1].set_ylabel('Residue i')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('predictions_vs_ground_truth.png', dpi=150)
plt.show()

# %% Load trained checkpoint (if available)
checkpoint_path = '../checkpoints/best_checkpoint.pt'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Best validation loss: {checkpoint['best_val_loss']:.4f}")
except FileNotFoundError:
    print(f"⚠️  No checkpoint found at {checkpoint_path}")
    print("   Train a model first using: python ../scripts/train.py")

print("\n✅ Example complete!")
