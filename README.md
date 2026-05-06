# ProXtal-LM: Protein Crystal Language Model

A deep learning framework for predicting protein crystallization outcomes using ESM2 embeddings and triangular attention mechanisms.

## 📋 Overview

ProXtal-LM predicts inter-residue distances in protein structures by:
1. Taking ESM2 protein embeddings as input
2. Processing them through a triangular attention network
3. Predicting distance binswhere each protein has multiple possible crystallization outcomes

The model architecture combines:
- **Sequence Encoder**: Transformer layers to process ESM2 embeddings
- **Pairwise Representations**: Symmetric initialization of residue pair features
- **Triangular Updates**: Geometric consistency through multiplicative updates
- **Axial Attention**: Efficient global context via row/column attention
- **Multi-target Loss**: Handles multiple ground truth structures per protein

## 🗂️ Project Structure

```
ProXtal-LM/
├── proxtal_lm/                      # Main package
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Neural network architectures
│   ├── data.py               # Dataset and data loading
│   ├── training.py           # Training/validation loops
│   ├── utils.py              # Loss functions and metrics
│   └── config.py             # Configuration management
│
├── scripts/                   # Executable scripts
│   └── train.py              # Main training script
│
├── configs/                   # Configuration files (optional)
├── checkpoints/              # Model checkpoints (created during training)
├── notebooks/                # Jupyter notebooks for analysis
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to ProXtal-LM directory
cd ProXtal-LM

# Install dependencies
pip install -r requirements.txt
```

### Basic Training

```bash
# Train with default configuration
python scripts/train.py

# Use small model for testing
python scripts/train.py --config small

# Resume from checkpoint
python scripts/train.py --resume checkpoints/latest_checkpoint.pt

# Custom experiment name
python scripts/train.py --name my_experiment --checkpoint-dir checkpoints/my_exp
```

### Using as a Library

```python
from proxtal_lm.models import CrystalTriangularModel
from proxtal_lm.data import CrystalContactsDataset, collate_pad
from proxtal_lm.training import train_one_epoch, validate_one_epoch
from proxtal_lm.config import get_default_config

# Create model
config = get_default_config()
model = CrystalTriangularModel(
    emb_dim=1280,
    d_model=256,
    d_pair=64,
    n_blocks=4
)

# Load dataset
dataset = CrystalContactsDataset("path/to/data.h5")
```

## 📊 Data Format

The model expects HDF5 files with the following structure:

```python
h5_file[
    '0': {                          # Sample index
        'embedding': [L, 1280],     # ESM2 embedding
        'contact_0': [L, L],        # Target 1 (integer bin indices)
        'contact_1': [L, L],        # Target 2 (optional)
        ...
    },
    '1': { ... },
    ...
]
```

- **embedding**: ESM2 protein embeddings (dimension 1280)
- **contact_N**: Ground truth distance maps as integer bin indices (0-63)
  - `-1` indicates padding
  - Bins < 27 typically represent contacts (< 8 Å)

## 🔧 Configuration

### Preset Configurations

- **default**: Baseline configuration (d_model=256, d_pair=64, 4 blocks)
- **small**: Faster for testing (d_model=128, d_pair=32, 2 blocks)
- **large**: Higher capacity (d_model=512, d_pair=128, 6 blocks)

### Custom Configuration

```python
from proxtal_lm.config import ExperimentConfig, ModelConfig, DataConfig

config = ExperimentConfig(
    name="my_experiment",
    model=ModelConfig(
        d_model=256,
        d_pair=64,
        n_blocks=4,
    ),
    data=DataConfig(
        train_path="path/to/train.h5",
        batch_size=4,
    ),
    training=TrainingConfig(
        learning_rate=2e-5,
        max_epochs=60,
    )
)
```

### Key Hyperparameters

**Model Architecture:**
- `emb_dim`: Input embedding dimension (1280 for ESM2)
- `d_model`: Sequence representation dimension
- `d_pair`: Pairwise representation dimension
- `n_blocks`: Number of TriAxial blocks
- `tri_hidden`: Hidden dimension for triangle updates

**Training:**
- `learning_rate`: Initial learning rate (default: 2e-5)
- `weight_decay`: L2 regularization (default: 1e-4)
- `batch_size`: Samples per batch (default: 3)
- `accum_steps`: Gradient accumulation steps (default: 64)
- `grad_clip`: Gradient clipping threshold (default: 1.0)
- `loss_mode`: Multi-target aggregation ('min' or 'mean')

**Data:**
- `num_bins`: Number of distance bins (default: 64)
- `smooth_sigma`: Gaussian smoothing for targets (default: 0.8)
- `pad_multiple`: Pad sequences to multiple for efficiency (default: 8)

## 📈 Metrics

The training process logs:
- **Loss**: Binary cross-entropy on smoothed distance distributions
- **Precision@L/L2/L5**: Precision in top L, L/2, L/5 predicted contacts
- **Recall**: Recall at 0.5 threshold
- **F1**: F1 score at 0.5 threshold

All metrics are calculated for residue pairs with sequence separation ≥ 6.

## 🧪 Model Architecture Details

### Sequence Encoder
- Transformer encoder layers on ESM2 embeddings
- Projects to lower dimension for efficiency

### Pairwise Representation
- **Initialization**: Symmetric sum and product of query/key projections
- **TriAxial Blocks**: Each block contains:
  1. **Axial Attention**: Row and column attention for global context
  2. **Outgoing Triangle Update**: `z_ij = Σ_k (a_ik × b_jk)`
  3. **Incoming Triangle Update**: `z_ij = Σ_k (a_ki × b_kj)`
  4. **Feed Forward**: Channel mixing

### Multi-Target Loss
- Handles proteins with multiple ground truth structures
- Two modes:
  - `min`: Take minimum loss (at least one structure correct)
  - `mean`: Average loss (all structures should match)

## 💾 Checkpointing

The training script automatically saves:
- **latest_checkpoint.pt**: After every epoch
- **best_checkpoint.pt**: When validation loss improves
- **checkpoint_epoch_N.pt**: Periodic saves (if configured)

Checkpoints include:
- Model state dict
- Optimizer state
- Scheduler state
- Training epoch
- Best validation loss
- Full configuration

## 🔬 Advanced Features

### Gradient Checkpointing
Enabled by default (`use_checkpoint=True`) to save memory during training by recomputing activations during backward pass.

### Mixed Precision Training
Automatic mixed precision (AMP) is enabled by default for faster training and lower memory usage.

### Gradient Accumulation
Simulates larger batch sizes by accumulating gradients over multiple forward passes before updating weights.

### Memory Optimization
- Chunked attention: Processes attention in chunks to reduce memory
- Flash Attention compatible (PyTorch 2.0+)
- Expandable memory segments for CUDA

## 📝 Notes

### Multiple Targets
The dataset contains multiple possible structures per protein (histograms vs crystograms). Currently, the model trains on all available targets. Future updates will add:
- Filtering to train only on crystograms
- Evaluation metrics specific to crystal vs. structure differences

### Sequence Separation
Contact prediction metrics only consider residue pairs with |i-j| ≥ 6 to focus on long-range interactions.

### Bin Definitions
- Bins 0-26: Contacts (< 8 Å typically)
- Bins 27-63: Non-contacts
- -1: Padding/invalid

## 🤝 Development Notes

This is a research codebase for protein crystallization prediction. The implementation is based on ideas from:
- AlphaFold2's triangular updates and attention mechanisms
- ESM2 protein language models
- Axial attention for efficient 2D processing

## 📧 Contact

For questions or issues, please refer to the project documentation or contact the development team.

## 🔄 Relationship to Other Folders

This reorganized codebase is based on the working implementation in:
- `../crystalpred/`: Original development folder with notebooks and experiments
- Training was done with checkpoints in `fsdp_checkpoints9/`
- Data files are referenced from the parent crystalpred directory

The ProXtal-LM folder provides a cleaner, modular organization while maintaining full compatibility with existing data and checkpoints.
