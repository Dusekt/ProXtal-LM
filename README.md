# ProXtal-LM: Protein Crystal Language Model

A deep learning framework for predicting protein crystallization outcomes using ESM2 embeddings and triangular attention mechanisms.

## 📋 Overview

ProXtal-LM predicts inter-residue distances in protein structures by:
1. Taking ESMC-300M protein embeddings as input
2. Processing them through a triangular attention network
3. Predicting distance bins where each protein has multiple possible crystallization outcomes

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
│   ├── config.py             # Configuration management
│   └── validation/           # Validation module
│
├── scripts/                   # Executable scripts
│   └── train.py              # Main training script
│
├── configs/                  # Configuration files (optional)
├── checkpoints/              # Model checkpoints (created during training)
├── notebooks/                # Jupyter notebooks for analysis (and fun exploration graph-maker)
│
└── requirements.txt          # Python dependencies
```

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

- **embedding**: ESMC-300M protein embeddings (dimension 1152)
- **contact_N**: Ground truth distance maps as integer bin indices (0-63)
  - `-1` indicates padding
  - Bins < 27 typically represent contacts (< 8 Å)

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


## Notes

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


##  Contact

For questions or issues, please contact the development team.

