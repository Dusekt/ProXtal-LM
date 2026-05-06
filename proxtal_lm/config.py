"""
Configuration management for ProXtal-LM training.

Provides dataclass-based configuration with preset profiles:
- default:   Baseline with recycling disabled
- small:     Quick testing on small GPU
- large:     High-capacity model  
- optimized: Speed + memory optimised with recycling
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # --- Embedding dimensions ---
    emb_dim: int = 1152              # ESM-2 embedding dimension
    d_model: int = 256               # Sequence encoder hidden dimension
    d_pair: int = 64                 # Pair representation dimension

    # --- Architecture depth ---
    n_seq_layers: int = 3            # Number of sequence transformer layers
    n_blocks: int = 4                # Number of TriAxial blocks

    # --- Component dimensions ---
    tri_hidden: int = 32             # Triangle update hidden dimension
    out_ch: int = 64                 # Number of output distance bins

    # --- Attention ---
    n_heads: int = 8                 # Attention heads in sequence encoder
    dropout: float = 0.1             # Global dropout rate

    # --- Training efficiency ---
    use_checkpoint: bool = True      # Gradient checkpointing

    # --- Recycling ---
    n_recycles: int = 0              # Recycling iterations (0 = single pass)

    # --- Positional / crystal encodings ---
    max_rel_pos: int = 32            # Max relative position for encoding
    num_space_groups: int = 231      # 230 space groups + 1 unknown

    # --- Windowed attention ---
    attention_window_size: int = 0   # 0 → full attention, >0 → local window

    # --- Multi-hypothesis ---
    n_hypotheses: int = 1            # Number of output hypotheses (1 = single prediction)

    def __repr__(self) -> str:
        return (
            f"ModelConfig(emb={self.emb_dim}, d_model={self.d_model}, "
            f"d_pair={self.d_pair}, blocks={self.n_blocks}, "
            f"recycles={self.n_recycles}, n_hyp={self.n_hypotheses})"
        )


@dataclass
class DataConfig:
    """Data loading configuration."""

    # Dataset paths
    train_path: str = "../crystalpred/train_data_3d_cln"
    val_path: str = "../crystalpred/valid_data_3d_cln"
    test_path: str = "../crystalpred/test_data_3d_cln"

    # Data processing
    num_bins: int = 64
    smooth_sigma: float = 0.8
    pad_multiple: int = 8

    # DataLoader
    batch_size: int = 3
    num_workers: int = 4
    pin_memory: bool = True

    def __repr__(self) -> str:
        return (
            f"DataConfig(batch={self.batch_size}, bins={self.num_bins}, "
            f"workers={self.num_workers})"
        )


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Optimisation
    learning_rate: float = 4e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    accum_steps: int = 64

    # Duration
    max_epochs: int = 60

    # LR scheduling
    scheduler_type: str = "none"   # 'plateau', 'cosine', or 'none'
    scheduler_factor: float = 0.8
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-7

    # Early stopping
    early_stop_patience: int = 10

    # Mixed precision
    use_amp: bool = True

    # Compilation
    use_compile: bool = False
    compile_mode: str = "default"

    # Workers & checkpointing
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 0

    # Multi-target matching
    matching_mode: str = "none"       # 'none', 'hungarian', or 'greedy'
    crystal_og_weight: float = 0.0    # Weight for original-structure loss (unmatched hyps)
    diversity_weight: float = 0.0     # Weight for diversity loss
    crystal_contact_weight: float = 0.0  # Weight for universal contact loss (ALL hyps)
    contact_threshold: float = 0.1    # KL threshold to detect contact positions
    contact_emphasis: float = 1.0     # Weight multiplier for contacts in matched loss (1.0=uniform)

    # Logging
    log_file: str = "training_metrics.csv"
    verbose: bool = True

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(lr={self.learning_rate}, epochs={self.max_epochs}, "
            f"accum={self.accum_steps})"
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    name: str = "default_experiment"
    description: str = ""
    resume_from: Optional[str] = None
    reset_lr_on_resume: bool = False   # Override saved LR with config LR on resume
    device: str = "cuda"

    def __post_init__(self):
        self.training.checkpoint_dir = os.path.abspath(self.training.checkpoint_dir)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)

        # Always place the log file inside checkpoint_dir, even if a
        # previous __post_init__ call already made it absolute (happens
        # when the CLI overrides checkpoint_dir after preset construction).
        self.training.log_file = os.path.join(
            self.training.checkpoint_dir,
            os.path.basename(self.training.log_file),
        )
        os.makedirs(os.path.dirname(self.training.log_file), exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  name='{self.name}'\n"
            f"  {self.model}\n"
            f"  {self.data}\n"
            f"  {self.training}\n"
            f")"
        )


# ================================================================
# Preset Configurations
# ================================================================

def get_default_config() -> ExperimentConfig:
    """Baseline: single-pass model (no recycling)."""
    return ExperimentConfig(
        name="baseline",
        description="Default single-pass configuration",
    )


def get_small_config() -> ExperimentConfig:
    """Smaller model for quick testing."""
    config = ExperimentConfig(
        name="small",
        description="Small model for quick testing",
    )
    config.model.d_model = 128
    config.model.d_pair = 32
    config.model.n_blocks = 2
    config.model.tri_hidden = 16
    config.model.n_recycles = 0
    config.data.batch_size = 8
    config.training.accum_steps = 32
    return config


def get_small_config_og() -> ExperimentConfig:
    """Smaller model for quick testing."""
    config = ExperimentConfig(
        name="small_og",
        description="Small model for quick testing",
    )

    config.data.train_path = "data/train_data_3d_cln.h5"
    config.data.val_path = "data/valid_data_3d_cln.h5"
    config.data.test_path = "data/test_data_3d_cln.h5"

    config.model.d_model = 128
    config.model.d_pair = 32
    config.model.n_blocks = 2
    config.model.tri_hidden = 16
    config.model.n_recycles = 0
    config.data.batch_size = 8
    config.training.accum_steps = 32
    return config

def get_large_config() -> ExperimentConfig:
    """Smaller model for quick testing."""
    config = ExperimentConfig(
        name="small_og",
        description="Small model for quick testing",
    )

    config.data.train_path = "data/train_data_3d_esmc"
    config.data.val_path = "data/valid_data_3d_esmc"
    config.data.test_path = "data/test_data_3d_esmc"

    config.emb_dim = 1152

    config.model.d_model = 128
    config.model.d_pair = 32
    config.model.n_blocks = 2
    config.model.tri_hidden = 16
    config.model.n_recycles = 0
    config.data.batch_size = 3
    config.training.accum_steps = 32

    config.training.scheduler_type = "cosine"
    config.training.scheduler_patience = 50
    config.training.early_stop_patience = 150
    
    return config


def get_optimized_config() -> ExperimentConfig:
    """Speed + memory optimised with recycling and windowed attention."""
    config = ExperimentConfig(
        name="optimized",
        description="Optimised configuration with recycling and windowed attention",
    )

    # Model
    config.model.d_model = 256
    config.model.d_pair = 64
    config.model.n_blocks = 4
    config.model.tri_hidden = 32
    config.model.n_recycles = 1
    config.model.attention_window_size = 0   # full attention by default

    # Data
    config.data.batch_size = 6
    config.data.num_bins = 64
    config.data.pad_multiple = 32

    # Training
    config.training.learning_rate = 3e-5
    config.training.weight_decay = 1e-4
    config.training.grad_clip = 1.0
    config.training.accum_steps = 64
    config.training.max_epochs = 80

    config.training.use_compile = False
    config.training.num_workers = 4
    config.training.use_amp = True

    config.training.scheduler_type = "cosine"
    config.training.scheduler_patience = 5
    config.training.early_stop_patience = 15

    return config


def get_cap_config() -> ExperimentConfig:
    """Maximise A100-80GB utilisation (~95% VRAM target).

    Same architecture as optimized but with batch_size pushed up to fill
    the available 80 GB.  Accumulation steps reduced proportionally so the
    effective batch stays large.
    """
    config = get_optimized_config()
    config.name = "cap"
    config.description = "A100-80GB capacity-fill config (~95% VRAM)"

    # batch 4 → 46 GB   ⇒  ~11.6 GB / sample
    # batch 7 → ~81 GB  ⇒  fits in 80 GB with AMP headroom
    config.data.batch_size = 7
    # keep effective batch roughly comparable: 7 * 36 = 252  (was 4 * 64 = 256)
    config.training.accum_steps = 36

    return config
