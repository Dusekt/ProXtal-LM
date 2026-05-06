"""
ProXtal-LM — Protein Crystal Language Model (V2)

A deep learning framework for predicting crystal distograms from ESM-2
protein embeddings, featuring pair-biased attention, chain-aware masking,
memory-safe recycling, and windowed axial attention.
"""

__version__ = "0.2.0"

from .models import CrystalTriangularModel
from .data import CrystalContactsDataset, collate_pad
from .training import train_one_epoch, validate_one_epoch
from .utils import (
    TargetSmoother,
    make_pair_mask,
    make_seq_sep_mask,
    get_distogram_loss,
    get_multi_target_distogram_loss,
    get_loss_from_multiple_targets,
    calculate_metrics,
    Metric_Tracker,
)
from .validation_metrics import (
    calculate_crystal_specific_metrics,
    calculate_comprehensive_metrics,
    calculate_per_hypothesis_metrics,
)
from .config import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    get_default_config,
    get_small_config,
    get_large_config,
    get_optimized_config,
    get_cap_config,
)

__all__ = [
    # Model
    "CrystalTriangularModel",
    # Data
    "CrystalContactsDataset",
    "collate_pad",
    # Training
    "train_one_epoch",
    "validate_one_epoch",
    # Utilities
    "TargetSmoother",
    "make_pair_mask",
    "make_seq_sep_mask",
    "get_distogram_loss",
    "get_multi_target_distogram_loss",
    "get_loss_from_multiple_targets",
    "calculate_metrics",
    "Metric_Tracker",
    # Crystal metrics
    "calculate_crystal_specific_metrics",
    "calculate_comprehensive_metrics",
    "calculate_per_hypothesis_metrics",
    # Config
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "get_default_config",
    "get_small_config",
    "get_large_config",
    "get_optimized_config",
    "get_cap_config",
]
