"""
ProXtal-LM Validation Subpackage
=================================

Crystal structure search and MLIP-based relaxation for validating
predicted distograms against real crystal structures.

Two-stage pipeline:
  1. Crystal Search (DE optimizer): Finds best space group / unit cell
     given predicted crystogram contacts.
  2. MLIP Ranker (MACE + ASE): Relaxes candidates with a machine-learned
     interatomic potential and ranks by energy.

Extra dependencies (not required by core proxtal_lm):
  - gemmi       (crystallography I/O & symmetry)
  - ase         (atomic simulation environment)
  - mace-torch  (MACE-OFF MLIP calculator)
  - scipy       (spatial rotations)

Install them via:
  conda env create -f proxtal_lm/validation/environment.yaml
"""

# Target builder only needs torch/numpy — always available
from .target_builder import (
    CrystalTargetSmoother,
    make_distogram_targets,
    make_cristogram_targets,
    pairwise_distances,
    get_distances,
)

# The following modules have heavy optional deps (gemmi, scipy, ase, mace).
# Import them lazily so ``import proxtal_lm.validation`` never fails due to
# missing optional packages — you just get an ImportError when you actually
# try to use the class.

def __getattr__(name):
    """Lazy import for validation submodules with optional dependencies."""
    _CRYSTAL_SEARCH_NAMES = {
        "CrystalSearchDynamic", "PyTorchDifferentialEvolution",
        "CrystalContact", "extract_crystal_contacts",
        "get_chiral_space_groups", "get_crystal_system_params",
    }
    _MLIP_RANKER_NAMES = {
        "MLIPResolver", "ForceCappedCalculator",
        "CandidateResult", "SurvivalStatus",
        "build_crystal_and_squeeze", "get_crystal_system", "get_strain_mask",
    }
    _TARGET_BUILDER_NAMES = {
        "structure_to_CB_matrix", "structure_to_CA_matrix",
        "get_fraction_matrices", "get_minimum_shift_distances",
        "process_protein", "process_esmfold_protein",
    }

    if name in _CRYSTAL_SEARCH_NAMES:
        from . import crystal_search as _cs
        return getattr(_cs, name)
    if name in _MLIP_RANKER_NAMES:
        from . import mlip_ranker as _mr
        return getattr(_mr, name)
    if name in _TARGET_BUILDER_NAMES:
        from . import target_builder as _tb
        return getattr(_tb, name)

    raise AttributeError(f"module 'proxtal_lm.validation' has no attribute {name!r}")


__all__ = [
    # Always available (torch/numpy only)
    "CrystalTargetSmoother",
    "make_distogram_targets",
    "make_cristogram_targets",
    "pairwise_distances",
    "get_distances",
    # Lazy (need gemmi)
    "structure_to_CB_matrix",
    "structure_to_CA_matrix",
    "get_fraction_matrices",
    "get_minimum_shift_distances",
    "process_protein",
    "process_esmfold_protein",
    # Lazy (need gemmi + scipy)
    "CrystalSearchDynamic",
    "PyTorchDifferentialEvolution",
    "CrystalContact",
    "extract_crystal_contacts",
    "get_chiral_space_groups",
    "get_crystal_system_params",
    # Lazy (need ase + mace)
    "MLIPResolver",
    "ForceCappedCalculator",
    "CandidateResult",
    "SurvivalStatus",
    "build_crystal_and_squeeze",
    "get_crystal_system",
    "get_strain_mask",
]
