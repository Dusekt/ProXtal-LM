"""
ProXtal-LM Validation — Crystal Target Builder
===============================================

Utilities for building distogram and crystogram targets from PDB/mmCIF
structures using gemmi. Used to generate ground-truth validation data.

Key functions:
  - process_protein:          Build both distogram + crystogram from a crystal structure
  - process_esmfold_protein:  Build distogram only from an ESMFold PDB (no crystal)
  - make_distogram_targets:   Bin pairwise CB distances into discrete bins
  - make_cristogram_targets:  Bin crystal-contact distances (with symmetry mates) into bins
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gemmi
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False


# ── Amino-acid mapping ──────────────────────────────────────────────────────

THREE_TO_ONE = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Mse': 'M', 'Sec': 'C', 'Pyl': 'K', 'Hyp': 'P',
    'Sep': 'S', 'Tpo': 'T', 'Ptr': 'Y',
    'Asx': 'D', 'Glx': 'Q', 'Xle': 'I',
}


def _three_to_one(three_letter_code: str) -> str:
    """Convert 3-letter amino acid code to 1-letter."""
    key = three_letter_code[0].upper() + three_letter_code[1:].lower()
    return THREE_TO_ONE[key]


# ── Geometry helpers ────────────────────────────────────────────────────────

def get_metric_tensor(cell) -> np.ndarray:
    """Compute the metric tensor for a given unit cell."""
    a, b, c = cell.a, cell.b, cell.c
    alpha, beta, gamma = np.radians([cell.alpha, cell.beta, cell.gamma])
    return np.array([
        [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
        [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)],
        [a * c * np.cos(beta), b * c * np.cos(alpha), c**2],
    ])


def pairwise_distances(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """All-pairs Euclidean distances between two sets of coordinates."""
    diff = mat1[:, np.newaxis, :] - mat2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def get_distances(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for a single coordinate set."""
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=-1))


# ── Structure → coordinate matrices ─────────────────────────────────────────

def structure_to_CB_matrix(structure) -> np.ndarray:
    """Extract CB positions (falling back to CA for Gly) from a gemmi Structure."""
    if not GEMMI_AVAILABLE:
        raise ImportError("gemmi is required for structure_to_CB_matrix")
    structure.remove_alternative_conformations()
    matrix = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.name.upper()
                try:
                    _three_to_one(resname)
                except KeyError:
                    continue
                if not residue.entity_type == gemmi.EntityType.Polymer:
                    continue
                cb_found = False
                ca_pos = None
                for atom in residue:
                    if atom.name == "CB":
                        matrix.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        cb_found = True
                        break
                    if atom.name == "CA":
                        ca_pos = [atom.pos.x, atom.pos.y, atom.pos.z]
                if not cb_found and ca_pos:
                    matrix.append(ca_pos)
    return np.array(matrix)


def structure_to_CA_matrix(structure) -> np.ndarray:
    """Extract CA positions from a gemmi Structure."""
    if not GEMMI_AVAILABLE:
        raise ImportError("gemmi is required for structure_to_CA_matrix")
    matrix = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.name.upper()
                try:
                    _three_to_one(resname)
                except (KeyError, IndexError):
                    continue
                if not residue.entity_type == gemmi.EntityType.Polymer:
                    continue
                for atom in residue:
                    if atom.name == "CA":
                        matrix.append([atom.pos.x, atom.pos.y, atom.pos.z])
                        break
    return np.array(matrix)


# ── Fractional-coordinate symmetry expansion ────────────────────────────────

def get_fraction_matrices(structure, structure_to_matrix_fn) -> List[np.ndarray]:
    """Apply all symmetry operations and return fractional coordinate matrices."""
    if not GEMMI_AVAILABLE:
        raise ImportError("gemmi is required for get_fraction_matrices")

    structure.setup_entities()
    cell = structure.cell
    sg_ops = gemmi.find_spacegroup_by_name(structure.spacegroup_hm).operations()

    transformations = []
    for op in sg_ops.sym_ops:
        rot_matrix = np.array(op.rot).reshape(3, 3) / 24.0
        trans_vector = np.array(op.tran) / 24.0
        transformations.append((rot_matrix, trans_vector))

    struct_matrix = structure_to_matrix_fn(structure)
    frac_mat = cell.frac.mat
    matrices = []
    for op in transformations:
        trans_matrix = struct_matrix @ frac_mat
        trans_matrix = trans_matrix @ op[0] + op[1]
        # Pack to unit cell
        com = np.mean(trans_matrix, axis=0)
        transl = -com + np.array([(ac - np.floor(ac)) for ac in com])
        trans_matrix = trans_matrix + transl
        matrices.append(trans_matrix)
    return matrices


def get_minimum_shift_distances(sym_matrices, shifts, orth_mat) -> np.ndarray:
    """Compute minimum inter-molecular distances across symmetry mates + periodic shifts."""
    mat0 = sym_matrices[0] @ orth_mat
    distance_mat = get_distances(mat0)
    for j in range(len(sym_matrices)):
        for s in range(len(shifts)):
            shifted = (sym_matrices[j] + shifts[s]) @ orth_mat
            dists = pairwise_distances(mat0, shifted)
            distance_mat = np.minimum(dists, distance_mat)
    return distance_mat


# ── Binning / target generation ─────────────────────────────────────────────

def make_distogram_targets(
    coords: np.ndarray,
    num_bins: int = 64,
    min_bin: float = 2.0,
    max_bin: float = 22.0,
) -> np.ndarray:
    """
    Bin pairwise CB distances into integer bin indices.

    Returns:
        (N, N) int8 array of bin indices in [0, num_bins-1].
    """
    dists = get_distances(coords)
    frac = (dists - min_bin) / (max_bin - min_bin) * (num_bins - 1)
    idx = np.floor(frac + 0.5).astype(np.int8)
    return np.clip(idx, 0, num_bins - 1)


def make_cristogram_targets(
    coords: np.ndarray,
    dists: np.ndarray,
    num_bins: int = 64,
    min_bin: float = 2.0,
    max_bin: float = 22.0,
) -> np.ndarray:
    """
    Bin crystal-contact distances (precomputed via symmetry expansion).

    Returns:
        (N, N) int8 array of bin indices in [0, num_bins-1].
    """
    frac = (dists - min_bin) / (max_bin - min_bin) * (num_bins - 1)
    idx = np.floor(frac + 0.5).astype(np.int8)
    return np.clip(idx, 0, num_bins - 1)


# ── TargetSmoother (standalone version for validation pipeline) ─────────────

class CrystalTargetSmoother(nn.Module):
    """
    Converts ground-truth integer bin indices to Gaussian-smoothed soft targets.

    This is a standalone copy for the validation subpackage (does not depend on
    ``proxtal_lm.utils.TargetSmoother``) to keep the validation environment
    self-contained.

    Args:
        num_bins:     Number of distance bins (default 64)
        sigma:        Standard deviation of the Gaussian kernel (default 0.8)
        ignore_index: Index value indicating padding / invalid data (default -1)
    """

    def __init__(self, num_bins: int = 64, sigma: float = 0.8, ignore_index: int = -1):
        super().__init__()
        self.num_bins = num_bins
        self.ignore_index = ignore_index

        k_size = int(math.ceil(3 * sigma) * 2 + 1)
        k_range = torch.arange(k_size, dtype=torch.float32) - (k_size - 1) / 2
        kernel = torch.exp(-0.5 * (k_range / sigma) ** 2)
        kernel = kernel / kernel.sum()

        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.pad = k_size // 2

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: arbitrary-shape integer tensor (padded with -1)

        Returns:
            same shape + trailing num_bins dim, float tensor
        """
        orig_shape = indices.shape
        flat_idx = indices.reshape(-1)

        mask = flat_idx != self.ignore_index
        safe_idx = flat_idx.clamp(0, self.num_bins - 1)

        one_hot = F.one_hot(safe_idx, num_classes=self.num_bins).float()
        one_hot = one_hot.unsqueeze(1)
        smoothed = F.conv1d(one_hot, self.kernel, padding=self.pad).squeeze(1)

        smoothed_sum = smoothed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        smoothed = smoothed / smoothed_sum

        smoothed = smoothed * mask.unsqueeze(-1)
        return smoothed.view(*orig_shape, self.num_bins)


# ── High-level processing functions ─────────────────────────────────────────

def process_protein(mmcif_file: str, output_mat: str) -> None:
    """
    Build both distogram and crystogram targets from a crystal mmCIF file.

    Saves:
        ``{output_mat}_indices.npy``    — distogram targets (B, L, L, S)
        ``{output_mat}_indices_C.npy``  — crystogram targets (B, L, L, S)
    """
    if not GEMMI_AVAILABLE:
        raise ImportError("gemmi is required for process_protein")

    smoother = CrystalTargetSmoother()

    structure = gemmi.read_structure(mmcif_file)
    cell = structure.cell
    orth_mat = cell.orth.mat

    B = structure_to_CB_matrix(structure)
    Btargets = smoother(torch.Tensor(make_distogram_targets(B)).long()).numpy()
    np.save(f"{output_mat}_indices.npy", Btargets)

    CB_mats = get_fraction_matrices(structure, structure_to_CB_matrix)
    shifts = np.array(
        [[k, l, m] for k in range(-1, 2) for l in range(-1, 2) for m in range(-1, 2)],
        dtype=np.float64,
    )
    Bdists = get_minimum_shift_distances(CB_mats, shifts, orth_mat)
    Ctargets = smoother(torch.Tensor(make_cristogram_targets(B, Bdists)).long()).numpy()
    np.save(f"{output_mat}_indices_C.npy", Ctargets)


def process_predicted_protein(pdb_file: str, output_mat: str) -> None:
    """
    Build distogram targets from a predicted PDB file (no crystal info).

    Saves:
        ``{output_mat}_indices.npy`` — distogram targets (L, L, S)
    """
    if not GEMMI_AVAILABLE:
        raise ImportError("gemmi is required for process_predicted_protein")

    smoother = CrystalTargetSmoother()
    structure = gemmi.read_structure(pdb_file)
    B_coords = structure_to_CB_matrix(structure)

    if len(B_coords) == 0:
        raise ValueError(f"No valid CB/CA coordinates extracted from {pdb_file}")

    Btargets = smoother(torch.Tensor(make_distogram_targets(B_coords)).long()).numpy()
    np.save(f"{output_mat}_indices.npy", Btargets)
