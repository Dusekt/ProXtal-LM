"""
Data handling for crystal distogram prediction.

This module provides:
- CrystalContactsDataset:  HDF5 dataset with automatic polymorph flattening
- collate_pad:              Collation with padding, chain_id, and space_group support

Polymorph Flattening
--------------------
Each protein in the HDF5 file may have multiple crystal polymorphs
(``contact_0``, ``contact_1``, …).  The dataset **flattens** these into
separate samples so that the loss function only ever sees a single target
distogram per sample.  This eliminates the need for ``min`` / ``mean`` /
``weighted`` multi-target aggregation in the loss.

Optional Fields
---------------
The dataset gracefully handles h5 files that lack ``chain_id`` or
``space_group`` data — they will simply be absent from the returned dict.

These were currently not used for training, but are included for potential future use in chain-aware masking
"""

import math
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CrystalContactsDataset(Dataset):
    """
    HDF5 dataset for crystal distogram prediction.

    Each item returns a **single** polymorph (protein + crystal form).
    Polymorphs are automatically flattened into separate indices.

    Expected HDF5 layout per protein group::

        h5[str(i)] = {
            'embedding':     [L, D],       # ESM-2 embedding
            'contact_0':     [L, L],       # crystal distogram (polymorph 0)
            'contact_1':     [L, L],       # crystal distogram (polymorph 1)  (optional)
            'contact_og_0':  [L, L],       # original / PDB distogram         (optional)
            'chain_id':      [L],          # chain assignment per residue      (optional)
        }
        h5[str(i)].attrs['space_group']    # int space-group ID               (optional)

    Args:
        path:     Path to the HDF5 file
        num_bins: Number of distance bins (for reference; default 64)
    """

    def __init__(self, path: str, num_bins: int = 64):
        self.path = path
        self.num_bins = num_bins
        self.file: Optional[h5py.File] = None

        # Build (protein_idx, polymorph_idx) index map
        self.index_map: List[tuple] = []
        with h5py.File(self.path, "r") as f:
            num_proteins = int(f.attrs["num_samples"])
            for i in range(num_proteins):
                group = f[str(i)]
                # Count crystal contact maps (contact_0, contact_1, …)
                n_poly = sum(
                    1
                    for k in group.keys()
                    if k.startswith("contact_") and not k.startswith("contact_og_")
                )
                n_poly = max(n_poly, 1)  # at least one sample
                for j in range(n_poly):
                    self.index_map.append((i, j))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.file is None:
            self.file = h5py.File(self.path, "r", swmr=True, libver="latest")

        protein_idx, poly_idx = self.index_map[idx]
        group = self.file[str(protein_idx)]

        # 1. Embedding
        embedding = torch.from_numpy(group["embedding"][:])  # [L, D]
        L = embedding.shape[0]
        mask = torch.ones(L, dtype=torch.bool)

        # 2. Crystal contact map (single polymorph)
        contact_key = f"contact_{poly_idx}"
        if contact_key in group:
            contact = torch.from_numpy(group[contact_key][:].astype(np.int64))
        else:
            # Fallback: first available contact map
            contact = torch.from_numpy(group["contact_0"][:].astype(np.int64))

        # 3. Original contact map (if present)
        contact_og = None
        if "contact_og_0" in group:
            contact_og = torch.from_numpy(group["contact_og_0"][:].astype(np.int64))

        # 4. Chain ID (if present)
        chain_id = None
        # Try polymorph-specific first, then generic
        if f"chain_id_{poly_idx}" in group:
            chain_id = torch.from_numpy(group[f"chain_id_{poly_idx}"][:].astype(np.int64))
        elif "chain_id" in group:
            chain_id = torch.from_numpy(group["chain_id"][:].astype(np.int64))

        # 5. Space group (if present)
        space_group = 0
        if "space_group" in group.attrs:
            space_group = int(group.attrs["space_group"])

        result: Dict[str, Any] = {
            "embedding": embedding,
            "mask": mask,
            "contact": contact,           # [L, L]
            "length": torch.tensor(L, dtype=torch.long),
            "space_group": torch.tensor(space_group, dtype=torch.long),
            "protein_idx": torch.tensor(protein_idx, dtype=torch.long),
        }
        if contact_og is not None:
            result["contact_og"] = contact_og  # [L, L]
        if chain_id is not None:
            result["chain_id"] = chain_id      # [L]

        # --- All polymorphs (for multi-target matching) ---
        all_contacts = [contact]
        for k in sorted(group.keys()):
            if k.startswith("contact_") and not k.startswith("contact_og_"):
                j = int(k.split("_")[1])
                if j != poly_idx:
                    all_contacts.append(
                        torch.from_numpy(group[k][:].astype(np.int64))
                    )
        result["all_contacts"] = all_contacts  # list of [L, L] tensors

        return result


def collate_pad(batch :List[Dict[str, Any]], pad_multiple: int = 1) -> Dict[str, Any]:
    """
    Collate samples into a padded batch.

    Handles variable-length sequences by padding to the longest in the batch
    (optionally rounded up to ``pad_multiple``).

    All contact / chain_id tensors are padded with ``-1`` to distinguish
    from valid data.

    Args:
        batch:        List of samples from CrystalContactsDataset
        pad_multiple: Round max_len up to nearest multiple (1 = no rounding)

    Returns:
        Dictionary with:
        - embedding:   [B, max_len, D]
        - mask:        [B, max_len]
        - contact:     [B, max_len, max_len]   (padded with -1)
        - lengths:     [B]
        - space_group: [B]
        - contact_og:  [B, max_len, max_len]   (if present; padded with -1)
        - chain_id:    [B, max_len]             (if present; padded with -1)
    """
    B = len(batch)
    lengths = [b["length"].item() for b in batch]
    max_len = max(lengths)

    if pad_multiple > 1:
        max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple

    D = batch[0]["embedding"].shape[-1]

    emb_batch     = torch.zeros(B, max_len, D, dtype=batch[0]["embedding"].dtype)
    mask_batch    = torch.zeros(B, max_len, dtype=torch.bool)
    contact_batch = torch.full((B, max_len, max_len), -1, dtype=torch.long)
    sg_batch      = torch.zeros(B, dtype=torch.long)

    has_og  = any("contact_og" in b for b in batch)
    has_cid = any("chain_id" in b for b in batch)

    og_batch  = torch.full((B, max_len, max_len), -1, dtype=torch.long) if has_og  else None
    cid_batch = torch.full((B, max_len),          -1, dtype=torch.long) if has_cid else None

    for i, b in enumerate(batch):
        L = b["length"].item()
        emb_batch[i, :L]        = b["embedding"]
        mask_batch[i, :L]       = b["mask"]
        contact_batch[i, :L, :L] = b["contact"]
        sg_batch[i]             = b.get("space_group", torch.tensor(0))

        if has_og and "contact_og" in b:
            og_batch[i, :L, :L] = b["contact_og"]
        if has_cid and "chain_id" in b:
            cid_batch[i, :L] = b["chain_id"]

    result: Dict[str, Any] = {
        "embedding":   emb_batch,
        "mask":        mask_batch,
        "contact":     contact_batch,
        "lengths":     torch.tensor(lengths, dtype=torch.long),
        "space_group": sg_batch,
    }
    if og_batch is not None:
        result["contact_og"] = og_batch
    if cid_batch is not None:
        result["chain_id"] = cid_batch

    # --- multi-target contacts ---
    has_multi = any("all_contacts" in b for b in batch)
    if has_multi:
        max_c = max(len(b.get("all_contacts", [b["contact"]])) for b in batch)
        multi_batch = torch.full(
            (B, max_c, max_len, max_len), -1, dtype=torch.long
        )
        multi_mask = torch.zeros(B, max_c, dtype=torch.bool)
        for i, b in enumerate(batch):
            all_c = b.get("all_contacts", [b["contact"]])
            Li = b["length"].item()
            for c_idx, ct in enumerate(all_c):
                multi_batch[i, c_idx, :Li, :Li] = ct
                multi_mask[i, c_idx] = True
        result["all_contacts"] = multi_batch   # [B, C, max_len, max_len]
        result["target_mask"] = multi_mask     # [B, C]

    # --- protein index ---
    has_pidx = any("protein_idx" in b for b in batch)
    if has_pidx:
        result["protein_idx"] = torch.stack(
            [b.get("protein_idx", torch.tensor(-1, dtype=torch.long)) for b in batch]
        )

    return result
