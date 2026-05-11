"""
Neural network architectures for crystal distogram prediction.

V2 Architecture features:
- Pair-biased self-attention in the sequence encoder (Evoformer-style)
- Relative positional encoding with chain-awareness for multi-chain crystals (unused)
- Space group embedding injected into sequence representation
- Memory-safe recycling loop with detached intermediate representations
- Windowed axial attention for memory efficiency with long multi-chain sequences
- Triangle multiplicative updates for geometric consistency
"""

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils import make_pair_mask


# ================================================================
# Positional & Identity Encodings
# ================================================================

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for the 2D pair representation.

    Combines:
    - Relative sequence position embedding, clipped to [-max_rel_pos, max_rel_pos]
    - Binary same-chain / different-chain embedding (when chain_id is provided)

    The same-chain indicator lets the model distinguish intra-chain and
    inter-chain residue pairs, which is critical for crystal lattice modelling.

    Args:
        d_pair:      Pair feature dimension
        max_rel_pos: Maximum relative position (positions beyond are clipped)
    """

    def __init__(self, d_pair: int, max_rel_pos: int = 32):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        self.rel_pos_emb = nn.Embedding(2 * max_rel_pos + 1, d_pair)
        # 0 = different chain, 1 = same chain
        self.same_chain_emb = nn.Embedding(2, d_pair)

    def forward(
        self,
        L: int,
        chain_id: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Args:
            L:        Sequence length
            chain_id: [B, L] integer chain assignments (optional)
            device:   Torch device

        Returns:
            [L, L, d_pair]    when chain_id is None
            [B, L, L, d_pair] when chain_id is provided
        """
        idx = torch.arange(L, device=device)
        rel_pos = idx.unsqueeze(1) - idx.unsqueeze(0)                    # [L, L]
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos)
        rel_pos = rel_pos + self.max_rel_pos                              # shift → [0, 2*max]
        pos_feat = self.rel_pos_emb(rel_pos)                              # [L, L, d_pair]

        if chain_id is not None:
            # chain_id: [B, L]
            same = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).long()  # [B, L, L]
            chain_feat = self.same_chain_emb(same)                         # [B, L, L, d_pair]
            return pos_feat.unsqueeze(0) + chain_feat                      # [B, L, L, d_pair]

        return pos_feat  # [L, L, d_pair]


class SpaceGroupEmbedding(nn.Module):
    """
    Embed the crystal space group ID and add it to the 1D sequence representation.

    There are 230, 65 for proteins crystallographic space groups; index 0 is reserved for
    "unknown / not provided".

    Args:
        num_space_groups: Total number of space-group entries (default 66)
        d_model:          Sequence feature dimension
    """

    def __init__(self, num_space_groups: int = 66, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_space_groups, d_model, padding_idx=0)

    def forward(self, space_group_id: torch.Tensor, seq_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            space_group_id: [B] integer tensor (0 = unknown)
            seq_repr:       [B, L, d_model]

        Returns:
            [B, L, d_model] with space-group embedding added
        """
        sg_emb = self.embedding(space_group_id)  # [B, d_model]
        return seq_repr + sg_emb.unsqueeze(1)


# ================================================================
# Pair-Biased Self-Attention
# ================================================================

class PairBiasedSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional pair-representation bias.

    The 2D pair representation is projected to *per-head scalar* biases that
    are added to the attention logits (QK^T / sqrt(d)).  This mechanism lets
    the current geometric/distance information modulate which sequence
    positions attend to each other — the same trick used in AlphaFold-2's
    Evoformer.

    Args:
        d_model: Sequence feature dimension
        n_heads: Number of attention heads
        d_pair:  Pair feature dimension (None → no pair bias)
        dropout: Attention dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, d_pair: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pair_bias_proj: Optional[nn.Linear] = None
        if d_pair is not None:
            self.pair_bias_proj = nn.Linear(d_pair, n_heads, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, L, d_model]
            pair_repr:        [B, L, L, d_pair] optional pair features
            key_padding_mask: [B, L]  True for **padding** positions

        Returns:
            [B, L, d_model]
        """
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        k = self.k_proj(x).reshape(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, H, Dh).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # --- pair bias ---
        if pair_repr is not None and self.pair_bias_proj is not None:
            pair_bias = self.pair_bias_proj(pair_repr)          # [B, L, L, H]
            attn = attn + pair_bias.permute(0, 3, 1, 2)        # [B, H, L, L]

        # --- padding mask ---
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),     # [B, 1, 1, L]
                float("-inf"),
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                             # [B, H, L, Dh]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class PairBiasedSeqEncoderLayer(nn.Module):
    """Pre-norm Transformer layer with optional pair-representation bias."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_pair: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4

        self.self_attn = PairBiasedSelfAttention(d_model, n_heads, d_pair, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), pair_repr, key_padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ================================================================
# Sequence Encoder
# ================================================================

class SimpleSeqEncoder(nn.Module):
    """
    Pair-biased Transformer encoder (layers only — no input projection).

    The input projection is handled externally by the model so that it can
    be computed once and reused across recycling iterations.

    Args:
        d_model:  Sequence hidden dimension
        d_pair:   Pair dimension for pair-biased attention (None to disable)
        n_layers: Number of Transformer encoder layers
        n_heads:  Number of attention heads
        dropout:  Dropout probability
    """

    def __init__(
        self,
        d_model: int = 256,
        d_pair: Optional[int] = None,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PairBiasedSeqEncoderLayer(d_model, n_heads, d_pair, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pad_mask = ~seq_mask if seq_mask is not None else None
        for layer in self.layers:
            x = layer(x, pair_repr, pad_mask)
        return self.norm(x)


# ================================================================
# Pair Initialization & Update
# ================================================================

class PairInit(nn.Module):
    """
    Construct initial 2D pair representation from 1D sequence features
    using broadcasting and concatenation.

    Uses symmetric operations (element-wise sum and product) to ensure
    pair[i,j] == pair[j,i].

    Args:
        d_model: Sequence feature dimension
        d_pair:  Pair feature dimension
    """

    def __init__(self, d_model: int, d_pair: int):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_pair)
        self.proj_k = nn.Linear(d_model, d_pair)
        self.out_proj = nn.Sequential(
            nn.Linear(d_pair * 2, d_pair),
            nn.LayerNorm(d_pair),
            nn.ReLU(),
            nn.Linear(d_pair, d_pair),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [B, L, d_model]

        Returns:
            [B, L, L, d_pair]
        """
        B, L, _ = seq.shape
        Q = self.proj_q(seq)  # [B, L, d_pair]
        K = self.proj_k(seq)

        Qi = Q.unsqueeze(2).expand(-1, -1, L, -1)
        Kj = K.unsqueeze(1).expand(-1, L, -1, -1)

        feat_sum  = Qi + Kj
        feat_prod = Qi * Kj
        pair = torch.cat([feat_sum, feat_prod], dim=-1)  # [B, L, L, 2·d_pair]
        return self.out_proj(pair)


class PairSeqUpdate(nn.Module):
    """
    Residual update of the pair representation from (updated) sequence features.

    Used after the sequence encoder runs so that the pair representation
    reflects the latest sequence state.  Employs a gated outer-product.

    Args:
        d_model: Sequence feature dimension
        d_pair:  Pair feature dimension
    """

    def __init__(self, d_model: int, d_pair: int):
        super().__init__()
        self.proj_left = nn.Linear(d_model, d_pair)
        self.proj_right = nn.Linear(d_model, d_pair)
        self.gate = nn.Sequential(
            nn.Linear(d_pair, d_pair),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_pair)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [B, L, d_model]

        Returns:
            [B, L, L, d_pair]
        """
        left  = self.proj_left(seq)    # [B, L, d_pair]
        right = self.proj_right(seq)
        update = left.unsqueeze(2) * right.unsqueeze(1)   # [B, L, L, d_pair]
        return self.gate(self.norm(update)) * update


# ================================================================
# Triangular Multiplicative Updates
# ================================================================

class TriangleMultiplicationOutgoing(nn.Module):
    """
    Outgoing triangle update: z_ij = Σ_k  a_ik · b_jk

    Enforces geometric consistency: if i→k and j→k are both contacts,
    then i→j is likely a contact too.

    Args:
        d_pair:   Pair feature dimension
        d_hidden: Hidden dimension for projections
    """

    def __init__(self, d_pair: int, d_hidden: int = 128):
        super().__init__()
        self.norm       = nn.LayerNorm(d_pair)
        self.left_proj  = nn.Linear(d_pair, d_hidden)
        self.right_proj = nn.Linear(d_pair, d_hidden)
        self.left_gate  = nn.Linear(d_pair, d_hidden)
        self.right_gate = nn.Linear(d_pair, d_hidden)
        self.out_gate   = nn.Linear(d_pair, d_pair)
        self.out_proj   = nn.LayerNorm(d_hidden)
        self.final_proj = nn.Linear(d_hidden, d_pair)

    def forward(self, pair: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(pair)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        left  = torch.sigmoid(self.left_gate(x))  * self.left_proj(x)
        right = torch.sigmoid(self.right_gate(x)) * self.right_proj(x)

        update = torch.einsum("bikc,bjkc->bijc", left, right)
        update = self.final_proj(self.out_proj(update))

        g = torch.sigmoid(self.out_gate(x))
        return pair + g * update


class TriangleMultiplicationIncoming(nn.Module):
    """
    Incoming triangle update: z_ij = Σ_k  a_ki · b_kj

    Complementary to the outgoing update.

    Args:
        d_pair:   Pair feature dimension
        d_hidden: Hidden dimension for projections
    """

    def __init__(self, d_pair: int, d_hidden: int = 128):
        super().__init__()
        self.norm       = nn.LayerNorm(d_pair)
        self.left_proj  = nn.Linear(d_pair, d_hidden)
        self.right_proj = nn.Linear(d_pair, d_hidden)
        self.left_gate  = nn.Linear(d_pair, d_hidden)
        self.right_gate = nn.Linear(d_pair, d_hidden)
        self.out_gate   = nn.Linear(d_pair, d_pair)
        self.out_proj   = nn.LayerNorm(d_hidden)
        self.final_proj = nn.Linear(d_hidden, d_pair)

    def forward(self, pair: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(pair)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        left  = torch.sigmoid(self.left_gate(x))  * self.left_proj(x)
        right = torch.sigmoid(self.right_gate(x)) * self.right_proj(x)

        update = torch.einsum("bkic,bkjc->bijc", left, right)
        update = self.final_proj(self.out_proj(update))

        g = torch.sigmoid(self.out_gate(x))
        return pair + g * update


# ================================================================
# Axial Attention (with Optional Windowing)
# ================================================================

class _AxialSelfAttention(nn.Module):
    """
    Manual multi-head self-attention using F.scaled_dot_product_attention.

    Unlike nn.MultiheadAttention this has no internal fast-path, so tensor
    shapes are deterministic across the two forward passes of gradient
    checkpointing — eliminating the metadata-mismatch crash.

    Args:
        d_model: Feature dimension
        n_heads: Number of attention heads
        dropout: Attention dropout (applied only during training)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = dropout

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [N, L, D]
            key_padding_mask: [N, L] bool, True = **padding** position

        Returns:
            [N, L, D]
        """
        N, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        qkv = self.qkv_proj(x).reshape(N, L, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [N, H, L, Dh]

        # Build additive mask from key_padding_mask
        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            # [N, 1, 1, L] — broadcast over heads and query positions
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        out = out.transpose(1, 2).reshape(N, L, D)
        return self.out_proj(out)


class AxialAttentionBlock(nn.Module):
    """
    Row + column axial attention on the 2D pair matrix.

    Uses a manual SDPA-based attention implementation that is fully
    compatible with ``torch.utils.checkpoint`` (no fast-path shape
    mismatches).

    Supports two modes:
    - **Full attention** (default): every position in a row (or column)
      attends to all others.  Chunking over rows reduces peak memory.
    - **Windowed attention** (``window_size > 0``): non-overlapping windows.
      Reduces memory from O(L²) to O(L·W) per row.

    Args:
        d_pair:      Pair feature dimension
        n_heads:     Number of attention heads
        dropout:     Dropout probability
        chunk_size:  Batch-chunk size for attention (reduces peak memory)
        window_size: Local window size (``None`` or ``0`` → full attention)
    """

    def __init__(
        self,
        d_pair: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 16,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.row_attn = _AxialSelfAttention(d_pair, n_heads, dropout)
        self.col_attn = _AxialSelfAttention(d_pair, n_heads, dropout)
        self.norm_row = nn.LayerNorm(d_pair)
        self.norm_col = nn.LayerNorm(d_pair)
        self.dropout  = nn.Dropout(dropout)
        self.chunk_size = chunk_size
        self.window_size = window_size if (window_size and window_size > 0) else None

    # ---- internal helpers ----

    def _chunked_attention(
        self,
        attn_layer: _AxialSelfAttention,
        x: torch.Tensor,
        mask_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Full attention processed in row-chunks to cap peak memory."""
        N = x.shape[0]
        outs: list[torch.Tensor] = []
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            xc = x[i:end]
            mc = mask_flat[i:end] if mask_flat is not None else None
            outs.append(attn_layer(xc, key_padding_mask=mc))
        return torch.cat(outs, dim=0)

    def _windowed_attention(
        self,
        attn_layer: _AxialSelfAttention,
        x: torch.Tensor,
        mask_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Non-overlapping windowed attention within each row/column."""
        assert self.window_size is not None
        BL, seq_len, D = x.shape
        W = self.window_size

        pad = (W - seq_len % W) % W
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            if mask_flat is not None:
                mask_flat = F.pad(mask_flat, (0, pad), value=True)

        n_win = (seq_len + pad) // W
        x = x.reshape(BL, n_win, W, D).reshape(BL * n_win, W, D)
        m = None
        if mask_flat is not None:
            m = mask_flat.reshape(BL, n_win, W).reshape(BL * n_win, W)

        out = self._chunked_attention(attn_layer, x, m)
        out = out.reshape(BL, n_win, W, D).reshape(BL, seq_len + pad, D)
        if pad > 0:
            out = out[:, :seq_len, :]
        return out

    def _apply_attention(
        self,
        attn_layer: _AxialSelfAttention,
        x: torch.Tensor,
        mask_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.window_size is not None and x.shape[1] > self.window_size:
            return self._windowed_attention(attn_layer, x, mask_flat)
        return self._chunked_attention(attn_layer, x, mask_flat)

    # ---- forward ----

    def forward(self, pair: torch.Tensor, pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _, D = pair.shape

        # --- Row attention ---
        x = pair
        row_in = self.norm_row(x).view(B * L, L, D)
        row_mask = ~pair_mask.view(B * L, L) if pair_mask is not None else None
        row_out = self._apply_attention(self.row_attn, row_in, row_mask)
        x = x + self.dropout(row_out.view(B, L, L, D))

        # --- Column attention ---
        col_raw = x.transpose(1, 2)
        col_in  = self.norm_col(col_raw).reshape(B * L, L, D)
        col_mask = ~pair_mask.transpose(1, 2).reshape(B * L, L) if pair_mask is not None else None
        col_out = self._apply_attention(self.col_attn, col_in, col_mask)
        col_out = col_out.view(B, L, L, D).transpose(1, 2)
        x = x + self.dropout(col_out)

        return x


# ================================================================
# Main Processing Block
# ================================================================

class TriAxialBlock(nn.Module):
    """
    Combined processing block:
    1. Axial Attention (global or windowed)
    2. Outgoing Triangle Update
    3. Incoming Triangle Update
    4. Feed-Forward (channel mixing)

    Args:
        d_pair:      Pair feature dimension
        tri_hidden:  Hidden dim for triangle updates
        n_heads:     Number of attention heads
        dropout:     Dropout probability
        window_size: Local window size for axial attention (None → full)
    """

    def __init__(
        self,
        d_pair: int,
        tri_hidden: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.axial_attn  = AxialAttentionBlock(d_pair, n_heads, dropout, window_size=window_size)
        self.tri_mul_out = TriangleMultiplicationOutgoing(d_pair, tri_hidden)
        self.tri_mul_in  = TriangleMultiplicationIncoming(d_pair, tri_hidden)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair * 4),
            nn.GELU(),
            nn.Linear(d_pair * 4, d_pair),
        )

    def forward(self, pair: torch.Tensor, pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pair = self.axial_attn(pair, pair_mask)
        pair = self.tri_mul_out(pair, pair_mask)
        pair = self.tri_mul_in(pair, pair_mask)
        pair = pair + self.ff(pair)
        return pair


# ================================================================
# Output Head
# ================================================================

class PairOutputHead(nn.Module):
    """Final MLP that maps pair features to distance-bin logits."""

    def __init__(self, d_pair: int, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair),
            nn.ReLU(),
            nn.Linear(d_pair, out_ch),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.net(pair)


# ================================================================
# Full Model with Recycling
# ================================================================

class CrystalTriangularModel(nn.Module):
    """
    Crystal distogram prediction model.

    Architecture overview (per recycling iteration):
      1. Input projection  (+space-group embedding)
      2. Inject recycled 1D/2D representations  +  logit bias
      3. Pair initialisation  (+relative positional encoding)
      4. Sequence encoder with pair-biased attention
      5. Residual pair update from sequence
      6. Stack of TriAxial blocks (axial attention + triangle updates)
      7. Output head(s) → distance-bin logits

    When ``n_hypotheses > 1`` the model produces **N independent hypotheses**
    from the same shared pair representation via N lightweight output heads.
    The output shape becomes ``[B, N, L, L, out_ch]`` (vs ``[B, L, L, out_ch]``
    for the single-hypothesis default).

    Recycling: the final 1D/2D representations and distogram logits from the
    previous cycle are **detached** and fed back as biases to the next cycle.
    Non-final cycles run under ``torch.no_grad()`` to prevent OOM graph leaks.

    Args:
        emb_dim:               Input embedding dim (1280 for ESM-2)
        d_model:               Sequence hidden dim
        d_pair:                Pair hidden dim
        n_seq_layers:          Number of sequence-encoder layers
        n_blocks:              Number of TriAxial blocks
        tri_hidden:            Triangle-update hidden dim
        out_ch:                Number of distance bins
        use_checkpoint:        Use gradient checkpointing in final cycle
        n_recycles:            Number of recycling iterations (0 = single pass)
        max_rel_pos:           Clip value for relative-position encoding
        num_space_groups:      Vocabulary size for space-group embedding
        n_heads:               Attention heads in sequence encoder
        attention_window_size: Window size for axial attention (None → full)
        dropout:               Global dropout rate
        n_hypotheses:          Number of output hypotheses (1 = single prediction)
    """

    def __init__(
        self,
        emb_dim: int = 1152,
        d_model: int = 256,
        d_pair: int = 64,
        n_seq_layers: int = 3,
        n_blocks: int = 4,
        tri_hidden: int = 32,
        out_ch: int = 64,
        use_checkpoint: bool = True,
        n_recycles: int = 0,
        max_rel_pos: int = 32,
        num_space_groups: int = 231,
        n_heads: int = 8,
        attention_window_size: Optional[int] = None,
        dropout: float = 0.1,
        n_hypotheses: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair  = d_pair
        self.n_recycles = n_recycles
        self.use_checkpoint = use_checkpoint
        self.n_hypotheses = n_hypotheses

        # --- Input embedding ---
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.sg_emb = SpaceGroupEmbedding(num_space_groups, d_model)

        # --- Sequence encoder (pair-biased) ---
        self.seq_encoder = SimpleSeqEncoder(
            d_model=d_model, d_pair=d_pair, n_layers=n_seq_layers,
            n_heads=n_heads, dropout=dropout,
        )

        # --- Pair initialisation ---
        self.pair_init = PairInit(d_model, d_pair)
        self.pair_seq_update = PairSeqUpdate(d_model, d_pair)
        self.relpos_enc = RelativePositionalEncoding(d_pair, max_rel_pos)

        # --- Recycling components ---
        self.recycle_seq_norm  = nn.LayerNorm(d_model)
        self.recycle_pair_norm = nn.LayerNorm(d_pair)
        self.recycle_logit_proj = nn.Linear(out_ch, d_pair)

        # --- TriAxial blocks ---
        win = attention_window_size if (attention_window_size and attention_window_size > 0) else None
        self.blocks = nn.ModuleList([
            TriAxialBlock(d_pair, tri_hidden=tri_hidden, n_heads=4,
                          dropout=dropout, window_size=win)
            for _ in range(n_blocks)
        ])

        # --- Output head(s) ---
        self.out_norm = nn.LayerNorm(d_pair)
        if n_hypotheses > 1:
            self.out_heads = nn.ModuleList([
                PairOutputHead(d_pair, out_ch) for _ in range(n_hypotheses)
            ])
        else:
            self.out_head = PairOutputHead(d_pair, out_ch)

    def forward(
        self,
        emb: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        chain_id: Optional[torch.Tensor] = None,
        space_group: Optional[torch.Tensor] = None,
        n_recycles: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            emb:         [B, L, emb_dim] ESM-2 embeddings
            seq_mask:    [B, L] boolean (True = valid)
            chain_id:    [B, L] integer chain assignment (optional)
            space_group: [B]    integer space-group ID (optional)
            n_recycles:  Override default recycling count

        Returns:
            [B, L, L, out_ch]    when n_hypotheses == 1
            [B, N, L, L, out_ch] when n_hypotheses > 1
        """
        n_cyc = n_recycles if n_recycles is not None else self.n_recycles
        B, L, _ = emb.shape
        device = emb.device

        # ----- computed once --------------------------------------------------
        seq_init = self.input_proj(emb)
        if space_group is not None:
            seq_init = self.sg_emb(space_group, seq_init)

        pair_mask = make_pair_mask(seq_mask) if seq_mask is not None else None

        relpos = self.relpos_enc(L, chain_id, device=device)
        if relpos.dim() == 3:                                     # no chain_id
            relpos = relpos.unsqueeze(0).expand(B, -1, -1, -1)

        # ----- recycled state (initialised to zero) ---------------------------
        prev_seq    = torch.zeros(B, L, self.d_model, device=device)
        prev_pair   = torch.zeros(B, L, L, self.d_pair, device=device)
        prev_logits: Optional[torch.Tensor] = None

        # ----- recycling loop -------------------------------------------------
        for cycle in range(n_cyc + 1):
            is_last = cycle == n_cyc

            # Disable gradient graph for non-final cycles to prevent OOM
            ctx = torch.no_grad() if (not is_last and self.training) else nullcontext()

            with ctx:
                # Inject recycled representations
                seq  = seq_init + self.recycle_seq_norm(prev_seq)
                pair = (self.pair_init(seq)
                        + relpos
                        + self.recycle_pair_norm(prev_pair))

                if prev_logits is not None:
                    pair = pair + self.recycle_logit_proj(prev_logits)

                # Sequence encoder with pair-biased attention
                seq = self.seq_encoder(seq, pair_repr=pair, seq_mask=seq_mask)

                # Residual pair update from new sequence
                pair = pair + self.pair_seq_update(seq)

                # TriAxial blocks
                for block in self.blocks:
                    if self.use_checkpoint and self.training and is_last:
                        # use_reentrant=True avoids metadata-mismatch errors
                        # that occur with AMP + non-reentrant checkpointing,
                        # where SDPA and other ops may take different internal
                        # paths at different precision levels across the two
                        # forward passes.
                        pair = checkpoint(block, pair, pair_mask, use_reentrant=True)
                    else:
                        pair = block(pair, pair_mask)

                normed = self.out_norm(pair)
                if self.n_hypotheses > 1:
                    logits = torch.stack(
                        [head(normed) for head in self.out_heads], dim=1
                    )  # [B, N, L, L, out_ch]
                    # For recycling, average across hypotheses
                    recycle_logits = logits.mean(dim=1)  # [B, L, L, out_ch]
                else:
                    logits = self.out_head(normed)        # [B, L, L, out_ch]
                    recycle_logits = logits

            # Prepare detached state for next cycle
            if not is_last:
                prev_seq    = seq.detach()
                prev_pair   = pair.detach()
                prev_logits = recycle_logits.detach()

        return logits
