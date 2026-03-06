"""
Semantic Time Warp Module for HADSB

This module implements semantic-aware time warping for diffusion models.
The warp function g(t; region, organs) -> s allows different anatomical
regions to follow different noise schedules.

Key properties:
- Monotonicity: g is strictly increasing (guaranteed by sigmoid-logit form)
- Boundary preservation: g(0) ≈ 0, g(1) ≈ 1
- Identity initialization: g(t) ≈ t at start of training
- Graceful fallback: works with missing organ annotations

Mathematical formulation:
    g(t; r, S) = sigmoid(a(e) * logit(t) + b(e))

    where:
    - e = concat(e_r, e_S) is the semantic embedding
    - e_r = Embed(region_id) is the region embedding
    - e_S = AttentionPool(e_r, organ_embeddings, organ_mask) is the organ set embedding
    - a > 0 controls the "speed" of the warp (a > 1 stretches endpoints, a < 1 compresses)
    - b controls the "shift" of the warp

Reference:
    This is designed for use with HADSB (Image-to-Image Schrödinger Bridge)
    to enable per-sample adaptive noise schedules based on anatomical context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class SemanticTimeWarp(nn.Module):
    """
    Semantic-aware monotonic time warping.

    g(t; region, organs) -> s ∈ [0, 1]

    The warp is applied to the continuous time coordinate, and the warped
    time is used to interpolate all schedule tables (std_fwd, mu_x0, etc.).
    This creates a per-sample adaptive diffusion schedule.

    Args:
        num_regions: Number of body part/region classes
        num_organs: Number of organ classes
        embed_dim: Embedding dimension for regions and organs
        hidden_dim: Hidden dimension for attention and MLP
        a_scale: Scale for tanh parameterization of 'a', controls warp range
                 a ∈ [1 - a_scale, 1 + a_scale]
    """

    def __init__(
        self,
        num_regions: int,
        num_organs: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        a_scale: float = 0.5,
    ):
        super().__init__()

        self.num_regions = num_regions
        self.num_organs = num_organs
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.a_scale = a_scale

        # === Region Embedding ===
        self.region_embed = nn.Embedding(num_regions, embed_dim)
        nn.init.normal_(self.region_embed.weight, std=0.02)

        # === Organ Embedding ===
        self.organ_embed = nn.Embedding(num_organs, embed_dim)
        nn.init.normal_(self.organ_embed.weight, std=0.02)

        # === Attention Pooling for Organs ===
        # Query from region, Key from organs
        self.Wq = nn.Linear(embed_dim, hidden_dim)
        self.Wk = nn.Linear(embed_dim, hidden_dim)

        # === Warp Parameter Prediction MLP ===
        # Input: concat(e_r, e_S) of dim 2 * embed_dim
        # Output: (a_raw, b) of dim 2
        self.warp_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Identity initialization
        self._init_identity()

    def _init_identity(self):
        """
        Initialize to identity warp: g(t) = t.

        With tanh parameterization: a = 1 + a_scale * tanh(a_raw)
        At init, a_raw = 0 -> a = 1
        At init, b = 0 -> no shift
        """
        # Zero-init the final layer so output ≈ bias
        nn.init.zeros_(self.warp_mlp[-1].weight)
        nn.init.zeros_(self.warp_mlp[-1].bias)
        # Now: a_raw = 0, b = 0 -> a = 1, b = 0 -> g(t) = sigmoid(1 * logit(t) + 0) = t

    def _compute_organ_embedding(
        self,
        e_r: torch.Tensor,
        organ_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention-pooled organ embedding with graceful fallback.

        Args:
            e_r: [B, embed_dim] region embeddings
            organ_mask: [B, num_organs] multi-hot mask, or None

        Returns:
            e_S: [B, embed_dim] aggregated organ embedding
            attn: [B, num_organs] attention weights (for logging), or None
        """
        B, D = e_r.shape
        device = e_r.device

        # No organ info at all -> zero embedding
        if organ_mask is None:
            return torch.zeros(B, D, device=device), None

        # Check which samples have at least one organ
        has_organs = organ_mask.any(dim=1)  # [B] bool

        # Initialize output as zeros (fallback for empty organ sets)
        e_S = torch.zeros(B, D, device=device)
        attn_out = torch.zeros(B, self.num_organs, device=device)

        if not has_organs.any():
            # All samples have empty organ sets
            return e_S, attn_out

        # === Process only samples with organs ===
        valid_idx = has_organs.nonzero(as_tuple=True)[0]

        e_r_valid = e_r[valid_idx]           # [B_valid, D]
        mask_valid = organ_mask[valid_idx]   # [B_valid, num_organs]

        # Query from region embedding
        q = self.Wq(e_r_valid)  # [B_valid, H]

        # Keys from all organ embeddings
        all_organs = self.organ_embed.weight  # [num_organs, D]
        k = self.Wk(all_organs)               # [num_organs, H]

        # Attention scores: [B_valid, num_organs]
        scores = torch.matmul(q, k.T) / math.sqrt(self.hidden_dim)

        # Mask out absent organs (use -1e9 instead of -inf for numerical stability)
        scores = scores.masked_fill(~mask_valid.bool(), -1e9)

        # Softmax over present organs only
        attn = F.softmax(scores, dim=-1)  # [B_valid, num_organs]

        # Weighted sum of organ embeddings
        e_S_valid = torch.matmul(attn, all_organs)  # [B_valid, D]

        # Scatter back to full batch
        e_S[valid_idx] = e_S_valid
        attn_out[valid_idx] = attn

        return e_S, attn_out

    def forward(
        self,
        t: torch.Tensor,
        region_id: torch.Tensor,
        organ_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply semantic time warp.

        Args:
            t: [B] normalized time in [0, 1]. MUST be 1D.
            region_id: [B] region/body-part indices
            organ_mask: [B, num_organs] multi-hot organ mask, or None

        Returns:
            s: [B] warped time in [0, 1]
            info: dict containing:
                - 'a': [B] warp slope parameter
                - 'b': [B] warp shift parameter
                - 'e_r': [B, D] region embeddings
                - 'e_S': [B, D] organ embeddings
                - 'attn': [B, num_organs] organ attention weights (or None)
        """
        assert t.dim() == 1, f"t must be 1D [B], got shape {t.shape}"
        assert region_id.dim() == 1, f"region_id must be 1D [B], got shape {region_id.shape}"

        B = t.shape[0]

        # === Step 1: Get region embedding ===
        e_r = self.region_embed(region_id)  # [B, D]

        # === Step 2: Get organ embedding via attention pooling ===
        e_S, attn = self._compute_organ_embedding(e_r, organ_mask)  # [B, D], [B, num_organs]

        # === Step 3: Combine embeddings ===
        e = torch.cat([e_r, e_S], dim=-1)  # [B, 2D]

        # === Step 4: Predict warp parameters ===
        params = self.warp_mlp(e)  # [B, 2]
        a_raw, b = params[:, 0], params[:, 1]

        # a = 1 + a_scale * tanh(a_raw), ensuring a ∈ [1-scale, 1+scale]
        # At init: a_raw = 0 -> a = 1 (identity)
        a = 1.0 + self.a_scale * torch.tanh(a_raw)

        # === Step 5: Apply monotonic warp ===
        # g(t) = sigmoid(a * logit(t) + b)
        # Clamp t to avoid log(0) or division by zero
        t_clamped = t.clamp(1e-6, 1 - 1e-6)

        # logit(t) = log(t / (1 - t))
        logit_t = torch.log(t_clamped / (1 - t_clamped))

        # Clamp logit for numerical stability (esp. in FP16)
        logit_t = logit_t.clamp(-15, 15)

        # Apply warp
        s = torch.sigmoid(a * logit_t + b)

        return s, {
            'a': a,
            'b': b,
            'e_r': e_r,
            'e_S': e_S,
            'attn': attn,
        }

    def warp_grid(
        self,
        t_grid: torch.Tensor,
        region_id: torch.Tensor,
        organ_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Warp an entire time grid at once (efficient for sampling).

        Args:
            t_grid: [num_steps] normalized time grid in [0, 1]
            region_id: [B] region indices
            organ_mask: [B, num_organs] organ mask

        Returns:
            s_grid: [B, num_steps] warped time grid
        """
        B = region_id.shape[0]
        num_steps = t_grid.shape[0]
        device = region_id.device

        # Get embeddings (once per batch, not per step)
        e_r = self.region_embed(region_id)  # [B, D]
        e_S, _ = self._compute_organ_embedding(e_r, organ_mask)  # [B, D]
        e = torch.cat([e_r, e_S], dim=-1)  # [B, 2D]

        # Predict warp parameters (once per sample)
        params = self.warp_mlp(e)  # [B, 2]
        a = 1.0 + self.a_scale * torch.tanh(params[:, 0])  # [B]
        b = params[:, 1]  # [B]

        # Expand t_grid to [B, num_steps]
        t_expanded = t_grid.unsqueeze(0).expand(B, -1)  # [B, num_steps]
        t_clamped = t_expanded.clamp(1e-6, 1 - 1e-6)

        # Compute logit
        logit_t = torch.log(t_clamped / (1 - t_clamped)).clamp(-15, 15)  # [B, num_steps]

        # Apply warp: broadcast a, b from [B] to [B, num_steps]
        s_grid = torch.sigmoid(a.unsqueeze(-1) * logit_t + b.unsqueeze(-1))  # [B, num_steps]

        return s_grid


def compute_warp_regularization(
    a: torch.Tensor,
    b: torch.Tensor,
    lambda_a: float = 0.1,
    lambda_b: float = 0.1,
    a_target: float = 1.0,
) -> torch.Tensor:
    """
    Regularization loss to encourage identity-like warp.

    L_reg = λ_a * (a - 1)² + λ_b * b²

    With tanh parameterization, a naturally stays near 1, so this
    regularization is gentle encouragement rather than fighting init.

    Args:
        a: [B] warp slope parameters
        b: [B] warp shift parameters
        lambda_a: Weight for slope regularization
        lambda_b: Weight for shift regularization
        a_target: Target value for a (default 1.0 = identity)

    Returns:
        Scalar regularization loss
    """
    reg_a = lambda_a * ((a - a_target) ** 2).mean()
    reg_b = lambda_b * (b ** 2).mean()
    return reg_a + reg_b


def visualize_warp_curves(
    time_warp: SemanticTimeWarp,
    num_regions: int,
    device: torch.device,
    num_points: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Generate warp curves for visualization.

    Args:
        time_warp: SemanticTimeWarp module
        num_regions: Number of regions to visualize
        device: Device to use
        num_points: Number of points in the curve

    Returns:
        Dict with:
            - 't': [num_points] original time grid
            - 's': [num_regions, num_points] warped time for each region
            - 'a': [num_regions] slope parameters
            - 'b': [num_regions] shift parameters
    """
    time_warp.eval()

    t = torch.linspace(0, 1, num_points, device=device)

    s_all = []
    a_all = []
    b_all = []

    with torch.no_grad():
        for region_id in range(min(num_regions, time_warp.num_regions)):
            region_tensor = torch.full((1,), region_id, device=device, dtype=torch.long)

            # Warp with no organs (region-only)
            s_grid = time_warp.warp_grid(t, region_tensor, organ_mask=None)  # [1, num_points]

            # Get parameters
            e_r = time_warp.region_embed(region_tensor)
            e_S = torch.zeros_like(e_r)
            e = torch.cat([e_r, e_S], dim=-1)
            params = time_warp.warp_mlp(e)
            a = 1.0 + time_warp.a_scale * torch.tanh(params[:, 0])
            b = params[:, 1]

            s_all.append(s_grid.squeeze(0))
            a_all.append(a.item())
            b_all.append(b.item())

    return {
        't': t.cpu(),
        's': torch.stack(s_all).cpu(),  # [num_regions, num_points]
        'a': torch.tensor(a_all),
        'b': torch.tensor(b_all),
    }
