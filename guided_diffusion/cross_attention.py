"""
Cross-Attention modules for conditioning on auxiliary modalities (e.g., PET scans)

This module implements cross-attention mechanisms that allow the UNet to attend
to external conditioning information (like PET probability maps) at various
resolution levels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
    normalization,
)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block that allows features to attend to conditioning information.

    This is similar to the cross-attention used in Stable Diffusion for text conditioning,
    but adapted for spatial conditioning (e.g., PET scans as probability maps).

    Args:
        channels: Number of channels in the main feature map
        context_channels: Number of channels in the conditioning context (PET)
        num_heads: Number of attention heads
        num_head_channels: Channels per head (alternative to num_heads)
        use_checkpoint: Whether to use gradient checkpointing
    """

    def __init__(
        self,
        channels,
        context_channels=None,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels or channels
        self.use_checkpoint = use_checkpoint

        # Determine number of heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.head_dim = channels // self.num_heads

        # Normalization layers
        self.norm = normalization(channels)

        # For context normalization: handle single-channel PET (not divisible by 32)
        if self.context_channels < 32:
            # Project to 32 channels first, then normalize
            self.context_preproc = nn.Sequential(
                conv_nd(1, self.context_channels, 32, 1),
            )
            self.context_norm = normalization(32)
            context_dim_for_kv = 32
        else:
            self.context_preproc = None
            self.context_norm = normalization(self.context_channels)
            context_dim_for_kv = self.context_channels

        # Query projection from main features
        self.q_proj = conv_nd(1, channels, channels, 1)

        # Key and Value projections from context (PET)
        self.k_proj = conv_nd(1, context_dim_for_kv, channels, 1)
        self.v_proj = conv_nd(1, context_dim_for_kv, channels, 1)

        # Output projection (initialized to zero for stable training)
        self.out_proj = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context, gate=None):
        """
        Apply cross-attention with optional gating.

        Args:
            x: Main features [B, C, H, W]
            context: Conditioning context (PET) [B, C_ctx, H_ctx, W_ctx]
                    Note: context can have different spatial dimensions
            gate: Optional gate value [B, 1, 1, 1] to scale cross-attention output
                  When gate=0, output = x (no cross-attention effect)
                  When gate=1, output = x + attn (full cross-attention)

        Returns:
            Output features [B, C, H, W] (same shape as input x)
        """
        return checkpoint(
            self._forward, (x, context, gate), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, context, gate=None):
        """Internal forward pass (for gradient checkpointing)"""
        b, c, h, w = x.shape
        b_ctx, c_ctx, h_ctx, w_ctx = context.shape

        assert b == b_ctx, "Batch size mismatch between x and context"

        # Reshape to [B, C, HW] for attention
        x_flat = x.reshape(b, c, h * w)
        context_flat = context.reshape(b_ctx, c_ctx, h_ctx * w_ctx)

        # Normalize
        x_norm = self.norm(x_flat)

        # Preprocess context if needed (for single-channel PET)
        if self.context_preproc is not None:
            context_flat = self.context_preproc(context_flat)
        context_norm = self.context_norm(context_flat)

        # Project to Q, K, V
        q = self.q_proj(x_norm)         # [B, C, HW]
        k = self.k_proj(context_norm)   # [B, C, H_ctx*W_ctx]
        v = self.v_proj(context_norm)   # [B, C, H_ctx*W_ctx]

        # Reshape for multi-head attention
        # [B, C, HW] -> [B, num_heads, head_dim, HW] -> [B, num_heads, HW, head_dim]
        q = q.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        k = k.reshape(b, self.num_heads, self.head_dim, h_ctx * w_ctx).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, h_ctx * w_ctx).transpose(2, 3)

        # Scaled dot-product attention
        # Q: [B, num_heads, HW, head_dim]
        # K: [B, num_heads, H_ctx*W_ctx, head_dim]
        # Attention weights: [B, num_heads, HW, H_ctx*W_ctx]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # FP16 stability: compute softmax in FP32 to avoid overflow/underflow
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type(q.dtype)

        # Apply attention to values
        # [B, num_heads, HW, H_ctx*W_ctx] @ [B, num_heads, H_ctx*W_ctx, head_dim]
        # -> [B, num_heads, HW, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to [B, C, HW]
        attn_output = attn_output.transpose(2, 3).reshape(b, c, h * w)

        # Output projection
        out = self.out_proj(attn_output)

        # Apply gate to cross-attention output (before residual connection)
        # gate=0 -> output = x (no cross-attention effect)
        # gate=1 -> output = x + out (full cross-attention)
        if gate is not None:
            # gate: [B, 1, 1, 1] -> [B, 1, 1] for broadcasting with [B, C, HW]
            gate_flat = gate.view(b, 1, 1)
            out = gate_flat * out

        # Residual connection and reshape back to spatial
        out = (x_flat + out).reshape(b, c, h, w)

        return out


class SpatialCrossAttentionBlock(nn.Module):
    """
    Spatially-aware cross-attention that preserves spatial structure.

    This variant first downsamples the context to match the feature resolution
    before applying cross-attention, making it more efficient and spatially aligned.

    Useful when PET and feature maps should be spatially aligned (e.g., both 512x512).
    """

    def __init__(
        self,
        channels,
        context_channels=None,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        adaptive_pool=True,  # Whether to adaptively pool context to match feature size
    ):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels or channels
        self.use_checkpoint = use_checkpoint
        self.adaptive_pool = adaptive_pool

        # Determine number of heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.head_dim = channels // self.num_heads

        # Context projection to match feature channels
        # First project context to a channels that's divisible by 32 for GroupNorm
        # For single-channel PET, project to 32 first
        intermediate_channels = max(32, self.context_channels)

        if self.context_channels < 32:
            # Project 1-channel PET to 32 channels first (without normalization)
            self.context_proj = nn.Sequential(
                conv_nd(2, self.context_channels, intermediate_channels, 1),
                normalization(intermediate_channels),
                nn.SiLU(),
                conv_nd(2, intermediate_channels, channels, 1),
            )
        else:
            # Normal case: context has enough channels for GroupNorm
            self.context_proj = nn.Sequential(
                normalization(self.context_channels),
                conv_nd(2, self.context_channels, channels, 1),
            )

        # Normalization (for concatenated x + context)
        self.norm = normalization(channels * 2)

        # QKV projections (all from same channel count now)
        self.qkv = conv_nd(1, channels * 2, channels * 3, 1)  # 2x input (x + context)

        # Output projection
        self.out_proj = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context, gate=None):
        """
        Args:
            x: Main features [B, C, H, W]
            context: PET context [B, C_ctx, H_ctx, W_ctx]
            gate: Optional gate value [B, 1, 1, 1] to scale cross-attention output
                  When gate=0, output = x (no cross-attention effect)
                  When gate=1, output = x + attn (full cross-attention)

        Returns:
            Output [B, C, H, W]
        """
        return checkpoint(
            self._forward, (x, context, gate), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, context, gate=None):
        b, c, h, w = x.shape

        # Adaptively pool context to match feature spatial size
        if self.adaptive_pool:
            context = F.adaptive_avg_pool2d(context, (h, w))

        # Project context to match feature channels
        context_proj = self.context_proj(context)  # [B, C, H, W]

        # Concatenate x and context_proj along channel dimension
        x_ctx = torch.cat([x, context_proj], dim=1)  # [B, 2C, H, W]

        # Reshape to [B, 2C, HW]
        x_ctx_flat = x_ctx.reshape(b, c * 2, h * w)
        x_flat = x.reshape(b, c, h * w)

        # Normalize and project to QKV
        qkv = self.qkv(self.norm(x_ctx_flat))  # [B, 3C, HW]

        # Split into Q, K, V
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # FP16 stability: compute softmax in FP32 to avoid overflow/underflow
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type(q.dtype)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(2, 3).reshape(b, c, h * w)

        # Output projection
        out = self.out_proj(attn_output)

        # Apply gate to cross-attention output (before residual connection)
        # gate=0 -> output = x (no cross-attention effect)
        # gate=1 -> output = x + out (full cross-attention)
        if gate is not None:
            # gate: [B, 1, 1, 1] -> [B, 1, 1] for broadcasting with [B, C, HW]
            gate_flat = gate.view(b, 1, 1)
            out = gate_flat * out

        # Residual connection
        out = (x_flat + out).reshape(b, c, h, w)

        return out


class PETConditionedBlock(nn.Module):
    """
    Combined ResBlock + CrossAttention block for PET conditioning.

    This wraps a ResBlock followed by a CrossAttentionBlock, allowing
    easy insertion into the UNet architecture.
    """

    def __init__(
        self,
        resblock,
        cross_attn_block,
    ):
        super().__init__()
        self.resblock = resblock
        self.cross_attn = cross_attn_block

    def forward(self, x, emb, context=None):
        """
        Args:
            x: Input features [B, C, H, W]
            emb: Timestep embeddings [B, emb_dim]
            context: PET conditioning [B, C_ctx, H_ctx, W_ctx]

        Returns:
            Output features [B, C, H, W]
        """
        # First apply ResBlock with timestep conditioning
        x = self.resblock(x, emb)

        # Then apply cross-attention if context is provided
        if context is not None:
            x = self.cross_attn(x, context)

        return x
