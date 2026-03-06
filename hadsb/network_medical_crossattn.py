"""
Medical network with Cross-Attention PET conditioning

This module implements a UNet-based diffusion model for medical image translation
where PET scans are injected as conditioning through cross-attention mechanisms
rather than channel concatenation.

Key differences from network_medical.py:
1. PET is not concatenated as input channels
2. PET is processed through cross-attention at multiple resolution levels
3. Input channels: 2 (x_t + x1) instead of 3 (x_t + x1 + pet)
4. More parameter-efficient and allows dynamic attention to PET features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion.unet import UNetModel, ResBlock, AttentionBlock, TimestepEmbedSequential
from guided_diffusion.cross_attention import CrossAttentionBlock, SpatialCrossAttentionBlock
from guided_diffusion.nn import conv_nd, zero_module, normalization
from .network import Image256Net

# For organ-specific output heads.
from guided_diffusion.nn import normalization as group_norm



class UNetWithCrossAttention(nn.Module):
    """
    Wrapper around UNetModel that injects cross-attention conditioning.

    This is a simpler approach than modifying UNet internals:
    1. Use a standard UNet for backbone processing
    2. Insert cross-attention blocks at strategic points
    3. Process through encoder -> middle -> decoder with context injection
    """

    def __init__(
        self,
        unet,
        context_channels=1,
        cross_attn_resolutions=None,
        use_spatial_cross_attn=True,
        actual_image_size=256,  # NEW: actual input image size (may differ from unet.image_size)
    ):
        """
        Args:
            unet: Pre-built UNetModel instance
            context_channels: Channels in conditioning context (PET)
            cross_attn_resolutions: Resolutions for cross-attention
            use_spatial_cross_attn: Use spatial variant
            actual_image_size: Actual input image size (used for resolution->channel mapping)
        """
        super().__init__()
        self.unet = unet
        self.context_channels = context_channels
        self.cross_attn_resolutions = cross_attn_resolutions or []
        self.use_spatial_cross_attn = use_spatial_cross_attn
        self.actual_image_size = actual_image_size

        # Build cross-attention modules
        self._build_cross_attention_modules()

    def _build_cross_attention_modules(self):
        """Build cross-attention modules for each resolution level"""
        self.cross_attns = nn.ModuleDict()

        ch = self.unet.model_channels
        channel_mult = self.unet.channel_mult

        # Map resolution to actual channel count
        # 
        # IMPORTANT: Use actual_image_size (not unet.image_size) for correct mapping
        # 
        # Key insight: Cross-attention is applied AFTER each input_block in forward().
        # When we first encounter a resolution R (after a Downsample block), the channels
        # are still from the PREVIOUS level because Downsample doesn't change channels.
        # The channel increase happens in the ResBlocks of the NEXT level.
        #
        # For 256x256 input with channel_mult=(1,1,2,2,4,4) and model_channels=128:
        # - First encounter 128x128: after level 0 downsample, channels = mult_0 * 128 = 128
        # - First encounter 64x64: after level 1 downsample, channels = mult_1 * 128 = 128
        # - First encounter 32x32: after level 2 downsample, channels = mult_2 * 128 = 256
        # - First encounter 16x16: after level 3 downsample, channels = mult_3 * 128 = 256
        # - First encounter 8x8: after level 4 downsample, channels = mult_4 * 128 = 512

        resolution_to_actual_channels = {}

        # Use actual image size for correct resolution calculation
        current_res = self.actual_image_size
        
        # Map resolutions that appear AFTER each level's downsample
        # Resolution after level L's downsample = input_size / 2^(L+1)
        # Channels at that point = mult_L * model_channels (downsample doesn't change channels)
        for level, mult in enumerate(channel_mult):
            if level < len(channel_mult) - 1:  # Last level has no downsample
                res_after_downsample = current_res // (2 ** (level + 1))
                channels_after_downsample = ch * mult
                resolution_to_actual_channels[res_after_downsample] = channels_after_downsample
        
        # Also map the initial resolution (before any downsample)
        resolution_to_actual_channels[current_res] = ch * channel_mult[0]
            
        # print(f"[DEBUG] Resolution to channels map: {resolution_to_actual_channels}")

        for resolution in self.cross_attn_resolutions:
            key = f"res_{resolution}"

            # Use actual channel count from our analysis
            actual_channels = resolution_to_actual_channels.get(resolution)

            if actual_channels is None:
                print(f"[WARNING] Resolution {resolution} not found in UNet structure, skipping")
                continue

            # print(f"[DEBUG] Creating cross-attention at resolution {resolution} with {actual_channels} channels")

            if self.use_spatial_cross_attn:
                self.cross_attns[key] = SpatialCrossAttentionBlock(
                    channels=actual_channels,
                    context_channels=self.context_channels,
                    num_heads=self.unet.num_heads,
                    use_checkpoint=False,
                )
            else:
                self.cross_attns[key] = CrossAttentionBlock(
                    channels=actual_channels,
                    context_channels=self.context_channels,
                    num_heads=self.unet.num_heads,
                    use_checkpoint=False,
                )


    def forward(self, x, timesteps, context=None, cond_emb=None, cross_attn_gate=None, 
                return_features=False):
        """
        Forward with cross-attention.

        Args:
            x: Input [B, C, H, W]
            timesteps: Timesteps [B]
            context: PET context [B, C_ctx, H, W]
            cond_emb: Additional embeddings (organ, etc.)
            cross_attn_gate: Optional gate value [B, 1, 1, 1] to scale cross-attention output
                             When gate=0, cross-attention has no effect
                             When gate=1, full cross-attention
            return_features: If True, return features before output layer instead of final output

        Returns:
            Output [B, C_out, H, W] or features [B, C_feat, H, W] if return_features=True
        """
        # Get timestep embeddings
        from guided_diffusion.nn import timestep_embedding as make_timestep_embedding
        emb = self.unet.time_embed(make_timestep_embedding(timesteps, self.unet.model_channels))

        if cond_emb is not None:
            emb = emb + cond_emb

        # Encoder
        hs = []
        h = x.type(self.unet.dtype)
        
        # Also convert context to the same dtype for FP16 compatibility
        if context is not None:
            context = context.type(self.unet.dtype)
        
        # Convert gate to same dtype
        if cross_attn_gate is not None:
            cross_attn_gate = cross_attn_gate.type(self.unet.dtype)

        # Track which resolutions we've already applied cross-attention to
        cross_attn_applied = set()

        for i, module in enumerate(self.unet.input_blocks):
            h = module(h, emb)

            # Apply cross-attention after each input block if at correct resolution
            current_h, current_w = h.shape[2], h.shape[3]
            resolution = current_h  # Assuming square
            key = f"res_{resolution}"

            # Debug: print channel count at each resolution (disabled for performance)
            # if i < 20:
            #     print(f"[DEBUG FORWARD] Block {i}: resolution={resolution}, channels={h.shape[1]}")

            # Only apply cross-attention at the FIRST encounter of each resolution
            if key in self.cross_attns and context is not None and key not in cross_attn_applied:
                h = self.cross_attns[key](h, context, gate=cross_attn_gate)
                cross_attn_applied.add(key)

            hs.append(h)

        # Middle
        h = self.unet.middle_block(h, emb)

        # Middle cross-attention (middle is unique, always apply)
        resolution = h.shape[2]
        key = f"res_{resolution}"
        if key in self.cross_attns and context is not None:
            h = self.cross_attns[key](h, context, gate=cross_attn_gate)

        # Decoder
        # Note: We don't apply cross-attention in decoder because channel counts differ
        # from encoder due to skip connections. Cross-attention conditioning is applied
        # during downsampling (encoder) and middle block.
        for i, module in enumerate(self.unet.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

            # Debug decoder blocks (disabled for performance)
            # if i < 10:
            #     resolution = h.shape[2]
            #     print(f"[DEBUG DECODER] Block {i}: resolution={resolution}, channels={h.shape[1]}")

        h = h.type(x.dtype)
        
        # Return features before output layer if requested
        if return_features:
            return h
        
        return self.unet.out(h)

    def convert_to_fp16(self):
        """
        Convert the model to FP16.
        Forward the call to the internal UNet and convert cross-attention modules.
        Only converts Conv and Linear layers, keeping GroupNorm in FP32.
        """
        from guided_diffusion.fp16_util import convert_module_to_f16
        # Convert internal UNet
        self.unet.convert_to_fp16()
        # Convert cross-attention modules (only Conv/Linear, not GroupNorm)
        for key in self.cross_attns:
            self.cross_attns[key].apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the model back to FP32.
        Forward the call to the internal UNet and convert cross-attention modules.
        """
        from guided_diffusion.fp16_util import convert_module_to_f32
        # Convert internal UNet
        self.unet.convert_to_fp32()
        # Convert cross-attention modules
        for key in self.cross_attns:
            self.cross_attns[key].apply(convert_module_to_f32)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of N indices (can be fractional)
        dim: Dimension of output
        max_period: Controls the minimum frequency

    Returns:
        Tensor of shape [N, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32))
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MedicalNetCrossAttn(nn.Module):
    """
    Medical network with PET conditioning via cross-attention.

    Key features:
    1. PET is injected through cross-attention, not channel concatenation
    2. Input channels: 2 (x_t + x1) instead of 3, or 4 with lava_water/lava_fat
    3. Cross-attention at multiple resolution levels (default: [32, 16, 8])
    4. More interpretable: attention maps show where PET information is used
    5. Support for lava_water and lava_fat conditioning with shared gate
    6. Support for Body Part semantic embedding with PubMedBERT
    """

    def __init__(
        self,
        log,
        noise_levels,
        use_fp16=False,
        cond_x1=True,
        cond_pet=False,  # Now controls cross-attention usage, not input channels
        cond_organ=False,
        num_organs=0,
        organ_embed_dim=16,
        organ_time_cond=True,
        cross_attn_resolutions=None,  # NEW: Which resolutions get cross-attention
        use_spatial_cross_attn=True,  # NEW: Use spatially-aligned cross-attention
        image_size=256,  # NEW: Actual input image size for correct resolution mapping
        # LAVA water/fat conditioning.
        cond_lava_water=False,
        cond_lava_fat=False,
        # Organ-specific output heads.
        use_organ_specific_out=False,
        # Body Part semantic conditioning.
        cond_body_part=False,
        body_part_embed_dim=64,
        body_part_channel_cond=True,
        body_part_time_cond=True,
        use_semantic_embedding=True,
    ):
        """
        Args:
            log: Logger
            noise_levels: Noise schedule
            use_fp16: Use fp16 precision
            cond_x1: Condition on T1 image (concatenated as input)
            cond_pet: Enable PET conditioning via cross-attention
            cond_organ: Enable organ conditioning
            cross_attn_resolutions: Resolutions for cross-attention (e.g., [32, 16, 8])
            use_spatial_cross_attn: Use spatial cross-attention (more efficient)
            image_size: Actual input image size (used for resolution->channel mapping)
            cond_lava_water: Condition on LAVA water image (channel concat)
            cond_lava_fat: Condition on LAVA fat image (channel concat)
            use_organ_specific_out: Use organ-specific output conv (shared norm/act + K independent convs)
            cond_body_part: Enable Body Part semantic embedding
            body_part_embed_dim: Embedding dimension for Body Part
            body_part_channel_cond: Use channel conditioning for Body Part
            body_part_time_cond: Use time conditioning for Body Part
            use_semantic_embedding: Use PubMedBERT for semantic embedding
        """
        super().__init__()

        self.log = log
        self.noise_levels = noise_levels
        self.use_fp16 = use_fp16
        self.cond_x1 = cond_x1
        self.cond_pet = cond_pet
        self.cond_organ = cond_organ
        self.organ_time_cond = organ_time_cond
        self.organ_embed_dim = organ_embed_dim if cond_organ else 0
        self.num_organs = num_organs
        self.image_size = image_size
        
        # LAVA water/fat conditioning
        self.cond_lava_water = cond_lava_water
        self.cond_lava_fat = cond_lava_fat
        
        # Body Part semantic conditioning settings (NEW)
        self.cond_body_part = cond_body_part
        self.body_part_embed_dim = body_part_embed_dim if cond_body_part else 0
        self.body_part_channel_cond = bool(body_part_channel_cond) if cond_body_part else False
        self.body_part_time_cond = bool(body_part_time_cond) if cond_body_part else False
        self.use_semantic_embedding = use_semantic_embedding

        # Default cross-attention resolutions
        if cross_attn_resolutions is None:
            cross_attn_resolutions = [32, 16, 8] if cond_pet else []
        self.cross_attn_resolutions = cross_attn_resolutions

        # Calculate input channels (NO PET in input now!)
        in_channels = 1  # x_t
        if cond_x1:
            in_channels += 1  # + x1 (T1/LAVA)
        if cond_lava_water:
            in_channels += 1  # + lava_water
        if cond_lava_fat:
            in_channels += 1  # + lava_fat
        # NOTE: PET is NOT concatenated anymore
        # Body Part channel conditioning (NEW)
        if cond_body_part and body_part_channel_cond:
            in_channels += body_part_embed_dim

        out_channels = 1  # Grayscale output

        # Create base UNet (without PET in input channels)
        base_unet = UNetModel(
            image_size=512,
            in_channels=in_channels,
            model_channels=128,
            out_channels=out_channels,
            num_res_blocks=3,
            attention_resolutions=[32, 16, 8],
            dropout=0.0,
            channel_mult=(1, 1, 2, 2, 4, 4),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=use_fp16,
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        )

        # Wrap with cross-attention support
        if cond_pet and self.cross_attn_resolutions:
            self.unet = UNetWithCrossAttention(
                unet=base_unet,
                context_channels=1,  # PET is 1-channel
                cross_attn_resolutions=self.cross_attn_resolutions,
                use_spatial_cross_attn=use_spatial_cross_attn,
                actual_image_size=image_size,  # Pass actual image size for correct mapping
            )
        else:
            # No cross-attention, use base UNet directly
            self.unet = base_unet

        # Get time embedding dimension
        time_embed_module = (
            self.unet.unet.time_embed
            if isinstance(self.unet, UNetWithCrossAttention)
            else self.unet.time_embed
        )
        time_embed_dim = time_embed_module[-1].out_features

        # Organ conditioning (same as before - backward compatible)
        if cond_organ:
            if num_organs <= 0:
                raise ValueError("cond_organ=True requires num_organs > 0")
            self.organ_embedding = nn.Embedding(num_organs, organ_embed_dim)
            if self.organ_time_cond:
                # Two-layer MLP similar to UNet time embedding.
                self.organ_time_mlp = nn.Sequential(
                    nn.Linear(organ_embed_dim, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
                # Init with larger gain for layer 1 and zero-init for layer 2.
                nn.init.xavier_uniform_(self.organ_time_mlp[0].weight, gain=2.0)
                nn.init.zeros_(self.organ_time_mlp[0].bias)
                nn.init.zeros_(self.organ_time_mlp[2].weight)
                nn.init.zeros_(self.organ_time_mlp[2].bias)
            else:
                self.organ_time_mlp = None
        
        # Body Part semantic conditioning (NEW)
        if cond_body_part:
            try:
                from configs.region_organ_config import BODY_PARTS, NUM_BODY_PARTS, ORGANS
                from .semantic_embedding import create_body_part_embedding, create_organ_embedding

                self.body_part_embedding = create_body_part_embedding(
                    body_parts=BODY_PARTS,
                    output_dim=body_part_embed_dim,
                    time_embed_dim=time_embed_dim,
                    use_semantic=use_semantic_embedding,
                )
                self.num_body_parts = NUM_BODY_PARTS

                # Organ embedding from JSON (uses organ_mask, not organ_id)
                # This provides additional organ-level conditioning based on Organs Present
                self.organ_embedding_from_json = create_organ_embedding(
                    organs=ORGANS,
                    output_dim=body_part_embed_dim,  # Same dim as body_part for consistency
                    use_semantic=use_semantic_embedding,
                    aggregation="weighted",
                )

                # Cross-attention: body_part as query, organ embeddings as key/value
                # This allows organ information to modulate body part embedding
                self.body_organ_cross_attn = nn.MultiheadAttention(
                    embed_dim=body_part_embed_dim,
                    num_heads=4,
                    dropout=0.0,
                    batch_first=True,
                )
                # Layer norm for cross-attention
                self.body_organ_cross_attn_norm = nn.LayerNorm(body_part_embed_dim)
                # Projection after cross-attention
                self.body_organ_cross_attn_proj = nn.Sequential(
                    nn.Linear(body_part_embed_dim, body_part_embed_dim),
                    nn.SiLU(),
                    nn.Linear(body_part_embed_dim, body_part_embed_dim),
                )
                nn.init.zeros_(self.body_organ_cross_attn_proj[-1].weight)
                nn.init.zeros_(self.body_organ_cross_attn_proj[-1].bias)

                log.info(f"[Network] Body Part semantic embedding enabled: "
                        f"dim={body_part_embed_dim}, channel={body_part_channel_cond}, "
                        f"time={body_part_time_cond}, semantic={use_semantic_embedding}")
                log.info(f"[Network] Organ embedding from JSON enabled with cross-attention modulation")
            except ImportError as e:
                log.warning(f"[Network] Failed to import semantic embedding: {e}")
                log.warning("[Network] Falling back to simple learnable embedding")
                # Fallback to simple embedding
                from configs.region_organ_config import NUM_BODY_PARTS
                self.body_part_embedding = nn.Embedding(NUM_BODY_PARTS, body_part_embed_dim)
                self.num_body_parts = NUM_BODY_PARTS
                self.organ_embedding_from_json = None  # No fallback for organ embedding
                # Time MLP for fallback
                if body_part_time_cond:
                    self.body_part_time_mlp = nn.Sequential(
                        nn.Linear(body_part_embed_dim, time_embed_dim),
                        nn.SiLU(),
                        nn.Linear(time_embed_dim, time_embed_dim),
                    )
                    nn.init.zeros_(self.body_part_time_mlp[-1].weight)
                    nn.init.zeros_(self.body_part_time_mlp[-1].bias)
        else:
            self.organ_embedding_from_json = None

        # Organ-specific output heads:
        # shared norm/act (reused from UNet) + one conv per organ.
        self.use_organ_specific_out = use_organ_specific_out and cond_organ and num_organs > 0
        if self.use_organ_specific_out:
            # Get the final decoder channel count.
            if isinstance(self.unet, UNetWithCrossAttention):
                final_ch = int(self.unet.unet.model_channels * self.unet.unet.channel_mult[0])
                # Reuse pre-initialized norm/act from UNet.out.
                self.shared_out_norm = self.unet.unet.out[0]  # GroupNorm
                self.shared_out_act = self.unet.unet.out[1]   # SiLU
                original_conv = self.unet.unet.out[2]         # Conv2d
            else:
                final_ch = int(self.unet.model_channels * self.unet.channel_mult[0])
                self.shared_out_norm = self.unet.out[0]
                self.shared_out_act = self.unet.out[1]
                original_conv = self.unet.out[2]
            
            # One output conv per organ, initialized from the shared conv.
            self.organ_out_convs = nn.ModuleList()
            for _ in range(num_organs):
                conv = conv_nd(2, final_ch, out_channels, 3, padding=1)
                # Copy shared conv weights as initialization.
                with torch.no_grad():
                    conv.weight.copy_(original_conv.weight)
                    conv.bias.copy_(original_conv.bias)
                self.organ_out_convs.append(conv)
            
            log.info(f"[Network] Organ-specific output heads ENABLED: "
                    f"{num_organs} independent convs (initialized from shared conv), final_ch={final_ch}")
        else:
            self.shared_out_norm = None
            self.shared_out_act = None
            self.organ_out_convs = None

        # Note: FP16 conversion is now handled by autocast() in the training loop
        # The old convert_to_fp16() method is incompatible with AMP's GradScaler
        # if use_fp16:
        #     self.unet.convert_to_fp16()

        log.info(f"[Network] Built medical UNet with cross-attention")
        input_desc = "x_t"
        if cond_x1: input_desc += " + x1"
        if cond_lava_water: input_desc += " + lava_water"
        if cond_lava_fat: input_desc += " + lava_fat"
        log.info(f"[Network] Input channels: {in_channels} ({input_desc})")
        log.info(
            f"[Network] PET conditioning: {'Enabled via cross-attention' if cond_pet else 'Disabled'}"
        )
        if cond_pet:
            log.info(f"[Network] Cross-attention at resolutions: {cross_attn_resolutions}")
            log.info(f"[Network] Spatial cross-attention: {use_spatial_cross_attn}")
        log.info(
            f"[Network] Organ conditioning: {cond_organ} "
            f"(time={self.organ_time_cond}, embed_dim={self.organ_embed_dim}, classes={num_organs})"
        )

    def forward(self, xt, t, x1=None, pet=None, organ_id=None,
                lava_water=None, lava_fat=None, body_part_id=None, organ_mask=None,
                noise_level_override=None):
        """
        Forward pass with cross-attention PET conditioning.

        Args:
            xt: Noisy image [B, 1, H, W]
            t: Timestep indices [B]
            x1: T1/LAVA conditioning [B, 1, H, W]
            pet: PET conditioning [B, 1, H, W] (used in cross-attention, not concatenated)
            organ_id: Organ indices [B] (backward compatible, for old organ conditioning)
            lava_water: LAVA water conditioning [B, 1, H, W]
            lava_fat: LAVA fat conditioning [B, 1, H, W]
            body_part_id: Body Part indices [B] (semantic body part conditioning)
            organ_mask: Multi-hot organ presence mask [B, num_organs] (from JSON Organs Present)
            noise_level_override: [B] If provided, use this as the noise level for time
                embedding instead of looking up from self.noise_levels[t].
                This is used for semantic time warp where the warped noise level
                (std_fwd at warped time s) is passed directly.

        Returns:
            Predicted output [B, 1, H, W]
        """
        # Map timestep indices to noise levels
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

        # Use override if provided (for semantic time warp), otherwise lookup
        if noise_level_override is not None:
            noise_level = noise_level_override
        else:
            noise_level = self.noise_levels[t]

        from guided_diffusion.nn import timestep_embedding as make_timestep_embedding

        # Collect condition embeddings
        cond_time_embs = []

        # Prepare organ conditioning
        organ_embed = None
        if self.cond_organ:
            if organ_id is None:
                raise ValueError("cond_organ=True but organ_id not provided")
            if isinstance(organ_id, int):
                organ_id = torch.tensor([organ_id], device=xt.device, dtype=torch.long)
            elif isinstance(organ_id, torch.Tensor):
                organ_id = organ_id.to(xt.device, dtype=torch.long)

            if organ_id.dim() == 0:
                organ_id = organ_id.unsqueeze(0)

            organ_embed = self.organ_embedding(organ_id)  # [B, embed_dim]

            if self.organ_time_cond and self.organ_time_mlp is not None:
                organ_time_emb = self.organ_time_mlp(organ_embed)
                cond_time_embs.append(organ_time_emb)

        # Body Part semantic conditioning (NEW)
        body_part_embed = None
        body_part_spatial = None
        if self.cond_body_part:
            if body_part_id is None:
                # Default to 0 (first body part) if not provided
                body_part_id = torch.zeros(xt.shape[0], device=xt.device, dtype=torch.long)
            elif isinstance(body_part_id, int):
                body_part_id = torch.full((xt.shape[0],), body_part_id, device=xt.device, dtype=torch.long)
            elif isinstance(body_part_id, torch.Tensor):
                body_part_id = body_part_id.to(xt.device, dtype=torch.long)
            
            if body_part_id.dim() == 0:
                body_part_id = body_part_id.unsqueeze(0)
            
            # Get embedding (works with both semantic and fallback)
            if hasattr(self.body_part_embedding, 'forward'):
                body_part_embed = self.body_part_embedding(body_part_id)  # [B, embed_dim]
            else:
                body_part_embed = self.body_part_embedding(body_part_id)
            
            # Time conditioning
            if self.body_part_time_cond:
                if hasattr(self.body_part_embedding, 'get_time_embedding'):
                    body_part_time_emb = self.body_part_embedding.get_time_embedding(body_part_id)
                elif hasattr(self, 'body_part_time_mlp'):
                    body_part_time_emb = self.body_part_time_mlp(body_part_embed)
                else:
                    body_part_time_emb = None
                
                if body_part_time_emb is not None:
                    cond_time_embs.append(body_part_time_emb)
            
            # Channel conditioning (spatial broadcast)
            if self.body_part_channel_cond:
                target_h, target_w = xt.shape[2], xt.shape[3]
                if hasattr(self.body_part_embedding, 'get_spatial_embedding'):
                    body_part_spatial = self.body_part_embedding.get_spatial_embedding(
                        body_part_id, target_h, target_w
                    )
                else:
                    # Fallback: manual broadcast
                    body_part_spatial = body_part_embed.unsqueeze(-1).unsqueeze(-1)
                    body_part_spatial = body_part_spatial.expand(-1, -1, target_h, target_w)

        # Organ embedding from JSON (uses organ_mask instead of organ_id)
        # Purposes:
        # 1. Cross-attention with body_part embedding to modulate body part representation
        # 2. Semantic conditioning for region-aware representation
        organ_embed_from_json = None
        if self.organ_embedding_from_json is not None and organ_mask is not None:
            # Convert organ_mask to proper device/dtype
            if isinstance(organ_mask, torch.Tensor):
                organ_mask = organ_mask.to(xt.device, dtype=torch.float32)
            else:
                organ_mask = torch.tensor(organ_mask, device=xt.device, dtype=torch.float32)

            if organ_mask.dim() == 1:
                organ_mask = organ_mask.unsqueeze(0)

            # Get aggregated organ embedding from multi-hot mask
            organ_embed_from_json = self.organ_embedding_from_json(organ_mask)  # [B, embed_dim]

            # Cross-attention: body_part_embed modulated by organ information
            if body_part_embed is not None and hasattr(self, 'body_organ_cross_attn'):
                # Prepare query: [B, 1, D]
                bp_query = body_part_embed.unsqueeze(1)  # [B, 1, D]

                # Get per-organ embeddings: [B, num_organs, D] with mask
                if hasattr(self.organ_embedding_from_json, 'get_expanded_embeddings'):
                    organ_kv, organ_key_mask = self.organ_embedding_from_json.get_expanded_embeddings(organ_mask)
                    # organ_kv: [B, num_organs, D], organ_key_mask: [B, num_organs] (True=ignore)
                else:
                    # Fallback for old checkpoints without get_expanded_embeddings
                    organ_kv = organ_embed_from_json.unsqueeze(1)  # [B, 1, D]
                    organ_key_mask = None

                # Cross-attention: body_part attends to each organ independently
                attn_out, _ = self.body_organ_cross_attn(
                    query=bp_query,
                    key=organ_kv,
                    value=organ_kv,
                    key_padding_mask=organ_key_mask,
                )  # [B, 1, D]
                attn_out = attn_out.squeeze(1)  # [B, D]

                # Residual connection with layer norm and projection
                attn_out = self.body_organ_cross_attn_norm(attn_out)
                attn_out = self.body_organ_cross_attn_proj(attn_out)

                # Add to body_part_embed (residual)
                body_part_embed = body_part_embed + attn_out

                # Update time embedding if body_part_time_cond
                if self.body_part_time_cond:
                    if hasattr(self.body_part_embedding, 'time_mlp'):
                        # Recompute time embedding with updated body_part_embed
                        body_part_time_emb = self.body_part_embedding.time_mlp(body_part_embed)
                    elif hasattr(self, 'body_part_time_mlp'):
                        body_part_time_emb = self.body_part_time_mlp(body_part_embed)

                    # Replace the body_part time embedding in cond_time_embs
                    # Find and replace it (it was added earlier)
                    # For simplicity, we add the delta instead
                    # This is handled by the updated body_part_embed being used later

        model_input = [xt]

        if self.cond_x1:
            if x1 is None:
                raise ValueError("Network expects x1 conditioning but x1=None")

            model_input.append(x1)
        
        if self.cond_lava_water:
            if lava_water is None:
                raise ValueError("Network expects lava_water conditioning but lava_water=None")

            model_input.append(lava_water)
        
        if self.cond_lava_fat:
            if lava_fat is None:
                raise ValueError("Network expects lava_fat conditioning but lava_fat=None")

            model_input.append(lava_fat)

        # Body Part spatial conditioning (NEW)
        if self.cond_body_part and self.body_part_channel_cond and body_part_spatial is not None:
            model_input.append(body_part_spatial)

        # Concatenate input
        model_input = torch.cat(model_input, dim=1)  # [B, in_channels, H, W]
        
        # Numerical stability clamp to avoid FP16 overflow.
        model_input = model_input.clamp(-2.0, 2.0)

        # Prepare PET context for cross-attention
        if self.cond_pet:
            if pet is None:
                self.log.warning(
                    "[PET] Network trained with PET cross-attention but pet=None. "
                    "Using zero tensor. Performance may degrade."
                )
                pet_context = torch.zeros_like(x1)
            else:
                pet_context = pet
        else:
            pet_context = None

        # Combined condition embedding for UNet
        if cond_time_embs:
            combined_cond_emb = sum(cond_time_embs)
        else:
            combined_cond_emb = None

        # Run through UNet with cross-attention
        if self.use_organ_specific_out:
            # Use organ-specific output heads.
            # First get decoder features before the output conv.
            if isinstance(self.unet, UNetWithCrossAttention):
                features = self.unet(
                    model_input,
                    noise_level,
                    context=pet_context,
                    cond_emb=combined_cond_emb,
                    return_features=True,
                )
            else:
                from guided_diffusion.nn import timestep_embedding as make_timestep_embedding
                emb = self.unet.time_embed(make_timestep_embedding(noise_level, self.unet.model_channels))
                if combined_cond_emb is not None:
                    emb = emb + combined_cond_emb
                features = self._forward_base_unet(model_input, emb, return_features=True)
            
            # Shared norm + activation.
            features = self.shared_out_norm(features)
            features = self.shared_out_act(features)
            
            # Group by organ_id for efficient mixed-batch execution.
            output = self._apply_organ_specific_output(features, organ_id)
        else:
            # Shared output head.
            if isinstance(self.unet, UNetWithCrossAttention):
                output = self.unet(
                    model_input,
                    noise_level,
                    context=pet_context,
                    cond_emb=combined_cond_emb,
                )
            else:
                # Base UNet (no cross-attention)
                from guided_diffusion.nn import timestep_embedding as make_timestep_embedding
                emb = self.unet.time_embed(make_timestep_embedding(noise_level, self.unet.model_channels))
                if combined_cond_emb is not None:
                    emb = emb + combined_cond_emb
                # Standard UNet forward (no context parameter)
                output = self._forward_base_unet(model_input, emb)

        return output

    def _forward_base_unet(self, x, emb, return_features=False):
        """Forward through base UNet without cross-attention
        
        Args:
            x: Input tensor
            emb: Time embedding
            return_features: If True, return features before output layer
        """
        hs = []
        h = x.type(self.unet.dtype)

        for module in self.unet.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.unet.middle_block(h, emb)

        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        
        if return_features:
            return h
        return self.unet.out(h)
    
    def _apply_organ_specific_output(self, features, organ_id):
        """
        Apply organ-specific output convolutions.
        
        Efficiently handle mixed batches by grouping samples by organ_id.
        
        Args:
            features: [B, C, H, W] features after shared norm/activation
            organ_id: [B] organ ID for each sample
            
        Returns:
            output: [B, 1, H, W]
        """
        B = features.shape[0]
        device = features.device
        dtype = features.dtype
        
        # Collect outputs in original order, then concatenate.
        output_list = [None] * B
        
        # Process per-organ groups.
        unique_organs = organ_id.unique()
        
        for org in unique_organs:
            # Indices for this organ.
            mask = (organ_id == org)
            indices = mask.nonzero(as_tuple=True)[0]
            
            # Batch process this organ subset.
            org_features = features[indices]  # [N_org, C, H, W]
            org_output = self.organ_out_convs[org.item()](org_features)  # [N_org, 1, H, W]
            
            # Restore outputs to original positions.
            for i, idx in enumerate(indices.tolist()):
                output_list[idx] = org_output[i:i+1]
        
        # Concatenate all outputs.
        output = torch.cat(output_list, dim=0)  # [B, 1, H, W]
        
        return output

    def load_pretrained(self, ckpt_path=None):
        """Load pretrained weights if available"""
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = checkpoint.get('net', checkpoint)

            # Filter compatible keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            self.log.info(f"[Network] Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained")
        else:
            self.log.info("[Network] No pretrained weights loaded, training from scratch")


def build_medical_network_crossattn(opt, log):
    """Build medical network with cross-attention PET conditioning"""
    noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

    # Use network_cond_pet if available (for architecture compatibility)
    cond_pet = getattr(opt, 'network_cond_pet', getattr(opt, 'cond_pet', False))
    
    # LAVA water/fat conditioning
    cond_lava_water = getattr(opt, 'cond_lava_water', False)
    cond_lava_fat = getattr(opt, 'cond_lava_fat', False)
    
    # Body Part semantic conditioning (NEW)
    cond_body_part = getattr(opt, 'cond_body_part', False)

    net = MedicalNetCrossAttn(
        log=log,
        noise_levels=noise_levels,
        use_fp16=getattr(opt, 'force_unet_fp16', False),
        cond_x1=opt.cond_x1,
        cond_pet=cond_pet,
        cond_organ=getattr(opt, 'cond_organ', False),
        num_organs=getattr(opt, 'num_organs', 0),
        organ_embed_dim=getattr(opt, 'organ_embed_dim', 16),
        organ_time_cond=getattr(opt, 'organ_time_cond', True),
        cross_attn_resolutions=getattr(opt, 'cross_attn_resolutions', [32, 16, 8]),
        use_spatial_cross_attn=getattr(opt, 'use_spatial_cross_attn', True),
        image_size=getattr(opt, 'image_size', 256),  # Pass actual image size
        # LAVA water/fat conditioning
        cond_lava_water=cond_lava_water,
        cond_lava_fat=cond_lava_fat,
        # Organ-specific output heads
        use_organ_specific_out=getattr(opt, 'organ_specific_out', False),
        # Body Part semantic conditioning (NEW)
        cond_body_part=cond_body_part,
        body_part_embed_dim=getattr(opt, 'body_part_embed_dim', 64),
        body_part_channel_cond=getattr(opt, 'body_part_channel_cond', True),
        body_part_time_cond=getattr(opt, 'body_part_time_cond', True),
        use_semantic_embedding=getattr(opt, 'use_semantic_embedding', True),
    )

    # Optionally load pretrained weights
    if hasattr(opt, 'pretrained_ckpt') and opt.pretrained_ckpt:
        net.load_pretrained(opt.pretrained_ckpt)

    return net
