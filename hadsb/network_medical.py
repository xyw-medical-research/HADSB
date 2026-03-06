"""
Modified network for medical modality conversion with PET conditioning
"""

import torch
import torch.nn as nn
import numpy as np
from guided_diffusion.unet import UNetModel
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .network import Image256Net


class MedicalNet(Image256Net):
    """
    Modified network for medical modality conversion
    Handles grayscale medical images and PET conditioning
    """
    
    def __init__(
        self,
        log,
        noise_levels,
        use_fp16=False,
        cond_x1=True,
        cond_pet=False,
        cond_organ=False,
        num_organs=0,
        organ_embed_dim=16,
        organ_channel_cond=True,
        organ_time_cond=True,
    ):
        """
        Args:
            log: Logger
            noise_levels: Noise levels for timesteps
            use_fp16: Whether to use fp16
            cond_x1: Whether to condition on x1 (T1 image)
            cond_pet: Whether to condition on PET image
            cond_organ: Whether to condition on organ embeddings
            num_organs: Number of organ classes (including unknown)
            organ_embed_dim: Embedding dimension for organ conditioning
        """
        # Don't call parent init, we'll set up our own UNet
        nn.Module.__init__(self)
        
        self.log = log
        self.noise_levels = noise_levels
        self.use_fp16 = use_fp16
        self.cond_x1 = cond_x1
        self.cond_pet = cond_pet
        self.cond_organ = cond_organ
        self.organ_channel_cond = bool(organ_channel_cond) if cond_organ else False
        self.organ_time_cond = bool(organ_time_cond) if cond_organ else False
        self.organ_embed_dim = organ_embed_dim if cond_organ else 0
        self.num_organs = num_organs
        
        # Calculate input channels
        # Base: 1 channel for grayscale image
        # +1 if conditioning on x1 (T1)
        # +1 if conditioning on PET
        in_channels = 1
        if cond_x1:
            in_channels += 1
        if cond_pet:
            in_channels += 1
        if cond_organ:
            if num_organs <= 0:
                raise ValueError("cond_organ=True requires num_organs > 0")
            if organ_embed_dim <= 0:
                raise ValueError("cond_organ=True requires organ_embed_dim > 0")
            if self.organ_channel_cond:
                in_channels += organ_embed_dim
        
        out_channels = 1  # Grayscale output
        
        # Create UNet with proper medical configuration
        self.unet = UNetModel(
            image_size=512,  # Medical images are 512x512
            in_channels=in_channels,
            model_channels=128,  # RESTORED: Back to 128 for sufficient capacity
            out_channels=out_channels,
            num_res_blocks=3,    # RESTORED: Back to 3 for better representation
            attention_resolutions=[32, 16, 8],  # RESTORED: Full attention hierarchy
            dropout=0.0,         # RESTORED: No dropout for medical precision
            channel_mult=(1, 1, 2, 2, 4, 4),   # RESTORED: Original progression
            num_classes=None,
            use_checkpoint=False,
            use_fp16=use_fp16,
            num_heads=4,         # RESTORED: Back to 4 heads
            num_head_channels=64, # RESTORED: Back to 64
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
        )
        
        # Get time embedding dimension from UNet
        time_embed_dim = self.unet.time_embed[-1].out_features
        
        # Organ conditioning layers
        if cond_organ:
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
        
        # Semantic Time Warp (for per-sample adaptive schedules)
        # This is stored but NOT applied here - it's applied in the runner
        # which has access to the diffusion object for consistent schedule warping
        self.time_warp = None  # Will be set by runner if enabled

        # Note: FP16 conversion is now handled by autocast() in the training loop
        # The old convert_to_fp16() method is incompatible with AMP's GradScaler
        # if use_fp16:
        #     self.unet.convert_to_fp16()
        
        log.info(f"[Network] Built medical UNet with {in_channels} input channels, {out_channels} output channels")
        log.info(
            f"[Network] Conditioning: x1={cond_x1}, PET={cond_pet}, "
            f"organ={cond_organ} (channel={self.organ_channel_cond}, time={self.organ_time_cond}, "
            f"embed_dim={self.organ_embed_dim}, classes={num_organs})"
        )
    
    def forward(self, xt, t, x1=None, pet=None, organ_id=None,
                noise_level_override=None,
                lava_water=None, lava_fat=None, body_part_id=None, organ_mask=None):
        """
        Args:
            xt: Noisy image at timestep t [B, 1, H, W]
            t: Timestep indices [B] (used if noise_level_override is None)
            x1: T1 conditioning image [B, 1, H, W]
            pet: PET conditioning image [B, 1, H, W]
            organ_id: Organ class indices [B]
            noise_level_override: [B] If provided, use this as the noise level for time
                embedding instead of looking up from self.noise_levels[t].
                This is used for semantic time warp where the warped noise level
                (std_fwd at warped time s) is passed directly.
            lava_water: Ignored (for interface compatibility with crossattn variant)
            lava_fat: Ignored (for interface compatibility with crossattn variant)
            body_part_id: Ignored (for interface compatibility with crossattn variant)
            organ_mask: Ignored (for interface compatibility with crossattn variant)

        Returns:
            Predicted noise/direction [B, 1, H, W]
        """
        # Map timestep indices to noise levels
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

        # Use override if provided (for semantic time warp), otherwise lookup
        if noise_level_override is not None:
            noise_level = noise_level_override
        else:
            noise_level = self.noise_levels[t]
        
        # Build input by concatenating conditioning
        model_input = [xt]

        if self.cond_x1:
            if x1 is None:
                raise ValueError("Network expects x1 conditioning but x1=None")
            model_input.append(x1)

        if self.cond_pet:
            # IMPORTANT: If network was built with PET, always include PET channel
            # Even if it's zeros (for inference without PET on PET-trained model)
            if pet is None:
                raise ValueError(
                    "Network expects PET conditioning but pet=None. "
                    "Use torch.zeros_like(x1) if you want to sample without PET."
                )
            model_input.append(pet)
        
        # Collect time embeddings from various conditions
        cond_time_embs = []
        
        # Organ conditioning
        if self.cond_organ:
            if organ_id is None:
                raise ValueError("cond_organ=True but organ_id not provided")
            if isinstance(organ_id, int):
                organ_id = torch.tensor([organ_id], device=xt.device, dtype=torch.long)
            elif isinstance(organ_id, torch.Tensor):
                organ_id = organ_id.to(xt.device, dtype=torch.long)
            else:
                raise TypeError(f"Unsupported organ_id type: {type(organ_id)}")

            if organ_id.dim() == 0:
                organ_id = organ_id.unsqueeze(0)

            organ_embed = self.organ_embedding(organ_id)  # [B, embed_dim]

            if self.organ_time_cond and self.organ_time_mlp is not None:
                organ_time_emb = self.organ_time_mlp(organ_embed)
                cond_time_embs.append(organ_time_emb)

            if self.organ_channel_cond:
                organ_spatial = organ_embed.unsqueeze(-1).unsqueeze(-1)
                target_h, target_w = xt.shape[2], xt.shape[3]
                organ_cond = organ_spatial.expand(-1, -1, target_h, target_w)
                model_input.append(organ_cond)
        
        # Concatenate along channel dimension
        model_input = torch.cat(model_input, dim=1)
        
        # Combine all time condition embeddings
        if cond_time_embs:
            combined_cond_emb = sum(cond_time_embs)
        else:
            combined_cond_emb = None
        
        # Run through UNet
        output = self.unet(model_input, noise_level, cond_emb=combined_cond_emb)
        
        return output
    
    def load_pretrained(self, ckpt_path=None):
        """Load pretrained weights if available"""
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # Filter out incompatible keys due to channel mismatch
            model_dict = self.state_dict()
            pretrained_dict = checkpoint.get('net', checkpoint)
            
            # Filter out input/output conv layers with different channels
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            self.log.info(f"[Network] Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained")
        else:
            self.log.info("[Network] No pretrained weights loaded, training from scratch")


def build_medical_network(opt, log):
    """Build medical network with optional PET conditioning"""
    noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

    # IMPORTANT: Use network_cond_pet if available (for architecture)
    # This ensures network architecture matches training, even if inference disables PET
    cond_pet = getattr(opt, 'network_cond_pet', getattr(opt, 'cond_pet', False))
    
    net = MedicalNet(
        log=log,
        noise_levels=noise_levels,
        use_fp16=getattr(opt, 'force_unet_fp16', False),
        cond_x1=opt.cond_x1,
        cond_pet=cond_pet,  # Use network_cond_pet for architecture
        cond_organ=getattr(opt, 'cond_organ', False),
        num_organs=getattr(opt, 'num_organs', 0),
        organ_embed_dim=getattr(opt, 'organ_embed_dim', 16),
        organ_channel_cond=getattr(opt, 'organ_channel_cond', True),
        organ_time_cond=getattr(opt, 'organ_time_cond', True),
    )
    
    # Optionally load pretrained weights
    if hasattr(opt, 'pretrained_ckpt') and opt.pretrained_ckpt:
        net.load_pretrained(opt.pretrained_ckpt)
    
    return net
