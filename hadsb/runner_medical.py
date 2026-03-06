"""
Modified runner for medical modality conversion with visualization
Supports both single-GPU and multi-GPU (DDP) training.
"""

import os
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torch_ema import ExponentialMovingAverage
import numpy as np
import pickle

from .util import unsqueeze_xdim
from .network_medical import build_medical_network
from .network_medical_crossattn import build_medical_network_crossattn
from .diffusion import Diffusion
from visualization import MedicalTrainingVisualizer


# ===================== Distributed Utilities =====================
def is_distributed():
    """Check if running in distributed mode"""
    return torch.distributed.is_initialized()


def get_world_size():
    """Get world size (number of processes)"""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def get_rank():
    """Get global rank"""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_local_rank():
    """Get local rank (GPU id on this node)"""
    if is_distributed():
        return int(os.environ.get('LOCAL_RANK', 0))
    return 0


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_rank() == 0


def barrier():
    """Synchronize all processes"""
    if is_distributed():
        torch.distributed.barrier()


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()


class MedicalRunner(object):
    """Runner adapted for medical modality conversion with PET conditioning.
    
    Supports both single-GPU and multi-GPU (DDP) training modes.
    """

    def __init__(self, opt, log, save_opt=True, wrap_ddp=False):
        """Initialize medical runner.
        
        Args:
            opt: Training options
            log: Logger
            save_opt: Whether to save options to disk (should be True only on main process)
            wrap_ddp: Whether to wrap the network with DDP
        """
        # Store distributed info
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.is_main = is_main_process()
        
        # Save options (only on main process)
        if save_opt and self.is_main:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info(f"Saved options to {opt_pkl_path}")

        # Build diffusion
        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built medical HADSB diffusion: steps={len(betas)}")

        # Build medical network with conditioning
        # Use MedicalNetCrossAttn when any advanced features are enabled:
        # - Cross-attention (with PET or other context)
        # - LAVA water/fat conditioning
        # - Body part semantic conditioning
        use_crossattn_network = (
            getattr(opt, 'use_cross_attn', False) or
            getattr(opt, 'cond_lava_water', False) or
            getattr(opt, 'cond_lava_fat', False) or
            getattr(opt, 'cond_body_part', False)
        )
        
        if use_crossattn_network:
            features = []
            if getattr(opt, 'use_cross_attn', False) and getattr(opt, 'cond_pet', False):
                features.append("PET cross-attention")
            if getattr(opt, 'cond_lava_water', False):
                features.append("LAVA water")
            if getattr(opt, 'cond_lava_fat', False):
                features.append("LAVA fat")
            if getattr(opt, 'cond_body_part', False):
                features.append("Body part")
            log.info(f"[Network] Using MedicalNetCrossAttn with features: {', '.join(features)}")
            # Parse cross-attention resolutions
            if hasattr(opt, 'cross_attn_resolutions') and isinstance(opt.cross_attn_resolutions, str):
                opt.cross_attn_resolutions = [int(x.strip()) for x in opt.cross_attn_resolutions.split(',')]
            self.net = build_medical_network_crossattn(opt, log)
        else:
            if getattr(opt, 'cond_pet', False):
                log.info("[Network] Using Channel Concatenation for PET conditioning")
            self.net = build_medical_network(opt, log)

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        # Load checkpoint if provided
        if opt.load:
            self.load_checkpoint(opt.load, log)

        # Move to device
        self.net.to(opt.device)
        self.ema.to(opt.device)
        
        # Wrap with DDP if requested
        self._net_module = self.net  # Keep reference to unwrapped module
        if wrap_ddp and self.world_size > 1:
            # Cases that usually require find_unused_parameters=True:
            # - organ_specific_out: only a subset of organ heads is used per batch
            # - use_semantic_embedding: some aggregation weights may be inactive per step
            # - use_cross_attn: cross-attention parameters can be conditionally used
            # - cond_organ_crossattn: organ cross-attention can be conditionally used
            find_unused = (
                getattr(opt, 'organ_specific_out', False) or
                getattr(opt, 'use_semantic_embedding', False) or
                getattr(opt, 'use_cross_attn', False) or
                getattr(opt, 'cond_organ_crossattn', False)
            )
            self.net = DDP(
                self.net,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=find_unused
            )
            if find_unused:
                log.info(f"Model wrapped with DDP on GPU {self.local_rank} (find_unused_parameters=True)")
            else:
                log.info(f"Model wrapped with DDP on GPU {self.local_rank}")

        self.log = log
        self.opt = opt
        
        # Initialize Semantic Time Warp if enabled
        # Check both use_time_warp (CLI) and use_semantic_time_warp (legacy)
        self.time_warp = None
        self.time_warp_ema = None
        use_time_warp = getattr(opt, 'use_time_warp', False) or getattr(opt, 'use_semantic_time_warp', False)
        if use_time_warp:
            from .semantic_time_warp import SemanticTimeWarp
            num_regions = getattr(opt, 'num_body_parts', 11)
            num_organs = getattr(opt, 'num_organs', 36)
            self.time_warp = SemanticTimeWarp(
                num_regions=num_regions,
                num_organs=num_organs,
                embed_dim=getattr(opt, 'time_warp_embed_dim', 64),
                hidden_dim=getattr(opt, 'time_warp_hidden_dim', 128),
                a_scale=getattr(opt, 'time_warp_a_scale', 0.5),
            )
            self.time_warp.to(opt.device)
            # Create separate EMA for time_warp
            self.time_warp_ema = ExponentialMovingAverage(self.time_warp.parameters(), decay=opt.ema)
            self.time_warp_ema.to(opt.device)
            # Store reference in network for checkpoint saving
            self._net_module.time_warp = self.time_warp
            log.info(f"[TimeWarp] Semantic Time Warp ENABLED: regions={num_regions}, organs={num_organs}")
            # Load pending checkpoint if available
            self._load_pending_time_warp_checkpoint()

        # Initialize PET Noise Modulator if enabled
        # This learns to spatially modulate the noise injection based on PET + semantic features
        self.pet_noise_modulator = None
        self.pet_noise_modulator_ema = None
        use_pet_noise_modulation = getattr(opt, 'use_pet_noise_modulation', False)
        if use_pet_noise_modulation:
            from .diffusion import PETNoiseModulator
            self.pet_noise_modulator = PETNoiseModulator(
                in_ch=1,
                hidden_ch=getattr(opt, 'pet_noise_hidden_ch', 32),
                min_scale=getattr(opt, 'pet_noise_min_scale', 0.3),
                max_scale=getattr(opt, 'pet_noise_max_scale', 1.5),
                # Semantic conditioning parameters.
                num_body_parts=getattr(opt, 'num_body_parts', 11),
                num_organs=getattr(opt, 'num_organs', 36),
                semantic_embed_dim=getattr(opt, 'pet_noise_semantic_embed_dim', 64),
            )
            self.pet_noise_modulator.to(opt.device)
            # Create separate EMA for pet_noise_modulator
            self.pet_noise_modulator_ema = ExponentialMovingAverage(
                self.pet_noise_modulator.parameters(), decay=opt.ema
            )
            self.pet_noise_modulator_ema.to(opt.device)
            log.info(
                f"[PETNoiseModulator] ENABLED: "
                f"min_scale={getattr(opt, 'pet_noise_min_scale', 0.3)}, "
                f"max_scale={getattr(opt, 'pet_noise_max_scale', 1.5)}, "
                f"semantic_embed_dim={getattr(opt, 'pet_noise_semantic_embed_dim', 64)}"
            )
            # Load pending checkpoint if available
            self._load_pending_pet_noise_modulator_checkpoint()

        # Initialize mixed precision training
        self.use_amp = getattr(opt, 'use_fp16', False)
        self.use_bf16 = getattr(opt, 'use_bf16', False)  # BF16 support.
        self.autocast_device = 'cuda' if opt.device.type == 'cuda' else 'cpu'

        # CPU fallback: disable CUDA AMP options.
        if opt.device.type != 'cuda':
            self.use_amp = False
            self.use_bf16 = False
        
        if self.use_bf16:
            # BF16 does not need GradScaler (similar dynamic range to FP32).
            self.scaler = None
            self.amp_dtype = torch.bfloat16
            log.info("[AMP] BFloat16 training enabled (no GradScaler needed)")
        elif self.use_amp:
            self.scaler = GradScaler('cuda')
            self.amp_dtype = torch.float16
            log.info("[AMP] Mixed precision (FP16) training enabled with GradScaler")
        else:
            self.scaler = None
            self.amp_dtype = torch.float32
        
        # Initialize visualizer (only on main process to avoid conflicts)
        if self.is_main and not getattr(opt, 'disable_viz', False):
            viz_dir = opt.ckpt_path / 'visualizations'
            self.visualizer = MedicalTrainingVisualizer(
                save_dir=viz_dir,
                log_freq=100,
                viz_freq=getattr(opt, 'viz_freq', 1000)
            )
        else:
            self.visualizer = None

    
    @property
    def net_module(self):
        """Get the unwrapped network module (without DDP wrapper)"""
        if isinstance(self.net, DDP):
            return self.net.module
        return self.net

    def load_checkpoint(self, ckpt_path, log):
        """Load checkpoint"""
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        except pickle.UnpicklingError as err:
            log.warning(
                "[Checkpoint] Failed default torch.load (%s). Retrying with weights_only=False.",
                err
            )
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Load network
        if 'net' in checkpoint:
            self.net.load_state_dict(checkpoint['net'], strict=False)
            log.info(f"[Net] Loaded network from {ckpt_path}")

        # Load EMA
        if 'ema' in checkpoint:
            try:
                self.ema.load_state_dict(checkpoint['ema'])
                log.info(f"[EMA] Loaded EMA from {ckpt_path}")
            except (RuntimeError, ValueError) as err:
                log.warning(
                    "[EMA] Skipping EMA load from %s due to mismatch: %s. "
                    "EMA weights will be re-initialized.",
                    ckpt_path, err
                )
                decay = getattr(self.ema, 'decay', 0.999)
                self.ema = ExponentialMovingAverage(self.net.parameters(), decay=decay)
                device = next(self.net.parameters()).device
                self.ema.to(device)

        # Load time_warp if present (will be initialized later if needed)
        self._pending_time_warp_ckpt = None
        if 'time_warp' in checkpoint:
            self._pending_time_warp_ckpt = {
                'time_warp': checkpoint['time_warp'],
                'time_warp_ema': checkpoint.get('time_warp_ema'),
            }
            log.info(f"[TimeWarp] Found time_warp state in checkpoint (will load after init)")

        # Load pet_noise_modulator if present (will be initialized later if needed)
        self._pending_pet_noise_modulator_ckpt = None
        if 'pet_noise_modulator' in checkpoint:
            self._pending_pet_noise_modulator_ckpt = {
                'pet_noise_modulator': checkpoint['pet_noise_modulator'],
                'pet_noise_modulator_ema': checkpoint.get('pet_noise_modulator_ema'),
            }
            log.info(f"[PETNoiseModulator] Found pet_noise_modulator state in checkpoint (will load after init)")

    def _load_pending_time_warp_checkpoint(self):
        """Load time_warp checkpoint if one was found during load_checkpoint."""
        if hasattr(self, '_pending_time_warp_ckpt') and self._pending_time_warp_ckpt is not None:
            if self.time_warp is not None:
                try:
                    self.time_warp.load_state_dict(self._pending_time_warp_ckpt['time_warp'])
                    self.log.info("[TimeWarp] Loaded time_warp weights from checkpoint")
                    if self._pending_time_warp_ckpt.get('time_warp_ema') and self.time_warp_ema is not None:
                        self.time_warp_ema.load_state_dict(self._pending_time_warp_ckpt['time_warp_ema'])
                        self.log.info("[TimeWarp] Loaded time_warp EMA from checkpoint")
                except Exception as e:
                    self.log.warning(f"[TimeWarp] Failed to load time_warp state: {e}")
            self._pending_time_warp_ckpt = None

    def _load_pending_pet_noise_modulator_checkpoint(self):
        """Load pet_noise_modulator checkpoint if one was found during load_checkpoint."""
        if hasattr(self, '_pending_pet_noise_modulator_ckpt') and self._pending_pet_noise_modulator_ckpt is not None:
            if self.pet_noise_modulator is not None:
                try:
                    self.pet_noise_modulator.load_state_dict(
                        self._pending_pet_noise_modulator_ckpt['pet_noise_modulator']
                    )
                    self.log.info("[PETNoiseModulator] Loaded pet_noise_modulator weights from checkpoint")
                    if (self._pending_pet_noise_modulator_ckpt.get('pet_noise_modulator_ema') 
                        and self.pet_noise_modulator_ema is not None):
                        self.pet_noise_modulator_ema.load_state_dict(
                            self._pending_pet_noise_modulator_ckpt['pet_noise_modulator_ema']
                        )
                        self.log.info("[PETNoiseModulator] Loaded pet_noise_modulator EMA from checkpoint")
                except Exception as e:
                    self.log.warning(f"[PETNoiseModulator] Failed to load pet_noise_modulator state: {e}")
            self._pending_pet_noise_modulator_ckpt = None

    def compute_loss(self, x0, x1, pet, mask=None,
                     lava_water=None, lava_fat=None, body_part=None, organ_mask=None):
        """
        Compute HADSB loss for medical modality conversion

        Args:
            x0: Target T2 images [B, 1, H, W]
            x1: Source T1 images [B, 1, H, W]
            pet: PET conditioning [B, 1, H, W]
            mask: Optional mask [B, 1, H, W]
            lava_water: LAVA water conditioning [B, 1, H, W]
            lava_fat: LAVA fat conditioning [B, 1, H, W]
            body_part: Body Part indices [B]
            organ_mask: Multi-hot organ mask [B, num_organs] (from JSON)
        """
        batch = x0.shape[0]
        device = x0.device

        # Numerical stability: sanitize NaN/Inf in inputs.
        if torch.isnan(x0).any() or torch.isinf(x0).any():
            x0 = torch.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(x1).any() or torch.isinf(x1).any():
            x1 = torch.nan_to_num(x1, nan=0.0, posinf=1.0, neginf=-1.0)

        # Sample random timesteps
        num_timesteps = len(self.diffusion.betas)
        step = torch.randint(0, num_timesteps, (batch,), device=device)

        # =====================================================================
        # Semantic Time Warp (if enabled)
        # Warp the time coordinate based on body part and organ context.
        # This creates a per-sample adaptive schedule.
        # =====================================================================
        use_time_warp = (
            self.time_warp is not None and
            body_part is not None and
            (getattr(self.opt, 'use_time_warp', False) or
             getattr(self.opt, 'use_semantic_time_warp', False))
        )

        # =====================================================================
        # PET Noise Modulation (if enabled)
        # Learn to spatially modulate noise based on PET + semantic features
        # =====================================================================
        noise_scale = None
        if self.pet_noise_modulator is not None and pet is not None:
            # PET noise modulator: PET spatial features + semantic context.
            noise_scale = self.pet_noise_modulator(
                pet, 
                body_part=body_part,      # [B] body part index
                organ_mask=organ_mask,    # [B, num_organs] multi-hot
            )  # [B, 1, H, W]

        if use_time_warp:
            # Normalize step to [0, 1]
            t_normalized = step.float() / num_timesteps

            # Apply semantic time warp: s = g(t; region, organs)
            s_normalized, warp_info = self.time_warp(t_normalized, body_part, organ_mask)

            # Convert back to continuous index in [0, T-1]
            s_continuous = s_normalized * num_timesteps

            # Use WARPED time for forward noising and target computation
            xt = self.diffusion.q_sample_warped(
                s_continuous, x0, x1, 
                ot_ode=getattr(self.opt, 'ot_ode', False),
                noise_scale=noise_scale  # PET-guided noise modulation
            )
            std_fwd = self.diffusion.get_std_fwd_warped(s_continuous, xdim=x0.shape[1:])

            # Get warped noise level for network (std_fwd at warped time)
            noise_level_override = self.diffusion.get_noise_level(s_continuous)
        else:
            # Standard (non-warped) forward pass
            xt = self.diffusion.q_sample(
                step, x0, x1, 
                ot_ode=getattr(self.opt, 'ot_ode', False),
                noise_scale=noise_scale  # PET-guided noise modulation
            )
            std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
            noise_level_override = None
            warp_info = None

        # Compute target (score/direction)
        # Numerical stability: avoid division overflow when std_fwd is too small.
        # HADSB std_fwd minimum is around 0.05; 0.01 is a safe lower bound.
        std_fwd_safe = std_fwd.clamp(min=0.01)
        label = (xt - x0) / std_fwd_safe

        # Numerical stability: clamp labels to an FP16-safe range.
        label = label.clamp(-50.0, 50.0)

        # Network prediction with optional conditioning
        # Note: During training, we always use actual data if available
        # The zero-padding logic is only for sampling/inference
        pet_cond = pet if getattr(self.opt, 'cond_pet', False) else None
        water_cond = lava_water if getattr(self.opt, 'cond_lava_water', False) else None
        fat_cond = lava_fat if getattr(self.opt, 'cond_lava_fat', False) else None
        body_part_cond = body_part if getattr(self.opt, 'cond_body_part', False) else None
        organ_mask_cond = organ_mask if getattr(self.opt, 'cond_body_part', False) else None

        # Numerical stability: sanitize conditioning inputs.
        if pet_cond is not None and (torch.isnan(pet_cond).any() or torch.isinf(pet_cond).any()):
            pet_cond = torch.nan_to_num(pet_cond, nan=0.0, posinf=1.0, neginf=-1.0)
        if water_cond is not None and (torch.isnan(water_cond).any() or torch.isinf(water_cond).any()):
            water_cond = torch.nan_to_num(water_cond, nan=0.0, posinf=1.0, neginf=-1.0)
        if fat_cond is not None and (torch.isnan(fat_cond).any() or torch.isinf(fat_cond).any()):
            fat_cond = torch.nan_to_num(fat_cond, nan=0.0, posinf=1.0, neginf=-1.0)

        # Use autocast for mixed precision forward, but keep loss in fp32 for stability.
        amp_enabled = self.use_amp or self.use_bf16
        with autocast(self.autocast_device, enabled=amp_enabled, dtype=self.amp_dtype):
            pred = self.net(xt, step, x1=x1, pet=pet_cond,
                           lava_water=water_cond, lava_fat=fat_cond,
                           body_part_id=body_part_cond, organ_mask=organ_mask_cond,
                           noise_level_override=noise_level_override)

        pred_fp32 = pred.float()
        label_fp32 = label.float()

        # Compute loss in fp32 to avoid fp16 overflow on large targets.
        if mask is not None:
            # Masked loss for inpainting-style tasks
            loss = F.mse_loss(pred_fp32 * mask, label_fp32 * mask, reduction='none')
            loss = loss.sum() / mask.sum()
        else:
            loss = F.mse_loss(pred_fp32, label_fp32)

        # Build metrics dict
        metrics = {
            'loss': loss.item(),
            'pred_mean': pred_fp32.mean().item(),
            'pred_std': pred_fp32.std().item(),
        }

        # Add time warp regularization if enabled
        if use_time_warp and warp_info is not None:
            from .semantic_time_warp import compute_warp_regularization
            warp_reg = compute_warp_regularization(
                warp_info['a'], warp_info['b'],
                lambda_a=getattr(self.opt, 'time_warp_lambda_a', 0.1),
                lambda_b=getattr(self.opt, 'time_warp_lambda_b', 0.1),
            )
            loss = loss + warp_reg

            # Log warp statistics
            metrics['warp_reg'] = warp_reg.item()
            metrics['warp_a_mean'] = warp_info['a'].mean().item()
            metrics['warp_a_std'] = warp_info['a'].std().item()
            metrics['warp_b_mean'] = warp_info['b'].mean().item()
            metrics['warp_b_std'] = warp_info['b'].std().item()

        return loss, metrics

    def sample_batch_from_dataloader(self, loader):
        """Sample a batch from medical dataloader"""
        batch = next(loader)

        # Extract modalities
        x0 = batch['t2'].to(self.opt.device)  # Target T2
        x1 = batch['t1'].to(self.opt.device)  # Source T1
        pet = batch['pet'].to(self.opt.device)  # PET conditioning

        # Extract LAVA water/fat if available
        lava_water = batch.get('lava_water')
        if lava_water is not None:
            lava_water = lava_water.to(self.opt.device)
        
        lava_fat = batch.get('lava_fat')
        if lava_fat is not None:
            lava_fat = lava_fat.to(self.opt.device)
        
        # Extract Body Part ID if available
        body_part = batch.get('body_part_id')
        if body_part is not None:
            body_part = body_part.to(self.opt.device)

        # Extract organ mask if available (from JSON, for organ embedding)
        organ_mask = batch.get('organ_mask')
        if organ_mask is not None:
            organ_mask = organ_mask.to(self.opt.device)

        return x0, x1, pet, None, lava_water, lava_fat, body_part, organ_mask

    def train(self, opt, train_dataset, val_dataset, test_nfe_callback=None):
        """Training loop for medical modality conversion.
        
        Supports both single-GPU and multi-GPU (DDP) training.
        
        Args:
            opt: Training options
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_nfe_callback: Optional callback for NFE testing (takes runner, val_dataset, device)
        """
        # ===================== Setup Data Loaders =====================
        # Adjust batch size for distributed training
        batch_size_per_gpu = opt.batch_size // self.world_size
        if opt.batch_size % self.world_size != 0 and self.is_main:
            self.log.warning(f"Batch size {opt.batch_size} not divisible by world size {self.world_size}")
            self.log.warning(f"Using batch size {batch_size_per_gpu} per GPU")
        
        # Create distributed sampler if needed
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=getattr(opt, 'seed', 42)
            )
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        
        # Effective batch size per loader = batch_size_per_gpu // microbatch
        loader_batch_size = batch_size_per_gpu // opt.microbatch
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=loader_batch_size,
            sampler=train_sampler,
            shuffle=shuffle,
            num_workers=getattr(opt, 'num_workers', 4),
            pin_memory=(opt.device.type == "cuda"),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(opt, 'num_workers', 4),
            pin_memory=(opt.device.type == "cuda")
        )
        
        # ===================== Setup Optimizer =====================
        # Get parameters (from unwrapped module if using DDP)
        # Use layerwise learning rate: higher LR for newly initialized modules

        new_module_lr_mult = getattr(opt, 'new_module_lr_mult', 1.0)

        # Define which parameter names belong to new modules (Stage 2/3 additions)
        # Note: 'time_warp' is included here because it's attached to net_module
        # as a submodule (self._net_module.time_warp = self.time_warp)
        new_module_keywords = [
            'body_part_embedding', 'body_part_time_mlp',
            'organ_embedding', 'organ_time_mlp', 'organ_embedding_from_json',
            'time_warp',  # Time warp is a new module attached to net_module
        ]

        # Separate parameters into base (pretrained) and new (randomly initialized)
        base_params = []
        new_params = []
        time_warp_param_count = 0

        for name, param in self.net_module.named_parameters():
            if not param.requires_grad:
                continue
            is_new_module = any(keyword in name for keyword in new_module_keywords)
            if is_new_module:
                new_params.append(param)
                # Count time_warp params for logging
                if 'time_warp' in name:
                    time_warp_param_count += param.numel()
            else:
                base_params.append(param)

        # Log time_warp params if present (they're already included via net_module)
        if self.time_warp is not None and time_warp_param_count > 0:
            self.log.info(f"[Optimizer] Including time_warp parameters ({time_warp_param_count} params)")

        # Build optimizer with layerwise learning rate
        base_lr = opt.lr
        new_lr = opt.lr * new_module_lr_mult

        param_groups = []
        if base_params:
            param_groups.append({'params': base_params, 'lr': base_lr, 'name': 'base'})
        if new_params:
            param_groups.append({'params': new_params, 'lr': new_lr, 'name': 'new_modules'})

        # Add PET noise modulator parameters (if enabled)
        if self.pet_noise_modulator is not None:
            pet_noise_params = list(self.pet_noise_modulator.parameters())
            pet_noise_param_count = sum(p.numel() for p in pet_noise_params)
            param_groups.append({
                'params': pet_noise_params, 
                'lr': new_lr,  # Use new-module learning rate.
                'name': 'pet_noise_modulator'
            })
            self.log.info(f"[Optimizer] Including pet_noise_modulator parameters ({pet_noise_param_count:,} params)")

        if not param_groups:
            # Fallback: no parameters to optimize (shouldn't happen)
            param_groups = [{'params': list(self.net_module.parameters()), 'lr': base_lr}]

        optimizer = AdamW(param_groups, weight_decay=opt.l2_norm)

        # Log parameter group info
        if self.is_main:
            base_param_count = sum(p.numel() for p in base_params)
            new_param_count = sum(p.numel() for p in new_params)
            self.log.info(f"[Optimizer] Layerwise LR enabled (mult={new_module_lr_mult}x)")
            self.log.info(f"  Base params: {base_param_count:,} @ lr={base_lr:.2e}")
            self.log.info(f"  New module params: {new_param_count:,} @ lr={new_lr:.2e}")
        
        if opt.lr_gamma < 1.0:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
        else:
            scheduler = None

        # ===================== Learning Rate Warmup =====================
        # Warmup helps stabilize early training by gradually increasing LR
        warmup_iters = getattr(opt, 'warmup_iters', 1000)
        if warmup_iters > 0 and self.is_main:
            self.log.info(f"[Warmup] LR warmup enabled for {warmup_iters} iterations")
        
        # ===================== Log Training Config =====================
        if self.is_main:
            self.log.info("-" * 60)
            self.log.info("TRAINING CONFIGURATION")
            self.log.info(f"  World size: {self.world_size} GPUs")
            self.log.info(f"  Batch size (total): {opt.batch_size}")
            self.log.info(f"  Batch size (per GPU): {batch_size_per_gpu}")
            self.log.info(f"  Microbatch: {opt.microbatch}")
            self.log.info(f"  Loader batch size: {loader_batch_size}")
            if hasattr(opt, 'timestep_strategy'):
                self.log.info(f"  Timestep strategy: {opt.timestep_strategy}")
            self.log.info("-" * 60)
            
        # ===================== Training Loop =====================
        self.net.train()
        train_iter = iter(train_loader)
        viz_freq = getattr(opt, 'viz_freq', 1000)
        test_nfe_interval = getattr(opt, 'test_nfe_interval', 5000)
        param_check_interval = getattr(opt, 'param_check_interval', 5000)

        if self.is_main:
            self.log.info("Starting training loop...")
        
        epoch = 0
        samples_per_epoch = len(train_dataset)
        iters_per_epoch = samples_per_epoch // opt.batch_size if opt.batch_size > 0 else 1
        
        # Set initial epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(0)
        
        for iteration in range(opt.num_itr):
            
            # Set epoch for distributed sampler (for proper shuffling) - only at epoch boundaries
            if train_sampler is not None:
                new_epoch = iteration // max(iters_per_epoch, 1)
                if new_epoch != epoch:
                    epoch = new_epoch
                    train_sampler.set_epoch(epoch)
            
            # Accumulate gradients over microbatches
            optimizer.zero_grad()
            total_loss = 0
            batch_metrics = {'loss': 0, 'pred_mean': 0, 'pred_std': 0}
            # Add time warp metrics if enabled
            if self.time_warp is not None:
                batch_metrics.update({
                    'warp_reg': 0, 'warp_a_mean': 0, 'warp_a_std': 0,
                    'warp_b_mean': 0, 'warp_b_std': 0
                })
            viz_data = None
            
            for micro_idx in range(opt.microbatch):
                # Get next batch
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                # Extract data
                x0 = batch['t2'].to(opt.device)
                x1 = batch['t1'].to(opt.device)
                pet = batch['pet'].to(opt.device)
                # Extract LAVA water/fat if available
                lava_water = batch.get('lava_water')
                if lava_water is not None:
                    lava_water = lava_water.to(opt.device)
                lava_fat = batch.get('lava_fat')
                if lava_fat is not None:
                    lava_fat = lava_fat.to(opt.device)
                
                # Extract Body Part ID if available
                body_part = batch.get('body_part_id')
                if body_part is not None:
                    body_part = body_part.to(opt.device)

                # Extract organ mask if available (from JSON)
                organ_mask = batch.get('organ_mask')
                if organ_mask is not None:
                    organ_mask = organ_mask.to(opt.device)

                # Compute loss
                loss, metrics = self.compute_loss(x0, x1, pet,
                                                  lava_water=lava_water, lava_fat=lava_fat,
                                                  body_part=body_part, organ_mask=organ_mask)
                loss = loss / opt.microbatch

                # ===================== Stability: Loss Anomaly Detection =====================
                # Skip batches with abnormally high loss to prevent gradient explosion
                loss_threshold = getattr(opt, 'loss_anomaly_threshold', 5.0)
                if loss.item() > loss_threshold:
                    if self.is_main and micro_idx == 0:
                        self.log.warning(
                            f"[Iter {iteration:06d}] ⚠️ Anomaly detected: loss={loss.item():.4f} > threshold={loss_threshold:.1f}, skipping batch"
                        )
                    # Zero out gradients and skip this microbatch
                    optimizer.zero_grad()
                    continue

                # Backward with gradient scaling for AMP
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()
                
                # Accumulate metrics
                for key in batch_metrics:
                    if key in metrics:
                        batch_metrics[key] += metrics[key] / opt.microbatch
                
                # Store first microbatch for visualization (main process only)
                # Use net_module (unwrapped) to avoid DDP sync issues
                if self.is_main and micro_idx == 0 and self.visualizer and iteration % viz_freq == 0:
                    amp_enabled = self.use_amp or self.use_bf16
                    with torch.no_grad(), autocast(self.autocast_device, enabled=amp_enabled, dtype=self.amp_dtype):
                        step = torch.randint(0, len(self.diffusion.betas), (x0.shape[0],), device=x0.device)
                        xt = self.diffusion.q_sample(step, x0, x1)
                        pet_cond = pet if getattr(self.opt, 'cond_pet', False) else None
                        water_cond = lava_water if getattr(self.opt, 'cond_lava_water', False) else None
                        fat_cond = lava_fat if getattr(self.opt, 'cond_lava_fat', False) else None
                        body_part_cond = body_part if getattr(self.opt, 'cond_body_part', False) else None
                        organ_mask_cond = organ_mask if getattr(self.opt, 'cond_body_part', False) else None
                        pred = self.net_module(xt, step, x1=x1, pet=pet_cond,
                                              lava_water=water_cond, lava_fat=fat_cond,
                                              body_part_id=body_part_cond, organ_mask=organ_mask_cond)
                        
                        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
                        t2_pred_viz = xt - std_fwd * pred
                        
                        viz_data = {
                            't1': x1.clone(),
                            't2_true': x0.clone(),
                            't2_pred': t2_pred_viz.clamp(-1, 1),
                            'pet': pet.clone() if getattr(self.opt, 'cond_pet', False) else torch.zeros_like(x1)
                        }
            
            # Unscale gradients before clipping (for AMP)
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)

            # Compute gradient norm
            total_norm = 0
            for p in self.net_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5

            # ===================== Stability: Gradient Clipping =====================
            # Reduced from 1.0 to 0.5 for better stability
            grad_clip_norm = getattr(opt, 'grad_clip_norm', 0.5)
            torch.nn.utils.clip_grad_norm_(self.net_module.parameters(), grad_clip_norm)

            # ===================== Learning Rate Warmup =====================
            # Linearly increase LR from 0 to target during warmup period
            if warmup_iters > 0 and iteration < warmup_iters:
                warmup_factor = (iteration + 1) / warmup_iters
                for param_group in optimizer.param_groups:
                    if 'initial_lr' not in param_group:
                        param_group['initial_lr'] = param_group['lr']
                    param_group['lr'] = param_group['initial_lr'] * warmup_factor
            
            if self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            self.ema.update()
            if self.time_warp_ema is not None:
                self.time_warp_ema.update()
            if self.pet_noise_modulator_ema is not None:
                self.pet_noise_modulator_ema.update()

            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()
            
            # ===================== Logging & Visualization (Main Process Only) =====================
            if self.is_main:
                # Log metrics to visualizer
                if self.visualizer:
                    self.visualizer.log_iteration(
                        iteration, total_loss, batch_metrics,
                        lr=current_lr, grad_norm=grad_norm
                    )
                    
                    # Create training visualizations
                    if viz_data is not None:
                        self.visualizer.visualize_training_batch(
                            iteration, viz_data['t1'], viz_data['t2_true'],
                            viz_data['t2_pred'], viz_data['pet'],
                            total_loss, batch_metrics, lr=current_lr
                        )
                        self.visualizer.visualize_data_batch(iteration, viz_data)
                    
                    self.visualizer.update_metrics_plot(iteration)
                
                # Console logging
                if iteration % 100 == 0:
                    log_msg = (
                        f"[Iter {iteration:06d}] Loss: {total_loss:.4f} | "
                        f"LR: {current_lr:.2e} | GradNorm: {grad_norm:.4f}"
                    )
                    # Add time warp metrics if present
                    if 'warp_a_mean' in batch_metrics:
                        log_msg += f" | Warp(a={batch_metrics['warp_a_mean']:.3f}, b={batch_metrics['warp_b_mean']:.3f})"
                    self.log.info(log_msg)
                
                # Save checkpoint
                if iteration % opt.save_interval == 0 and iteration > 0:
                    self.save_checkpoint(opt, iteration)
            
            # Validation (all processes must participate for model state sync)
            if iteration % opt.val_interval == 0 and iteration > 0:
                val_loss = self.validate(opt, val_loader, iteration)
                if self.is_main:
                    if self.visualizer:
                        self.visualizer.log_validation(iteration, val_loss)
                    
                # Synchronize after validation
                if self.world_size > 1:
                    dist.barrier()
            
            # NFE testing (main process only)
            if self.is_main:
                if test_nfe_callback and iteration % test_nfe_interval == 0 and iteration > 0:
                    self.log.info(f"Testing NFE performance at iteration {iteration}...")
                    # Ensure net is in train mode after NFE test
                    try:
                        test_nfe_callback(self, val_dataset, opt.device, iteration)
                    finally:
                        self.net.train()  # Always restore train mode

    def save_checkpoint(self, opt, iteration):
        """Save checkpoint (should only be called from main process)"""
        if not self.is_main:
            return
            
        ckpt_path = opt.ckpt_path / f"ckpt_{iteration:06d}.pt"

        # Use unwrapped module state dict
        checkpoint = {
            'net': self.net_module.state_dict(),
            'ema': self.ema.state_dict(),
            'iteration': iteration,
            'opt': opt,
        }

        # Include time_warp if enabled
        if self.time_warp is not None:
            checkpoint['time_warp'] = self.time_warp.state_dict()
            checkpoint['time_warp_ema'] = self.time_warp_ema.state_dict()

        # Include pet_noise_modulator if enabled
        if self.pet_noise_modulator is not None:
            checkpoint['pet_noise_modulator'] = self.pet_noise_modulator.state_dict()
            checkpoint['pet_noise_modulator_ema'] = self.pet_noise_modulator_ema.state_dict()

        torch.save(checkpoint, ckpt_path)

        # Also save as latest
        latest_path = opt.ckpt_path / "latest.pt"
        torch.save(checkpoint, latest_path)

        self.log.info(f"[Checkpoint] Saved to {ckpt_path}")

    @torch.no_grad()
    def validate(self, opt, val_loader, iteration):
        """Validation step - all processes participate for DDP sync"""
        self.net.eval()

        val_losses = []

        for batch in val_loader:
            x0 = batch['t2'].to(opt.device)
            x1 = batch['t1'].to(opt.device)
            pet = batch['pet'].to(opt.device)

            # Extract LAVA water/fat if available
            lava_water = batch.get('lava_water')
            if lava_water is not None:
                lava_water = lava_water.to(opt.device)
            lava_fat = batch.get('lava_fat')
            if lava_fat is not None:
                lava_fat = lava_fat.to(opt.device)

            # Extract Body Part ID if available
            body_part = batch.get('body_part_id')
            if body_part is not None:
                body_part = body_part.to(opt.device)

            # Extract organ mask if available (from JSON)
            organ_mask = batch.get('organ_mask')
            if organ_mask is not None:
                organ_mask = organ_mask.to(opt.device)

            loss, _ = self.compute_loss(x0, x1, pet,
                                        lava_water=lava_water, lava_fat=lava_fat,
                                        body_part=body_part, organ_mask=organ_mask)
            val_losses.append(loss.item())

            if len(val_losses) >= 50:  # Validate on subset
                break

        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        
        if self.is_main:
            self.log.info(f"[Val] Iter {iteration:06d} | Loss: {avg_val_loss:.4f}")

        self.net.train()

        return avg_val_loss

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: 
            pred_x0.clamp_(-1., 1.)
        return pred_x0

    def ddpm_sampling(
        self, opt, x1, pet, mask=None, cond=None, clip_denoise=False,
        nfe=None, log_count=10, verbose=True,
        visualize_trajectory=False, viz_save_path=None,
        early_stop=None, early_stop_ratio=None,
        sigma_min=2e-3,        # σ floor
        match_intensity=False, # intensity normalization
        ref_mean=0.0, ref_std=1.0,  # training set statistics
        histogram_match=None,  # histogram matching
        histogram_ref=None,    # reference image (ground truth)
        histogram_method="full", # matching method
        integrator="heun",     # ODE integrator ("euler" or "heun")
        lava_water=None,       # LAVA water conditioning
        lava_fat=None,         # LAVA fat conditioning
        body_part=None,        # Body Part conditioning
        organ_mask=None,       # Organ mask [B, num_organs] (from JSON)
        ):
        """
        DDPM sampling for medical modality conversion (matches original HADSB interface)

        Args:
            early_stop: Stop N steps before completion
            early_stop_ratio: Stop at ratio of total steps
            sigma_min: Clamp σ >= sigma_min to avoid last-step explosion
            match_intensity: If True, linearly rescale final x0 to match training statistics
            integrator: ODE integration method - "euler" (default) or "heun" (RK2, more accurate)
            lava_water: LAVA water conditioning [B, 1, H, W]
            lava_fat: LAVA fat conditioning [B, 1, H, W]
            body_part: Body Part indices [B]
            organ_mask: Organ mask [B, num_organs] (from JSON)
        """

        # --- Create timesteps ---
        nfe = nfe or opt.interval - 1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)

        from .util import space_indices
        steps = space_indices(opt.interval, nfe + 1)

        # --- Early-stop logic ---
        original_nfe = nfe
        if early_stop is not None:
            effective_nfe = max(1, nfe - early_stop)
            steps = space_indices(opt.interval, effective_nfe + 1)
            self.log.info(f"[Early Stop] Stopping {early_stop} steps before completion: {nfe} -> {effective_nfe}")
        elif early_stop_ratio is not None:
            effective_nfe = max(1, int(nfe * early_stop_ratio))
            steps = space_indices(opt.interval, effective_nfe + 1)
            self.log.info(f"[Early Stop] Stopping at {early_stop_ratio:.1%} of steps: {nfe} -> {effective_nfe}")

        # --- Build logging steps ---
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        if early_stop is not None or early_stop_ratio is not None:
            self.log.info(f"[Medical DDPM Sampling] steps={opt.interval}, original_nfe={original_nfe}, effective_nfe={len(steps)-1}, {log_steps=}!")
        else:
            self.log.info(f"[Medical DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        # --- Preprocess inputs ---
        x1 = x1.to(opt.device)
        pet = pet.to(opt.device) if pet is not None else None
        lava_water = lava_water.to(opt.device) if lava_water is not None else None
        lava_fat = lava_fat.to(opt.device) if lava_fat is not None else None
        body_part = body_part.to(opt.device, dtype=torch.long) if body_part is not None else None
        organ_mask = organ_mask.to(opt.device, dtype=torch.float32) if organ_mask is not None else None
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step_t = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)

                # Handle PET conditioning with zero-padding fallback
                # If network was trained with PET but sampling disables it, use zeros
                if getattr(opt, 'cond_pet', False):
                    # Sampling config enables PET - use actual PET data
                    pet_cond = pet
                elif self.net.cond_pet:
                    # Network expects PET but sampling disabled it - use zero padding
                    # Create zeros with same shape as x1 (which should match pet)
                    pet_cond = torch.zeros_like(x1)
                    if not hasattr(self, '_pet_zero_warning_shown'):
                        self.log.info("[PET] Network trained with PET but sampling disabled. Using zero-padding to maintain channel compatibility.")
                        self._pet_zero_warning_shown = True
                else:
                    # Network doesn't expect PET
                    pet_cond = None

                # Handle LAVA water/fat conditioning
                water_cond = lava_water if getattr(opt, 'cond_lava_water', False) else None
                if water_cond is not None and water_cond.shape[0] != xt.shape[0]:
                    water_cond = water_cond[:xt.shape[0]]
                fat_cond = lava_fat if getattr(opt, 'cond_lava_fat', False) else None
                if fat_cond is not None and fat_cond.shape[0] != xt.shape[0]:
                    fat_cond = fat_cond[:xt.shape[0]]
                
                # Handle Body Part conditioning
                body_part_cond = body_part if getattr(opt, 'cond_body_part', False) else None
                if body_part_cond is not None and body_part_cond.shape[0] != xt.shape[0]:
                    body_part_cond = body_part_cond[:xt.shape[0]]

                # Handle organ mask conditioning (from JSON)
                organ_mask_cond = organ_mask if getattr(opt, 'cond_body_part', False) else None
                if organ_mask_cond is not None and organ_mask_cond.shape[0] != xt.shape[0]:
                    organ_mask_cond = organ_mask_cond[:xt.shape[0]]

                # Use autocast for mixed precision during inference
                amp_enabled = self.use_amp or self.use_bf16
                with autocast(self.autocast_device, enabled=amp_enabled, dtype=self.amp_dtype):
                    out = self.net(xt, step_t, x1=x1, pet=pet_cond,
                                  lava_water=water_cond, lava_fat=fat_cond,
                                  body_part_id=body_part_cond, organ_mask=organ_mask_cond)

                # Force a sigma floor for stability.
                std_fwd = self.diffusion.get_std_fwd(step_t, xdim=xt.shape[1:])
                std_fwd = std_fwd.clamp_min(sigma_min)
                pred_x0 = xt - std_fwd * out

                if clip_denoise:
                    pred_x0.clamp_(-1., 1.)
                return pred_x0

            # Create noise_scale_fn for PET-guided noise modulation in reverse process
            noise_scale_fn = None
            use_pet_noise_mod = getattr(opt, 'use_pet_noise_modulation', False)
            
            if use_pet_noise_mod and self.pet_noise_modulator is not None and pet is not None:
                # Use EMA parameters if available (copy to model, then use model)
                modulator = self.pet_noise_modulator
                if self.pet_noise_modulator_ema is not None:
                    # ExponentialMovingAverage stores averaged params, copy them to model for inference
                    self.pet_noise_modulator_ema.copy_to(modulator.parameters())
                modulator.eval()
                
                def noise_scale_fn(pred_x0):
                    """Compute noise_scale from PET and semantic features"""
                    with torch.no_grad():
                        # Get body_part and organ_mask for semantic conditioning
                        bp = body_part if body_part is not None else None
                        om = organ_mask if organ_mask is not None else None
                        
                        # Ensure batch size matches
                        pet_batch = pet
                        if pet_batch.shape[0] != pred_x0.shape[0]:
                            pet_batch = pet_batch[:pred_x0.shape[0]]
                        if bp is not None and bp.shape[0] != pred_x0.shape[0]:
                            bp = bp[:pred_x0.shape[0]]
                        if om is not None and om.shape[0] != pred_x0.shape[0]:
                            om = om[:pred_x0.shape[0]]
                        
                        return modulator(pet_batch, body_part=bp, organ_mask=om)
                
                if not hasattr(self, '_pet_noise_mod_sampling_info_shown'):
                    self.log.info("[PET Noise Modulation] Enabled in reverse sampling process")
                    self._pet_noise_mod_sampling_info_shown = True

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode,
                integrator=integrator, log_steps=log_steps, verbose=verbose,
                noise_scale_fn=noise_scale_fn,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        # --- Intensity normalization (final step) ---
        if match_intensity:
            last_x0 = pred_x0[:, -1]
            m = last_x0.mean(dim=(1,2,3), keepdim=True)
            s = last_x0.std(dim=(1,2,3), keepdim=True) + 1e-6
            last_x0 = (last_x0 - m) / s * ref_std + ref_mean
            pred_x0[:, -1] = last_x0

        # --- Improved histogram matching (final step) ---
        if histogram_match and histogram_ref is not None:
            from tools.histogram_matching import (
                histogram_matching_torch, match_histogram_percentiles,
                apply_improved_histogram_matching, hybrid_histogram_matching_torch
            )

            # Get final prediction
            final_pred = pred_x0[:, -1]  # [B, C, H, W]

            # Apply histogram matching with new methods
            if histogram_method in ["structure_preserving", "local_adaptive", "tissue_aware"]:
                # Use improved histogram matching methods
                matched_pred, structure_metrics = apply_improved_histogram_matching(
                    final_pred, histogram_ref,
                    method=histogram_method,
                    evaluate_structure=True,
                    structure_weight=getattr(self, 'structure_weight', 0.4)
                )

                if verbose and structure_metrics:
                    self.log.info(f"[Structure Metrics] Gradient correlation: {structure_metrics['mean_gradient_correlation']:.3f}, "
                                f"Edge preservation: {structure_metrics['edge_preservation_ratio']:.3f}")

            elif histogram_method == "percentile":
                # Legacy percentile method
                matched_pred = torch.zeros_like(final_pred)
                for b in range(final_pred.shape[0]):
                    matched_pred[b] = match_histogram_percentiles(
                        final_pred[b], histogram_ref[b] if histogram_ref.ndim == 4 else histogram_ref[0]
                    )

            else:  # "full" or "masked" - legacy methods
                matched_pred = torch.zeros_like(final_pred)
                for b in range(final_pred.shape[0]):
                    matched_pred[b] = histogram_matching_torch(
                        final_pred[b], histogram_ref[b] if histogram_ref.ndim == 4 else histogram_ref[0]
                    )

            # Update the final prediction
            pred_x0[:, -1] = matched_pred
            xs[:, -1] = matched_pred  # Also update xs for consistency

            if verbose:
                self.log.info(f"[Histogram Matching] Applied {histogram_method} matching to final predictions")

        # --- Trajectory visualization ---
        if visualize_trajectory and viz_save_path is not None:
            self._create_trajectory_visualization(
                x1, pet, xs, pred_x0, log_steps, nfe, steps, opt, viz_save_path
            )

        return xs, pred_x0

    def ddpm_sampling_warped(
        self, opt, x1, pet, body_part, organ_mask,
        mask=None, clip_denoise=False,
        nfe=None, log_count=10, verbose=True,
        sigma_min=2e-3,
        lava_water=None, lava_fat=None,
    ):
        """
        DDPM sampling with semantic time warp.

        Uses warped time coordinates for sampling, where different body parts
        and organ contexts follow different effective noise schedules.

        Args:
            body_part: [B] body part indices (required for time warp)
            organ_mask: [B, num_organs] organ mask
            Other args same as ddpm_sampling

        Returns:
            xs: [B, log_count, C, H, W] sampled states at log steps
            pred_x0: [B, log_count, C, H, W] predicted clean images
        """
        if self.time_warp is None:
            self.log.warning("[TimeWarp] time_warp is None, falling back to standard sampling")
            return self.ddpm_sampling(
                opt, x1, pet, mask=mask,
                clip_denoise=clip_denoise, nfe=nfe, log_count=log_count,
                verbose=verbose, sigma_min=sigma_min,
                body_part=body_part, organ_mask=organ_mask,
                lava_water=lava_water, lava_fat=lava_fat,
            )

        # --- Setup time steps ---
        nfe = nfe or opt.interval - 1
        T = opt.interval
        assert 0 < nfe < T == len(self.diffusion.betas)

        from .util import space_indices
        steps = space_indices(T, nfe + 1)

        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        if verbose:
            self.log.info(f"[Warped DDPM Sampling] steps={T}, {nfe=}, {log_steps=}")

        # --- Prepare inputs ---
        B = x1.shape[0]
        device = opt.device
        x1 = x1.to(device)
        pet = pet.to(device) if pet is not None else None
        body_part = body_part.to(device, dtype=torch.long)
        organ_mask = organ_mask.to(device, dtype=torch.float32) if organ_mask is not None else None

        if mask is not None:
            mask = mask.to(device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        # --- Pre-compute warped time grid ---
        steps_tensor = torch.tensor(steps, device=device, dtype=torch.float32)
        steps_normalized = steps_tensor / T  # [nfe+1], normalized to [0, 1]

        # Use EMA for time_warp during sampling
        time_warp_ctx = self.time_warp_ema.average_parameters() if self.time_warp_ema else nullcontext()

        with time_warp_ctx:
            with torch.no_grad():
                s_grid = self.time_warp.warp_grid(steps_normalized, body_part, organ_mask)  # [B, nfe+1]
                s_grid = s_grid * T  # Convert to continuous indices [0, T)

        # --- Sampling loop ---
        xt = x1.clone()
        xs = []
        pred_x0s = []

        with self.ema.average_parameters():
            self.net.eval()

            steps_reversed = list(range(len(steps) - 1, 0, -1))
            if verbose:
                from tqdm import tqdm
                steps_reversed = tqdm(steps_reversed, desc='Warped DDPM sampling')

            for i in steps_reversed:
                s_curr = s_grid[:, i]      # [B]
                s_prev = s_grid[:, i - 1]  # [B]

                # Get noise level at warped time
                noise_level_s = self.diffusion.get_noise_level(s_curr)

                # Handle conditioning
                pet_cond = pet if getattr(opt, 'cond_pet', False) else None
                if pet_cond is None and hasattr(self.net_module, 'cond_pet') and self.net_module.cond_pet:
                    pet_cond = torch.zeros_like(x1)

                water_cond = lava_water if getattr(opt, 'cond_lava_water', False) else None
                fat_cond = lava_fat if getattr(opt, 'cond_lava_fat', False) else None
                body_part_cond = body_part if getattr(opt, 'cond_body_part', False) else None
                organ_mask_cond = organ_mask if getattr(opt, 'cond_body_part', False) else None

                # Network forward with warped noise level
                amp_enabled = self.use_amp or self.use_bf16
                with torch.no_grad():
                    with autocast(self.autocast_device, enabled=amp_enabled, dtype=self.amp_dtype):
                        out = self.net(xt, steps[i], x1=x1, pet=pet_cond,
                                      lava_water=water_cond, lava_fat=fat_cond,
                                      body_part_id=body_part_cond, organ_mask=organ_mask_cond,
                                      noise_level_override=noise_level_s)

                # Compute pred_x0 using warped schedule
                std_fwd_s = self.diffusion.get_std_fwd_warped(s_curr, xdim=xt.shape[1:])
                std_fwd_s = std_fwd_s.clamp_min(sigma_min)
                pred_x0 = xt - std_fwd_s * out

                if clip_denoise:
                    pred_x0 = pred_x0.clamp(-1., 1.)

                # Compute noise_scale for PET-guided noise modulation
                noise_scale = None
                use_pet_noise_mod = getattr(opt, 'use_pet_noise_modulation', False)
                ot_ode = getattr(opt, 'ot_ode', False)
                
                if use_pet_noise_mod and self.pet_noise_modulator is not None and pet is not None and not ot_ode:
                    modulator = self.pet_noise_modulator
                    # Note: EMA params should already be copied to modulator in ddpm_sampling setup
                    with torch.no_grad():
                        noise_scale = modulator(pet, body_part=body_part, organ_mask=organ_mask)

                # Posterior step using warped schedule
                xt = self.diffusion.p_posterior_warped(
                    s_prev, s_curr, xt, pred_x0, ot_ode=ot_ode, noise_scale=noise_scale
                )

                # Inpainting mask
                if mask is not None:
                    xt_true = x1
                    if not getattr(opt, 'ot_ode', False):
                        std_sb_s = self.diffusion.interp_schedule(self.diffusion.std_sb, s_prev)
                        std_sb_s = unsqueeze_xdim(std_sb_s, x1.shape[1:])
                        xt_true = xt_true + std_sb_s * torch.randn_like(xt_true)
                    xt = (1. - mask) * xt_true + mask * xt

                # Log
                if steps[i - 1] in log_steps:
                    xs.append(xt.detach().cpu())
                    pred_x0s.append(pred_x0.detach().cpu())

        # Stack results
        xs = torch.stack(xs[::-1], dim=1)  # Reverse to get chronological order
        pred_x0s = torch.stack(pred_x0s[::-1], dim=1)

        return xs, pred_x0s

    def _create_trajectory_visualization(self, x1, pet, xs, pred_x0, log_steps, nfe, full_steps, opt, viz_save_path):
        """Create and save trajectory visualization"""
        from visualization.medical_viz import create_sampling_trajectory_visualization

        # Prepare NFE information
        nfe_info = {
            'actual_nfe': nfe,
            'total_steps': len(full_steps),
            'solver': 'OT-ODE' if getattr(opt, 'ot_ode', False) else 'DDPM',
            'logged_steps': len(log_steps)
        }

        try:
            create_sampling_trajectory_visualization(
                x1, pet, xs, pred_x0, log_steps, nfe_info, viz_save_path,
                show_pet=getattr(opt, 'cond_pet', False), max_steps=10
            )
            self.log.info(f"[Trajectory Viz] Saved to {viz_save_path}")
        except Exception as e:
            self.log.warning(f"[Trajectory Viz] Failed to create visualization: {e}")

    def _compute_trajectory_metrics(self, xs, pred_x0):
        """Compute metrics along the sampling trajectory"""
        import torch.nn.functional as F

        metrics = {
            'similarities': [],
            'variances': []
        }

        # Final prediction for comparison
        final_pred = pred_x0[:, 0, ...]  # First logged step is final

        for step_idx in range(pred_x0.shape[1]):
            current_pred = pred_x0[:, step_idx, ...]

            # Compute similarity to final prediction
            similarity = F.cosine_similarity(
                current_pred.flatten(1), final_pred.flatten(1), dim=1
            ).mean().item()
            metrics['similarities'].append(similarity)

            # Compute spatial variance
            variance = current_pred.var().item()
            metrics['variances'].append(variance)

        return metrics

    @torch.no_grad()
    def sample(self, x1, pet, clip_denoise=True, nfe=None, verbose=True, visualize_trajectory=False, viz_save_path=None,
               early_stop=None, early_stop_ratio=None, histogram_match=None, histogram_ref=None, histogram_method="structure_preserving",
               structure_weight=0.4, integrator="euler", lava_water=None, lava_fat=None,
               body_part=None, organ_mask=None):
        """
        Sample T2 from T1 and PET conditioning (simplified interface)

        Args:
            x1: Source T1 images [B, 1, H, W]
            pet: PET conditioning [B, 1, H, W]
            clip_denoise: Whether to clip to [-1, 1]
            nfe: Number of function evaluations
            verbose: Whether to show progress
            visualize_trajectory: Whether to create trajectory visualization
            viz_save_path: Path to save trajectory visualization
            early_stop: Stop N steps before completion
            early_stop_ratio: Stop at ratio of total steps
            histogram_match: Whether to apply histogram matching
            histogram_ref: Reference images for histogram matching [B, 1, H, W]
            histogram_method: Histogram matching method ("full", "percentile", "masked")
            lava_water: Optional LAVA water images [B, 1, H, W]
            lava_fat: Optional LAVA fat images [B, 1, H, W]
            body_part: Optional Body Part indices [B]
            organ_mask: Optional organ mask [B, num_organs] (from JSON)

        Returns:
            Sampled T2 images [B, 1, H, W]
        """
        # Use more log points for trajectory visualization
        log_count = 10 if visualize_trajectory else 1

        # Store structure_weight for access in ddpm_sampling
        self.structure_weight = structure_weight

        # Use warped sampling if time_warp is available and body_part is provided
        use_warped = (
            self.time_warp is not None and
            body_part is not None and
            (getattr(self.opt, 'use_time_warp', False) or
             getattr(self.opt, 'use_semantic_time_warp', False))
        )

        if use_warped:
            # Use time-warped sampling
            xs, _ = self.ddpm_sampling_warped(
                self.opt, x1, pet, body_part, organ_mask,
                lava_water=lava_water, lava_fat=lava_fat,
                clip_denoise=clip_denoise, nfe=nfe,
                log_count=log_count, verbose=verbose,
                sigma_min=2e-3,
            )
        else:
            # Use standard sampling
            xs, _ = self.ddpm_sampling(
                self.opt, x1, pet,
                lava_water=lava_water, lava_fat=lava_fat,
                body_part=body_part, organ_mask=organ_mask,
                clip_denoise=clip_denoise, nfe=nfe,
                log_count=log_count, verbose=verbose,
                visualize_trajectory=visualize_trajectory, viz_save_path=viz_save_path,
                early_stop=early_stop, early_stop_ratio=early_stop_ratio,
                histogram_match=histogram_match, histogram_ref=histogram_ref,
                histogram_method=histogram_method, integrator=integrator
            )
        # Return the final result (first logged step is the final denoised result)
        return xs[:, 0, ...]

    @torch.no_grad()
    def sample_with_trajectory(self, x1, pet, clip_denoise=True, nfe=None, verbose=True, log_count=10,
                              early_stop=None, early_stop_ratio=None, histogram_match=None, histogram_ref=None, histogram_method="structure_preserving",
                              structure_weight=0.4, integrator="euler", lava_water=None, lava_fat=None,
                              body_part=None, organ_mask=None):
        """
        Sample T2 from T1 with full trajectory return

        Args:
            x1: Source T1 images [B, 1, H, W]
            pet: PET conditioning [B, 1, H, W]
            clip_denoise: Whether to clip to [-1, 1]
            nfe: Number of function evaluations
            verbose: Whether to show progress
            log_count: Number of trajectory steps to return
            lava_water: Optional LAVA water images [B, 1, H, W]
            lava_fat: Optional LAVA fat images [B, 1, H, W]
            body_part: Optional Body Part indices [B]
            organ_mask: Optional organ mask [B, num_organs] (from JSON)
            early_stop: Stop N steps before completion
            early_stop_ratio: Stop at ratio of total steps

        Returns:
            final_t2: Final sampled T2 images [B, 1, H, W]
            xs_trajectory: Full sampling trajectory [B, log_count, 1, H, W]
            pred_x0_trajectory: Predicted clean trajectory [B, log_count, 1, H, W]
            trajectory_data: Dict with step indices and metrics
        """
        # Store structure_weight for access in ddpm_sampling
        self.structure_weight = structure_weight

        # Use the standardized ddpm_sampling interface
        xs, pred_x0 = self.ddpm_sampling(
            self.opt, x1, pet,
            lava_water=lava_water, lava_fat=lava_fat,
            body_part=body_part, organ_mask=organ_mask,
            clip_denoise=clip_denoise, nfe=nfe,
            log_count=log_count, verbose=verbose,
            early_stop=early_stop, early_stop_ratio=early_stop_ratio,
            histogram_match=histogram_match, histogram_ref=histogram_ref,
            histogram_method=histogram_method, integrator=integrator
        )

        # Compute trajectory metrics
        step_metrics = self._compute_trajectory_metrics(xs, pred_x0)

        # Create step indices - need to match the actual sampling that was done
        from .util import space_indices
        original_nfe = nfe or self.opt.interval - 1

        # Apply same early stopping logic as in ddpm_sampling
        if early_stop is not None:
            effective_nfe = max(1, original_nfe - early_stop)
        elif early_stop_ratio is not None:
            effective_nfe = max(1, int(original_nfe * early_stop_ratio))
        else:
            effective_nfe = original_nfe

        steps = space_indices(self.opt.interval, effective_nfe + 1)
        log_count_actual = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count_actual)]

        # Ensure step_metrics and log_steps have same length
        actual_logged = xs.shape[1]  # This is the real number of logged steps
        if len(log_steps) != actual_logged:
            # Adjust log_steps to match actual trajectory length
            log_steps = log_steps[:actual_logged]

        # Also trim step_metrics to match
        for key in step_metrics:
            step_metrics[key] = step_metrics[key][:len(log_steps)]

        trajectory_data = {
            'step_indices': log_steps,
            'step_metrics': step_metrics,
            'nfe_info': {
                'actual_nfe': effective_nfe,
                'original_nfe': original_nfe,
                'total_steps': len(steps),
                'solver': 'OT-ODE' if getattr(self.opt, 'ot_ode', False) else 'DDPM',
                'early_stopped': early_stop is not None or early_stop_ratio is not None
            }
        }

        return xs[:, 0, ...], xs, pred_x0, trajectory_data
