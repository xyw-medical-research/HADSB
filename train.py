#!/usr/bin/env python
"""
Fixed training script for Medical HADSB with NFE performance fix
Supports both single-GPU and multi-GPU (DDP) training

This script now uses the unified training loop in MedicalRunner.train()
"""

import os
import sys
import argparse
import warnings

# Suppress pkg_resources deprecation warning from torchmetrics
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning)

import torch
import torch.distributed as dist
from datetime import datetime
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger import Logger
from hadsb.runner_medical import MedicalRunner, is_main_process, get_world_size, get_rank
from dataset.medical import build_medical_dataset


def setup_distributed(requested_device: str = "auto"):
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single-process mode
        rank = 0
        world_size = 1
        local_rank = 0

    if requested_device == "auto":
        has_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{local_rank}" if has_cuda else "cpu")
    else:
        device = torch.device(requested_device)

    if world_size > 1:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{local_rank}")
            )
        else:
            dist.init_process_group(backend="gloo")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_nfe_performance_callback(runner, val_dataset, device, iteration):
    """Callback for testing NFE performance during training"""
    nfe_values = [10, 20, 50]

    # Get a test sample
    sample = val_dataset[0]
    x1 = sample['t1'].unsqueeze(0).to(device)
    x0_true = sample['t2'].unsqueeze(0).to(device)
    pet = sample['pet'].unsqueeze(0).to(device)

    # Handle body part conditioning (from JSON-based region info)
    body_part = sample.get('body_part_id')
    if body_part is not None:
        if isinstance(body_part, torch.Tensor):
            body_part = body_part.unsqueeze(0).to(device)
        else:
            body_part = torch.tensor([body_part], device=device, dtype=torch.long)

    # Handle organ mask conditioning (multi-hot vector from JSON)
    organ_mask = sample.get('organ_mask')
    if organ_mask is not None:
        if isinstance(organ_mask, torch.Tensor):
            organ_mask = organ_mask.unsqueeze(0).to(device)
        else:
            organ_mask = torch.tensor([organ_mask], device=device).float()

    # Extract LAVA water/fat conditioning if available
    lava_water = sample.get('lava_water')
    if lava_water is not None:
        lava_water = lava_water.unsqueeze(0).to(device)
    lava_fat = sample.get('lava_fat')
    if lava_fat is not None:
        lava_fat = lava_fat.unsqueeze(0).to(device)

    results = {}
    with torch.no_grad():
        for nfe in nfe_values:
            try:
                # Sample with specific NFE (include all conditioning inputs)
                x0_pred = runner.sample(x1, pet, clip_denoise=True, nfe=nfe, verbose=False,
                                        body_part=body_part, organ_mask=organ_mask,
                                        lava_water=lava_water, lava_fat=lava_fat)
                x0_pred = x0_pred.to(device)
                
                # Compute MSE
                mse = F.mse_loss(x0_pred, x0_true).item()
                results[nfe] = mse
            except Exception as e:
                runner.log.warning(f"NFE test failed for nfe={nfe}: {e}")
    
    # Log results
    runner.log.info("  NFE Performance:")
    for nfe, mse in results.items():
        runner.log.info(f"    NFE={nfe}: MSE={mse:.6f}")
    
    # Check for degradation
    if len(results) >= 2:
        nfe_vals = sorted(results.keys())
        if results[nfe_vals[-1]] > results[nfe_vals[0]] * 1.1:
            degradation = (results[nfe_vals[-1]] / results[nfe_vals[0]] - 1) * 100
            if degradation > 20:
                runner.log.info(f"  ❌ High degradation ({degradation:.1f}%) - May need more training")
            else:
                runner.log.info(f"  ⚠️  Moderate degradation ({degradation:.1f}%)")
        else:
            improvement = (1 - results[nfe_vals[-1]] / results[nfe_vals[0]]) * 100
            if improvement > 0:
                runner.log.info(f"  ✅ NFE={nfe_vals[-1]} improved by {improvement:.1f}%")
            else:
                runner.log.info(f"  ✅ Low degradation ({-improvement:.1f}%) - Fix is working!")


def main():
    parser = argparse.ArgumentParser(description="Train Medical HADSB (Open-Source Core, Multi-GPU)")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./data/paired_dataset",
                       help="Path to medical dataset")
    parser.add_argument("--image-size", type=int, default=512,
                       help="Image size for training")
    parser.add_argument("--pet-log-scale", action="store_true", default=False,
                       help="Apply log1p + min-max normalization to PET inputs")
    parser.add_argument("--pet-log-eps", type=float, default=1e-6,
                       help="Numerical epsilon to guard PET log scaling")
    parser.add_argument("--pet-norm", type=str, default="max",
                       choices=["max", "p1p99", "zscore"],
                       help="PET normalization method (default: max)")
    
    # Model arguments
    parser.add_argument("--in-channels", type=int, default=1,
                       help="Input channels (T1)")
    parser.add_argument("--out-channels", type=int, default=1,
                       help="Output channels (T2)")
    parser.add_argument("--model-channels", type=int, default=128,
                       help="Base channel count for UNet")
    parser.add_argument("--num-res-blocks", type=int, default=3,
                       help="Number of residual blocks per resolution")
    
    # Diffusion arguments
    parser.add_argument("--beta-max", type=float, default=0.8,
                       help="Maximum beta for diffusion")
    parser.add_argument("--interval", type=int, default=1000,
                       help="Number of diffusion steps")
    parser.add_argument("--t0", type=float, default=1e-4,
                       help="Starting time for diffusion")
    parser.add_argument("--T", type=float, default=1.0,
                       help="End time for diffusion")
    parser.add_argument("--ot-ode", action="store_true", default=False,
                       help="Use OT-ODE for deterministic sampling")
    parser.add_argument("--no-ot-ode", action="store_true", default=False,
                       help="Disable OT-ODE (use stochastic sampling)")
    
    # Training arguments
    parser.add_argument("--num-itr", type=int, default=300000,
                       help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Total batch size across all GPUs")
    parser.add_argument("--microbatch", type=int, default=1,
                       help="Microbatch size for gradient accumulation")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--new-module-lr-mult", type=float, default=1.0,
                       help="Learning rate multiplier for new modules (body_part, etc.) "
                            "relative to base LR. Use >1.0 for Stage 2/3 fine-tuning (e.g., 10.0)")
    parser.add_argument("--lr-gamma", type=float, default=0.99,
                       help="Learning rate decay factor")
    parser.add_argument("--lr-step", type=int, default=1000,
                       help="Learning rate decay step")
    parser.add_argument("--l2-norm", type=float, default=0.0,
                       help="L2 regularization weight")
    parser.add_argument("--ema", type=float, default=0.9999,
                       help="EMA decay rate")

    # Stability arguments (prevent gradient explosion)
    parser.add_argument("--warmup-iters", type=int, default=1000,
                       help="Learning rate warmup iterations (default: 1000)")
    parser.add_argument("--grad-clip-norm", type=float, default=0.5,
                       help="Gradient clipping norm threshold (default: 0.5)")
    parser.add_argument("--loss-anomaly-threshold", type=float, default=5.0,
                       help="Skip batches with loss > threshold (default: 5.0)")
    parser.add_argument("--use-fp16", action="store_true", default=False,
                       help="Use mixed precision training (FP16)")
    parser.add_argument("--use-bf16", action="store_true", default=False,
                       help="Use BFloat16 mixed precision (recommended for RTX 40 series)")
    
    # NFE Fix arguments
    parser.add_argument("--timestep-strategy", type=str, default="mixed",
                       choices=["original", "mixed", "adaptive", "consecutive"],
                       help="Timestep sampling strategy for NFE fix")
    parser.add_argument("--test-nfe-interval", type=int, default=5000,
                       help="Test NFE performance every N iterations")
    
    # Visualization arguments
    parser.add_argument("--viz-freq", type=int, default=500,
                       help="Visualization frequency (iterations)")
    parser.add_argument("--disable-viz", action="store_true",
                       help="Disable training visualizations")
    
    # Conditioning arguments
    parser.add_argument("--cond-x1", action="store_true", default=True,
                       help="Condition on source image x1")
    parser.add_argument("--cond-pet", action="store_true", default=False,
                       help="Condition on PET image")
    # Cross-Attention arguments (for PET conditioning)
    parser.add_argument("--use-cross-attn", action="store_true", default=False,
                       help="Use cross-attention for PET conditioning instead of channel concatenation")
    parser.add_argument("--cross-attn-resolutions", type=str, default="32,16",
                       help="Resolutions for cross-attention (comma-separated, e.g., '32,16,8')")
    parser.add_argument("--use-spatial-cross-attn", action="store_true", default=True,
                       help="Use spatial cross-attention (recommended for better performance)")

    # LAVA water/fat conditioning arguments
    parser.add_argument("--cond-lava-water", action="store_true", default=False,
                       help="Condition on LAVA water image (concatenated with gate)")
    parser.add_argument("--cond-lava-fat", action="store_true", default=False,
                       help="Condition on LAVA fat image (concatenated with gate)")
    parser.add_argument("--flat-data-structure", action="store_true", default=False,
                       help="Use flat data structure (data_dir/modality/ instead of data_dir/split/modality/)")
    parser.add_argument("--val-patients-file", type=str, default=None,
                       help="Optional text file with one patient/group id per line for val split (flat structure)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                       help="Validation split ratio in [0,1] for flat structure when no val file is provided")
    parser.add_argument("--split-seed", type=int, default=42,
                       help="Random seed for ratio-based split in flat structure mode")

    # Organ-specific output heads
    parser.add_argument("--organ-specific-out", action="store_true", default=False,
                       help="Use organ-specific output conv layers (shared norm/act + K independent convs)")

    # Body Part / Organ Semantic Conditioning arguments
    parser.add_argument("--cond-body-part", action="store_true", default=False,
                       help="Condition on body part (semantic embedding using PubMedBERT)")
    parser.add_argument("--body-part-embed-dim", type=int, default=64,
                       help="Embedding dimension for body part conditioning")
    parser.add_argument("--body-part-channel-cond", action="store_true", default=False,
                       help="Enable channel conditioning for body part")
    parser.add_argument("--body-part-time-cond", action="store_true", default=True,
                       help="Enable time conditioning for body part")
    parser.add_argument("--use-semantic-embedding", action="store_true", default=True,
                       help="Use PubMedBERT semantic embedding (vs learnable)")
    parser.add_argument("--region-info-file", type=str, default=None,
                       help="Path to region info JSON (e.g., mri_analysis_results_T2.json)")
    parser.add_argument("--cond-organ-crossattn", action="store_true", default=False,
                       help="Enable organ cross-attention (Phase 3)")
    parser.add_argument("--organ-crossattn-dim", type=int, default=64,
                       help="Cross-attention dimension for organs")

    # Semantic Time Warp arguments
    # Learns a monotonic time-warp g(t; region, organs) -> s for per-sample adaptive schedules
    parser.add_argument("--use-time-warp", action="store_true", default=False,
                       help="Enable semantic time warp for per-sample adaptive schedules")
    parser.add_argument("--time-warp-embed-dim", type=int, default=64,
                       help="Embedding dimension for time warp region/organ embeddings")
    parser.add_argument("--time-warp-hidden-dim", type=int, default=128,
                       help="Hidden dimension for time warp MLP")
    parser.add_argument("--time-warp-a-scale", type=float, default=0.5,
                       help="Scale for warp parameter 'a' (a ∈ [1-scale, 1+scale])")
    parser.add_argument("--time-warp-lambda-a", type=float, default=0.1,
                       help="Regularization weight for warp slope parameter")
    parser.add_argument("--time-warp-lambda-b", type=float, default=0.1,
                       help="Regularization weight for warp shift parameter")
    parser.add_argument("--time-warp-warmup", type=int, default=5000,
                       help="Number of iterations before enabling time warp (warmup period)")
    parser.add_argument("--freeze-base-for-warp", action="store_true", default=False,
                       help="Freeze base UNet when training time warp (Stage 3)")

    # === PET Noise Modulation arguments ===
    # Learns spatial noise modulation from PET + semantic features
    parser.add_argument("--use-pet-noise-modulation", action="store_true", default=False,
                       help="Enable PET-guided noise modulation (learn spatial noise scaling from PET + semantics)")
    parser.add_argument("--pet-noise-min-scale", type=float, default=0.3,
                       help="Minimum noise scale factor (prevents noise from vanishing)")
    parser.add_argument("--pet-noise-max-scale", type=float, default=1.5,
                       help="Maximum noise scale factor")
    parser.add_argument("--pet-noise-hidden-ch", type=int, default=32,
                       help="Hidden channels for PET noise modulator")
    parser.add_argument("--pet-noise-semantic-embed-dim", type=int, default=64,
                       help="Semantic embedding dimension for body part/organ conditioning in noise modulator")

    # Checkpoint arguments
    parser.add_argument("--save-interval", type=int, default=10000,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--val-interval", type=int, default=1000,
                       help="Validate every N iterations")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto/cpu/cuda[:id]")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")

    args = parser.parse_args()

    if not (0.0 <= args.val_ratio <= 1.0):
        parser.error("--val-ratio must be in [0, 1]")

    # Process body part conditioning arguments
    if args.cond_body_part:
        if args.body_part_embed_dim <= 0:
            parser.error("--body-part-embed-dim must be positive when --cond-body-part is set")
        # region_info_file is optional - will use default mapping if not provided
    else:
        args.body_part_channel_cond = False
        args.body_part_time_cond = False
    
    # ===================== Setup Distributed Training =====================
    rank, world_size, local_rank, device = setup_distributed(args.device)
    
    # Set random seed (different for each rank for proper shuffling)
    set_seed(args.seed + rank)
    
    # Create experiment name if not provided
    if args.name is None:
        args.name = f"medical_fixed_{args.timestep_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup paths
    ckpt_path = Path(f"results/{args.name}")
    if is_main_process():
        ckpt_path.mkdir(parents=True, exist_ok=True)
    
    # Wait for main process to create directories
    if world_size > 1:
        dist.barrier()
    
    # ===================== Setup Logger =====================
    log = Logger(rank=rank, log_dir=str(ckpt_path))
    
    # ===================== Create Options =====================
    opt = edict(vars(args))
    opt.ckpt_path = ckpt_path
    opt.device = device
    opt.world_size = world_size
    opt.rank = rank
    opt.local_rank = local_rank
    opt.load = args.resume  # MedicalRunner expects 'load' instead of 'resume'
    
    # ===================== Log Configuration =====================
    if is_main_process():
        log.info("=" * 60)
        log.info("FIXED MEDICAL HADSB TRAINING (UNIFIED)")
        log.info("=" * 60)
        log.info(f"Experiment: {args.name}")
        log.info(f"Timestep Strategy: {args.timestep_strategy}")
        log.info(f"World Size: {world_size} GPUs")
        log.info(f"Device: {device}")
        log.info(f"Checkpoint Path: {ckpt_path}")
        log.info("-" * 60)
        
        # Log all arguments
        for key, value in vars(args).items():
            log.info(f"  {key}: {value}")
        log.info("-" * 60)

        if args.cond_body_part:
            log.info("-" * 60)
            log.info("BODY PART CONDITIONING ENABLED")
            log.info(f"  Embed dim: {args.body_part_embed_dim}")
            log.info(f"  Channel cond: {args.body_part_channel_cond}")
            log.info(f"  Time cond: {args.body_part_time_cond}")
            log.info(f"  Semantic embedding: {args.use_semantic_embedding}")
            log.info(f"  Region info file: {args.region_info_file}")
    
    # ===================== Load Datasets =====================
    log.info("Loading datasets...")
    train_dataset = build_medical_dataset(opt, log, train=True)
    val_dataset = build_medical_dataset(opt, log, train=False)
    
    if is_main_process():
        log.info(f"Train samples: {len(train_dataset)}")
        log.info(f"Val samples: {len(val_dataset)}")
    
    # ===================== Initialize Runner with DDP =====================
    log.info("Initializing model with NFE fix...")
    
    # Create runner with DDP wrapping enabled
    runner = MedicalRunner(
        opt, 
        log, 
        save_opt=is_main_process(),
        wrap_ddp=(world_size > 1)  # Auto-wrap with DDP if multi-GPU
    )
    
    # ===================== Start Training =====================
    try:
        # Use unified training loop with optional NFE callback
        runner.train(
            opt, 
            train_dataset, 
            val_dataset,
            test_nfe_callback=test_nfe_performance_callback if args.test_nfe_interval > 0 else None
        )
        
        if is_main_process():
            log.info("=" * 60)
            log.info("Training completed successfully!")
            log.info("=" * 60)
            
    except KeyboardInterrupt:
        if is_main_process():
            log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
