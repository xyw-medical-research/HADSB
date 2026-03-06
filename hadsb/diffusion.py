# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn

from .util import unsqueeze_xdim
from utils.intensity_calib import match_mean_std, pwl_quantile_match

class PETNoiseModulator(nn.Module):
    """
    Learn a spatial noise modulation factor from PET + semantic context.

    The model uses:
    - PET spatial features
    - Body part semantics
    - Organ presence semantics

    to decide the noise strength per pixel.

    Output is constrained to [min_scale, max_scale].
    """
    def __init__(
        self, 
        in_ch: int = 1, 
        hidden_ch: int = 32,
        min_scale: float = 0.3,
        max_scale: float = 1.5,
        # Semantic conditioning.
        num_body_parts: int = 11,
        num_organs: int = 36,
        semantic_embed_dim: int = 64,
    ):
        """
        Args:
            in_ch: Input channel count (PET is usually 1)
            hidden_ch: Hidden channel count
            min_scale: Minimum noise scale
            max_scale: Maximum noise scale
            num_body_parts: Number of body-part classes
            num_organs: Number of organ classes
            semantic_embed_dim: Semantic embedding dimension
        """
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # ===================== Semantic Embeddings =====================
        # Body part embedding (discrete labels).
        self.body_part_embed = nn.Embedding(num_body_parts + 1, semantic_embed_dim)  # +1 for unknown
        
        # Organ embedding (multi-label linear projection).
        self.organ_proj = nn.Linear(num_organs, semantic_embed_dim)
        
        # Semantic fusion MLP: body_part + organ -> modulation feature.
        self.semantic_mlp = nn.Sequential(
            nn.Linear(semantic_embed_dim * 2, hidden_ch),
            nn.SiLU(),
            nn.Linear(hidden_ch, hidden_ch),
        )
        
        # ===================== Spatial Feature Encoder =====================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(),
        )
        
        # Semantic-spatial fusion (FiLM style).
        self.semantic_scale = nn.Linear(hidden_ch, hidden_ch)
        self.semantic_shift = nn.Linear(hidden_ch, hidden_ch)
        
        # Output head.
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_ch // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_ch // 2, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1].
        )
    
    def forward(
        self, 
        pet: torch.Tensor, 
        body_part: torch.Tensor = None,
        organ_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pet: PET image [B, 1, H, W]
            body_part: Body part index [B] (long tensor)
            organ_mask: Multi-hot organ mask [B, num_organs] (float tensor)
        
        Returns:
            noise_scale: Noise scale [B, 1, H, W] in [min_scale, max_scale]
        """
        batch = pet.shape[0]
        device = pet.device
        
        # ===================== PET Spatial Features =====================
        feat = self.encoder(pet)  # [B, hidden_ch, H, W]
        
        # ===================== Semantic Conditioning =====================
        # Body Part embedding
        if body_part is not None:
            bp_embed = self.body_part_embed(body_part)  # [B, semantic_embed_dim]
        else:
            # Use 0 (unknown) by default.
            bp_embed = self.body_part_embed(torch.zeros(batch, dtype=torch.long, device=device))
        
        # Organ embedding (multi-hot -> embed)
        if organ_mask is not None:
            organ_embed = self.organ_proj(organ_mask)  # [B, semantic_embed_dim]
        else:
            organ_embed = torch.zeros(batch, bp_embed.shape[-1], device=device)
        
        # Fuse semantic features.
        semantic_feat = torch.cat([bp_embed, organ_embed], dim=-1)  # [B, semantic_embed_dim * 2]
        semantic_feat = self.semantic_mlp(semantic_feat)  # [B, hidden_ch]
        
        # ===================== FiLM Modulation =====================
        # Modulate spatial features: feat = scale * feat + shift.
        scale = self.semantic_scale(semantic_feat).unsqueeze(-1).unsqueeze(-1)  # [B, hidden_ch, 1, 1]
        shift = self.semantic_shift(semantic_feat).unsqueeze(-1).unsqueeze(-1)  # [B, hidden_ch, 1, 1]
        
        feat = scale * feat + shift
        
        # ===================== Predict Noise Scale =====================
        scale_01 = self.head(feat)  # [B, 1, H, W], range [0, 1]
        
        # Map to [min_scale, max_scale].
        noise_scale = self.min_scale + (self.max_scale - self.min_scale) * scale_01
        
        return noise_scale

def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

class Diffusion():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    # =========================================================================
    # Continuous Index Methods (for Semantic Time Warp)
    # =========================================================================

    def interp_schedule(self, table: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation of a schedule table at continuous indices.

        This enables using warped (non-integer) time indices to look up
        schedule values, which is essential for per-sample adaptive schedules.

        Args:
            table: [T] 1D schedule tensor (e.g., std_fwd, mu_x0, std_sb)
            s: [B] 1D continuous indices in [0, T-1]. MUST be 1D.

        Returns:
            [B] interpolated values

        Note:
            Caller is responsible for:
            - Ensuring s is 1D [B] (use s.view(-1) if needed)
            - Using unsqueeze_xdim() after for spatial broadcasting
        """
        assert table.dim() == 1, f"table must be 1D, got {table.shape}"
        assert s.dim() == 1, f"s must be 1D [B], got {s.shape}"

        T = len(table)
        s = s.float().clamp(0, T - 1 - 1e-6)

        s_low = s.long()
        s_high = (s_low + 1).clamp(max=T - 1)
        frac = s - s_low.float()  # [B], in [0, 1)

        val_low = table[s_low]    # [B]
        val_high = table[s_high]  # [B]

        return val_low + frac * (val_high - val_low)  # [B]

    def get_noise_level(self, s: torch.Tensor) -> torch.Tensor:
        """
        Get noise level (std_fwd) at continuous time index s.

        This is the value the network should receive as time conditioning.
        Using std_fwd directly gives the network a physically meaningful
        signal about the current noise level.

        Args:
            s: [B] continuous time index in [0, T-1]

        Returns:
            [B] noise levels (std_fwd values)
        """
        return self.interp_schedule(self.std_fwd, s)

    def get_std_fwd_warped(self, s: torch.Tensor, xdim=None) -> torch.Tensor:
        """
        Get std_fwd at continuous (warped) time index.

        Args:
            s: [B] continuous time index
            xdim: If provided, unsqueeze result for spatial broadcasting

        Returns:
            [B] or [B, 1, 1, 1] std_fwd values
        """
        std_fwd = self.interp_schedule(self.std_fwd, s)
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample_warped(self, s: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor,
                        ot_ode: bool = False, noise_scale: torch.Tensor = None) -> torch.Tensor:
        """
        Forward noising with continuous (warped) time index.

        q(x_s | x_0, x_1) using interpolated schedule values at s.

        Args:
            s: [B] continuous time index in [0, T-1]
            x0: [B, C, H, W] clean target image
            x1: [B, C, H, W] corrupted source image
            ot_ode: If True, skip stochastic noise (deterministic bridge)
            noise_scale: Optional spatial noise scale [B, 1, H, W]
                        from PETNoiseModulator for PET-guided noise injection

        Returns:
            x_s: [B, C, H, W] noised image at warped time s
        """
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        # Interpolate schedule values at continuous s
        mu_x0 = unsqueeze_xdim(self.interp_schedule(self.mu_x0, s), xdim)
        mu_x1 = unsqueeze_xdim(self.interp_schedule(self.mu_x1, s), xdim)
        std_sb = unsqueeze_xdim(self.interp_schedule(self.std_sb, s), xdim)

        x_s = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            noise = torch.randn_like(x_s)
            if noise_scale is not None:
                # PET-guided spatially adaptive noise.
                x_s = x_s + std_sb * noise_scale * noise
            else:
                x_s = x_s + std_sb * noise

        return x_s.detach()

    def p_posterior_warped(self, s_prev: torch.Tensor, s_curr: torch.Tensor,
                           x_curr: torch.Tensor, x0: torch.Tensor,
                           ot_ode: bool = False, noise_scale: torch.Tensor = None) -> torch.Tensor:
        """
        Posterior sampling with continuous (warped) time indices.

        p(x_{s_prev} | x_{s_curr}, x_0)

        Args:
            s_prev: [B] previous (target) continuous time index
            s_curr: [B] current continuous time index
            x_curr: [B, C, H, W] current state
            x0: [B, C, H, W] predicted clean image
            ot_ode: If True, deterministic update
            noise_scale: optional spatial noise scaling factor [B, 1, H, W]
                        from PETNoiseModulator, used to modulate the reverse noise

        Returns:
            x_prev: [B, C, H, W] state at previous time
        """
        batch, *xdim = x_curr.shape

        # Get std_fwd at both times
        std_curr = self.interp_schedule(self.std_fwd, s_curr)
        std_prev = self.interp_schedule(self.std_fwd, s_prev)

        # Compute std_delta: sqrt(std_curr^2 - std_prev^2)
        std_delta_sq = (std_curr ** 2 - std_prev ** 2).clamp(min=1e-8)
        std_delta = std_delta_sq.sqrt()

        # Compute Gaussian product coefficients
        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_prev, std_delta)

        # Unsqueeze for spatial dimensions
        mu_x0 = unsqueeze_xdim(mu_x0, xdim)
        mu_xn = unsqueeze_xdim(mu_xn, xdim)
        var = unsqueeze_xdim(var, xdim)

        x_prev = mu_x0 * x0 + mu_xn * x_curr

        if not ot_ode:
            # Only add noise if not at s=0
            # Use a soft mask to handle per-sample s_prev values
            noise_mask = (s_prev > 1e-3).float().view(-1, *([1] * len(xdim)))
            noise = torch.randn_like(x_prev)
            if noise_scale is not None:
                # Apply spatial noise modulation (matches forward process)
                x_prev = x_prev + noise_mask * var.sqrt() * noise_scale * noise
            else:
                x_prev = x_prev + noise_mask * var.sqrt() * noise

        return x_prev

    def q_sample(self, step, x0, x1, ot_ode=False, noise_scale=None):
        """ 
        Sample q(x_t | x_0, x_1), i.e. eq 11 
        
        Args:
            step: Timestep [B] or scalar
            x0: Target image (T2) [B, C, H, W]
            x1: Source image (T1) [B, C, H, W]
            ot_ode: Whether to use OT-ODE (deterministic, no random noise)
            noise_scale: Optional spatial noise scale [B, 1, H, W]
                        from PETNoiseModulator for PET-guided noise injection
        
        Returns:
            xt: Noised image [B, C, H, W]
        """
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            noise = torch.randn_like(xt)
            if noise_scale is not None:
                # PET-guided spatially adaptive noise.
                xt = xt + std_sb * noise_scale * noise
            else:
                xt = xt + std_sb * noise
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False, noise_scale=None):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4
        
        Args:
            nprev: previous time step
            n: current time step
            x_n: current noisy image
            x0: predicted clean image
            ot_ode: if True, deterministic (no noise added)
            noise_scale: optional spatial noise scaling factor [B, 1, H, W]
                        from PETNoiseModulator, used to modulate the reverse noise
        """

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            noise = torch.randn_like(xt_prev)
            if noise_scale is not None:
                # Apply spatial noise modulation (matches forward process)
                xt_prev = xt_prev + var.sqrt() * noise_scale * noise
            else:
                xt_prev = xt_prev + var.sqrt() * noise

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, integrator="euler",
                      log_steps=None, verbose=True, calibrate=False, calib_mode='affine', ref=None,
                      noise_scale_fn=None):
        """
        DDPM sampling with optional Heun integration for ODE mode.

        Args:
            steps: list of time steps
            pred_x0_fn: function to predict x0 given (xt, step)
            x1: source image (T1)
            mask: optional inpainting mask
            ot_ode: if True, deterministic sampling (no noise)
            integrator: "euler" (default) or "heun" - only affects ot_ode=True mode
            log_steps: steps to log
            verbose: show progress bar
            calibrate: apply intensity calibration
            calib_mode: calibration mode ('affine' or 'pwl')
            ref: reference for calibration
            noise_scale_fn: optional function to compute noise_scale given predicted x0
                           signature: noise_scale_fn(pred_x0) -> [B, 1, H, W]
                           Used for PET-guided noise modulation in reverse process
        """
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            if ot_ode and integrator == "heun":
                # Heun's method (2-stage RK2) for deterministic ODE
                # 
                # Standard Heun in diffusion models:
                # 1. Predict x0 at current point (xt, step)
                # 2. Do Euler step to get intermediate point
                # 3. Predict x0 at intermediate point (x_euler, prev_step)
                # 4. Use average of two x0 predictions for final update
                #
                # This is more accurate than single-step Euler because it
                # corrects for curvature in the ODE trajectory.
                
                # Stage 1: Euler prediction at current point
                pred_x0_1 = pred_x0_fn(xt, step)
                x_euler = self.p_posterior(prev_step, step, xt, pred_x0_1, ot_ode=True)

                # Stage 2: Evaluate at the Euler-predicted point
                pred_x0_2 = pred_x0_fn(x_euler, prev_step)
                
                # Heun correction: use average of two x0 predictions
                # This is the standard approach for diffusion model Heun sampling
                pred_x0_avg = 0.5 * (pred_x0_1 + pred_x0_2)
                xt = self.p_posterior(prev_step, step, xt, pred_x0_avg, ot_ode=True)

                # Store averaged prediction for trajectory logging
                pred_x0 = pred_x0_avg
            else:
                # Original Euler method (for both stochastic and deterministic ODE)
                pred_x0 = pred_x0_fn(xt, step)
                
                # Compute noise_scale if provided (for PET-guided noise modulation)
                noise_scale = None
                if noise_scale_fn is not None and not ot_ode:
                    noise_scale = noise_scale_fn(pred_x0)
                
                xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode, noise_scale=noise_scale)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        xs_stacked = stack_bwd_traj(xs)
        pred_x0s_stacked = stack_bwd_traj(pred_x0s)

        # Apply calibration if requested
        if calibrate and ref is not None:
            # Get the final sample (first in the stacked trajectory)
            final_x0 = xs_stacked[:, 0, ...]

            if calib_mode == 'affine':
                calibrated_x0 = match_mean_std(final_x0, ref)
            elif calib_mode == 'pwl':
                calibrated_x0 = pwl_quantile_match(final_x0, ref)
            else:
                raise ValueError(f"Unknown calibration mode: {calib_mode}. Use 'affine' or 'pwl'")

            # Replace the final sample in the trajectory
            xs_stacked = xs_stacked.clone()
            xs_stacked[:, 0, ...] = calibrated_x0

        return xs_stacked, pred_x0s_stacked
