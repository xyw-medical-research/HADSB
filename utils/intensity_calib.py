import numpy as np
import torch
from typing import Union, Tuple, Optional


def match_mean_std(x: Union[np.ndarray, torch.Tensor],
                   ref: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Affine intensity correction by matching mean and standard deviation.

    Args:
        x: Input image/tensor to be calibrated (generated image)
        ref: Reference image/tensor or statistics to match

    Returns:
        Calibrated image with matched mean and std
    """
    is_numpy = isinstance(x, np.ndarray)

    if is_numpy:
        x_tensor = torch.from_numpy(x).float()
        ref_tensor = torch.from_numpy(ref).float() if isinstance(ref, np.ndarray) else ref.float()
    else:
        x_tensor = x.float()
        ref_tensor = ref.float()

    # Ensure both tensors are on the same device
    device = x_tensor.device
    ref_tensor = ref_tensor.to(device)

    # Compute statistics along spatial dimensions
    # For batched tensors (N,C,H,W) or (N,C,D,H,W), compute per sample
    if x_tensor.ndim >= 4:
        # Batched tensor
        dims = list(range(2, x_tensor.ndim))  # All spatial dims
        x_mean = x_tensor.mean(dim=dims, keepdim=True)
        x_std = x_tensor.std(dim=dims, keepdim=True)
        ref_mean = ref_tensor.mean(dim=dims, keepdim=True)
        ref_std = ref_tensor.std(dim=dims, keepdim=True)
    else:
        # 2D/3D image
        x_mean = x_tensor.mean()
        x_std = x_tensor.std()
        ref_mean = ref_tensor.mean()
        ref_std = ref_tensor.std()

    # Avoid division by zero
    x_std = torch.where(x_std < 1e-8, torch.ones_like(x_std), x_std)

    # Apply affine transformation: y = (x - mean_x) / std_x * std_ref + mean_ref
    calibrated = (x_tensor - x_mean) / x_std * ref_std + ref_mean

    if is_numpy:
        return calibrated.numpy()
    return calibrated


def pwl_quantile_match(x: Union[np.ndarray, torch.Tensor],
                        ref: Union[np.ndarray, torch.Tensor],
                        qs: Tuple[float, ...] = (0.01, 0.1, 0.5, 0.9, 0.99)) -> Union[np.ndarray, torch.Tensor]:
    """
    Piecewise linear intensity correction using quantile matching.

    Args:
        x: Input image/tensor to be calibrated (generated image)
        ref: Reference image/tensor to match
        qs: Quantiles to use for matching (default: 1%, 10%, 50%, 90%, 99%)

    Returns:
        Calibrated image with matched quantiles via piecewise linear mapping
    """
    is_numpy = isinstance(x, np.ndarray)

    if is_numpy:
        x_tensor = torch.from_numpy(x).float()
        ref_tensor = torch.from_numpy(ref).float() if isinstance(ref, np.ndarray) else ref.float()
    else:
        x_tensor = x.float()
        ref_tensor = ref.float()

    device = x_tensor.device
    # Ensure ref_tensor is on the same device as x_tensor
    ref_tensor = ref_tensor.to(device)
    quantiles = torch.tensor(qs, device=device, dtype=torch.float32)

    # Handle batched tensors
    if x_tensor.ndim >= 4:
        # Process each sample in the batch independently
        batch_size = x_tensor.shape[0]
        calibrated = torch.zeros_like(x_tensor)

        for b in range(batch_size):
            x_sample = x_tensor[b]
            ref_sample = ref_tensor[b] if ref_tensor.shape[0] > 1 else ref_tensor[0] if ref_tensor.ndim >= 4 else ref_tensor

            # Flatten for quantile computation
            x_flat = x_sample.flatten()
            ref_flat = ref_sample.flatten()

            # Compute quantiles
            x_quantiles = torch.quantile(x_flat, quantiles)
            ref_quantiles = torch.quantile(ref_flat, quantiles)

            # Add boundary points (0 and 1)
            x_quantiles = torch.cat([x_flat.min().unsqueeze(0), x_quantiles, x_flat.max().unsqueeze(0)])
            ref_quantiles = torch.cat([ref_flat.min().unsqueeze(0), ref_quantiles, ref_flat.max().unsqueeze(0)])

            # Apply piecewise linear mapping
            calibrated_sample = torch.zeros_like(x_sample)
            for i in range(len(x_quantiles) - 1):
                mask = (x_sample >= x_quantiles[i]) & (x_sample <= x_quantiles[i + 1])

                # Linear interpolation within this segment
                if x_quantiles[i + 1] - x_quantiles[i] > 1e-8:
                    slope = (ref_quantiles[i + 1] - ref_quantiles[i]) / (x_quantiles[i + 1] - x_quantiles[i])
                    calibrated_sample[mask] = ref_quantiles[i] + slope * (x_sample[mask] - x_quantiles[i])
                else:
                    calibrated_sample[mask] = ref_quantiles[i]

            calibrated[b] = calibrated_sample
    else:
        # 2D/3D image
        x_flat = x_tensor.flatten()
        ref_flat = ref_tensor.flatten()

        # Compute quantiles
        x_quantiles = torch.quantile(x_flat, quantiles)
        ref_quantiles = torch.quantile(ref_flat, quantiles)

        # Add boundary points
        x_quantiles = torch.cat([x_flat.min().unsqueeze(0), x_quantiles, x_flat.max().unsqueeze(0)])
        ref_quantiles = torch.cat([ref_flat.min().unsqueeze(0), ref_quantiles, ref_flat.max().unsqueeze(0)])

        # Apply piecewise linear mapping
        calibrated = torch.zeros_like(x_tensor)
        for i in range(len(x_quantiles) - 1):
            mask = (x_tensor >= x_quantiles[i]) & (x_tensor <= x_quantiles[i + 1])

            if x_quantiles[i + 1] - x_quantiles[i] > 1e-8:
                slope = (ref_quantiles[i + 1] - ref_quantiles[i]) / (x_quantiles[i + 1] - x_quantiles[i])
                calibrated[mask] = ref_quantiles[i] + slope * (x_tensor[mask] - x_quantiles[i])
            else:
                calibrated[mask] = ref_quantiles[i]

    if is_numpy:
        return calibrated.cpu().numpy()
    return calibrated