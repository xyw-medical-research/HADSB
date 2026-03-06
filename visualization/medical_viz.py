"""
Visualization utilities for medical modality conversion training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def denormalize_medical_image(img_tensor):
    """Convert from [-1, 1] to [0, 1] for visualization"""
    return (img_tensor + 1.0) / 2.0


def create_unified_medical_visualization(t1, t2_true, t2_pred, pet, save_path, 
                                       mode='sample', step_info=None, show_diff=True, 
                                       max_samples=4, idx=0):
    """
    Unified visualization function for both training and sampling
    
    Args:
        t1: T1 input images [B, 1, H, W] or single [1, H, W]
        t2_true: Ground truth T2 [B, 1, H, W] or single [1, H, W]
        t2_pred: Predicted T2 [B, 1, H, W] or single [1, H, W]  
        pet: PET conditioning [B, 1, H, W] or single [1, H, W]
        save_path: Path to save visualization
        mode: 'training' or 'sample' - affects layout and info display
        step_info: Dict with training step information (for training mode)
        show_diff: Whether to include difference map
        max_samples: Maximum number of samples to visualize
        idx: Sample index (for single sample mode)
    """
    # Handle single sample vs batch
    if t1.dim() == 3:  # Single sample [1, H, W]
        t1 = t1.unsqueeze(0)
        t2_true = t2_true.unsqueeze(0)
        t2_pred = t2_pred.unsqueeze(0)
        pet = pet.unsqueeze(0)
    
    batch_size = min(max_samples, t1.shape[0])
    n_cols = 5 if show_diff else 4
    
    # Determine figure layout based on mode
    if mode == 'sample' and batch_size == 1:
        # Single row for sampling
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        axes = axes.reshape(1, -1)
    else:
        # Multiple rows for training or batch sampling
        fig, axes = plt.subplots(batch_size, n_cols, figsize=(4 * n_cols, 4 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
    
    # Process each sample in batch
    for i in range(batch_size):
        # Convert to numpy and denormalize
        t1_np = denormalize_medical_image(t1[i, 0]).cpu().numpy()
        t2_true_np = denormalize_medical_image(t2_true[i, 0]).cpu().numpy()
        t2_pred_np = denormalize_medical_image(t2_pred[i, 0]).cpu().numpy()
        pet_np = denormalize_medical_image(pet[i, 0]).cpu().numpy()
        
        # Plot core images
        sample_label = f"(Sample {i+1})" if batch_size > 1 else ""
        
        axes[i, 0].imshow(t1_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'T1 Input\n{sample_label}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pet_np, cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title(f'PET Condition\n{sample_label}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(t2_pred_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'T2 Predicted\n{sample_label}')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(t2_true_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f'T2 Ground Truth\n{sample_label}')
        axes[i, 3].axis('off')
        
        # Optional difference map
        if show_diff:
            diff_np = np.abs(t2_true_np - t2_pred_np)
            im = axes[i, 4].imshow(diff_np, cmap='jet', vmin=0, vmax=0.5)
            axes[i, 4].set_title(f'|Pred - True|\n{sample_label}')
            axes[i, 4].axis('off')
            
            # Add colorbar for difference map (first sample only)
            if i == 0:
                plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)
    
    # Add mode-specific title information
    if mode == 'training' and step_info:
        info_text = f"[VALIDATION DATA] Iteration: {step_info['iteration']:06d} | Loss: {step_info['loss']:.4f}"
        if 'lr' in step_info:
            info_text += f" | LR: {step_info['lr']:.2e}"
        if 'pred_mean' in step_info:
            info_text += f" | Pred μ: {step_info['pred_mean']:.3f}"
        if 'pred_std' in step_info:
            info_text += f" | Pred σ: {step_info['pred_std']:.3f}"
        
        fig.suptitle(info_text, fontsize=14, y=0.98)
        plt.subplots_adjust(top=0.92)
    elif mode == 'sample':
        if batch_size == 1:
            fig.suptitle(f'Medical Sample {idx:04d}', fontsize=14, y=0.95)
        else:
            fig.suptitle(f'Medical Samples (Batch {idx})', fontsize=14, y=0.95)
    
    # Save with appropriate filename
    if mode == 'sample':
        if isinstance(save_path, Path) and save_path.is_dir():
            final_path = save_path / f"sample_{idx:04d}.png"
        elif isinstance(save_path, Path) and not save_path.suffix:
            # Path without extension, assume it's meant to be a directory
            save_path.mkdir(parents=True, exist_ok=True)
            final_path = save_path / f"sample_{idx:04d}.png"
        else:
            final_path = save_path
    else:
        final_path = save_path
    
    plt.tight_layout()
    plt.savefig(final_path, dpi=150, bbox_inches='tight')
    plt.close()


# Backward compatibility functions
def create_medical_training_visualization(t1, t2_true, t2_pred, pet, step_info, save_path):
    """Backward compatible training visualization wrapper"""
    create_unified_medical_visualization(
        t1, t2_true, t2_pred, pet, save_path, 
        mode='training', step_info=step_info, show_diff=True
    )


def create_medical_sampling_visualization(t1, t2_true, t2_pred, pet, save_path, idx=0):
    """Backward compatible sampling visualization wrapper"""
    create_unified_medical_visualization(
        t1, t2_true, t2_pred, pet, save_path,
        mode='sample', show_diff=False, idx=idx
    )


def create_diffusion_process_visualization(x1, xt_sequence, x0_pred_sequence, save_path):
    """
    Visualize the diffusion process during sampling

    Args:
        x1: Starting T1 image [1, 1, H, W]
        xt_sequence: List of intermediate diffusion states
        x0_pred_sequence: List of predicted clean images
        save_path: Path to save visualization
    """
    num_steps = len(xt_sequence)

    fig, axes = plt.subplots(3, num_steps, figsize=(2 * num_steps, 6))
    if num_steps == 1:
        axes = axes.reshape(-1, 1)

    x1_np = denormalize_medical_image(x1[0, 0]).cpu().numpy()

    for i, (xt, x0_pred) in enumerate(zip(xt_sequence, x0_pred_sequence)):
        xt_np = denormalize_medical_image(xt[0, 0]).cpu().numpy()
        x0_pred_np = denormalize_medical_image(x0_pred[0, 0]).cpu().numpy()

        # Show T1 reference in first row
        axes[0, i].imshow(x1_np, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'T1 (Ref)\nStep {i}')
        axes[0, i].axis('off')

        # Show current diffusion state
        axes[1, i].imshow(xt_np, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'xt\nStep {i}')
        axes[1, i].axis('off')

        # Show predicted clean image
        axes[2, i].imshow(x0_pred_np, cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Pred T2\nStep {i}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_sampling_trajectory_visualization(x1, pet, xs_trajectory, pred_x0_trajectory,
                                           step_indices, nfe_info, save_path,
                                           show_pet=True, max_steps=10):
    """
    Enhanced visualization for sampling trajectory with step information

    Args:
        x1: Source T1 image [B, 1, H, W]
        pet: PET conditioning [B, 1, H, W] (can be None)
        xs_trajectory: Intermediate states [B, log_count, 1, H, W]
        pred_x0_trajectory: Predicted clean images [B, log_count, 1, H, W]
        step_indices: List of step indices corresponding to logged steps
        nfe_info: Dict with NFE information (total_steps, actual_nfe, etc.)
        save_path: Path to save visualization
        show_pet: Whether to include PET conditioning in visualization
        max_steps: Maximum number of steps to show (will sample evenly if more)
    """
    batch_size = x1.shape[0]
    total_logged_steps = xs_trajectory.shape[1]

    # Sample steps if too many
    if total_logged_steps > max_steps:
        step_idxs = np.linspace(0, total_logged_steps-1, max_steps, dtype=int)
        xs_trajectory = xs_trajectory[:, step_idxs]
        pred_x0_trajectory = pred_x0_trajectory[:, step_idxs]
        step_indices = [step_indices[i] for i in step_idxs]
        display_steps = max_steps
    else:
        display_steps = total_logged_steps

    # Determine layout
    n_rows = 4 if show_pet and pet is not None else 3
    row_labels = ['T1 Source', 'PET Cond', 'Sampling Traj', 'T2 Prediction'] if show_pet and pet is not None else ['T1 Source', 'Sampling Traj', 'T2 Prediction']

    # Create figure
    fig, axes = plt.subplots(n_rows, display_steps, figsize=(2.5 * display_steps, 2.5 * n_rows))
    if display_steps == 1:
        axes = axes.reshape(-1, 1)

    # Process first sample in batch
    sample_idx = 0
    x1_np = denormalize_medical_image(x1[sample_idx, 0]).cpu().numpy()
    if show_pet and pet is not None:
        pet_np = denormalize_medical_image(pet[sample_idx, 0]).cpu().numpy()

    for step_idx in range(display_steps):
        # Extract trajectory data
        xt_np = denormalize_medical_image(xs_trajectory[sample_idx, step_idx, 0]).cpu().numpy()
        pred_x0_np = denormalize_medical_image(pred_x0_trajectory[sample_idx, step_idx, 0]).cpu().numpy()

        # Current step info
        current_step = step_indices[step_idx] if step_idx < len(step_indices) else step_idx
        progress = (step_idx + 1) / display_steps

        row = 0
        # Row 1: T1 Source (reference)
        axes[row, step_idx].imshow(x1_np, cmap='gray', vmin=0, vmax=1)
        axes[row, step_idx].set_title(f'T1 Source\nStep {current_step}')
        axes[row, step_idx].axis('off')

        # Row 2: PET Conditioning (if available)
        if show_pet and pet is not None:
            row += 1
            axes[row, step_idx].imshow(pet_np, cmap='hot', vmin=0, vmax=1, alpha=0.8)
            axes[row, step_idx].set_title(f'PET Cond\nStep {current_step}')
            axes[row, step_idx].axis('off')

        # Row 3: Sampling trajectory (xt)
        row += 1
        axes[row, step_idx].imshow(xt_np, cmap='gray', vmin=0, vmax=1)
        axes[row, step_idx].set_title(f'xt (Traj)\nProgress: {progress:.1%}')
        axes[row, step_idx].axis('off')

        # Add progress indicator border
        border_color = plt.cm.viridis(progress)
        for spine in axes[row, step_idx].spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

        # Row 4: Predicted T2
        row += 1
        axes[row, step_idx].imshow(pred_x0_np, cmap='gray', vmin=0, vmax=1)
        axes[row, step_idx].set_title(f'Pred T2\nStep {current_step}')
        axes[row, step_idx].axis('off')

    # Add overall title with NFE information
    title_text = f"Sampling Trajectory - NFE: {nfe_info.get('actual_nfe', 'N/A')}"
    if 'total_steps' in nfe_info:
        title_text += f" (Total Steps: {nfe_info['total_steps']})"
    if 'solver' in nfe_info:
        title_text += f" | Solver: {nfe_info['solver']}"

    # Add early stopping info if applicable
    if nfe_info.get('early_stopped', False):
        original_nfe = nfe_info.get('original_nfe', nfe_info.get('actual_nfe', 'N/A'))
        title_text += f" | Early Stop: {original_nfe} → {nfe_info.get('actual_nfe', 'N/A')}"

    fig.suptitle(title_text, fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_trajectory_comparison_plot(x1, final_pred, gt_t2, trajectory_data, save_path):
    """
    Create a comparison plot showing the final result and trajectory progression

    Args:
        x1: Source T1 [B, 1, H, W]
        final_pred: Final predicted T2 [B, 1, H, W]
        gt_t2: Ground truth T2 (if available) [B, 1, H, W]
        trajectory_data: Dict with trajectory info and metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert to numpy for visualization
    sample_idx = 0
    x1_np = denormalize_medical_image(x1[sample_idx, 0]).cpu().numpy()
    pred_np = denormalize_medical_image(final_pred[sample_idx, 0]).cpu().numpy()

    # Top row: Input, Prediction, Ground Truth (if available)
    axes[0, 0].imshow(x1_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('T1 Source')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Final T2 Prediction')
    axes[0, 1].axis('off')

    if gt_t2 is not None:
        gt_np = denormalize_medical_image(gt_t2[sample_idx, 0]).cpu().numpy()
        axes[0, 2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('Ground Truth T2')
        axes[0, 2].axis('off')

        # Difference map
        diff_np = np.abs(pred_np - gt_np)
        im = axes[1, 2].imshow(diff_np, cmap='jet', vmin=0, vmax=0.5)
        axes[1, 2].set_title('|Pred - GT|')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        axes[0, 2].axis('off')
        axes[1, 2].axis('off')

    # Bottom row: Trajectory progression plots
    if 'step_metrics' in trajectory_data:
        steps = trajectory_data['step_indices']
        step_metrics = trajectory_data['step_metrics']

        # Ensure all arrays have the same length
        min_len = min(len(steps),
                     len(step_metrics.get('similarities', [])),
                     len(step_metrics.get('variances', [])))

        if min_len > 0:
            steps = steps[:min_len]

            # Plot image similarity progression (if available)
            if 'similarities' in step_metrics and len(step_metrics['similarities']) > 0:
                similarities = step_metrics['similarities'][:min_len]
                axes[1, 0].plot(steps, similarities, 'b-o', linewidth=2, markersize=4)
                axes[1, 0].set_title('Image Similarity vs Steps')
                axes[1, 0].set_xlabel('Diffusion Step')
                axes[1, 0].set_ylabel('Similarity to Final')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No similarity data', transform=axes[1, 0].transAxes,
                              ha='center', va='center')
                axes[1, 0].set_title('Image Similarity vs Steps')

            # Plot prediction confidence/variance
            if 'variances' in step_metrics and len(step_metrics['variances']) > 0:
                variances = step_metrics['variances'][:min_len]
                axes[1, 1].plot(steps, variances, 'r-o', linewidth=2, markersize=4)
                axes[1, 1].set_title('Prediction Variance vs Steps')
                axes[1, 1].set_xlabel('Diffusion Step')
                axes[1, 1].set_ylabel('Pixel Variance')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_yscale('log')
            else:
                axes[1, 1].text(0.5, 0.5, 'No variance data', transform=axes[1, 1].transAxes,
                              ha='center', va='center')
                axes[1, 1].set_title('Prediction Variance vs Steps')
        else:
            # No valid data to plot
            for ax in [axes[1, 0], axes[1, 1]]:
                ax.text(0.5, 0.5, 'No trajectory data', transform=ax.transAxes,
                       ha='center', va='center')
            axes[1, 0].set_title('Image Similarity vs Steps')
            axes[1, 1].set_title('Prediction Variance vs Steps')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_metrics(metrics_history, save_path):
    """
    Plot training metrics over time
    
    Args:
        metrics_history: Dict with lists of metrics over iterations
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    if 'train_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['iterations'], metrics_history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['val_iterations'], metrics_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale('log')
    
    # Learning rate plot
    if 'learning_rate' in metrics_history:
        axes[0, 1].plot(metrics_history['iterations'], metrics_history['learning_rate'], 'g-')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')
    
    # Prediction statistics
    if 'pred_mean' in metrics_history:
        axes[1, 0].plot(metrics_history['iterations'], metrics_history['pred_mean'], 'c-', label='Mean')
        if 'pred_std' in metrics_history:
            axes[1, 0].plot(metrics_history['iterations'], metrics_history['pred_std'], 'm-', label='Std')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Prediction Statistics')
        axes[1, 0].set_title('Model Output Statistics')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Gradient norm (if available)
    if 'grad_norm' in metrics_history and len(metrics_history['grad_norm']) > 0:
        # Ensure dimensions match before plotting
        grad_norms = metrics_history['grad_norm']
        iterations = metrics_history['iterations']
        min_len = min(len(iterations), len(grad_norms))
        if min_len > 0:
            axes[1, 1].plot(iterations[:min_len], grad_norms[:min_len], 'orange')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Magnitude')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_data_distribution_plot(data_dict, save_path):
    """
    Visualize data distributions for debugging
    
    Args:
        data_dict: Dict with 't1', 't2', 'pet' tensors
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    modalities = ['t1', 't2', 'pet']
    colors = ['blue', 'green', 'red']
    
    for i, (mod, color) in enumerate(zip(modalities, colors)):
        if mod in data_dict:
            data = data_dict[mod].cpu().numpy().flatten()
            
            # Histogram
            axes[0, i].hist(data, bins=50, color=color, alpha=0.7, density=True)
            axes[0, i].set_title(f'{mod.upper()} Value Distribution')
            axes[0, i].set_xlabel('Pixel Value')
            axes[0, i].set_ylabel('Density')
            axes[0, i].grid(True)
            
            # Box plot
            axes[1, i].boxplot(data, vert=True)
            axes[1, i].set_title(f'{mod.upper()} Statistics')
            axes[1, i].set_ylabel('Pixel Value')
            axes[1, i].grid(True)
            
            # Add statistics text
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
            axes[0, i].text(0.02, 0.98, stats_text, transform=axes[0, i].transAxes, 
                          verticalalignment='top', fontsize=10, 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class MedicalTrainingVisualizer:
    """Class to manage visualizations during training"""
    
    def __init__(self, save_dir, log_freq=100, viz_freq=1000):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_freq = log_freq
        self.viz_freq = viz_freq
        
        # Create subdirectories
        (self.save_dir / 'training_images').mkdir(exist_ok=True)
        (self.save_dir / 'metrics_plots').mkdir(exist_ok=True)
        (self.save_dir / 'distributions').mkdir(exist_ok=True)
        
        # Metrics tracking
        self.metrics_history = {
            'iterations': [],
            'train_loss': [],
            'val_iterations': [],
            'val_loss': [],
            'learning_rate': [],
            'pred_mean': [],
            'pred_std': [],
            'grad_norm': []
        }
    
    def log_iteration(self, iteration, loss, metrics, lr=None, grad_norm=None):
        """Log metrics for an iteration"""
        self.metrics_history['iterations'].append(iteration)
        self.metrics_history['train_loss'].append(loss)
        
        if 'pred_mean' in metrics:
            self.metrics_history['pred_mean'].append(metrics['pred_mean'])
        if 'pred_std' in metrics:
            self.metrics_history['pred_std'].append(metrics['pred_std'])
            
        if lr is not None:
            self.metrics_history['learning_rate'].append(lr)
        if grad_norm is not None:
            self.metrics_history['grad_norm'].append(grad_norm)
    
    def log_validation(self, iteration, val_loss):
        """Log validation metrics"""
        self.metrics_history['val_iterations'].append(iteration)
        self.metrics_history['val_loss'].append(val_loss)
    
    def visualize_training_batch(self, iteration, t1, t2_true, t2_pred, pet, loss, metrics, lr=None):
        """Create and save training visualization"""
        if iteration % self.viz_freq == 0:
            step_info = {
                'iteration': iteration,
                'loss': loss,
                'lr': lr,
                'pred_mean': metrics.get('pred_mean', 0),
                'pred_std': metrics.get('pred_std', 0)
            }
            
            save_path = self.save_dir / 'training_images' / f'train_viz_{iteration:06d}.png'
            create_unified_medical_visualization(
                t1, t2_true, t2_pred, pet, save_path,
                mode='training', step_info=step_info, show_diff=True
            )
    
    def update_metrics_plot(self, iteration):
        """Update and save metrics plot"""
        if iteration % (self.viz_freq * 2) == 0 and len(self.metrics_history['iterations']) > 1:
            save_path = self.save_dir / 'metrics_plots' / f'metrics_{iteration:06d}.png'
            plot_training_metrics(self.metrics_history, save_path)
    
    def visualize_data_batch(self, iteration, data_dict):
        """Visualize data distributions"""
        if iteration % (self.viz_freq * 5) == 0:  # Less frequent
            save_path = self.save_dir / 'distributions' / f'data_dist_{iteration:06d}.png'
            create_data_distribution_plot(data_dict, save_path)