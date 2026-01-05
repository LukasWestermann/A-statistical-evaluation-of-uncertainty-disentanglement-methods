import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.results_save import save_plot

# Try to import IPython display for better notebook support
try:
    from IPython.display import display
    _ipython_available = True
except ImportError:
    _ipython_available = False


def plot_toy_data(x_train, y_train, x_grid, y_clean, title="Toy Regression Data", save_plot_file=True):
    """Plot the training data and clean function"""
    fig = plt.figure(figsize=(12, 6))
    
    # Plot training data points
    plt.scatter(x_train, y_train, alpha=0.6, s=20, label="Training data", color='blue')
    
    # Plot clean function
    plt.plot(x_grid, y_clean, 'r--', linewidth=2)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot_file:
        save_plot(fig, title, subfolder='toy_data')
    
    plt.show()
    plt.close(fig)


def plot_data_with_ood_regions(x_train, y_train, x_grid, y_grid_clean, train_range=None, train_ranges=None, ood_ranges=None, 
                                title="Data Setup with OOD Regions", func_type='', save_plot_file=True):
    """
    Plot training data, evaluation grid, and highlight OOD regions.
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Full evaluation grid (ID + OOD)
        y_grid_clean: Clean function values at grid points
        train_range: Tuple (min, max) for single training range (for backward compatibility)
        train_ranges: List of tuples [(min1, max1), (min2, max2), ...] for multiple training ranges
                     If provided, overrides train_range
        ood_ranges: List of tuples [(min1, max1), (min2, max2), ...] for OOD regions
        title: Plot title
        func_type: Function type identifier (e.g., 'linear', 'sin')
        save_plot_file: Whether to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Flatten arrays for plotting
    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
    x_grid_flat = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    y_grid_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean
    
    # Handle train_ranges: if provided, use it; otherwise use train_range as single range
    if train_ranges is None:
        if train_range is not None:
            train_ranges = [train_range]
        else:
            train_ranges = []
    
    # Create OOD mask: everything NOT in training ranges is OOD
    ood_mask = np.ones(len(x_grid_flat), dtype=bool)  # Start with all True (OOD)
    
    # Mark training ranges as ID (False in ood_mask)
    for train_r in train_ranges:
        train_start, train_end = train_r
        train_mask = (x_grid_flat >= train_start) & (x_grid_flat <= train_end)
        ood_mask[train_mask] = False  # Training regions are ID, not OOD
    
    # If explicit ood_ranges provided, ensure they are marked as OOD
    if ood_ranges:
        for ood_range in ood_ranges:
            ood_start, ood_end = ood_range
            ood_mask |= (x_grid_flat >= ood_start) & (x_grid_flat <= ood_end)
    
    id_mask = ~ood_mask
    
    # Plot clean function - ID region
    ax.plot(x_grid_flat[id_mask], y_grid_clean_flat[id_mask], 'r-', 
            linewidth=2, alpha=0.8, label="Clean function (ID)")
    
    # Plot clean function - OOD region
    if np.any(ood_mask):
        ax.plot(x_grid_flat[ood_mask], y_grid_clean_flat[ood_mask], 'r-', 
                linewidth=2, alpha=0.8, label="Clean function (OOD)")
    
    # Plot training data
    ax.scatter(x_train_flat, y_train_flat, alpha=0.5, s=30, color='blue', 
               label="Training data", zorder=5, edgecolors='darkblue', linewidths=0.5)
    
    # Handle train_ranges: if provided, use it; otherwise use train_range as single range
    if train_ranges is None:
        if train_range is not None:
            train_ranges = [train_range]
        else:
            train_ranges = []
    
    # Highlight OOD regions with shaded background
    if ood_ranges:
        for idx, ood_range in enumerate(ood_ranges):
            ood_start, ood_end = ood_range
            ax.axvspan(ood_start, ood_end, alpha=0.2, color='red', 
                       label=f"OOD Region {idx+1}" if idx == 0 else "")
    
    # Add vertical lines to mark training range boundaries
    for idx, train_r in enumerate(train_ranges):
        train_start, train_end = train_r
        ax.axvline(x=train_start, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7, label="Train range" if idx == 0 else "")
        ax.axvline(x=train_end, color='green', linestyle=':', linewidth=2, 
                   alpha=0.7)
    
    # Add vertical lines for OOD boundaries
    if ood_ranges:
        for idx, ood_range in enumerate(ood_ranges):
            ood_start, ood_end = ood_range
            ax.axvline(x=ood_start, color='orange', linestyle='--', linewidth=1.5, 
                       alpha=0.7)
            ax.axvline(x=ood_end, color='orange', linestyle='--', linewidth=1.5, 
                       alpha=0.7)
    
    # Add text annotations
    y_range = ax.get_ylim()
    
    # Annotate training ranges
    for idx, train_r in enumerate(train_ranges):
        train_start, train_end = train_r
        train_mid = (train_start + train_end) / 2
        label = f"Training Range {idx+1}" if len(train_ranges) > 1 else "Training Range"
        ax.text(train_mid, y_range[1] * 0.95, f"{label}\n({train_start:.1f} - {train_end:.1f})", 
                ha="center", va="top", fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    # Annotate OOD ranges
    if ood_ranges:
        for idx, ood_range in enumerate(ood_ranges):
            ood_start, ood_end = ood_range
            ood_mid = (ood_start + ood_end) / 2
            ax.text(ood_mid, y_range[1] * 0.95, f"OOD Region {idx+1}\n({ood_start:.1f} - {ood_end:.1f})", 
                    ha="center", va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
    
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    
    func_label = f" ({func_type})" if func_type else ""
    train_str = f"Train Ranges: {train_ranges}" if train_ranges else "No training ranges"
    ood_str = f"OOD Ranges: {ood_ranges}" if ood_ranges else "No OOD ranges"
    ax.set_title(f"{title}{func_label}\n{train_str} | {ood_str}", fontsize=13)
    
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot_file:
        save_plot(fig, title, subfolder='data_setup')
    
    plt.show()
    plt.close(fig)


# Simplified plotting function without OOD
def plot_uncertainties_no_ood(x_train_subset, y_train_subset, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, title, noise_type='heteroscedastic', func_type=''):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0]
    
    # Ensure mu_pred is 1D for plotting (handle both 1D and 2D inputs)
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    
    # Plot 1: Predictive mean + Total uncertainty
    axes[0].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[0].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[0].fill_between(x, mu_pred - np.sqrt(tot_var), mu_pred + np.sqrt(tot_var), 
                        alpha=0.3, color='blue', label="±σ(total)")
    axes[0].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric uncertainty only
    axes[1].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[1].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[1].fill_between(x, mu_pred - np.sqrt(ale_var), mu_pred + np.sqrt(ale_var), 
                        alpha=0.3, color='green', label="±σ(aleatoric)")
    axes[1].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Predictive mean + Epistemic uncertainty only
    axes[2].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[2].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[2].fill_between(x, mu_pred - np.sqrt(epi_var), mu_pred + np.sqrt(epi_var), 
                        alpha=0.3, color='orange', label="±σ(epistemic)")
    axes[2].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8)
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties/{noise_type}/{func_type}/
    subfolder = f"uncertainties/{noise_type}/{func_type}" if func_type else f"uncertainties/{noise_type}"
    save_plot(fig, title, subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def plot_uncertainties_ood(x_train, y_train, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, ood_mask, title, noise_type='heteroscedastic', func_type=''):
    """Plot uncertainties with OOD regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_var: Aleatoric uncertainty variance
        epi_var: Epistemic uncertainty variance
        tot_var: Total uncertainty variance
        ood_mask: Boolean mask indicating OOD regions
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure mu_pred is 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_var.ndim > 1:
        ale_var = ale_var.squeeze()
    if epi_var.ndim > 1:
        epi_var = epi_var.squeeze()
    if tot_var.ndim > 1:
        tot_var = tot_var.squeeze()
    
    # Split data into training (ID) and OOD regions
    id_mask = ~ood_mask
    ood_mask_bool = ood_mask
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Find ID/OOD boundaries for vertical separator lines
    boundary_x = []
    if np.any(ood_mask_bool):
        # Find transitions: where mask changes from ID to OOD or vice versa
        transitions = np.where(np.diff(ood_mask_bool.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = x[transitions + 1]  # x values at boundaries
    
    # Plot 1: Predictive mean + Total uncertainty
    axes[0].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Add vertical separator lines
    for bx in boundary_x:
        axes[0].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[0].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[0].fill_between(x[id_mask], mu_pred[id_mask] - np.sqrt(tot_var[id_mask]), 
                        mu_pred[id_mask] + np.sqrt(tot_var[id_mask]), 
                        alpha=0.3, color='blue', label="±σ(total)")
    axes[0].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - np.sqrt(tot_var[ood_mask_bool]), 
                        mu_pred[ood_mask_bool] + np.sqrt(tot_var[ood_mask_bool]), 
                        alpha=0.3, color='lightblue')
    
    # Plot clean function - ID and OOD separately (no labels)
    axes[0].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[0].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        # Mark OOD grid points (no label)
        axes[0].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric uncertainty only
    axes[1].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Add vertical separator lines
    for bx in boundary_x:
        axes[1].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[1].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[1].fill_between(x[id_mask], mu_pred[id_mask] - np.sqrt(ale_var[id_mask]), 
                        mu_pred[id_mask] + np.sqrt(ale_var[id_mask]), 
                        alpha=0.3, color='green', label="±σ(aleatoric)")
    axes[1].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - np.sqrt(ale_var[ood_mask_bool]), 
                        mu_pred[ood_mask_bool] + np.sqrt(ale_var[ood_mask_bool]), 
                        alpha=0.3, color='lightgreen')
    
    # Plot clean function - ID and OOD separately (no labels)
    axes[1].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[1].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        # Mark OOD grid points (no label)
        axes[1].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Predictive mean + Epistemic uncertainty only
    axes[2].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Add vertical separator lines
    for bx in boundary_x:
        axes[2].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[2].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[2].fill_between(x[id_mask], mu_pred[id_mask] - np.sqrt(epi_var[id_mask]), 
                        mu_pred[id_mask] + np.sqrt(epi_var[id_mask]), 
                        alpha=0.3, color='red', label="±σ(epistemic)")
    axes[2].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - np.sqrt(epi_var[ood_mask_bool]), 
                        mu_pred[ood_mask_bool] + np.sqrt(epi_var[ood_mask_bool]), 
                        alpha=0.3, color='coral')
    
    # Plot clean function - ID and OOD separately (no labels)
    axes[2].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[2].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        # Mark OOD grid points (no label)
        axes[2].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties_ood/{noise_type}/{func_type}/
    subfolder = f"uncertainties_ood/{noise_type}/{func_type}" if func_type else f"uncertainties_ood/{noise_type}"
    save_plot(fig, title, subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def plot_uncertainties_undersampling(x_train, y_train, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, 
                                    region_masks, sampling_regions, title, noise_type='heteroscedastic', func_type=''):
    """Plot uncertainties with different sampling regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_var: Aleatoric uncertainty variance
        epi_var: Epistemic uncertainty variance
        tot_var: Total uncertainty variance
        region_masks: List of boolean masks, one for each region
        sampling_regions: List of tuples (region_tuple, density_factor)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure mu_pred is 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_var.ndim > 1:
        ale_var = ale_var.squeeze()
    if epi_var.ndim > 1:
        epi_var = epi_var.squeeze()
    if tot_var.ndim > 1:
        tot_var = tot_var.squeeze()
    
    # Define colors for different regions
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Prepare training data
    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
    
    # Plot training data in ALL three graphs
    for ax in axes:
        for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
            if region_tuple is not None:
                region_mask_train = (x_train_flat >= region_tuple[0]) & (x_train_flat <= region_tuple[1])
            else:
                region_mask_train = region_masks[idx] if idx < len(region_masks) else np.ones(len(x_train_flat), dtype=bool)
            color = colors[idx % len(colors)]
            alpha = 0.3 if density_factor < 0.5 else 0.6
            # Only add label in first iteration to avoid duplicate legend entries
            label = "Training data" if idx == 0 else None
            ax.scatter(x_train_flat[region_mask_train], y_train_flat[region_mask_train], 
                      alpha=alpha, s=10, color=color, label=label, zorder=3)
    
    # Plot predictive mean and uncertainties for each region
    # Use simplified legend: one entry per uncertainty type, regions distinguished by line style/color
    pred_mean_plotted = False
    tot_unc_plotted = False
    ale_unc_plotted = False
    epi_unc_plotted = False
    
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        # Plot 1: Total uncertainty
        if not pred_mean_plotted:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
            pred_mean_plotted = True
        else:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not tot_unc_plotted:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - np.sqrt(tot_var[region_mask]), 
                                mu_pred[region_mask] + np.sqrt(tot_var[region_mask]), 
                                alpha=0.2, color=color, label="Total uncertainty")
            tot_unc_plotted = True
        else:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - np.sqrt(tot_var[region_mask]), 
                                mu_pred[region_mask] + np.sqrt(tot_var[region_mask]), 
                                alpha=0.2, color=color)
        
        # Plot 2: Aleatoric uncertainty
        if idx == 0:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not ale_unc_plotted:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - np.sqrt(ale_var[region_mask]), 
                                mu_pred[region_mask] + np.sqrt(ale_var[region_mask]), 
                                alpha=0.2, color=color, label="Aleatoric uncertainty")
            ale_unc_plotted = True
        else:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - np.sqrt(ale_var[region_mask]), 
                                mu_pred[region_mask] + np.sqrt(ale_var[region_mask]), 
                                alpha=0.2, color=color)
        
        # Plot 3: Epistemic uncertainty
        if idx == 0:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not epi_unc_plotted:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - np.sqrt(epi_var[region_mask]), 
                                mu_pred[region_mask] + np.sqrt(epi_var[region_mask]), 
                                alpha=0.2, color=color, label="Epistemic uncertainty")
            epi_unc_plotted = True
        else:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - np.sqrt(epi_var[region_mask]), 
                                mu_pred[region_mask] + np.sqrt(epi_var[region_mask]), 
                                alpha=0.2, color=color)
    
    # Add vertical lines for region boundaries
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        for ax in axes:
            ax.axvline(x=region_tuple[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=region_tuple[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot clean function
    for ax in axes:
        ax.plot(x, y_clean[:, 0] if y_clean.ndim > 1 else y_clean, 'r--', linewidth=1.5, alpha=0.8, label="Clean function")
    
    # Set labels and titles
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties_undersampling/{noise_type}/{func_type}/
    subfolder = f"uncertainties_undersampling/{noise_type}/{func_type}" if func_type else f"uncertainties_undersampling/{noise_type}"
    save_plot(fig, title, subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def analyze_uncertainties_by_region(x_values, ale_var, epi_var, tot_var, mu_pred=None, 
                                   y_clean=None, region_ranges=None, title="Uncertainty Analysis by Region",
                                   save_plot_file=True, subfolder='region_analysis'):
    """
    General function to compute and plot averaged uncertainties for specific regions.
    
    This function can be used with any saved model outputs to analyze uncertainties
    in specific x-ranges without needing to rerun the model.
    
    Args:
        x_values: Array of x values (1D or 2D, will be flattened)
        ale_var: Array of aleatoric uncertainty variances (1D or 2D, will be flattened)
        epi_var: Array of epistemic uncertainty variances (1D or 2D, will be flattened)
        tot_var: Array of total uncertainty variances (1D or 2D, will be flattened)
        mu_pred: Optional array of predictive means (for plotting)
        y_clean: Optional array of clean function values (for plotting)
        region_ranges: List of (min, max) tuples for regions to analyze. 
                      If None, analyzes entire range as single region.
        title: Plot title
        save_plot_file: Whether to save the plot
        subfolder: Subfolder for saving plots
    
    Returns:
        dict: Dictionary with statistics for each region:
            - 'regions': List of region names
            - 'avg_ale': List of average aleatoric uncertainties per region
            - 'avg_epi': List of average epistemic uncertainties per region
            - 'avg_tot': List of average total uncertainties per region
            - 'std_ale': List of std dev of aleatoric uncertainties per region
            - 'std_epi': List of std dev of epistemic uncertainties per region
            - 'std_tot': List of std dev of total uncertainties per region
            - 'correlations': List of correlations (epi vs ale) per region
            - 'mse': List of MSE values per region (if y_clean provided)
            - 'stats_df': DataFrame with all statistics
    """
    # Flatten arrays if needed
    x_flat = x_values.flatten() if x_values.ndim > 1 else x_values
    ale_flat = ale_var.flatten() if ale_var.ndim > 1 else ale_var
    epi_flat = epi_var.flatten() if epi_var.ndim > 1 else epi_var
    tot_flat = tot_var.flatten() if tot_var.ndim > 1 else tot_var
    
    if mu_pred is not None:
        mu_flat = mu_pred.flatten() if mu_pred.ndim > 1 else mu_pred
    else:
        mu_flat = None
    
    if y_clean is not None:
        y_clean_flat = y_clean.flatten() if y_clean.ndim > 1 else y_clean
    else:
        y_clean_flat = None
    
    # If no regions specified, analyze entire range as single region
    if region_ranges is None:
        region_ranges = [(x_flat.min(), x_flat.max())]
    
    # Compute statistics for each region
    region_names = []
    avg_ale_list = []
    avg_epi_list = []
    avg_tot_list = []
    std_ale_list = []
    std_epi_list = []
    std_tot_list = []
    corr_list = []
    mse_list = []
    region_masks = []
    
    print(f"\n{'='*60}")
    print(f"Uncertainty Analysis by Region: {title}")
    print(f"{'='*60}")
    print(f"\n{'Region':<20} {'Range':<20} {'Avg Ale':<15} {'Avg Epi':<15} {'Avg Tot':<15} {'Correlation':<15} {'MSE':<15}")
    print("-" * 120)
    
    for idx, (region_min, region_max) in enumerate(region_ranges):
        # Create mask for this region
        mask = (x_flat >= region_min) & (x_flat <= region_max)
        region_masks.append(mask)
        
        if not np.any(mask):
            print(f"Region {idx+1:<18} ({region_min:.2f}, {region_max:.2f}) {'No data in range':<20}")
            continue
        
        region_name = f"Region_{idx+1}"
        region_names.append(region_name)
        
        # Extract uncertainties for this region
        ale_region = ale_flat[mask]
        epi_region = epi_flat[mask]
        tot_region = tot_flat[mask]
        
        # Compute statistics
        avg_ale = np.mean(ale_region)
        avg_epi = np.mean(epi_region)
        avg_tot = np.mean(tot_region)
        
        std_ale = np.std(ale_region)
        std_epi = np.std(epi_region)
        std_tot = np.std(tot_region)
        
        correlation = np.corrcoef(epi_region, ale_region)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Compute MSE if y_clean provided
        mse = None
        if mu_flat is not None and y_clean_flat is not None:
            mse = np.mean((mu_flat[mask] - y_clean_flat[mask])**2)
            mse_list.append(mse)
            print(f"{region_name:<20} ({region_min:.2f}, {region_max:.2f}) {avg_ale:>14.6f}  {avg_epi:>14.6f}  {avg_tot:>14.6f}  {correlation:>14.6f}  {mse:>14.6f}")
        else:
            mse_list.append(None)
            print(f"{region_name:<20} ({region_min:.2f}, {region_max:.2f}) {avg_ale:>14.6f}  {avg_epi:>14.6f}  {avg_tot:>14.6f}  {correlation:>14.6f}  {'N/A':<15}")
        
        avg_ale_list.append(avg_ale)
        avg_epi_list.append(avg_epi)
        avg_tot_list.append(avg_tot)
        std_ale_list.append(std_ale)
        std_epi_list.append(std_epi)
        std_tot_list.append(std_tot)
        corr_list.append(correlation)
    
    print(f"{'='*60}\n")
    
    # Create DataFrame with statistics
    stats_dict = {
        'Region': region_names,
        'Range_Min': [r[0] for r in region_ranges[:len(region_names)]],
        'Range_Max': [r[1] for r in region_ranges[:len(region_names)]],
        'Avg_Aleatoric': avg_ale_list,
        'Avg_Epistemic': avg_epi_list,
        'Avg_Total': avg_tot_list,
        'Std_Aleatoric': std_ale_list,
        'Std_Epistemic': std_epi_list,
        'Std_Total': std_tot_list,
        'Correlation_Epi_Ale': corr_list
    }
    
    if any(mse is not None for mse in mse_list):
        stats_dict['MSE'] = mse_list
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average uncertainties by region (bar chart)
    x_pos = np.arange(len(region_names))
    width = 0.25
    
    axes[0, 0].bar(x_pos - width, avg_ale_list, width, label='Aleatoric', color='green', alpha=0.7)
    axes[0, 0].bar(x_pos, avg_epi_list, width, label='Epistemic', color='orange', alpha=0.7)
    axes[0, 0].bar(x_pos + width, avg_tot_list, width, label='Total', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('Average Uncertainty')
    axes[0, 0].set_title('Average Uncertainties by Region')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(region_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Standard deviations by region
    axes[0, 1].bar(x_pos - width, std_ale_list, width, label='Aleatoric', color='green', alpha=0.7)
    axes[0, 1].bar(x_pos, std_epi_list, width, label='Epistemic', color='orange', alpha=0.7)
    axes[0, 1].bar(x_pos + width, std_tot_list, width, label='Total', color='blue', alpha=0.7)
    axes[0, 1].set_xlabel('Region')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Uncertainty Standard Deviations by Region')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(region_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Correlations by region
    axes[1, 0].bar(x_pos, corr_list, width=0.5, color='purple', alpha=0.7)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Region')
    axes[1, 0].set_ylabel('Correlation (Epi vs Ale)')
    axes[1, 0].set_title('Correlation: Epistemic vs Aleatoric by Region')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(region_names, rotation=45, ha='right')
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: MSE by region (if available) or uncertainty spread
    if any(mse is not None for mse in mse_list):
        axes[1, 1].bar(x_pos, [m if m is not None else 0 for m in mse_list], width=0.5, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('MSE by Region')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(region_names, rotation=45, ha='right')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        # Plot coefficient of variation instead
        cv_ale = [std/avg if avg > 0 else 0 for std, avg in zip(std_ale_list, avg_ale_list)]
        cv_epi = [std/avg if avg > 0 else 0 for std, avg in zip(std_epi_list, avg_epi_list)]
        cv_tot = [std/avg if avg > 0 else 0 for std, avg in zip(std_tot_list, avg_tot_list)]
        axes[1, 1].bar(x_pos - width, cv_ale, width, label='Aleatoric', color='green', alpha=0.7)
        axes[1, 1].bar(x_pos, cv_epi, width, label='Epistemic', color='orange', alpha=0.7)
        axes[1, 1].bar(x_pos + width, cv_tot, width, label='Total', color='blue', alpha=0.7)
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        axes[1, 1].set_title('Uncertainty Variability (CV) by Region')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(region_names, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_plot_file:
        save_plot(fig, title, subfolder=subfolder)
    
    plt.show()
    plt.close(fig)
    
    return {
        'regions': region_names,
        'avg_ale': avg_ale_list,
        'avg_epi': avg_epi_list,
        'avg_tot': avg_tot_list,
        'std_ale': std_ale_list,
        'std_epi': std_epi_list,
        'std_tot': std_tot_list,
        'correlations': corr_list,
        'mse': mse_list,
        'stats_df': stats_df,
        'region_masks': region_masks
    }


# ========== Entropy-Based Plotting Functions ==========

def plot_uncertainties_entropy_no_ood(x_train_subset, y_train_subset, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy, title, noise_type='heteroscedastic', func_type=''):
    """Plot entropy-based uncertainties without OOD regions
    
    Args:
        x_train_subset: Training data x values
        y_train_subset: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure mu_pred is 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Convert entropy to equivalent standard deviation for visualization
    # For Gaussian: H = 0.5 * log(2πe * σ²), so σ = exp(H) / sqrt(2πe)
    sqrt_2pi_e = np.sqrt(2 * np.pi * np.e)
    tot_std_equiv = np.exp(tot_entropy) / sqrt_2pi_e
    ale_std_equiv = np.exp(ale_entropy) / sqrt_2pi_e
    epi_std_equiv = np.exp(epi_entropy) / sqrt_2pi_e
    
    # Plot 0: Predictive mean (for reference)
    axes[0].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[0].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[0].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean function")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Predictive mean + Total entropy (as bands)
    axes[1].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[1].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[1].fill_between(x, mu_pred - tot_std_equiv, mu_pred + tot_std_equiv, 
                        alpha=0.3, color='blue', label="±σ(total, from entropy)")
    axes[1].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean function")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Entropy (as ±σ)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric entropy (as bands)
    axes[2].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[2].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[2].fill_between(x, mu_pred - ale_std_equiv, mu_pred + ale_std_equiv, 
                        alpha=0.3, color='green', label="±σ(aleatoric, from entropy)")
    axes[2].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean function")
    axes[2].set_ylabel("y")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Entropy (as ±σ)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    # Plot 3: Predictive mean + Epistemic entropy (as bands)
    axes[3].scatter(x_train_subset[:, 0], y_train_subset[:, 0], alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    axes[3].plot(x, mu_pred, 'b-', linewidth=2, label="Predictive mean")
    axes[3].fill_between(x, mu_pred - epi_std_equiv, mu_pred + epi_std_equiv, 
                        alpha=0.3, color='orange', label="±σ(epistemic, from entropy)")
    axes[3].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean function")
    axes[3].set_ylabel("y")
    axes[3].set_xlabel("x")
    axes[3].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Entropy (as ±σ)")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties_entropy/{noise_type}/{func_type}/
    subfolder = f"uncertainties_entropy/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy/{noise_type}"
    save_plot(fig, f"{title}_entropy", subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def plot_uncertainties_entropy_ood(x_train, y_train, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask, title, noise_type='heteroscedastic', func_type=''):
    """Plot entropy-based uncertainties with OOD regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        ood_mask: Boolean mask indicating OOD regions
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Split data into training (ID) and OOD regions
    id_mask = ~ood_mask
    ood_mask_bool = ood_mask
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Convert entropy to equivalent standard deviation for visualization
    # For Gaussian: H = 0.5 * log(2πe * σ²), so σ = exp(H) / sqrt(2πe)
    sqrt_2pi_e = np.sqrt(2 * np.pi * np.e)
    tot_std_equiv = np.exp(tot_entropy) / sqrt_2pi_e
    ale_std_equiv = np.exp(ale_entropy) / sqrt_2pi_e
    epi_std_equiv = np.exp(epi_entropy) / sqrt_2pi_e
    
    # Find ID/OOD boundaries for vertical separator lines
    boundary_x = []
    if np.any(ood_mask_bool):
        # Find transitions: where mask changes from ID to OOD or vice versa
        transitions = np.where(np.diff(ood_mask_bool.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = x[transitions + 1]  # x values at boundaries
    
    # Plot 0: Predictive mean + Total entropy (as bands)
    axes[0].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Add vertical separator lines
    for bx in boundary_x:
        axes[0].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[0].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[0].fill_between(x[id_mask], mu_pred[id_mask] - tot_std_equiv[id_mask], 
                        mu_pred[id_mask] + tot_std_equiv[id_mask], 
                        alpha=0.3, color='blue', label="±σ(total, from entropy)")
    axes[0].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - tot_std_equiv[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + tot_std_equiv[ood_mask_bool], 
                        alpha=0.3, color='lightblue')
    
    # Plot clean function - ID and OOD separately (no labels)
    axes[0].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[0].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        # Mark OOD grid points (no label)
        axes[0].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Entropy (as ±σ)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Predictive mean + Aleatoric entropy (as bands)
    axes[1].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Add vertical separator lines
    for bx in boundary_x:
        axes[1].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[1].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[1].fill_between(x[id_mask], mu_pred[id_mask] - ale_std_equiv[id_mask], 
                        mu_pred[id_mask] + ale_std_equiv[id_mask], 
                        alpha=0.3, color='green', label="±σ(aleatoric, from entropy)")
    axes[1].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - ale_std_equiv[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + ale_std_equiv[ood_mask_bool], 
                        alpha=0.3, color='lightgreen')
    
    # Plot clean function - ID and OOD separately (no labels)
    axes[1].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[1].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        # Mark OOD grid points (no label)
        axes[1].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Entropy (as ±σ)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Epistemic entropy (as bands)
    axes[2].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Add vertical separator lines
    for bx in boundary_x:
        axes[2].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[2].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[2].fill_between(x[id_mask], mu_pred[id_mask] - epi_std_equiv[id_mask], 
                        mu_pred[id_mask] + epi_std_equiv[id_mask], 
                        alpha=0.3, color='red', label="±σ(epistemic, from entropy)")
    axes[2].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - epi_std_equiv[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + epi_std_equiv[ood_mask_bool], 
                        alpha=0.3, color='coral')
    
    # Plot clean function - ID and OOD separately (no labels)
    axes[2].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[2].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        # Mark OOD grid points (no label)
        axes[2].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Entropy (as ±σ)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties_entropy_ood/{noise_type}/{func_type}/
    subfolder = f"uncertainties_entropy_ood/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_ood/{noise_type}"
    save_plot(fig, f"{title}_entropy", subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def _normalize_values(values, vmin=None, vmax=None):
    """Normalize values to [0, 1] range"""
    values = np.asarray(values)
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    if vmax - vmin == 0:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def plot_uncertainties_ood_normalized(x_train, y_train, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, ood_mask, title, noise_type='heteroscedastic', func_type='', scale_factor=0.3):
    """Plot normalized variance-based uncertainties with OOD regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_var: Aleatoric uncertainty variance
        epi_var: Epistemic uncertainty variance
        tot_var: Total uncertainty variance
        ood_mask: Boolean mask indicating OOD regions
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        scale_factor: Scaling factor for normalized bands (default 0.3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure mu_pred is 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_var.ndim > 1:
        ale_var = ale_var.squeeze()
    if epi_var.ndim > 1:
        epi_var = epi_var.squeeze()
    if tot_var.ndim > 1:
        tot_var = tot_var.squeeze()
    
    # Split data into training (ID) and OOD regions
    id_mask = ~ood_mask
    ood_mask_bool = ood_mask
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Compute y-axis range
    y_range = y_clean_flat.max() - y_clean_flat.min()
    
    # Convert variance to standard deviation and normalize ale and epi separately
    std_ale = np.sqrt(ale_var)
    std_epi = np.sqrt(epi_var)
    
    # Normalize ale and epi separately
    std_ale_norm = _normalize_values(std_ale)
    std_epi_norm = _normalize_values(std_epi)
    
    # Total is sum of normalized ale and epi
    std_tot_norm = std_ale_norm + std_epi_norm
    
    # Scale normalized values to y-axis range
    band_width_tot = std_tot_norm * y_range * scale_factor
    band_width_ale = std_ale_norm * y_range * scale_factor
    band_width_epi = std_epi_norm * y_range * scale_factor
    
    # Find ID/OOD boundaries for vertical separator lines
    boundary_x = []
    if np.any(ood_mask_bool):
        transitions = np.where(np.diff(ood_mask_bool.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = x[transitions + 1]
    
    # Plot 1: Predictive mean + Total uncertainty (normalized)
    axes[0].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[0].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[0].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[0].fill_between(x[id_mask], mu_pred[id_mask] - band_width_tot[id_mask], 
                        mu_pred[id_mask] + band_width_tot[id_mask], 
                        alpha=0.3, color='blue', label="±norm(total)")
    axes[0].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - band_width_tot[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + band_width_tot[ood_mask_bool], 
                        alpha=0.3, color='lightblue')
    
    axes[0].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[0].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[0].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty (Normalized)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric uncertainty (normalized)
    axes[1].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[1].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[1].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[1].fill_between(x[id_mask], mu_pred[id_mask] - band_width_ale[id_mask], 
                        mu_pred[id_mask] + band_width_ale[id_mask], 
                        alpha=0.3, color='green', label="±norm(aleatoric)")
    axes[1].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - band_width_ale[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + band_width_ale[ood_mask_bool], 
                        alpha=0.3, color='lightgreen')
    
    axes[1].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[1].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[1].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty (Normalized)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Predictive mean + Epistemic uncertainty (normalized)
    axes[2].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[2].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[2].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[2].fill_between(x[id_mask], mu_pred[id_mask] - band_width_epi[id_mask], 
                        mu_pred[id_mask] + band_width_epi[id_mask], 
                        alpha=0.3, color='red', label="±norm(epistemic)")
    axes[2].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - band_width_epi[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + band_width_epi[ood_mask_bool], 
                        alpha=0.3, color='coral')
    
    axes[2].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[2].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[2].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty (Normalized)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_ood/{noise_type}/{func_type}" if func_type else f"uncertainties_ood/{noise_type}"
    save_plot(fig, f"{title}_normalized", subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def plot_uncertainties_entropy_ood_normalized(x_train, y_train, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask, title, noise_type='heteroscedastic', func_type='', scale_factor=0.3):
    """Plot normalized entropy-based uncertainties with OOD regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        ood_mask: Boolean mask indicating OOD regions
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        scale_factor: Scaling factor for normalized bands (default 0.3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Split data into training (ID) and OOD regions
    id_mask = ~ood_mask
    ood_mask_bool = ood_mask
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Compute y-axis range
    y_range = y_clean_flat.max() - y_clean_flat.min()
    
    # Normalize ale and epi entropy separately
    ale_entropy_norm = _normalize_values(ale_entropy)
    epi_entropy_norm = _normalize_values(epi_entropy)
    
    # Total is sum of normalized ale and epi
    tot_entropy_norm = ale_entropy_norm + epi_entropy_norm
    
    # Scale normalized values to y-axis range
    band_width_tot = tot_entropy_norm * y_range * scale_factor
    band_width_ale = ale_entropy_norm * y_range * scale_factor
    band_width_epi = epi_entropy_norm * y_range * scale_factor
    
    # Find ID/OOD boundaries for vertical separator lines
    boundary_x = []
    if np.any(ood_mask_bool):
        transitions = np.where(np.diff(ood_mask_bool.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = x[transitions + 1]
    
    # Plot 0: Predictive mean + Total entropy (normalized)
    axes[0].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[0].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[0].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[0].fill_between(x[id_mask], mu_pred[id_mask] - band_width_tot[id_mask], 
                        mu_pred[id_mask] + band_width_tot[id_mask], 
                        alpha=0.3, color='blue', label="±norm(entropy, total)")
    axes[0].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - band_width_tot[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + band_width_tot[ood_mask_bool], 
                        alpha=0.3, color='lightblue')
    
    axes[0].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[0].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[0].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Entropy (Normalized)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Predictive mean + Aleatoric entropy (normalized)
    axes[1].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[1].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[1].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[1].fill_between(x[id_mask], mu_pred[id_mask] - band_width_ale[id_mask], 
                        mu_pred[id_mask] + band_width_ale[id_mask], 
                        alpha=0.3, color='green', label="±norm(entropy, aleatoric)")
    axes[1].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - band_width_ale[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + band_width_ale[ood_mask_bool], 
                        alpha=0.3, color='lightgreen')
    
    axes[1].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[1].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[1].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Entropy (Normalized)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Epistemic entropy (normalized)
    axes[2].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[2].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    axes[2].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean")
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2)
    axes[2].fill_between(x[id_mask], mu_pred[id_mask] - band_width_epi[id_mask], 
                        mu_pred[id_mask] + band_width_epi[id_mask], 
                        alpha=0.3, color='red', label="±norm(entropy, epistemic)")
    axes[2].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - band_width_epi[ood_mask_bool], 
                        mu_pred[ood_mask_bool] + band_width_epi[ood_mask_bool], 
                        alpha=0.3, color='coral')
    
    axes[2].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[2].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[2].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color='red', 
                       alpha=0.4, marker='x', zorder=4)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Entropy (Normalized)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_entropy_ood/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_ood/{noise_type}"
    save_plot(fig, f"{title}_entropy_normalized", subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def plot_entropy_lines_ood(x_train, y_train, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask, title, noise_type='heteroscedastic', func_type=''):
    """Plot entropy values directly as line plots (in nats) with OOD regions highlighted
    
    Shows entropy values on y-axis, separate from predictive mean.
    Useful for understanding actual entropy magnitudes in nats.
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        ood_mask: Boolean mask indicating OOD regions
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Split data into training (ID) and OOD regions
    id_mask = ~ood_mask
    ood_mask_bool = ood_mask
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Find ID/OOD boundaries for vertical separator lines
    boundary_x = []
    if np.any(ood_mask_bool):
        transitions = np.where(np.diff(ood_mask_bool.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = x[transitions + 1]
    
    # Plot 0: Total entropy as line plot
    axes[0].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[0].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    # Plot predictive mean on primary y-axis
    ax1_twin = axes[0].twinx()
    axes[0].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2, alpha=0.5)
    
    # Plot entropy on secondary y-axis
    ax1_twin.plot(x[id_mask], tot_entropy[id_mask], 'g-', linewidth=2, label="Total entropy (nats)")
    ax1_twin.plot(x[ood_mask_bool], tot_entropy[ood_mask_bool], 'g-', linewidth=2, alpha=0.7)
    
    axes[0].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    if np.any(ood_mask_bool):
        axes[0].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[0].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[0].set_ylabel("y / Predictive mean", fontsize=11)
    ax1_twin.set_ylabel("Entropy (nats)", fontsize=11, color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Total Entropy (nats)")
    axes[0].legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Aleatoric entropy as line plot
    axes[1].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[1].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    # Plot predictive mean on primary y-axis
    ax2_twin = axes[1].twinx()
    axes[1].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2, alpha=0.5)
    
    # Plot entropy on secondary y-axis
    ax2_twin.plot(x[id_mask], ale_entropy[id_mask], 'g-', linewidth=2, label="Aleatoric entropy (nats)")
    ax2_twin.plot(x[ood_mask_bool], ale_entropy[ood_mask_bool], 'g-', linewidth=2, alpha=0.7)
    
    axes[1].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    if np.any(ood_mask_bool):
        axes[1].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[1].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[1].set_ylabel("y / Predictive mean", fontsize=11)
    ax2_twin.set_ylabel("Entropy (nats)", fontsize=11, color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Aleatoric Entropy (nats)")
    axes[1].legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Epistemic entropy as line plot
    axes[2].scatter(x_train[:, 0] if x_train.ndim > 1 else x_train, 
                   y_train[:, 0] if y_train.ndim > 1 else y_train, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    for bx in boundary_x:
        axes[2].axvline(x=bx, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    
    # Plot predictive mean on primary y-axis
    ax3_twin = axes[2].twinx()
    axes[2].plot(x[id_mask], mu_pred[id_mask], 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b-', linewidth=1.2, alpha=0.5)
    
    # Plot entropy on secondary y-axis
    ax3_twin.plot(x[id_mask], epi_entropy[id_mask], 'r-', linewidth=2, label="Epistemic entropy (nats)")
    ax3_twin.plot(x[ood_mask_bool], epi_entropy[ood_mask_bool], 'r-', linewidth=2, alpha=0.7)
    
    axes[2].plot(x[id_mask], y_clean_flat[id_mask], 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    if np.any(ood_mask_bool):
        axes[2].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], 'r-', linewidth=1.5, alpha=0.8)
        axes[2].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color='red', 
                       alpha=0.3, marker='x', zorder=4)
    
    axes[2].set_ylabel("y / Predictive mean", fontsize=11)
    axes[2].set_xlabel("x", fontsize=11)
    ax3_twin.set_ylabel("Entropy (nats)", fontsize=11, color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Epistemic Entropy (nats)")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_entropy_lines_ood/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_lines_ood/{noise_type}"
    save_plot(fig, f"{title}_entropy_lines", subfolder=subfolder)
    
    plt.show()
    plt.close(fig)


def plot_entropy_lines_no_ood(x_train_subset, y_train_subset, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy, title, noise_type='heteroscedastic', func_type=''):
    """Plot entropy values directly as line plots (in nats) without OOD regions
    
    Shows entropy values on y-axis, separate from predictive mean.
    Useful for understanding actual entropy magnitudes in nats.
    
    Args:
        x_train_subset: Training data x values
        y_train_subset: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Plot 0: Total entropy as line plot
    axes[0].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Plot predictive mean on primary y-axis
    ax1_twin = axes[0].twinx()
    axes[0].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
    
    # Plot entropy on secondary y-axis
    ax1_twin.plot(x, tot_entropy, 'g-', linewidth=2, label="Total entropy (nats)")
    
    axes[0].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[0].set_ylabel("y / Predictive mean", fontsize=11)
    ax1_twin.set_ylabel("Entropy (nats)", fontsize=11, color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Total Entropy (nats)")
    axes[0].legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Aleatoric entropy as line plot
    axes[1].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Plot predictive mean on primary y-axis
    ax2_twin = axes[1].twinx()
    axes[1].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
    
    # Plot entropy on secondary y-axis
    ax2_twin.plot(x, ale_entropy, 'g-', linewidth=2, label="Aleatoric entropy (nats)")
    
    axes[1].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[1].set_ylabel("y / Predictive mean", fontsize=11)
    ax2_twin.set_ylabel("Entropy (nats)", fontsize=11, color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Aleatoric Entropy (nats)")
    axes[1].legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Epistemic entropy as line plot
    axes[2].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    # Plot predictive mean on primary y-axis
    ax3_twin = axes[2].twinx()
    axes[2].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
    
    # Plot entropy on secondary y-axis
    ax3_twin.plot(x, epi_entropy, 'r-', linewidth=2, label="Epistemic entropy (nats)")
    
    axes[2].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[2].set_ylabel("y / Predictive mean", fontsize=11)
    axes[2].set_xlabel("x", fontsize=11)
    ax3_twin.set_ylabel("Entropy (nats)", fontsize=11, color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Epistemic Entropy (nats)")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_entropy_lines_no_ood/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_lines_no_ood/{noise_type}"
    save_plot(fig, f"{title}_entropy_lines", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_uncertainties_no_ood_normalized(x_train_subset, y_train_subset, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, title, noise_type='heteroscedastic', func_type='', scale_factor=0.3):
    """Plot normalized variance-based uncertainties without OOD regions
    
    Args:
        x_train_subset: Training data x values
        y_train_subset: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_var: Aleatoric uncertainty variance
        epi_var: Epistemic uncertainty variance
        tot_var: Total uncertainty variance
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        scale_factor: Scaling factor for normalized bands (default 0.3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure mu_pred is 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_var.ndim > 1:
        ale_var = ale_var.squeeze()
    if epi_var.ndim > 1:
        epi_var = epi_var.squeeze()
    if tot_var.ndim > 1:
        tot_var = tot_var.squeeze()
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Compute y-axis range
    y_range = y_clean_flat.max() - y_clean_flat.min()
    
    # Convert variance to standard deviation and normalize ale and epi separately
    std_ale = np.sqrt(ale_var)
    std_epi = np.sqrt(epi_var)
    
    # Normalize ale and epi separately
    std_ale_norm = _normalize_values(std_ale)
    std_epi_norm = _normalize_values(std_epi)
    
    # Total is sum of normalized ale and epi
    std_tot_norm = std_ale_norm + std_epi_norm
    
    # Scale normalized values to y-axis range
    band_width_tot = std_tot_norm * y_range * scale_factor
    band_width_ale = std_ale_norm * y_range * scale_factor
    band_width_epi = std_epi_norm * y_range * scale_factor
    
    # Plot 1: Predictive mean + Total uncertainty (normalized)
    axes[0].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    axes[0].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean")
    axes[0].fill_between(x, mu_pred - band_width_tot, mu_pred + band_width_tot, 
                        alpha=0.3, color='blue', label="±norm(total)")
    
    axes[0].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty (Normalized)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric uncertainty (normalized)
    axes[1].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    axes[1].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean")
    axes[1].fill_between(x, mu_pred - band_width_ale, mu_pred + band_width_ale, 
                        alpha=0.3, color='green', label="±norm(aleatoric)")
    
    axes[1].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty (Normalized)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Predictive mean + Epistemic uncertainty (normalized)
    axes[2].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    axes[2].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean")
    axes[2].fill_between(x, mu_pred - band_width_epi, mu_pred + band_width_epi, 
                        alpha=0.3, color='red', label="±norm(epistemic)")
    
    axes[2].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty (Normalized)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_no_ood_normalized/{noise_type}/{func_type}" if func_type else f"uncertainties_no_ood_normalized/{noise_type}"
    save_plot(fig, f"{title}_normalized", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_uncertainties_entropy_no_ood_normalized(x_train_subset, y_train_subset, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy, title, noise_type='heteroscedastic', func_type='', scale_factor=0.3):
    """Plot normalized entropy-based uncertainties without OOD regions
    
    Args:
        x_train_subset: Training data x values
        y_train_subset: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        scale_factor: Scaling factor for normalized bands (default 0.3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Compute y-axis range
    y_range = y_clean_flat.max() - y_clean_flat.min()
    
    # Normalize ale and epi entropy separately
    ale_entropy_norm = _normalize_values(ale_entropy)
    epi_entropy_norm = _normalize_values(epi_entropy)
    
    # Total is sum of normalized ale and epi
    tot_entropy_norm = ale_entropy_norm + epi_entropy_norm
    
    # Scale normalized values to y-axis range
    band_width_tot = tot_entropy_norm * y_range * scale_factor
    band_width_ale = ale_entropy_norm * y_range * scale_factor
    band_width_epi = epi_entropy_norm * y_range * scale_factor
    
    # Plot 0: Predictive mean + Total entropy (normalized)
    axes[0].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    axes[0].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean")
    axes[0].fill_between(x, mu_pred - band_width_tot, mu_pred + band_width_tot, 
                        alpha=0.3, color='blue', label="±norm(entropy, total)")
    
    axes[0].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Entropy (Normalized)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Predictive mean + Aleatoric entropy (normalized)
    axes[1].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    axes[1].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean")
    axes[1].fill_between(x, mu_pred - band_width_ale, mu_pred + band_width_ale, 
                        alpha=0.3, color='green', label="±norm(entropy, aleatoric)")
    
    axes[1].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Entropy (Normalized)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Epistemic entropy (normalized)
    axes[2].scatter(x_train_subset[:, 0] if x_train_subset.ndim > 1 else x_train_subset, 
                   y_train_subset[:, 0] if y_train_subset.ndim > 1 else y_train_subset, 
                   alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
    
    axes[2].plot(x, mu_pred, 'b-', linewidth=1.2, label="Predictive mean")
    axes[2].fill_between(x, mu_pred - band_width_epi, mu_pred + band_width_epi, 
                        alpha=0.3, color='red', label="±norm(entropy, epistemic)")
    
    axes[2].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Entropy (Normalized)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_entropy_no_ood_normalized/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_no_ood_normalized/{noise_type}"
    save_plot(fig, f"{title}_entropy_normalized", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_uncertainties_entropy_undersampling(x_train, y_train, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy,
                                            region_masks, sampling_regions, title, noise_type='heteroscedastic', func_type=''):
    """Plot entropy-based uncertainties with different sampling regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        region_masks: List of boolean masks, one for each region
        sampling_regions: List of tuples (region_tuple, density_factor)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Convert entropy to equivalent standard deviation for visualization
    # For Gaussian: H = 0.5 * log(2πe * σ²), so σ = exp(H) / sqrt(2πe)
    sqrt_2pi_e = np.sqrt(2 * np.pi * np.e)
    tot_std_equiv = np.exp(tot_entropy) / sqrt_2pi_e
    ale_std_equiv = np.exp(ale_entropy) / sqrt_2pi_e
    epi_std_equiv = np.exp(epi_entropy) / sqrt_2pi_e
    
    # Define colors for different regions
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Prepare training data
    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
    
    # Plot training data in ALL three graphs
    for ax in axes:
        for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
            if region_tuple is not None:
                region_mask_train = (x_train_flat >= region_tuple[0]) & (x_train_flat <= region_tuple[1])
            else:
                region_mask_train = region_masks[idx] if idx < len(region_masks) else np.ones(len(x_train_flat), dtype=bool)
            color = colors[idx % len(colors)]
            alpha = 0.3 if density_factor < 0.5 else 0.6
            # Only add label in first iteration to avoid duplicate legend entries
            label = "Training data" if idx == 0 else None
            ax.scatter(x_train_flat[region_mask_train], y_train_flat[region_mask_train], 
                      alpha=alpha, s=10, color=color, label=label, zorder=3)
    
    # Plot predictive mean and uncertainties for each region
    # Use simplified legend: one entry per uncertainty type, regions distinguished by line style/color
    pred_mean_plotted = False
    tot_unc_plotted = False
    ale_unc_plotted = False
    epi_unc_plotted = False
    
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        # Plot 1: Total uncertainty
        if not pred_mean_plotted:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
            pred_mean_plotted = True
        else:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not tot_unc_plotted:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - tot_std_equiv[region_mask], 
                                mu_pred[region_mask] + tot_std_equiv[region_mask], 
                                alpha=0.2, color=color, label="Total uncertainty")
            tot_unc_plotted = True
        else:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - tot_std_equiv[region_mask], 
                                mu_pred[region_mask] + tot_std_equiv[region_mask], 
                                alpha=0.2, color=color)
        
        # Plot 2: Aleatoric uncertainty
        if idx == 0:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not ale_unc_plotted:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - ale_std_equiv[region_mask], 
                                mu_pred[region_mask] + ale_std_equiv[region_mask], 
                                alpha=0.2, color=color, label="Aleatoric uncertainty")
            ale_unc_plotted = True
        else:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - ale_std_equiv[region_mask], 
                                mu_pred[region_mask] + ale_std_equiv[region_mask], 
                                alpha=0.2, color=color)
        
        # Plot 3: Epistemic uncertainty
        if idx == 0:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not epi_unc_plotted:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - epi_std_equiv[region_mask], 
                                mu_pred[region_mask] + epi_std_equiv[region_mask], 
                                alpha=0.2, color=color, label="Epistemic uncertainty")
            epi_unc_plotted = True
        else:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - epi_std_equiv[region_mask], 
                                mu_pred[region_mask] + epi_std_equiv[region_mask], 
                                alpha=0.2, color=color)
    
    # Add vertical lines for region boundaries
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        for ax in axes:
            ax.axvline(x=region_tuple[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=region_tuple[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot clean function
    for ax in axes:
        ax.plot(x, y_clean[:, 0] if y_clean.ndim > 1 else y_clean, 'r--', linewidth=1.5, alpha=0.8, label="Clean function")
    
    # Set labels and titles
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty (Entropy)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty (Entropy)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty (Entropy)")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure: uncertainties_entropy_undersampling/{noise_type}/{func_type}/
    subfolder = f"uncertainties_entropy_undersampling/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_undersampling/{noise_type}"
    save_plot(fig, f"{title}_entropy", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_entropy_lines_undersampling(x_train, y_train, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy,
                                     region_masks, sampling_regions, title, noise_type='heteroscedastic', func_type=''):
    """Plot entropy values directly as line plots (in nats) with undersampling regions highlighted
    
    Shows entropy values on y-axis, separate from predictive mean.
    Useful for understanding actual entropy magnitudes in nats.
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        region_masks: List of boolean masks, one for each region
        sampling_regions: List of tuples (region_tuple, density_factor)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Define colors for different regions
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Prepare training data
    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
    
    # Plot training data in ALL three graphs
    for ax in axes:
        for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
            if region_tuple is not None:
                region_mask_train = (x_train_flat >= region_tuple[0]) & (x_train_flat <= region_tuple[1])
            else:
                region_mask_train = region_masks[idx] if idx < len(region_masks) else np.ones(len(x_train_flat), dtype=bool)
            color = colors[idx % len(colors)]
            alpha = 0.3 if density_factor < 0.5 else 0.6
            label = "Training data" if idx == 0 else None
            ax.scatter(x_train_flat[region_mask_train], y_train_flat[region_mask_train], 
                      alpha=alpha, s=10, color=color, label=label, zorder=3)
    
    # Add vertical lines for region boundaries
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        for ax in axes:
            ax.axvline(x=region_tuple[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=region_tuple[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot predictive mean on primary y-axis and entropy on secondary y-axis for each region
    pred_mean_plotted = False
    
    # Plot 0: Total entropy
    ax1_twin = axes[0].twinx()
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        if not pred_mean_plotted:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean", alpha=0.5)
            ax1_twin.plot(x[region_mask], tot_entropy[region_mask], linestyle=linestyle, 
                         linewidth=2, color=color, label="Total entropy (nats)")
            pred_mean_plotted = True
        else:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, alpha=0.5)
            ax1_twin.plot(x[region_mask], tot_entropy[region_mask], linestyle=linestyle, 
                         linewidth=2, color=color)
    
    axes[0].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[0].set_ylabel("y / Predictive mean", fontsize=11)
    ax1_twin.set_ylabel("Entropy (nats)", fontsize=11, color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Total Entropy (nats)")
    axes[0].legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 1: Aleatoric entropy
    ax2_twin = axes[1].twinx()
    pred_mean_plotted = False
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        if idx == 0:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean", alpha=0.5)
            ax2_twin.plot(x[region_mask], ale_entropy[region_mask], linestyle=linestyle, 
                         linewidth=2, color=color, label="Aleatoric entropy (nats)")
        else:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, alpha=0.5)
            ax2_twin.plot(x[region_mask], ale_entropy[region_mask], linestyle=linestyle, 
                         linewidth=2, color=color)
    
    axes[1].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[1].set_ylabel("y / Predictive mean", fontsize=11)
    ax2_twin.set_ylabel("Entropy (nats)", fontsize=11, color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Aleatoric Entropy (nats)")
    axes[1].legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 2: Epistemic entropy
    ax3_twin = axes[2].twinx()
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        if idx == 0:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean", alpha=0.5)
            ax3_twin.plot(x[region_mask], epi_entropy[region_mask], linestyle=linestyle, 
                         linewidth=2, color=color, label="Epistemic entropy (nats)")
        else:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, alpha=0.5)
            ax3_twin.plot(x[region_mask], epi_entropy[region_mask], linestyle=linestyle, 
                         linewidth=2, color=color)
    
    axes[2].plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    axes[2].set_ylabel("y / Predictive mean", fontsize=11)
    axes[2].set_xlabel("x", fontsize=11)
    ax3_twin.set_ylabel("Entropy (nats)", fontsize=11, color='red')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Epistemic Entropy (nats)")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_entropy_lines_undersampling/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_lines_undersampling/{noise_type}"
    save_plot(fig, f"{title}_entropy_lines", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_uncertainties_undersampling_normalized(x_train, y_train, x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var,
                                                region_masks, sampling_regions, title, noise_type='heteroscedastic', func_type='', scale_factor=0.3):
    """Plot normalized variance-based uncertainties with undersampling regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_var: Aleatoric uncertainty variance
        epi_var: Epistemic uncertainty variance
        tot_var: Total uncertainty variance
        region_masks: List of boolean masks, one for each region
        sampling_regions: List of tuples (region_tuple, density_factor)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        scale_factor: Scaling factor for normalized bands (default 0.3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure mu_pred is 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_var.ndim > 1:
        ale_var = ale_var.squeeze()
    if epi_var.ndim > 1:
        epi_var = epi_var.squeeze()
    if tot_var.ndim > 1:
        tot_var = tot_var.squeeze()
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Compute y-axis range
    y_range = y_clean_flat.max() - y_clean_flat.min()
    
    # Convert variance to standard deviation and normalize ale and epi separately
    std_ale = np.sqrt(ale_var)
    std_epi = np.sqrt(epi_var)
    
    # Normalize ale and epi separately
    std_ale_norm = _normalize_values(std_ale)
    std_epi_norm = _normalize_values(std_epi)
    
    # Total is sum of normalized ale and epi
    std_tot_norm = std_ale_norm + std_epi_norm
    
    # Scale normalized values to y-axis range
    band_width_tot = std_tot_norm * y_range * scale_factor
    band_width_ale = std_ale_norm * y_range * scale_factor
    band_width_epi = std_epi_norm * y_range * scale_factor
    
    # Define colors for different regions
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Prepare training data
    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
    
    # Plot training data in ALL three graphs
    for ax in axes:
        for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
            if region_tuple is not None:
                region_mask_train = (x_train_flat >= region_tuple[0]) & (x_train_flat <= region_tuple[1])
            else:
                region_mask_train = region_masks[idx] if idx < len(region_masks) else np.ones(len(x_train_flat), dtype=bool)
            color = colors[idx % len(colors)]
            alpha = 0.3 if density_factor < 0.5 else 0.6
            label = "Training data" if idx == 0 else None
            ax.scatter(x_train_flat[region_mask_train], y_train_flat[region_mask_train], 
                      alpha=alpha, s=10, color=color, label=label, zorder=3)
    
    # Add vertical lines for region boundaries
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        for ax in axes:
            ax.axvline(x=region_tuple[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=region_tuple[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot predictive mean and uncertainties for each region
    pred_mean_plotted = False
    tot_unc_plotted = False
    ale_unc_plotted = False
    epi_unc_plotted = False
    
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        # Plot 0: Total uncertainty (normalized)
        if not pred_mean_plotted:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
            pred_mean_plotted = True
        else:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not tot_unc_plotted:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - band_width_tot[region_mask], 
                                mu_pred[region_mask] + band_width_tot[region_mask], 
                                alpha=0.3, color=color, label="±norm(total)")
            tot_unc_plotted = True
        else:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - band_width_tot[region_mask], 
                                mu_pred[region_mask] + band_width_tot[region_mask], 
                                alpha=0.3, color=color)
        
        # Plot 1: Aleatoric uncertainty (normalized)
        if idx == 0:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not ale_unc_plotted:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - band_width_ale[region_mask], 
                                mu_pred[region_mask] + band_width_ale[region_mask], 
                                alpha=0.3, color=color, label="±norm(aleatoric)")
            ale_unc_plotted = True
        else:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - band_width_ale[region_mask], 
                                mu_pred[region_mask] + band_width_ale[region_mask], 
                                alpha=0.3, color=color)
        
        # Plot 2: Epistemic uncertainty (normalized)
        if idx == 0:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not epi_unc_plotted:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - band_width_epi[region_mask], 
                                mu_pred[region_mask] + band_width_epi[region_mask], 
                                alpha=0.3, color=color, label="±norm(epistemic)")
            epi_unc_plotted = True
        else:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - band_width_epi[region_mask], 
                                mu_pred[region_mask] + band_width_epi[region_mask], 
                                alpha=0.3, color=color)
    
    # Plot clean function
    for ax in axes:
        ax.plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    # Set labels and titles
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty (Normalized)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty (Normalized)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty (Normalized)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_undersampling_normalized/{noise_type}/{func_type}" if func_type else f"uncertainties_undersampling_normalized/{noise_type}"
    save_plot(fig, f"{title}_normalized", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_uncertainties_entropy_undersampling_normalized(x_train, y_train, x_grid, y_clean, mu_pred, ale_entropy, epi_entropy, tot_entropy,
                                                         region_masks, sampling_regions, title, noise_type='heteroscedastic', func_type='', scale_factor=0.3):
    """Plot normalized entropy-based uncertainties with undersampling regions highlighted
    
    Args:
        x_train: Training data x values
        y_train: Training data y values
        x_grid: Grid x values for evaluation
        y_clean: Clean function values at grid points
        mu_pred: Predictive mean
        ale_entropy: Aleatoric entropy (nats)
        epi_entropy: Epistemic entropy (nats)
        tot_entropy: Total entropy (nats)
        region_masks: List of boolean masks, one for each region
        sampling_regions: List of tuples (region_tuple, density_factor)
        title: Plot title
        noise_type: Type of noise ('heteroscedastic' or 'homoscedastic')
        func_type: Function type identifier (e.g., 'linear', 'sin')
        scale_factor: Scaling factor for normalized bands (default 0.3)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    
    # Ensure arrays are 1D for plotting
    if mu_pred.ndim > 1:
        mu_pred = mu_pred.squeeze()
    if ale_entropy.ndim > 1:
        ale_entropy = ale_entropy.squeeze()
    if epi_entropy.ndim > 1:
        epi_entropy = epi_entropy.squeeze()
    if tot_entropy.ndim > 1:
        tot_entropy = tot_entropy.squeeze()
    
    # Prepare clean function values
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean
    
    # Compute y-axis range
    y_range = y_clean_flat.max() - y_clean_flat.min()
    
    # Normalize ale and epi entropy separately
    ale_entropy_norm = _normalize_values(ale_entropy)
    epi_entropy_norm = _normalize_values(epi_entropy)
    
    # Total is sum of normalized ale and epi
    tot_entropy_norm = ale_entropy_norm + epi_entropy_norm
    
    # Scale normalized values to y-axis range
    band_width_tot = tot_entropy_norm * y_range * scale_factor
    band_width_ale = ale_entropy_norm * y_range * scale_factor
    band_width_epi = epi_entropy_norm * y_range * scale_factor
    
    # Define colors for different regions
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    # Prepare training data
    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
    
    # Plot training data in ALL three graphs
    for ax in axes:
        for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
            if region_tuple is not None:
                region_mask_train = (x_train_flat >= region_tuple[0]) & (x_train_flat <= region_tuple[1])
            else:
                region_mask_train = region_masks[idx] if idx < len(region_masks) else np.ones(len(x_train_flat), dtype=bool)
            color = colors[idx % len(colors)]
            alpha = 0.3 if density_factor < 0.5 else 0.6
            label = "Training data" if idx == 0 else None
            ax.scatter(x_train_flat[region_mask_train], y_train_flat[region_mask_train], 
                      alpha=alpha, s=10, color=color, label=label, zorder=3)
    
    # Add vertical lines for region boundaries
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        for ax in axes:
            ax.axvline(x=region_tuple[0], color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=region_tuple[1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot predictive mean and uncertainties for each region
    pred_mean_plotted = False
    tot_unc_plotted = False
    ale_unc_plotted = False
    epi_unc_plotted = False
    
    for idx, (region_tuple, density_factor) in enumerate(sampling_regions):
        region_mask = region_masks[idx]
        color = colors[idx % len(colors)]
        linestyle = '--' if density_factor < 0.5 else '-'
        linewidth = 1.5 if density_factor < 0.5 else 2
        
        # Plot 0: Total entropy (normalized)
        if not pred_mean_plotted:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
            pred_mean_plotted = True
        else:
            axes[0].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not tot_unc_plotted:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - band_width_tot[region_mask], 
                                mu_pred[region_mask] + band_width_tot[region_mask], 
                                alpha=0.3, color=color, label="±norm(entropy, total)")
            tot_unc_plotted = True
        else:
            axes[0].fill_between(x[region_mask], mu_pred[region_mask] - band_width_tot[region_mask], 
                                mu_pred[region_mask] + band_width_tot[region_mask], 
                                alpha=0.3, color=color)
        
        # Plot 1: Aleatoric entropy (normalized)
        if idx == 0:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[1].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not ale_unc_plotted:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - band_width_ale[region_mask], 
                                mu_pred[region_mask] + band_width_ale[region_mask], 
                                alpha=0.3, color=color, label="±norm(entropy, aleatoric)")
            ale_unc_plotted = True
        else:
            axes[1].fill_between(x[region_mask], mu_pred[region_mask] - band_width_ale[region_mask], 
                                mu_pred[region_mask] + band_width_ale[region_mask], 
                                alpha=0.3, color=color)
        
        # Plot 2: Epistemic entropy (normalized)
        if idx == 0:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color, label="Predictive mean")
        else:
            axes[2].plot(x[region_mask], mu_pred[region_mask], linestyle=linestyle, 
                        linewidth=linewidth, color=color)
        
        if not epi_unc_plotted:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - band_width_epi[region_mask], 
                                mu_pred[region_mask] + band_width_epi[region_mask], 
                                alpha=0.3, color=color, label="±norm(entropy, epistemic)")
            epi_unc_plotted = True
        else:
            axes[2].fill_between(x[region_mask], mu_pred[region_mask] - band_width_epi[region_mask], 
                                mu_pred[region_mask] + band_width_epi[region_mask], 
                                alpha=0.3, color=color)
    
    # Plot clean function
    for ax in axes:
        ax.plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
    
    # Set labels and titles
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Total Entropy (Normalized)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Entropy (Normalized)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title} ({noise_type.capitalize()}): Predictive Mean + Epistemic Entropy (Normalized)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with organized folder structure
    subfolder = f"uncertainties_entropy_undersampling_normalized/{noise_type}/{func_type}" if func_type else f"uncertainties_entropy_undersampling_normalized/{noise_type}"
    save_plot(fig, f"{title}_entropy_normalized", subfolder=subfolder)
    
    # Display plot in notebook
    if _ipython_available:
        display(fig)
    plt.show()
    plt.close(fig)