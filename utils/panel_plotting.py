"""
Panel plotting functions for all experiment types.

This module creates multi-subplot panel figures that combine plots across
different parameter values (tau, percentage, regions) for easier comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.results_save import save_plot, sanitize_filename
from utils.noise_level_experiments import _accumulated_plot_data as _accumulated_noise_level_plot_data
from utils.sample_size_experiments import _accumulated_sample_size_plot_data
from utils.ood_experiments import _accumulated_ood_plot_data
from utils.undersampling_experiments import _accumulated_undersampling_plot_data


def create_tau_panel_plot(
    plot_data_dict,  # {tau: plot_data}
    plot_type='variance',
    model_name='',
    func_type='',
    noise_type='heteroscedastic',
    distribution='normal',
    tau_values=None
):
    """
    Create a panel plot showing all tau values for a given model/function/distribution.
    
    Each subplot shows one tau value. Layout depends on plot_type:
    - variance: 3 subplots per tau (Total, Aleatoric, Epistemic)
    - entropy: 4 subplots per tau (Mean, Total, Aleatoric, Epistemic)
    - normalized_variance: 3 subplots per tau (normalized bands)
    - normalized_entropy: 4 subplots per tau (normalized bands)
    - entropy_lines: 3 subplots per tau (Total, Aleatoric, Epistemic entropy lines)
    
    Args:
        plot_data_dict: Dictionary mapping tau values to plot data dictionaries
        plot_type: Type of plot ('variance', 'entropy', 'normalized_variance', 'normalized_entropy', 'entropy_lines')
        model_name: Model name for title
        func_type: Function type for title
        noise_type: Noise type for title
        distribution: Distribution type for title
        tau_values: Optional list of tau values to include (if None, uses sorted keys from dict)
    
    Returns:
        matplotlib Figure object
    """
    if tau_values is None:
        tau_values = sorted([t for t in plot_data_dict.keys() if t is not None])
    
    if not tau_values:
        return None
    
    n_tau = len(tau_values)
    n_cols = min(3, n_tau)  # Max 3 columns
    n_rows = (n_tau + n_cols - 1) // n_cols
    
    # Determine subplots per tau based on plot type
    if plot_type == 'variance':
        n_subplots_per_tau = 3  # Total, Aleatoric, Epistemic
    elif plot_type == 'entropy':
        n_subplots_per_tau = 4  # Mean, Total, Aleatoric, Epistemic
    elif plot_type == 'normalized_variance':
        n_subplots_per_tau = 3  # Total, Aleatoric, Epistemic (normalized)
    elif plot_type == 'normalized_entropy':
        n_subplots_per_tau = 4  # Mean, Total, Aleatoric, Epistemic (normalized)
    elif plot_type == 'entropy_lines':
        n_subplots_per_tau = 3  # Total, Aleatoric, Epistemic entropy lines
    else:
        n_subplots_per_tau = 3  # Default
    
    total_rows = n_rows * n_subplots_per_tau
    fig, axes = plt.subplots(total_rows, n_cols, 
                            figsize=(6*n_cols, 4*total_rows),
                            sharex=True, squeeze=False)
    
    axes = axes.flatten()
    
    for idx, tau in enumerate(tau_values):
        if tau not in plot_data_dict:
            continue
        
        data = plot_data_dict[tau]
        
        # Extract and flatten arrays
        x_grid = data['x_grid']
        x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
        
        x_train = data['x_train']
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
        
        y_train = data['y_train']
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        
        y_grid_clean = data['y_grid_clean']
        y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean
        
        mu_pred_flat = data['mu_pred']
        
        if plot_type == 'variance':
            # Row 1: Total uncertainty
            ax_idx = idx * n_subplots_per_tau
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - np.sqrt(data['tot_var']), 
                              mu_pred_flat + np.sqrt(data['tot_var']), 
                              alpha=0.3, color='blue', label="±σ(total)")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"τ={tau} - Total Variance")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Row 2: Aleatoric
            if ax_idx + 1 < len(axes):
                ax = axes[ax_idx + 1]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - np.sqrt(data['ale_var']), 
                              mu_pred_flat + np.sqrt(data['ale_var']), 
                              alpha=0.3, color='green', label="±σ(aleatoric)")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"τ={tau} - Aleatoric Variance")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Row 3: Epistemic
            if ax_idx + 2 < len(axes):
                ax = axes[ax_idx + 2]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - np.sqrt(data['epi_var']), 
                              mu_pred_flat + np.sqrt(data['epi_var']), 
                              alpha=0.3, color='orange', label="±σ(epistemic)")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_xlabel("x")
                ax.set_title(f"τ={tau} - Epistemic Variance")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
        
        elif plot_type == 'entropy':
            ale_entropy_flat = data['ale_entropy'].squeeze()
            epi_entropy_flat = data['epi_entropy'].squeeze()
            tot_entropy_flat = data['tot_entropy'].squeeze()

            sqrt_2pi_e = np.sqrt(2 * np.pi * np.e)
            tot_std_equiv = np.exp(tot_entropy_flat) / sqrt_2pi_e
            ale_std_equiv = np.exp(ale_entropy_flat) / sqrt_2pi_e
            epi_std_equiv = np.exp(epi_entropy_flat) / sqrt_2pi_e

            # Row 1: Predictive mean
            ax_idx = idx * n_subplots_per_tau
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"τ={tau} - Predictive Mean")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Row 2: Total entropy (as bands)
            if ax_idx + 1 < len(axes):
                ax = axes[ax_idx + 1]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - tot_std_equiv, mu_pred_flat + tot_std_equiv, 
                                alpha=0.3, color='blue', label="±σ(total, from entropy)")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"τ={tau} - Total Entropy (as ±σ)")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Row 3: Aleatoric entropy (as bands)
            if ax_idx + 2 < len(axes):
                ax = axes[ax_idx + 2]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - ale_std_equiv, mu_pred_flat + ale_std_equiv, 
                                alpha=0.3, color='green', label="±σ(aleatoric, from entropy)")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"τ={tau} - Aleatoric Entropy (as ±σ)")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Row 4: Epistemic entropy (as bands)
            if ax_idx + 3 < len(axes):
                ax = axes[ax_idx + 3]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - epi_std_equiv, mu_pred_flat + epi_std_equiv, 
                                alpha=0.3, color='orange', label="±σ(epistemic, from entropy)")
                ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_xlabel("x")
                ax.set_title(f"τ={tau} - Epistemic Entropy (as ±σ)")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
        
        elif plot_type == 'entropy_lines':
            ale_entropy_flat = data['ale_entropy'].squeeze()
            epi_entropy_flat = data['epi_entropy'].squeeze()
            tot_entropy_flat = data['tot_entropy'].squeeze()

            # Plot 1: Total entropy as line plot
            ax_idx = idx * n_subplots_per_tau
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', label="Training data", zorder=3)
                ax1_twin = ax.twinx()
                ax.plot(x, mu_pred_flat, 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
                ax1_twin.plot(x, tot_entropy_flat, 'g-', linewidth=2, label="Total entropy (nats)")
                ax.plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
                ax.set_ylabel("y / Predictive mean")
                ax1_twin.set_ylabel("Entropy (nats)", color='green')
                ax1_twin.tick_params(axis='y', labelcolor='green')
                ax.set_title(f"τ={tau} - Total Entropy (nats)")
                ax.legend(loc="upper left", fontsize=8)
                ax1_twin.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Plot 2: Aleatoric entropy as line plot
            if ax_idx + 1 < len(axes):
                ax = axes[ax_idx + 1]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax2_twin = ax.twinx()
                ax.plot(x, mu_pred_flat, 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
                ax2_twin.plot(x, ale_entropy_flat, 'g-', linewidth=2, label="Aleatoric entropy (nats)")
                ax.plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
                ax.set_ylabel("y / Predictive mean")
                ax2_twin.set_ylabel("Entropy (nats)", color='green')
                ax2_twin.tick_params(axis='y', labelcolor='green')
                ax.set_title(f"τ={tau} - Aleatoric Entropy (nats)")
                ax.legend(loc="upper left", fontsize=8)
                ax2_twin.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Plot 3: Epistemic entropy as line plot
            if ax_idx + 2 < len(axes):
                ax = axes[ax_idx + 2]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax3_twin = ax.twinx()
                ax.plot(x, mu_pred_flat, 'b-', linewidth=1.2, label="Predictive mean", alpha=0.5)
                ax3_twin.plot(x, epi_entropy_flat, 'r-', linewidth=2, label="Epistemic entropy (nats)")
                ax.plot(x, y_clean_flat, 'r-', linewidth=1.5, alpha=0.8, label="Clean function")
                ax.set_ylabel("y / Predictive mean")
                ax.set_xlabel("x")
                ax3_twin.set_ylabel("Entropy (nats)", color='red')
                ax3_twin.tick_params(axis='y', labelcolor='red')
                ax.set_title(f"τ={tau} - Epistemic Entropy (nats)")
                ax.legend(loc="upper left", fontsize=8)
                ax3_twin.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
        
        elif plot_type in ['normalized_variance', 'normalized_entropy']:
            # For normalized plots, skip for now (similar to regular plots)
            # Can be implemented later if needed
            continue
    
    # Hide unused subplots
    for i in range(len(axes)):
        if i >= n_tau * n_subplots_per_tau:
            fig.delaxes(axes[i])
    
    title = f"{model_name} - {func_type.capitalize()} Function ({noise_type}, {distribution}) - {plot_type.replace('_', ' ').capitalize()}"
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def create_pct_panel_plot(
    plot_data_dict,  # {pct: plot_data}
    plot_type='variance',
    model_name='',
    func_type='',
    noise_type='heteroscedastic',
    pct_values=None
):
    """
    Create a panel plot showing all percentage values for sample size experiments.
    Similar to create_tau_panel_plot but for percentages.
    """
    if pct_values is None:
        pct_values = sorted([p for p in plot_data_dict.keys() if p is not None])
    
    if not pct_values:
        return None
    
    # Reuse tau panel plot logic but with pct labels
    # Convert pct to tau-like structure for reuse
    tau_like_dict = {pct: plot_data_dict[pct] for pct in pct_values}
    
    fig = create_tau_panel_plot(
        tau_like_dict,
        plot_type=plot_type,
        model_name=model_name,
        func_type=func_type,
        noise_type=noise_type,
        distribution='',  # Not used for sample size
        tau_values=pct_values
    )
    
    if fig is not None:
        # Update titles to show percentages instead of tau
        for idx, pct in enumerate(pct_values):
            if plot_type == 'variance':
                n_subplots_per_tau = 3
            elif plot_type == 'entropy':
                n_subplots_per_tau = 4
            elif plot_type == 'entropy_lines':
                n_subplots_per_tau = 3
            else:
                n_subplots_per_tau = 3
            
            ax_idx = idx * n_subplots_per_tau
            for i in range(n_subplots_per_tau):
                if ax_idx + i < len(fig.axes):
                    ax = fig.axes[ax_idx + i]
                    title = ax.get_title()
                    title = title.replace(f'τ={pct}', f'{pct}%')
                    ax.set_title(title)
        
        # Update main title
        title = f"{model_name} - {func_type.capitalize()} Function ({noise_type}) - {plot_type.replace('_', ' ').capitalize()}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    return fig


def create_region_panel_plot(
    plot_data_dict,  # {region_key: plot_data}
    plot_type='variance',
    model_name='',
    func_type='',
    noise_type='heteroscedastic',
    region_keys=None,
    experiment_type='ood'  # 'ood' or 'undersampling'
):
    """
    Create a panel plot showing different regions for OOD or undersampling experiments.
    """
    if region_keys is None:
        region_keys = sorted([k for k in plot_data_dict.keys() if k is not None])
    
    if not region_keys:
        return None
    
    n_regions = len(region_keys)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    # Determine subplots per region based on plot type
    if plot_type == 'variance':
        n_subplots_per_region = 3
    elif plot_type == 'entropy':
        n_subplots_per_region = 4
    elif plot_type == 'entropy_lines':
        n_subplots_per_region = 3
    else:
        n_subplots_per_region = 3
    
    total_rows = n_rows * n_subplots_per_region
    fig, axes = plt.subplots(total_rows, n_cols, 
                            figsize=(6*n_cols, 4*total_rows),
                            sharex=True, squeeze=False)
    
    axes = axes.flatten()
    
    for idx, region_key in enumerate(region_keys):
        if region_key not in plot_data_dict:
            continue
        
        data = plot_data_dict[region_key]
        
        # Extract and flatten arrays
        x_grid = data['x_grid']
        x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
        
        x_train = data['x_train']
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
        
        y_train = data['y_train']
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        
        y_grid_clean = data['y_grid_clean']
        y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean
        
        mu_pred_flat = data['mu_pred']
        
        # Format region label
        if isinstance(region_key, tuple):
            region_label = f"Region {region_key}"
        else:
            region_label = str(region_key)
        
        if plot_type == 'variance':
            ax_idx = idx * n_subplots_per_region
            
            # Total
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - np.sqrt(data['tot_var']), 
                              mu_pred_flat + np.sqrt(data['tot_var']), 
                              alpha=0.3, color='blue', label="±σ(total)")
                if experiment_type == 'ood' and data.get('ood_mask') is not None:
                    ood_mask = data['ood_mask']
                    id_mask = ~ood_mask
                    ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=1.5, alpha=0.8, label="True function")
                    if np.any(ood_mask):
                        ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=1.5, alpha=0.8)
                        ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=8, color='red', alpha=0.3, marker='x', zorder=4)
                else:
                    ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"{region_label} - Total Variance")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Aleatoric
            if ax_idx + 1 < len(axes):
                ax = axes[ax_idx + 1]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - np.sqrt(data['ale_var']), 
                              mu_pred_flat + np.sqrt(data['ale_var']), 
                              alpha=0.3, color='green', label="±σ(aleatoric)")
                if experiment_type == 'ood' and data.get('ood_mask') is not None:
                    ood_mask = data['ood_mask']
                    id_mask = ~ood_mask
                    ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=1.5, alpha=0.8, label="True function")
                    if np.any(ood_mask):
                        ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=1.5, alpha=0.8)
                        ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=8, color='red', alpha=0.3, marker='x', zorder=4)
                else:
                    ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_title(f"{region_label} - Aleatoric Variance")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Epistemic
            if ax_idx + 2 < len(axes):
                ax = axes[ax_idx + 2]
                ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color='blue', zorder=3)
                ax.plot(x, mu_pred_flat, 'b-', linewidth=2, label="Predictive mean")
                ax.fill_between(x, mu_pred_flat - np.sqrt(data['epi_var']), 
                              mu_pred_flat + np.sqrt(data['epi_var']), 
                              alpha=0.3, color='orange', label="±σ(epistemic)")
                if experiment_type == 'ood' and data.get('ood_mask') is not None:
                    ood_mask = data['ood_mask']
                    id_mask = ~ood_mask
                    ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=1.5, alpha=0.8, label="True function")
                    if np.any(ood_mask):
                        ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=1.5, alpha=0.8)
                        ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=8, color='red', alpha=0.3, marker='x', zorder=4)
                else:
                    ax.plot(x, y_clean_flat, 'r--', linewidth=1.5, alpha=0.8, label="True function")
                ax.set_ylabel("y")
                ax.set_xlabel("x")
                ax.set_title(f"{region_label} - Epistemic Variance")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
        
        # Similar logic for entropy and entropy_lines can be added here
    
    # Hide unused subplots
    for i in range(len(axes)):
        if i >= n_regions * n_subplots_per_region:
            fig.delaxes(axes[i])
    
    title = f"{model_name} - {func_type.capitalize()} Function ({noise_type}) - {plot_type.replace('_', ' ').capitalize()}"
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def save_all_panel_plots(date=None, subfolder='noise_level'):
    """
    Generate and save all panel plots from accumulated plot data for noise level experiments.
    Called after all experiments are complete.
    
    Args:
        date: Optional date string in YYYYMMDD format. If None, uses current date.
        subfolder: Subfolder path for saving plots
    """
    from datetime import datetime
    
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    if not _accumulated_noise_level_plot_data:
        print("No plot data accumulated for noise level panel plots.")
        return
    
    print(f"\n{'='*80}")
    print("Generating noise level panel plots...")
    print(f"{'='*80}\n")
    
    saved_count = 0
    for (noise_type, distribution), plot_dict in _accumulated_noise_level_plot_data.items():
        # Group by (model_name, func_type, plot_type)
        for (model_name, func_type, plot_type), tau_data in plot_dict.items():
            if not tau_data:
                continue
            
            try:
                # Create tau panel plot
                fig = create_tau_panel_plot(
                    tau_data,
                    plot_type=plot_type,
                    model_name=model_name,
                    func_type=func_type,
                    noise_type=noise_type,
                    distribution=distribution
                )
                
                if fig is None:
                    continue
                
                filename = f"{date}_panel_{model_name}_{func_type}_{plot_type}_{noise_type}_{distribution}"
                filename = sanitize_filename(filename)
                
                save_plot(fig, filename, subfolder=f"{subfolder}/panels/{noise_type}/{func_type}")
                plt.close(fig)
                
                saved_count += 1
                print(f"  Saved panel plot: {filename}")
                
            except Exception as e:
                print(f"  Error creating panel plot for {model_name}/{func_type}/{plot_type}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Finished generating noise level panel plots. Saved {saved_count} panel plots.")
    print(f"{'='*80}\n")


def save_all_sample_size_panel_plots(date=None, subfolder='sample_size'):
    """
    Generate and save all panel plots for sample size experiments.
    """
    from datetime import datetime
    
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    if not _accumulated_sample_size_plot_data:
        print("No plot data accumulated for sample size panel plots.")
        return
    
    print(f"\n{'='*80}")
    print("Generating sample size panel plots...")
    print(f"{'='*80}\n")
    
    saved_count = 0
    for noise_type, plot_dict in _accumulated_sample_size_plot_data.items():
        for (model_name, func_type, plot_type), pct_data in plot_dict.items():
            if not pct_data:
                continue
            
            try:
                fig = create_pct_panel_plot(
                    pct_data,
                    plot_type=plot_type,
                    model_name=model_name,
                    func_type=func_type,
                    noise_type=noise_type
                )
                
                if fig is None:
                    continue
                
                filename = f"{date}_panel_{model_name}_{func_type}_{plot_type}_{noise_type}"
                filename = sanitize_filename(filename)
                
                save_plot(fig, filename, subfolder=f"{subfolder}/panels/{noise_type}/{func_type}")
                plt.close(fig)
                
                saved_count += 1
                print(f"  Saved panel plot: {filename}")
                
            except Exception as e:
                print(f"  Error creating panel plot for {model_name}/{func_type}/{plot_type}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Finished generating sample size panel plots. Saved {saved_count} panel plots.")
    print(f"{'='*80}\n")


def create_model_comparison_panel_plot(
    model_name='',
    model_data=None,  # {'variance': plot_data_dict, 'entropy': plot_data_dict}
    func_type='',
    noise_type='heteroscedastic'
):
    """
    Create a panel plot for a single model showing variance and entropy side-by-side.
    
    Layout:
    - Columns: 2 plot types (Variance, Entropy)
    - Rows: 3 subplots stacked vertically (Total, Aleatoric, Epistemic)
    
    Args:
        model_name: Model name (e.g., 'MC_Dropout', 'Deep_Ensemble', 'BNN', 'BAMLSS')
        model_data: Dictionary {'variance': plot_data_dict, 'entropy': plot_data_dict}
                    where plot_data_dict is {region_key: plot_data}
        func_type: Function type ('linear' or 'sin')
        noise_type: Noise type ('heteroscedastic' or 'homoscedastic')
    
    Returns:
        matplotlib Figure object
    """
    if model_data is None:
        return None
    
    n_cols = 2  # Variance and Entropy
    n_rows = 3  # Total, Aleatoric, Epistemic
    
    # Create figure with GridSpec for better control - much larger height
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(18, 12))  # Increased width and height
    gs = GridSpec(n_rows, n_cols, 
                  figure=fig, hspace=0.35, wspace=0.25,
                  left=0.08, right=0.96, top=0.94, bottom=0.06)
    
    # Helper function to get region data (prefer 'Combined', otherwise first available)
    def get_region_data(plot_data_dict):
        if 'Combined' in plot_data_dict:
            return plot_data_dict['Combined']
        elif plot_data_dict:
            return list(plot_data_dict.values())[0]
        return None
    
    # Get variance and entropy data
    variance_data = get_region_data(model_data.get('variance', {}))
    entropy_data = get_region_data(model_data.get('entropy', {}))
    
    # Plot variance column
    if variance_data is not None:
        # Extract arrays
        x_grid = variance_data['x_grid']
        x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
        
        x_train = variance_data['x_train']
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
        
        y_train = variance_data['y_train']
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        
        y_grid_clean = variance_data['y_grid_clean']
        y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean
        
        mu_pred_flat = variance_data['mu_pred']
        ood_mask = variance_data.get('ood_mask')
        
        # Total Variance
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color='#2E86AB', zorder=3, edgecolors='none')
        ax.plot(x, mu_pred_flat, 'b-', linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(x, mu_pred_flat - np.sqrt(variance_data['tot_var']), 
                      mu_pred_flat + np.sqrt(variance_data['tot_var']), 
                      alpha=0.35, color='#2E86AB', label="±σ(total)", zorder=1)
        if ood_mask is not None:
            id_mask = ~ood_mask
            ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=20, color='red', alpha=0.4, marker='x', zorder=6, linewidths=1.5)
        else:
            ax.plot(x, y_clean_flat, 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        ax.set_title("Variance - Total", fontweight='bold', fontsize=13, pad=10)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=10)
        
        # Aleatoric Variance
        ax = fig.add_subplot(gs[1, 0])
        ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color='#2E86AB', zorder=3, edgecolors='none')
        ax.plot(x, mu_pred_flat, 'b-', linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(x, mu_pred_flat - np.sqrt(variance_data['ale_var']), 
                      mu_pred_flat + np.sqrt(variance_data['ale_var']), 
                      alpha=0.35, color='#06A77D', label="±σ(aleatoric)", zorder=1)
        if ood_mask is not None:
            id_mask = ~ood_mask
            ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=20, color='red', alpha=0.4, marker='x', zorder=6, linewidths=1.5)
        else:
            ax.plot(x, y_clean_flat, 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        ax.set_title("Variance - Aleatoric", fontweight='bold', fontsize=13, pad=10)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=10)
        
        # Epistemic Variance
        ax = fig.add_subplot(gs[2, 0])
        ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color='#2E86AB', zorder=3, edgecolors='none')
        ax.plot(x, mu_pred_flat, 'b-', linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(x, mu_pred_flat - np.sqrt(variance_data['epi_var']), 
                      mu_pred_flat + np.sqrt(variance_data['epi_var']), 
                      alpha=0.35, color='#F18F01', label="±σ(epistemic)", zorder=1)
        if ood_mask is not None:
            id_mask = ~ood_mask
            ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=20, color='red', alpha=0.4, marker='x', zorder=6, linewidths=1.5)
        else:
            ax.plot(x, y_clean_flat, 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
        ax.set_xlabel("x", fontsize=12, fontweight='bold')
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        ax.set_title("Variance - Epistemic", fontweight='bold', fontsize=13, pad=10)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=10)
    
    # Plot entropy column
    if entropy_data is not None and 'ale_entropy' in entropy_data:
        # Extract arrays
        x_grid = entropy_data['x_grid']
        x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
        
        x_train = entropy_data['x_train']
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
        
        y_train = entropy_data['y_train']
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train
        
        y_grid_clean = entropy_data['y_grid_clean']
        y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean
        
        mu_pred_flat = entropy_data['mu_pred']
        ood_mask = entropy_data.get('ood_mask')
        
        ale_entropy_flat = entropy_data['ale_entropy'].squeeze()
        epi_entropy_flat = entropy_data['epi_entropy'].squeeze()
        tot_entropy_flat = entropy_data['tot_entropy'].squeeze()
        
        sqrt_2pi_e = np.sqrt(2 * np.pi * np.e)
        tot_std_equiv = np.exp(tot_entropy_flat) / sqrt_2pi_e
        ale_std_equiv = np.exp(ale_entropy_flat) / sqrt_2pi_e
        epi_std_equiv = np.exp(epi_entropy_flat) / sqrt_2pi_e
        
        # Total Entropy
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color='#2E86AB', zorder=3, edgecolors='none')
        ax.plot(x, mu_pred_flat, 'b-', linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(x, mu_pred_flat - tot_std_equiv, mu_pred_flat + tot_std_equiv, 
                      alpha=0.35, color='#2E86AB', label="±σ(total, from entropy)", zorder=1)
        if ood_mask is not None:
            id_mask = ~ood_mask
            ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=20, color='red', alpha=0.4, marker='x', zorder=6, linewidths=1.5)
        else:
            ax.plot(x, y_clean_flat, 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        ax.set_title("Entropy - Total", fontweight='bold', fontsize=13, pad=10)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=10)
        
        # Aleatoric Entropy
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color='#2E86AB', zorder=3, edgecolors='none')
        ax.plot(x, mu_pred_flat, 'b-', linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(x, mu_pred_flat - ale_std_equiv, mu_pred_flat + ale_std_equiv, 
                      alpha=0.35, color='#06A77D', label="±σ(aleatoric, from entropy)", zorder=1)
        if ood_mask is not None:
            id_mask = ~ood_mask
            ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=20, color='red', alpha=0.4, marker='x', zorder=6, linewidths=1.5)
        else:
            ax.plot(x, y_clean_flat, 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        ax.set_title("Entropy - Aleatoric", fontweight='bold', fontsize=13, pad=10)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=10)
        
        # Epistemic Entropy
        ax = fig.add_subplot(gs[2, 1])
        ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color='#2E86AB', zorder=3, edgecolors='none')
        ax.plot(x, mu_pred_flat, 'b-', linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(x, mu_pred_flat - epi_std_equiv, mu_pred_flat + epi_std_equiv, 
                      alpha=0.35, color='#F18F01', label="±σ(epistemic, from entropy)", zorder=1)
        if ood_mask is not None:
            id_mask = ~ood_mask
            ax.plot(x[id_mask], y_clean_flat[id_mask], 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y_clean_flat[ood_mask], 'r--', linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=20, color='red', alpha=0.4, marker='x', zorder=6, linewidths=1.5)
        else:
            ax.plot(x, y_clean_flat, 'r--', linewidth=2, alpha=0.9, label="True function", zorder=4)
        ax.set_xlabel("x", fontsize=12, fontweight='bold')
        ax.set_ylabel("y", fontsize=12, fontweight='bold')
        ax.set_title("Entropy - Epistemic", fontweight='bold', fontsize=13, pad=10)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=10)
    
    # Create a nicer title with model name
    title = f"{model_name} - {func_type.capitalize()} Function ({noise_type})"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def save_all_ood_panel_plots(date=None, subfolder='ood'):
    """
    Generate and save combined model comparison panel plots for OOD experiments.
    Creates one panel plot per (func_type, noise_type) combination showing all models side-by-side.
    """
    from datetime import datetime
    
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    if not _accumulated_ood_plot_data:
        print("No plot data accumulated for OOD panel plots.")
        print(f"Debug: _accumulated_ood_plot_data = {_accumulated_ood_plot_data}")
        print("Note: Make sure you've run the OOD experiments AFTER adding accumulate_plot_data calls.")
        return
    
    print(f"Debug: Found plot data for {len(_accumulated_ood_plot_data)} noise type(s)")
    for noise_type, plot_dict in _accumulated_ood_plot_data.items():
        print(f"  - {noise_type}: {len(plot_dict)} plot entries")
    
    print(f"\n{'='*80}")
    print("Generating combined model comparison panel plots...")
    print(f"{'='*80}\n")
    
    # Group data by (func_type, noise_type)
    grouped_data = {}
    for noise_type, plot_dict in _accumulated_ood_plot_data.items():
        for (model_name, func_type, plot_type), region_data in plot_dict.items():
            if not region_data:
                continue
            
            key = (func_type, noise_type)
            if key not in grouped_data:
                grouped_data[key] = {}
            
            if model_name not in grouped_data[key]:
                grouped_data[key][model_name] = {}
            
            grouped_data[key][model_name][plot_type] = region_data
    
    saved_count = 0
    # Create individual plots for each model and (func_type, noise_type) combination
    for (func_type, noise_type), models_dict in grouped_data.items():
        # Create a plot for each model
        for model_name, model_data in models_dict.items():
            # Check if we have both variance and entropy for this model
            has_variance = 'variance' in model_data
            has_entropy = 'entropy' in model_data
            
            if not (has_variance and has_entropy):
                print(f"  Skipping {model_name}/{func_type}/{noise_type}: missing variance or entropy data")
                continue
            
            try:
                fig = create_model_comparison_panel_plot(
                    model_name=model_name,
                    model_data=model_data,
                    func_type=func_type,
                    noise_type=noise_type
                )
                
                if fig is None:
                    continue
                
                filename = f"{date}_panel_{model_name}_{func_type}_{noise_type}"
                filename = sanitize_filename(filename)
                
                save_plot(fig, filename, subfolder=f"{subfolder}/panels/{noise_type}/{func_type}")
                plt.close(fig)
                
                saved_count += 1
                print(f"  Saved panel plot: {filename}")
                
            except Exception as e:
                print(f"  Error creating panel plot for {model_name}/{func_type}/{noise_type}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Finished generating combined model comparison panel plots. Saved {saved_count} panel plots.")
    print(f"{'='*80}\n")


def save_all_undersampling_panel_plots(date=None, subfolder='undersampling'):
    """
    Generate and save all panel plots for undersampling experiments.
    """
    from datetime import datetime
    
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    if not _accumulated_undersampling_plot_data:
        print("No plot data accumulated for undersampling panel plots.")
        return
    
    print(f"\n{'='*80}")
    print("Generating undersampling panel plots...")
    print(f"{'='*80}\n")
    
    saved_count = 0
    for noise_type, plot_dict in _accumulated_undersampling_plot_data.items():
        for (model_name, func_type, plot_type), region_data in plot_dict.items():
            if not region_data:
                continue
            
            try:
                fig = create_region_panel_plot(
                    region_data,
                    plot_type=plot_type,
                    model_name=model_name,
                    func_type=func_type,
                    noise_type=noise_type,
                    experiment_type='undersampling'
                )
                
                if fig is None:
                    continue
                
                filename = f"{date}_panel_{model_name}_{func_type}_{plot_type}_{noise_type}"
                filename = sanitize_filename(filename)
                
                save_plot(fig, filename, subfolder=f"{subfolder}/panels/{noise_type}/{func_type}")
                plt.close(fig)
                
                saved_count += 1
                print(f"  Saved panel plot: {filename}")
                
            except Exception as e:
                print(f"  Error creating panel plot for {model_name}/{func_type}/{plot_type}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Finished generating undersampling panel plots. Saved {saved_count} panel plots.")
    print(f"{'='*80}\n")
