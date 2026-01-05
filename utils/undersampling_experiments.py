"""
Helper functions for running spatial undersampling experiments across different models.

This module provides functions to run undersampling experiments for various
uncertainty quantification models, handling the common pattern of:
1. Generating non-uniformly distributed training data
2. Training models on spatially undersampled data
3. Evaluating uncertainties across different regions
4. Computing and saving statistics separately for each region
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing
from datetime import datetime

from utils.results_save import save_summary_statistics_undersampling, save_model_outputs
from utils.plotting import plot_uncertainties_undersampling
from utils.entropy_uncertainty import entropy_uncertainty_analytical, entropy_uncertainty_numerical
from utils.device import get_device_for_worker, get_num_gpus
from utils.metrics import (
    compute_predictive_aggregation,
    compute_gaussian_nll,
    compute_crps_gaussian,
    compute_true_noise_variance,
    compute_uncertainty_disentanglement
)

# Set multiprocessing start method for CUDA support (Linux: can use 'fork' or 'spawn')
# Using 'spawn' for better CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


# ========== Data Generation Helper ==========

def generate_data_with_undersampling(generate_toy_regression_func, n_train, train_range, 
                                   sampling_regions, grid_points, noise_type, func_type, 
                                   seed=42, **kwargs):
    """
    Generate training data with non-uniform spatial sampling.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data
    n_train : int
        Total number of training samples
    train_range : tuple
        (min, max) range for training data
    sampling_regions : list of tuples
        List of (region_tuple, density_factor) where:
        - region_tuple: (min, max) range for this region
        - density_factor: float (0.0-1.0+) representing relative sampling density
    grid_points : int
        Number of grid points for evaluation
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str
        Function type identifier (e.g., 'linear', 'sin')
    seed : int
        Random seed
    **kwargs : dict
        Additional arguments to pass to generate_toy_regression_func
    
    Returns:
    --------
    tuple: (x_train, y_train, x_grid, y_grid_clean, region_masks)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_min, train_max = train_range
    train_width = train_max - train_min
    
    # Calculate relative widths and weighted densities for each region
    region_info = []
    total_weighted_width = 0.0
    
    for region_tuple, density_factor in sampling_regions:
        region_min, region_max = region_tuple
        region_width = max(0, min(region_max, train_max) - max(region_min, train_min))
        weighted_width = region_width * density_factor
        region_info.append({
            'region': region_tuple,
            'density': density_factor,
            'width': region_width,
            'weighted_width': weighted_width
        })
        total_weighted_width += weighted_width
    
    # Allocate samples to each region proportionally
    x_train_list = []
    y_train_list = []
    
    for info in region_info:
        if info['weighted_width'] > 0 and total_weighted_width > 0:
            # Calculate number of samples for this region
            n_samples_region = int(n_train * info['weighted_width'] / total_weighted_width)
            n_samples_region = max(1, n_samples_region)  # At least 1 sample
            
            # Generate samples in this region
            region_min, region_max = info['region']
            region_min_clipped = max(region_min, train_min)
            region_max_clipped = min(region_max, train_max)
            
            if region_max_clipped > region_min_clipped:
                x_region = np.random.uniform(region_min_clipped, region_max_clipped, 
                                            size=(n_samples_region, 1))
                
                # Generate y values directly using the function
                if func_type == "linear":
                    f_clean = lambda x: 0.7 * x + 0.5
                elif func_type == "sin":
                    f_clean = lambda x: x * np.sin(x) + x
                else:
                    raise ValueError(f"Unknown func_type: {func_type}")
                
                y_clean_region = f_clean(x_region)
                
                # Add noise
                if noise_type == 'homoscedastic':
                    sigma = 2
                    sigma_region = np.full_like(x_region, sigma)
                elif noise_type == 'heteroscedastic':
                    sigma_region = np.abs(2.5 * np.sin(0.5*x_region +5))
                else:
                    raise ValueError("noise_type must be 'homoscedastic' or 'heteroscedastic'")
                
                epsilon = np.random.normal(0.0, sigma_region, size=(n_samples_region, 1))
                y_region = y_clean_region + epsilon
                
                x_train_list.append(x_region)
                y_train_list.append(y_region)
    
    # Combine all regions
    if x_train_list:
        x_train = np.vstack(x_train_list)
        y_train = np.vstack(y_train_list)
        
        # Shuffle the combined data
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]
    else:
        # Fallback: uniform sampling
        x_train, y_train, _, _ = generate_toy_regression_func(
            n_train=n_train,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            type=func_type,
            **kwargs
        )
    
    # Generate evaluation grid spanning entire training range
    x_grid = np.linspace(train_min, train_max, grid_points).reshape(-1, 1)
    
    # Generate clean function values for the grid
    if func_type == "linear":
        f_clean = lambda x: 0.7 * x + 0.5
    elif func_type == "sin":
        f_clean = lambda x: x * np.sin(x) + x
    else:
        raise ValueError(f"Unknown func_type: {func_type}")
    
    y_grid_clean = f_clean(x_grid)
    
    # Create region masks for grid points
    region_masks = []
    for region_tuple, _ in sampling_regions:
        region_min, region_max = region_tuple
        mask = (x_grid[:, 0] >= region_min) & (x_grid[:, 0] <= region_max)
        region_masks.append(mask)
    
    return (x_train.astype(np.float32), y_train.astype(np.float32),
            x_grid.astype(np.float32), y_grid_clean.astype(np.float32), region_masks)


# ========== Parallel execution wrapper functions ==========
# These functions are top-level to be picklable for multiprocessing

def _train_single_undersampling_mc_dropout(args):
    """Wrapper function for training MC Dropout for undersampling experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, region_masks,
     seed, p, beta, epochs, lr, batch_size, mc_samples, func_type, noise_type, entropy_method) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    torch.cuda.set_device(device) if device.type == 'cuda' else None
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    # Create dataloader
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Train model
    model = MCDropoutRegressor(p=p)
    train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
    
    # Make predictions with raw arrays
    result = mc_dropout_predict(model, x_grid, M=mc_samples, return_raw_arrays=True)
    mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
    
    # Compute MSE for each region
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    mse_by_region = []
    for mask in region_masks:
        if np.any(mask):
            mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
        else:
            mse = 0.0
        mse_by_region.append(mse)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_by_region, mu_samples, sigma2_samples


def _train_single_undersampling_deep_ensemble(args):
    """Wrapper function for training Deep Ensemble for undersampling experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, region_masks,
     seed, beta, batch_size, K, epochs, func_type, noise_type, entropy_method) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    torch.cuda.set_device(device) if device.type == 'cuda' else None
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from Models.Deep_Ensemble import (
        train_ensemble_deep,
        ensemble_predict_deep
    )
    from Models.MC_Dropout import (
        normalize_x,
        normalize_x_data
    )
    
    # Normalize input
    x_mean, x_std = normalize_x(x_train)
    x_train_norm = normalize_x_data(x_train, x_mean, x_std)
    x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
    
    # Train ensemble
    ensemble = train_ensemble_deep(
        x_train_norm, y_train,
        batch_size=batch_size, K=K,
        loss_type='beta_nll', beta=beta, parallel=True, epochs=epochs
    )
    
    # Make predictions with raw arrays
    result = ensemble_predict_deep(ensemble, x_grid_norm, return_raw_arrays=True)
    mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
    
    # Compute MSE for each region
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    mse_by_region = []
    for mask in region_masks:
        if np.any(mask):
            mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
        else:
            mse = 0.0
        mse_by_region.append(mse)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_by_region, mu_samples, sigma2_samples


def _train_single_undersampling_bnn(args):
    """Wrapper function for training BNN for undersampling experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, region_masks,
     seed, hidden_width, weight_scale, warmup, samples, chains, func_type, noise_type, entropy_method) = args
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from Models.BNN import (
        train_bnn,
        bnn_predict,
        normalize_x as bnn_normalize_x,
        normalize_x_data as bnn_normalize_x_data
    )
    
    # Normalize input
    x_mean, x_std = bnn_normalize_x(x_train)
    x_train_norm = bnn_normalize_x_data(x_train, x_mean, x_std)
    x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
    
    # Train BNN with MCMC (uses CPU internally)
    mcmc = train_bnn(
        x_train_norm, y_train,
        hidden_width=hidden_width, weight_scale=weight_scale,
        warmup=warmup, samples=samples, chains=chains, seed=seed
    )
    
    # Make predictions with raw arrays
    result = bnn_predict(
        mcmc, x_grid_norm,
        hidden_width=hidden_width, weight_scale=weight_scale,
        return_raw_arrays=True
    )
    mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
    
    # Compute MSE for each region
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    mse_by_region = []
    for mask in region_masks:
        if np.any(mask):
            mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
        else:
            mse = 0.0
        mse_by_region.append(mse)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_by_region, mu_samples, sigma2_samples


def _train_single_undersampling_bamlss(args):
    """Wrapper function for training BAMLSS for undersampling experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, region_masks,
     seed, n_iter, burnin, thin, nsamples, func_type, noise_type, entropy_method) = args
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from Models.BAMLSS import bamlss_predict
    
    # BAMLSS fits directly - get raw arrays
    result = bamlss_predict(
        x_train, y_train, x_grid,
        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples,
        return_raw_arrays=True
    )
    mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
    
    # Compute MSE for each region
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    mse_by_region = []
    for mask in region_masks:
        if np.any(mask):
            mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
        else:
            mse = 0.0
        mse_by_region.append(mse)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_by_region, mu_samples, sigma2_samples


# ========== Statistics Computation ==========

def compute_and_save_statistics_undersampling(
    uncertainties_by_region: list,
    mse_by_region: list,
    sampling_regions: list,
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    date: str = None,
    dropout_p: float = None,
    mc_samples: int = None,
    n_nets: int = None,
    nll_by_region: list = None,
    crps_by_region: list = None,
    spearman_aleatoric_by_region: list = None,
    spearman_epistemic_by_region: list = None
):
    """
    Shared function to compute normalized statistics and save results for undersampling experiments.
    
    This function normalizes uncertainties, computes averages and correlations,
    prints formatted statistics, and saves results separately for each region.
    
    Parameters:
    -----------
    uncertainties_by_region : list
        List of dictionaries, one per region, each with 'ale', 'epi', 'tot' arrays
    mse_by_region : list
        List of MSE values, one per region
    sampling_regions : list
        List of (region_tuple, density_factor) tuples
    function_name : str
        Human-readable function name (e.g., "Linear", "Sinusoidal")
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str
        Function type identifier (e.g., 'linear', 'sin')
    model_name : str
        Model name for saving results (e.g., 'MC_Dropout', 'Deep_Ensemble')
    date : str, optional
        Date string in YYYYMMDD format
    dropout_p : float, optional
        Dropout probability for MC Dropout
    mc_samples : int, optional
        Number of MC samples for MC Dropout
    n_nets : int, optional
        Number of nets for Deep Ensemble
    
    Returns:
    --------
    dict : Statistics dictionary with results for each region
    """
    def normalize(values, vmin, vmax):
        """Normalize values to [0, 1] range"""
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    # Collect all values for normalization (across all regions)
    all_ale = np.concatenate([unc['ale'] for unc in uncertainties_by_region])
    all_epi = np.concatenate([unc['epi'] for unc in uncertainties_by_region])
    all_tot = np.concatenate([unc['tot'] for unc in uncertainties_by_region])
    
    # Compute min/max for normalization
    ale_min, ale_max = all_ale.min(), all_ale.max()
    epi_min, epi_max = all_epi.min(), all_epi.max()
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Undersampling Experiment Statistics - {function_name} Function - {model_name}")
    print(f"{'='*60}")
    header = f"\n{'Region':<15} {'Density':<12} {'Avg Ale (norm)':<20} {'Avg Epi (norm)':<20} {'Avg Tot (norm)':<20} {'Correlation':<15} {'MSE':<15} {'NLL':<15} {'CRPS':<15} {'Spear_Ale':<15} {'Spear_Epi':<15}"
    print(header)
    print("-" * len(header))
    
    for idx, ((region_tuple, density_factor), uncertainties, mse) in enumerate(
        zip(sampling_regions, uncertainties_by_region, mse_by_region)
    ):
        region_name = f"Region_{idx+1}"
        
        ale_vals = uncertainties['ale']
        epi_vals = uncertainties['epi']
        tot_vals = uncertainties['tot']
        
        ale_norm = normalize(ale_vals, ale_min, ale_max)
        epi_norm = normalize(epi_vals, epi_min, epi_max)
        tot_norm = epi_norm + ale_norm
        
        avg_ale_norm = np.mean(ale_norm)
        avg_epi_norm = np.mean(epi_norm)
        avg_tot_norm = np.mean(tot_norm)
        
        correlation = np.corrcoef(epi_vals, ale_vals)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Get metrics for this region
        nll_val = nll_by_region[idx] if nll_by_region is not None else None
        crps_val = crps_by_region[idx] if crps_by_region is not None else None
        spear_ale_val = spearman_aleatoric_by_region[idx] if spearman_aleatoric_by_region is not None else None
        spear_epi_val = spearman_epistemic_by_region[idx] if spearman_epistemic_by_region is not None else None
        
        # Print statistics
        nll_str = f"{nll_val:>14.6f}" if nll_val is not None else f"{'N/A':>14}"
        crps_str = f"{crps_val:>14.6f}" if crps_val is not None else f"{'N/A':>14}"
        spear_ale_str = f"{spear_ale_val:>14.6f}" if spear_ale_val is not None else f"{'N/A':>14}"
        spear_epi_str = f"{spear_epi_val:>14.6f}" if spear_epi_val is not None else f"{'N/A':>14}"
        print_line = f"{region_name:<15} {density_factor:<12.2f} {avg_ale_norm:>19.6f}  {avg_epi_norm:>19.6f}  {avg_tot_norm:>19.6f}  {correlation:>14.6f}  {mse:>14.6f} {nll_str} {crps_str} {spear_ale_str} {spear_epi_str}"
        print(print_line)
        
        # Save statistics for this region
        stats_df, fig = save_summary_statistics_undersampling(
            [avg_ale_norm], [avg_epi_norm], [avg_tot_norm], [correlation], [mse],
            function_name, noise_type=noise_type,
            func_type=func_type, model_name=model_name, region_name=region_name,
            date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets,
            density_factor=density_factor,
            nll_list=[nll_val] if nll_val is not None else None,
            crps_list=[crps_val] if crps_val is not None else None,
            spearman_aleatoric_list=[spear_ale_val] if spear_ale_val is not None else None,
            spearman_epistemic_list=[spear_epi_val] if spear_epi_val is not None else None
        )
        plt.show()
        plt.close(fig)
        
        results[region_name.lower()] = {
            'avg_ale_norm': avg_ale_norm,
            'avg_epi_norm': avg_epi_norm,
            'avg_tot_norm': avg_tot_norm,
            'correlation': correlation,
            'mse': mse,
            'density_factor': density_factor,
            'nll': nll_val,
            'crps': crps_val,
            'spearman_aleatoric': spear_ale_val,
            'spearman_epistemic': spear_epi_val,
            'stats_df': stats_df
        }
    
    # Compute comparison: undersampled vs well-sampled
    undersampled_regions = [i for i, (_, df) in enumerate(sampling_regions) if df < 0.5]
    well_sampled_regions = [i for i, (_, df) in enumerate(sampling_regions) if df >= 0.5]
    
    if undersampled_regions and well_sampled_regions:
        # Combine undersampled regions
        undersampled_ale = np.concatenate([uncertainties_by_region[i]['ale'] for i in undersampled_regions])
        undersampled_epi = np.concatenate([uncertainties_by_region[i]['epi'] for i in undersampled_regions])
        undersampled_tot = np.concatenate([uncertainties_by_region[i]['tot'] for i in undersampled_regions])
        undersampled_mse = np.mean([mse_by_region[i] for i in undersampled_regions])
        
        # Combine well-sampled regions
        well_sampled_ale = np.concatenate([uncertainties_by_region[i]['ale'] for i in well_sampled_regions])
        well_sampled_epi = np.concatenate([uncertainties_by_region[i]['epi'] for i in well_sampled_regions])
        well_sampled_tot = np.concatenate([uncertainties_by_region[i]['tot'] for i in well_sampled_regions])
        well_sampled_mse = np.mean([mse_by_region[i] for i in well_sampled_regions])
        
        # Normalize
        undersampled_ale_norm = normalize(undersampled_ale, ale_min, ale_max)
        undersampled_epi_norm = normalize(undersampled_epi, epi_min, epi_max)
        undersampled_tot_norm = undersampled_epi_norm + undersampled_ale_norm
        
        well_sampled_ale_norm = normalize(well_sampled_ale, ale_min, ale_max)
        well_sampled_epi_norm = normalize(well_sampled_epi, epi_min, epi_max)
        well_sampled_tot_norm = well_sampled_epi_norm + well_sampled_ale_norm
        
        avg_undersampled_ale = np.mean(undersampled_ale_norm)
        avg_undersampled_epi = np.mean(undersampled_epi_norm)
        avg_undersampled_tot = np.mean(undersampled_tot_norm)
        corr_undersampled = np.corrcoef(undersampled_epi, undersampled_ale)[0, 1]
        if np.isnan(corr_undersampled):
            corr_undersampled = 0.0
        
        avg_well_sampled_ale = np.mean(well_sampled_ale_norm)
        avg_well_sampled_epi = np.mean(well_sampled_epi_norm)
        avg_well_sampled_tot = np.mean(well_sampled_tot_norm)
        corr_well_sampled = np.corrcoef(well_sampled_epi, well_sampled_ale)[0, 1]
        if np.isnan(corr_well_sampled):
            corr_well_sampled = 0.0
        
        print(f"\n{'='*60}")
        print("Comparison: Undersampled vs Well-sampled Regions")
        print(f"{'='*60}")
        print(f"{'Region':<20} {'Avg Ale (norm)':<20} {'Avg Epi (norm)':<20} {'Avg Tot (norm)':<20} {'Correlation':<15} {'MSE':<15}")
        print("-" * 140)
        print(f"{'Undersampled':<20} {avg_undersampled_ale:>19.6f}  {avg_undersampled_epi:>19.6f}  {avg_undersampled_tot:>19.6f}  {corr_undersampled:>14.6f}  {undersampled_mse:>14.6f}")
        print(f"{'Well-sampled':<20} {avg_well_sampled_ale:>19.6f}  {avg_well_sampled_epi:>19.6f}  {avg_well_sampled_tot:>19.6f}  {corr_well_sampled:>14.6f}  {well_sampled_mse:>14.6f}")
        
        # Save comparison statistics
        for region_name, stats in [('Undersampled', (avg_undersampled_ale, avg_undersampled_epi, avg_undersampled_tot, corr_undersampled, undersampled_mse)),
                                   ('Well_sampled', (avg_well_sampled_ale, avg_well_sampled_epi, avg_well_sampled_tot, corr_well_sampled, well_sampled_mse))]:
            stats_df, fig = save_summary_statistics_undersampling(
                [stats[0]], [stats[1]], [stats[2]], [stats[3]], [stats[4]],
                function_name, noise_type=noise_type,
                func_type=func_type, model_name=model_name, region_name=region_name,
                date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets
            )
            plt.show()
            plt.close(fig)
    
    # Compute overall statistics (across all regions combined)
    all_ale_combined = np.concatenate([unc['ale'] for unc in uncertainties_by_region])
    all_epi_combined = np.concatenate([unc['epi'] for unc in uncertainties_by_region])
    all_tot_combined = np.concatenate([unc['tot'] for unc in uncertainties_by_region])
    overall_mse = np.mean(mse_by_region)
    
    # Normalize overall
    ale_norm_overall = normalize(all_ale_combined, ale_min, ale_max)
    epi_norm_overall = normalize(all_epi_combined, epi_min, epi_max)
    tot_norm_overall = epi_norm_overall + ale_norm_overall
    
    avg_ale_overall = np.mean(ale_norm_overall)
    avg_epi_overall = np.mean(epi_norm_overall)
    avg_tot_overall = np.mean(tot_norm_overall)
    corr_overall = np.corrcoef(all_epi_combined, all_ale_combined)[0, 1]
    if np.isnan(corr_overall):
        corr_overall = 0.0
    
    print(f"\n{'='*60}")
    print("Overall Statistics (All Regions Combined)")
    print(f"{'='*60}")
    print(f"{'Region':<15} {'Avg Ale (norm)':<20} {'Avg Epi (norm)':<20} {'Avg Tot (norm)':<20} {'Correlation':<15} {'MSE':<15}")
    print("-" * 140)
    print(f"{'Overall':<15} {avg_ale_overall:>19.6f}  {avg_epi_overall:>19.6f}  {avg_tot_overall:>19.6f}  {corr_overall:>14.6f}  {overall_mse:>14.6f}")
    
    # Save overall statistics
    stats_df, fig = save_summary_statistics_undersampling(
        [avg_ale_overall], [avg_epi_overall], [avg_tot_overall], [corr_overall], [overall_mse],
        function_name, noise_type=noise_type,
        func_type=func_type, model_name=model_name, region_name='Overall',
        date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets
    )
    plt.show()
    plt.close(fig)
    
    results['overall'] = {
        'avg_ale_norm': avg_ale_overall,
        'avg_epi_norm': avg_epi_overall,
        'avg_tot_norm': avg_tot_overall,
        'correlation': corr_overall,
        'mse': overall_mse,
        'stats_df': stats_df
    }
    
    print(f"\n{'='*60}")
    print("Note: Average values are normalized to [0, 1] range across all regions")
    print("      Correlation is computed on original (non-normalized) uncertainty values")
    print(f"{'='*60}")
    
    return results


def compute_and_save_statistics_entropy_undersampling(
    uncertainties_entropy_by_region: list,
    mse_by_region: list,
    sampling_regions: list,
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    date: str = None,
    dropout_p: float = None,
    mc_samples: int = None,
    n_nets: int = None
):
    """
    Compute and save normalized entropy-based statistics for undersampling experiments.
    
    Parameters:
    -----------
    uncertainties_entropy_by_region : list
        List of dictionaries, one per region, each with 'ale', 'epi', 'tot' entropy arrays
    mse_by_region : list
        List of MSE values, one per region
    sampling_regions : list
        List of (region_tuple, density_factor) tuples
    function_name : str
        Human-readable function name (e.g., "Linear", "Sinusoidal")
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str
        Function type identifier (e.g., 'linear', 'sin')
    model_name : str
        Model name for saving results (e.g., 'MC_Dropout', 'Deep_Ensemble')
    date : str, optional
        Date string in YYYYMMDD format
    dropout_p : float, optional
        Dropout probability for MC Dropout
    mc_samples : int, optional
        Number of MC samples for MC Dropout
    n_nets : int, optional
        Number of nets for Deep Ensemble
    
    Returns:
    --------
    dict : Statistics dictionary with results for each region
    """
    from utils.results_save import save_summary_statistics_entropy_undersampling
    
    def normalize(values, vmin, vmax):
        """Normalize values to [0, 1] range"""
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    # Collect all entropy values for normalization (across all regions)
    all_ale = np.concatenate([unc['ale'] for unc in uncertainties_entropy_by_region])
    all_epi = np.concatenate([unc['epi'] for unc in uncertainties_entropy_by_region])
    all_tot = np.concatenate([unc['tot'] for unc in uncertainties_entropy_by_region])
    
    # Compute min/max for normalization
    ale_min, ale_max = all_ale.min(), all_ale.max()
    epi_min, epi_max = all_epi.min(), all_epi.max()
    tot_min, tot_max = all_tot.min(), all_tot.max()
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Undersampling Experiment Statistics (Entropy) - {function_name} Function - {model_name}")
    print(f"{'='*60}")
    print(f"\n{'Region':<15} {'Density':<12} {'Avg Ale (norm)':<20} {'Avg Epi (norm)':<20} {'Avg Tot (norm)':<20} {'Correlation':<15} {'MSE':<15}")
    print("-" * 140)
    
    for idx, ((region_tuple, density_factor), uncertainties, mse) in enumerate(
        zip(sampling_regions, uncertainties_entropy_by_region, mse_by_region)
    ):
        region_name = f"Region_{idx+1}"
        
        ale_entropy = uncertainties['ale']
        epi_entropy = uncertainties['epi']
        tot_entropy = uncertainties['tot']
        
        # Normalize entropy values (ale and epi separately)
        ale_norm = normalize(ale_entropy, ale_min, ale_max)
        epi_norm = normalize(epi_entropy, epi_min, epi_max)
        # Total is sum of normalized ale and epi (not normalized separately)
        tot_norm = ale_norm + epi_norm
        
        # Compute normalized averages
        avg_ale_entropy_norm = np.mean(ale_norm)
        avg_epi_entropy_norm = np.mean(epi_norm)
        avg_tot_entropy_norm = np.mean(tot_norm)
        
        # Correlation computed on original (non-normalized) values
        correlation = np.corrcoef(epi_entropy, ale_entropy)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        print(f"{region_name:<15} {density_factor:<12.2f} {avg_ale_entropy_norm:>19.6f}  {avg_epi_entropy_norm:>19.6f}  {avg_tot_entropy_norm:>19.6f}  {correlation:>14.6f}  {mse:>14.6f}")
        
        # Save normalized statistics for this region
        stats_df, fig = save_summary_statistics_entropy_undersampling(
            [avg_ale_entropy_norm], [avg_epi_entropy_norm], [avg_tot_entropy_norm], [correlation], [mse],
            function_name, noise_type=noise_type,
            func_type=func_type, model_name=model_name, region_name=region_name,
            date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets,
            density_factor=density_factor
        )
        plt.show()
        plt.close(fig)
        
        results[region_name.lower()] = {
            'avg_ale_norm': avg_ale_entropy_norm,
            'avg_epi_norm': avg_epi_entropy_norm,
            'avg_tot_norm': avg_tot_entropy_norm,
            'correlation': correlation,
            'mse': mse,
            'density_factor': density_factor,
            'stats_df': stats_df
        }
    
    # Compute comparison: undersampled vs well-sampled
    undersampled_regions = [i for i, (_, df) in enumerate(sampling_regions) if df < 0.5]
    well_sampled_regions = [i for i, (_, df) in enumerate(sampling_regions) if df >= 0.5]
    
    if undersampled_regions and well_sampled_regions:
        # Combine undersampled regions
        undersampled_ale = np.concatenate([uncertainties_entropy_by_region[i]['ale'] for i in undersampled_regions])
        undersampled_epi = np.concatenate([uncertainties_entropy_by_region[i]['epi'] for i in undersampled_regions])
        undersampled_tot = np.concatenate([uncertainties_entropy_by_region[i]['tot'] for i in undersampled_regions])
        undersampled_mse = np.mean([mse_by_region[i] for i in undersampled_regions])
        
        # Combine well-sampled regions
        well_sampled_ale = np.concatenate([uncertainties_entropy_by_region[i]['ale'] for i in well_sampled_regions])
        well_sampled_epi = np.concatenate([uncertainties_entropy_by_region[i]['epi'] for i in well_sampled_regions])
        well_sampled_tot = np.concatenate([uncertainties_entropy_by_region[i]['tot'] for i in well_sampled_regions])
        well_sampled_mse = np.mean([mse_by_region[i] for i in well_sampled_regions])
        
        # Normalize (ale and epi separately)
        undersampled_ale_norm = normalize(undersampled_ale, ale_min, ale_max)
        undersampled_epi_norm = normalize(undersampled_epi, epi_min, epi_max)
        # Total is sum of normalized ale and epi (not normalized separately)
        undersampled_tot_norm = undersampled_ale_norm + undersampled_epi_norm
        
        well_sampled_ale_norm = normalize(well_sampled_ale, ale_min, ale_max)
        well_sampled_epi_norm = normalize(well_sampled_epi, epi_min, epi_max)
        # Total is sum of normalized ale and epi (not normalized separately)
        well_sampled_tot_norm = well_sampled_ale_norm + well_sampled_epi_norm
        
        avg_undersampled_ale = np.mean(undersampled_ale_norm)
        avg_undersampled_epi = np.mean(undersampled_epi_norm)
        avg_undersampled_tot = np.mean(undersampled_tot_norm)
        corr_undersampled = np.corrcoef(undersampled_epi, undersampled_ale)[0, 1]
        if np.isnan(corr_undersampled):
            corr_undersampled = 0.0
        
        avg_well_sampled_ale = np.mean(well_sampled_ale_norm)
        avg_well_sampled_epi = np.mean(well_sampled_epi_norm)
        avg_well_sampled_tot = np.mean(well_sampled_tot_norm)
        corr_well_sampled = np.corrcoef(well_sampled_epi, well_sampled_ale)[0, 1]
        if np.isnan(corr_well_sampled):
            corr_well_sampled = 0.0
        
        print(f"\n{'='*60}")
        print("Comparison: Undersampled vs Well-sampled Regions (Entropy)")
        print(f"{'='*60}")
        print(f"{'Region':<20} {'Avg Ale (norm)':<20} {'Avg Epi (norm)':<20} {'Avg Tot (norm)':<20} {'Correlation':<15} {'MSE':<15}")
        print("-" * 140)
        print(f"{'Undersampled':<20} {avg_undersampled_ale:>19.6f}  {avg_undersampled_epi:>19.6f}  {avg_undersampled_tot:>19.6f}  {corr_undersampled:>14.6f}  {undersampled_mse:>14.6f}")
        print(f"{'Well-sampled':<20} {avg_well_sampled_ale:>19.6f}  {avg_well_sampled_epi:>19.6f}  {avg_well_sampled_tot:>19.6f}  {corr_well_sampled:>14.6f}  {well_sampled_mse:>14.6f}")
        
        # Save comparison statistics
        for region_name, stats in [('Undersampled', (avg_undersampled_ale, avg_undersampled_epi, avg_undersampled_tot, corr_undersampled, undersampled_mse)),
                                   ('Well_sampled', (avg_well_sampled_ale, avg_well_sampled_epi, avg_well_sampled_tot, corr_well_sampled, well_sampled_mse))]:
            stats_df, fig = save_summary_statistics_entropy_undersampling(
                [stats[0]], [stats[1]], [stats[2]], [stats[3]], [stats[4]],
                function_name, noise_type=noise_type,
                func_type=func_type, model_name=model_name, region_name=region_name,
                date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets
            )
            plt.show()
            plt.close(fig)
    
    # Compute overall statistics (across all regions combined)
    all_ale_combined = np.concatenate([unc['ale'] for unc in uncertainties_entropy_by_region])
    all_epi_combined = np.concatenate([unc['epi'] for unc in uncertainties_entropy_by_region])
    all_tot_combined = np.concatenate([unc['tot'] for unc in uncertainties_entropy_by_region])
    overall_mse = np.mean(mse_by_region)
    
    # Normalize overall (ale and epi separately)
    ale_norm_overall = normalize(all_ale_combined, ale_min, ale_max)
    epi_norm_overall = normalize(all_epi_combined, epi_min, epi_max)
    # Total is sum of normalized ale and epi (not normalized separately)
    tot_norm_overall = ale_norm_overall + epi_norm_overall
    
    avg_ale_overall = np.mean(ale_norm_overall)
    avg_epi_overall = np.mean(epi_norm_overall)
    avg_tot_overall = np.mean(tot_norm_overall)
    corr_overall = np.corrcoef(all_epi_combined, all_ale_combined)[0, 1]
    if np.isnan(corr_overall):
        corr_overall = 0.0
    
    print(f"\n{'='*60}")
    print("Overall Statistics (All Regions Combined) - Entropy")
    print(f"{'='*60}")
    print(f"{'Region':<15} {'Avg Ale (norm)':<20} {'Avg Epi (norm)':<20} {'Avg Tot (norm)':<20} {'Correlation':<15} {'MSE':<15}")
    print("-" * 140)
    print(f"{'Overall':<15} {avg_ale_overall:>19.6f}  {avg_epi_overall:>19.6f}  {avg_tot_overall:>19.6f}  {corr_overall:>14.6f}  {overall_mse:>14.6f}")
    
    # Save overall statistics
    stats_df, fig = save_summary_statistics_entropy_undersampling(
        [avg_ale_overall], [avg_epi_overall], [avg_tot_overall], [corr_overall], [overall_mse],
        function_name, noise_type=noise_type,
        func_type=func_type, model_name=model_name, region_name='Overall',
        date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets
    )
    plt.show()
    plt.close(fig)
    
    results['overall'] = {
        'avg_ale_norm': avg_ale_overall,
        'avg_epi_norm': avg_epi_overall,
        'avg_tot_norm': avg_tot_overall,
        'correlation': corr_overall,
        'mse': overall_mse,
        'stats_df': stats_df
    }
    
    print(f"\n{'='*60}")
    print("Note: Average entropy values are normalized to [0, 1] range across all regions")
    print("      Correlation is computed on original (non-normalized) entropy values")
    print(f"{'='*60}")
    
    return results


# ========== Experiment Functions ==========

def run_mc_dropout_undersampling_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    sampling_regions: list = None,
    n_train: int = 1000,
    grid_points: int = 1000,
    seed: int = 42,
    p: float = 0.2,
    beta: float = 0.5,
    epochs: int = 250,
    lr: float = 1e-3,
    batch_size: int = 32,
    mc_samples: int = 20,
    parallel: bool = True,
    entropy_method: str = 'analytical'
):
    """
    Run undersampling experiment for MC Dropout model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test (e.g., ['linear', 'sin'])
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    train_range : tuple
        (min, max) range for training data
    sampling_regions : list of tuples
        List of (region_tuple, density_factor) tuples. If None, uses default: [((-5, 0), 0.2), ((0, 5), 1.0), ((5, 10), 0.2)]
    n_train : int
        Number of training samples
    grid_points : int
        Number of grid points for evaluation
    seed : int
        Random seed
    p : float
        Dropout probability
    beta : float
        Beta parameter for beta-NLL loss
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    mc_samples : int
        Number of Monte Carlo samples for prediction
    parallel : bool
        Whether to use parallel execution
    """
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    if sampling_regions is None:
        # Default: undersampled regions on the sides, well-sampled in the middle
        sampling_regions = [((-5, 0), 0.2), ((0, 5), 1.0), ((5, 10), 0.2)]
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - MC Dropout - Undersampling Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with undersampling
        x_train, y_train, x_grid, y_grid_clean, region_masks = generate_data_with_undersampling(
            generate_toy_regression_func, n_train, train_range, sampling_regions,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"Sampling regions: {sampling_regions}")
        print(f"Total training samples: {len(x_train)}")
        for idx, ((region_tuple, density_factor), mask) in enumerate(zip(sampling_regions, region_masks)):
            n_samples_in_region = np.sum((x_train[:, 0] >= region_tuple[0]) & (x_train[:, 0] <= region_tuple[1]))
            print(f"  Region {idx+1} {region_tuple}: density={density_factor:.2f}, samples={n_samples_in_region}, grid_points={np.sum(mask)}")
        print()
        
        # Train model
        print(f"{'='*60}")
        print(f"Training model...")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        model = MCDropoutRegressor(p=p)
        train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
        
        # Make predictions with raw arrays
        result = mc_dropout_predict(model, x_grid, M=mc_samples, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        elif entropy_method == 'numerical':
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        else:
            raise ValueError(f"Unknown entropy_method: {entropy_method}. Must be 'analytical' or 'numerical'")
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Split uncertainties by region
        uncertainties_by_region = []
        uncertainties_entropy_by_region = []
        mse_by_region = []
        
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        for mask in region_masks:
            uncertainties_by_region.append({
                'ale': ale_var[mask].flatten() if ale_var.ndim > 1 else ale_var[mask],
                'epi': epi_var[mask].flatten() if epi_var.ndim > 1 else epi_var[mask],
                'tot': tot_var[mask].flatten() if tot_var.ndim > 1 else tot_var[mask]
            })
            
            uncertainties_entropy_by_region.append({
                'ale': ale_entropy[mask] if ale_entropy.ndim == 1 else ale_entropy[mask].flatten(),
                'epi': epi_entropy[mask] if epi_entropy.ndim == 1 else epi_entropy[mask].flatten(),
                'tot': tot_entropy[mask] if tot_entropy.ndim == 1 else tot_entropy[mask].flatten()
            })
            
            if np.any(mask):
                mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
            else:
                mse = 0.0
            mse_by_region.append(mse)
        
        # Compute predictive aggregation (μ*, σ*²)
        mu_star, sigma2_star = compute_predictive_aggregation(mu_samples, sigma2_samples)
        
        # Compute true noise variance for grid points
        true_noise_var = compute_true_noise_variance(x_grid, noise_type, func_type)
        
        # Compute NLL, CRPS, and disentanglement metrics for each region
        nll_by_region = []
        crps_by_region = []
        spearman_aleatoric_by_region = []
        spearman_epistemic_by_region = []
        
        for mask in region_masks:
            if np.any(mask):
                nll = compute_gaussian_nll(y_grid_clean_flat[mask], mu_star[mask], sigma2_star[mask])
                crps = compute_crps_gaussian(y_grid_clean_flat[mask], mu_star[mask], sigma2_star[mask])
                disentangle = compute_uncertainty_disentanglement(
                    y_grid_clean_flat[mask], mu_star[mask],
                    ale_var[mask], epi_var[mask], true_noise_var[mask]
                )
                nll_by_region.append(nll)
                crps_by_region.append(crps)
                spearman_aleatoric_by_region.append(disentangle['spearman_aleatoric'])
                spearman_epistemic_by_region.append(disentangle['spearman_epistemic'])
            else:
                nll_by_region.append(0.0)
                crps_by_region.append(0.0)
                spearman_aleatoric_by_region.append(0.0)
                spearman_epistemic_by_region.append(0.0)
        
        # Save raw model outputs
        save_model_outputs(
            mu_samples=mu_samples,
            sigma2_samples=sigma2_samples,
            x_grid=x_grid,
            y_grid_clean=y_grid_clean,
            x_train_subset=x_train,
            y_train_subset=y_train,
            model_name='MC_Dropout',
            noise_type=noise_type,
            func_type=func_type,
            subfolder='undersampling',
            dropout_p=p,
            mc_samples=mc_samples,
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, region_masks, sampling_regions,
            title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - Undersampling - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        from utils.plotting import plot_uncertainties_entropy_undersampling
        plot_uncertainties_entropy_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, region_masks, sampling_regions,
            title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - Undersampling - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_undersampling(
            uncertainties_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'MC_Dropout',
            date=date, dropout_p=p, mc_samples=mc_samples,
            nll_by_region=nll_by_region, crps_by_region=crps_by_region,
            spearman_aleatoric_by_region=spearman_aleatoric_by_region,
            spearman_epistemic_by_region=spearman_epistemic_by_region
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_undersampling(
            uncertainties_entropy_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'MC_Dropout',
            date=date, dropout_p=p, mc_samples=mc_samples
        )


def run_deep_ensemble_undersampling_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    sampling_regions: list = None,
    n_train: int = 1000,
    grid_points: int = 1000,
    seed: int = 42,
    beta: float = 0.5,
    batch_size: int = 32,
    K: int = 5,
    epochs: int = 250,
    parallel: bool = True,
    entropy_method: str = 'analytical'
):
    """
    Run undersampling experiment for Deep Ensemble model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    train_range : tuple
        (min, max) range for training data
    sampling_regions : list of tuples
        List of (region_tuple, density_factor) tuples. If None, uses default.
    n_train : int
        Number of training samples
    grid_points : int
        Number of grid points for evaluation
    seed : int
        Random seed
    beta : float
        Beta parameter for beta-NLL loss
    batch_size : int
        Batch size for training
    K : int
        Number of ensemble members
    parallel : bool
        Whether to use parallel execution
    """
    from Models.Deep_Ensemble import (
        train_ensemble_deep,
        ensemble_predict_deep
    )
    from Models.MC_Dropout import (
        normalize_x,
        normalize_x_data
    )
    
    if sampling_regions is None:
        sampling_regions = [((-5, 0), 0.2), ((0, 5), 1.0), ((5, 10), 0.2)]
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - Deep Ensemble - Undersampling Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with undersampling
        x_train, y_train, x_grid, y_grid_clean, region_masks = generate_data_with_undersampling(
            generate_toy_regression_func, n_train, train_range, sampling_regions,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"Sampling regions: {sampling_regions}")
        print(f"Total training samples: {len(x_train)}")
        for idx, ((region_tuple, density_factor), mask) in enumerate(zip(sampling_regions, region_masks)):
            n_samples_in_region = np.sum((x_train[:, 0] >= region_tuple[0]) & (x_train[:, 0] <= region_tuple[1]))
            print(f"  Region {idx+1} {region_tuple}: density={density_factor:.2f}, samples={n_samples_in_region}, grid_points={np.sum(mask)}")
        print()
        
        # Train model
        print(f"{'='*60}")
        print(f"Training ensemble...")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x_mean, x_std = normalize_x(x_train)
        x_train_norm = normalize_x_data(x_train, x_mean, x_std)
        x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
        
        ensemble = train_ensemble_deep(
            x_train_norm, y_train,
            batch_size=batch_size, K=K,
            loss_type='beta_nll', beta=beta, parallel=True, epochs=epochs
        )
        
        # Make predictions with raw arrays
        result = ensemble_predict_deep(ensemble, x_grid_norm, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        elif entropy_method == 'numerical':
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        else:
            raise ValueError(f"Unknown entropy_method: {entropy_method}. Must be 'analytical' or 'numerical'")
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Split uncertainties by region
        uncertainties_by_region = []
        uncertainties_entropy_by_region = []
        mse_by_region = []
        
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        for mask in region_masks:
            uncertainties_by_region.append({
                'ale': ale_var[mask].flatten() if ale_var.ndim > 1 else ale_var[mask],
                'epi': epi_var[mask].flatten() if epi_var.ndim > 1 else epi_var[mask],
                'tot': tot_var[mask].flatten() if tot_var.ndim > 1 else tot_var[mask]
            })
            
            uncertainties_entropy_by_region.append({
                'ale': ale_entropy[mask] if ale_entropy.ndim == 1 else ale_entropy[mask].flatten(),
                'epi': epi_entropy[mask] if epi_entropy.ndim == 1 else epi_entropy[mask].flatten(),
                'tot': tot_entropy[mask] if tot_entropy.ndim == 1 else tot_entropy[mask].flatten()
            })
            
            if np.any(mask):
                mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
            else:
                mse = 0.0
            mse_by_region.append(mse)
        
        # Save raw model outputs
        save_model_outputs(
            mu_samples=mu_samples,
            sigma2_samples=sigma2_samples,
            x_grid=x_grid,
            y_grid_clean=y_grid_clean,
            x_train_subset=x_train,
            y_train_subset=y_train,
            model_name='Deep_Ensemble',
            noise_type=noise_type,
            func_type=func_type,
            subfolder='undersampling',
            n_nets=K,
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, region_masks, sampling_regions,
            title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - Undersampling - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        from utils.plotting import plot_uncertainties_entropy_undersampling
        plot_uncertainties_entropy_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, region_masks, sampling_regions,
            title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - Undersampling - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_undersampling(
            uncertainties_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'Deep_Ensemble',
            date=date, n_nets=K,
            nll_by_region=nll_by_region, crps_by_region=crps_by_region,
            spearman_aleatoric_by_region=spearman_aleatoric_by_region,
            spearman_epistemic_by_region=spearman_epistemic_by_region
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_undersampling(
            uncertainties_entropy_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'Deep_Ensemble',
            date=date, n_nets=K
        )


def run_bnn_undersampling_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    sampling_regions: list = None,
    n_train: int = 1000,
    grid_points: int = 1000,
    seed: int = 42,
    hidden_width: int = 16,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    chains: int = 1,
    parallel: bool = True,
    entropy_method: str = 'analytical'
):
    """
    Run undersampling experiment for BNN (Bayesian Neural Network) model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    train_range : tuple
        (min, max) range for training data
    sampling_regions : list of tuples
        List of (region_tuple, density_factor) tuples. If None, uses default.
    n_train : int
        Number of training samples
    grid_points : int
        Number of grid points for evaluation
    seed : int
        Random seed
    hidden_width : int
        Hidden layer width
    weight_scale : float
        Weight scale for prior
    warmup : int
        Number of warmup steps for MCMC
    samples : int
        Number of MCMC samples
    chains : int
        Number of MCMC chains
    parallel : bool
        Whether to use parallel execution
    """
    from Models.BNN import (
        train_bnn,
        bnn_predict,
        normalize_x as bnn_normalize_x,
        normalize_x_data as bnn_normalize_x_data
    )
    
    if sampling_regions is None:
        sampling_regions = [((-5, 0), 0.2), ((0, 5), 1.0), ((5, 10), 0.2)]
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - BNN - Undersampling Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with undersampling
        x_train, y_train, x_grid, y_grid_clean, region_masks = generate_data_with_undersampling(
            generate_toy_regression_func, n_train, train_range, sampling_regions,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"Sampling regions: {sampling_regions}")
        print(f"Total training samples: {len(x_train)}")
        for idx, ((region_tuple, density_factor), mask) in enumerate(zip(sampling_regions, region_masks)):
            n_samples_in_region = np.sum((x_train[:, 0] >= region_tuple[0]) & (x_train[:, 0] <= region_tuple[1]))
            print(f"  Region {idx+1} {region_tuple}: density={density_factor:.2f}, samples={n_samples_in_region}, grid_points={np.sum(mask)}")
        print()
        
        # Train model
        print(f"{'='*60}")
        print(f"Training BNN with MCMC...")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x_mean, x_std = bnn_normalize_x(x_train)
        x_train_norm = bnn_normalize_x_data(x_train, x_mean, x_std)
        x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
        
        mcmc = train_bnn(
            x_train_norm, y_train,
            hidden_width=hidden_width, weight_scale=weight_scale,
            warmup=warmup, samples=samples, chains=chains, seed=seed
        )
        
        # Make predictions with raw arrays
        result = bnn_predict(
            mcmc, x_grid_norm,
            hidden_width=hidden_width, weight_scale=weight_scale,
            return_raw_arrays=True
        )
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        elif entropy_method == 'numerical':
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        else:
            raise ValueError(f"Unknown entropy_method: {entropy_method}. Must be 'analytical' or 'numerical'")
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Compute predictive aggregation (μ*, σ*²)
        mu_star, sigma2_star = compute_predictive_aggregation(mu_samples, sigma2_samples)
        
        # Compute true noise variance for grid points
        true_noise_var = compute_true_noise_variance(x_grid, noise_type, func_type)
        
        # Split uncertainties by region
        uncertainties_by_region = []
        uncertainties_entropy_by_region = []
        mse_by_region = []
        nll_by_region = []
        crps_by_region = []
        spearman_aleatoric_by_region = []
        spearman_epistemic_by_region = []
        
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        for mask in region_masks:
            uncertainties_by_region.append({
                'ale': ale_var[mask].flatten() if ale_var.ndim > 1 else ale_var[mask],
                'epi': epi_var[mask].flatten() if epi_var.ndim > 1 else epi_var[mask],
                'tot': tot_var[mask].flatten() if tot_var.ndim > 1 else tot_var[mask]
            })
            
            uncertainties_entropy_by_region.append({
                'ale': ale_entropy[mask] if ale_entropy.ndim == 1 else ale_entropy[mask].flatten(),
                'epi': epi_entropy[mask] if epi_entropy.ndim == 1 else epi_entropy[mask].flatten(),
                'tot': tot_entropy[mask] if tot_entropy.ndim == 1 else tot_entropy[mask].flatten()
            })
            
            if np.any(mask):
                mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
                
                # Compute NLL, CRPS, and disentanglement metrics for this region
                nll = compute_gaussian_nll(y_grid_clean_flat[mask], mu_star[mask], sigma2_star[mask])
                crps = compute_crps_gaussian(y_grid_clean_flat[mask], mu_star[mask], sigma2_star[mask])
                disentangle = compute_uncertainty_disentanglement(
                    y_grid_clean_flat[mask], mu_star[mask],
                    ale_var[mask] if ale_var.ndim == 1 else ale_var[mask].flatten(),
                    epi_var[mask] if epi_var.ndim == 1 else epi_var[mask].flatten(),
                    true_noise_var[mask]
                )
            else:
                mse = 0.0
                nll = 0.0
                crps = 0.0
                disentangle = {'spearman_aleatoric': 0.0, 'spearman_epistemic': 0.0}
            
            mse_by_region.append(mse)
            nll_by_region.append(nll)
            crps_by_region.append(crps)
            spearman_aleatoric_by_region.append(disentangle['spearman_aleatoric'])
            spearman_epistemic_by_region.append(disentangle['spearman_epistemic'])
        
        # Save raw model outputs
        save_model_outputs(
            mu_samples=mu_samples,
            sigma2_samples=sigma2_samples,
            x_grid=x_grid,
            y_grid_clean=y_grid_clean,
            x_train_subset=x_train,
            y_train_subset=y_train,
            model_name='BNN',
            noise_type=noise_type,
            func_type=func_type,
            subfolder='undersampling',
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, region_masks, sampling_regions,
            title=f"BNN (Pyro NUTS) - {function_names[func_type]} - Undersampling - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        from utils.plotting import plot_uncertainties_entropy_undersampling
        plot_uncertainties_entropy_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, region_masks, sampling_regions,
            title=f"BNN (Pyro NUTS) - {function_names[func_type]} - Undersampling - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_undersampling(
            uncertainties_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'BNN',
            date=date,
            nll_by_region=nll_by_region, crps_by_region=crps_by_region,
            spearman_aleatoric_by_region=spearman_aleatoric_by_region,
            spearman_epistemic_by_region=spearman_epistemic_by_region
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_undersampling(
            uncertainties_entropy_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'BNN',
            date=date
        )


def run_bamlss_undersampling_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    sampling_regions: list = None,
    n_train: int = 1000,
    grid_points: int = 1000,
    seed: int = 42,
    n_iter: int = 12000,
    burnin: int = 2000,
    thin: int = 10,
    nsamples: int = 1000,
    parallel: bool = True,
    entropy_method: str = 'analytical'
):
    """
    Run undersampling experiment for BAMLSS model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    train_range : tuple
        (min, max) range for training data
    sampling_regions : list of tuples
        List of (region_tuple, density_factor) tuples. If None, uses default.
    n_train : int
        Number of training samples
    grid_points : int
        Number of grid points for evaluation
    seed : int
        Random seed
    n_iter : int
        Number of MCMC iterations
    burnin : int
        Number of burn-in samples
    thin : int
        Thinning interval
    nsamples : int
        Number of samples to use
    parallel : bool
        Whether to use parallel execution
    """
    from Models.BAMLSS import bamlss_predict
    
    if sampling_regions is None:
        sampling_regions = [((-5, 0), 0.2), ((0, 5), 1.0), ((5, 10), 0.2)]
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - BAMLSS - Undersampling Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with undersampling
        x_train, y_train, x_grid, y_grid_clean, region_masks = generate_data_with_undersampling(
            generate_toy_regression_func, n_train, train_range, sampling_regions,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"Sampling regions: {sampling_regions}")
        print(f"Total training samples: {len(x_train)}")
        for idx, ((region_tuple, density_factor), mask) in enumerate(zip(sampling_regions, region_masks)):
            n_samples_in_region = np.sum((x_train[:, 0] >= region_tuple[0]) & (x_train[:, 0] <= region_tuple[1]))
            print(f"  Region {idx+1} {region_tuple}: density={density_factor:.2f}, samples={n_samples_in_region}, grid_points={np.sum(mask)}")
        print()
        
        # Train model
        print(f"{'='*60}")
        print(f"Fitting BAMLSS...")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # BAMLSS fits directly - get raw arrays
        result = bamlss_predict(
            x_train, y_train, x_grid,
            n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples,
            return_raw_arrays=True
        )
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        elif entropy_method == 'numerical':
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        else:
            raise ValueError(f"Unknown entropy_method: {entropy_method}. Must be 'analytical' or 'numerical'")
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Compute predictive aggregation (μ*, σ*²)
        mu_star, sigma2_star = compute_predictive_aggregation(mu_samples, sigma2_samples)
        
        # Compute true noise variance for grid points
        true_noise_var = compute_true_noise_variance(x_grid, noise_type, func_type)
        
        # Split uncertainties by region
        uncertainties_by_region = []
        uncertainties_entropy_by_region = []
        mse_by_region = []
        nll_by_region = []
        crps_by_region = []
        spearman_aleatoric_by_region = []
        spearman_epistemic_by_region = []
        
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        for mask in region_masks:
            uncertainties_by_region.append({
                'ale': ale_var[mask].flatten() if ale_var.ndim > 1 else ale_var[mask],
                'epi': epi_var[mask].flatten() if epi_var.ndim > 1 else epi_var[mask],
                'tot': tot_var[mask].flatten() if tot_var.ndim > 1 else tot_var[mask]
            })
            
            uncertainties_entropy_by_region.append({
                'ale': ale_entropy[mask] if ale_entropy.ndim == 1 else ale_entropy[mask].flatten(),
                'epi': epi_entropy[mask] if epi_entropy.ndim == 1 else epi_entropy[mask].flatten(),
                'tot': tot_entropy[mask] if tot_entropy.ndim == 1 else tot_entropy[mask].flatten()
            })
            
            if np.any(mask):
                mse = np.mean((mu_pred_flat[mask] - y_grid_clean_flat[mask])**2)
                
                # Compute NLL, CRPS, and disentanglement metrics for this region
                nll = compute_gaussian_nll(y_grid_clean_flat[mask], mu_star[mask], sigma2_star[mask])
                crps = compute_crps_gaussian(y_grid_clean_flat[mask], mu_star[mask], sigma2_star[mask])
                disentangle = compute_uncertainty_disentanglement(
                    y_grid_clean_flat[mask], mu_star[mask],
                    ale_var[mask] if ale_var.ndim == 1 else ale_var[mask].flatten(),
                    epi_var[mask] if epi_var.ndim == 1 else epi_var[mask].flatten(),
                    true_noise_var[mask]
                )
            else:
                mse = 0.0
                nll = 0.0
                crps = 0.0
                disentangle = {'spearman_aleatoric': 0.0, 'spearman_epistemic': 0.0}
            
            mse_by_region.append(mse)
            nll_by_region.append(nll)
            crps_by_region.append(crps)
            spearman_aleatoric_by_region.append(disentangle['spearman_aleatoric'])
            spearman_epistemic_by_region.append(disentangle['spearman_epistemic'])
        
        # Save raw model outputs
        save_model_outputs(
            mu_samples=mu_samples,
            sigma2_samples=sigma2_samples,
            x_grid=x_grid,
            y_grid_clean=y_grid_clean,
            x_train_subset=x_train,
            y_train_subset=y_train,
            model_name='BAMLSS',
            noise_type=noise_type,
            func_type=func_type,
            subfolder='undersampling',
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, region_masks, sampling_regions,
            title=f"BAMLSS - {function_names[func_type]} - Undersampling - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        from utils.plotting import plot_uncertainties_entropy_undersampling
        plot_uncertainties_entropy_undersampling(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, region_masks, sampling_regions,
            title=f"BAMLSS - {function_names[func_type]} - Undersampling - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_undersampling(
            uncertainties_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'BAMLSS',
            date=date,
            nll_by_region=nll_by_region, crps_by_region=crps_by_region,
            spearman_aleatoric_by_region=spearman_aleatoric_by_region,
            spearman_epistemic_by_region=spearman_epistemic_by_region
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_undersampling(
            uncertainties_entropy_by_region, mse_by_region, sampling_regions,
            function_names[func_type], noise_type, func_type, 'BAMLSS',
            date=date
        )

