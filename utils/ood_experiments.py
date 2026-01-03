"""
Helper functions for running OOD (Out-of-Distribution) experiments across different models.

This module provides functions to run OOD experiments for various
uncertainty quantification models, handling the common pattern of:
1. Generating data with training range and OOD ranges
2. Training models on in-distribution data only
3. Evaluating uncertainties on both ID and OOD regions
4. Computing and saving statistics separately for ID, OOD, and combined
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing
from datetime import datetime

from utils.results_save import save_summary_statistics_ood, save_summary_statistics_entropy_ood, save_model_outputs
from utils.plotting import plot_uncertainties_ood, plot_uncertainties_entropy_ood
from utils.entropy_uncertainty import entropy_uncertainty_analytical, entropy_uncertainty_numerical
from utils.device import get_device_for_worker, get_num_gpus

# Set multiprocessing start method for CUDA support (Linux: can use 'fork' or 'spawn')
# Using 'spawn' for better CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


# ========== Data Generation Helper ==========

def generate_data_with_ood(generate_toy_regression_func, n_train, train_range, ood_ranges, 
                          grid_points, noise_type, func_type, seed=42, train_ranges=None, **kwargs):
    """
    Generate training data and evaluation grid with OOD regions.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (should accept ood_ranges and train_ranges parameters)
    n_train : int
        Number of training samples
    train_range : tuple
        (min, max) range for training data (for backward compatibility)
    train_ranges : list of tuples
        List of (min, max) ranges for training data, e.g., [(0, 5), (10, 15)]
        If provided, overrides train_range
    ood_ranges : list of tuples
        List of (min, max) ranges for OOD regions, e.g., [(30, 40), (50, 60)]
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
    tuple: (x_train, y_train, x_grid, y_grid_clean, ood_mask)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Call generate_toy_regression_func with ood_ranges and train_ranges parameters
    # It should now handle grid generation and OOD mask creation internally
    try:
        # Try calling with train_ranges and ood_ranges parameters (new signature)
        call_kwargs = {
            'n_train': n_train,
            'train_range': train_range,
            'ood_ranges': ood_ranges,
            'grid_points': grid_points,
            'noise_type': noise_type,
            'type': func_type,
            **kwargs
        }
        # Add train_ranges if provided
        if train_ranges is not None:
            call_kwargs['train_ranges'] = train_ranges
        
        result = generate_toy_regression_func(**call_kwargs)
        
        # Check if result has 5 values (new signature) or 4 values (old signature)
        if len(result) == 5:
            x_train, y_train, x_grid, y_grid_clean, ood_mask = result
            return (x_train.astype(np.float32), y_train.astype(np.float32),
                    x_grid.astype(np.float32), y_grid_clean.astype(np.float32), ood_mask)
        elif len(result) == 4:
            # Old signature - fall back to manual grid generation
            x_train, y_train, _, _ = result
            # Determine grid range: from train_range[0] to max of all OOD ranges
            max_ood_end = max([r[1] for r in ood_ranges]) if ood_ranges else train_range[1]
            grid_start = train_range[0]
            grid_end = max(train_range[1], max_ood_end)
            
            # Generate evaluation grid spanning both ID and OOD regions
            x_grid = np.linspace(grid_start, grid_end, grid_points).reshape(-1, 1)
            
            # Generate clean function values for the grid
            if func_type == "linear":
                f_clean = lambda x: 0.7 * x + 0.5
            elif func_type == "sin":
                f_clean = lambda x: x * np.sin(x) + x
            else:
                raise ValueError(f"Unknown func_type: {func_type}")
            
            y_grid_clean = f_clean(x_grid)
            
            # Create OOD mask: True for OOD regions, False for ID regions
            ood_mask = np.zeros(len(x_grid), dtype=bool)
            if ood_ranges:
                for ood_range in ood_ranges:
                    ood_start, ood_end = ood_range
                    ood_mask |= (x_grid[:, 0] >= ood_start) & (x_grid[:, 0] <= ood_end)
            
            return (x_train.astype(np.float32), y_train.astype(np.float32),
                    x_grid.astype(np.float32), y_grid_clean.astype(np.float32), ood_mask)
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
    except TypeError:
        # Function doesn't accept ood_ranges parameter - fall back to old method
        x_train, y_train, _, _ = generate_toy_regression_func(
            n_train=n_train,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            type=func_type,
            **kwargs
        )
        
        # Determine grid range: from train_range[0] to max of all OOD ranges
        max_ood_end = max([r[1] for r in ood_ranges]) if ood_ranges else train_range[1]
        grid_start = train_range[0]
        grid_end = max(train_range[1], max_ood_end)
        
        # Generate evaluation grid spanning both ID and OOD regions
        x_grid = np.linspace(grid_start, grid_end, grid_points).reshape(-1, 1)
        
        # Generate clean function values for the grid
        if func_type == "linear":
            f_clean = lambda x: 0.7 * x + 0.5
        elif func_type == "sin":
            f_clean = lambda x: x * np.sin(x) + x
        else:
            raise ValueError(f"Unknown func_type: {func_type}")
        
        y_grid_clean = f_clean(x_grid)
        
        # Create OOD mask: True for OOD regions, False for ID regions
        ood_mask = np.zeros(len(x_grid), dtype=bool)
        if ood_ranges:
            for ood_range in ood_ranges:
                ood_start, ood_end = ood_range
                ood_mask |= (x_grid[:, 0] >= ood_start) & (x_grid[:, 0] <= ood_end)
        
        return (x_train.astype(np.float32), y_train.astype(np.float32),
                x_grid.astype(np.float32), y_grid_clean.astype(np.float32), ood_mask)


# ========== Parallel execution wrapper functions ==========
# These functions are top-level to be picklable for multiprocessing

def _train_single_ood_mc_dropout(args):
    """Wrapper function for training MC Dropout for OOD experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, ood_mask,
     seed, p, beta, epochs, lr, batch_size, mc_samples, func_type, noise_type) = args
    
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
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = mc_dropout_predict(model, x_grid, M=mc_samples)
    
    # Compute MSE separately for ID and OOD
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    id_mask = ~ood_mask
    mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
    mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_id, mse_ood


def _train_single_ood_deep_ensemble(args):
    """Wrapper function for training Deep Ensemble for OOD experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, ood_mask,
     seed, beta, batch_size, K, func_type, noise_type) = args
    
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
        loss_type='beta_nll', beta=beta, parallel=True
    )
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = ensemble_predict_deep(ensemble, x_grid_norm)
    
    # Compute MSE separately for ID and OOD
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    id_mask = ~ood_mask
    mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
    mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_id, mse_ood


def _train_single_ood_bnn(args):
    """Wrapper function for training BNN for OOD experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, ood_mask,
     seed, hidden_width, weight_scale, warmup, samples, chains, func_type, noise_type) = args
    
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
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = bnn_predict(
        mcmc, x_grid_norm,
        hidden_width=hidden_width, weight_scale=weight_scale
    )
    
    # Compute MSE separately for ID and OOD
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    id_mask = ~ood_mask
    mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
    mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_id, mse_ood


def _train_single_ood_bamlss(args):
    """Wrapper function for training BAMLSS for OOD experiment (for parallel execution)."""
    (worker_id, x_train, y_train, x_grid, y_grid_clean, ood_mask,
     seed, n_iter, burnin, thin, nsamples, func_type, noise_type) = args
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    from Models.BAMLSS import bamlss_predict
    
    # BAMLSS fits directly - no normalization or training loop needed
    mu_pred, ale_var, epi_var, tot_var = bamlss_predict(
        x_train, y_train, x_grid,
        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples
    )
    
    # Compute MSE separately for ID and OOD
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    
    id_mask = ~ood_mask
    mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
    mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
    
    return mu_pred, ale_var, epi_var, tot_var, mse_id, mse_ood


# ========== Statistics Computation ==========

def compute_and_save_statistics_ood(
    uncertainties_id: dict,
    uncertainties_ood: dict,
    uncertainties_combined: dict,
    mse_id: float,
    mse_ood: float,
    mse_combined: float,
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
    Shared function to compute normalized statistics and save results for OOD experiments.
    
    This function normalizes uncertainties, computes averages and correlations,
    prints formatted statistics, and saves results separately for ID, OOD, and combined regions.
    
    Parameters:
    -----------
    uncertainties_id : dict
        Dictionary with 'ale', 'epi', 'tot' arrays for ID region
    uncertainties_ood : dict
        Dictionary with 'ale', 'epi', 'tot' arrays for OOD region
    uncertainties_combined : dict
        Dictionary with 'ale', 'epi', 'tot' arrays for combined (ID + OOD) region
    mse_id : float
        MSE for ID region
    mse_ood : float
        MSE for OOD region
    mse_combined : float
        MSE for combined region
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
    dict : Statistics dictionary with ID, OOD, and combined statistics
    """
    def normalize(values, vmin, vmax):
        """Normalize values to [0, 1] range"""
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    # Collect all values for normalization (across all regions)
    all_ale = np.concatenate([uncertainties_combined['ale']])
    all_epi = np.concatenate([uncertainties_combined['epi']])
    all_tot = np.concatenate([uncertainties_combined['tot']])
    
    # Compute min/max for normalization
    ale_min, ale_max = all_ale.min(), all_ale.max()
    epi_min, epi_max = all_epi.min(), all_epi.max()
    
    # Process each region
    regions = [
        ('ID', uncertainties_id, mse_id),
        ('OOD', uncertainties_ood, mse_ood),
        ('Combined', uncertainties_combined, mse_combined)
    ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"OOD Experiment Statistics - {function_name} Function - {model_name}")
    print(f"{'='*60}")
    print(f"\n{'Region':<12} {'Avg Aleatoric (norm)':<25} {'Avg Epistemic (norm)':<25} {'Avg Total (norm)':<25} {'Correlation (Epi-Ale)':<25} {'MSE':<15}")
    print("-" * 140)
    
    for region_name, uncertainties, mse in regions:
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
        
        print(f"{region_name:<12} {avg_ale_norm:>24.6f}  {avg_epi_norm:>24.6f}  {avg_tot_norm:>24.6f}  {correlation:>24.6f}  {mse:>15.6f}")
        
        # Save statistics for this region
        stats_df, fig = save_summary_statistics_ood(
            [avg_ale_norm], [avg_epi_norm], [avg_tot_norm], [correlation], [mse],
            function_name, noise_type=noise_type,
            func_type=func_type, model_name=model_name, region_type=region_name,
            date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets
        )
        plt.show()
        plt.close(fig)
        
        results[region_name.lower()] = {
            'avg_ale_norm': avg_ale_norm,
            'avg_epi_norm': avg_epi_norm,
            'avg_tot_norm': avg_tot_norm,
            'correlation': correlation,
            'mse': mse,
            'stats_df': stats_df
        }
    
    print(f"\n{'='*60}")
    print("Note: Average values are normalized to [0, 1] range across all regions")
    print("      Correlation is computed on original (non-normalized) uncertainty values")
    print(f"{'='*60}")
    
    return results


def compute_and_save_statistics_entropy_ood(
    uncertainties_id: dict,
    uncertainties_ood: dict,
    uncertainties_combined: dict,
    mse_id: float,
    mse_ood: float,
    mse_combined: float,
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
    Shared function to compute entropy-based statistics and save results for OOD experiments.
    
    This function computes averages and correlations for entropy-based uncertainties,
    prints formatted statistics, and saves results separately for ID, OOD, and combined regions.
    
    Parameters:
    -----------
    uncertainties_id : dict
        Dictionary with 'ale', 'epi', 'tot' entropy arrays for ID region
    uncertainties_ood : dict
        Dictionary with 'ale', 'epi', 'tot' entropy arrays for OOD region
    uncertainties_combined : dict
        Dictionary with 'ale', 'epi', 'tot' entropy arrays for combined (ID + OOD) region
    mse_id : float
        MSE for ID region
    mse_ood : float
        MSE for OOD region
    mse_combined : float
        MSE for combined region
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
    dict : Statistics dictionary with ID, OOD, and combined statistics
    """
    # Process each region
    regions = [
        ('ID', uncertainties_id, mse_id),
        ('OOD', uncertainties_ood, mse_ood),
        ('Combined', uncertainties_combined, mse_combined)
    ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"OOD Experiment Statistics (Entropy) - {function_name} Function - {model_name}")
    print(f"{'='*60}")
    print(f"\n{'Region':<12} {'Avg Aleatoric (nats)':<25} {'Avg Epistemic (nats)':<25} {'Avg Total (nats)':<25} {'Correlation (Epi-Ale)':<25} {'MSE':<15}")
    print("-" * 140)
    
    for region_name, uncertainties, mse in regions:
        ale_vals = uncertainties['ale']
        epi_vals = uncertainties['epi']
        tot_vals = uncertainties['tot']
        
        avg_ale_entropy = np.mean(ale_vals)
        avg_epi_entropy = np.mean(epi_vals)
        avg_tot_entropy = np.mean(tot_vals)
        
        correlation = np.corrcoef(epi_vals, ale_vals)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        print(f"{region_name:<12} {avg_ale_entropy:>24.6f}  {avg_epi_entropy:>24.6f}  {avg_tot_entropy:>24.6f}  {correlation:>24.6f}  {mse:>15.6f}")
        
        # Save statistics for this region
        stats_df, fig = save_summary_statistics_entropy_ood(
            [avg_ale_entropy], [avg_epi_entropy], [avg_tot_entropy], [correlation], [mse],
            function_name, noise_type=noise_type,
            func_type=func_type, model_name=model_name, region_type=region_name,
            date=date, dropout_p=dropout_p, mc_samples=mc_samples, n_nets=n_nets
        )
        plt.show()
        plt.close(fig)
        
        results[region_name.lower()] = {
            'avg_ale_entropy': avg_ale_entropy,
            'avg_epi_entropy': avg_epi_entropy,
            'avg_tot_entropy': avg_tot_entropy,
            'correlation': correlation,
            'mse': mse,
            'stats_df': stats_df
        }
    
    print(f"\n{'='*60}")
    print("Note: Entropy values are in nats (natural logarithm base)")
    print("      Correlation is computed on entropy values")
    print(f"{'='*60}")
    
    return results


# ========== Experiment Functions ==========

def run_mc_dropout_ood_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    ood_ranges: list = [(30, 40)],
    n_train: int = 1000,
    grid_points: int = 1000,
    seed: int = 42,
    p: float = 0.2,
    beta: float = 0.5,
    epochs: int = 700,
    lr: float = 1e-3,
    batch_size: int = 32,
    mc_samples: int = 20,
    parallel: bool = True,
    entropy_method: str = 'analytical'
):
    """
    Run OOD experiment for MC Dropout model.
    
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
    ood_ranges : list of tuples
        List of (min, max) ranges for OOD regions, e.g., [(30, 40), (50, 60)]
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
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - MC Dropout - OOD Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with OOD regions
        x_train, y_train, x_grid, y_grid_clean, ood_mask = generate_data_with_ood(
            generate_toy_regression_func, n_train, train_range, ood_ranges,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"OOD ranges: {ood_ranges}")
        print(f"Grid spans: [{x_grid[0, 0]:.2f}, {x_grid[-1, 0]:.2f}]")
        print(f"ID points: {np.sum(~ood_mask)}, OOD points: {np.sum(ood_mask)}\n")
        
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
        
        # Make predictions with raw arrays for entropy computation
        result = mc_dropout_predict(model, x_grid, M=mc_samples, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Split uncertainties by region
        id_mask = ~ood_mask
        
        uncertainties_id = {
            'ale': ale_var[id_mask] if ale_var.ndim == 1 else ale_var[id_mask].flatten(),
            'epi': epi_var[id_mask] if epi_var.ndim == 1 else epi_var[id_mask].flatten(),
            'tot': tot_var[id_mask] if tot_var.ndim == 1 else tot_var[id_mask].flatten()
        }
        
        uncertainties_ood = {
            'ale': ale_var[ood_mask] if ale_var.ndim == 1 else ale_var[ood_mask].flatten(),
            'epi': epi_var[ood_mask] if epi_var.ndim == 1 else epi_var[ood_mask].flatten(),
            'tot': tot_var[ood_mask] if tot_var.ndim == 1 else tot_var[ood_mask].flatten()
        }
        
        uncertainties_combined = {
            'ale': ale_var.flatten() if ale_var.ndim > 1 else ale_var,
            'epi': epi_var.flatten() if epi_var.ndim > 1 else epi_var,
            'tot': tot_var.flatten() if tot_var.ndim > 1 else tot_var
        }
        
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
        
        # Split entropy uncertainties by region
        uncertainties_entropy_id = {
            'ale': ale_entropy[id_mask] if ale_entropy.ndim == 1 else ale_entropy[id_mask].flatten(),
            'epi': epi_entropy[id_mask] if epi_entropy.ndim == 1 else epi_entropy[id_mask].flatten(),
            'tot': tot_entropy[id_mask] if tot_entropy.ndim == 1 else tot_entropy[id_mask].flatten()
        }
        
        uncertainties_entropy_ood = {
            'ale': ale_entropy[ood_mask] if ale_entropy.ndim == 1 else ale_entropy[ood_mask].flatten(),
            'epi': epi_entropy[ood_mask] if epi_entropy.ndim == 1 else epi_entropy[ood_mask].flatten(),
            'tot': tot_entropy[ood_mask] if tot_entropy.ndim == 1 else tot_entropy[ood_mask].flatten()
        }
        
        uncertainties_entropy_combined = {
            'ale': ale_entropy.flatten() if ale_entropy.ndim > 1 else ale_entropy,
            'epi': epi_entropy.flatten() if epi_entropy.ndim > 1 else epi_entropy,
            'tot': tot_entropy.flatten() if tot_entropy.ndim > 1 else tot_entropy
        }
        
        # Compute MSE separately
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
        mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
        mse_combined = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        
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
            subfolder='ood',
            dropout_p=p,
            mc_samples=mc_samples,
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, ood_mask,
            title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - OOD - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        plot_uncertainties_entropy_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask,
            title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - OOD - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_ood(
            uncertainties_id, uncertainties_ood, uncertainties_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'MC_Dropout',
            date=date, dropout_p=p, mc_samples=mc_samples
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_ood(
            uncertainties_entropy_id, uncertainties_entropy_ood, uncertainties_entropy_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'MC_Dropout',
            date=date, dropout_p=p, mc_samples=mc_samples
        )


def run_deep_ensemble_ood_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    ood_ranges: list = [(30, 40)],
    n_train: int = 1000,
    grid_points: int = 1000,
    seed: int = 42,
    beta: float = 0.5,
    batch_size: int = 32,
    K: int = 5,
    parallel: bool = True,
    entropy_method: str = 'analytical'
):
    """
    Run OOD experiment for Deep Ensemble model.
    
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
    ood_ranges : list of tuples
        List of (min, max) ranges for OOD regions
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
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - Deep Ensemble - OOD Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with OOD regions
        x_train, y_train, x_grid, y_grid_clean, ood_mask = generate_data_with_ood(
            generate_toy_regression_func, n_train, train_range, ood_ranges,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"OOD ranges: {ood_ranges}")
        print(f"Grid spans: [{x_grid[0, 0]:.2f}, {x_grid[-1, 0]:.2f}]")
        print(f"ID points: {np.sum(~ood_mask)}, OOD points: {np.sum(ood_mask)}\n")
        
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
            loss_type='beta_nll', beta=beta, parallel=True
        )
        
        # Make predictions with raw arrays for entropy computation
        result = ensemble_predict_deep(ensemble, x_grid_norm, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Split uncertainties by region
        id_mask = ~ood_mask
        
        uncertainties_id = {
            'ale': ale_var[id_mask] if ale_var.ndim == 1 else ale_var[id_mask].flatten(),
            'epi': epi_var[id_mask] if epi_var.ndim == 1 else epi_var[id_mask].flatten(),
            'tot': tot_var[id_mask] if tot_var.ndim == 1 else tot_var[id_mask].flatten()
        }
        
        uncertainties_ood = {
            'ale': ale_var[ood_mask] if ale_var.ndim == 1 else ale_var[ood_mask].flatten(),
            'epi': epi_var[ood_mask] if epi_var.ndim == 1 else epi_var[ood_mask].flatten(),
            'tot': tot_var[ood_mask] if tot_var.ndim == 1 else tot_var[ood_mask].flatten()
        }
        
        uncertainties_combined = {
            'ale': ale_var.flatten() if ale_var.ndim > 1 else ale_var,
            'epi': epi_var.flatten() if epi_var.ndim > 1 else epi_var,
            'tot': tot_var.flatten() if tot_var.ndim > 1 else tot_var
        }
        
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
        
        # Split entropy uncertainties by region
        uncertainties_entropy_id = {
            'ale': ale_entropy[id_mask] if ale_entropy.ndim == 1 else ale_entropy[id_mask].flatten(),
            'epi': epi_entropy[id_mask] if epi_entropy.ndim == 1 else epi_entropy[id_mask].flatten(),
            'tot': tot_entropy[id_mask] if tot_entropy.ndim == 1 else tot_entropy[id_mask].flatten()
        }
        
        uncertainties_entropy_ood = {
            'ale': ale_entropy[ood_mask] if ale_entropy.ndim == 1 else ale_entropy[ood_mask].flatten(),
            'epi': epi_entropy[ood_mask] if epi_entropy.ndim == 1 else epi_entropy[ood_mask].flatten(),
            'tot': tot_entropy[ood_mask] if tot_entropy.ndim == 1 else tot_entropy[ood_mask].flatten()
        }
        
        uncertainties_entropy_combined = {
            'ale': ale_entropy.flatten() if ale_entropy.ndim > 1 else ale_entropy,
            'epi': epi_entropy.flatten() if epi_entropy.ndim > 1 else epi_entropy,
            'tot': tot_entropy.flatten() if tot_entropy.ndim > 1 else tot_entropy
        }
        
        # Compute MSE separately
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
        mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
        mse_combined = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        
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
            subfolder='ood',
            n_nets=K,
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, ood_mask,
            title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - OOD - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        plot_uncertainties_entropy_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask,
            title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - OOD - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_ood(
            uncertainties_id, uncertainties_ood, uncertainties_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'Deep_Ensemble',
            date=date, n_nets=K
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_ood(
            uncertainties_entropy_id, uncertainties_entropy_ood, uncertainties_entropy_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'Deep_Ensemble',
            date=date, n_nets=K
        )


def run_bnn_ood_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    ood_ranges: list = [(30, 40)],
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
    Run OOD experiment for BNN (Bayesian Neural Network) model.
    
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
    ood_ranges : list of tuples
        List of (min, max) ranges for OOD regions
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
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - BNN - OOD Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with OOD regions
        x_train, y_train, x_grid, y_grid_clean, ood_mask = generate_data_with_ood(
            generate_toy_regression_func, n_train, train_range, ood_ranges,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"OOD ranges: {ood_ranges}")
        print(f"Grid spans: [{x_grid[0, 0]:.2f}, {x_grid[-1, 0]:.2f}]")
        print(f"ID points: {np.sum(~ood_mask)}, OOD points: {np.sum(ood_mask)}\n")
        
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
        
        # Make predictions with raw arrays for entropy computation
        result = bnn_predict(
            mcmc, x_grid_norm,
            hidden_width=hidden_width, weight_scale=weight_scale,
            return_raw_arrays=True
        )
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Split uncertainties by region
        id_mask = ~ood_mask
        
        uncertainties_id = {
            'ale': ale_var[id_mask] if ale_var.ndim == 1 else ale_var[id_mask].flatten(),
            'epi': epi_var[id_mask] if epi_var.ndim == 1 else epi_var[id_mask].flatten(),
            'tot': tot_var[id_mask] if tot_var.ndim == 1 else tot_var[id_mask].flatten()
        }
        
        uncertainties_ood = {
            'ale': ale_var[ood_mask] if ale_var.ndim == 1 else ale_var[ood_mask].flatten(),
            'epi': epi_var[ood_mask] if epi_var.ndim == 1 else epi_var[ood_mask].flatten(),
            'tot': tot_var[ood_mask] if tot_var.ndim == 1 else tot_var[ood_mask].flatten()
        }
        
        uncertainties_combined = {
            'ale': ale_var.flatten() if ale_var.ndim > 1 else ale_var,
            'epi': epi_var.flatten() if epi_var.ndim > 1 else epi_var,
            'tot': tot_var.flatten() if tot_var.ndim > 1 else tot_var
        }
        
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
        
        # Split entropy uncertainties by region
        uncertainties_entropy_id = {
            'ale': ale_entropy[id_mask] if ale_entropy.ndim == 1 else ale_entropy[id_mask].flatten(),
            'epi': epi_entropy[id_mask] if epi_entropy.ndim == 1 else epi_entropy[id_mask].flatten(),
            'tot': tot_entropy[id_mask] if tot_entropy.ndim == 1 else tot_entropy[id_mask].flatten()
        }
        
        uncertainties_entropy_ood = {
            'ale': ale_entropy[ood_mask] if ale_entropy.ndim == 1 else ale_entropy[ood_mask].flatten(),
            'epi': epi_entropy[ood_mask] if epi_entropy.ndim == 1 else epi_entropy[ood_mask].flatten(),
            'tot': tot_entropy[ood_mask] if tot_entropy.ndim == 1 else tot_entropy[ood_mask].flatten()
        }
        
        uncertainties_entropy_combined = {
            'ale': ale_entropy.flatten() if ale_entropy.ndim > 1 else ale_entropy,
            'epi': epi_entropy.flatten() if epi_entropy.ndim > 1 else epi_entropy,
            'tot': tot_entropy.flatten() if tot_entropy.ndim > 1 else tot_entropy
        }
        
        # Compute MSE separately
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
        mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
        mse_combined = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        
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
            subfolder='ood',
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, ood_mask,
            title=f"BNN (Pyro NUTS) - {function_names[func_type]} - OOD - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        plot_uncertainties_entropy_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask,
            title=f"BNN (Pyro NUTS) - {function_names[func_type]} - OOD - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_ood(
            uncertainties_id, uncertainties_ood, uncertainties_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'BNN',
            date=date
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_ood(
            uncertainties_entropy_id, uncertainties_entropy_ood, uncertainties_entropy_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'BNN',
            date=date
        )


def run_bamlss_ood_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    train_range: tuple = (-5, 10),
    ood_ranges: list = [(30, 40)],
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
    Run OOD experiment for BAMLSS model.
    
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
    ood_ranges : list of tuples
        List of (min, max) ranges for OOD regions
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
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    # Generate date at the start of the experiment
    date = datetime.now().strftime('%Y%m%d')
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - BAMLSS - OOD Experiment")
        print(f"{'#'*80}\n")
        
        # Generate data with OOD regions
        x_train, y_train, x_grid, y_grid_clean, ood_mask = generate_data_with_ood(
            generate_toy_regression_func, n_train, train_range, ood_ranges,
            grid_points, noise_type, func_type, seed
        )
        
        print(f"Training range: {train_range}")
        print(f"OOD ranges: {ood_ranges}")
        print(f"Grid spans: [{x_grid[0, 0]:.2f}, {x_grid[-1, 0]:.2f}]")
        print(f"ID points: {np.sum(~ood_mask)}, OOD points: {np.sum(ood_mask)}\n")
        
        # Train model
        print(f"{'='*60}")
        print(f"Fitting BAMLSS...")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # BAMLSS fits directly - get raw arrays for entropy computation
        result = bamlss_predict(
            x_train, y_train, x_grid,
            n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples,
            return_raw_arrays=True
        )
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Split uncertainties by region
        id_mask = ~ood_mask
        
        uncertainties_id = {
            'ale': ale_var[id_mask] if ale_var.ndim == 1 else ale_var[id_mask].flatten(),
            'epi': epi_var[id_mask] if epi_var.ndim == 1 else epi_var[id_mask].flatten(),
            'tot': tot_var[id_mask] if tot_var.ndim == 1 else tot_var[id_mask].flatten()
        }
        
        uncertainties_ood = {
            'ale': ale_var[ood_mask] if ale_var.ndim == 1 else ale_var[ood_mask].flatten(),
            'epi': epi_var[ood_mask] if epi_var.ndim == 1 else epi_var[ood_mask].flatten(),
            'tot': tot_var[ood_mask] if tot_var.ndim == 1 else tot_var[ood_mask].flatten()
        }
        
        uncertainties_combined = {
            'ale': ale_var.flatten() if ale_var.ndim > 1 else ale_var,
            'epi': epi_var.flatten() if epi_var.ndim > 1 else epi_var,
            'tot': tot_var.flatten() if tot_var.ndim > 1 else tot_var
        }
        
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
        
        # Split entropy uncertainties by region
        uncertainties_entropy_id = {
            'ale': ale_entropy[id_mask] if ale_entropy.ndim == 1 else ale_entropy[id_mask].flatten(),
            'epi': epi_entropy[id_mask] if epi_entropy.ndim == 1 else epi_entropy[id_mask].flatten(),
            'tot': tot_entropy[id_mask] if tot_entropy.ndim == 1 else tot_entropy[id_mask].flatten()
        }
        
        uncertainties_entropy_ood = {
            'ale': ale_entropy[ood_mask] if ale_entropy.ndim == 1 else ale_entropy[ood_mask].flatten(),
            'epi': epi_entropy[ood_mask] if epi_entropy.ndim == 1 else epi_entropy[ood_mask].flatten(),
            'tot': tot_entropy[ood_mask] if tot_entropy.ndim == 1 else tot_entropy[ood_mask].flatten()
        }
        
        uncertainties_entropy_combined = {
            'ale': ale_entropy.flatten() if ale_entropy.ndim > 1 else ale_entropy,
            'epi': epi_entropy.flatten() if epi_entropy.ndim > 1 else epi_entropy,
            'tot': tot_entropy.flatten() if tot_entropy.ndim > 1 else tot_entropy
        }
        
        # Compute MSE separately
        mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
        y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
        
        mse_id = np.mean((mu_pred_flat[id_mask] - y_grid_clean_flat[id_mask])**2)
        mse_ood = np.mean((mu_pred_flat[ood_mask] - y_grid_clean_flat[ood_mask])**2)
        mse_combined = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        
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
            subfolder='ood',
            date=date
        )
        
        # Plot variance-based uncertainties
        plot_uncertainties_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var, ood_mask,
            title=f"BAMLSS - {function_names[func_type]} - OOD - Variance",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Plot entropy-based uncertainties
        plot_uncertainties_entropy_ood(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask,
            title=f"BAMLSS - {function_names[func_type]} - OOD - Entropy",
            noise_type=noise_type,
            func_type=func_type
        )
        
        # Compute and save variance-based statistics
        compute_and_save_statistics_ood(
            uncertainties_id, uncertainties_ood, uncertainties_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'BAMLSS',
            date=date
        )
        
        # Compute and save entropy-based statistics
        compute_and_save_statistics_entropy_ood(
            uncertainties_entropy_id, uncertainties_entropy_ood, uncertainties_entropy_combined,
            mse_id, mse_ood, mse_combined,
            function_names[func_type], noise_type, func_type, 'BAMLSS',
            date=date
        )

