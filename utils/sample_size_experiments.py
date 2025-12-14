"""
Helper functions for running sample size experiments across different models.

This module provides functions to run sample size experiments for various
uncertainty quantification models, handling the common pattern of:
1. Generating/subsampling data
2. Training models at different sample sizes
3. Collecting uncertainties
4. Computing and saving statistics
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing

from utils.results_save import save_summary_statistics
from utils.plotting import plot_uncertainties_no_ood
from utils.device import get_device_for_worker, get_num_gpus

# Set multiprocessing start method for CUDA support (Linux: can use 'fork' or 'spawn')
# Using 'spawn' for better CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


# ========== Parallel execution wrapper functions ==========
# These functions are top-level to be picklable for multiprocessing

def _train_single_percentage_mc_dropout(args):
    """Wrapper function for training MC Dropout at a single percentage (for parallel execution)."""
    (worker_id, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
     seed, p, beta, epochs, lr, batch_size, mc_samples, func_type, noise_type) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    torch.cuda.set_device(device) if device.type == 'cuda' else None
    
    # Set seed for reproducibility
    np.random.seed(seed + int(pct))
    torch.manual_seed(seed + int(pct))
    
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    # Subsample data
    n_train_full_actual = len(x_train_full)
    n_samples = int(n_train_full_actual * pct / 100)
    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
    x_train_subset = x_train_full[indices]
    y_train_subset = y_train_full[indices]
    
    # Create dataloader
    ds = TensorDataset(torch.from_numpy(x_train_subset), torch.from_numpy(y_train_subset))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Train model
    model = MCDropoutRegressor(p=p)
    train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = mc_dropout_predict(model, x_grid, M=mc_samples)
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return pct, mu_pred, ale_var, epi_var, tot_var, mse


def _train_single_percentage_deep_ensemble(args):
    """Wrapper function for training Deep Ensemble at a single percentage (for parallel execution)."""
    (worker_id, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
     seed, beta, batch_size, K, func_type, noise_type) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    torch.cuda.set_device(device) if device.type == 'cuda' else None
    
    # Set seed for reproducibility
    np.random.seed(seed + int(pct))
    torch.manual_seed(seed + int(pct))
    
    from Models.Deep_Ensemble import (
        train_ensemble_deep,
        ensemble_predict_deep
    )
    from Models.MC_Dropout import (
        normalize_x,
        normalize_x_data
    )
    
    # Subsample data
    n_train_full_actual = len(x_train_full)
    n_samples = int(n_train_full_actual * pct / 100)
    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
    x_train_subset = x_train_full[indices]
    y_train_subset = y_train_full[indices]
    
    # Normalize input
    x_mean, x_std = normalize_x(x_train_subset)
    x_train_subset_norm = normalize_x_data(x_train_subset, x_mean, x_std)
    x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
    
    # Train ensemble
    ensemble = train_ensemble_deep(
        x_train_subset_norm, y_train_subset,
        batch_size=batch_size, K=K,
        loss_type='beta_nll', beta=beta, parallel=True
    )
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = ensemble_predict_deep(ensemble, x_grid_norm)
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return pct, mu_pred, ale_var, epi_var, tot_var, mse


def _train_single_percentage_bnn(args):
    """Wrapper function for training BNN at a single percentage (for parallel execution)."""
    (worker_id, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
     seed, hidden_width, weight_scale, warmup, samples, chains, func_type, noise_type) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    torch.cuda.set_device(device) if device.type == 'cuda' else None
    
    # Set seed for reproducibility
    np.random.seed(seed + int(pct))
    torch.manual_seed(seed + int(pct))
    
    from Models.BNN import (
        train_bnn,
        bnn_predict,
        normalize_x as bnn_normalize_x,
        normalize_x_data as bnn_normalize_x_data
    )
    
    # Subsample data
    n_train_full_actual = len(x_train_full)
    n_samples = int(n_train_full_actual * pct / 100)
    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
    x_train_subset = x_train_full[indices]
    y_train_subset = y_train_full[indices]
    
    # Normalize input
    x_mean, x_std = bnn_normalize_x(x_train_subset)
    x_train_subset_norm = bnn_normalize_x_data(x_train_subset, x_mean, x_std)
    x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
    
    # Train BNN with MCMC
    mcmc = train_bnn(
        x_train_subset_norm, y_train_subset,
        hidden_width=hidden_width, weight_scale=weight_scale,
        warmup=warmup, samples=samples, chains=chains, seed=seed + int(pct)
    )
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = bnn_predict(
        mcmc, x_grid_norm,
        hidden_width=hidden_width, weight_scale=weight_scale
    )
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return pct, mu_pred, ale_var, epi_var, tot_var, mse


def _train_single_percentage_bamlss(args):
    """Wrapper function for training BAMLSS at a single percentage (for parallel execution)."""
    (worker_id, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
     seed, n_iter, burnin, thin, nsamples, func_type, noise_type) = args
    
    # Set seed for reproducibility
    np.random.seed(seed + int(pct))
    torch.manual_seed(seed + int(pct))
    
    from Models.BAMLSS import bamlss_predict
    
    # Subsample data
    n_train_full_actual = len(x_train_full)
    n_samples = int(n_train_full_actual * pct / 100)
    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
    x_train_subset = x_train_full[indices]
    y_train_subset = y_train_full[indices]
    
    # BAMLSS fits directly - no normalization or training loop needed
    mu_pred, ale_var, epi_var, tot_var = bamlss_predict(
        x_train_subset, y_train_subset, x_grid,
        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples
    )
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return pct, mu_pred, ale_var, epi_var, tot_var, mse


def compute_and_save_statistics(
    uncertainties_by_pct: dict,
    percentages: list,
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    mse_by_pct: dict = None
):
    """
    Shared function to compute normalized statistics and save results.
    
    This function normalizes uncertainties across all percentages, computes
    averages and correlations, prints formatted statistics, and saves results.
    
    Parameters:
    -----------
    uncertainties_by_pct : dict
        Dictionary mapping percentage to dict with 'ale', 'epi', 'tot' lists
    percentages : list
        List of percentages tested
    function_name : str
        Human-readable function name (e.g., "Linear", "Sinusoidal")
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str
        Function type identifier (e.g., 'linear', 'sin')
    model_name : str
        Model name for saving results (e.g., 'MC_Dropout', 'Deep_Ensemble')
    mse_by_pct : dict, optional
        Dictionary mapping percentage to list of MSE values
    
    Returns:
    --------
    dict : Statistics dictionary with percentages, averages, correlations, MSE, and stats_df
    """
    # Collect all values for normalization
    all_ale = np.concatenate([np.concatenate(uncertainties_by_pct[pct]['ale']) 
                              for pct in percentages])
    all_epi = np.concatenate([np.concatenate(uncertainties_by_pct[pct]['epi']) 
                              for pct in percentages])
    all_tot = np.concatenate([np.concatenate(uncertainties_by_pct[pct]['tot']) 
                              for pct in percentages])
    
    # Compute min/max for normalization
    ale_min, ale_max = all_ale.min(), all_ale.max()
    epi_min, epi_max = all_epi.min(), all_epi.max()
    
    def normalize(values, vmin, vmax):
        """Normalize values to [0, 1] range"""
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    # Compute statistics for each percentage
    avg_ale_norm_list = []
    avg_epi_norm_list = []
    avg_tot_norm_list = []
    correlation_list = []
    mse_list = []
    
    print(f"\n{'='*60}")
    print(f"Normalized Average Uncertainties by Percentage - {function_name} Function - {model_name}")
    print(f"{'='*60}")
    if mse_by_pct is not None:
        print(f"\n{'Percentage':<12} {'Avg Aleatoric (norm)':<25} {'Avg Epistemic (norm)':<25} {'Avg Total (norm)':<25} {'Correlation (Epi-Ale)':<25} {'MSE':<15}")
        print("-" * 140)
    else:
        print(f"\n{'Percentage':<12} {'Avg Aleatoric (norm)':<25} {'Avg Epistemic (norm)':<25} {'Avg Total (norm)':<25} {'Correlation (Epi-Ale)':<25}")
        print("-" * 120)
    
    for pct in percentages:
        ale_vals = np.concatenate(uncertainties_by_pct[pct]['ale'])
        epi_vals = np.concatenate(uncertainties_by_pct[pct]['epi'])
        tot_vals = np.concatenate(uncertainties_by_pct[pct]['tot'])
        
        ale_norm = normalize(ale_vals, ale_min, ale_max)
        epi_norm = normalize(epi_vals, epi_min, epi_max)
        tot_norm = epi_norm + ale_norm
        
        avg_ale_norm = np.mean(ale_norm)
        avg_epi_norm = np.mean(epi_norm)
        avg_tot_norm = np.mean(tot_norm)
        
        correlation = np.corrcoef(epi_vals, ale_vals)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        avg_ale_norm_list.append(avg_ale_norm)
        avg_epi_norm_list.append(avg_epi_norm)
        avg_tot_norm_list.append(avg_tot_norm)
        correlation_list.append(correlation)
        
        # Compute average MSE if provided
        if mse_by_pct is not None and pct in mse_by_pct:
            avg_mse = np.mean(mse_by_pct[pct])
            mse_list.append(avg_mse)
            print(f"{pct:>3}%        {avg_ale_norm:>24.6f}  {avg_epi_norm:>24.6f}  {avg_tot_norm:>24.6f}  {correlation:>24.6f}  {avg_mse:>15.6f}")
        else:
            print(f"{pct:>3}%        {avg_ale_norm:>24.6f}  {avg_epi_norm:>24.6f}  {avg_tot_norm:>24.6f}  {correlation:>24.6f}")
    
    print(f"\n{'='*60}")
    print("Note: Average values are normalized to [0, 1] range across all percentages")
    print("      Correlation is computed on original (non-normalized) uncertainty values")
    print(f"{'='*60}")
    
    # Save summary statistics
    stats_df, fig = save_summary_statistics(
        percentages, avg_ale_norm_list, avg_epi_norm_list,
        avg_tot_norm_list, correlation_list,
        function_name, noise_type=noise_type,
        func_type=func_type, model_name=model_name,
        mse_list=mse_list if mse_list else None
    )
    plt.show()
    plt.close(fig)
    
    result = {
        'percentages': percentages,
        'avg_ale_norm': avg_ale_norm_list,
        'avg_epi_norm': avg_epi_norm_list,
        'avg_tot_norm': avg_tot_norm_list,
        'correlations': correlation_list,
        'stats_df': stats_df
    }
    
    if mse_list:
        result['mse'] = mse_list
    
    return result


def run_mc_dropout_sample_size_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    percentages: list = [5, 10, 15, 25, 50, 100],
    n_train_full: int = 1000,
    train_range: tuple = (10, 30),
    grid_points: int = 600,
    seed: int = 42,
    p: float = 0.1,
    beta: float = 0.5,
    epochs: int = 700,
    lr: float = 1e-3,
    batch_size: int = 32,
    mc_samples: int = 20,
    parallel: bool = True
):
    """
    Run sample size experiment for MC Dropout model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test (e.g., ['linear', 'sin'])
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    percentages : list
        List of training data percentages to test
    n_train_full : int
        Full training dataset size
    train_range : tuple
        Range for training data
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
    """
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - MC Dropout")
        print(f"{'#'*80}\n")
        
        # Generate data
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x_train_full, y_train_full, x_grid, y_grid_clean = generate_toy_regression_func(
            n_train=n_train_full,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            type=func_type
        )
        
        n_train_full_actual = len(x_train_full)
        uncertainties_by_pct = {pct: {'ale': [], 'epi': [], 'tot': []} for pct in percentages}
        mse_by_pct = {pct: [] for pct in percentages}
        
        # Prepare arguments for parallel execution
        num_gpus = get_num_gpus()
        use_gpu = num_gpus > 0 and parallel
        
        if parallel and len(percentages) > 1:
            # Determine number of workers
            if use_gpu:
                max_workers = min(len(percentages), num_gpus)
                print(f"Using GPU parallelization with {max_workers} workers across {num_gpus} GPU(s)")
            else:
                max_workers = min(len(percentages), multiprocessing.cpu_count())
                print(f"Using CPU parallelization with {max_workers} workers")
            
            # Prepare arguments for each worker
            args_list = []
            for idx, pct in enumerate(percentages):
                args = (idx, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
                       seed, p, beta, epochs, lr, batch_size, mc_samples, func_type, noise_type)
                args_list.append(args)
            
            # Execute in parallel
            try:
                if use_gpu:
                    # Use torch.multiprocessing for GPU support
                    with mp.Pool(processes=max_workers) as pool:
                        results = pool.map(_train_single_percentage_mc_dropout, args_list)
                else:
                    # Use ProcessPoolExecutor for CPU
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(_train_single_percentage_mc_dropout, args_list))
                
                # Process results and plot
                for pct, mu_pred, ale_var, epi_var, tot_var, mse in results:
                    uncertainties_by_pct[pct]['ale'].append(ale_var)
                    uncertainties_by_pct[pct]['epi'].append(epi_var)
                    uncertainties_by_pct[pct]['tot'].append(tot_var)
                    mse_by_pct[pct].append(mse)
                    
                    # Get subset for plotting
                    n_samples = int(n_train_full_actual * pct / 100)
                    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                    x_train_subset = x_train_full[indices]
                    y_train_subset = y_train_full[indices]
                    
                    # Plot with stored predictions
                    plot_uncertainties_no_ood(
                        x_train_subset, y_train_subset, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - {pct}% training data",
                        noise_type=noise_type,
                        func_type=func_type
                    )
            except Exception as e:
                print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                parallel = False
        
        # Sequential execution (fallback or if parallel=False)
        if not parallel or len(percentages) == 1:
            for pct in percentages:
                print(f"\n{'='*60}")
                print(f"Training with {pct}% of training data ({int(n_train_full_actual * pct / 100)} samples)")
                print(f"{'='*60}")
                
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Subsample data
                n_samples = int(n_train_full_actual * pct / 100)
                indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                x_train_subset = x_train_full[indices]
                y_train_subset = y_train_full[indices]
                
                # Create dataloader
                ds = TensorDataset(torch.from_numpy(x_train_subset), torch.from_numpy(y_train_subset))
                loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                
                # Train model
                model = MCDropoutRegressor(p=p)
                train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
                
                # Make predictions
                mu_pred, ale_var, epi_var, tot_var = mc_dropout_predict(model, x_grid, M=mc_samples)
                
                # Compute MSE (handle shape differences)
                mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                
                # Store uncertainties and MSE
                uncertainties_by_pct[pct]['ale'].append(ale_var)
                uncertainties_by_pct[pct]['epi'].append(epi_var)
                uncertainties_by_pct[pct]['tot'].append(tot_var)
                mse_by_pct[pct].append(mse)
                
                # Plot
                plot_uncertainties_no_ood(
                    x_train_subset, y_train_subset, x_grid, y_grid_clean,
                    mu_pred, ale_var, epi_var, tot_var,
                    title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - {pct}% training data",
                    noise_type=noise_type,
                    func_type=func_type
                )
        
        # Compute and save statistics
        compute_and_save_statistics(
            uncertainties_by_pct, percentages,
            function_names[func_type], noise_type, func_type, 'MC_Dropout',
            mse_by_pct=mse_by_pct
        )


def run_deep_ensemble_sample_size_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    percentages: list = [5, 10, 15, 25, 50, 100],
    n_train_full: int = 1000,
    train_range: tuple = (10, 30),
    grid_points: int = 600,
    seed: int = 42,
    beta: float = 0.5,
    batch_size: int = 32,
    K: int = 5,
    parallel: bool = True
):
    """
    Run sample size experiment for Deep Ensemble model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    percentages : list
        List of training data percentages to test
    n_train_full : int
        Full training dataset size
    train_range : tuple
        Range for training data
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
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - Deep Ensemble")
        print(f"{'#'*80}\n")
        
        # Generate data
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x_train_full, y_train_full, x_grid, y_grid_clean = generate_toy_regression_func(
            n_train=n_train_full,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            type=func_type
        )
        
        n_train_full_actual = len(x_train_full)
        uncertainties_by_pct = {pct: {'ale': [], 'epi': [], 'tot': []} for pct in percentages}
        mse_by_pct = {pct: [] for pct in percentages}
        
        # Prepare arguments for parallel execution
        num_gpus = get_num_gpus()
        use_gpu = num_gpus > 0 and parallel
        
        if parallel and len(percentages) > 1:
            max_workers = min(len(percentages), num_gpus if use_gpu else multiprocessing.cpu_count())
            print(f"Using {'GPU' if use_gpu else 'CPU'} parallelization with {max_workers} workers")
            
            args_list = [(idx, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
                         seed, beta, batch_size, K, func_type, noise_type)
                        for idx, pct in enumerate(percentages)]
            
            try:
                if use_gpu:
                    with mp.Pool(processes=max_workers) as pool:
                        results = pool.map(_train_single_percentage_deep_ensemble, args_list)
                else:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(_train_single_percentage_deep_ensemble, args_list))
                
                for pct, mu_pred, ale_var, epi_var, tot_var, mse in results:
                    uncertainties_by_pct[pct]['ale'].append(ale_var)
                    uncertainties_by_pct[pct]['epi'].append(epi_var)
                    uncertainties_by_pct[pct]['tot'].append(tot_var)
                    mse_by_pct[pct].append(mse)
                    
                    n_samples = int(n_train_full_actual * pct / 100)
                    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                    x_train_subset = x_train_full[indices]
                    y_train_subset = y_train_full[indices]
                    
                    plot_uncertainties_no_ood(
                        x_train_subset, y_train_subset, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - {pct}% training data",
                        noise_type=noise_type,
                        func_type=func_type
                    )
            except Exception as e:
                print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                parallel = False
        
        # Sequential execution
        if not parallel or len(percentages) == 1:
            for pct in percentages:
                print(f"\n{'='*60}")
                print(f"Training with {pct}% of training data ({int(n_train_full_actual * pct / 100)} samples)")
                print(f"{'='*60}")
                
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                n_samples = int(n_train_full_actual * pct / 100)
                indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                x_train_subset = x_train_full[indices]
                y_train_subset = y_train_full[indices]
                
                x_mean, x_std = normalize_x(x_train_subset)
                x_train_subset_norm = normalize_x_data(x_train_subset, x_mean, x_std)
                x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
                
                ensemble = train_ensemble_deep(
                    x_train_subset_norm, y_train_subset,
                    batch_size=batch_size, K=K,
                    loss_type='beta_nll', beta=beta, parallel=True
                )
                
                mu_pred, ale_var, epi_var, tot_var = ensemble_predict_deep(ensemble, x_grid_norm)
                
                mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                
                uncertainties_by_pct[pct]['ale'].append(ale_var)
                uncertainties_by_pct[pct]['epi'].append(epi_var)
                uncertainties_by_pct[pct]['tot'].append(tot_var)
                mse_by_pct[pct].append(mse)
                
                plot_uncertainties_no_ood(
                    x_train_subset, y_train_subset, x_grid, y_grid_clean,
                    mu_pred, ale_var, epi_var, tot_var,
                    title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - {pct}% training data",
                    noise_type=noise_type,
                    func_type=func_type
                )
        
        # Compute and save statistics
        compute_and_save_statistics(
            uncertainties_by_pct, percentages,
            function_names[func_type], noise_type, func_type, 'Deep_Ensemble',
            mse_by_pct=mse_by_pct
        )


def run_bnn_sample_size_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    percentages: list = [5, 10, 15, 25, 50, 100],
    n_train_full: int = 1000,
    train_range: tuple = (10, 30),
    grid_points: int = 600,
    seed: int = 42,
    hidden_width: int = 16,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    chains: int = 1,
    parallel: bool = True
):
    """
    Run sample size experiment for BNN (Bayesian Neural Network) model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    percentages : list
        List of training data percentages to test
    n_train_full : int
        Full training dataset size
    train_range : tuple
        Range for training data
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
    """
    from Models.BNN import (
        train_bnn,
        bnn_predict,
        normalize_x as bnn_normalize_x,
        normalize_x_data as bnn_normalize_x_data
    )
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - BNN")
        print(f"{'#'*80}\n")
        
        # Generate data
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x_train_full, y_train_full, x_grid, y_grid_clean = generate_toy_regression_func(
            n_train=n_train_full,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            type=func_type
        )
        
        n_train_full_actual = len(x_train_full)
        uncertainties_by_pct = {pct: {'ale': [], 'epi': [], 'tot': []} for pct in percentages}
        mse_by_pct = {pct: [] for pct in percentages}
        
        num_gpus = get_num_gpus()
        use_gpu = num_gpus > 0 and parallel
        
        if parallel and len(percentages) > 1:
            max_workers = min(len(percentages), num_gpus if use_gpu else multiprocessing.cpu_count())
            print(f"Using {'GPU' if use_gpu else 'CPU'} parallelization with {max_workers} workers")
            
            args_list = [(idx, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
                         seed, hidden_width, weight_scale, warmup, samples, chains, func_type, noise_type)
                        for idx, pct in enumerate(percentages)]
            
            try:
                if use_gpu:
                    with mp.Pool(processes=max_workers) as pool:
                        results = pool.map(_train_single_percentage_bnn, args_list)
                else:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(_train_single_percentage_bnn, args_list))
                
                for pct, mu_pred, ale_var, epi_var, tot_var, mse in results:
                    uncertainties_by_pct[pct]['ale'].append(ale_var)
                    uncertainties_by_pct[pct]['epi'].append(epi_var)
                    uncertainties_by_pct[pct]['tot'].append(tot_var)
                    mse_by_pct[pct].append(mse)
                    
                    n_samples = int(n_train_full_actual * pct / 100)
                    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                    x_train_subset = x_train_full[indices]
                    y_train_subset = y_train_full[indices]
                    
                    plot_uncertainties_no_ood(
                        x_train_subset, y_train_subset, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"BNN (Pyro NUTS) - {function_names[func_type]} - {pct}% training data",
                        noise_type=noise_type,
                        func_type=func_type
                    )
            except Exception as e:
                print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                parallel = False
        
        if not parallel or len(percentages) == 1:
            for pct in percentages:
                print(f"\n{'='*60}")
                print(f"Training with {pct}% of training data ({int(n_train_full_actual * pct / 100)} samples)")
                print(f"{'='*60}")
                
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                n_samples = int(n_train_full_actual * pct / 100)
                indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                x_train_subset = x_train_full[indices]
                y_train_subset = y_train_full[indices]
                
                x_mean, x_std = bnn_normalize_x(x_train_subset)
                x_train_subset_norm = bnn_normalize_x_data(x_train_subset, x_mean, x_std)
                x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
                
                mcmc = train_bnn(
                    x_train_subset_norm, y_train_subset,
                    hidden_width=hidden_width, weight_scale=weight_scale,
                    warmup=warmup, samples=samples, chains=chains, seed=seed
                )
                
                mu_pred, ale_var, epi_var, tot_var = bnn_predict(
                    mcmc, x_grid_norm,
                    hidden_width=hidden_width, weight_scale=weight_scale
                )
                
                mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                
                uncertainties_by_pct[pct]['ale'].append(ale_var)
                uncertainties_by_pct[pct]['epi'].append(epi_var)
                uncertainties_by_pct[pct]['tot'].append(tot_var)
                mse_by_pct[pct].append(mse)
                
                plot_uncertainties_no_ood(
                    x_train_subset, y_train_subset, x_grid, y_grid_clean,
                    mu_pred, ale_var, epi_var, tot_var,
                    title=f"BNN (Pyro NUTS) - {function_names[func_type]} - {pct}% training data",
                    noise_type=noise_type,
                    func_type=func_type
                )
        
        # Compute and save statistics
        compute_and_save_statistics(
            uncertainties_by_pct, percentages,
            function_names[func_type], noise_type, func_type, 'BNN',
            mse_by_pct=mse_by_pct
        )


def run_bamlss_sample_size_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    percentages: list = [5, 10, 15, 25, 50, 100],
    n_train_full: int = 1000,
    train_range: tuple = (10, 30),
    grid_points: int = 600,
    seed: int = 42,
    n_iter: int = 12000,
    burnin: int = 2000,
    thin: int = 10,
    nsamples: int = 1000,
    parallel: bool = True
):
    """
    Run sample size experiment for BAMLSS model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    percentages : list
        List of training data percentages to test
    n_train_full : int
        Full training dataset size
    train_range : tuple
        Range for training data
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
    """
    from Models.BAMLSS import bamlss_predict
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    for func_type in function_types:
        print(f"\n{'#'*80}")
        print(f"# Function Type: {function_names[func_type]} ({func_type}) - BAMLSS")
        print(f"{'#'*80}\n")
        
        # Generate data
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        x_train_full, y_train_full, x_grid, y_grid_clean = generate_toy_regression_func(
            n_train=n_train_full,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            type=func_type
        )
        
        n_train_full_actual = len(x_train_full)
        uncertainties_by_pct = {pct: {'ale': [], 'epi': [], 'tot': []} for pct in percentages}
        mse_by_pct = {pct: [] for pct in percentages}
        
        # BAMLSS uses R, so CPU-only parallelization
        if parallel and len(percentages) > 1:
            max_workers = min(len(percentages), multiprocessing.cpu_count())
            print(f"Using CPU parallelization with {max_workers} workers (BAMLSS is CPU-only)")
            
            args_list = [(idx, pct, x_train_full, y_train_full, x_grid, y_grid_clean,
                         seed, n_iter, burnin, thin, nsamples, func_type, noise_type)
                        for idx, pct in enumerate(percentages)]
            
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(_train_single_percentage_bamlss, args_list))
                
                for pct, mu_pred, ale_var, epi_var, tot_var, mse in results:
                    uncertainties_by_pct[pct]['ale'].append(ale_var)
                    uncertainties_by_pct[pct]['epi'].append(epi_var)
                    uncertainties_by_pct[pct]['tot'].append(tot_var)
                    mse_by_pct[pct].append(mse)
                    
                    n_samples = int(n_train_full_actual * pct / 100)
                    indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                    x_train_subset = x_train_full[indices]
                    y_train_subset = y_train_full[indices]
                    
                    plot_uncertainties_no_ood(
                        x_train_subset, y_train_subset, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"BAMLSS - {function_names[func_type]} - {pct}% training data",
                        noise_type=noise_type,
                        func_type=func_type
                    )
            except Exception as e:
                print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                parallel = False
        
        if not parallel or len(percentages) == 1:
            for pct in percentages:
                print(f"\n{'='*60}")
                print(f"Training with {pct}% of training data ({int(n_train_full_actual * pct / 100)} samples)")
                print(f"{'='*60}")
                
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                n_samples = int(n_train_full_actual * pct / 100)
                indices = np.random.choice(n_train_full_actual, size=n_samples, replace=False)
                x_train_subset = x_train_full[indices]
                y_train_subset = y_train_full[indices]
                
                mu_pred, ale_var, epi_var, tot_var = bamlss_predict(
                    x_train_subset, y_train_subset, x_grid,
                    n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples
                )
                
                mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                
                uncertainties_by_pct[pct]['ale'].append(ale_var)
                uncertainties_by_pct[pct]['epi'].append(epi_var)
                uncertainties_by_pct[pct]['tot'].append(tot_var)
                mse_by_pct[pct].append(mse)
                
                plot_uncertainties_no_ood(
                    x_train_subset, y_train_subset, x_grid, y_grid_clean,
                    mu_pred, ale_var, epi_var, tot_var,
                    title=f"BAMLSS - {function_names[func_type]} - {pct}% training data",
                    noise_type=noise_type,
                    func_type=func_type
                )
        
        # Compute and save statistics
        compute_and_save_statistics(
            uncertainties_by_pct, percentages,
            function_names[func_type], noise_type, func_type, 'BAMLSS',
            mse_by_pct=mse_by_pct
        )

