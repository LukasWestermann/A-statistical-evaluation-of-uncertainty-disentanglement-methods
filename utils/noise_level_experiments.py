"""
Helper functions for running noise level experiments across different models.

This module provides functions to run noise level experiments for various
uncertainty quantification models, handling the common pattern of:
1. Generating data with different tau (noise level) values
2. Training models at different noise levels
3. Collecting uncertainties and MSE
4. Computing and saving statistics
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing

from utils.results_save import save_summary_statistics_noise_level
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

def _train_single_tau_mc_dropout(args):
    """Wrapper function for training MC Dropout at a single tau value (for parallel execution)."""
    (worker_id, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
     seed, p, beta, epochs, lr, batch_size, mc_samples, func_type, noise_type) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Set seed for reproducibility
    np.random.seed(seed + int(tau * 100))
    torch.manual_seed(seed + int(tau * 100))
    
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict,
        normalize_x,
        normalize_x_data
    )
    
    # Normalize input (using passed data)
    x_mean, x_std = normalize_x(x_train)
    x_train_norm = normalize_x_data(x_train, x_mean, x_std)
    x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
    
    # Create dataloader
    ds = TensorDataset(torch.from_numpy(x_train_norm), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Train model
    model = MCDropoutRegressor(p=p)
    train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
    
    # Make predictions
    mu_pred, ale_var, epi_var, tot_var = mc_dropout_predict(model, x_grid_norm, M=mc_samples)
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return tau, distribution, mu_pred, ale_var, epi_var, tot_var, mse


def _train_single_tau_deep_ensemble(args):
    """Wrapper function for training Deep Ensemble at a single tau value (for parallel execution)."""
    (worker_id, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
     seed, beta, batch_size, K, func_type, noise_type) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Set seed for reproducibility
    np.random.seed(seed + int(tau * 100))
    torch.manual_seed(seed + int(tau * 100))
    
    from Models.Deep_Ensemble import (
        train_ensemble_deep,
        ensemble_predict_deep
    )
    from Models.MC_Dropout import (
        normalize_x,
        normalize_x_data
    )
    
    # Normalize input (using passed data)
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
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return tau, distribution, mu_pred, ale_var, epi_var, tot_var, mse


def _train_single_tau_bnn(args):
    """Wrapper function for training BNN at a single tau value (for parallel execution)."""
    (worker_id, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
     seed, hidden_width, weight_scale, warmup, samples, chains, func_type, noise_type) = args
    
    # Set seed for reproducibility
    np.random.seed(seed + int(tau * 100))
    torch.manual_seed(seed + int(tau * 100))
    
    from Models.BNN import (
        train_bnn,
        bnn_predict,
        normalize_x as bnn_normalize_x,
        normalize_x_data as bnn_normalize_x_data
    )
    
    # Normalize input (using passed data)
    x_mean, x_std = bnn_normalize_x(x_train)
    x_train_norm = bnn_normalize_x_data(x_train, x_mean, x_std)
    x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
    
    # Train BNN with MCMC (uses CPU internally)
    mcmc = train_bnn(
        x_train_norm, y_train,
        hidden_width=hidden_width, weight_scale=weight_scale,
        warmup=warmup, samples=samples, chains=chains, seed=seed + int(tau * 100)
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
    
    return tau, distribution, mu_pred, ale_var, epi_var, tot_var, mse


def _train_single_tau_bamlss(args):
    """Wrapper function for training BAMLSS at a single tau value (for parallel execution)."""
    (worker_id, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
     seed, n_iter, burnin, thin, nsamples, func_type, noise_type) = args
    
    # Set seed for reproducibility (BAMLSS is CPU-only, no GPU assignment needed)
    np.random.seed(seed + int(tau * 100))
    torch.manual_seed(seed + int(tau * 100))
    
    from Models.BAMLSS import bamlss_predict
    
    # BAMLSS fits directly using passed data
    mu_pred, ale_var, epi_var, tot_var = bamlss_predict(
        x_train, y_train, x_grid,
        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples
    )
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    return tau, distribution, mu_pred, ale_var, epi_var, tot_var, mse


def compute_and_save_statistics_noise_level(
    uncertainties_by_tau: dict,
    mse_by_tau: dict,
    tau_values: list,
    function_name: str,
    distribution: str,
    noise_type: str,
    func_type: str,
    model_name: str
):
    """
    Shared function to compute normalized statistics and save results for noise level experiments.
    
    This function normalizes uncertainties across all tau values, computes
    averages and correlations, prints formatted statistics, and saves results.
    
    Parameters:
    -----------
    uncertainties_by_tau : dict
        Dictionary mapping tau to dict with 'ale', 'epi', 'tot' lists
    mse_by_tau : dict
        Dictionary mapping tau to list of MSE values
    tau_values : list
        List of tau values tested
    function_name : str
        Human-readable function name (e.g., "Linear", "Sinusoidal")
    distribution : str
        Noise distribution ('normal' or 'laplace')
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str
        Function type identifier (e.g., 'linear', 'sin')
    model_name : str
        Model name for saving results (e.g., 'MC_Dropout', 'Deep_Ensemble')
    
    Returns:
    --------
    dict : Statistics dictionary with tau values, averages, correlations, MSE, and stats_df
    """
    # Collect all values for normalization
    all_ale = np.concatenate([np.concatenate(uncertainties_by_tau[tau]['ale']) 
                              for tau in tau_values])
    all_epi = np.concatenate([np.concatenate(uncertainties_by_tau[tau]['epi']) 
                              for tau in tau_values])
    all_tot = np.concatenate([np.concatenate(uncertainties_by_tau[tau]['tot']) 
                              for tau in tau_values])
    
    # Compute min/max for normalization
    ale_min, ale_max = all_ale.min(), all_ale.max()
    epi_min, epi_max = all_epi.min(), all_epi.max()
    
    
    def normalize(values, vmin, vmax):
        """Normalize values to [0, 1] range"""
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)
    
    # Compute statistics for each tau value
    avg_ale_norm_list = []
    avg_epi_norm_list = []
    avg_tot_norm_list = []
    correlation_list = []
    mse_list = []
    
    print(f"\n{'='*60}")
    print(f"Normalized Average Uncertainties by Tau - {function_name} Function - {model_name} - {distribution}")
    print(f"{'='*60}")
    print(f"\n{'Tau':<12} {'Avg Aleatoric (norm)':<25} {'Avg Epistemic (norm)':<25} {'Avg Total (norm)':<25} {'Correlation (Epi-Ale)':<25} {'MSE':<15}")
    print("-" * 140)
    
    for tau in tau_values:
        ale_vals = np.concatenate(uncertainties_by_tau[tau]['ale'])
        epi_vals = np.concatenate(uncertainties_by_tau[tau]['epi'])
        tot_vals = np.concatenate(uncertainties_by_tau[tau]['tot'])
        mse_vals = mse_by_tau[tau]
        
        ale_norm = normalize(ale_vals, ale_min, ale_max)
        epi_norm = normalize(epi_vals, epi_min, epi_max)
        tot_norm = ale_norm + epi_norm
        
        avg_ale_norm = np.mean(ale_norm)
        avg_epi_norm = np.mean(epi_norm)
        avg_tot_norm = np.mean(tot_norm)
        avg_mse = np.mean(mse_vals)
        
        correlation = np.corrcoef(epi_vals, ale_vals)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        avg_ale_norm_list.append(avg_ale_norm)
        avg_epi_norm_list.append(avg_epi_norm)
        avg_tot_norm_list.append(avg_tot_norm)
        correlation_list.append(correlation)
        mse_list.append(avg_mse)
        
        print(f"{tau:>12.2f}  {avg_ale_norm:>24.6f}  {avg_epi_norm:>24.6f}  {avg_tot_norm:>24.6f}  {correlation:>24.6f}  {avg_mse:>15.6f}")
    
    print(f"\n{'='*60}")
    print("Note: Average values are normalized to [0, 1] range across all tau values")
    print("      Correlation is computed on original (non-normalized) uncertainty values")
    print(f"{'='*60}")
    
    # Save summary statistics
    stats_df, fig = save_summary_statistics_noise_level(
        tau_values, avg_ale_norm_list, avg_epi_norm_list,
        avg_tot_norm_list, correlation_list, mse_list,
        function_name, distribution=distribution,
        noise_type=noise_type, func_type=func_type, model_name=model_name
    )
    plt.show()
    plt.close(fig)
    
    return {
        'tau_values': tau_values,
        'avg_ale_norm': avg_ale_norm_list,
        'avg_epi_norm': avg_epi_norm_list,
        'avg_tot_norm': avg_tot_norm_list,
        'correlations': correlation_list,
        'mse': mse_list,
        'stats_df': stats_df
    }


def run_mc_dropout_noise_level_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    tau_values: list = [0.5, 1, 2, 2.5, 5, 10],
    distributions: list = ['normal', 'laplace'],
    n_train: int = 1000,
    train_range: tuple = (-5, 10),
    grid_points: int = 1000,
    seed: int = 42,
    p: float = 0.2,
    beta: float = 0.5,
    epochs: int = 700,
    lr: float = 1e-3,
    batch_size: int = 32,
    mc_samples: int = 20,
    parallel: bool = True
):
    """
    Run noise level experiment for MC Dropout model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test (e.g., ['linear', 'sin'])
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    tau_values : list
        List of tau (noise level) values to test
    distributions : list
        List of noise distributions to test (e.g., ['normal', 'laplace'])
    n_train : int
        Training dataset size
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
        mc_dropout_predict,
        normalize_x,
        normalize_x_data
    )
    
    function_names = {"linear": "Linear", "sin": "Sinusoidal"}
    
    for func_type in function_types:
        for distribution in distributions:
            print(f"\n{'#'*80}")
            print(f"# Function Type: {function_names[func_type]} ({func_type}) - Distribution: {distribution} - MC Dropout")
            print(f"{'#'*80}\n")
            
            # Generate data
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            uncertainties_by_tau = {tau: {'ale': [], 'epi': [], 'tot': []} for tau in tau_values}
            mse_by_tau = {tau: [] for tau in tau_values}
            
            # Prepare arguments for parallel execution
            num_gpus = get_num_gpus()
            use_gpu = num_gpus > 0 and parallel
            total_tasks = len(tau_values)
            
            if parallel and total_tasks > 1:
                max_workers = min(total_tasks, num_gpus if use_gpu else multiprocessing.cpu_count())
                print(f"Using {'GPU' if use_gpu else 'CPU'} parallelization with {max_workers} workers")
                
                # Generate data for all tau values first (needed for parallel execution)
                args_list = []
                for idx, tau in enumerate(tau_values):
                    np.random.seed(seed + int(tau * 100))
                    torch.manual_seed(seed + int(tau * 100))
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train,
                        train_range=train_range,
                        grid_points=grid_points,
                        noise_type=noise_type,
                        type=func_type,
                        tau=tau,
                        distribution=distribution
                    )
                    args = (idx, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
                           seed, p, beta, epochs, lr, batch_size, mc_samples, func_type, noise_type)
                    args_list.append(args)
                
                try:
                    if use_gpu:
                        with mp.Pool(processes=max_workers) as pool:
                            results = pool.map(_train_single_tau_mc_dropout, args_list)
                    else:
                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            results = list(executor.map(_train_single_tau_mc_dropout, args_list))
                    
                    # Process results
                    for tau, dist, mu_pred, ale_var, epi_var, tot_var, mse in results:
                        uncertainties_by_tau[tau]['ale'].append(ale_var)
                        uncertainties_by_tau[tau]['epi'].append(epi_var)
                        uncertainties_by_tau[tau]['tot'].append(tot_var)
                        mse_by_tau[tau].append(mse)
                        
                        # Get training data for plotting
                        np.random.seed(seed + int(tau * 100))
                        x_train_plot, y_train_plot, _, _ = generate_toy_regression_func(
                            n_train=n_train,
                            train_range=train_range,
                            grid_points=grid_points,
                            noise_type=noise_type,
                            type=func_type,
                            tau=tau,
                            distribution=distribution
                        )
                        
                        plot_uncertainties_no_ood(
                            x_train_plot, y_train_plot, x_grid, y_grid_clean,
                            mu_pred, ale_var, epi_var, tot_var,
                            title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - τ={tau} ({distribution})",
                            noise_type=noise_type,
                            func_type=func_type
                        )
                except Exception as e:
                    print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                    parallel = False
            
            # Sequential execution
            if not parallel or total_tasks == 1:
                for tau in tau_values:
                    print(f"\n{'='*60}")
                    print(f"Training with tau={tau} (noise level)")
                    print(f"{'='*60}")
                    
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train,
                        train_range=train_range,
                        grid_points=grid_points,
                        noise_type=noise_type,
                        type=func_type,
                        tau=tau,
                        distribution=distribution
                    )
                    
                    x_mean, x_std = normalize_x(x_train)
                    x_train_norm = normalize_x_data(x_train, x_mean, x_std)
                    x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
                    
                    ds = TensorDataset(torch.from_numpy(x_train_norm), torch.from_numpy(y_train))
                    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                    
                    model = MCDropoutRegressor(p=p)
                    train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
                    
                    mu_pred, ale_var, epi_var, tot_var = mc_dropout_predict(model, x_grid_norm, M=mc_samples)
                    
                    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                    
                    uncertainties_by_tau[tau]['ale'].append(ale_var)
                    uncertainties_by_tau[tau]['epi'].append(epi_var)
                    uncertainties_by_tau[tau]['tot'].append(tot_var)
                    mse_by_tau[tau].append(mse)
                    
                    plot_uncertainties_no_ood(
                        x_train, y_train, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"MC Dropout (β-NLL, β={beta}) - {function_names[func_type]} - τ={tau} ({distribution})",
                        noise_type=noise_type,
                        func_type=func_type
                    )
            
            # Compute and save statistics
            compute_and_save_statistics_noise_level(
                uncertainties_by_tau, mse_by_tau, tau_values,
                function_names[func_type], distribution,
                noise_type, func_type, 'MC_Dropout'
            )


def run_deep_ensemble_noise_level_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    tau_values: list = [0.5, 1, 2, 2.5, 5, 10],
    distributions: list = ['normal', 'laplace'],
    n_train: int = 1000,
    train_range: tuple = (-5, 10),
    grid_points: int = 1000,
    seed: int = 42,
    beta: float = 0.5,
    batch_size: int = 32,
    K: int = 5,
    parallel: bool = True
):
    """
    Run noise level experiment for Deep Ensemble model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    tau_values : list
        List of tau (noise level) values to test
    distributions : list
        List of noise distributions to test
    n_train : int
        Training dataset size
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
        for distribution in distributions:
            print(f"\n{'#'*80}")
            print(f"# Function Type: {function_names[func_type]} ({func_type}) - Distribution: {distribution} - Deep Ensemble")
            print(f"{'#'*80}\n")
            
            # Generate data
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            uncertainties_by_tau = {tau: {'ale': [], 'epi': [], 'tot': []} for tau in tau_values}
            mse_by_tau = {tau: [] for tau in tau_values}
            
            num_gpus = get_num_gpus()
            use_gpu = num_gpus > 0 and parallel
            total_tasks = len(tau_values)
            
            if parallel and total_tasks > 1:
                max_workers = min(total_tasks, num_gpus if use_gpu else multiprocessing.cpu_count())
                print(f"Using {'GPU' if use_gpu else 'CPU'} parallelization with {max_workers} workers")
                
                args_list = []
                for idx, tau in enumerate(tau_values):
                    np.random.seed(seed + int(tau * 100))
                    torch.manual_seed(seed + int(tau * 100))
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train, train_range=train_range, grid_points=grid_points,
                        noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                    )
                    args = (idx, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
                           seed, beta, batch_size, K, func_type, noise_type)
                    args_list.append(args)
                
                try:
                    if use_gpu:
                        with mp.Pool(processes=max_workers) as pool:
                            results = pool.map(_train_single_tau_deep_ensemble, args_list)
                    else:
                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            results = list(executor.map(_train_single_tau_deep_ensemble, args_list))
                    
                    for tau, dist, mu_pred, ale_var, epi_var, tot_var, mse in results:
                        uncertainties_by_tau[tau]['ale'].append(ale_var)
                        uncertainties_by_tau[tau]['epi'].append(epi_var)
                        uncertainties_by_tau[tau]['tot'].append(tot_var)
                        mse_by_tau[tau].append(mse)
                        
                        np.random.seed(seed + int(tau * 100))
                        x_train_plot, y_train_plot, _, _ = generate_toy_regression_func(
                            n_train=n_train, train_range=train_range, grid_points=grid_points,
                            noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                        )
                        
                        plot_uncertainties_no_ood(
                            x_train_plot, y_train_plot, x_grid, y_grid_clean,
                            mu_pred, ale_var, epi_var, tot_var,
                            title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - τ={tau} ({distribution})",
                            noise_type=noise_type, func_type=func_type
                        )
                except Exception as e:
                    print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                    parallel = False
            
            if not parallel or total_tasks == 1:
                for tau in tau_values:
                    print(f"\n{'='*60}")
                    print(f"Training with tau={tau} (noise level)")
                    print(f"{'='*60}")
                    
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train, train_range=train_range, grid_points=grid_points,
                        noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                    )
                    
                    x_mean, x_std = normalize_x(x_train)
                    x_train_norm = normalize_x_data(x_train, x_mean, x_std)
                    x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
                    
                    ensemble = train_ensemble_deep(
                        x_train_norm, y_train,
                        batch_size=batch_size, K=K,
                        loss_type='beta_nll', beta=beta, parallel=True
                    )
                    
                    mu_pred, ale_var, epi_var, tot_var = ensemble_predict_deep(ensemble, x_grid_norm)
                    
                    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                    
                    uncertainties_by_tau[tau]['ale'].append(ale_var)
                    uncertainties_by_tau[tau]['epi'].append(epi_var)
                    uncertainties_by_tau[tau]['tot'].append(tot_var)
                    mse_by_tau[tau].append(mse)
                    
                    plot_uncertainties_no_ood(
                        x_train, y_train, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"Deep Ensemble (β-NLL, β={beta}) - {function_names[func_type]} - τ={tau} ({distribution})",
                        noise_type=noise_type, func_type=func_type
                    )
            
            # Compute and save statistics
            compute_and_save_statistics_noise_level(
                uncertainties_by_tau, mse_by_tau, tau_values,
                function_names[func_type], distribution,
                noise_type, func_type, 'Deep_Ensemble'
            )


def run_bnn_noise_level_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    tau_values: list = [0.5, 1, 2, 2.5, 5, 10],
    distributions: list = ['normal', 'laplace'],
    n_train: int = 1000,
    train_range: tuple = (-5, 10),
    grid_points: int = 1000,
    seed: int = 42,
    hidden_width: int = 16,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    chains: int = 1,
    parallel: bool = True
):
    """
    Run noise level experiment for BNN (Bayesian Neural Network) model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    tau_values : list
        List of tau (noise level) values to test
    distributions : list
        List of noise distributions to test
    n_train : int
        Training dataset size
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
        for distribution in distributions:
            print(f"\n{'#'*80}")
            print(f"# Function Type: {function_names[func_type]} ({func_type}) - Distribution: {distribution} - BNN")
            print(f"{'#'*80}\n")
            
            # Generate data
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            uncertainties_by_tau = {tau: {'ale': [], 'epi': [], 'tot': []} for tau in tau_values}
            mse_by_tau = {tau: [] for tau in tau_values}
            
            num_gpus = get_num_gpus()
            use_gpu = num_gpus > 0 and parallel
            total_tasks = len(tau_values)
            
            if parallel and total_tasks > 1:
                max_workers = min(total_tasks, num_gpus if use_gpu else multiprocessing.cpu_count())
                print(f"Using {'GPU' if use_gpu else 'CPU'} parallelization with {max_workers} workers")
                
                args_list = []
                for idx, tau in enumerate(tau_values):
                    np.random.seed(seed + int(tau * 100))
                    torch.manual_seed(seed + int(tau * 100))
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train, train_range=train_range, grid_points=grid_points,
                        noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                    )
                    args = (idx, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
                           seed, hidden_width, weight_scale, warmup, samples, chains, func_type, noise_type)
                    args_list.append(args)
                
                try:
                    if use_gpu:
                        with mp.Pool(processes=max_workers) as pool:
                            results = pool.map(_train_single_tau_bnn, args_list)
                    else:
                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            results = list(executor.map(_train_single_tau_bnn, args_list))
                    
                    for tau, dist, mu_pred, ale_var, epi_var, tot_var, mse in results:
                        uncertainties_by_tau[tau]['ale'].append(ale_var)
                        uncertainties_by_tau[tau]['epi'].append(epi_var)
                        uncertainties_by_tau[tau]['tot'].append(tot_var)
                        mse_by_tau[tau].append(mse)
                        
                        np.random.seed(seed + int(tau * 100))
                        x_train_plot, y_train_plot, _, _ = generate_toy_regression_func(
                            n_train=n_train, train_range=train_range, grid_points=grid_points,
                            noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                        )
                        
                        plot_uncertainties_no_ood(
                            x_train_plot, y_train_plot, x_grid, y_grid_clean,
                            mu_pred, ale_var, epi_var, tot_var,
                            title=f"BNN (Pyro NUTS) - {function_names[func_type]} - τ={tau} ({distribution})",
                            noise_type=noise_type, func_type=func_type
                        )
                except Exception as e:
                    print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                    parallel = False
            
            if not parallel or total_tasks == 1:
                for tau in tau_values:
                    print(f"\n{'='*60}")
                    print(f"Training with tau={tau} (noise level)")
                    print(f"{'='*60}")
                    
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train, train_range=train_range, grid_points=grid_points,
                        noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                    )
                    
                    x_mean, x_std = bnn_normalize_x(x_train)
                    x_train_norm = bnn_normalize_x_data(x_train, x_mean, x_std)
                    x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
                    
                    mcmc = train_bnn(
                        x_train_norm, y_train,
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
                    
                    uncertainties_by_tau[tau]['ale'].append(ale_var)
                    uncertainties_by_tau[tau]['epi'].append(epi_var)
                    uncertainties_by_tau[tau]['tot'].append(tot_var)
                    mse_by_tau[tau].append(mse)
                    
                    plot_uncertainties_no_ood(
                        x_train, y_train, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"BNN (Pyro NUTS) - {function_names[func_type]} - τ={tau} ({distribution})",
                        noise_type=noise_type, func_type=func_type
                    )
            
            # Compute and save statistics
            compute_and_save_statistics_noise_level(
                uncertainties_by_tau, mse_by_tau, tau_values,
                function_names[func_type], distribution,
                noise_type, func_type, 'BNN'
            )


def run_bamlss_noise_level_experiment(
    generate_toy_regression_func,
    function_types: list = ['linear', 'sin'],
    noise_type: str = 'heteroscedastic',
    tau_values: list = [0.5, 1, 2, 2.5, 5, 10],
    distributions: list = ['normal', 'laplace'],
    n_train: int = 1000,
    train_range: tuple = (-5, 10),
    grid_points: int = 1000,
    seed: int = 42,
    n_iter: int = 12000,
    burnin: int = 2000,
    thin: int = 10,
    nsamples: int = 1000,
    parallel: bool = True
):
    """
    Run noise level experiment for BAMLSS model.
    
    Parameters:
    -----------
    generate_toy_regression_func : callable
        Function to generate toy regression data (from notebook)
    function_types : list
        List of function types to test
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    tau_values : list
        List of tau (noise level) values to test
    distributions : list
        List of noise distributions to test
    n_train : int
        Training dataset size
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
        for distribution in distributions:
            print(f"\n{'#'*80}")
            print(f"# Function Type: {function_names[func_type]} ({func_type}) - Distribution: {distribution} - BAMLSS")
            print(f"{'#'*80}\n")
            
            # Generate data
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            uncertainties_by_tau = {tau: {'ale': [], 'epi': [], 'tot': []} for tau in tau_values}
            mse_by_tau = {tau: [] for tau in tau_values}
            
            # BAMLSS uses R, so CPU-only parallelization
            total_tasks = len(tau_values)
            
            if parallel and total_tasks > 1:
                max_workers = min(total_tasks, multiprocessing.cpu_count())
                print(f"Using CPU parallelization with {max_workers} workers (BAMLSS is CPU-only)")
                
                args_list = []
                for idx, tau in enumerate(tau_values):
                    np.random.seed(seed + int(tau * 100))
                    torch.manual_seed(seed + int(tau * 100))
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train, train_range=train_range, grid_points=grid_points,
                        noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                    )
                    args = (idx, tau, distribution, x_train, y_train, x_grid, y_grid_clean,
                           seed, n_iter, burnin, thin, nsamples, func_type, noise_type)
                    args_list.append(args)
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(_train_single_tau_bamlss, args_list))
                    
                    for tau, dist, mu_pred, ale_var, epi_var, tot_var, mse in results:
                        uncertainties_by_tau[tau]['ale'].append(ale_var)
                        uncertainties_by_tau[tau]['epi'].append(epi_var)
                        uncertainties_by_tau[tau]['tot'].append(tot_var)
                        mse_by_tau[tau].append(mse)
                        
                        np.random.seed(seed + int(tau * 100))
                        x_train_plot, y_train_plot, _, _ = generate_toy_regression_func(
                            n_train=n_train, train_range=train_range, grid_points=grid_points,
                            noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                        )
                        
                        plot_uncertainties_no_ood(
                            x_train_plot, y_train_plot, x_grid, y_grid_clean,
                            mu_pred, ale_var, epi_var, tot_var,
                            title=f"BAMLSS - {function_names[func_type]} - τ={tau} ({distribution})",
                            noise_type=noise_type, func_type=func_type
                        )
                except Exception as e:
                    print(f"Parallel execution failed: {e}. Falling back to sequential execution.")
                    parallel = False
            
            if not parallel or total_tasks == 1:
                for tau in tau_values:
                    print(f"\n{'='*60}")
                    print(f"Training with tau={tau} (noise level)")
                    print(f"{'='*60}")
                    
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    
                    x_train, y_train, x_grid, y_grid_clean = generate_toy_regression_func(
                        n_train=n_train, train_range=train_range, grid_points=grid_points,
                        noise_type=noise_type, type=func_type, tau=tau, distribution=distribution
                    )
                    
                    mu_pred, ale_var, epi_var, tot_var = bamlss_predict(
                        x_train, y_train, x_grid,
                        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples
                    )
                    
                    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
                    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
                    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
                    
                    uncertainties_by_tau[tau]['ale'].append(ale_var)
                    uncertainties_by_tau[tau]['epi'].append(epi_var)
                    uncertainties_by_tau[tau]['tot'].append(tot_var)
                    mse_by_tau[tau].append(mse)
                    
                    plot_uncertainties_no_ood(
                        x_train, y_train, x_grid, y_grid_clean,
                        mu_pred, ale_var, epi_var, tot_var,
                        title=f"BAMLSS - {function_names[func_type]} - τ={tau} ({distribution})",
                        noise_type=noise_type, func_type=func_type
                    )
            
            # Compute and save statistics
            compute_and_save_statistics_noise_level(
                uncertainties_by_tau, mse_by_tau, tau_values,
                function_names[func_type], distribution,
                noise_type, func_type, 'BAMLSS'
            )

