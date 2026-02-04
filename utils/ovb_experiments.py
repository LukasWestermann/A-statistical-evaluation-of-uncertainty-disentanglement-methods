"""
Helper functions for running OVB (Omitted Variable Bias) experiments with uncertainty quantification.

This module provides functions to run OVB experiments for MC Dropout,
handling the pattern of:
1. Generating data with correlated latent variable Z
2. Training models on X only (omitting Z)
3. Collecting uncertainties (variance and entropy-based)
4. Computing and saving statistics
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import multiprocessing
from datetime import datetime
import pandas as pd
from pathlib import Path

from utils.entropy_uncertainty import entropy_uncertainty_analytical, entropy_uncertainty_numerical
from utils.device import get_device_for_worker, get_num_gpus
from utils.plotting import (
    plot_uncertainties_no_ood,
    plot_uncertainties_no_ood_normalized,
    plot_uncertainties_entropy_no_ood,
    plot_uncertainties_entropy_no_ood_normalized,
    plot_entropy_lines_no_ood
)


# ============================================================================
# Helper Functions
# ============================================================================

def _normalize_minmax(arr):
    """Normalize array to [0, 1] using its own min/max."""
    arr_flat = arr.flatten()
    vmin, vmax = arr_flat.min(), arr_flat.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


# ============================================================================
# OVB Data Generation
# ============================================================================

def generate_ovb_data(
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 1000,
    noise_type: str = 'heteroscedastic',
    func_type: str = 'linear',
    rho: float = 0.7,
    beta2: float = 1.0,
    seed: int = 42
):
    """
    Generate toy regression data with OVB potential.
    
    Model: Y = f(X) + beta2*Z + epsilon
    Where X and Z are correlated via rho.
    
    Parameters:
    -----------
    n_train : int
        Number of training samples
    train_range : tuple
        (min, max) range for X values
    grid_points : int
        Number of grid points for evaluation
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str
        'linear' (f(x) = 0.7x + 0.5) or 'sin' (f(x) = x*sin(x) + x)
    rho : float
        Correlation between X and Z
    beta2 : float
        Effect of omitted variable Z on Y
    seed : int
        Random seed
        
    Returns:
    --------
    X, Z, Y, x_grid, y_grid_clean : tuple
    """
    rng = np.random.default_rng(seed)
    
    # Generate latent Z ~ N(0, 1)
    Z = rng.standard_normal(n_train)
    
    # Generate X correlated with Z
    nu = rng.standard_normal(n_train)
    X = rho * Z + np.sqrt(1 - rho**2) * nu
    
    # Scale X to desired range
    X_scaled = X * (train_range[1] - train_range[0]) / 4 + (train_range[0] + train_range[1]) / 2
    
    # Define the function f(x)
    if func_type == 'linear':
        f_clean = lambda x: 0.7 * x + 0.5
    elif func_type == 'sin':
        f_clean = lambda x: x * np.sin(x) + x
    else:
        raise ValueError("func_type must be 'linear' or 'sin'")
    
    # Compute clean function output
    y_clean = f_clean(X_scaled)
    
    # Generate noise epsilon
    if noise_type == 'homoscedastic':
        sigma = 1.0
        epsilon = rng.normal(0, sigma, n_train)
    elif noise_type == 'heteroscedastic':
        sigma = np.abs(2.5 * np.sin(0.5 * X_scaled + 5))
        epsilon = rng.normal(0, sigma)
    else:
        raise ValueError("noise_type must be 'homoscedastic' or 'heteroscedastic'")
    
    # Generate Y with OVB structure
    Y = y_clean + beta2 * Z + epsilon
    
    # Create evaluation grid
    x_grid = np.linspace(train_range[0], train_range[1], grid_points)
    y_grid_clean = f_clean(x_grid)
    
    return (X_scaled.astype(np.float32).reshape(-1, 1), 
            Z.astype(np.float32), 
            Y.astype(np.float32).reshape(-1, 1),
            x_grid.astype(np.float32).reshape(-1, 1), 
            y_grid_clean.astype(np.float32).reshape(-1, 1))


# ============================================================================
# Training helper for single configuration
# ============================================================================

def _train_single_ovb_config(args):
    """Train MC Dropout at a single OVB configuration."""
    (worker_id, param_value, param_name, fixed_param_value, fixed_param_name,
     n_train, train_range, grid_points, noise_type, func_type,
     seed, p, beta, epochs, lr, batch_size, mc_samples, entropy_method) = args
    
    # Set device for this worker
    device = get_device_for_worker(worker_id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Set seeds
    np.random.seed(seed + int(param_value * 100))
    torch.manual_seed(seed + int(param_value * 100))
    
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    # Set rho and beta2 based on which parameter is being varied
    if param_name == 'rho':
        rho = param_value
        beta2 = fixed_param_value
    else:  # param_name == 'beta2'
        rho = fixed_param_value
        beta2 = param_value
    
    # Generate OVB data
    X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        noise_type=noise_type,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed + int(param_value * 100)
    )
    
    # Create dataloader (training on X only, omitting Z)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Train model
    model = MCDropoutRegressor(p=p)
    train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
    
    # Make predictions
    result = mc_dropout_predict(model, x_grid, M=mc_samples, return_raw_arrays=True)
    mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
    
    # Compute entropy-based uncertainties
    if entropy_method == 'analytical':
        entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
    else:
        entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
    
    ale_entropy = entropy_results['aleatoric']
    epi_entropy = entropy_results['epistemic']
    tot_entropy = entropy_results['total']
    
    # Compute MSE
    mu_pred_flat = mu_pred.squeeze() if mu_pred.ndim > 1 else mu_pred
    y_grid_clean_flat = y_grid_clean.squeeze() if y_grid_clean.ndim > 1 else y_grid_clean
    mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
    
    # Compute empirical correlation between X and Z
    empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
    
    return {
        'param_value': param_value,
        'rho': rho,
        'beta2': beta2,
        'mu_pred': mu_pred,
        'ale_var': ale_var,
        'epi_var': epi_var,
        'tot_var': tot_var,
        'ale_entropy': ale_entropy,
        'epi_entropy': epi_entropy,
        'tot_entropy': tot_entropy,
        'mse': mse,
        'empirical_corr': empirical_corr,
        'X': X,
        'Z': Z,
        'Y': Y,
        'x_grid': x_grid,
        'y_grid_clean': y_grid_clean,
        'mu_samples': mu_samples,
        'sigma2_samples': sigma2_samples
    }


# ============================================================================
# Save Functions
# ============================================================================

def save_ovb_model_outputs(X, Z, Y, x_grid, y_grid_clean, mu_pred, mu_samples, sigma2_samples,
                           ale_var, epi_var, tot_var, ale_entropy, epi_entropy, tot_entropy,
                           rho, beta2, func_type, noise_type, results_dir, param_name='rho'):
    """
    Save all OVB model outputs for later reuse.
    
    Parameters:
    -----------
    X, Z, Y : np.ndarray
        Training data (X observed, Z latent/omitted, Y outcome)
    x_grid, y_grid_clean : np.ndarray
        Evaluation grid
    mu_pred, mu_samples, sigma2_samples : np.ndarray
        Raw model predictions
    ale_var, epi_var, tot_var : np.ndarray
        Variance decomposition
    ale_entropy, epi_entropy, tot_entropy : np.ndarray
        Entropy (IT) decomposition
    rho, beta2 : float
        OVB parameters
    func_type, noise_type : str
        Experiment configuration
    results_dir : Path
        Directory to save outputs
    param_name : str
        'rho' or 'beta2' - which parameter is being varied
    """
    date = datetime.now().strftime('%Y%m%d')
    
    if param_name == 'rho':
        filename = f"ovb_outputs_rho{rho:.2f}_beta2{beta2:.2f}_{date}.npz"
    else:
        filename = f"ovb_outputs_beta2{beta2:.2f}_rho{rho:.2f}_{date}.npz"
    
    filepath = results_dir / filename
    
    np.savez_compressed(
        filepath,
        X=X, Z=Z, Y=Y,
        x_grid=x_grid, y_grid_clean=y_grid_clean,
        mu_pred=mu_pred, mu_samples=mu_samples, sigma2_samples=sigma2_samples,
        ale_var=ale_var, epi_var=epi_var, tot_var=tot_var,
        ale_entropy=ale_entropy, epi_entropy=epi_entropy, tot_entropy=tot_entropy,
        rho=np.array([rho]), beta2=np.array([beta2]),
        func_type=np.array([func_type]), noise_type=np.array([noise_type])
    )
    print(f"  Saved model outputs to: {filepath}")


# ============================================================================
# Main Experiment Functions
# ============================================================================

def run_mc_dropout_ovb_rho_experiment(
    rho_values: list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type: str = 'linear',
    noise_type: str = 'heteroscedastic',
    seed: int = 42,
    p: float = 0.25,
    beta: float = 0.5,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,
    mc_samples: int = 100,
    entropy_method: str = 'analytical',
    parallel: bool = False,
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run OVB experiment varying rho (correlation strength) with fixed beta2.
    
    Parameters func_type and noise_type can be either:
    - A single string: 'linear' or 'heteroscedastic'
    - A list of strings: ['linear', 'sin'] - will iterate through all combinations
    
    Returns:
    --------
    results_df : pd.DataFrame
        Summary statistics for each rho value (includes func_type/noise_type columns if multiple)
    all_results : dict
        Full results. If multiple configs: keyed by (func_type, noise_type) tuple
    """
    # Handle list inputs for func_type and noise_type
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    # If multiple configurations, iterate through all combinations
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                
                df, results = run_mc_dropout_ovb_rho_experiment(
                    rho_values=rho_values,
                    beta2=beta2,
                    n_train=n_train,
                    train_range=train_range,
                    grid_points=grid_points,
                    func_type=ft,
                    noise_type=nt,
                    seed=seed,
                    p=p,
                    beta=beta,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    mc_samples=mc_samples,
                    entropy_method=entropy_method,
                    parallel=parallel,
                    save_plots=save_plots,
                    results_dir=results_dir
                )
                
                # Add config columns to dataframe
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    # Single configuration - original logic
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    # Create subdirectory structure: noise_type/func_type/
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# OVB Experiment: Varying rho (X-Z Correlation)")
    print(f"# Fixed beta2 = {beta2}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"{'#'*80}\n")
    
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    all_results = {}
    summary_rows = []
    
    for rho in rho_values:
        print(f"\n{'='*60}")
        print(f"Training with rho = {rho}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(rho * 100))
        torch.manual_seed(seed + int(rho * 100))
        
        # Generate OVB data
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            func_type=func_type,
            rho=rho,
            beta2=beta2,
            seed=seed + int(rho * 100)
        )
        
        # Create dataloader (training on X only)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        # Train model
        model = MCDropoutRegressor(p=p)
        train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
        
        # Make predictions
        result = mc_dropout_predict(model, x_grid, M=mc_samples, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Compute statistics for omitted model
        mu_pred_flat = mu_pred.squeeze()
        y_grid_clean_flat = y_grid_clean.squeeze()
        mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # =====================================================================
        # FULL MODEL: Train on (X, Z) to compare with omitted model
        # =====================================================================
        print(f"  Training full model (with Z)...")
        
        # Create 2D input by concatenating X and Z
        X_full = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
        
        # Train full model
        model_full = MCDropoutRegressor(p=p, input_dim=2)
        ds_full = TensorDataset(torch.from_numpy(X_full), torch.from_numpy(Y))
        loader_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True)
        train_model(model_full, loader_full, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
        
        # Make predictions on training points (X, Z)
        result_full = mc_dropout_predict(model_full, X_full, M=mc_samples, return_raw_arrays=True)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        # Compute entropy-based uncertainties for full model
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        # Compute MSE for full model (on training points)
        mu_pred_full_flat = mu_pred_full.squeeze()
        Y_flat = Y.squeeze()
        mse_full = np.mean((mu_pred_full_flat - Y_flat)**2)
        
        # =====================================================================
        # Evaluate full model on 2D grid for heatmap visualization
        # =====================================================================
        heatmap_grid_points = 50
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        XZ_grid_flat = np.column_stack([X_grid_2d.ravel(), Z_grid_2d.ravel()]).astype(np.float32)
        
        # Predict on 2D grid
        result_2d = mc_dropout_predict(model_full, XZ_grid_flat, M=mc_samples, return_raw_arrays=True)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, (mu_samples_2d, sigma2_samples_2d) = result_2d
        
        # Compute entropy on 2D grid
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(mu_samples_2d, sigma2_samples_2d)
        else:
            entropy_2d = entropy_uncertainty_numerical(mu_samples_2d, sigma2_samples_2d, n_samples=5000, seed=seed)
        
        ale_entropy_2d = entropy_2d['aleatoric']
        epi_entropy_2d = entropy_2d['epistemic']
        tot_entropy_2d = entropy_2d['total']
        
        # Reshape to 2D grids
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        ale_var_2d_grid = ale_var_2d.reshape(grid_shape)
        epi_var_2d_grid = epi_var_2d.reshape(grid_shape)
        tot_var_2d_grid = tot_var_2d.reshape(grid_shape)
        ale_entropy_2d_grid = ale_entropy_2d.reshape(grid_shape)
        epi_entropy_2d_grid = epi_entropy_2d.reshape(grid_shape)
        tot_entropy_2d_grid = tot_entropy_2d.reshape(grid_shape)
        
        # Store results (both omitted and full model)
        all_results[rho] = {
            # Omitted model results
            'mu_pred': mu_pred,
            'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            # Full model results (on training points)
            'mu_pred_full': mu_pred_full,
            'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            # Full model results (on 2D grid for heatmaps)
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d_grid, 'epi_var_2d_grid': epi_var_2d_grid, 'tot_var_2d_grid': tot_var_2d_grid,
            'ale_entropy_2d_grid': ale_entropy_2d_grid, 'epi_entropy_2d_grid': epi_entropy_2d_grid, 'tot_entropy_2d_grid': tot_entropy_2d_grid,
        }
        
        # Compute normalized uncertainties for omitted model
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        
        # Compute normalized uncertainties for full model
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        # Compute inflation ratios (how much OVB inflates uncertainty)
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        # Summary row
        summary_rows.append({
            'rho': rho,
            'beta2': beta2,
            'empirical_corr': empirical_corr,
            # Omitted model - MSE
            'mse': mse,
            # Omitted model - Variance decomposition (raw)
            'mean_ale_var': np.mean(ale_var),
            'mean_epi_var': np.mean(epi_var),
            'mean_tot_var': np.mean(tot_var),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            # Omitted model - Variance decomposition (normalized)
            'mean_ale_var_norm': np.mean(ale_var_norm),
            'mean_epi_var_norm': np.mean(epi_var_norm),
            'mean_tot_var_norm': np.mean(tot_var_norm),
            # Omitted model - Entropy decomposition (raw)
            'mean_ale_entropy': np.mean(ale_entropy),
            'mean_epi_entropy': np.mean(epi_entropy),
            'mean_tot_entropy': np.mean(tot_entropy),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            # Omitted model - Entropy decomposition (normalized)
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm),
            'mean_epi_entropy_norm': np.mean(epi_entropy_norm),
            'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            # Full model - MSE
            'mse_full': mse_full,
            # Full model - Variance decomposition (raw)
            'mean_ale_var_full': mean_ale_var_full,
            'mean_epi_var_full': mean_epi_var_full,
            'mean_tot_var_full': np.mean(tot_var_full),
            # Full model - Variance decomposition (normalized)
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm),
            'mean_epi_var_full_norm': np.mean(epi_var_full_norm),
            'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            # Full model - Entropy decomposition (raw)
            'mean_ale_entropy_full': mean_ale_entropy_full,
            'mean_epi_entropy_full': mean_epi_entropy_full,
            'mean_tot_entropy_full': np.mean(tot_entropy_full),
            # Full model - Entropy decomposition (normalized)
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm),
            'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm),
            'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            # Inflation ratios
            'au_inflation_var': au_inflation_var,
            'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy,
            'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        # Print comparison
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Omitted] Variance - AU: {np.mean(ale_var):.4f}, EU: {np.mean(epi_var):.4f}")
        print(f"  [Omitted] Entropy  - AU: {np.mean(ale_entropy):.4f}, EU: {np.mean(epi_entropy):.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  [Full]    Variance - AU: {mean_ale_var_full:.4f}, EU: {mean_epi_var_full:.4f}")
        print(f"  [Full]    Entropy  - AU: {mean_ale_entropy_full:.4f}, EU: {mean_epi_entropy_full:.4f}")
        print(f"  Inflation (Var)     - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
        print(f"  Inflation (Entropy) - AU: {au_inflation_entropy:+.2%}, EU: {eu_inflation_entropy:+.2%}")
        
        # Generate individual plots for this rho value
        if save_plots:
            title_base = f"MC Dropout OVB (rho={rho}, beta2={beta2})"
            
            # Variance-based plots (std deviation bands)
            plot_uncertainties_no_ood(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_var, epi_var, tot_var,
                title=f"{title_base} - Variance",
                noise_type=noise_type, func_type=func_type
            )
            
            # Normalized variance-based
            plot_uncertainties_no_ood_normalized(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_var, epi_var, tot_var,
                title=f"{title_base} - Variance Normalized",
                noise_type=noise_type, func_type=func_type
            )
            
            # Entropy-based plots
            plot_uncertainties_entropy_no_ood(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_entropy, epi_entropy, tot_entropy,
                title=f"{title_base} - Entropy",
                noise_type=noise_type, func_type=func_type
            )
            
            # Normalized entropy-based
            plot_uncertainties_entropy_no_ood_normalized(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_entropy, epi_entropy, tot_entropy,
                title=f"{title_base} - Entropy Normalized",
                noise_type=noise_type, func_type=func_type
            )
            
            # Entropy lines (raw nats)
            plot_entropy_lines_no_ood(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_entropy, epi_entropy, tot_entropy,
                title=f"{title_base} - Entropy Lines",
                noise_type=noise_type, func_type=func_type
            )
        
        # Save model outputs for later reuse
        if save_dir:
            save_ovb_model_outputs(
                X=X, Z=Z, Y=Y, x_grid=x_grid, y_grid_clean=y_grid_clean,
                mu_pred=mu_pred, mu_samples=mu_samples, sigma2_samples=sigma2_samples,
                ale_var=ale_var, epi_var=epi_var, tot_var=tot_var,
                ale_entropy=ale_entropy, epi_entropy=epi_entropy, tot_entropy=tot_entropy,
                rho=rho, beta2=beta2, func_type=func_type, noise_type=noise_type,
                results_dir=save_dir, param_name='rho'
            )
    
    results_df = pd.DataFrame(summary_rows)
    
    # Generate summary plots
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'rho', beta2, func_type, noise_type, save_dir)
    
    # Save summary stats to Excel
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"ovb_rho_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


def run_mc_dropout_ovb_beta2_experiment(
    beta2_values: list = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type: str = 'linear',
    noise_type: str = 'heteroscedastic',
    seed: int = 42,
    p: float = 0.25,
    beta: float = 0.5,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 32,
    mc_samples: int = 100,
    entropy_method: str = 'analytical',
    parallel: bool = False,
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run OVB experiment varying beta2 (effect of omitted Z) with fixed rho.
    
    Parameters func_type and noise_type can be either:
    - A single string: 'linear' or 'heteroscedastic'
    - A list of strings: ['linear', 'sin'] - will iterate through all combinations
    
    Returns:
    --------
    results_df : pd.DataFrame
        Summary statistics for each beta2 value (includes func_type/noise_type columns if multiple)
    all_results : dict
        Full results. If multiple configs: keyed by (func_type, noise_type) tuple
    """
    # Handle list inputs for func_type and noise_type
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    # If multiple configurations, iterate through all combinations
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                
                df, results = run_mc_dropout_ovb_beta2_experiment(
                    beta2_values=beta2_values,
                    rho=rho,
                    n_train=n_train,
                    train_range=train_range,
                    grid_points=grid_points,
                    func_type=ft,
                    noise_type=nt,
                    seed=seed,
                    p=p,
                    beta=beta,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    mc_samples=mc_samples,
                    entropy_method=entropy_method,
                    parallel=parallel,
                    save_plots=save_plots,
                    results_dir=results_dir
                )
                
                # Add config columns to dataframe
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    # Single configuration - original logic
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    # Create subdirectory structure: noise_type/func_type/
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# OVB Experiment: Varying beta2 (Effect of Omitted Z)")
    print(f"# Fixed rho = {rho}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"{'#'*80}\n")
    
    from Models.MC_Dropout import (
        MCDropoutRegressor,
        train_model,
        mc_dropout_predict
    )
    
    all_results = {}
    summary_rows = []
    
    for beta2_val in beta2_values:
        print(f"\n{'='*60}")
        print(f"Training with beta2 = {beta2_val}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(beta2_val * 100))
        torch.manual_seed(seed + int(beta2_val * 100))
        
        # Generate OVB data
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train,
            train_range=train_range,
            grid_points=grid_points,
            noise_type=noise_type,
            func_type=func_type,
            rho=rho,
            beta2=beta2_val,
            seed=seed + int(beta2_val * 100)
        )
        
        # Create dataloader (training on X only)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        # Train model
        model = MCDropoutRegressor(p=p)
        train_model(model, loader, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
        
        # Make predictions
        result = mc_dropout_predict(model, x_grid, M=mc_samples, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Compute statistics for omitted model
        mu_pred_flat = mu_pred.squeeze()
        y_grid_clean_flat = y_grid_clean.squeeze()
        mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # =====================================================================
        # FULL MODEL: Train on (X, Z) to compare with omitted model
        # =====================================================================
        print(f"  Training full model (with Z)...")
        
        # Create 2D input by concatenating X and Z
        X_full = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
        
        # Train full model
        model_full = MCDropoutRegressor(p=p, input_dim=2)
        ds_full = TensorDataset(torch.from_numpy(X_full), torch.from_numpy(Y))
        loader_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True)
        train_model(model_full, loader_full, epochs=epochs, lr=lr, loss_type='beta_nll', beta=beta)
        
        # Make predictions on training points (X, Z)
        result_full = mc_dropout_predict(model_full, X_full, M=mc_samples, return_raw_arrays=True)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        # Compute entropy-based uncertainties for full model
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        # Compute MSE for full model (on training points)
        mu_pred_full_flat = mu_pred_full.squeeze()
        Y_flat = Y.squeeze()
        mse_full = np.mean((mu_pred_full_flat - Y_flat)**2)
        
        # =====================================================================
        # Evaluate full model on 2D grid for heatmap visualization
        # =====================================================================
        heatmap_grid_points = 50
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        XZ_grid_flat = np.column_stack([X_grid_2d.ravel(), Z_grid_2d.ravel()]).astype(np.float32)
        
        # Predict on 2D grid
        result_2d = mc_dropout_predict(model_full, XZ_grid_flat, M=mc_samples, return_raw_arrays=True)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, (mu_samples_2d, sigma2_samples_2d) = result_2d
        
        # Compute entropy on 2D grid
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(mu_samples_2d, sigma2_samples_2d)
        else:
            entropy_2d = entropy_uncertainty_numerical(mu_samples_2d, sigma2_samples_2d, n_samples=5000, seed=seed)
        
        ale_entropy_2d = entropy_2d['aleatoric']
        epi_entropy_2d = entropy_2d['epistemic']
        tot_entropy_2d = entropy_2d['total']
        
        # Reshape to 2D grids
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        ale_var_2d_grid = ale_var_2d.reshape(grid_shape)
        epi_var_2d_grid = epi_var_2d.reshape(grid_shape)
        tot_var_2d_grid = tot_var_2d.reshape(grid_shape)
        ale_entropy_2d_grid = ale_entropy_2d.reshape(grid_shape)
        epi_entropy_2d_grid = epi_entropy_2d.reshape(grid_shape)
        tot_entropy_2d_grid = tot_entropy_2d.reshape(grid_shape)
        
        # Store results (both omitted and full model)
        all_results[beta2_val] = {
            # Omitted model results
            'mu_pred': mu_pred,
            'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            # Full model results (on training points)
            'mu_pred_full': mu_pred_full,
            'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            # Full model results (on 2D grid for heatmaps)
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d_grid, 'epi_var_2d_grid': epi_var_2d_grid, 'tot_var_2d_grid': tot_var_2d_grid,
            'ale_entropy_2d_grid': ale_entropy_2d_grid, 'epi_entropy_2d_grid': epi_entropy_2d_grid, 'tot_entropy_2d_grid': tot_entropy_2d_grid,
        }
        
        # Compute normalized uncertainties for omitted model
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        
        # Compute normalized uncertainties for full model
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        # Compute inflation ratios (how much OVB inflates uncertainty)
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        # Summary row
        summary_rows.append({
            'rho': rho,
            'beta2': beta2_val,
            'empirical_corr': empirical_corr,
            # Omitted model - MSE
            'mse': mse,
            # Omitted model - Variance decomposition (raw)
            'mean_ale_var': np.mean(ale_var),
            'mean_epi_var': np.mean(epi_var),
            'mean_tot_var': np.mean(tot_var),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            # Omitted model - Variance decomposition (normalized)
            'mean_ale_var_norm': np.mean(ale_var_norm),
            'mean_epi_var_norm': np.mean(epi_var_norm),
            'mean_tot_var_norm': np.mean(tot_var_norm),
            # Omitted model - Entropy decomposition (raw)
            'mean_ale_entropy': np.mean(ale_entropy),
            'mean_epi_entropy': np.mean(epi_entropy),
            'mean_tot_entropy': np.mean(tot_entropy),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            # Omitted model - Entropy decomposition (normalized)
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm),
            'mean_epi_entropy_norm': np.mean(epi_entropy_norm),
            'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            # Full model - MSE
            'mse_full': mse_full,
            # Full model - Variance decomposition (raw)
            'mean_ale_var_full': mean_ale_var_full,
            'mean_epi_var_full': mean_epi_var_full,
            'mean_tot_var_full': np.mean(tot_var_full),
            # Full model - Variance decomposition (normalized)
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm),
            'mean_epi_var_full_norm': np.mean(epi_var_full_norm),
            'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            # Full model - Entropy decomposition (raw)
            'mean_ale_entropy_full': mean_ale_entropy_full,
            'mean_epi_entropy_full': mean_epi_entropy_full,
            'mean_tot_entropy_full': np.mean(tot_entropy_full),
            # Full model - Entropy decomposition (normalized)
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm),
            'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm),
            'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            # Inflation ratios
            'au_inflation_var': au_inflation_var,
            'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy,
            'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        # Print comparison
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Omitted] Variance - AU: {np.mean(ale_var):.4f}, EU: {np.mean(epi_var):.4f}")
        print(f"  [Omitted] Entropy  - AU: {np.mean(ale_entropy):.4f}, EU: {np.mean(epi_entropy):.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  [Full]    Variance - AU: {mean_ale_var_full:.4f}, EU: {mean_epi_var_full:.4f}")
        print(f"  [Full]    Entropy  - AU: {mean_ale_entropy_full:.4f}, EU: {mean_epi_entropy_full:.4f}")
        print(f"  Inflation (Var)     - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
        print(f"  Inflation (Entropy) - AU: {au_inflation_entropy:+.2%}, EU: {eu_inflation_entropy:+.2%}")
        
        # Generate individual plots for this beta2 value
        if save_plots:
            title_base = f"MC Dropout OVB (rho={rho}, beta2={beta2_val})"
            
            # Variance-based plots (std deviation bands)
            plot_uncertainties_no_ood(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_var, epi_var, tot_var,
                title=f"{title_base} - Variance",
                noise_type=noise_type, func_type=func_type
            )
            
            # Normalized variance-based
            plot_uncertainties_no_ood_normalized(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_var, epi_var, tot_var,
                title=f"{title_base} - Variance Normalized",
                noise_type=noise_type, func_type=func_type
            )
            
            # Entropy-based plots
            plot_uncertainties_entropy_no_ood(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_entropy, epi_entropy, tot_entropy,
                title=f"{title_base} - Entropy",
                noise_type=noise_type, func_type=func_type
            )
            
            # Normalized entropy-based
            plot_uncertainties_entropy_no_ood_normalized(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_entropy, epi_entropy, tot_entropy,
                title=f"{title_base} - Entropy Normalized",
                noise_type=noise_type, func_type=func_type
            )
            
            # Entropy lines (raw nats)
            plot_entropy_lines_no_ood(
                X, Y, x_grid, y_grid_clean, mu_pred,
                ale_entropy, epi_entropy, tot_entropy,
                title=f"{title_base} - Entropy Lines",
                noise_type=noise_type, func_type=func_type
            )
        
        # Save model outputs for later reuse
        if save_dir:
            save_ovb_model_outputs(
                X=X, Z=Z, Y=Y, x_grid=x_grid, y_grid_clean=y_grid_clean,
                mu_pred=mu_pred, mu_samples=mu_samples, sigma2_samples=sigma2_samples,
                ale_var=ale_var, epi_var=epi_var, tot_var=tot_var,
                ale_entropy=ale_entropy, epi_entropy=epi_entropy, tot_entropy=tot_entropy,
                rho=rho, beta2=beta2_val, func_type=func_type, noise_type=noise_type,
                results_dir=save_dir, param_name='beta2'
            )
    
    results_df = pd.DataFrame(summary_rows)
    
    # Generate summary plots
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'beta2', rho, func_type, noise_type, save_dir)
    
    # Save summary stats to Excel
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"ovb_beta2_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


# ============================================================================
# Deep Ensemble OVB Experiments
# ============================================================================

def run_deep_ensemble_ovb_rho_experiment(
    rho_values: list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type = 'linear',
    noise_type = 'heteroscedastic',
    seed: int = 42,
    K: int = 10,
    epochs: int = 500,
    batch_size: int = 32,
    entropy_method: str = 'analytical',
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run Deep Ensemble OVB experiment varying rho (correlation strength) with fixed beta2.
    
    Parameters func_type and noise_type can be either:
    - A single string: 'linear' or 'heteroscedastic'
    - A list of strings: ['linear', 'sin'] - will iterate through all combinations
    """
    # Handle list inputs for func_type and noise_type
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    # If multiple configurations, iterate through all combinations
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                
                df, results = run_deep_ensemble_ovb_rho_experiment(
                    rho_values=rho_values, beta2=beta2, n_train=n_train,
                    train_range=train_range, grid_points=grid_points,
                    func_type=ft, noise_type=nt, seed=seed,
                    K=K, epochs=epochs, batch_size=batch_size,
                    entropy_method=entropy_method, save_plots=save_plots,
                    results_dir=results_dir
                )
                
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    # Single configuration
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    # Create subdirectory structure
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# Deep Ensemble OVB Experiment: Varying rho (X-Z Correlation)")
    print(f"# Fixed beta2 = {beta2}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"# K = {K} ensemble members")
    print(f"{'#'*80}\n")
    
    from Models.Deep_Ensemble import train_ensemble_deep, ensemble_predict_deep
    
    all_results = {}
    summary_rows = []
    
    for rho in rho_values:
        print(f"\n{'='*60}")
        print(f"Training with rho = {rho}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(rho * 100))
        torch.manual_seed(seed + int(rho * 100))
        
        # Generate OVB data
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train, train_range=train_range, grid_points=grid_points,
            noise_type=noise_type, func_type=func_type,
            rho=rho, beta2=beta2, seed=seed + int(rho * 100)
        )
        
        # Train omitted model (on X only)
        print(f"  Training omitted ensemble (X only)...")
        ensemble = train_ensemble_deep(X, Y, batch_size=batch_size, K=K, 
                                        loss_type='beta_nll', beta=0.5, 
                                        epochs=epochs, input_dim=1)
        
        # Predict on grid
        result = ensemble_predict_deep(ensemble, x_grid, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy-based uncertainties
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        # Compute statistics for omitted model
        mu_pred_flat = mu_pred.squeeze()
        y_grid_clean_flat = y_grid_clean.squeeze()
        mse = np.mean((mu_pred_flat - y_grid_clean_flat)**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # Train full model (on X, Z)
        print(f"  Training full ensemble (with Z)...")
        X_full = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
        ensemble_full = train_ensemble_deep(X_full, Y, batch_size=batch_size, K=K,
                                             loss_type='beta_nll', beta=0.5,
                                             epochs=epochs, input_dim=2)
        
        # Predict on training points
        result_full = ensemble_predict_deep(ensemble_full, X_full, return_raw_arrays=True)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        # Compute entropy for full model
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        # MSE for full model
        mse_full = np.mean((mu_pred_full.squeeze() - Y.squeeze())**2)
        
        # 2D grid evaluation for heatmaps
        heatmap_grid_points = 50
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        XZ_grid_flat = np.column_stack([X_grid_2d.ravel(), Z_grid_2d.ravel()]).astype(np.float32)
        
        result_2d = ensemble_predict_deep(ensemble_full, XZ_grid_flat, return_raw_arrays=True)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, (mu_samples_2d, sigma2_samples_2d) = result_2d
        
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(mu_samples_2d, sigma2_samples_2d)
        else:
            entropy_2d = entropy_uncertainty_numerical(mu_samples_2d, sigma2_samples_2d, n_samples=5000, seed=seed)
        
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        
        # Store results
        all_results[rho] = {
            'mu_pred': mu_pred, 'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            'mu_pred_full': mu_pred_full, 'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d.reshape(grid_shape), 'epi_var_2d_grid': epi_var_2d.reshape(grid_shape), 'tot_var_2d_grid': tot_var_2d.reshape(grid_shape),
            'ale_entropy_2d_grid': entropy_2d['aleatoric'].reshape(grid_shape), 'epi_entropy_2d_grid': entropy_2d['epistemic'].reshape(grid_shape), 'tot_entropy_2d_grid': entropy_2d['total'].reshape(grid_shape),
        }
        
        # Compute normalized uncertainties and inflation
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        summary_rows.append({
            'rho': rho, 'beta2': beta2, 'empirical_corr': empirical_corr,
            'mse': mse, 'mse_full': mse_full,
            'mean_ale_var': np.mean(ale_var), 'mean_epi_var': np.mean(epi_var), 'mean_tot_var': np.mean(tot_var),
            'mean_ale_var_norm': np.mean(ale_var_norm), 'mean_epi_var_norm': np.mean(epi_var_norm), 'mean_tot_var_norm': np.mean(tot_var_norm),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            'mean_ale_entropy': np.mean(ale_entropy), 'mean_epi_entropy': np.mean(epi_entropy), 'mean_tot_entropy': np.mean(tot_entropy),
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm), 'mean_epi_entropy_norm': np.mean(epi_entropy_norm), 'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            'mean_ale_var_full': mean_ale_var_full, 'mean_epi_var_full': mean_epi_var_full, 'mean_tot_var_full': np.mean(tot_var_full),
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm), 'mean_epi_var_full_norm': np.mean(epi_var_full_norm), 'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            'mean_ale_entropy_full': mean_ale_entropy_full, 'mean_epi_entropy_full': mean_epi_entropy_full, 'mean_tot_entropy_full': np.mean(tot_entropy_full),
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm), 'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm), 'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            'au_inflation_var': au_inflation_var, 'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy, 'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Omitted] Variance - AU: {np.mean(ale_var):.4f}, EU: {np.mean(epi_var):.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  [Full]    Variance - AU: {mean_ale_var_full:.4f}, EU: {mean_epi_var_full:.4f}")
        print(f"  Inflation (Var) - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
    
    results_df = pd.DataFrame(summary_rows)
    
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'rho', beta2, func_type, noise_type, save_dir)
    
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"deep_ensemble_ovb_rho_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


def run_deep_ensemble_ovb_beta2_experiment(
    beta2_values: list = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type = 'linear',
    noise_type = 'heteroscedastic',
    seed: int = 42,
    K: int = 10,
    epochs: int = 500,
    batch_size: int = 32,
    entropy_method: str = 'analytical',
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run Deep Ensemble OVB experiment varying beta2 (effect of omitted Z) with fixed rho.
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                df, results = run_deep_ensemble_ovb_beta2_experiment(
                    beta2_values=beta2_values, rho=rho, n_train=n_train,
                    train_range=train_range, grid_points=grid_points,
                    func_type=ft, noise_type=nt, seed=seed,
                    K=K, epochs=epochs, batch_size=batch_size,
                    entropy_method=entropy_method, save_plots=save_plots,
                    results_dir=results_dir
                )
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# Deep Ensemble OVB Experiment: Varying beta2 (Effect of Omitted Z)")
    print(f"# Fixed rho = {rho}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"# K = {K} ensemble members")
    print(f"{'#'*80}\n")
    
    from Models.Deep_Ensemble import train_ensemble_deep, ensemble_predict_deep
    
    all_results = {}
    summary_rows = []
    
    for beta2_val in beta2_values:
        print(f"\n{'='*60}")
        print(f"Training with beta2 = {beta2_val}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(beta2_val * 100))
        torch.manual_seed(seed + int(beta2_val * 100))
        
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train, train_range=train_range, grid_points=grid_points,
            noise_type=noise_type, func_type=func_type,
            rho=rho, beta2=beta2_val, seed=seed + int(beta2_val * 100)
        )
        
        # Train omitted model
        print(f"  Training omitted ensemble (X only)...")
        ensemble = train_ensemble_deep(X, Y, batch_size=batch_size, K=K,
                                        loss_type='beta_nll', beta=0.5,
                                        epochs=epochs, input_dim=1)
        
        result = ensemble_predict_deep(ensemble, x_grid, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        mse = np.mean((mu_pred.squeeze() - y_grid_clean.squeeze())**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # Train full model
        print(f"  Training full ensemble (with Z)...")
        X_full = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
        ensemble_full = train_ensemble_deep(X_full, Y, batch_size=batch_size, K=K,
                                             loss_type='beta_nll', beta=0.5,
                                             epochs=epochs, input_dim=2)
        
        result_full = ensemble_predict_deep(ensemble_full, X_full, return_raw_arrays=True)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        mse_full = np.mean((mu_pred_full.squeeze() - Y.squeeze())**2)
        
        # 2D grid
        heatmap_grid_points = 50
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        XZ_grid_flat = np.column_stack([X_grid_2d.ravel(), Z_grid_2d.ravel()]).astype(np.float32)
        
        result_2d = ensemble_predict_deep(ensemble_full, XZ_grid_flat, return_raw_arrays=True)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, _ = result_2d
        
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(result_2d[4][0], result_2d[4][1])
        else:
            entropy_2d = entropy_uncertainty_numerical(result_2d[4][0], result_2d[4][1], n_samples=5000, seed=seed)
        
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        
        all_results[beta2_val] = {
            'mu_pred': mu_pred, 'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            'mu_pred_full': mu_pred_full, 'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d.reshape(grid_shape), 'epi_var_2d_grid': epi_var_2d.reshape(grid_shape), 'tot_var_2d_grid': tot_var_2d.reshape(grid_shape),
            'ale_entropy_2d_grid': entropy_2d['aleatoric'].reshape(grid_shape), 'epi_entropy_2d_grid': entropy_2d['epistemic'].reshape(grid_shape), 'tot_entropy_2d_grid': entropy_2d['total'].reshape(grid_shape),
        }
        
        # Normalized and inflation
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        summary_rows.append({
            'rho': rho, 'beta2': beta2_val, 'empirical_corr': empirical_corr,
            'mse': mse, 'mse_full': mse_full,
            'mean_ale_var': np.mean(ale_var), 'mean_epi_var': np.mean(epi_var), 'mean_tot_var': np.mean(tot_var),
            'mean_ale_var_norm': np.mean(ale_var_norm), 'mean_epi_var_norm': np.mean(epi_var_norm), 'mean_tot_var_norm': np.mean(tot_var_norm),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            'mean_ale_entropy': np.mean(ale_entropy), 'mean_epi_entropy': np.mean(epi_entropy), 'mean_tot_entropy': np.mean(tot_entropy),
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm), 'mean_epi_entropy_norm': np.mean(epi_entropy_norm), 'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            'mean_ale_var_full': mean_ale_var_full, 'mean_epi_var_full': mean_epi_var_full, 'mean_tot_var_full': np.mean(tot_var_full),
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm), 'mean_epi_var_full_norm': np.mean(epi_var_full_norm), 'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            'mean_ale_entropy_full': mean_ale_entropy_full, 'mean_epi_entropy_full': mean_epi_entropy_full, 'mean_tot_entropy_full': np.mean(tot_entropy_full),
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm), 'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm), 'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            'au_inflation_var': au_inflation_var, 'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy, 'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  Inflation (Var) - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
    
    results_df = pd.DataFrame(summary_rows)
    
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'beta2', rho, func_type, noise_type, save_dir)
    
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"deep_ensemble_ovb_beta2_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


# ============================================================================
# BNN OVB Experiments
# ============================================================================

def run_bnn_ovb_rho_experiment(
    rho_values: list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type = 'linear',
    noise_type = 'heteroscedastic',
    seed: int = 42,
    hidden_width: int = 16,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    entropy_method: str = 'analytical',
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run BNN OVB experiment varying rho (correlation strength) with fixed beta2.
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                df, results = run_bnn_ovb_rho_experiment(
                    rho_values=rho_values, beta2=beta2, n_train=n_train,
                    train_range=train_range, grid_points=grid_points,
                    func_type=ft, noise_type=nt, seed=seed,
                    hidden_width=hidden_width, weight_scale=weight_scale,
                    warmup=warmup, samples=samples,
                    entropy_method=entropy_method, save_plots=save_plots,
                    results_dir=results_dir
                )
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# BNN OVB Experiment: Varying rho (X-Z Correlation)")
    print(f"# Fixed beta2 = {beta2}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"# hidden_width = {hidden_width}, warmup = {warmup}, samples = {samples}")
    print(f"{'#'*80}\n")
    
    from Models.BNN import train_bnn, bnn_predict
    
    all_results = {}
    summary_rows = []
    
    for rho in rho_values:
        print(f"\n{'='*60}")
        print(f"Training with rho = {rho}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(rho * 100))
        torch.manual_seed(seed + int(rho * 100))
        
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train, train_range=train_range, grid_points=grid_points,
            noise_type=noise_type, func_type=func_type,
            rho=rho, beta2=beta2, seed=seed + int(rho * 100)
        )
        
        # Train omitted model (on X only)
        print(f"  Training omitted BNN (X only)...")
        mcmc = train_bnn(X, Y, hidden_width=hidden_width, weight_scale=weight_scale,
                         warmup=warmup, samples=samples, seed=seed + int(rho * 100), input_dim=1)
        
        result = bnn_predict(mcmc, x_grid, hidden_width=hidden_width, weight_scale=weight_scale,
                             return_raw_arrays=True, input_dim=1)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        mse = np.mean((mu_pred.squeeze() - y_grid_clean.squeeze())**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # Train full model (on X, Z)
        print(f"  Training full BNN (with Z)...")
        X_full = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
        mcmc_full = train_bnn(X_full, Y, hidden_width=hidden_width, weight_scale=weight_scale,
                              warmup=warmup, samples=samples, seed=seed + int(rho * 100), input_dim=2)
        
        result_full = bnn_predict(mcmc_full, X_full, hidden_width=hidden_width, weight_scale=weight_scale,
                                  return_raw_arrays=True, input_dim=2)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        mse_full = np.mean((mu_pred_full.squeeze() - Y.squeeze())**2)
        
        # 2D grid evaluation
        heatmap_grid_points = 50
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        XZ_grid_flat = np.column_stack([X_grid_2d.ravel(), Z_grid_2d.ravel()]).astype(np.float32)
        
        result_2d = bnn_predict(mcmc_full, XZ_grid_flat, hidden_width=hidden_width, weight_scale=weight_scale,
                                return_raw_arrays=True, input_dim=2)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, (mu_samples_2d, sigma2_samples_2d) = result_2d
        
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(mu_samples_2d, sigma2_samples_2d)
        else:
            entropy_2d = entropy_uncertainty_numerical(mu_samples_2d, sigma2_samples_2d, n_samples=5000, seed=seed)
        
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        
        all_results[rho] = {
            'mu_pred': mu_pred, 'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            'mu_pred_full': mu_pred_full, 'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d.reshape(grid_shape), 'epi_var_2d_grid': epi_var_2d.reshape(grid_shape), 'tot_var_2d_grid': tot_var_2d.reshape(grid_shape),
            'ale_entropy_2d_grid': entropy_2d['aleatoric'].reshape(grid_shape), 'epi_entropy_2d_grid': entropy_2d['epistemic'].reshape(grid_shape), 'tot_entropy_2d_grid': entropy_2d['total'].reshape(grid_shape),
        }
        
        # Compute normalized uncertainties and inflation
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        summary_rows.append({
            'rho': rho, 'beta2': beta2, 'empirical_corr': empirical_corr,
            'mse': mse, 'mse_full': mse_full,
            'mean_ale_var': np.mean(ale_var), 'mean_epi_var': np.mean(epi_var), 'mean_tot_var': np.mean(tot_var),
            'mean_ale_var_norm': np.mean(ale_var_norm), 'mean_epi_var_norm': np.mean(epi_var_norm), 'mean_tot_var_norm': np.mean(tot_var_norm),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            'mean_ale_entropy': np.mean(ale_entropy), 'mean_epi_entropy': np.mean(epi_entropy), 'mean_tot_entropy': np.mean(tot_entropy),
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm), 'mean_epi_entropy_norm': np.mean(epi_entropy_norm), 'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            'mean_ale_var_full': mean_ale_var_full, 'mean_epi_var_full': mean_epi_var_full, 'mean_tot_var_full': np.mean(tot_var_full),
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm), 'mean_epi_var_full_norm': np.mean(epi_var_full_norm), 'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            'mean_ale_entropy_full': mean_ale_entropy_full, 'mean_epi_entropy_full': mean_epi_entropy_full, 'mean_tot_entropy_full': np.mean(tot_entropy_full),
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm), 'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm), 'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            'au_inflation_var': au_inflation_var, 'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy, 'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Omitted] Variance - AU: {np.mean(ale_var):.4f}, EU: {np.mean(epi_var):.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  [Full]    Variance - AU: {mean_ale_var_full:.4f}, EU: {mean_epi_var_full:.4f}")
        print(f"  Inflation (Var) - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
    
    results_df = pd.DataFrame(summary_rows)
    
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'rho', beta2, func_type, noise_type, save_dir)
    
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"bnn_ovb_rho_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


def run_bnn_ovb_beta2_experiment(
    beta2_values: list = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type = 'linear',
    noise_type = 'heteroscedastic',
    seed: int = 42,
    hidden_width: int = 16,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    entropy_method: str = 'analytical',
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run BNN OVB experiment varying beta2 (effect of omitted Z) with fixed rho.
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                df, results = run_bnn_ovb_beta2_experiment(
                    beta2_values=beta2_values, rho=rho, n_train=n_train,
                    train_range=train_range, grid_points=grid_points,
                    func_type=ft, noise_type=nt, seed=seed,
                    hidden_width=hidden_width, weight_scale=weight_scale,
                    warmup=warmup, samples=samples,
                    entropy_method=entropy_method, save_plots=save_plots,
                    results_dir=results_dir
                )
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# BNN OVB Experiment: Varying beta2 (Effect of Omitted Z)")
    print(f"# Fixed rho = {rho}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"# hidden_width = {hidden_width}, warmup = {warmup}, samples = {samples}")
    print(f"{'#'*80}\n")
    
    from Models.BNN import train_bnn, bnn_predict
    
    all_results = {}
    summary_rows = []
    
    for beta2_val in beta2_values:
        print(f"\n{'='*60}")
        print(f"Training with beta2 = {beta2_val}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(beta2_val * 100))
        torch.manual_seed(seed + int(beta2_val * 100))
        
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train, train_range=train_range, grid_points=grid_points,
            noise_type=noise_type, func_type=func_type,
            rho=rho, beta2=beta2_val, seed=seed + int(beta2_val * 100)
        )
        
        # Train omitted model
        print(f"  Training omitted BNN (X only)...")
        mcmc = train_bnn(X, Y, hidden_width=hidden_width, weight_scale=weight_scale,
                         warmup=warmup, samples=samples, seed=seed + int(beta2_val * 100), input_dim=1)
        
        result = bnn_predict(mcmc, x_grid, hidden_width=hidden_width, weight_scale=weight_scale,
                             return_raw_arrays=True, input_dim=1)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        mse = np.mean((mu_pred.squeeze() - y_grid_clean.squeeze())**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # Train full model
        print(f"  Training full BNN (with Z)...")
        X_full = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
        mcmc_full = train_bnn(X_full, Y, hidden_width=hidden_width, weight_scale=weight_scale,
                              warmup=warmup, samples=samples, seed=seed + int(beta2_val * 100), input_dim=2)
        
        result_full = bnn_predict(mcmc_full, X_full, hidden_width=hidden_width, weight_scale=weight_scale,
                                  return_raw_arrays=True, input_dim=2)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        mse_full = np.mean((mu_pred_full.squeeze() - Y.squeeze())**2)
        
        # 2D grid
        heatmap_grid_points = 50
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        XZ_grid_flat = np.column_stack([X_grid_2d.ravel(), Z_grid_2d.ravel()]).astype(np.float32)
        
        result_2d = bnn_predict(mcmc_full, XZ_grid_flat, hidden_width=hidden_width, weight_scale=weight_scale,
                                return_raw_arrays=True, input_dim=2)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, _ = result_2d
        
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(result_2d[4][0], result_2d[4][1])
        else:
            entropy_2d = entropy_uncertainty_numerical(result_2d[4][0], result_2d[4][1], n_samples=5000, seed=seed)
        
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        
        all_results[beta2_val] = {
            'mu_pred': mu_pred, 'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            'mu_pred_full': mu_pred_full, 'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d.reshape(grid_shape), 'epi_var_2d_grid': epi_var_2d.reshape(grid_shape), 'tot_var_2d_grid': tot_var_2d.reshape(grid_shape),
            'ale_entropy_2d_grid': entropy_2d['aleatoric'].reshape(grid_shape), 'epi_entropy_2d_grid': entropy_2d['epistemic'].reshape(grid_shape), 'tot_entropy_2d_grid': entropy_2d['total'].reshape(grid_shape),
        }
        
        # Normalized and inflation
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        summary_rows.append({
            'rho': rho, 'beta2': beta2_val, 'empirical_corr': empirical_corr,
            'mse': mse, 'mse_full': mse_full,
            'mean_ale_var': np.mean(ale_var), 'mean_epi_var': np.mean(epi_var), 'mean_tot_var': np.mean(tot_var),
            'mean_ale_var_norm': np.mean(ale_var_norm), 'mean_epi_var_norm': np.mean(epi_var_norm), 'mean_tot_var_norm': np.mean(tot_var_norm),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            'mean_ale_entropy': np.mean(ale_entropy), 'mean_epi_entropy': np.mean(epi_entropy), 'mean_tot_entropy': np.mean(tot_entropy),
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm), 'mean_epi_entropy_norm': np.mean(epi_entropy_norm), 'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            'mean_ale_var_full': mean_ale_var_full, 'mean_epi_var_full': mean_epi_var_full, 'mean_tot_var_full': np.mean(tot_var_full),
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm), 'mean_epi_var_full_norm': np.mean(epi_var_full_norm), 'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            'mean_ale_entropy_full': mean_ale_entropy_full, 'mean_epi_entropy_full': mean_epi_entropy_full, 'mean_tot_entropy_full': np.mean(tot_entropy_full),
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm), 'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm), 'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            'au_inflation_var': au_inflation_var, 'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy, 'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  Inflation (Var) - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
    
    results_df = pd.DataFrame(summary_rows)
    
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'beta2', rho, func_type, noise_type, save_dir)
    
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"bnn_ovb_beta2_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


# ============================================================================
# BAMLSS OVB Experiments
# ============================================================================

def run_bamlss_ovb_rho_experiment(
    rho_values: list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type = 'linear',
    noise_type = 'heteroscedastic',
    seed: int = 42,
    n_iter: int = 12000,
    burnin: int = 2000,
    thin: int = 10,
    nsamples: int = 1000,
    entropy_method: str = 'analytical',
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run BAMLSS OVB experiment varying rho (correlation strength) with fixed beta2.
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                df, results = run_bamlss_ovb_rho_experiment(
                    rho_values=rho_values, beta2=beta2, n_train=n_train,
                    train_range=train_range, grid_points=grid_points,
                    func_type=ft, noise_type=nt, seed=seed,
                    n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples,
                    entropy_method=entropy_method, save_plots=save_plots,
                    results_dir=results_dir
                )
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# BAMLSS OVB Experiment: Varying rho (X-Z Correlation)")
    print(f"# Fixed beta2 = {beta2}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"# n_iter = {n_iter}, burnin = {burnin}, nsamples = {nsamples}")
    print(f"{'#'*80}\n")
    
    from Models.BAMLSS import bamlss_predict, bamlss_predict_2d
    
    all_results = {}
    summary_rows = []
    
    for rho in rho_values:
        print(f"\n{'='*60}")
        print(f"Training with rho = {rho}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(rho * 100))
        
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train, train_range=train_range, grid_points=grid_points,
            noise_type=noise_type, func_type=func_type,
            rho=rho, beta2=beta2, seed=seed + int(rho * 100)
        )
        
        # Train omitted model (on X only)
        print(f"  Training omitted BAMLSS (X only)...")
        result = bamlss_predict(X, Y, x_grid, return_raw_arrays=True, 
                                n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        mse = np.mean((mu_pred.squeeze() - y_grid_clean.squeeze())**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # Train full model (on X, Z) - predict on training points
        print(f"  Training full BAMLSS (with Z)...")
        result_full = bamlss_predict_2d(X.squeeze(), Z, Y, X.squeeze(), Z, return_raw_arrays=True,
                                        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        mse_full = np.mean((mu_pred_full.squeeze() - Y.squeeze())**2)
        
        # 2D grid evaluation for heatmaps
        heatmap_grid_points = 30  # Smaller for BAMLSS (slower)
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        
        print(f"  Evaluating on 2D grid ({heatmap_grid_points}x{heatmap_grid_points})...")
        result_2d = bamlss_predict_2d(X.squeeze(), Z, Y, X_grid_2d.ravel(), Z_grid_2d.ravel(), 
                                      return_raw_arrays=True, n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, (mu_samples_2d, sigma2_samples_2d) = result_2d
        
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(mu_samples_2d, sigma2_samples_2d)
        else:
            entropy_2d = entropy_uncertainty_numerical(mu_samples_2d, sigma2_samples_2d, n_samples=5000, seed=seed)
        
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        
        all_results[rho] = {
            'mu_pred': mu_pred, 'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            'mu_pred_full': mu_pred_full, 'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d.reshape(grid_shape), 'epi_var_2d_grid': epi_var_2d.reshape(grid_shape), 'tot_var_2d_grid': tot_var_2d.reshape(grid_shape),
            'ale_entropy_2d_grid': entropy_2d['aleatoric'].reshape(grid_shape), 'epi_entropy_2d_grid': entropy_2d['epistemic'].reshape(grid_shape), 'tot_entropy_2d_grid': entropy_2d['total'].reshape(grid_shape),
        }
        
        # Compute normalized uncertainties and inflation
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        summary_rows.append({
            'rho': rho, 'beta2': beta2, 'empirical_corr': empirical_corr,
            'mse': mse, 'mse_full': mse_full,
            'mean_ale_var': np.mean(ale_var), 'mean_epi_var': np.mean(epi_var), 'mean_tot_var': np.mean(tot_var),
            'mean_ale_var_norm': np.mean(ale_var_norm), 'mean_epi_var_norm': np.mean(epi_var_norm), 'mean_tot_var_norm': np.mean(tot_var_norm),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            'mean_ale_entropy': np.mean(ale_entropy), 'mean_epi_entropy': np.mean(epi_entropy), 'mean_tot_entropy': np.mean(tot_entropy),
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm), 'mean_epi_entropy_norm': np.mean(epi_entropy_norm), 'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            'mean_ale_var_full': mean_ale_var_full, 'mean_epi_var_full': mean_epi_var_full, 'mean_tot_var_full': np.mean(tot_var_full),
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm), 'mean_epi_var_full_norm': np.mean(epi_var_full_norm), 'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            'mean_ale_entropy_full': mean_ale_entropy_full, 'mean_epi_entropy_full': mean_epi_entropy_full, 'mean_tot_entropy_full': np.mean(tot_entropy_full),
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm), 'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm), 'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            'au_inflation_var': au_inflation_var, 'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy, 'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Omitted] Variance - AU: {np.mean(ale_var):.4f}, EU: {np.mean(epi_var):.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  [Full]    Variance - AU: {mean_ale_var_full:.4f}, EU: {mean_epi_var_full:.4f}")
        print(f"  Inflation (Var) - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
    
    results_df = pd.DataFrame(summary_rows)
    
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'rho', beta2, func_type, noise_type, save_dir)
    
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"bamlss_ovb_rho_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


def run_bamlss_ovb_beta2_experiment(
    beta2_values: list = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (0.0, 10.0),
    grid_points: int = 600,
    func_type = 'linear',
    noise_type = 'heteroscedastic',
    seed: int = 42,
    n_iter: int = 12000,
    burnin: int = 2000,
    thin: int = 10,
    nsamples: int = 1000,
    entropy_method: str = 'analytical',
    save_plots: bool = True,
    results_dir: Path = None
):
    """
    Run BAMLSS OVB experiment varying beta2 (effect of omitted Z) with fixed rho.
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    noise_types = [noise_type] if isinstance(noise_type, str) else list(noise_type)
    
    if len(func_types) > 1 or len(noise_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            for nt in noise_types:
                print(f"\n{'='*80}")
                print(f"Running config: func_type={ft}, noise_type={nt}")
                print(f"{'='*80}")
                df, results = run_bamlss_ovb_beta2_experiment(
                    beta2_values=beta2_values, rho=rho, n_train=n_train,
                    train_range=train_range, grid_points=grid_points,
                    func_type=ft, noise_type=nt, seed=seed,
                    n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples,
                    entropy_method=entropy_method, save_plots=save_plots,
                    results_dir=results_dir
                )
                df['func_type'] = ft
                df['noise_type'] = nt
                combined_dfs.append(df)
                combined_results[(ft, nt)] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    noise_type = noise_types[0]
    
    if results_dir:
        save_dir = results_dir / noise_type / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    print(f"\n{'#'*80}")
    print(f"# BAMLSS OVB Experiment: Varying beta2 (Effect of Omitted Z)")
    print(f"# Fixed rho = {rho}, func_type = {func_type}, noise_type = {noise_type}")
    print(f"# n_iter = {n_iter}, burnin = {burnin}, nsamples = {nsamples}")
    print(f"{'#'*80}\n")
    
    from Models.BAMLSS import bamlss_predict, bamlss_predict_2d
    
    all_results = {}
    summary_rows = []
    
    for beta2_val in beta2_values:
        print(f"\n{'='*60}")
        print(f"Training with beta2 = {beta2_val}")
        print(f"{'='*60}")
        
        np.random.seed(seed + int(beta2_val * 100))
        
        X, Z, Y, x_grid, y_grid_clean = generate_ovb_data(
            n_train=n_train, train_range=train_range, grid_points=grid_points,
            noise_type=noise_type, func_type=func_type,
            rho=rho, beta2=beta2_val, seed=seed + int(beta2_val * 100)
        )
        
        # Train omitted model
        print(f"  Training omitted BAMLSS (X only)...")
        result = bamlss_predict(X, Y, x_grid, return_raw_arrays=True,
                                n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        if entropy_method == 'analytical':
            entropy_results = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        else:
            entropy_results = entropy_uncertainty_numerical(mu_samples, sigma2_samples, n_samples=5000, seed=seed)
        
        ale_entropy = entropy_results['aleatoric']
        epi_entropy = entropy_results['epistemic']
        tot_entropy = entropy_results['total']
        
        mse = np.mean((mu_pred.squeeze() - y_grid_clean.squeeze())**2)
        empirical_corr = np.corrcoef(X.squeeze(), Z)[0, 1]
        
        # Train full model
        print(f"  Training full BAMLSS (with Z)...")
        result_full = bamlss_predict_2d(X.squeeze(), Z, Y, X.squeeze(), Z, return_raw_arrays=True,
                                        n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples)
        mu_pred_full, ale_var_full, epi_var_full, tot_var_full, (mu_samples_full, sigma2_samples_full) = result_full
        
        if entropy_method == 'analytical':
            entropy_results_full = entropy_uncertainty_analytical(mu_samples_full, sigma2_samples_full)
        else:
            entropy_results_full = entropy_uncertainty_numerical(mu_samples_full, sigma2_samples_full, n_samples=5000, seed=seed)
        
        ale_entropy_full = entropy_results_full['aleatoric']
        epi_entropy_full = entropy_results_full['epistemic']
        tot_entropy_full = entropy_results_full['total']
        
        mse_full = np.mean((mu_pred_full.squeeze() - Y.squeeze())**2)
        
        # 2D grid
        heatmap_grid_points = 30
        x_vals_2d = np.linspace(train_range[0], train_range[1], heatmap_grid_points)
        z_vals_2d = np.linspace(Z.min(), Z.max(), heatmap_grid_points)
        X_grid_2d, Z_grid_2d = np.meshgrid(x_vals_2d, z_vals_2d)
        
        print(f"  Evaluating on 2D grid ({heatmap_grid_points}x{heatmap_grid_points})...")
        result_2d = bamlss_predict_2d(X.squeeze(), Z, Y, X_grid_2d.ravel(), Z_grid_2d.ravel(),
                                      return_raw_arrays=True, n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples)
        mu_pred_2d, ale_var_2d, epi_var_2d, tot_var_2d, (mu_samples_2d, sigma2_samples_2d) = result_2d
        
        if entropy_method == 'analytical':
            entropy_2d = entropy_uncertainty_analytical(mu_samples_2d, sigma2_samples_2d)
        else:
            entropy_2d = entropy_uncertainty_numerical(mu_samples_2d, sigma2_samples_2d, n_samples=5000, seed=seed)
        
        grid_shape = (heatmap_grid_points, heatmap_grid_points)
        
        all_results[beta2_val] = {
            'mu_pred': mu_pred, 'ale_var': ale_var, 'epi_var': epi_var, 'tot_var': tot_var,
            'ale_entropy': ale_entropy, 'epi_entropy': epi_entropy, 'tot_entropy': tot_entropy,
            'X': X, 'Z': Z, 'Y': Y, 'x_grid': x_grid, 'y_grid_clean': y_grid_clean,
            'mu_pred_full': mu_pred_full, 'ale_var_full': ale_var_full, 'epi_var_full': epi_var_full, 'tot_var_full': tot_var_full,
            'ale_entropy_full': ale_entropy_full, 'epi_entropy_full': epi_entropy_full, 'tot_entropy_full': tot_entropy_full,
            'X_grid_2d': X_grid_2d, 'Z_grid_2d': Z_grid_2d,
            'ale_var_2d_grid': ale_var_2d.reshape(grid_shape), 'epi_var_2d_grid': epi_var_2d.reshape(grid_shape), 'tot_var_2d_grid': tot_var_2d.reshape(grid_shape),
            'ale_entropy_2d_grid': entropy_2d['aleatoric'].reshape(grid_shape), 'epi_entropy_2d_grid': entropy_2d['epistemic'].reshape(grid_shape), 'tot_entropy_2d_grid': entropy_2d['total'].reshape(grid_shape),
        }
        
        # Normalized and inflation
        ale_var_norm = _normalize_minmax(ale_var)
        epi_var_norm = _normalize_minmax(epi_var)
        tot_var_norm = _normalize_minmax(tot_var)
        ale_var_full_norm = _normalize_minmax(ale_var_full)
        epi_var_full_norm = _normalize_minmax(epi_var_full)
        tot_var_full_norm = _normalize_minmax(tot_var_full)
        ale_entropy_norm = _normalize_minmax(ale_entropy)
        epi_entropy_norm = _normalize_minmax(epi_entropy)
        tot_entropy_norm = _normalize_minmax(tot_entropy)
        ale_entropy_full_norm = _normalize_minmax(ale_entropy_full)
        epi_entropy_full_norm = _normalize_minmax(epi_entropy_full)
        tot_entropy_full_norm = _normalize_minmax(tot_entropy_full)
        
        mean_ale_var_full = np.mean(ale_var_full)
        mean_epi_var_full = np.mean(epi_var_full)
        au_inflation_var = (np.mean(ale_var) - mean_ale_var_full) / (mean_ale_var_full + 1e-10)
        eu_inflation_var = (np.mean(epi_var) - mean_epi_var_full) / (mean_epi_var_full + 1e-10)
        mean_ale_entropy_full = np.mean(ale_entropy_full)
        mean_epi_entropy_full = np.mean(epi_entropy_full)
        au_inflation_entropy = (np.mean(ale_entropy) - mean_ale_entropy_full) / (np.abs(mean_ale_entropy_full) + 1e-10)
        eu_inflation_entropy = (np.mean(epi_entropy) - mean_epi_entropy_full) / (np.abs(mean_epi_entropy_full) + 1e-10)
        
        summary_rows.append({
            'rho': rho, 'beta2': beta2_val, 'empirical_corr': empirical_corr,
            'mse': mse, 'mse_full': mse_full,
            'mean_ale_var': np.mean(ale_var), 'mean_epi_var': np.mean(epi_var), 'mean_tot_var': np.mean(tot_var),
            'mean_ale_var_norm': np.mean(ale_var_norm), 'mean_epi_var_norm': np.mean(epi_var_norm), 'mean_tot_var_norm': np.mean(tot_var_norm),
            'au_eu_corr_var': np.corrcoef(ale_var.flatten(), epi_var.flatten())[0, 1],
            'mean_ale_entropy': np.mean(ale_entropy), 'mean_epi_entropy': np.mean(epi_entropy), 'mean_tot_entropy': np.mean(tot_entropy),
            'mean_ale_entropy_norm': np.mean(ale_entropy_norm), 'mean_epi_entropy_norm': np.mean(epi_entropy_norm), 'mean_tot_entropy_norm': np.mean(tot_entropy_norm),
            'au_eu_corr_entropy': np.corrcoef(ale_entropy.flatten(), epi_entropy.flatten())[0, 1],
            'mean_ale_var_full': mean_ale_var_full, 'mean_epi_var_full': mean_epi_var_full, 'mean_tot_var_full': np.mean(tot_var_full),
            'mean_ale_var_full_norm': np.mean(ale_var_full_norm), 'mean_epi_var_full_norm': np.mean(epi_var_full_norm), 'mean_tot_var_full_norm': np.mean(tot_var_full_norm),
            'mean_ale_entropy_full': mean_ale_entropy_full, 'mean_epi_entropy_full': mean_epi_entropy_full, 'mean_tot_entropy_full': np.mean(tot_entropy_full),
            'mean_ale_entropy_full_norm': np.mean(ale_entropy_full_norm), 'mean_epi_entropy_full_norm': np.mean(epi_entropy_full_norm), 'mean_tot_entropy_full_norm': np.mean(tot_entropy_full_norm),
            'au_inflation_var': au_inflation_var, 'eu_inflation_var': eu_inflation_var,
            'au_inflation_entropy': au_inflation_entropy, 'eu_inflation_entropy': eu_inflation_entropy,
        })
        
        print(f"  [Omitted] MSE: {mse:.4f}")
        print(f"  [Full]    MSE: {mse_full:.4f}")
        print(f"  Inflation (Var) - AU: {au_inflation_var:+.2%}, EU: {eu_inflation_var:+.2%}")
    
    results_df = pd.DataFrame(summary_rows)
    
    if save_plots:
        _plot_ovb_experiment_results(results_df, all_results, 'beta2', rho, func_type, noise_type, save_dir)
    
    if save_dir:
        date = datetime.now().strftime('%Y%m%d')
        excel_path = save_dir / f"bamlss_ovb_beta2_stats_{date}.xlsx"
        results_df.to_excel(excel_path, index=False)
        print(f"\nSaved summary stats to: {excel_path}")
    
    return results_df, all_results


# ============================================================================
# Plotting Functions
# ============================================================================

def _plot_ovb_experiment_results(results_df, all_results, vary_param, fixed_param_value, func_type, noise_type, results_dir=None):
    """Generate plots for OVB experiment results with omitted vs full model comparison."""
    
    if vary_param == 'rho':
        x_label = 'rho (X-Z Correlation)'
        x_col = 'rho'
        title_suffix = f'beta2={fixed_param_value}'
    else:
        x_label = 'beta2 (Effect of Omitted Z)'
        x_col = 'beta2'
        title_suffix = f'rho={fixed_param_value}'
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    # Row 1: Omitted model - Variance and Entropy (normalized)
    ax1 = axes[0, 0]
    ax1.plot(results_df[x_col], results_df['mean_ale_var_norm'], 'o-', label='AU (Omitted)', color='blue')
    ax1.plot(results_df[x_col], results_df['mean_epi_var_norm'], 's-', label='EU (Omitted)', color='orange')
    ax1.plot(results_df[x_col], results_df['mean_ale_var_full_norm'], 'o--', label='AU (Full)', color='blue', alpha=0.5)
    ax1.plot(results_df[x_col], results_df['mean_epi_var_full_norm'], 's--', label='EU (Full)', color='orange', alpha=0.5)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Mean Normalized Variance')
    ax1.set_title(f'Variance: Omitted vs Full ({title_suffix})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(results_df[x_col], results_df['mean_ale_entropy_norm'], 'o-', label='AU (Omitted)', color='blue')
    ax2.plot(results_df[x_col], results_df['mean_epi_entropy_norm'], 's-', label='EU (Omitted)', color='orange')
    ax2.plot(results_df[x_col], results_df['mean_ale_entropy_full_norm'], 'o--', label='AU (Full)', color='blue', alpha=0.5)
    ax2.plot(results_df[x_col], results_df['mean_epi_entropy_full_norm'], 's--', label='EU (Full)', color='orange', alpha=0.5)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Mean Normalized Entropy')
    ax2.set_title(f'Entropy: Omitted vs Full ({title_suffix})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    ax3.plot(results_df[x_col], results_df['mse'], 'o-', label='Omitted', color='red')
    ax3.plot(results_df[x_col], results_df['mse_full'], 's--', label='Full', color='red', alpha=0.5)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel('MSE')
    ax3.set_title(f'MSE: Omitted vs Full')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Inflation ratios
    ax4 = axes[1, 0]
    ax4.plot(results_df[x_col], results_df['au_inflation_var'] * 100, 'o-', label='AU Inflation', color='blue')
    ax4.plot(results_df[x_col], results_df['eu_inflation_var'] * 100, 's-', label='EU Inflation', color='orange')
    ax4.set_xlabel(x_label)
    ax4.set_ylabel('Inflation (%)')
    ax4.set_title(f'Variance Inflation from OVB')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    ax5.plot(results_df[x_col], results_df['au_inflation_entropy'] * 100, 'o-', label='AU Inflation', color='blue')
    ax5.plot(results_df[x_col], results_df['eu_inflation_entropy'] * 100, 's-', label='EU Inflation', color='orange')
    ax5.set_xlabel(x_label)
    ax5.set_ylabel('Inflation (%)')
    ax5.set_title(f'Entropy Inflation from OVB')
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = axes[1, 2]
    ax6.plot(results_df[x_col], results_df['empirical_corr'], 'o-', color='teal')
    ax6.set_xlabel(x_label)
    ax6.set_ylabel('Empirical Corr(X, Z)')
    ax6.set_title(f'Empirical X-Z Correlation')
    ax6.grid(True, alpha=0.3)
    
    # Row 3: AU-EU correlations and raw values
    ax7 = axes[2, 0]
    ax7.plot(results_df[x_col], results_df['au_eu_corr_var'], 'o-', color='purple')
    ax7.set_xlabel(x_label)
    ax7.set_ylabel('AU-EU Correlation')
    ax7.set_title(f'AU-EU Correlation (Variance)')
    ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax7.grid(True, alpha=0.3)
    
    ax8 = axes[2, 1]
    ax8.plot(results_df[x_col], results_df['au_eu_corr_entropy'], 'o-', color='purple')
    ax8.set_xlabel(x_label)
    ax8.set_ylabel('AU-EU Correlation')
    ax8.set_title(f'AU-EU Correlation (Entropy)')
    ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax8.grid(True, alpha=0.3)
    
    ax9 = axes[2, 2]
    ax9.plot(results_df[x_col], results_df['mean_ale_var'], 'o-', label='AU (Omitted)', color='blue')
    ax9.plot(results_df[x_col], results_df['mean_epi_var'], 's-', label='EU (Omitted)', color='orange')
    ax9.plot(results_df[x_col], results_df['mean_ale_var_full'], 'o--', label='AU (Full)', color='blue', alpha=0.5)
    ax9.plot(results_df[x_col], results_df['mean_epi_var_full'], 's--', label='EU (Full)', color='orange', alpha=0.5)
    ax9.set_xlabel(x_label)
    ax9.set_ylabel('Mean Raw Variance')
    ax9.set_title(f'Raw Variance Comparison')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'OVB Experiment: Omitted vs Full Model - {func_type} - {noise_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if results_dir:
        save_path = results_dir / f"ovb_{vary_param}_experiment.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_ovb_uncertainty_comparison(results_df, vary_param='rho'):
    """
    Create a comparison bar plot of variance vs entropy decomposition.
    """
    if vary_param == 'rho':
        x_col = 'rho'
        x_label = 'rho'
    else:
        x_col = 'beta2'
        x_label = 'beta2'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    # Variance
    ax1 = axes[0]
    ax1.bar(x - width/2, results_df['mean_ale_var'], width, label='AU', color='tab:blue', alpha=0.7)
    ax1.bar(x + width/2, results_df['mean_epi_var'], width, label='EU', color='tab:orange', alpha=0.7)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Mean Variance')
    ax1.set_title('Variance Decomposition')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{v:.2f}' for v in results_df[x_col]])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Entropy
    ax2 = axes[1]
    ax2.bar(x - width/2, results_df['mean_ale_entropy'], width, label='AU', color='tab:blue', alpha=0.7)
    ax2.bar(x + width/2, results_df['mean_epi_entropy'], width, label='EU', color='tab:orange', alpha=0.7)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Mean Entropy (nats)')
    ax2.set_title('Entropy (IT) Decomposition')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{v:.2f}' for v in results_df[x_col]])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Detailed Comparison Visualizations
# ============================================================================

def plot_ovb_heatmap_comparison(
    all_results: dict,
    param_value,
    decomposition: str = 'variance',
    results_dir=None
):
    """
    Plot 2D heatmaps comparing full model AU/EU over (X, Z) space
    with omitted model overlay.
    
    Parameters:
    -----------
    all_results : dict
        Results from run_mc_dropout_ovb_rho_experiment or run_mc_dropout_ovb_beta2_experiment
    param_value : float
        The rho or beta2 value to visualize
    decomposition : str
        'variance' or 'entropy'
    results_dir : Path, optional
        Directory to save plot
    """
    if param_value not in all_results:
        raise ValueError(f"param_value {param_value} not found in results. Available: {list(all_results.keys())}")
    
    res = all_results[param_value]
    
    # Get 2D grid data for full model
    X_grid_2d = res['X_grid_2d']
    Z_grid_2d = res['Z_grid_2d']
    
    if decomposition == 'variance':
        au_2d = res['ale_var_2d_grid']
        eu_2d = res['epi_var_2d_grid']
        tu_2d = res['tot_var_2d_grid']
        au_omitted = res['ale_var']
        eu_omitted = res['epi_var']
        title_base = 'Variance'
        cbar_label = 'Variance'
    else:
        au_2d = res['ale_entropy_2d_grid']
        eu_2d = res['epi_entropy_2d_grid']
        tu_2d = res['tot_entropy_2d_grid']
        au_omitted = res['ale_entropy']
        eu_omitted = res['epi_entropy']
        title_base = 'Entropy'
        cbar_label = 'Entropy (nats)'
    
    x_grid = res['x_grid']
    X_train = res['X']
    Z_train = res['Z']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Full model heatmaps
    # AU heatmap
    im1 = axes[0, 0].pcolormesh(X_grid_2d, Z_grid_2d, au_2d, shading='auto', cmap='viridis')
    axes[0, 0].scatter(X_train.squeeze(), Z_train, c='white', s=5, alpha=0.3, edgecolors='none')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Z')
    axes[0, 0].set_title(f'Full Model - AU ({title_base})')
    plt.colorbar(im1, ax=axes[0, 0], label=cbar_label)
    
    # EU heatmap
    im2 = axes[0, 1].pcolormesh(X_grid_2d, Z_grid_2d, eu_2d, shading='auto', cmap='plasma')
    axes[0, 1].scatter(X_train.squeeze(), Z_train, c='white', s=5, alpha=0.3, edgecolors='none')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title(f'Full Model - EU ({title_base})')
    plt.colorbar(im2, ax=axes[0, 1], label=cbar_label)
    
    # TU heatmap
    im3 = axes[0, 2].pcolormesh(X_grid_2d, Z_grid_2d, tu_2d, shading='auto', cmap='inferno')
    axes[0, 2].scatter(X_train.squeeze(), Z_train, c='white', s=5, alpha=0.3, edgecolors='none')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Z')
    axes[0, 2].set_title(f'Full Model - TU ({title_base})')
    plt.colorbar(im3, ax=axes[0, 2], label=cbar_label)
    
    # Row 2: Omitted model (1D, shown as function of X)
    # AU line plot
    axes[1, 0].plot(x_grid.squeeze(), au_omitted.squeeze(), 'b-', linewidth=2, label='Omitted AU')
    axes[1, 0].fill_between(x_grid.squeeze(), 0, au_omitted.squeeze(), alpha=0.3)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel(cbar_label)
    axes[1, 0].set_title(f'Omitted Model - AU ({title_base})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # EU line plot
    axes[1, 1].plot(x_grid.squeeze(), eu_omitted.squeeze(), 'orange', linewidth=2, label='Omitted EU')
    axes[1, 1].fill_between(x_grid.squeeze(), 0, eu_omitted.squeeze(), alpha=0.3, color='orange')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel(cbar_label)
    axes[1, 1].set_title(f'Omitted Model - EU ({title_base})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Comparison: Mean AU/EU across Z for full model vs omitted
    au_2d_mean_z = np.mean(au_2d, axis=0)  # Mean across Z dimension
    eu_2d_mean_z = np.mean(eu_2d, axis=0)
    x_vals_2d = X_grid_2d[0, :]
    
    axes[1, 2].plot(x_vals_2d, au_2d_mean_z, 'b--', linewidth=2, label='Full AU (mean over Z)')
    axes[1, 2].plot(x_vals_2d, eu_2d_mean_z, 'orange', linestyle='--', linewidth=2, label='Full EU (mean over Z)')
    axes[1, 2].plot(x_grid.squeeze(), au_omitted.squeeze(), 'b-', linewidth=2, alpha=0.7, label='Omitted AU')
    axes[1, 2].plot(x_grid.squeeze(), eu_omitted.squeeze(), 'orange', linewidth=2, alpha=0.7, label='Omitted EU')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel(cbar_label)
    axes[1, 2].set_title(f'Comparison: Omitted vs Full (marginalized)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'OVB Heatmap Comparison - {title_base} (param={param_value})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if results_dir:
        save_path = results_dir / f"ovb_heatmap_{decomposition}_param{param_value}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_ovb_marginal_comparison(
    all_results: dict,
    param_value,
    decomposition: str = 'variance',
    results_dir=None
):
    """
    Plot marginal histograms comparing AU/EU distributions between omitted and full models.
    
    Parameters:
    -----------
    all_results : dict
        Results from run_mc_dropout_ovb_rho_experiment or run_mc_dropout_ovb_beta2_experiment
    param_value : float
        The rho or beta2 value to visualize
    decomposition : str
        'variance' or 'entropy'
    results_dir : Path, optional
        Directory to save plot
    """
    if param_value not in all_results:
        raise ValueError(f"param_value {param_value} not found in results. Available: {list(all_results.keys())}")
    
    res = all_results[param_value]
    
    if decomposition == 'variance':
        au_omitted = res['ale_var'].flatten()
        eu_omitted = res['epi_var'].flatten()
        au_full = res['ale_var_full'].flatten()
        eu_full = res['epi_var_full'].flatten()
        au_2d = res['ale_var_2d_grid'].flatten()
        eu_2d = res['epi_var_2d_grid'].flatten()
        title_base = 'Variance'
        x_label = 'Variance'
    else:
        au_omitted = res['ale_entropy'].flatten()
        eu_omitted = res['epi_entropy'].flatten()
        au_full = res['ale_entropy_full'].flatten()
        eu_full = res['epi_entropy_full'].flatten()
        au_2d = res['ale_entropy_2d_grid'].flatten()
        eu_2d = res['epi_entropy_2d_grid'].flatten()
        title_base = 'Entropy'
        x_label = 'Entropy (nats)'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Row 1: AU comparison
    # Omitted vs Full (training points)
    ax1 = axes[0, 0]
    ax1.hist(au_omitted, bins=30, alpha=0.6, label='Omitted', color='blue', density=True)
    ax1.hist(au_full, bins=30, alpha=0.6, label='Full (train pts)', color='green', density=True)
    ax1.axvline(np.mean(au_omitted), color='blue', linestyle='--', linewidth=2, label=f'Omitted mean: {np.mean(au_omitted):.4f}')
    ax1.axvline(np.mean(au_full), color='green', linestyle='--', linewidth=2, label=f'Full mean: {np.mean(au_full):.4f}')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Density')
    ax1.set_title(f'AU Distribution - Omitted vs Full (training points)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Omitted vs Full (2D grid)
    ax2 = axes[0, 1]
    ax2.hist(au_omitted, bins=30, alpha=0.6, label='Omitted', color='blue', density=True)
    ax2.hist(au_2d, bins=30, alpha=0.6, label='Full (2D grid)', color='purple', density=True)
    ax2.axvline(np.mean(au_omitted), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(au_2d), color='purple', linestyle='--', linewidth=2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Density')
    ax2.set_title(f'AU Distribution - Omitted vs Full (2D grid)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Row 2: EU comparison
    ax3 = axes[1, 0]
    ax3.hist(eu_omitted, bins=30, alpha=0.6, label='Omitted', color='orange', density=True)
    ax3.hist(eu_full, bins=30, alpha=0.6, label='Full (train pts)', color='green', density=True)
    ax3.axvline(np.mean(eu_omitted), color='orange', linestyle='--', linewidth=2, label=f'Omitted mean: {np.mean(eu_omitted):.4f}')
    ax3.axvline(np.mean(eu_full), color='green', linestyle='--', linewidth=2, label=f'Full mean: {np.mean(eu_full):.4f}')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel('Density')
    ax3.set_title(f'EU Distribution - Omitted vs Full (training points)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.hist(eu_omitted, bins=30, alpha=0.6, label='Omitted', color='orange', density=True)
    ax4.hist(eu_2d, bins=30, alpha=0.6, label='Full (2D grid)', color='purple', density=True)
    ax4.axvline(np.mean(eu_omitted), color='orange', linestyle='--', linewidth=2)
    ax4.axvline(np.mean(eu_2d), color='purple', linestyle='--', linewidth=2)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel('Density')
    ax4.set_title(f'EU Distribution - Omitted vs Full (2D grid)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'OVB Marginal Distribution Comparison - {title_base} (param={param_value})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if results_dir:
        save_path = results_dir / f"ovb_marginal_{decomposition}_param{param_value}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_ovb_z_slices(
    all_results: dict,
    param_value,
    z_percentiles: list = [10, 50, 90],
    decomposition: str = 'variance',
    results_dir=None
):
    """
    Plot AU/EU vs X at specific Z slices (percentiles of Z distribution).
    
    Shows how full model uncertainty varies with Z while omitted model stays constant.
    
    Parameters:
    -----------
    all_results : dict
        Results from run_mc_dropout_ovb_rho_experiment or run_mc_dropout_ovb_beta2_experiment
    param_value : float
        The rho or beta2 value to visualize
    z_percentiles : list
        Percentiles of Z distribution to use as slices (e.g., [10, 50, 90])
    decomposition : str
        'variance' or 'entropy'
    results_dir : Path, optional
        Directory to save plot
    """
    if param_value not in all_results:
        raise ValueError(f"param_value {param_value} not found in results. Available: {list(all_results.keys())}")
    
    res = all_results[param_value]
    
    # Get Z values at specified percentiles
    Z_train = res['Z']
    z_values = np.percentile(Z_train, z_percentiles)
    
    # Get 2D grid data
    X_grid_2d = res['X_grid_2d']
    Z_grid_2d = res['Z_grid_2d']
    x_vals = X_grid_2d[0, :]  # X values along the grid
    
    if decomposition == 'variance':
        au_2d = res['ale_var_2d_grid']
        eu_2d = res['epi_var_2d_grid']
        au_omitted = res['ale_var']
        eu_omitted = res['epi_var']
        title_base = 'Variance'
        y_label = 'Variance'
    else:
        au_2d = res['ale_entropy_2d_grid']
        eu_2d = res['epi_entropy_2d_grid']
        au_omitted = res['ale_entropy']
        eu_omitted = res['epi_entropy']
        title_base = 'Entropy'
        y_label = 'Entropy (nats)'
    
    x_grid_omitted = res['x_grid']
    n_slices = len(z_percentiles)
    
    fig, axes = plt.subplots(2, n_slices, figsize=(5 * n_slices, 8))
    
    for col, (z_pct, z_val) in enumerate(zip(z_percentiles, z_values)):
        # Find the closest row in Z_grid_2d to the target z value
        z_grid_vals = Z_grid_2d[:, 0]
        z_idx = np.argmin(np.abs(z_grid_vals - z_val))
        actual_z = z_grid_vals[z_idx]
        
        # Extract AU/EU slices at this Z
        au_slice = au_2d[z_idx, :]
        eu_slice = eu_2d[z_idx, :]
        
        # Row 1: AU comparison
        ax1 = axes[0, col] if n_slices > 1 else axes[0]
        ax1.plot(x_vals, au_slice, 'b-', linewidth=2, label='Full model')
        ax1.plot(x_grid_omitted.squeeze(), au_omitted.squeeze(), 'r--', linewidth=2, label='Omitted model', alpha=0.7)
        ax1.fill_between(x_vals, 0, au_slice, alpha=0.2, color='blue')
        ax1.set_xlabel('X')
        ax1.set_ylabel(y_label)
        ax1.set_title(f'AU at Z = {actual_z:.2f}\n(p{z_pct})')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Row 2: EU comparison
        ax2 = axes[1, col] if n_slices > 1 else axes[1]
        ax2.plot(x_vals, eu_slice, 'orange', linewidth=2, label='Full model')
        ax2.plot(x_grid_omitted.squeeze(), eu_omitted.squeeze(), 'r--', linewidth=2, label='Omitted model', alpha=0.7)
        ax2.fill_between(x_vals, 0, eu_slice, alpha=0.2, color='orange')
        ax2.set_xlabel('X')
        ax2.set_ylabel(y_label)
        ax2.set_title(f'EU at Z = {actual_z:.2f}\n(p{z_pct})')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'OVB Z-Slice Comparison - {title_base} (param={param_value})\n'
                 f'Omitted model (red dashed) is constant across Z; Full model (solid) varies',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if results_dir:
        save_path = results_dir / f"ovb_zslice_{decomposition}_param{param_value}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
