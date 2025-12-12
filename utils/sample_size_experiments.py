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
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from utils.results_save import save_summary_statistics
from utils.plotting import plot_uncertainties_no_ood


def compute_and_save_statistics(
    uncertainties_by_pct: dict,
    percentages: list,
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str
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
    
    Returns:
    --------
    dict : Statistics dictionary with percentages, averages, correlations, and stats_df
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
    tot_min, tot_max = all_tot.min(), all_tot.max()
    
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
    
    print(f"\n{'='*60}")
    print(f"Normalized Average Uncertainties by Percentage - {function_name} Function - {model_name}")
    print(f"{'='*60}")
    print(f"\n{'Percentage':<12} {'Avg Aleatoric (norm)':<25} {'Avg Epistemic (norm)':<25} {'Avg Total (norm)':<25} {'Correlation (Epi-Ale)':<25}")
    print("-" * 120)
    
    for pct in percentages:
        ale_vals = np.concatenate(uncertainties_by_pct[pct]['ale'])
        epi_vals = np.concatenate(uncertainties_by_pct[pct]['epi'])
        tot_vals = np.concatenate(uncertainties_by_pct[pct]['tot'])
        
        ale_norm = normalize(ale_vals, ale_min, ale_max)
        epi_norm = normalize(epi_vals, epi_min, epi_max)
        tot_norm = normalize(tot_vals, tot_min, tot_max)
        
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
        func_type=func_type, model_name=model_name
    )
    plt.show()
    plt.close(fig)
    
    return {
        'percentages': percentages,
        'avg_ale_norm': avg_ale_norm_list,
        'avg_epi_norm': avg_epi_norm_list,
        'avg_tot_norm': avg_tot_norm_list,
        'correlations': correlation_list,
        'stats_df': stats_df
    }


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
    mc_samples: int = 20
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
        
        # Train and evaluate for each percentage
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
            
            # Store uncertainties
            uncertainties_by_pct[pct]['ale'].append(ale_var)
            uncertainties_by_pct[pct]['epi'].append(epi_var)
            uncertainties_by_pct[pct]['tot'].append(tot_var)
            
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
            function_names[func_type], noise_type, func_type, 'MC_Dropout'
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
    K: int = 5
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
        
        # Train and evaluate for each percentage
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
            
            # Normalize input
            x_mean, x_std = normalize_x(x_train_subset)
            x_train_subset_norm = normalize_x_data(x_train_subset, x_mean, x_std)
            x_grid_norm = normalize_x_data(x_grid, x_mean, x_std)
            
            # Train ensemble
            ensemble = train_ensemble_deep(
                x_train_subset_norm, y_train_subset,
                batch_size=batch_size, K=K,
                loss_type='beta_nll', beta=beta
            )
            
            # Make predictions
            mu_pred, ale_var, epi_var, tot_var = ensemble_predict_deep(ensemble, x_grid_norm)
            
            # Store uncertainties
            uncertainties_by_pct[pct]['ale'].append(ale_var)
            uncertainties_by_pct[pct]['epi'].append(epi_var)
            uncertainties_by_pct[pct]['tot'].append(tot_var)
            
            # Plot
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
            function_names[func_type], noise_type, func_type, 'Deep_Ensemble'
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
    chains: int = 1
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
        
        # Train and evaluate for each percentage
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
            
            # Normalize input
            x_mean, x_std = bnn_normalize_x(x_train_subset)
            x_train_subset_norm = bnn_normalize_x_data(x_train_subset, x_mean, x_std)
            x_grid_norm = bnn_normalize_x_data(x_grid, x_mean, x_std)
            
            # Train BNN with MCMC
            mcmc = train_bnn(
                x_train_subset_norm, y_train_subset,
                hidden_width=hidden_width, weight_scale=weight_scale,
                warmup=warmup, samples=samples, chains=chains, seed=seed
            )
            
            # Make predictions
            mu_pred, ale_var, epi_var, tot_var = bnn_predict(
                mcmc, x_grid_norm,
                hidden_width=hidden_width, weight_scale=weight_scale
            )
            
            # Store uncertainties
            uncertainties_by_pct[pct]['ale'].append(ale_var)
            uncertainties_by_pct[pct]['epi'].append(epi_var)
            uncertainties_by_pct[pct]['tot'].append(tot_var)
            
            # Plot
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
            function_names[func_type], noise_type, func_type, 'BNN'
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
    nsamples: int = 1000
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
        
        # Train and evaluate for each percentage
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
            
            # BAMLSS fits directly - no normalization or training loop needed
            # Make predictions (fitting happens inside bamlss_predict)
            mu_pred, ale_var, epi_var, tot_var = bamlss_predict(
                x_train_subset, y_train_subset, x_grid,
                n_iter=n_iter, burnin=burnin, thin=thin, nsamples=nsamples
            )
            
            # Store uncertainties
            uncertainties_by_pct[pct]['ale'].append(ale_var)
            uncertainties_by_pct[pct]['epi'].append(epi_var)
            uncertainties_by_pct[pct]['tot'].append(tot_var)
            
            # Plot
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
            function_names[func_type], noise_type, func_type, 'BAMLSS'
        )

