"""
Omitted Variable Bias (OVB) experiments for classification.

This module provides experiment functions for studying the effects of omitting
a confounding variable Z on uncertainty estimation in binary classification.

DGP:
    Z ~ N(0, 1)
    X = rho * Z + sqrt(1 - rho^2) * nu, nu ~ N(0, 1)
    latent = f(X_scaled) + beta2 * Z
    P(Y=1) = sigmoid(latent)
    Y ~ Bernoulli(P(Y=1))
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils.classification_models import (
    train_mc_dropout_it,
    train_mc_dropout_gl,
    mc_dropout_predict_it,
    mc_dropout_predict_gl,
    train_deep_ensemble_it,
    train_deep_ensemble_gl,
    ensemble_predict_it,
    ensemble_predict_gl,
    train_bnn_it,
    train_bnn_gl,
    bnn_predict_it,
    bnn_predict_gl,
    sampling_softmax_np,
)
from utils.classification_experiments import it_uncertainty, gl_uncertainty, entropy


# ============================================================================
# Helper Functions
# ============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def _compute_nll_binary(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-8) -> float:
    """Compute negative log-likelihood for binary classification."""
    # probs is [N, 2] for binary, or [N] for P(Y=1)
    if probs.ndim == 2:
        p_y1 = probs[:, 1]
    else:
        p_y1 = probs
    p_y1 = np.clip(p_y1, eps, 1 - eps)
    nll = -np.mean(y_true * np.log(p_y1) + (1 - y_true) * np.log(1 - p_y1))
    return float(nll)


def _compute_accuracy_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute accuracy for binary classification."""
    if probs.ndim == 2:
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = (probs > 0.5).astype(int)
    return float(np.mean(y_pred == y_true))


# ============================================================================
# Data Generation
# ============================================================================

def generate_ovb_classification_data(
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: str = 'linear',
    rho: float = 0.7,
    beta2: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate binary classification data with OVB potential.
    
    DGP:
        Z ~ N(0, 1)
        X = rho * Z + sqrt(1 - rho^2) * nu, nu ~ N(0, 1)
        latent = f(X_scaled) + beta2 * Z
        P(Y=1) = sigmoid(latent)
        Y ~ Bernoulli(P(Y=1))
    
    Parameters:
    -----------
    n_train : int
        Number of training samples
    train_range : tuple
        (min, max) range for X values
    grid_points : int
        Number of grid points per dimension for evaluation
    func_type : str
        'linear' or 'sin'
    rho : float
        Correlation between X and Z (0 to 1)
    beta2 : float
        Effect of omitted variable Z on Y
    seed : int
        Random seed
        
    Returns:
    --------
    X, Z, Y, true_probs, x_grid, z_grid : tuple
        - X: (n_train, 1) observed feature
        - Z: (n_train,) omitted variable
        - Y: (n_train,) binary labels {0, 1}
        - true_probs: (n_train,) P(Y=1|X,Z) ground truth
        - x_grid: (grid_points,) x values for evaluation grid
        - z_grid: (grid_points,) z values for evaluation grid
    """
    rng = np.random.default_rng(seed)
    
    # Generate latent Z ~ N(0, 1)
    Z = rng.standard_normal(n_train)
    
    # Generate X correlated with Z
    nu = rng.standard_normal(n_train)
    X = rho * Z + np.sqrt(1 - rho**2) * nu
    
    # Scale X to desired range
    X_scaled = X * (train_range[1] - train_range[0]) / 4 + (train_range[0] + train_range[1]) / 2
    
    # Define the function f(x) for latent logit
    if func_type == 'linear':
        f_x = lambda x: 0.5 * x  # Simple linear decision boundary
    elif func_type == 'sin':
        f_x = lambda x: 0.5 * x + np.sin(x)  # Nonlinear decision boundary
    else:
        raise ValueError("func_type must be 'linear' or 'sin'")
    
    # Compute latent logit: f(X) + beta2 * Z
    latent = f_x(X_scaled) + beta2 * Z
    
    # Compute true probabilities
    true_probs = _sigmoid(latent)
    
    # Sample binary labels
    Y = rng.binomial(1, true_probs).astype(np.int64)
    
    # Create evaluation grids
    x_grid = np.linspace(train_range[0], train_range[1], grid_points).astype(np.float32)
    z_grid = np.linspace(-3.0, 3.0, grid_points).astype(np.float32)
    
    return (
        X_scaled.astype(np.float32).reshape(-1, 1),
        Z.astype(np.float32),
        Y,
        true_probs.astype(np.float32),
        x_grid,
        z_grid
    )


def generate_ovb_2d_grid(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    func_type: str = 'linear',
    beta2: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D evaluation grid with true probabilities.
    
    Returns:
    --------
    X_grid_2d, Z_grid_2d, true_probs_2d : tuple
        - X_grid_2d: (grid_points^2, 1) flattened X values
        - Z_grid_2d: (grid_points^2,) flattened Z values
        - true_probs_2d: (grid_points^2,) P(Y=1|X,Z) for each grid point
    """
    xx, zz = np.meshgrid(x_grid, z_grid)
    X_flat = xx.ravel().astype(np.float32)
    Z_flat = zz.ravel().astype(np.float32)
    
    # Compute true probabilities on grid
    if func_type == 'linear':
        f_x = lambda x: 0.5 * x
    elif func_type == 'sin':
        f_x = lambda x: 0.5 * x + np.sin(x)
    else:
        raise ValueError("func_type must be 'linear' or 'sin'")
    
    latent = f_x(X_flat) + beta2 * Z_flat
    true_probs = _sigmoid(latent).astype(np.float32)
    
    return X_flat.reshape(-1, 1), Z_flat, true_probs


# ============================================================================
# Save Utilities
# ============================================================================

def save_ovb_classification_outputs(
    X: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    true_probs: np.ndarray,
    probs_omitted: np.ndarray,
    probs_full: np.ndarray,
    au_omitted: np.ndarray,
    eu_omitted: np.ndarray,
    au_full: np.ndarray,
    eu_full: np.ndarray,
    rho: float,
    beta2: float,
    func_type: str,
    results_dir: Path,
    param_name: str = 'rho',
    model_type: str = 'mc_dropout',
    decomposition: str = 'it'
):
    """
    Save all OVB classification outputs for later reuse.
    """
    date = datetime.now().strftime('%Y%m%d')
    
    if param_name == 'rho':
        filename = f"ovb_cls_outputs_rho{rho:.2f}_beta2{beta2:.2f}_{date}.npz"
    else:
        filename = f"ovb_cls_outputs_beta2{beta2:.2f}_rho{rho:.2f}_{date}.npz"
    
    filepath = results_dir / filename
    results_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        filepath,
        X=X, Z=Z, Y=Y, true_probs=true_probs,
        probs_omitted=probs_omitted, probs_full=probs_full,
        au_omitted=au_omitted, eu_omitted=eu_omitted,
        au_full=au_full, eu_full=eu_full,
        rho=np.array([rho]), beta2=np.array([beta2]),
        func_type=np.array([func_type]),
        model_type=np.array([model_type]),
        decomposition=np.array([decomposition])
    )
    print(f"  Saved outputs to: {filepath}")


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_ovb_classification_heatmap(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    values_omitted: np.ndarray,
    values_full: np.ndarray,
    title: str = "Uncertainty Comparison",
    value_name: str = "AU",
    rho: float = 0.0,
    beta2: float = 1.0,
    save_path: Path = None,
    show: bool = True
):
    """
    Plot 2D heatmaps comparing omitted vs full model uncertainties.
    """
    n_x = len(x_grid)
    n_z = len(z_grid)
    
    # Reshape to 2D
    values_omitted_2d = values_omitted.reshape(n_z, n_x)
    values_full_2d = values_full.reshape(n_z, n_x)
    
    # Common colorbar range
    vmin = min(values_omitted.min(), values_full.min())
    vmax = max(values_omitted.max(), values_full.max())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Omitted model
    im1 = axes[0].imshow(
        values_omitted_2d, extent=[x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()],
        origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax
    )
    axes[0].set_xlabel('X (observed)')
    axes[0].set_ylabel('Z (omitted)')
    axes[0].set_title(f'Omitted Model - {value_name}')
    plt.colorbar(im1, ax=axes[0])
    
    # Full model
    im2 = axes[1].imshow(
        values_full_2d, extent=[x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()],
        origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax
    )
    axes[1].set_xlabel('X (observed)')
    axes[1].set_ylabel('Z (omitted)')
    axes[1].set_title(f'Full Model - {value_name}')
    plt.colorbar(im2, ax=axes[1])
    
    fig.suptitle(f'{title}\nrho={rho:.2f}, beta2={beta2:.2f}', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ovb_classification_marginals(
    au_omitted: np.ndarray,
    eu_omitted: np.ndarray,
    au_full: np.ndarray,
    eu_full: np.ndarray,
    rho: float = 0.0,
    beta2: float = 1.0,
    save_path: Path = None,
    show: bool = True
):
    """
    Plot marginal distributions of AU/EU for omitted vs full models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # AU histogram
    axes[0].hist(au_omitted.flatten(), bins=50, alpha=0.6, label='Omitted', density=True)
    axes[0].hist(au_full.flatten(), bins=50, alpha=0.6, label='Full', density=True)
    axes[0].set_xlabel('Aleatoric Uncertainty')
    axes[0].set_ylabel('Density')
    axes[0].set_title('AU Distribution')
    axes[0].legend()
    
    # EU histogram
    axes[1].hist(eu_omitted.flatten(), bins=50, alpha=0.6, label='Omitted', density=True)
    axes[1].hist(eu_full.flatten(), bins=50, alpha=0.6, label='Full', density=True)
    axes[1].set_xlabel('Epistemic Uncertainty')
    axes[1].set_ylabel('Density')
    axes[1].set_title('EU Distribution')
    axes[1].legend()
    
    fig.suptitle(f'Uncertainty Marginals (rho={rho:.2f}, beta2={beta2:.2f})', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ovb_classification_z_slices(
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    au_omitted: np.ndarray,
    eu_omitted: np.ndarray,
    au_full: np.ndarray,
    eu_full: np.ndarray,
    z_percentiles: List[float] = [10, 50, 90],
    rho: float = 0.0,
    beta2: float = 1.0,
    save_path: Path = None,
    show: bool = True
):
    """
    Plot AU/EU as function of X at specific Z values (slices).
    """
    n_x = len(x_grid)
    n_z = len(z_grid)
    
    # Reshape to 2D: [n_z, n_x]
    au_omitted_2d = au_omitted.reshape(n_z, n_x)
    eu_omitted_2d = eu_omitted.reshape(n_z, n_x)
    au_full_2d = au_full.reshape(n_z, n_x)
    eu_full_2d = eu_full.reshape(n_z, n_x)
    
    # Get Z indices for percentiles
    z_indices = [int(p / 100 * (n_z - 1)) for p in z_percentiles]
    z_values = [z_grid[i] for i in z_indices]
    
    fig, axes = plt.subplots(2, len(z_percentiles), figsize=(5*len(z_percentiles), 8))
    if len(z_percentiles) == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.tab10.colors
    
    for col, (z_idx, z_val, z_pct) in enumerate(zip(z_indices, z_values, z_percentiles)):
        # AU slice
        axes[0, col].plot(x_grid, au_omitted_2d[z_idx, :], label='Omitted', color=colors[0], linewidth=2)
        axes[0, col].plot(x_grid, au_full_2d[z_idx, :], label='Full', color=colors[1], linewidth=2)
        axes[0, col].set_xlabel('X')
        axes[0, col].set_ylabel('AU')
        axes[0, col].set_title(f'AU at Z={z_val:.2f} ({z_pct}th pct)')
        axes[0, col].legend()
        
        # EU slice
        axes[1, col].plot(x_grid, eu_omitted_2d[z_idx, :], label='Omitted', color=colors[0], linewidth=2)
        axes[1, col].plot(x_grid, eu_full_2d[z_idx, :], label='Full', color=colors[1], linewidth=2)
        axes[1, col].set_xlabel('X')
        axes[1, col].set_ylabel('EU')
        axes[1, col].set_title(f'EU at Z={z_val:.2f} ({z_pct}th pct)')
        axes[1, col].legend()
    
    fig.suptitle(f'Z-Slice Comparison (rho={rho:.2f}, beta2={beta2:.2f})', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ovb_classification_summary(
    results_df: pd.DataFrame,
    param_name: str = 'rho',
    save_path: Path = None,
    show: bool = True
):
    """
    Plot summary metrics across rho or beta2 values.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    param_vals = results_df[param_name].values
    
    # Row 1: Accuracy and NLL
    axes[0, 0].plot(param_vals, results_df['acc_omitted'], 'o-', label='Omitted', markersize=8)
    axes[0, 0].plot(param_vals, results_df['acc_full'], 's-', label='Full', markersize=8)
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Classification Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(param_vals, results_df['nll_omitted'], 'o-', label='Omitted', markersize=8)
    axes[0, 1].plot(param_vals, results_df['nll_full'], 's-', label='Full', markersize=8)
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('NLL')
    axes[0, 1].set_title('Negative Log-Likelihood')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy difference
    axes[0, 2].bar(param_vals, results_df['acc_full'] - results_df['acc_omitted'], width=0.1)
    axes[0, 2].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 2].set_xlabel(param_name)
    axes[0, 2].set_ylabel('Acc Diff (Full - Omitted)')
    axes[0, 2].set_title('Accuracy Improvement from Full Model')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: AU and EU
    axes[1, 0].plot(param_vals, results_df['au_mean_omitted'], 'o-', label='Omitted', markersize=8)
    axes[1, 0].plot(param_vals, results_df['au_mean_full'], 's-', label='Full', markersize=8)
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Mean AU')
    axes[1, 0].set_title('Aleatoric Uncertainty')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(param_vals, results_df['eu_mean_omitted'], 'o-', label='Omitted', markersize=8)
    axes[1, 1].plot(param_vals, results_df['eu_mean_full'], 's-', label='Full', markersize=8)
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Mean EU')
    axes[1, 1].set_title('Epistemic Uncertainty')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # AU/EU ratio
    axes[1, 2].plot(param_vals, results_df['au_mean_omitted'] / (results_df['eu_mean_omitted'] + 1e-8), 
                    'o-', label='Omitted', markersize=8)
    axes[1, 2].plot(param_vals, results_df['au_mean_full'] / (results_df['eu_mean_full'] + 1e-8), 
                    's-', label='Full', markersize=8)
    axes[1, 2].set_xlabel(param_name)
    axes[1, 2].set_ylabel('AU/EU Ratio')
    axes[1, 2].set_title('AU/EU Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved summary plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# MC Dropout IT Experiments
# ============================================================================

def _run_single_ovb_classification_mc_dropout_it(
    rho: float,
    beta2: float,
    n_train: int,
    train_range: tuple,
    grid_points: int,
    func_type: str,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    dropout_p: float,
    mc_samples: int,
    results_dir: Path = None,
    save_plots: bool = True,
    param_name: str = 'rho'
) -> Dict[str, Any]:
    """Run a single OVB classification experiment with MC Dropout IT."""
    
    print(f"\n  Running MC Dropout IT: rho={rho:.2f}, beta2={beta2:.2f}, func={func_type}")
    
    # Generate data
    X, Z, Y, true_probs, x_grid, z_grid = generate_ovb_classification_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed
    )
    
    # Generate 2D evaluation grid
    X_grid_2d, Z_grid_2d, true_probs_2d = generate_ovb_2d_grid(
        x_grid, z_grid, func_type=func_type, beta2=beta2
    )
    
    # Prepare full input [X, Z]
    XZ_train = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
    XZ_grid = np.hstack([X_grid_2d, Z_grid_2d.reshape(-1, 1)]).astype(np.float32)
    
    # Train omitted model (X only)
    print("    Training omitted model (X only)...")
    model_omitted = train_mc_dropout_it(
        X, Y,
        input_dim=1,
        num_classes=2,
        p=dropout_p,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )
    
    # Train full model (X and Z)
    print("    Training full model (X, Z)...")
    model_full = train_mc_dropout_it(
        XZ_train, Y,
        input_dim=2,
        num_classes=2,
        p=dropout_p,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )
    
    # Predict on 2D grid
    print("    Predicting on evaluation grid...")
    probs_members_omitted = mc_dropout_predict_it(model_omitted, X_grid_2d, mc_samples=mc_samples)
    probs_members_full = mc_dropout_predict_it(model_full, XZ_grid, mc_samples=mc_samples)
    
    # Compute IT uncertainties
    unc_omitted = it_uncertainty(probs_members_omitted)
    unc_full = it_uncertainty(probs_members_full)
    
    # Get predictive probabilities
    probs_omitted = unc_omitted['p_bar']
    probs_full = unc_full['p_bar']
    
    # Compute metrics on training data
    # Predict on training data
    probs_members_train_omitted = mc_dropout_predict_it(model_omitted, X, mc_samples=mc_samples)
    probs_members_train_full = mc_dropout_predict_it(model_full, XZ_train, mc_samples=mc_samples)
    probs_train_omitted = probs_members_train_omitted.mean(axis=0)
    probs_train_full = probs_members_train_full.mean(axis=0)
    
    acc_omitted = _compute_accuracy_binary(Y, probs_train_omitted)
    acc_full = _compute_accuracy_binary(Y, probs_train_full)
    nll_omitted = _compute_nll_binary(Y, probs_train_omitted)
    nll_full = _compute_nll_binary(Y, probs_train_full)
    
    # Summary statistics
    result = {
        'rho': rho,
        'beta2': beta2,
        'func_type': func_type,
        'acc_omitted': acc_omitted,
        'acc_full': acc_full,
        'nll_omitted': nll_omitted,
        'nll_full': nll_full,
        'au_mean_omitted': float(unc_omitted['AU'].mean()),
        'au_mean_full': float(unc_full['AU'].mean()),
        'eu_mean_omitted': float(unc_omitted['EU'].mean()),
        'eu_mean_full': float(unc_full['EU'].mean()),
        'tu_mean_omitted': float(unc_omitted['TU'].mean()),
        'tu_mean_full': float(unc_full['TU'].mean()),
        # Store arrays for visualization
        'au_omitted': unc_omitted['AU'],
        'eu_omitted': unc_omitted['EU'],
        'au_full': unc_full['AU'],
        'eu_full': unc_full['EU'],
        'probs_omitted': probs_omitted,
        'probs_full': probs_full,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'X': X,
        'Z': Z,
        'Y': Y,
        'true_probs': true_probs,
    }
    
    print(f"    Acc: omitted={acc_omitted:.4f}, full={acc_full:.4f}")
    print(f"    NLL: omitted={nll_omitted:.4f}, full={nll_full:.4f}")
    print(f"    AU:  omitted={unc_omitted['AU'].mean():.4f}, full={unc_full['AU'].mean():.4f}")
    print(f"    EU:  omitted={unc_omitted['EU'].mean():.4f}, full={unc_full['EU'].mean():.4f}")
    
    # Save outputs and plots
    if results_dir and save_plots:
        # Create subdirectory
        save_dir = results_dir / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model outputs
        save_ovb_classification_outputs(
            X, Z, Y, true_probs,
            probs_omitted, probs_full,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho, beta2, func_type,
            save_dir, param_name,
            model_type='mc_dropout_it',
            decomposition='it'
        )
        
        # Plot heatmaps
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['AU'], unc_full['AU'],
            title='Aleatoric Uncertainty (MC Dropout IT)',
            value_name='AU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_au_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['EU'], unc_full['EU'],
            title='Epistemic Uncertainty (MC Dropout IT)',
            value_name='EU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_eu_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        # Plot marginals
        plot_ovb_classification_marginals(
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'marginals_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        # Plot Z-slices
        plot_ovb_classification_z_slices(
            x_grid, z_grid,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'z_slices_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
    
    return result


def run_mc_dropout_it_ovb_rho_experiment(
    rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    dropout_p: float = 0.25,
    mc_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run OVB classification experiment varying rho with MC Dropout IT.
    
    Parameters can be single values or lists (for multi-config iteration).
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_mc_dropout_it_ovb_rho_experiment(
                rho_values=rho_values, beta2=beta2,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, dropout_p=dropout_p,
                mc_samples=mc_samples, save_plots=save_plots,
                results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    # Single config execution
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"MC Dropout IT - Varying rho (beta2={beta2}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'mc_dropout_it'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for rho in rho_values:
        result = _run_single_ovb_classification_mc_dropout_it(
            rho=rho, beta2=beta2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, dropout_p=dropout_p,
            mc_samples=mc_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='rho'
        )
        all_results.append(result)
    
    # Create summary DataFrame
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full', 'tu_mean_omitted', 'tu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    # Save summary
    if save_dir:
        summary_path = save_dir / func_type / f'summary_rho_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        # Plot summary
        plot_ovb_classification_summary(
            results_df, param_name='rho',
            save_path=save_dir / func_type / 'summary_rho_plot.png',
            show=False
        )
    
    return results_df, {r['rho']: r for r in all_results}


def run_mc_dropout_it_ovb_beta2_experiment(
    beta2_values: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    dropout_p: float = 0.25,
    mc_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run OVB classification experiment varying beta2 with MC Dropout IT.
    """
    # Handle list inputs
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_mc_dropout_it_ovb_beta2_experiment(
                beta2_values=beta2_values, rho=rho,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, dropout_p=dropout_p,
                mc_samples=mc_samples, save_plots=save_plots,
                results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    # Single config execution
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"MC Dropout IT - Varying beta2 (rho={rho}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'mc_dropout_it'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for b2 in beta2_values:
        result = _run_single_ovb_classification_mc_dropout_it(
            rho=rho, beta2=b2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, dropout_p=dropout_p,
            mc_samples=mc_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='beta2'
        )
        all_results.append(result)
    
    # Create summary DataFrame
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full', 'tu_mean_omitted', 'tu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    # Save summary
    if save_dir:
        summary_path = save_dir / func_type / f'summary_beta2_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        # Plot summary
        plot_ovb_classification_summary(
            results_df, param_name='beta2',
            save_path=save_dir / func_type / 'summary_beta2_plot.png',
            show=False
        )
    
    return results_df, {r['beta2']: r for r in all_results}


# ============================================================================
# MC Dropout GL Experiments
# ============================================================================

def _run_single_ovb_classification_mc_dropout_gl(
    rho: float,
    beta2: float,
    n_train: int,
    train_range: tuple,
    grid_points: int,
    func_type: str,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    dropout_p: float,
    mc_samples: int,
    gl_samples: int,
    results_dir: Path = None,
    save_plots: bool = True,
    param_name: str = 'rho'
) -> Dict[str, Any]:
    """Run a single OVB classification experiment with MC Dropout GL."""
    
    print(f"\n  Running MC Dropout GL: rho={rho:.2f}, beta2={beta2:.2f}, func={func_type}")
    
    # Generate data
    X, Z, Y, true_probs, x_grid, z_grid = generate_ovb_classification_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed
    )
    
    # Generate 2D evaluation grid
    X_grid_2d, Z_grid_2d, true_probs_2d = generate_ovb_2d_grid(
        x_grid, z_grid, func_type=func_type, beta2=beta2
    )
    
    # Prepare full input [X, Z]
    XZ_train = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
    XZ_grid = np.hstack([X_grid_2d, Z_grid_2d.reshape(-1, 1)]).astype(np.float32)
    
    # Train omitted model (X only)
    print("    Training omitted model (X only)...")
    model_omitted = train_mc_dropout_gl(
        X, Y,
        input_dim=1,
        num_classes=2,
        p=dropout_p,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_samples=gl_samples
    )
    
    # Train full model (X and Z)
    print("    Training full model (X, Z)...")
    model_full = train_mc_dropout_gl(
        XZ_train, Y,
        input_dim=2,
        num_classes=2,
        p=dropout_p,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_samples=gl_samples
    )
    
    # Predict on 2D grid
    print("    Predicting on evaluation grid...")
    mu_members_omitted, sigma2_members_omitted = mc_dropout_predict_gl(
        model_omitted, X_grid_2d, mc_samples=mc_samples
    )
    mu_members_full, sigma2_members_full = mc_dropout_predict_gl(
        model_full, XZ_grid, mc_samples=mc_samples
    )
    
    # Compute GL uncertainties
    unc_omitted = gl_uncertainty(mu_members_omitted, sigma2_members_omitted, n_samples=gl_samples)
    unc_full = gl_uncertainty(mu_members_full, sigma2_members_full, n_samples=gl_samples)
    
    # Get predictive probabilities
    probs_omitted = unc_omitted['p_ale']
    probs_full = unc_full['p_ale']
    
    # Compute metrics on training data
    mu_train_omitted, sigma2_train_omitted = mc_dropout_predict_gl(model_omitted, X, mc_samples=mc_samples)
    mu_train_full, sigma2_train_full = mc_dropout_predict_gl(model_full, XZ_train, mc_samples=mc_samples)
    
    unc_train_omitted = gl_uncertainty(mu_train_omitted, sigma2_train_omitted, n_samples=gl_samples)
    unc_train_full = gl_uncertainty(mu_train_full, sigma2_train_full, n_samples=gl_samples)
    
    probs_train_omitted = unc_train_omitted['p_ale']
    probs_train_full = unc_train_full['p_ale']
    
    acc_omitted = _compute_accuracy_binary(Y, probs_train_omitted)
    acc_full = _compute_accuracy_binary(Y, probs_train_full)
    nll_omitted = _compute_nll_binary(Y, probs_train_omitted)
    nll_full = _compute_nll_binary(Y, probs_train_full)
    
    # Summary
    result = {
        'rho': rho,
        'beta2': beta2,
        'func_type': func_type,
        'acc_omitted': acc_omitted,
        'acc_full': acc_full,
        'nll_omitted': nll_omitted,
        'nll_full': nll_full,
        'au_mean_omitted': float(unc_omitted['AU'].mean()),
        'au_mean_full': float(unc_full['AU'].mean()),
        'eu_mean_omitted': float(unc_omitted['EU'].mean()),
        'eu_mean_full': float(unc_full['EU'].mean()),
        'au_omitted': unc_omitted['AU'],
        'eu_omitted': unc_omitted['EU'],
        'au_full': unc_full['AU'],
        'eu_full': unc_full['EU'],
        'probs_omitted': probs_omitted,
        'probs_full': probs_full,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'X': X,
        'Z': Z,
        'Y': Y,
        'true_probs': true_probs,
    }
    
    print(f"    Acc: omitted={acc_omitted:.4f}, full={acc_full:.4f}")
    print(f"    NLL: omitted={nll_omitted:.4f}, full={nll_full:.4f}")
    print(f"    AU:  omitted={unc_omitted['AU'].mean():.4f}, full={unc_full['AU'].mean():.4f}")
    print(f"    EU:  omitted={unc_omitted['EU'].mean():.4f}, full={unc_full['EU'].mean():.4f}")
    
    # Save outputs and plots
    if results_dir and save_plots:
        save_dir = results_dir / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_ovb_classification_outputs(
            X, Z, Y, true_probs,
            probs_omitted, probs_full,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho, beta2, func_type,
            save_dir, param_name,
            model_type='mc_dropout_gl',
            decomposition='gl'
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['AU'], unc_full['AU'],
            title='Aleatoric Uncertainty (MC Dropout GL)',
            value_name='AU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_au_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['EU'], unc_full['EU'],
            title='Epistemic Uncertainty (MC Dropout GL)',
            value_name='EU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_eu_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_marginals(
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'marginals_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_z_slices(
            x_grid, z_grid,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'z_slices_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
    
    return result


def run_mc_dropout_gl_ovb_rho_experiment(
    rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    dropout_p: float = 0.25,
    mc_samples: int = 100,
    gl_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying rho with MC Dropout GL."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_mc_dropout_gl_ovb_rho_experiment(
                rho_values=rho_values, beta2=beta2,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, dropout_p=dropout_p,
                mc_samples=mc_samples, gl_samples=gl_samples,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"MC Dropout GL - Varying rho (beta2={beta2}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'mc_dropout_gl'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for rho in rho_values:
        result = _run_single_ovb_classification_mc_dropout_gl(
            rho=rho, beta2=beta2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, dropout_p=dropout_p,
            mc_samples=mc_samples, gl_samples=gl_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='rho'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_rho_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='rho',
            save_path=save_dir / func_type / 'summary_rho_plot.png',
            show=False
        )
    
    return results_df, {r['rho']: r for r in all_results}


def run_mc_dropout_gl_ovb_beta2_experiment(
    beta2_values: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    dropout_p: float = 0.25,
    mc_samples: int = 100,
    gl_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying beta2 with MC Dropout GL."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_mc_dropout_gl_ovb_beta2_experiment(
                beta2_values=beta2_values, rho=rho,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, dropout_p=dropout_p,
                mc_samples=mc_samples, gl_samples=gl_samples,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"MC Dropout GL - Varying beta2 (rho={rho}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'mc_dropout_gl'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for b2 in beta2_values:
        result = _run_single_ovb_classification_mc_dropout_gl(
            rho=rho, beta2=b2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, dropout_p=dropout_p,
            mc_samples=mc_samples, gl_samples=gl_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='beta2'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_beta2_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='beta2',
            save_path=save_dir / func_type / 'summary_beta2_plot.png',
            show=False
        )
    
    return results_df, {r['beta2']: r for r in all_results}


# ============================================================================
# Deep Ensemble IT Experiments
# ============================================================================

def _run_single_ovb_classification_deep_ensemble_it(
    rho: float,
    beta2: float,
    n_train: int,
    train_range: tuple,
    grid_points: int,
    func_type: str,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    K: int,
    results_dir: Path = None,
    save_plots: bool = True,
    param_name: str = 'rho'
) -> Dict[str, Any]:
    """Run a single OVB classification experiment with Deep Ensemble IT."""
    
    print(f"\n  Running Deep Ensemble IT: rho={rho:.2f}, beta2={beta2:.2f}, func={func_type}")
    
    # Generate data
    X, Z, Y, true_probs, x_grid, z_grid = generate_ovb_classification_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed
    )
    
    # Generate 2D evaluation grid
    X_grid_2d, Z_grid_2d, true_probs_2d = generate_ovb_2d_grid(
        x_grid, z_grid, func_type=func_type, beta2=beta2
    )
    
    # Prepare full input [X, Z]
    XZ_train = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
    XZ_grid = np.hstack([X_grid_2d, Z_grid_2d.reshape(-1, 1)]).astype(np.float32)
    
    # Train omitted model (X only)
    print("    Training omitted ensemble (X only)...")
    ensemble_omitted = train_deep_ensemble_it(
        X, Y,
        input_dim=1,
        num_classes=2,
        K=K,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )
    
    # Train full model (X and Z)
    print("    Training full ensemble (X, Z)...")
    ensemble_full = train_deep_ensemble_it(
        XZ_train, Y,
        input_dim=2,
        num_classes=2,
        K=K,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size
    )
    
    # Predict on 2D grid
    print("    Predicting on evaluation grid...")
    probs_members_omitted = ensemble_predict_it(ensemble_omitted, X_grid_2d)
    probs_members_full = ensemble_predict_it(ensemble_full, XZ_grid)
    
    # Compute IT uncertainties
    unc_omitted = it_uncertainty(probs_members_omitted)
    unc_full = it_uncertainty(probs_members_full)
    
    probs_omitted = unc_omitted['p_bar']
    probs_full = unc_full['p_bar']
    
    # Compute metrics on training data
    probs_members_train_omitted = ensemble_predict_it(ensemble_omitted, X)
    probs_members_train_full = ensemble_predict_it(ensemble_full, XZ_train)
    probs_train_omitted = probs_members_train_omitted.mean(axis=0)
    probs_train_full = probs_members_train_full.mean(axis=0)
    
    acc_omitted = _compute_accuracy_binary(Y, probs_train_omitted)
    acc_full = _compute_accuracy_binary(Y, probs_train_full)
    nll_omitted = _compute_nll_binary(Y, probs_train_omitted)
    nll_full = _compute_nll_binary(Y, probs_train_full)
    
    result = {
        'rho': rho,
        'beta2': beta2,
        'func_type': func_type,
        'acc_omitted': acc_omitted,
        'acc_full': acc_full,
        'nll_omitted': nll_omitted,
        'nll_full': nll_full,
        'au_mean_omitted': float(unc_omitted['AU'].mean()),
        'au_mean_full': float(unc_full['AU'].mean()),
        'eu_mean_omitted': float(unc_omitted['EU'].mean()),
        'eu_mean_full': float(unc_full['EU'].mean()),
        'tu_mean_omitted': float(unc_omitted['TU'].mean()),
        'tu_mean_full': float(unc_full['TU'].mean()),
        'au_omitted': unc_omitted['AU'],
        'eu_omitted': unc_omitted['EU'],
        'au_full': unc_full['AU'],
        'eu_full': unc_full['EU'],
        'probs_omitted': probs_omitted,
        'probs_full': probs_full,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'X': X,
        'Z': Z,
        'Y': Y,
        'true_probs': true_probs,
    }
    
    print(f"    Acc: omitted={acc_omitted:.4f}, full={acc_full:.4f}")
    print(f"    NLL: omitted={nll_omitted:.4f}, full={nll_full:.4f}")
    print(f"    AU:  omitted={unc_omitted['AU'].mean():.4f}, full={unc_full['AU'].mean():.4f}")
    print(f"    EU:  omitted={unc_omitted['EU'].mean():.4f}, full={unc_full['EU'].mean():.4f}")
    
    if results_dir and save_plots:
        save_dir = results_dir / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_ovb_classification_outputs(
            X, Z, Y, true_probs,
            probs_omitted, probs_full,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho, beta2, func_type,
            save_dir, param_name,
            model_type='deep_ensemble_it',
            decomposition='it'
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['AU'], unc_full['AU'],
            title='Aleatoric Uncertainty (Deep Ensemble IT)',
            value_name='AU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_au_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['EU'], unc_full['EU'],
            title='Epistemic Uncertainty (Deep Ensemble IT)',
            value_name='EU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_eu_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_marginals(
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'marginals_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_z_slices(
            x_grid, z_grid,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'z_slices_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
    
    return result


def run_deep_ensemble_it_ovb_rho_experiment(
    rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    K: int = 10,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying rho with Deep Ensemble IT."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_deep_ensemble_it_ovb_rho_experiment(
                rho_values=rho_values, beta2=beta2,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, K=K,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"Deep Ensemble IT - Varying rho (beta2={beta2}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'deep_ensemble_it'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for rho in rho_values:
        result = _run_single_ovb_classification_deep_ensemble_it(
            rho=rho, beta2=beta2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, K=K,
            results_dir=save_dir, save_plots=save_plots,
            param_name='rho'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full', 'tu_mean_omitted', 'tu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_rho_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='rho',
            save_path=save_dir / func_type / 'summary_rho_plot.png',
            show=False
        )
    
    return results_df, {r['rho']: r for r in all_results}


def run_deep_ensemble_it_ovb_beta2_experiment(
    beta2_values: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    K: int = 10,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying beta2 with Deep Ensemble IT."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_deep_ensemble_it_ovb_beta2_experiment(
                beta2_values=beta2_values, rho=rho,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, K=K,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"Deep Ensemble IT - Varying beta2 (rho={rho}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'deep_ensemble_it'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for b2 in beta2_values:
        result = _run_single_ovb_classification_deep_ensemble_it(
            rho=rho, beta2=b2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, K=K,
            results_dir=save_dir, save_plots=save_plots,
            param_name='beta2'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full', 'tu_mean_omitted', 'tu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_beta2_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='beta2',
            save_path=save_dir / func_type / 'summary_beta2_plot.png',
            show=False
        )
    
    return results_df, {r['beta2']: r for r in all_results}


# ============================================================================
# Deep Ensemble GL Experiments
# ============================================================================

def _run_single_ovb_classification_deep_ensemble_gl(
    rho: float,
    beta2: float,
    n_train: int,
    train_range: tuple,
    grid_points: int,
    func_type: str,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    K: int,
    gl_samples: int,
    results_dir: Path = None,
    save_plots: bool = True,
    param_name: str = 'rho'
) -> Dict[str, Any]:
    """Run a single OVB classification experiment with Deep Ensemble GL."""
    
    print(f"\n  Running Deep Ensemble GL: rho={rho:.2f}, beta2={beta2:.2f}, func={func_type}")
    
    # Generate data
    X, Z, Y, true_probs, x_grid, z_grid = generate_ovb_classification_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed
    )
    
    # Generate 2D evaluation grid
    X_grid_2d, Z_grid_2d, true_probs_2d = generate_ovb_2d_grid(
        x_grid, z_grid, func_type=func_type, beta2=beta2
    )
    
    # Prepare full input [X, Z]
    XZ_train = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
    XZ_grid = np.hstack([X_grid_2d, Z_grid_2d.reshape(-1, 1)]).astype(np.float32)
    
    # Train omitted model (X only)
    print("    Training omitted ensemble (X only)...")
    ensemble_omitted = train_deep_ensemble_gl(
        X, Y,
        input_dim=1,
        num_classes=2,
        K=K,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_samples=gl_samples
    )
    
    # Train full model (X and Z)
    print("    Training full ensemble (X, Z)...")
    ensemble_full = train_deep_ensemble_gl(
        XZ_train, Y,
        input_dim=2,
        num_classes=2,
        K=K,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_samples=gl_samples
    )
    
    # Predict on 2D grid
    print("    Predicting on evaluation grid...")
    mu_members_omitted, sigma2_members_omitted = ensemble_predict_gl(ensemble_omitted, X_grid_2d)
    mu_members_full, sigma2_members_full = ensemble_predict_gl(ensemble_full, XZ_grid)
    
    # Compute GL uncertainties
    unc_omitted = gl_uncertainty(mu_members_omitted, sigma2_members_omitted, n_samples=gl_samples)
    unc_full = gl_uncertainty(mu_members_full, sigma2_members_full, n_samples=gl_samples)
    
    probs_omitted = unc_omitted['p_ale']
    probs_full = unc_full['p_ale']
    
    # Compute metrics on training data
    mu_train_omitted, sigma2_train_omitted = ensemble_predict_gl(ensemble_omitted, X)
    mu_train_full, sigma2_train_full = ensemble_predict_gl(ensemble_full, XZ_train)
    
    unc_train_omitted = gl_uncertainty(mu_train_omitted, sigma2_train_omitted, n_samples=gl_samples)
    unc_train_full = gl_uncertainty(mu_train_full, sigma2_train_full, n_samples=gl_samples)
    
    probs_train_omitted = unc_train_omitted['p_ale']
    probs_train_full = unc_train_full['p_ale']
    
    acc_omitted = _compute_accuracy_binary(Y, probs_train_omitted)
    acc_full = _compute_accuracy_binary(Y, probs_train_full)
    nll_omitted = _compute_nll_binary(Y, probs_train_omitted)
    nll_full = _compute_nll_binary(Y, probs_train_full)
    
    result = {
        'rho': rho,
        'beta2': beta2,
        'func_type': func_type,
        'acc_omitted': acc_omitted,
        'acc_full': acc_full,
        'nll_omitted': nll_omitted,
        'nll_full': nll_full,
        'au_mean_omitted': float(unc_omitted['AU'].mean()),
        'au_mean_full': float(unc_full['AU'].mean()),
        'eu_mean_omitted': float(unc_omitted['EU'].mean()),
        'eu_mean_full': float(unc_full['EU'].mean()),
        'au_omitted': unc_omitted['AU'],
        'eu_omitted': unc_omitted['EU'],
        'au_full': unc_full['AU'],
        'eu_full': unc_full['EU'],
        'probs_omitted': probs_omitted,
        'probs_full': probs_full,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'X': X,
        'Z': Z,
        'Y': Y,
        'true_probs': true_probs,
    }
    
    print(f"    Acc: omitted={acc_omitted:.4f}, full={acc_full:.4f}")
    print(f"    NLL: omitted={nll_omitted:.4f}, full={nll_full:.4f}")
    print(f"    AU:  omitted={unc_omitted['AU'].mean():.4f}, full={unc_full['AU'].mean():.4f}")
    print(f"    EU:  omitted={unc_omitted['EU'].mean():.4f}, full={unc_full['EU'].mean():.4f}")
    
    if results_dir and save_plots:
        save_dir = results_dir / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_ovb_classification_outputs(
            X, Z, Y, true_probs,
            probs_omitted, probs_full,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho, beta2, func_type,
            save_dir, param_name,
            model_type='deep_ensemble_gl',
            decomposition='gl'
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['AU'], unc_full['AU'],
            title='Aleatoric Uncertainty (Deep Ensemble GL)',
            value_name='AU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_au_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['EU'], unc_full['EU'],
            title='Epistemic Uncertainty (Deep Ensemble GL)',
            value_name='EU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_eu_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_marginals(
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'marginals_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_z_slices(
            x_grid, z_grid,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'z_slices_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
    
    return result


def run_deep_ensemble_gl_ovb_rho_experiment(
    rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    K: int = 10,
    gl_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying rho with Deep Ensemble GL."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_deep_ensemble_gl_ovb_rho_experiment(
                rho_values=rho_values, beta2=beta2,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, K=K, gl_samples=gl_samples,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"Deep Ensemble GL - Varying rho (beta2={beta2}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'deep_ensemble_gl'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for rho in rho_values:
        result = _run_single_ovb_classification_deep_ensemble_gl(
            rho=rho, beta2=beta2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, K=K, gl_samples=gl_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='rho'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_rho_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='rho',
            save_path=save_dir / func_type / 'summary_rho_plot.png',
            show=False
        )
    
    return results_df, {r['rho']: r for r in all_results}


def run_deep_ensemble_gl_ovb_beta2_experiment(
    beta2_values: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    K: int = 10,
    gl_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying beta2 with Deep Ensemble GL."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_deep_ensemble_gl_ovb_beta2_experiment(
                beta2_values=beta2_values, rho=rho,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, epochs=epochs, lr=lr,
                batch_size=batch_size, K=K, gl_samples=gl_samples,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"Deep Ensemble GL - Varying beta2 (rho={rho}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'deep_ensemble_gl'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for b2 in beta2_values:
        result = _run_single_ovb_classification_deep_ensemble_gl(
            rho=rho, beta2=b2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, epochs=epochs, lr=lr,
            batch_size=batch_size, K=K, gl_samples=gl_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='beta2'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_beta2_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='beta2',
            save_path=save_dir / func_type / 'summary_beta2_plot.png',
            show=False
        )
    
    return results_df, {r['beta2']: r for r in all_results}


# ============================================================================
# BNN IT Experiments
# ============================================================================

def _run_single_ovb_classification_bnn_it(
    rho: float,
    beta2: float,
    n_train: int,
    train_range: tuple,
    grid_points: int,
    func_type: str,
    seed: int,
    num_samples: int,
    warmup_steps: int,
    hidden_width: int,
    weight_scale: float,
    results_dir: Path = None,
    save_plots: bool = True,
    param_name: str = 'rho'
) -> Dict[str, Any]:
    """Run a single OVB classification experiment with BNN IT."""
    
    print(f"\n  Running BNN IT: rho={rho:.2f}, beta2={beta2:.2f}, func={func_type}")
    
    # Generate data
    X, Z, Y, true_probs, x_grid, z_grid = generate_ovb_classification_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed
    )
    
    # Generate 2D evaluation grid
    X_grid_2d, Z_grid_2d, true_probs_2d = generate_ovb_2d_grid(
        x_grid, z_grid, func_type=func_type, beta2=beta2
    )
    
    # Prepare full input [X, Z]
    XZ_train = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
    XZ_grid = np.hstack([X_grid_2d, Z_grid_2d.reshape(-1, 1)]).astype(np.float32)
    
    # Train omitted model (X only)
    print("    Training omitted BNN (X only)...")
    mcmc_omitted = train_bnn_it(
        X, Y,
        input_dim=1,
        num_classes=2,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        hidden_width=hidden_width,
        weight_scale=weight_scale
    )
    
    # Train full model (X and Z)
    print("    Training full BNN (X, Z)...")
    mcmc_full = train_bnn_it(
        XZ_train, Y,
        input_dim=2,
        num_classes=2,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        hidden_width=hidden_width,
        weight_scale=weight_scale
    )
    
    # Predict on 2D grid
    print("    Predicting on evaluation grid...")
    probs_members_omitted = bnn_predict_it(mcmc_omitted, X_grid_2d, input_dim=1, num_classes=2, hidden_width=hidden_width, weight_scale=weight_scale)
    probs_members_full = bnn_predict_it(mcmc_full, XZ_grid, input_dim=2, num_classes=2, hidden_width=hidden_width, weight_scale=weight_scale)
    
    # Compute IT uncertainties
    unc_omitted = it_uncertainty(probs_members_omitted)
    unc_full = it_uncertainty(probs_members_full)
    
    probs_omitted = unc_omitted['p_bar']
    probs_full = unc_full['p_bar']
    
    # Compute metrics on training data
    probs_members_train_omitted = bnn_predict_it(mcmc_omitted, X, input_dim=1, num_classes=2, hidden_width=hidden_width, weight_scale=weight_scale)
    probs_members_train_full = bnn_predict_it(mcmc_full, XZ_train, input_dim=2, num_classes=2, hidden_width=hidden_width, weight_scale=weight_scale)
    probs_train_omitted = probs_members_train_omitted.mean(axis=0)
    probs_train_full = probs_members_train_full.mean(axis=0)
    
    acc_omitted = _compute_accuracy_binary(Y, probs_train_omitted)
    acc_full = _compute_accuracy_binary(Y, probs_train_full)
    nll_omitted = _compute_nll_binary(Y, probs_train_omitted)
    nll_full = _compute_nll_binary(Y, probs_train_full)
    
    result = {
        'rho': rho,
        'beta2': beta2,
        'func_type': func_type,
        'acc_omitted': acc_omitted,
        'acc_full': acc_full,
        'nll_omitted': nll_omitted,
        'nll_full': nll_full,
        'au_mean_omitted': float(unc_omitted['AU'].mean()),
        'au_mean_full': float(unc_full['AU'].mean()),
        'eu_mean_omitted': float(unc_omitted['EU'].mean()),
        'eu_mean_full': float(unc_full['EU'].mean()),
        'tu_mean_omitted': float(unc_omitted['TU'].mean()),
        'tu_mean_full': float(unc_full['TU'].mean()),
        'au_omitted': unc_omitted['AU'],
        'eu_omitted': unc_omitted['EU'],
        'au_full': unc_full['AU'],
        'eu_full': unc_full['EU'],
        'probs_omitted': probs_omitted,
        'probs_full': probs_full,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'X': X,
        'Z': Z,
        'Y': Y,
        'true_probs': true_probs,
    }
    
    print(f"    Acc: omitted={acc_omitted:.4f}, full={acc_full:.4f}")
    print(f"    NLL: omitted={nll_omitted:.4f}, full={nll_full:.4f}")
    print(f"    AU:  omitted={unc_omitted['AU'].mean():.4f}, full={unc_full['AU'].mean():.4f}")
    print(f"    EU:  omitted={unc_omitted['EU'].mean():.4f}, full={unc_full['EU'].mean():.4f}")
    
    if results_dir and save_plots:
        save_dir = results_dir / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_ovb_classification_outputs(
            X, Z, Y, true_probs,
            probs_omitted, probs_full,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho, beta2, func_type,
            save_dir, param_name,
            model_type='bnn_it',
            decomposition='it'
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['AU'], unc_full['AU'],
            title='Aleatoric Uncertainty (BNN IT)',
            value_name='AU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_au_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['EU'], unc_full['EU'],
            title='Epistemic Uncertainty (BNN IT)',
            value_name='EU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_eu_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_marginals(
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'marginals_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_z_slices(
            x_grid, z_grid,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'z_slices_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
    
    return result


def run_bnn_it_ovb_rho_experiment(
    rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    num_samples: int = 200,
    warmup_steps: int = 100,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying rho with BNN IT."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_bnn_it_ovb_rho_experiment(
                rho_values=rho_values, beta2=beta2,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, num_samples=num_samples,
                warmup_steps=warmup_steps, hidden_width=hidden_width,
                weight_scale=weight_scale, save_plots=save_plots,
                results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"BNN IT - Varying rho (beta2={beta2}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'bnn_it'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for rho in rho_values:
        result = _run_single_ovb_classification_bnn_it(
            rho=rho, beta2=beta2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, num_samples=num_samples,
            warmup_steps=warmup_steps, hidden_width=hidden_width,
            weight_scale=weight_scale,
            results_dir=save_dir, save_plots=save_plots,
            param_name='rho'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full', 'tu_mean_omitted', 'tu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_rho_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='rho',
            save_path=save_dir / func_type / 'summary_rho_plot.png',
            show=False
        )
    
    return results_df, {r['rho']: r for r in all_results}


def run_bnn_it_ovb_beta2_experiment(
    beta2_values: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    num_samples: int = 200,
    warmup_steps: int = 100,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying beta2 with BNN IT."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_bnn_it_ovb_beta2_experiment(
                beta2_values=beta2_values, rho=rho,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, num_samples=num_samples,
                warmup_steps=warmup_steps, hidden_width=hidden_width,
                weight_scale=weight_scale, save_plots=save_plots,
                results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"BNN IT - Varying beta2 (rho={rho}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'bnn_it'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for b2 in beta2_values:
        result = _run_single_ovb_classification_bnn_it(
            rho=rho, beta2=b2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, num_samples=num_samples,
            warmup_steps=warmup_steps, hidden_width=hidden_width,
            weight_scale=weight_scale,
            results_dir=save_dir, save_plots=save_plots,
            param_name='beta2'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full', 'tu_mean_omitted', 'tu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_beta2_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='beta2',
            save_path=save_dir / func_type / 'summary_beta2_plot.png',
            show=False
        )
    
    return results_df, {r['beta2']: r for r in all_results}


# ============================================================================
# BNN GL Experiments
# ============================================================================

def _run_single_ovb_classification_bnn_gl(
    rho: float,
    beta2: float,
    n_train: int,
    train_range: tuple,
    grid_points: int,
    func_type: str,
    seed: int,
    num_samples: int,
    warmup_steps: int,
    hidden_width: int,
    weight_scale: float,
    gl_samples: int,
    results_dir: Path = None,
    save_plots: bool = True,
    param_name: str = 'rho'
) -> Dict[str, Any]:
    """Run a single OVB classification experiment with BNN GL."""
    
    print(f"\n  Running BNN GL: rho={rho:.2f}, beta2={beta2:.2f}, func={func_type}")
    
    # Generate data
    X, Z, Y, true_probs, x_grid, z_grid = generate_ovb_classification_data(
        n_train=n_train,
        train_range=train_range,
        grid_points=grid_points,
        func_type=func_type,
        rho=rho,
        beta2=beta2,
        seed=seed
    )
    
    # Generate 2D evaluation grid
    X_grid_2d, Z_grid_2d, true_probs_2d = generate_ovb_2d_grid(
        x_grid, z_grid, func_type=func_type, beta2=beta2
    )
    
    # Prepare full input [X, Z]
    XZ_train = np.hstack([X, Z.reshape(-1, 1)]).astype(np.float32)
    XZ_grid = np.hstack([X_grid_2d, Z_grid_2d.reshape(-1, 1)]).astype(np.float32)
    
    # Train omitted model (X only)
    print("    Training omitted BNN GL (X only)...")
    mcmc_omitted = train_bnn_gl(
        X, Y,
        input_dim=1,
        num_classes=2,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        hidden_width=hidden_width,
        weight_scale=weight_scale
    )
    
    # Train full model (X and Z)
    print("    Training full BNN GL (X, Z)...")
    mcmc_full = train_bnn_gl(
        XZ_train, Y,
        input_dim=2,
        num_classes=2,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        hidden_width=hidden_width,
        weight_scale=weight_scale
    )
    
    # Predict on 2D grid
    print("    Predicting on evaluation grid...")
    mu_members_omitted, sigma2_members_omitted = bnn_predict_gl(
        mcmc_omitted, X_grid_2d, input_dim=1, num_classes=2,
        hidden_width=hidden_width, weight_scale=weight_scale
    )
    mu_members_full, sigma2_members_full = bnn_predict_gl(
        mcmc_full, XZ_grid, input_dim=2, num_classes=2,
        hidden_width=hidden_width, weight_scale=weight_scale
    )
    
    # Compute GL uncertainties
    unc_omitted = gl_uncertainty(mu_members_omitted, sigma2_members_omitted, n_samples=gl_samples)
    unc_full = gl_uncertainty(mu_members_full, sigma2_members_full, n_samples=gl_samples)
    
    probs_omitted = unc_omitted['p_ale']
    probs_full = unc_full['p_ale']
    
    # Compute metrics on training data
    mu_train_omitted, sigma2_train_omitted = bnn_predict_gl(
        mcmc_omitted, X, input_dim=1, num_classes=2,
        hidden_width=hidden_width, weight_scale=weight_scale
    )
    mu_train_full, sigma2_train_full = bnn_predict_gl(
        mcmc_full, XZ_train, input_dim=2, num_classes=2,
        hidden_width=hidden_width, weight_scale=weight_scale
    )
    
    unc_train_omitted = gl_uncertainty(mu_train_omitted, sigma2_train_omitted, n_samples=gl_samples)
    unc_train_full = gl_uncertainty(mu_train_full, sigma2_train_full, n_samples=gl_samples)
    
    probs_train_omitted = unc_train_omitted['p_ale']
    probs_train_full = unc_train_full['p_ale']
    
    acc_omitted = _compute_accuracy_binary(Y, probs_train_omitted)
    acc_full = _compute_accuracy_binary(Y, probs_train_full)
    nll_omitted = _compute_nll_binary(Y, probs_train_omitted)
    nll_full = _compute_nll_binary(Y, probs_train_full)
    
    result = {
        'rho': rho,
        'beta2': beta2,
        'func_type': func_type,
        'acc_omitted': acc_omitted,
        'acc_full': acc_full,
        'nll_omitted': nll_omitted,
        'nll_full': nll_full,
        'au_mean_omitted': float(unc_omitted['AU'].mean()),
        'au_mean_full': float(unc_full['AU'].mean()),
        'eu_mean_omitted': float(unc_omitted['EU'].mean()),
        'eu_mean_full': float(unc_full['EU'].mean()),
        'au_omitted': unc_omitted['AU'],
        'eu_omitted': unc_omitted['EU'],
        'au_full': unc_full['AU'],
        'eu_full': unc_full['EU'],
        'probs_omitted': probs_omitted,
        'probs_full': probs_full,
        'x_grid': x_grid,
        'z_grid': z_grid,
        'X': X,
        'Z': Z,
        'Y': Y,
        'true_probs': true_probs,
    }
    
    print(f"    Acc: omitted={acc_omitted:.4f}, full={acc_full:.4f}")
    print(f"    NLL: omitted={nll_omitted:.4f}, full={nll_full:.4f}")
    print(f"    AU:  omitted={unc_omitted['AU'].mean():.4f}, full={unc_full['AU'].mean():.4f}")
    print(f"    EU:  omitted={unc_omitted['EU'].mean():.4f}, full={unc_full['EU'].mean():.4f}")
    
    if results_dir and save_plots:
        save_dir = results_dir / func_type
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_ovb_classification_outputs(
            X, Z, Y, true_probs,
            probs_omitted, probs_full,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho, beta2, func_type,
            save_dir, param_name,
            model_type='bnn_gl',
            decomposition='gl'
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['AU'], unc_full['AU'],
            title='Aleatoric Uncertainty (BNN GL)',
            value_name='AU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_au_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_heatmap(
            x_grid, z_grid, unc_omitted['EU'], unc_full['EU'],
            title='Epistemic Uncertainty (BNN GL)',
            value_name='EU', rho=rho, beta2=beta2,
            save_path=save_dir / f'heatmap_eu_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_marginals(
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'marginals_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
        
        plot_ovb_classification_z_slices(
            x_grid, z_grid,
            unc_omitted['AU'], unc_omitted['EU'],
            unc_full['AU'], unc_full['EU'],
            rho=rho, beta2=beta2,
            save_path=save_dir / f'z_slices_{param_name}{rho if param_name=="rho" else beta2:.2f}.png',
            show=False
        )
    
    return result


def run_bnn_gl_ovb_rho_experiment(
    rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 0.9],
    beta2: float = 1.0,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    num_samples: int = 200,
    warmup_steps: int = 100,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    gl_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying rho with BNN GL."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_bnn_gl_ovb_rho_experiment(
                rho_values=rho_values, beta2=beta2,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, num_samples=num_samples,
                warmup_steps=warmup_steps, hidden_width=hidden_width,
                weight_scale=weight_scale, gl_samples=gl_samples,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"BNN GL - Varying rho (beta2={beta2}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'bnn_gl'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for rho in rho_values:
        result = _run_single_ovb_classification_bnn_gl(
            rho=rho, beta2=beta2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, num_samples=num_samples,
            warmup_steps=warmup_steps, hidden_width=hidden_width,
            weight_scale=weight_scale, gl_samples=gl_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='rho'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_rho_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='rho',
            save_path=save_dir / func_type / 'summary_rho_plot.png',
            show=False
        )
    
    return results_df, {r['rho']: r for r in all_results}


def run_bnn_gl_ovb_beta2_experiment(
    beta2_values: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    rho: float = 0.7,
    n_train: int = 1000,
    train_range: tuple = (-3.0, 3.0),
    grid_points: int = 50,
    func_type: Union[str, List[str]] = 'linear',
    seed: int = 42,
    num_samples: int = 200,
    warmup_steps: int = 100,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    gl_samples: int = 100,
    save_plots: bool = True,
    results_dir: Path = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run OVB classification experiment varying beta2 with BNN GL."""
    
    func_types = [func_type] if isinstance(func_type, str) else list(func_type)
    
    if len(func_types) > 1:
        combined_dfs = []
        combined_results = {}
        for ft in func_types:
            print(f"\n{'='*60}")
            print(f"Running config: func_type={ft}")
            print(f"{'='*60}")
            df, results = run_bnn_gl_ovb_beta2_experiment(
                beta2_values=beta2_values, rho=rho,
                n_train=n_train, train_range=train_range,
                grid_points=grid_points, func_type=ft,
                seed=seed, num_samples=num_samples,
                warmup_steps=warmup_steps, hidden_width=hidden_width,
                weight_scale=weight_scale, gl_samples=gl_samples,
                save_plots=save_plots, results_dir=results_dir
            )
            df['func_type'] = ft
            combined_dfs.append(df)
            combined_results[ft] = results
        return pd.concat(combined_dfs, ignore_index=True), combined_results
    
    func_type = func_types[0]
    print(f"\n{'='*60}")
    print(f"BNN GL - Varying beta2 (rho={rho}, func={func_type})")
    print(f"{'='*60}")
    
    if results_dir:
        save_dir = results_dir / 'bnn_gl'
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
    
    all_results = []
    for b2 in beta2_values:
        result = _run_single_ovb_classification_bnn_gl(
            rho=rho, beta2=b2,
            n_train=n_train, train_range=train_range,
            grid_points=grid_points, func_type=func_type,
            seed=seed, num_samples=num_samples,
            warmup_steps=warmup_steps, hidden_width=hidden_width,
            weight_scale=weight_scale, gl_samples=gl_samples,
            results_dir=save_dir, save_plots=save_plots,
            param_name='beta2'
        )
        all_results.append(result)
    
    summary_cols = ['rho', 'beta2', 'func_type', 'acc_omitted', 'acc_full',
                    'nll_omitted', 'nll_full', 'au_mean_omitted', 'au_mean_full',
                    'eu_mean_omitted', 'eu_mean_full']
    results_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    
    if save_dir:
        summary_path = save_dir / func_type / f'summary_beta2_experiment.xlsx'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_excel(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        plot_ovb_classification_summary(
            results_df, param_name='beta2',
            save_path=save_dir / func_type / 'summary_beta2_plot.png',
            show=False
        )
    
    return results_df, {r['beta2']: r for r in all_results}
