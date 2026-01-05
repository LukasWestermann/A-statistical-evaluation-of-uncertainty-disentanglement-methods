"""
Utility functions for computing evaluation metrics for uncertainty-aware models.

This module provides functions to compute:
- Predictive aggregation (μ*, σ*²)
- Gaussian Negative Log-Likelihood (NLL)
- Continuous Ranked Probability Score (CRPS)
- Uncertainty disentanglement metrics (Spearman correlations)
- True noise variance computation
"""

import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.special import erf

try:
    # On Windows, cffi (used by properscoring) only supports ABI mode, not API mode
    # Try to set environment variable to suppress cffi warnings
    import os
    if os.name == 'nt':  # Windows
        os.environ.setdefault('CFFI_MODE', 'ABI')
    
    import properscoring as ps
    PROPERSCORING_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    # On Windows, properscoring may fail due to cffi issues:
    # - cffi API mode not supported on Windows (requires C compiler)
    # - ABI mode fallback may also fail in some configurations
    # We have a reliable fallback: closed-form Gaussian CRPS formula produces identical results
    PROPERSCORING_AVAILABLE = False
    # Suppress warning - fallback implementation is mathematically equivalent


def compute_predictive_aggregation(mu_samples, sigma2_samples):
    """
    Compute aggregated predictive mean and variance from per-sample predictions.
    
    For each input x:
    μ*(x) = mean_i μ_i(x)
    σ*²(x) = mean_i [σ_i²(x) + μ_i(x)²] − μ*(x)²
    
    Parameters:
    -----------
    mu_samples : np.ndarray, shape (M, N) or (M, 1, N)
        Per-sample predicted means for M samples/models and N data points
    sigma2_samples : np.ndarray, shape (M, N) or (M, 1, N)
        Per-sample predicted variances for M samples/models and N data points
    
    Returns:
    --------
    mu_star : np.ndarray, shape (N,)
        Aggregated predictive mean
    sigma2_star : np.ndarray, shape (N,)
        Aggregated predictive variance
    """
    mu_samples = np.asarray(mu_samples)
    sigma2_samples = np.asarray(sigma2_samples)
    
    # Ensure 2D shape [M, N]
    if mu_samples.ndim == 3:
        mu_samples = mu_samples.squeeze(1)
    if sigma2_samples.ndim == 3:
        sigma2_samples = sigma2_samples.squeeze(1)
    
    if mu_samples.ndim == 1:
        mu_samples = mu_samples.reshape(1, -1)
    if sigma2_samples.ndim == 1:
        sigma2_samples = sigma2_samples.reshape(1, -1)
    
    # Ensure correct orientation: [M, N]
    if mu_samples.shape[0] > mu_samples.shape[1] and mu_samples.shape[1] == 1:
        mu_samples = mu_samples.T
    if sigma2_samples.shape[0] > sigma2_samples.shape[1] and sigma2_samples.shape[1] == 1:
        sigma2_samples = sigma2_samples.T
    
    M, N = mu_samples.shape
    
    # Predictive mean: μ*(x) = mean_i μ_i(x)
    mu_star = np.mean(mu_samples, axis=0)
    
    # Predictive variance: σ*²(x) = mean_i [σ_i²(x) + μ_i(x)²] − μ*(x)²
    # This is equivalent to: Var(μ) + E[σ²]
    sigma2_star = np.mean(sigma2_samples + mu_samples**2, axis=0) - mu_star**2
    
    # Ensure non-negative variance
    sigma2_star = np.maximum(sigma2_star, 1e-10)
    
    return mu_star, sigma2_star


def compute_gaussian_nll(y_true, mu_star, sigma2_star):
    """
    Compute Gaussian Negative Log-Likelihood.
    
    NLL = mean_x [ 0.5 * log(2π * σ*²(x)) + (y − μ*(x))² / (2 * σ*²(x)) ]
    
    Parameters:
    -----------
    y_true : np.ndarray, shape (N,) or (N, 1)
        True target values
    mu_star : np.ndarray, shape (N,)
        Aggregated predictive mean
    sigma2_star : np.ndarray, shape (N,)
        Aggregated predictive variance
    
    Returns:
    --------
    nll : float
        Mean negative log-likelihood
    """
    y_true = np.asarray(y_true).flatten()
    mu_star = np.asarray(mu_star).flatten()
    sigma2_star = np.asarray(sigma2_star).flatten()
    
    # Ensure same length
    assert len(y_true) == len(mu_star) == len(sigma2_star), \
        f"Length mismatch: y_true={len(y_true)}, mu_star={len(mu_star)}, sigma2_star={len(sigma2_star)}"
    
    # Ensure positive variance
    sigma2_star = np.maximum(sigma2_star, 1e-10)
    
    # Compute NLL: 0.5 * log(2π * σ²) + (y - μ)² / (2 * σ²)
    nll_per_point = 0.5 * np.log(2 * np.pi * sigma2_star) + (y_true - mu_star)**2 / (2 * sigma2_star)
    
    # Return mean NLL
    return np.mean(nll_per_point)


def compute_crps_gaussian(y_true, mu_star, sigma2_star):
    """
    Compute Continuous Ranked Probability Score (CRPS) for Gaussian distributions.
    
    For Gaussian closed-form:
    z = (y − μ*(x)) / σ*(x)
    CRPS(x) = σ*(x) [ z (2Φ(z) − 1) + 2φ(z) − 1/√π ]
    CRPS = mean_x [CRPS(x)]
    
    Parameters:
    -----------
    y_true : np.ndarray, shape (N,) or (N, 1)
        True target values
    mu_star : np.ndarray, shape (N,)
        Aggregated predictive mean
    sigma2_star : np.ndarray, shape (N,)
        Aggregated predictive variance
    
    Returns:
    --------
    crps : float
        Mean CRPS
    """
    y_true = np.asarray(y_true).flatten()
    mu_star = np.asarray(mu_star).flatten()
    sigma2_star = np.asarray(sigma2_star).flatten()
    
    # Ensure same length
    assert len(y_true) == len(mu_star) == len(sigma2_star), \
        f"Length mismatch: y_true={len(y_true)}, mu_star={len(mu_star)}, sigma2_star={len(sigma2_star)}"
    
    # Ensure positive variance
    sigma2_star = np.maximum(sigma2_star, 1e-10)
    sigma_star = np.sqrt(sigma2_star)
    
    # Try using properscoring if available
    if PROPERSCORING_AVAILABLE:
        try:
            crps_per_point = ps.crps_gaussian(y_true, mu_star, sigma_star)
            return np.mean(crps_per_point)
        except Exception:
            # Fall back to closed-form if properscoring fails
            pass
    
    # Closed-form Gaussian CRPS
    z = (y_true - mu_star) / sigma_star
    
    # Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / √2))
    phi_z = 0.5 * (1 + erf(z / np.sqrt(2)))
    
    # Standard normal PDF: φ(z) = (1/√(2π)) * exp(-z²/2)
    phi_z_pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # CRPS formula: σ * [ z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π ]
    crps_per_point = sigma_star * (z * (2 * phi_z - 1) + 2 * phi_z_pdf - 1 / np.sqrt(np.pi))
    
    return np.mean(crps_per_point)


def compute_true_noise_variance(x_grid, noise_type, func_type=None, tau=2.5):
    """
    Compute true noise variance for grid points based on noise function.
    
    Parameters:
    -----------
    x_grid : np.ndarray, shape (N, 1) or (N,)
        Grid points
    noise_type : str
        'homoscedastic' or 'heteroscedastic'
    func_type : str, optional
        Function type (not used currently, but kept for consistency)
    tau : float, default=2.5
        Noise scale parameter (used for heteroscedastic and homoscedastic)
    
    Returns:
    --------
    true_noise_var : np.ndarray, shape (N,)
        True noise variance for each grid point
    """
    x_grid = np.asarray(x_grid).flatten()
    
    if noise_type == 'homoscedastic':
        # Homoscedastic: σ(x) = tau (default tau=2, but can vary)
        sigma = tau if tau != 2.5 else 2.0  # Default to 2.0 for OOD experiments
        true_noise_var = np.full_like(x_grid, sigma**2, dtype=np.float32)
    elif noise_type == 'heteroscedastic':
        # Heteroscedastic: σ(x) = |tau * sin(0.5*x + 5)|
        sigma = np.abs(tau * np.sin(0.5 * x_grid + 5))
        true_noise_var = sigma**2
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Must be 'homoscedastic' or 'heteroscedastic'")
    
    return true_noise_var


def compute_uncertainty_disentanglement(y_true, mu_star, ale_var, epi_var, true_noise_var=None):
    """
    Compute uncertainty disentanglement metrics using Spearman correlations.
    
    Computes:
    - Corr(true_noise, σ²_aleatoric) = Spearman correlation
    - Corr(error, σ²_epistemic) = Spearman correlation
    
    Parameters:
    -----------
    y_true : np.ndarray, shape (N,) or (N, 1)
        True target values
    mu_star : np.ndarray, shape (N,)
        Aggregated predictive mean
    ale_var : np.ndarray, shape (N,)
        Aleatoric variance
    epi_var : np.ndarray, shape (N,)
        Epistemic variance
    true_noise_var : np.ndarray, shape (N,), optional
        True noise variance. If None, aleatoric correlation is not computed.
    
    Returns:
    --------
    correlations : dict
        Dictionary with keys:
        - 'spearman_aleatoric': float or None (if true_noise_var not provided)
        - 'spearman_epistemic': float
    """
    y_true = np.asarray(y_true).flatten()
    mu_star = np.asarray(mu_star).flatten()
    ale_var = np.asarray(ale_var).flatten()
    epi_var = np.asarray(epi_var).flatten()
    
    # Ensure same length
    assert len(y_true) == len(mu_star) == len(ale_var) == len(epi_var), \
        f"Length mismatch: y_true={len(y_true)}, mu_star={len(mu_star)}, ale_var={len(ale_var)}, epi_var={len(epi_var)}"
    
    correlations = {}
    
    # Compute absolute prediction errors
    errors = np.abs(y_true - mu_star)
    
    # Spearman correlation: Corr(error, σ²_epistemic)
    if len(np.unique(epi_var)) > 1 and len(np.unique(errors)) > 1:
        corr_epi, _ = spearmanr(errors, epi_var)
        correlations['spearman_epistemic'] = corr_epi if not np.isnan(corr_epi) else 0.0
    else:
        correlations['spearman_epistemic'] = 0.0
    
    # Spearman correlation: Corr(true_noise, σ²_aleatoric)
    if true_noise_var is not None:
        true_noise_var = np.asarray(true_noise_var).flatten()
        assert len(true_noise_var) == len(ale_var), \
            f"Length mismatch: true_noise_var={len(true_noise_var)}, ale_var={len(ale_var)}"
        
        if len(np.unique(true_noise_var)) > 1 and len(np.unique(ale_var)) > 1:
            corr_ale, _ = spearmanr(true_noise_var, ale_var)
            correlations['spearman_aleatoric'] = corr_ale if not np.isnan(corr_ale) else 0.0
        else:
            correlations['spearman_aleatoric'] = 0.0
    else:
        correlations['spearman_aleatoric'] = None
    
    return correlations

