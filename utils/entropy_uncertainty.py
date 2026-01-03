"""
Entropy-based uncertainty decomposition for regression models.

This module provides functions to decompose uncertainty into aleatoric and epistemic
components using differential entropy instead of variance. Two methods are available:
1. Analytical: Fast approximation assuming single Gaussian
2. Numerical: Accurate Monte Carlo estimation of mixture entropy
"""

import numpy as np
from scipy.stats import norm


def _gaussian_entropy(variance, eps=1e-10):
    """
    Compute differential entropy for a Gaussian distribution.
    
    For a Gaussian with variance σ², differential entropy is:
    H(X) = 0.5 * log(2πe * σ²)
    
    Parameters:
    -----------
    variance : array-like
        Variance values (can be scalar or array)
    eps : float
        Small value to avoid log(0)
    
    Returns:
    --------
    entropy : array-like
        Differential entropy values (in nats)
    """
    variance = np.asarray(variance)
    variance = np.maximum(variance, eps)  # Ensure variance > 0
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def _sample_from_mixture(mu, sigma2, n_samples, seed=None):
    """
    Sample from a Gaussian mixture distribution.
    
    Parameters:
    -----------
    mu : array-like, shape (M, N)
        Means for M components and N data points
    sigma2 : array-like, shape (M, N)
        Variances for M components and N data points
    n_samples : int
        Number of samples to draw
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    samples : array, shape (n_samples, N)
        Samples from the mixture distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    M, N = mu.shape
    sigma = np.sqrt(sigma2)
    
    # Randomly select which Gaussian component for each sample
    component_idx = np.random.randint(0, M, size=(n_samples, N))
    
    # Sample from the selected Gaussian for each data point
    n_idx = np.arange(N)
    mu_selected = mu[component_idx, n_idx]  # shape (n_samples, N)
    sigma_selected = sigma[component_idx, n_idx]  # shape (n_samples, N)
    
    # Sample from selected Gaussians
    samples = np.random.normal(mu_selected, sigma_selected)
    
    return samples


def _mixture_pdf_log(y_samples, mu, sigma2):
    """
    Evaluate log PDF of Gaussian mixture at samples.
    
    Uses log-sum-exp trick for numerical stability.
    
    Parameters:
    -----------
    y_samples : array-like, shape (n_samples, N)
        Points at which to evaluate PDF
    mu : array-like, shape (M, N)
        Means for M components and N data points
    sigma2 : array-like, shape (M, N)
        Variances for M components and N data points
    
    Returns:
    --------
    log_pdf : array, shape (n_samples, N)
        Log PDF values at each sample point
    """
    y_samples = np.asarray(y_samples)
    mu = np.asarray(mu)
    sigma2 = np.asarray(sigma2)
    
    n_samples, N = y_samples.shape
    M = mu.shape[0]
    sigma = np.sqrt(sigma2)
    
    # Expand dimensions for broadcasting: (n_samples, 1, N) and (1, M, N)
    y_expanded = y_samples[:, np.newaxis, :]  # (n_samples, 1, N)
    mu_expanded = mu[np.newaxis, :, :]  # (1, M, N)
    sigma_expanded = sigma[np.newaxis, :, :]  # (1, M, N)
    
    # Compute log PDF for each component: log N(y; μ_m, σ_m)
    # log N(y; μ, σ) = -0.5 * log(2πσ²) - 0.5 * (y - μ)² / σ²
    log_2pi = np.log(2 * np.pi)
    log_component_pdfs = -0.5 * (log_2pi + np.log(sigma2[np.newaxis, :, :])) - \
                         0.5 * ((y_expanded - mu_expanded) ** 2) / sigma2[np.newaxis, :, :]
    # Shape: (n_samples, M, N)
    
    # Log-sum-exp trick: log(mean(exp(log_pdfs))) = max + log(mean(exp(log_pdfs - max)))
    max_log = np.max(log_component_pdfs, axis=1, keepdims=True)  # (n_samples, 1, N)
    log_mixture_pdf = np.squeeze(max_log) + np.log(np.mean(np.exp(log_component_pdfs - max_log), axis=1))
    # Shape: (n_samples, N)
    
    return log_mixture_pdf


def entropy_uncertainty_analytical(mu, sigma2):
    """
    Analytical entropy-based uncertainty decomposition.
    
    Approximates the Gaussian mixture by a single Gaussian with mean parameters.
    This is fast but may underestimate total uncertainty when ensemble members disagree.
    
    Parameters:
    -----------
    mu : array-like, shape (M, N)
        Predicted means from M forward passes/models for N data points
    sigma2 : array-like, shape (M, N)
        Predicted variances from M forward passes/models for N data points
    
    Returns:
    --------
    dict with keys:
        'aleatoric': array, shape (N,) - Aleatoric entropy (mean of individual entropies)
        'epistemic': array, shape (N,) - Epistemic entropy (TU - AU)
        'total': array, shape (N,) - Total entropy (entropy of approximated Gaussian)
    """
    mu = np.asarray(mu)
    sigma2 = np.asarray(sigma2)
    
    # Ensure correct shape: (M, N)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if sigma2.ndim == 1:
        sigma2 = sigma2.reshape(1, -1)
    
    M, N = mu.shape
    
    # Aleatoric uncertainty: mean of individual Gaussian entropies
    # AU = (1/M) Σ [0.5 * log(2πe * σ²_m)]
    individual_entropies = _gaussian_entropy(sigma2)  # shape (M, N)
    aleatoric_entropy = np.mean(individual_entropies, axis=0)  # shape (N,)
    
    # Total uncertainty: entropy of Gaussian with mean parameters
    # TU = 0.5 * log(2πe * σ̄²) where σ̄² = mean(σ²_m)
    mean_variance = np.mean(sigma2, axis=0)  # shape (N,)
    total_entropy = _gaussian_entropy(mean_variance)  # shape (N,)
    
    # Epistemic uncertainty: difference
    # EU = TU - AU
    epistemic_entropy = total_entropy - aleatoric_entropy
    
    return {
        'aleatoric': aleatoric_entropy,
        'epistemic': epistemic_entropy,
        'total': total_entropy
    }


def entropy_uncertainty_numerical(mu, sigma2, n_samples=5000, seed=None):
    """
    Monte Carlo entropy-based uncertainty decomposition.
    
    Estimates the true Gaussian mixture entropy via sampling. This is slower but
    more accurate than the analytical method, especially when ensemble members disagree.
    
    Parameters:
    -----------
    mu : array-like, shape (M, N)
        Predicted means from M forward passes/models for N data points
    sigma2 : array-like, shape (M, N)
        Predicted variances from M forward passes/models for N data points
    n_samples : int, default=5000
        Number of Monte Carlo samples for mixture entropy estimation
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict with keys:
        'aleatoric': array, shape (N,) - Aleatoric entropy (same as analytical)
        'epistemic': array, shape (N,) - Epistemic entropy (TU - AU)
        'total': array, shape (N,) - Total entropy (Monte Carlo estimate)
    """
    mu = np.asarray(mu)
    sigma2 = np.asarray(sigma2)
    
    # Ensure correct shape: (M, N)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if sigma2.ndim == 1:
        sigma2 = sigma2.reshape(1, -1)
    
    M, N = mu.shape
    
    # Aleatoric uncertainty: same as analytical (closed-form)
    individual_entropies = _gaussian_entropy(sigma2)  # shape (M, N)
    aleatoric_entropy = np.mean(individual_entropies, axis=0)  # shape (N,)
    
    # Total uncertainty: Monte Carlo estimate of mixture entropy
    # TU = H[p(y)] ≈ -(1/S) Σ log(p(y_s)), where y_s ~ p(y)
    # and p(y) = (1/M) Σ N(y; μ_m, σ_m)
    
    # Sample from mixture
    samples = _sample_from_mixture(mu, sigma2, n_samples, seed=seed)  # shape (n_samples, N)
    
    # Evaluate log PDF of full mixture at samples
    log_pdf = _mixture_pdf_log(samples, mu, sigma2)  # shape (n_samples, N)
    
    # Total entropy: negative mean of log PDF
    total_entropy = -np.mean(log_pdf, axis=0)  # shape (N,)
    
    # Epistemic uncertainty: difference
    epistemic_entropy = total_entropy - aleatoric_entropy
    
    return {
        'aleatoric': aleatoric_entropy,
        'epistemic': epistemic_entropy,
        'total': total_entropy
    }

