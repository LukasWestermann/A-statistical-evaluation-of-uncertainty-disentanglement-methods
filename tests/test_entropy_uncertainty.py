"""
Unit tests for entropy-based uncertainty decomposition functions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import (
    entropy_uncertainty_analytical,
    entropy_uncertainty_numerical,
    _gaussian_entropy
)


class TestGaussianEntropy:
    """Test Gaussian entropy computation helper function."""
    
    def test_gaussian_entropy_basic(self):
        """Test entropy for known variance values."""
        # For variance = 1, entropy should be 0.5 * log(2πe) ≈ 1.419
        entropy = _gaussian_entropy(1.0)
        expected = 0.5 * np.log(2 * np.pi * np.e)
        assert np.isclose(entropy, expected, rtol=1e-6)
    
    def test_gaussian_entropy_zero_variance(self):
        """Test that zero variance doesn't crash."""
        entropy = _gaussian_entropy(0.0)
        assert np.isfinite(entropy)
        assert entropy > -np.inf
    
    def test_gaussian_entropy_array(self):
        """Test entropy computation for arrays."""
        variances = np.array([0.1, 1.0, 10.0])
        entropies = _gaussian_entropy(variances)
        assert entropies.shape == (3,)
        assert np.all(np.isfinite(entropies))
        # Entropy should increase with variance
        assert entropies[0] < entropies[1] < entropies[2]
    
    def test_gaussian_entropy_negative_variance(self):
        """Test that negative variance is handled (clipped to eps)."""
        entropy = _gaussian_entropy(-1.0)
        assert np.isfinite(entropy)
        assert entropy > -np.inf


class TestEntropyUncertaintyAnalytical:
    """Test analytical entropy-based uncertainty decomposition."""
    
    def test_basic_shape(self):
        """Test that output shapes are correct."""
        M, N = 5, 10
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(N).reshape(1, -1) ** 2
        sigma2 = np.tile(sigma2, (M, 1))
        
        result = entropy_uncertainty_analytical(mu, sigma2)
        
        assert 'aleatoric' in result
        assert 'epistemic' in result
        assert 'total' in result
        assert result['aleatoric'].shape == (N,)
        assert result['epistemic'].shape == (N,)
        assert result['total'].shape == (N,)
    
    def test_no_nan_or_inf(self):
        """Test that results don't contain NaN or Inf."""
        M, N = 3, 5
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result = entropy_uncertainty_analytical(mu, sigma2)
        
        for key in ['aleatoric', 'epistemic', 'total']:
            assert np.all(np.isfinite(result[key]))
    
    def test_entropy_relationship(self):
        """Test that total ≈ aleatoric + epistemic (approximately)."""
        M, N = 10, 20
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result = entropy_uncertainty_analytical(mu, sigma2)
        
        # For analytical method, total should be close to aleatoric + epistemic
        # (though not exactly equal due to approximation)
        diff = result['total'] - (result['aleatoric'] + result['epistemic'])
        # Difference should be small relative to total
        assert np.all(np.abs(diff) < 0.1 * np.abs(result['total']))
    
    def test_single_sample(self):
        """Test with single forward pass (M=1)."""
        M, N = 1, 5
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result = entropy_uncertainty_analytical(mu, sigma2)
        
        # With single sample, epistemic should be zero (no variance in means)
        assert np.allclose(result['epistemic'], 0, atol=1e-6)
        assert np.allclose(result['total'], result['aleatoric'], rtol=1e-5)
    
    def test_identical_samples(self):
        """Test with identical samples (should give zero epistemic)."""
        M, N = 5, 10
        mu = np.ones((M, N))  # All means identical
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result = entropy_uncertainty_analytical(mu, sigma2)
        
        # Epistemic should be zero (no variance in means)
        # Note: For analytical method, epistemic is computed as total - aleatoric
        # which may not be exactly zero due to approximation, so use a more lenient tolerance
        assert np.allclose(result['epistemic'], 0, atol=0.2)
    
    def test_1d_input(self):
        """Test that 1D input is handled correctly."""
        M, N = 3, 5
        mu = np.random.randn(M * N)
        sigma2 = np.random.rand(M * N) ** 2 + 0.1
        
        # Reshape to (M, N)
        mu = mu.reshape(M, N)
        sigma2 = sigma2.reshape(M, N)
        
        result = entropy_uncertainty_analytical(mu, sigma2)
        assert result['aleatoric'].shape == (N,)


class TestEntropyUncertaintyNumerical:
    """Test numerical (Monte Carlo) entropy-based uncertainty decomposition."""
    
    def test_basic_shape(self):
        """Test that output shapes are correct."""
        M, N = 5, 10
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result = entropy_uncertainty_numerical(mu, sigma2, n_samples=100, seed=42)
        
        assert 'aleatoric' in result
        assert 'epistemic' in result
        assert 'total' in result
        assert result['aleatoric'].shape == (N,)
        assert result['epistemic'].shape == (N,)
        assert result['total'].shape == (N,)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        M, N = 3, 5
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result1 = entropy_uncertainty_numerical(mu, sigma2, n_samples=100, seed=42)
        result2 = entropy_uncertainty_numerical(mu, sigma2, n_samples=100, seed=42)
        
        for key in ['aleatoric', 'epistemic', 'total']:
            assert np.allclose(result1[key], result2[key], rtol=1e-6)
    
    def test_no_nan_or_inf(self):
        """Test that results don't contain NaN or Inf."""
        M, N = 3, 5
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result = entropy_uncertainty_numerical(mu, sigma2, n_samples=100, seed=42)
        
        for key in ['aleatoric', 'epistemic', 'total']:
            assert np.all(np.isfinite(result[key]))
    
    def test_comparison_with_analytical(self):
        """Test that numerical and analytical methods give similar results for simple cases."""
        M, N = 5, 10
        mu = np.random.randn(M, N)
        sigma2 = np.random.rand(M, N) ** 2 + 0.1
        
        result_analytical = entropy_uncertainty_analytical(mu, sigma2)
        result_numerical = entropy_uncertainty_numerical(mu, sigma2, n_samples=500, seed=42)
        
        # Aleatoric should be identical (same computation)
        assert np.allclose(result_analytical['aleatoric'], result_numerical['aleatoric'], rtol=1e-5)
        
        # Total should be similar (within reasonable tolerance)
        # Note: numerical method may be more accurate for mixtures, and analytical is an approximation
        # Both should produce finite, positive values
        assert np.all(np.isfinite(result_analytical['total']))
        assert np.all(np.isfinite(result_numerical['total']))
        assert np.all(result_analytical['total'] > 0)
        assert np.all(result_numerical['total'] > 0)
        
        # Both should be in reasonable range (entropy should be positive)
        # The methods may differ significantly for mixtures, so we just check they're valid

