"""
Smoke tests for plotting functions.
These tests ensure plotting functions don't crash, but don't verify plot content.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.plotting import (
    plot_uncertainties_no_ood,
    plot_uncertainties_ood,
    plot_uncertainties_entropy_no_ood,
    plot_uncertainties_entropy_ood,
    plot_uncertainties_undersampling,
    plot_uncertainties_entropy_undersampling
)


class TestPlottingFunctions:
    """Smoke tests for plotting functions - just check they don't crash."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy data for plotting."""
        n_train, n_grid = 20, 30
        x_train = np.random.randn(n_train, 1)
        y_train = np.random.randn(n_train, 1)
        x_grid = np.linspace(-2, 2, n_grid).reshape(-1, 1)
        y_clean = np.sin(x_grid)
        
        return x_train, y_train, x_grid, y_clean
    
    def test_plot_uncertainties_no_ood(self, dummy_data):
        """Test that variance plotting doesn't crash."""
        x_train, y_train, x_grid, y_clean = dummy_data
        
        mu_pred = np.sin(x_grid).squeeze()
        ale_var = np.ones(len(x_grid)) * 0.1
        epi_var = np.ones(len(x_grid)) * 0.05
        tot_var = ale_var + epi_var
        
        # Should not raise
        plot_uncertainties_no_ood(
            x_train, y_train, x_grid, y_clean,
            mu_pred, ale_var, epi_var, tot_var,
            title="Test Plot",
            noise_type='heteroscedastic',
            func_type='sin'
        )
    
    def test_plot_uncertainties_entropy_no_ood(self, dummy_data):
        """Test that entropy plotting doesn't crash."""
        x_train, y_train, x_grid, y_clean = dummy_data
        
        mu_pred = np.sin(x_grid).squeeze()
        ale_entropy = np.ones(len(x_grid)) * 0.5
        epi_entropy = np.ones(len(x_grid)) * 0.3
        tot_entropy = ale_entropy + epi_entropy
        
        # Should not raise
        plot_uncertainties_entropy_no_ood(
            x_train, y_train, x_grid, y_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy,
            title="Test Plot Entropy",
            noise_type='heteroscedastic',
            func_type='sin'
        )
    
    def test_plot_uncertainties_ood(self, dummy_data):
        """Test OOD plotting doesn't crash."""
        x_train, y_train, x_grid, y_clean = dummy_data
        
        mu_pred = np.sin(x_grid).squeeze()
        ale_var = np.ones(len(x_grid)) * 0.1
        epi_var = np.ones(len(x_grid)) * 0.05
        tot_var = ale_var + epi_var
        ood_mask = (x_grid.squeeze() > 1.5) | (x_grid.squeeze() < -1.5)
        
        plot_uncertainties_ood(
            x_train, y_train, x_grid, y_clean,
            mu_pred, ale_var, epi_var, tot_var, ood_mask,
            title="Test OOD Plot",
            noise_type='heteroscedastic',
            func_type='sin'
        )
    
    def test_plot_uncertainties_entropy_ood(self, dummy_data):
        """Test entropy OOD plotting doesn't crash."""
        x_train, y_train, x_grid, y_clean = dummy_data
        
        mu_pred = np.sin(x_grid).squeeze()
        ale_entropy = np.ones(len(x_grid)) * 0.5
        epi_entropy = np.ones(len(x_grid)) * 0.3
        tot_entropy = ale_entropy + epi_entropy
        ood_mask = (x_grid.squeeze() > 1.5) | (x_grid.squeeze() < -1.5)
        
        plot_uncertainties_entropy_ood(
            x_train, y_train, x_grid, y_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy, ood_mask,
            title="Test OOD Plot Entropy",
            noise_type='heteroscedastic',
            func_type='sin'
        )
    
    def test_plot_uncertainties_undersampling(self, dummy_data):
        """Test undersampling plotting doesn't crash."""
        x_train, y_train, x_grid, y_clean = dummy_data
        
        mu_pred = np.sin(x_grid).squeeze()
        ale_var = np.ones(len(x_grid)) * 0.1
        epi_var = np.ones(len(x_grid)) * 0.05
        tot_var = ale_var + epi_var
        
        # Create region masks
        region_masks = [
            (x_grid.squeeze() >= -1) & (x_grid.squeeze() <= 1),
            (x_grid.squeeze() < -1) | (x_grid.squeeze() > 1)
        ]
        sampling_regions = [
            ((-1, 1), 1.0),  # Well-sampled
            ((-2, -1), 0.5)  # Undersampled (use actual tuple instead of None)
        ]
        
        plot_uncertainties_undersampling(
            x_train, y_train, x_grid, y_clean,
            mu_pred, ale_var, epi_var, tot_var,
            region_masks, sampling_regions,
            title="Test Undersampling Plot",
            noise_type='heteroscedastic',
            func_type='sin'
        )
    
    def test_plot_uncertainties_entropy_undersampling(self, dummy_data):
        """Test entropy undersampling plotting doesn't crash."""
        x_train, y_train, x_grid, y_clean = dummy_data
        
        mu_pred = np.sin(x_grid).squeeze()
        ale_entropy = np.ones(len(x_grid)) * 0.5
        epi_entropy = np.ones(len(x_grid)) * 0.3
        tot_entropy = ale_entropy + epi_entropy
        
        # Create region masks
        region_masks = [
            (x_grid.squeeze() >= -1) & (x_grid.squeeze() <= 1),
            (x_grid.squeeze() < -1) | (x_grid.squeeze() > 1)
        ]
        sampling_regions = [
            ((-1, 1), 1.0),  # Well-sampled
            ((-2, -1), 0.5)  # Undersampled (use actual tuple instead of None)
        ]
        
        plot_uncertainties_entropy_undersampling(
            x_train, y_train, x_grid, y_clean,
            mu_pred, ale_entropy, epi_entropy, tot_entropy,
            region_masks, sampling_regions,
            title="Test Undersampling Plot Entropy",
            noise_type='heteroscedastic',
            func_type='sin'
        )

