"""
Quick smoke tests for experiment pipelines.
Uses minimal data/iterations for speed - goal is to catch integration errors.
"""
import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Models.MC_Dropout import MCDropoutRegressor, train_model, mc_dropout_predict
from utils.entropy_uncertainty import entropy_uncertainty_analytical
from utils.ood_experiments import generate_data_with_ood
from torch.utils.data import TensorDataset, DataLoader


class TestQuickPipeline:
    """Quick smoke tests for full pipelines."""
    
    def test_ood_pipeline_minimal(self):
        """Test OOD experiment pipeline with minimal data."""
        def dummy_generate(n_train, train_range, grid_points, noise_type, type):
            x_train = np.random.uniform(train_range[0], train_range[1], (n_train, 1))
            y_train = x_train + np.random.randn(n_train, 1) * 0.1
            x_grid = np.linspace(train_range[0]-5, train_range[1]+5, grid_points).reshape(-1, 1)
            y_clean = x_grid
            return x_train, y_train, x_grid, y_clean
        
        # Generate minimal data
        x_train, y_train, x_grid, y_grid_clean, ood_mask = generate_data_with_ood(
            dummy_generate, n_train=20, train_range=(-2, 2),
            ood_ranges=[(5, 7)], grid_points=30, noise_type='heteroscedastic',
            func_type='linear', seed=42
        )
        
        # Ensure float32 for dtype consistency
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        x_grid = x_grid.astype(np.float32)
        
        # Quick training
        model = MCDropoutRegressor(p=0.1)
        ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=10, shuffle=True)
        
        # Just 2 epochs for speed
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(2):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                mu, var = model(x_batch)
                loss = torch.nn.functional.gaussian_nll_loss(mu, y_batch, var)
                loss.backward()
                optimizer.step()
        
        # Predict with raw arrays
        result = mc_dropout_predict(model, x_grid, M=3, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy
        entropy_result = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        
        # Check that everything computed correctly
        assert np.all(np.isfinite(mu_pred))
        assert np.all(np.isfinite(ale_var))
        assert np.all(np.isfinite(epi_var))
        assert np.all(np.isfinite(entropy_result['aleatoric']))
        assert np.all(np.isfinite(entropy_result['epistemic']))
        
        # Check shapes
        assert mu_pred.shape == (len(x_grid),)
        assert entropy_result['aleatoric'].shape == (len(x_grid),)
        
        # Check OOD mask
        assert ood_mask.shape == (len(x_grid),)
        assert np.sum(ood_mask) > 0  # Should have some OOD points
    
    def test_sample_size_pipeline_minimal(self):
        """Test sample size experiment pipeline with minimal data."""
        # Generate data (ensure float32)
        n_train_full = 50
        x_train_full = np.random.randn(n_train_full, 1).astype(np.float32)
        y_train_full = (x_train_full + np.random.randn(n_train_full, 1) * 0.1).astype(np.float32)
        x_grid = np.linspace(-2, 2, 20).reshape(-1, 1).astype(np.float32)
        y_grid_clean = x_grid
        
        # Test with minimal percentage
        pct = 20  # 20% of data
        n_samples = int(n_train_full * pct / 100)
        indices = np.random.choice(n_train_full, size=n_samples, replace=False)
        x_train_subset = x_train_full[indices]
        y_train_subset = y_train_full[indices]
        
        # Quick training
        model = MCDropoutRegressor(p=0.1)
        ds = TensorDataset(torch.from_numpy(x_train_subset), torch.from_numpy(y_train_subset))
        loader = DataLoader(ds, batch_size=10, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(2):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                mu, var = model(x_batch)
                loss = torch.nn.functional.gaussian_nll_loss(mu, y_batch, var)
                loss.backward()
                optimizer.step()
        
        # Predict with raw arrays
        result = mc_dropout_predict(model, x_grid, M=3, return_raw_arrays=True)
        mu_pred, ale_var, epi_var, tot_var, (mu_samples, sigma2_samples) = result
        
        # Compute entropy
        entropy_result = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
        
        # Verify everything works
        assert np.all(np.isfinite(mu_pred))
        assert np.all(np.isfinite(entropy_result['aleatoric']))
        assert len(mu_pred) == len(x_grid)

