"""
Pytest configuration and shared fixtures for uncertainty estimation tests.
"""
import pytest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting tests

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Models.MC_Dropout import MCDropoutRegressor
from Models.Deep_Ensemble import train_ensemble_deep
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility across all tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield


@pytest.fixture
def dummy_data():
    """Create simple dummy data for testing."""
    n_train, n_grid = 20, 30
    x_train = np.random.randn(n_train, 1)
    y_train = x_train + np.random.randn(n_train, 1) * 0.1
    x_grid = np.linspace(-2, 2, n_grid).reshape(-1, 1)
    y_clean = np.sin(x_grid)
    
    return x_train, y_train, x_grid, y_clean


@pytest.fixture
def trained_mc_dropout_model():
    """Create a quickly trained MC Dropout model for testing."""
    model = MCDropoutRegressor(p=0.1)
    
    # Create dummy training data (ensure float32 for dtype consistency)
    x = torch.randn(30, 1, dtype=torch.float32)
    y = x + torch.randn(30, 1, dtype=torch.float32) * 0.1
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    
    # Quick training (just 3 epochs for speed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(3):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            mu, var = model(x_batch)
            loss = torch.nn.functional.gaussian_nll_loss(mu, y_batch, var)
            loss.backward()
            optimizer.step()
    
    return model


@pytest.fixture
def trained_deep_ensemble():
    """Create a quickly trained Deep Ensemble for testing."""
    # Note: train_ensemble_deep uses hardcoded epochs=700, lr=1e-3
    # For tests, we'll use minimal data and let it train (will be slow but works)
    # Alternatively, we could modify the function, but that's beyond test scope
    x = np.random.randn(30, 1).astype(np.float32)
    y = (x + np.random.randn(30, 1) * 0.1).astype(np.float32)
    
    # Use minimal K for speed
    ensemble = train_ensemble_deep(
        x, y, K=2, batch_size=10,
        loss_type='beta_nll', beta=0.5, parallel=False
    )
    
    return ensemble

