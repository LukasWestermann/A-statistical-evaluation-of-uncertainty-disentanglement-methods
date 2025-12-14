import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from utils.device import get_device

# Get device
device = get_device()

# Import loss functions from MC_Dropout
from Models.MC_Dropout import gaussian_nll, beta_nll

# Deep Ensemble model and functions

base_seed = 42

# ----- Baseline regression model (Appendix B) -----
class BaselineRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(32, 1)       # Linear mean
        self.sigma_head = nn.Sequential(      # Softplus std; variance = sigma^2
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self.eps = 1e-6

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h) + self.eps
        var = sigma ** 2
        return mu, var

# ----- Training loop for a single model -----
def train_single_ensemble(model, loader, epochs=700, lr=1e-3, loss_type='nll', beta=0.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, var = model(xb)
            if loss_type == 'nll':
                loss = gaussian_nll(yb, mu, var).mean()
            elif loss_type == 'beta_nll':
                loss = beta_nll(yb, mu, var, beta=beta).mean()
            else:
                raise ValueError("loss_type must be 'nll' or 'beta_nll'")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        # Optional: print every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"[{loss_type}] Epoch {epoch+1}/{epochs} - avg loss {total_loss/len(loader.dataset):.4f}")

# ----- Train an ensemble of K models -----
def train_ensemble_deep(x_train, y_train, batch_size=32, K=5, loss_type='nll', beta=0.5, parallel=True):
    """
    Train an ensemble of K models.
    
    Args:
        x_train: Training inputs
        y_train: Training targets
        batch_size: Batch size for training
        K: Number of ensemble members
        loss_type: 'nll' or 'beta_nll'
        beta: Beta parameter for beta-NLL loss
        parallel: If True, train ensemble members in parallel using ThreadPoolExecutor
    
    Returns:
        List of trained models
    """
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def train_member(k):
        """Train a single ensemble member."""
        model = BaselineRegressor()
        # Different seed per member to vary initialization and shuffling
        member_seed = base_seed + 1000 + k
        train_single_ensemble(model, loader, epochs=700, lr=1e-3, loss_type=loss_type, beta=beta, seed=member_seed)
        return model
    
    if parallel and K > 1:
        # Use ThreadPoolExecutor for parallel training
        # Note: Threads share the same GPU, which is fine for ensemble members
        max_workers = min(K, mp.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ensemble = list(executor.map(train_member, range(K)))
    else:
        # Sequential training
        ensemble = [train_member(k) for k in range(K)]
    
    return ensemble

# ----- Ensemble prediction and uncertainty decomposition -----
def ensemble_predict_deep(ensemble, x):
    x_t = torch.from_numpy(x).to(device)
    mus = []
    vars_ = []
    with torch.no_grad():
        for model in ensemble:
            model.eval()  # deterministic
            mu_i, var_i = model(x_t)
            mus.append(mu_i.cpu().numpy())
            vars_.append(var_i.cpu().numpy())

    mus = np.stack(mus, axis=0).squeeze(-1)    # [K, N]
    vars_ = np.stack(vars_, axis=0).squeeze(-1)  # [K, N]

    mu_pred = mus.mean(axis=0)             # predictive mean
    aleatoric_var = vars_.mean(axis=0)     # E[σ²] across ensemble
    epistemic_var = mus.var(axis=0)        # Var[μ] across ensemble
    total_var = aleatoric_var + epistemic_var
    return mu_pred, aleatoric_var, epistemic_var, total_var
