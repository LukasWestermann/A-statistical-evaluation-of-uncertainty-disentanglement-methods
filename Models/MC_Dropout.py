import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

# Add parent directory to path to import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.device import get_device

# Paper settings: MC-Dropout p=0.25, M=20 forward passes, NLL and β-NLL (β=0.5)
# Architecture from Appendix: two hidden layers (32 units, ReLU) + Dropout; μ head (Linear), σ head (Softplus)
# Training config: 700 epochs, batch size 32, Adam lr=1e-3
device = get_device()


# ----- Model per Appendix B (Regression) -----
class MCDropoutRegressor(nn.Module):
    def __init__(self, p=0.25):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p)
        )
        self.mu_head = nn.Linear(32, 1)  # Linear mean
        self.sigma_head = nn.Sequential( # Softplus std; we square to get variance
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self.eps = 1e-6

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h) + self.eps  # std > 0
        var = sigma ** 2
        return mu, var

# ----- Losses (Gaussian NLL and β-NLL with stop-grad on σ²) -----
def gaussian_nll(y, mu, var):
    # LNLL ≈ 0.5*log(σ²) + (y - μ)² / (2σ²)  (constant term omitted)
    return 0.5 * torch.log(var) + (y - mu) ** 2 / (2.0 * var)

def beta_nll(y, mu, var, beta=0.5):
    weight = (var.detach()) ** beta  # stop gradient
    return weight * gaussian_nll(y, mu, var)

# ----- Training loop -----
def train_model(model, loader, epochs=500, lr=1e-3, loss_type='nll', beta=0.5, device=None):
    if device is None:
        device = get_device()
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
        if (epoch + 1) % 100 == 0:
            print(f"[{loss_type}] Epoch {epoch+1}/{epochs} - avg loss {total_loss/len(loader.dataset):.4f}")

# ----- Input normalization utilities (for x only) -----
def normalize_x(x_train):
    """
    Compute normalization statistics for input x.
    Since we only normalize x (not y), predictions remain in original y scale.
    No back-transformation needed!
    
    Returns:
        x_mean: Mean of x_train
        x_std: Standard deviation of x_train
    """
    if isinstance(x_train, torch.Tensor):
        x_train = x_train.cpu().numpy()
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_std = np.std(x_train, axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0  # Avoid division by zero
    return x_mean, x_std

def normalize_x_data(x, x_mean, x_std):
    """
    Normalize input data using pre-computed statistics.
    
    Args:
        x: Input data to normalize (numpy array or torch tensor)
        x_mean: Mean used for normalization
        x_std: Std used for normalization
    
    Returns:
        Normalized x (same type as input)
    """
    if isinstance(x, torch.Tensor):
        x_norm = (x - torch.from_numpy(x_mean).to(x.device).to(x.dtype)) / torch.from_numpy(x_std).to(x.device).to(x.dtype)
        return x_norm
    else:
        return (x - x_mean) / x_std

# ----- MC-Dropout sampling and uncertainty decomposition -----
def mc_dropout_predict(model, x, M=100, device=None, return_raw_arrays=False):
    # Keep dropout active at inference by using train() but without gradients
    if device is None:
        device = get_device()
    model.to(device)
    model.train()
    x_t = torch.from_numpy(x).to(device)
    mus = []
    vars_ = []
    with torch.no_grad():
        for _ in range(M):
            mu_i, var_i = model(x_t)
            mus.append(mu_i.cpu().numpy())
            vars_.append(var_i.cpu().numpy())

    mus = np.stack(mus, axis=0).squeeze(-1)   # shape: [M, N]
    vars_ = np.stack(vars_, axis=0).squeeze(-1)  # shape: [M, N]

    mu_pred = mus.mean(axis=0)             # predictive mean μ*
    aleatoric_var = vars_.mean(axis=0)     # E[σ²] (aleatoric)
    epistemic_var = mus.var(axis=0)        # Var[μ] (epistemic)
    total_var = aleatoric_var + epistemic_var

    if return_raw_arrays:
        return mu_pred, aleatoric_var, epistemic_var, total_var, (mus, vars_)
    else:
        return mu_pred, aleatoric_var, epistemic_var, total_var

# ----- Plotting -----
def plot_toy_data(x_train, y_train, x_grid, y_clean, title="Toy Regression Data"):
    """Plot the training data and clean function"""
    plt.figure(figsize=(12, 6))
    
    # Plot training data points
    plt.scatter(x_train, y_train, alpha=0.6, s=20, label="Training data", color='blue')
    
    # Plot clean function
    plt.plot(x_grid, y_clean, 'r--', linewidth=2, label="Clean f(x) = 0.7x + 0.5")
    
    # Add vertical line to separate training and OOD regions
    plt.axvline(x=10, color='black', linestyle=':', alpha=0.7, linewidth=1)
    plt.text(5, plt.ylim()[1]*0.9, "Training (0-10)", ha="center", va="center", fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(12.5, plt.ylim()[1]*0.9, "OOD (10-15)", ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_uncertainties(x_grid, y_clean, mu_pred, ale_var, epi_var, tot_var, ood_mask, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    x = x_grid[:, 0]
    
    # Split data into training and OOD regions
    train_mask = ~ood_mask
    ood_mask_bool = ood_mask
    
    # Plot 1: Predictive mean + Total uncertainty
    axes[0].plot(x[train_mask], mu_pred[train_mask], 'b-', linewidth=2, label="Predictive mean (train)")
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b--', linewidth=2, label="Predictive mean (OOD)")
    axes[0].fill_between(x[train_mask], mu_pred[train_mask] - np.sqrt(tot_var[train_mask]), 
                        mu_pred[train_mask] + np.sqrt(tot_var[train_mask]), 
                        alpha=0.3, color='blue', label="±σ(total) train")
    axes[0].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - np.sqrt(tot_var[ood_mask_bool]), 
                        mu_pred[ood_mask_bool] + np.sqrt(tot_var[ood_mask_bool]), 
                        alpha=0.3, color='lightblue', label="±σ(total) OOD")
    axes[0].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean f(x) = 0.7x + 0.5")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{title}: Predictive Mean + Total Uncertainty")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictive mean + Aleatoric uncertainty only
    axes[1].plot(x[train_mask], mu_pred[train_mask], 'b-', linewidth=2, label="Predictive mean (train)")
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b--', linewidth=2, label="Predictive mean (OOD)")
    axes[1].fill_between(x[train_mask], mu_pred[train_mask] - np.sqrt(ale_var[train_mask]), 
                        mu_pred[train_mask] + np.sqrt(ale_var[train_mask]), 
                        alpha=0.3, color='green', label="±σ(aleatoric) train")
    axes[1].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - np.sqrt(ale_var[ood_mask_bool]), 
                        mu_pred[ood_mask_bool] + np.sqrt(ale_var[ood_mask_bool]), 
                        alpha=0.3, color='lightgreen', label="±σ(aleatoric) OOD")
    axes[1].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean f(x) = 0.7x + 0.5")
    axes[1].set_ylabel("y")
    axes[1].set_title(f"{title}: Predictive Mean + Aleatoric Uncertainty")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Predictive mean + Epistemic uncertainty only
    axes[2].plot(x[train_mask], mu_pred[train_mask], 'b-', linewidth=2, label="Predictive mean (train)")
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], 'b--', linewidth=2, label="Predictive mean (OOD)")
    axes[2].fill_between(x[train_mask], mu_pred[train_mask] - np.sqrt(epi_var[train_mask]), 
                        mu_pred[train_mask] + np.sqrt(epi_var[train_mask]), 
                        alpha=0.3, color='orange', label="±σ(epistemic) train")
    axes[2].fill_between(x[ood_mask_bool], mu_pred[ood_mask_bool] - np.sqrt(epi_var[ood_mask_bool]), 
                        mu_pred[ood_mask_bool] + np.sqrt(epi_var[ood_mask_bool]), 
                        alpha=0.3, color='moccasin', label="±σ(epistemic) OOD")
    axes[2].plot(x, y_clean[:, 0], 'r--', linewidth=1.5, alpha=0.8, label="Clean f(x) = 0.7x + 0.5")
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"{title}: Predictive Mean + Epistemic Uncertainty")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)
    
    # Add vertical line to separate training and OOD regions
    for ax in axes:
        ax.axvline(x=10, color='black', linestyle=':', alpha=0.7, linewidth=1)
        ax.text(5, ax.get_ylim()[1]*0.9, "Training (0-10)", ha="center", va="center", fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(12.5, ax.get_ylim()[1]*0.9, "OOD (10-15)", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    plt.tight_layout()
    plt.show()
