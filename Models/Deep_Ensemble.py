import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.device import get_device

# Get device
device = get_device()

# Import loss functions from MC_Dropout
from Models.MC_Dropout import gaussian_nll, beta_nll

# Deep Ensemble model and functions

base_seed = 42

# Stride between outer seeds, chosen far larger than any plausible K so that adjacent
# seeds cannot share member streams. The naive `seed + 1000 + k` would give seed 42's
# member 1 and seed 43's member 0 the identical seed 1043 -- across seeds 42..46 that
# collapses most of the per-seed ensemble variability seed replication exists to measure.
_SEED_STRIDE = 10_000


def _member_seed(seed, k):
    """Init/shuffle seed for ensemble member `k`.

    `seed=None` reproduces the historical `base_seed + 1000 + k` stream byte-for-byte,
    so every pre-existing caller -- and every raw_outputs.npz already on disk -- is
    unaffected. An explicit seed switches to a strided stream that keeps nearby outer
    seeds independent.
    """
    if seed is None:
        return base_seed + 1000 + k
    return int(seed) * _SEED_STRIDE + 1000 + k


# ----- Baseline regression model (Appendix B) -----
class BaselineRegressor(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 32),
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

# ----- Vectorized ensemble module (training only) -----
class _BatchedEnsemble(nn.Module):
    """K BaselineRegressor members stacked along a leading dim and trained with one
    batched matmul per layer instead of a Python loop or threads over K models.
    There is no mixing between members (each uses only its own weight slice), so a
    single Adam optimizer over the stacked parameters is equivalent to K independent
    optimizers: Adam's moment estimates and updates are elementwise per parameter."""

    def __init__(self, K, input_dim=1, hidden=32, seed=None):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.w1 = nn.Parameter(torch.empty(K, hidden, input_dim))
        self.b1 = nn.Parameter(torch.empty(K, hidden))
        self.w2 = nn.Parameter(torch.empty(K, hidden, hidden))
        self.b2 = nn.Parameter(torch.empty(K, hidden))
        self.w_mu = nn.Parameter(torch.empty(K, 1, hidden))
        self.b_mu = nn.Parameter(torch.empty(K, 1))
        self.w_sigma = nn.Parameter(torch.empty(K, 1, hidden))
        self.b_sigma = nn.Parameter(torch.empty(K, 1))
        self.eps = 1e-6
        self._init_members(input_dim, hidden, seed=seed)

    def _init_members(self, input_dim, hidden, seed=None):
        # Build each member's layers with nn.Linear's own default init, seeded the
        # same way each ensemble member has always been seeded, so per-member
        # initialization diversity matches training K separate models. See
        # _member_seed for how an explicit `seed` decorrelates outer seed replicates.
        with torch.no_grad():
            for k in range(self.K):
                torch.manual_seed(_member_seed(seed, k))
                l1, l2 = nn.Linear(input_dim, hidden), nn.Linear(hidden, hidden)
                lmu, lsig = nn.Linear(hidden, 1), nn.Linear(hidden, 1)
                self.w1[k].copy_(l1.weight); self.b1[k].copy_(l1.bias)
                self.w2[k].copy_(l2.weight); self.b2[k].copy_(l2.bias)
                self.w_mu[k].copy_(lmu.weight); self.b_mu[k].copy_(lmu.bias)
                self.w_sigma[k].copy_(lsig.weight); self.b_sigma[k].copy_(lsig.bias)

    def forward(self, x):
        # x: [K, B, input_dim]
        h = torch.relu(torch.bmm(x, self.w1.transpose(1, 2)) + self.b1.unsqueeze(1))
        h = torch.relu(torch.bmm(h, self.w2.transpose(1, 2)) + self.b2.unsqueeze(1))
        mu = torch.bmm(h, self.w_mu.transpose(1, 2)) + self.b_mu.unsqueeze(1)
        sigma = F.softplus(torch.bmm(h, self.w_sigma.transpose(1, 2)) + self.b_sigma.unsqueeze(1)) + self.eps
        return mu, sigma ** 2

    def unpack(self):
        """Split the trained batched weights into K standalone BaselineRegressor
        models, so callers (ensemble_predict_deep, tests) see the same list-of-models
        interface as before."""
        models = []
        with torch.no_grad():
            for k in range(self.K):
                m = BaselineRegressor(input_dim=self.input_dim)
                m.trunk[0].weight.copy_(self.w1[k]); m.trunk[0].bias.copy_(self.b1[k])
                m.trunk[2].weight.copy_(self.w2[k]); m.trunk[2].bias.copy_(self.b2[k])
                m.mu_head.weight.copy_(self.w_mu[k]); m.mu_head.bias.copy_(self.b_mu[k])
                m.sigma_head[0].weight.copy_(self.w_sigma[k]); m.sigma_head[0].bias.copy_(self.b_sigma[k])
                models.append(m.to(device))
        return models

# ----- Train an ensemble of K models -----
def train_ensemble_deep(x_train, y_train, batch_size=32, K=30, loss_type='nll', beta=0.5, parallel=True, epochs=500, input_dim=1, seed=None):
    """
    Train an ensemble of K models.

    Members are trained as one batched module (stacked weights, single Adam
    optimizer) instead of K sequential or multi-threaded model instances. This
    gives real GPU-parallel training across members on any backend, including
    Apple's MPS, where concurrent multi-threaded kernel submission crashes the
    process. `parallel` is kept for backward compatibility but no longer changes
    behavior.

    Args:
        x_train: Training inputs
        y_train: Training targets
        batch_size: Batch size for training
        K: Number of ensemble members
        loss_type: 'nll' or 'beta_nll'
        beta: Beta parameter for beta-NLL loss
        input_dim: Input dimension (1 for univariate, 2 for OVB with X and Z)
        seed: Optional base seed for member initialization and per-member batch
            order. Both were previously fixed to the module-level base_seed=42,
            which made every ensemble bit-identical regardless of what the caller
            seeded -- seed replicates need this to vary. None preserves the
            historical behavior.

    Returns:
        List of trained models
    """
    if loss_type not in ('nll', 'beta_nll'):
        raise ValueError("loss_type must be 'nll' or 'beta_nll'")

    N = x_train.shape[0]
    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)

    ensemble = _BatchedEnsemble(K, input_dim=input_dim, seed=seed).to(device)
    opt = optim.Adam(ensemble.parameters(), lr=1e-3)

    # Each member gets its own permutation stream, seeded the same way its
    # `torch.manual_seed(member_seed)` call used to be, so per-member batch order
    # stays independent across epochs instead of every member sharing one shuffle.
    perm_generators = [torch.Generator().manual_seed(_member_seed(seed, k)) for k in range(K)]

    for epoch in range(epochs):
        batch_indices = torch.stack([torch.randperm(N, generator=g) for g in perm_generators])  # [K, N]
        total_loss, n_batches = 0.0, 0
        for start in range(0, N, batch_size):
            idx = batch_indices[:, start:start + batch_size].to(device)  # [K, b]
            xb, yb = x_t[idx], y_t[idx]  # [K, b, input_dim], [K, b, 1]
            mu, var = ensemble(xb)
            if loss_type == 'beta_nll':
                loss_per_elem = beta_nll(yb, mu, var, beta=beta)
            else:
                loss_per_elem = gaussian_nll(yb, mu, var)
            # Mean over each member's own batch, summed across members so the
            # per-member gradient magnitude matches training K independent models.
            loss = loss_per_elem.mean(dim=(1, 2)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() / K
            n_batches += 1
        if (epoch + 1) % 100 == 0:
            print(f"[{loss_type}] Epoch {epoch+1}/{epochs} - avg member loss {total_loss/n_batches:.4f}")

    return ensemble.unpack()

# ----- Ensemble prediction and uncertainty decomposition -----
def ensemble_predict_deep(ensemble, x, return_raw_arrays=False):
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
    
    if return_raw_arrays:
        return mu_pred, aleatoric_var, epistemic_var, total_var, (mus, vars_)
    else:
        return mu_pred, aleatoric_var, epistemic_var, total_var
