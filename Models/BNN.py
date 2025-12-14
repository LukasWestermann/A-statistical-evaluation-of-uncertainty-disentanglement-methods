# PyTorch + Pyro BNN with NUTS MCMC for regression with uncertainty decomposition
# Heteroscedastic likelihood: y ~ Normal(mu(x), sigma(x)), sigma(x) via Softplus
# Decomposition: aleatoric = E[sigma^2], epistemic = Var(mu) across posterior samples
#
# NOTE: Pyro MCMC has known GPU compatibility issues, so we use CPU for all operations.
# This ensures stability and avoids device mismatch errors.

import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import sys
from pathlib import Path

# Add parent directory to path to import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set Pyro random seed (can be overridden)
pyro.set_rng_seed(0)

# Use CPU for all Pyro operations (Pyro MCMC has GPU compatibility issues)
cpu_device = torch.device("cpu")


# ---------- Model Architecture ----------
def forward_nn(x, W1, b1, W2, b2, W_mu, b_mu, W_rho, b_rho):
    """
    Forward pass through the neural network.
    
    Args:
        x: Input tensor [N, 1]
        W1, b1, W2, b2: Hidden layer weights and biases
        W_mu, b_mu: Mean head weights and bias
        W_rho, b_rho: Log-scale head weights and bias
    
    Returns:
        mu: Predictive mean [N]
        sigma: Predictive std [N] (positive via softplus)
    """
    h1 = F.relu(x @ W1 + b1)           # [N, H]
    h2 = F.relu(h1 @ W2 + b2)          # [N, H]
    mu = h2 @ W_mu + b_mu              # [N, 1]
    rho = h2 @ W_rho + b_rho           # [N, 1]
    sigma = F.softplus(rho) + 1e-6     # [N, 1], positive std
    return mu.squeeze(-1), sigma.squeeze(-1)


def bnn_model(x, y=None, hidden_width=16, weight_scale=1.0):
    """
    Pyro BNN model with heteroscedastic noise.
    
    Args:
        x: Input tensor [N, 1] (on CPU)
        y: Target tensor [N] or None for prediction
        hidden_width: Width of hidden layers
        weight_scale: Scale of weight priors
    
    Returns:
        Samples from likelihood: y ~ Normal(mu(x), sigma(x))
    """
    N = x.shape[0]
    H = hidden_width

    # Priors on weights/biases (Gaussian) - all on CPU
    W1   = pyro.sample("W1",   dist.Normal(0, weight_scale).expand([1, H]).to_event(2))
    b1   = pyro.sample("b1",   dist.Normal(0, weight_scale).expand([H]).to_event(1))
    W2   = pyro.sample("W2",   dist.Normal(0, weight_scale).expand([H, H]).to_event(2))
    b2   = pyro.sample("b2",   dist.Normal(0, weight_scale).expand([H]).to_event(1))
    W_mu = pyro.sample("W_mu", dist.Normal(0, weight_scale).expand([H, 1]).to_event(2))
    b_mu = pyro.sample("b_mu", dist.Normal(0, weight_scale).expand([1]).to_event(1))
    W_rho= pyro.sample("W_rho",dist.Normal(0, weight_scale).expand([H, 1]).to_event(2))
    b_rho= pyro.sample("b_rho",dist.Normal(0, weight_scale).expand([1]).to_event(1))

    mu, sigma = forward_nn(x, W1, b1, W2, b2, W_mu, b_mu, W_rho, b_rho)  # [N], [N]
    pyro.deterministic("mu",   mu)
    pyro.deterministic("sigma",sigma)

    with pyro.plate("data", N):
        # Handle both 1D and 2D y arrays
        y_obs = y if y is None else (y.squeeze(-1) if y.ndim > 1 and y.shape[-1] == 1 else y)
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y_obs)


# ---------- MCMC Training ----------
def run_nuts(x_train_t, y_train_t, hidden_width=16, weight_scale=1.0,
             warmup=200, samples=200, chains=1, seed=None):
    """
    Run NUTS MCMC to sample from posterior.
    
    Args:
        x_train_t: Training inputs [N, 1] torch tensor (on CPU)
        y_train_t: Training targets [N] torch tensor (on CPU)
        hidden_width: Width of hidden layers
        weight_scale: Scale of weight priors
        warmup: Number of warmup steps
        samples: Number of posterior samples
        chains: Number of chains
        seed: Random seed (optional)
    
    Returns:
        MCMC object with posterior samples
    """
    if seed is not None:
        pyro.set_rng_seed(seed)
    
    nuts_kernel = NUTS(bnn_model, target_accept_prob=0.8)
    mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=warmup, num_chains=chains)
    mcmc.run(x=x_train_t, y=y_train_t, hidden_width=hidden_width, weight_scale=weight_scale)
    return mcmc


def train_bnn(x_train, y_train, hidden_width=16, weight_scale=1.0,
              warmup=200, samples=200, chains=1, seed=None):
    """
    Train BNN using MCMC.
    
    NOTE: Uses CPU for all operations due to Pyro MCMC GPU compatibility issues.
    
    Args:
        x_train: Training inputs [N, 1] numpy array
        y_train: Training targets [N, 1] or [N] numpy array
        hidden_width: Width of hidden layers
        weight_scale: Scale of weight priors
        warmup: Number of warmup steps
        samples: Number of posterior samples
        chains: Number of chains
        seed: Random seed (optional)
    
    Returns:
        Trained MCMC object
    """
    # Convert to torch tensors - use CPU for Pyro MCMC
    x_train_t = torch.from_numpy(x_train).to(cpu_device)
    
    # Handle 2D y arrays
    if y_train.ndim > 1 and y_train.shape[-1] == 1:
        y_train_t = torch.from_numpy(y_train.squeeze(-1)).to(cpu_device)
    else:
        y_train_t = torch.from_numpy(y_train).to(cpu_device)
    
    mcmc = run_nuts(x_train_t, y_train_t, hidden_width, weight_scale, 
                    warmup, samples, chains, seed)
    return mcmc


# ---------- Prediction ----------
def posterior_predictive(mcmc, x_new_t, hidden_width=16, weight_scale=1.0):
    """
    Generate posterior predictive samples.
    
    Args:
        mcmc: Trained MCMC object
        x_new_t: New inputs [M, 1] torch tensor (on CPU)
        hidden_width: Width of hidden layers (must match training)
        weight_scale: Scale of weight priors (must match training)
    
    Returns:
        Dictionary with 'mu', 'sigma', 'obs' arrays
    """
    samples = mcmc.get_samples()
    predictive = Predictive(bnn_model, posterior_samples=samples, return_sites=("mu","sigma","obs"))
    preds = predictive(x=x_new_t, hidden_width=hidden_width, weight_scale=weight_scale)
    return {k: v.detach().cpu().numpy() for k, v in preds.items()}


def decompose_uncertainty(mu_samples, sigma_samples):
    """
    Decompose uncertainty into aleatoric and epistemic components.
    
    Args:
        mu_samples: Samples of predictive mean [S, N] or [S, 1, N] or [S, N, 1]
        sigma_samples: Samples of predictive std [S, N] or [S, 1, N] or [S, N, 1]
    
    Returns:
        mu_mean: Mean predictive mean [N]
        aleatoric_var: Aleatoric variance E[σ²] [N]
        epistemic_var: Epistemic variance Var[μ] [N]
        total_var: Total variance [N]
    """
    # Handle various shape scenarios
    # Expected: [S, N] but Pyro may return [S, 1, N] or [S, N, 1]
    
    # Squeeze out singleton dimensions
    while mu_samples.ndim > 2:
        mu_samples = mu_samples.squeeze()
    while sigma_samples.ndim > 2:
        sigma_samples = sigma_samples.squeeze()
    
    # Ensure we have [S, N] shape
    if mu_samples.ndim == 1:
        # Single sample case - reshape to [1, N]
        mu_samples = mu_samples.reshape(1, -1)
    if sigma_samples.ndim == 1:
        sigma_samples = sigma_samples.reshape(1, -1)
    
    # If we have [N, S] instead of [S, N], transpose
    if mu_samples.shape[0] > mu_samples.shape[1] and mu_samples.shape[1] == 1:
        mu_samples = mu_samples.T
    if sigma_samples.shape[0] > sigma_samples.shape[1] and sigma_samples.shape[1] == 1:
        sigma_samples = sigma_samples.T
    
    # Now should be [S, N]
    # mu_samples: [S, N], sigma_samples: [S, N]
    aleatoric_var = (sigma_samples**2).mean(axis=0)   # E[σ²] - average variance across samples
    epistemic_var = mu_samples.var(axis=0)            # Var[μ] - variance of means across samples
    total_var = aleatoric_var + epistemic_var
    mu_mean = mu_samples.mean(axis=0)
    
    return mu_mean, aleatoric_var, epistemic_var, total_var


def bnn_predict(mcmc, x, hidden_width=16, weight_scale=1.0):
    """
    Make predictions with uncertainty decomposition.
    
    NOTE: Uses CPU for all operations (consistent with training).
    
    Args:
        mcmc: Trained MCMC object
        x: Input array [N, 1] numpy array
        hidden_width: Width of hidden layers (must match training)
        weight_scale: Scale of weight priors (must match training)
    
    Returns:
        mu_pred: Predictive mean [N]
        ale_var: Aleatoric variance [N]
        epi_var: Epistemic variance [N]
        tot_var: Total variance [N]
    """
    # Use CPU for prediction (since MCMC was on CPU)
    x_t = torch.from_numpy(x).to(cpu_device)
    preds = posterior_predictive(mcmc, x_t, hidden_width, weight_scale)
    
    mu_samps = preds["mu"]       # [S, N] or [S, 1, N]
    sigma_samps = preds["sigma"] # [S, N] or [S, 1, N]
    
    mu_pred, ale_var, epi_var, tot_var = decompose_uncertainty(mu_samps, sigma_samps)
    
    return mu_pred, ale_var, epi_var, tot_var


# ---------- Input Normalization (optional, like Deep Ensemble) ----------
def normalize_x(x_train):
    """
    Compute normalization statistics for input x.
    Compatible with MC_Dropout.normalize_x interface.
    
    Args:
        x_train: Training inputs [N, 1] numpy array
    
    Returns:
        x_mean: Mean of x_train [1, 1]
        x_std: Standard deviation of x_train [1, 1]
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
    Compatible with MC_Dropout.normalize_x_data interface.
    
    Args:
        x: Input data to normalize [N, 1] numpy array
        x_mean: Mean used for normalization [1, 1]
        x_std: Std used for normalization [1, 1]
    
    Returns:
        Normalized x [N, 1] numpy array
    """
    if isinstance(x, torch.Tensor):
        x_norm = (x - torch.from_numpy(x_mean).to(x.device).to(x.dtype)) / torch.from_numpy(x_std).to(x.device).to(x.dtype)
        return x_norm.cpu().numpy()
    else:
        return (x - x_mean) / x_std

