"""
Classification models and training utilities for IT and GL variants.
This module is separate from regression Models/* and does not modify them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive

from utils.device import get_device


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)


def _ensure_class_indices(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    raise ValueError("y must be class indices or one-hot encoded.")


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_idx = _ensure_class_indices(y)
    return np.eye(num_classes, dtype=np.float32)[y_idx]


def sampling_softmax_torch(mu: torch.Tensor, sigma2: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
    """
    Sampling softmax for GL training: z ~ N(mu, diag(sigma2)), p = mean softmax(z).
    mu, sigma2: [N, K]
    """
    eps = torch.randn((n_samples, mu.shape[0], mu.shape[1]), device=mu.device, dtype=mu.dtype)
    z = mu.unsqueeze(0) + torch.sqrt(sigma2.unsqueeze(0)) * eps
    p = torch.softmax(z, dim=-1).mean(dim=0)
    return p


def sampling_softmax_np(mu: np.ndarray, sigma2: np.ndarray, n_samples: int = 100, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    eps = rng.standard_normal(size=(n_samples, mu.shape[0], mu.shape[1])).astype(np.float32)
    z = mu[None, :, :] + np.sqrt(sigma2[None, :, :]) * eps
    z = z - np.max(z, axis=2, keepdims=True)
    p = np.exp(z)
    p = p / np.sum(p, axis=2, keepdims=True)
    return p.mean(axis=0)


def cross_entropy_from_probs(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy from probabilities.
    y can be class indices [N] or one-hot [N, K].
    """
    eps = 1e-8
    if y.ndim == 1:
        logp = torch.log(probs + eps)
        return -logp[torch.arange(y.shape[0], device=y.device), y].mean()
    if y.ndim == 2:
        logp = torch.log(probs + eps)
        return -(y * logp).sum(dim=1).mean()
    raise ValueError("y must be class indices or one-hot encoded.")


# ---------------------------
# MC Dropout classifiers
# ---------------------------

class MCDropoutClassifierIT(nn.Module):
    def __init__(self, input_dim: int = 2, num_classes: int = 3, p: float = 0.25):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.logits_head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_head(self.trunk(x))


class MCDropoutClassifierGL(nn.Module):
    def __init__(self, input_dim: int = 2, num_classes: int = 3, p: float = 0.25):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.mu_head = nn.Linear(32, num_classes)
        self.sigma_head = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = self.mu_head(h)
        sigma2 = self.sigma_head(h) ** 2 + 1e-6
        return mu, sigma2


def train_mc_dropout_it(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 2,
    num_classes: int = 3,
    p: float = 0.25,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> MCDropoutClassifierIT:
    device = get_device()
    y_idx = _ensure_class_indices(y_train)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_idx))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = MCDropoutClassifierIT(input_dim=input_dim, num_classes=num_classes, p=p).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 100 == 0:
            print(f"[MC Dropout IT] Epoch {epoch+1}/{epochs} - avg loss {total_loss/len(loader.dataset):.4f}")
    return model


def train_mc_dropout_gl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 2,
    num_classes: int = 3,
    p: float = 0.25,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_samples: int = 100,
) -> MCDropoutClassifierGL:
    device = get_device()
    y_idx = _ensure_class_indices(y_train)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_idx))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = MCDropoutClassifierGL(input_dim=input_dim, num_classes=num_classes, p=p).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mu, sigma2 = model(xb)
            probs = sampling_softmax_torch(mu, sigma2, n_samples=n_samples)
            loss = cross_entropy_from_probs(probs, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 100 == 0:
            print(f"[MC Dropout GL] Epoch {epoch+1}/{epochs} - avg loss {total_loss/len(loader.dataset):.4f}")
    return model


def mc_dropout_predict_it(
    model: MCDropoutClassifierIT,
    x: np.ndarray,
    mc_samples: int = 100,
) -> np.ndarray:
    device = get_device()
    model.to(device)
    model.train()
    x_t = torch.from_numpy(x).to(device)
    probs_members = []
    with torch.no_grad():
        for _ in range(mc_samples):
            logits = model(x_t)
            probs = torch.softmax(logits, dim=-1)
            probs_members.append(probs.cpu().numpy())
    return np.stack(probs_members, axis=0)  # [M, N, K]


def mc_dropout_predict_gl(
    model: MCDropoutClassifierGL,
    x: np.ndarray,
    mc_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    device = get_device()
    model.to(device)
    model.train()
    x_t = torch.from_numpy(x).to(device)
    mu_members = []
    sigma2_members = []
    with torch.no_grad():
        for _ in range(mc_samples):
            mu, sigma2 = model(x_t)
            mu_members.append(mu.cpu().numpy())
            sigma2_members.append(sigma2.cpu().numpy())
    return np.stack(mu_members, axis=0), np.stack(sigma2_members, axis=0)  # [M, N, K]


# ---------------------------
# Deep ensemble classifiers
# ---------------------------

class DeepEnsembleClassifierIT(nn.Module):
    def __init__(self, input_dim: int = 2, num_classes: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.logits_head = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_head(self.trunk(x))


class DeepEnsembleClassifierGL(nn.Module):
    def __init__(self, input_dim: int = 2, num_classes: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(32, num_classes)
        self.sigma_head = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        mu = self.mu_head(h)
        sigma2 = self.sigma_head(h) ** 2 + 1e-6
        return mu, sigma2


def _train_single_it(model: nn.Module, loader: DataLoader, epochs: int, lr: float, seed: Optional[int]) -> nn.Module:
    device = get_device()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
    return model


def _train_single_gl(model: nn.Module, loader: DataLoader, epochs: int, lr: float, seed: Optional[int], n_samples: int) -> nn.Module:
    device = get_device()
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
            mu, sigma2 = model(xb)
            probs = sampling_softmax_torch(mu, sigma2, n_samples=n_samples)
            loss = cross_entropy_from_probs(probs, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
    return model


def train_deep_ensemble_it(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 2,
    num_classes: int = 3,
    K: int = 10,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> List[DeepEnsembleClassifierIT]:
    y_idx = _ensure_class_indices(y_train)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_idx))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ensemble = []
    for k in range(K):
        member = DeepEnsembleClassifierIT(input_dim=input_dim, num_classes=num_classes)
        ensemble.append(_train_single_it(member, loader, epochs, lr, seed + k))
    return ensemble


def train_deep_ensemble_gl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 2,
    num_classes: int = 3,
    K: int = 10,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    n_samples: int = 100,
) -> List[DeepEnsembleClassifierGL]:
    y_idx = _ensure_class_indices(y_train)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_idx))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ensemble = []
    for k in range(K):
        member = DeepEnsembleClassifierGL(input_dim=input_dim, num_classes=num_classes)
        ensemble.append(_train_single_gl(member, loader, epochs, lr, seed + k, n_samples))
    return ensemble


def ensemble_predict_it(ensemble: List[nn.Module], x: np.ndarray) -> np.ndarray:
    device = get_device()
    x_t = torch.from_numpy(x).to(device)
    probs_members = []
    with torch.no_grad():
        for model in ensemble:
            model.to(device)
            model.eval()
            logits = model(x_t)
            probs = torch.softmax(logits, dim=-1)
            probs_members.append(probs.cpu().numpy())
    return np.stack(probs_members, axis=0)  # [K, N, K]


def ensemble_predict_gl(ensemble: List[nn.Module], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    device = get_device()
    x_t = torch.from_numpy(x).to(device)
    mu_members = []
    sigma2_members = []
    with torch.no_grad():
        for model in ensemble:
            model.to(device)
            model.eval()
            mu, sigma2 = model(x_t)
            mu_members.append(mu.cpu().numpy())
            sigma2_members.append(sigma2.cpu().numpy())
    return np.stack(mu_members, axis=0), np.stack(sigma2_members, axis=0)


# ---------------------------
# BNN classifiers (Pyro)
# ---------------------------

def _bnn_forward_logits(x, W1, b1, W2, b2, W_mu, b_mu, W_rho=None, b_rho=None):
    h1 = torch.relu(x @ W1 + b1)
    h2 = torch.relu(h1 @ W2 + b2)
    mu = h2 @ W_mu + b_mu
    if W_rho is None:
        return mu
    rho = h2 @ W_rho + b_rho
    sigma2 = torch.nn.functional.softplus(rho) ** 2 + 1e-6
    return mu, sigma2


def bnn_classification_it(x, y=None, hidden_width=32, weight_scale=1.0, num_classes=3):
    N = x.shape[0]
    H = hidden_width
    W1 = pyro.sample("W1", dist.Normal(0, weight_scale).expand([x.shape[1], H]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0, weight_scale).expand([H]).to_event(1))
    W2 = pyro.sample("W2", dist.Normal(0, weight_scale).expand([H, H]).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(0, weight_scale).expand([H]).to_event(1))
    W_mu = pyro.sample("W_mu", dist.Normal(0, weight_scale).expand([H, num_classes]).to_event(2))
    b_mu = pyro.sample("b_mu", dist.Normal(0, weight_scale).expand([num_classes]).to_event(1))

    logits = _bnn_forward_logits(x, W1, b1, W2, b2, W_mu, b_mu)
    pyro.deterministic("logits", logits)
    with pyro.plate("data", N):
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y)


def bnn_classification_gl(x, y=None, hidden_width=32, weight_scale=1.0, num_classes=3):
    N = x.shape[0]
    H = hidden_width
    W1 = pyro.sample("W1", dist.Normal(0, weight_scale).expand([x.shape[1], H]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0, weight_scale).expand([H]).to_event(1))
    W2 = pyro.sample("W2", dist.Normal(0, weight_scale).expand([H, H]).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(0, weight_scale).expand([H]).to_event(1))
    W_mu = pyro.sample("W_mu", dist.Normal(0, weight_scale).expand([H, num_classes]).to_event(2))
    b_mu = pyro.sample("b_mu", dist.Normal(0, weight_scale).expand([num_classes]).to_event(1))
    W_rho = pyro.sample("W_rho", dist.Normal(0, weight_scale).expand([H, num_classes]).to_event(2))
    b_rho = pyro.sample("b_rho", dist.Normal(0, weight_scale).expand([num_classes]).to_event(1))

    mu, sigma2 = _bnn_forward_logits(x, W1, b1, W2, b2, W_mu, b_mu, W_rho, b_rho)
    pyro.deterministic("mu", mu)
    pyro.deterministic("sigma2", sigma2)

    z = pyro.sample("z", dist.Normal(mu, torch.sqrt(sigma2)).to_event(1))
    with pyro.plate("data", N):
        pyro.sample("obs", dist.Categorical(logits=z), obs=y)


def train_bnn_it(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    chains: int = 1,
    seed: int = 42,
):
    pyro.set_rng_seed(seed)
    x_t = torch.from_numpy(x_train).float()
    y_idx = torch.from_numpy(_ensure_class_indices(y_train)).long()
    nuts_kernel = NUTS(bnn_classification_it, target_accept_prob=0.8)
    mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=warmup, num_chains=chains)
    mcmc.run(x=x_t, y=y_idx, hidden_width=hidden_width, weight_scale=weight_scale, num_classes=int(y_idx.max().item() + 1))
    return mcmc


def train_bnn_gl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    warmup: int = 200,
    samples: int = 200,
    chains: int = 1,
    seed: int = 42,
):
    pyro.set_rng_seed(seed)
    x_t = torch.from_numpy(x_train).float()
    y_idx = torch.from_numpy(_ensure_class_indices(y_train)).long()
    nuts_kernel = NUTS(bnn_classification_gl, target_accept_prob=0.8)
    mcmc = MCMC(nuts_kernel, num_samples=samples, warmup_steps=warmup, num_chains=chains)
    mcmc.run(x=x_t, y=y_idx, hidden_width=hidden_width, weight_scale=weight_scale, num_classes=int(y_idx.max().item() + 1))
    return mcmc


def bnn_predict_it(
    mcmc,
    x: np.ndarray,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    num_classes: int = 3,
) -> np.ndarray:
    x_t = torch.from_numpy(x).float()
    samples = mcmc.get_samples()
    predictive = Predictive(bnn_classification_it, posterior_samples=samples, return_sites=("logits",))
    preds = predictive(x=x_t, hidden_width=hidden_width, weight_scale=weight_scale, num_classes=num_classes)
    logits = preds["logits"].detach().cpu().numpy()  # [S, N, K]
    logits = logits - logits.max(axis=2, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=2, keepdims=True)
    return probs


def bnn_predict_gl(
    mcmc,
    x: np.ndarray,
    hidden_width: int = 32,
    weight_scale: float = 1.0,
    num_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    x_t = torch.from_numpy(x).float()
    samples = mcmc.get_samples()
    predictive = Predictive(bnn_classification_gl, posterior_samples=samples, return_sites=("mu", "sigma2"))
    preds = predictive(x=x_t, hidden_width=hidden_width, weight_scale=weight_scale, num_classes=num_classes)
    mu = preds["mu"].detach().cpu().numpy()       # [S, N, K]
    sigma2 = preds["sigma2"].detach().cpu().numpy()  # [S, N, K]
    return mu, sigma2
