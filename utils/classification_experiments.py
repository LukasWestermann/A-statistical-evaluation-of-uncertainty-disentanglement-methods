"""
Classification experiment utilities for IT and GL uncertainty decompositions.

This module provides per-model experiment functions following the regression notebook pattern.
"""

from __future__ import annotations

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

import numpy as np

from utils.classification_data import simulate_dataset, simulate_rotation_ood_dataset, simulate_ring_ood_dataset
from utils.classification_models import (
    sampling_softmax_np,
    train_mc_dropout_it,
    train_mc_dropout_gl,
    mc_dropout_predict_it,
    mc_dropout_predict_gl,
    train_deep_ensemble_it,
    train_deep_ensemble_gl,
    ensemble_predict_it,
    ensemble_predict_gl,
    train_bnn_it,
    train_bnn_gl,
    bnn_predict_it,
    bnn_predict_gl,
)
from utils.classification_metrics import accuracy_from_probs, expected_calibration_error, roc_auc_score_manual
from utils.classification_plotting import plot_metric_curves, plot_uncertainty_heatmap, plot_misclassifications, plot_uncertainty_panel
from utils.results_save import save_statistics, save_classification_outputs


# ============================================================================
# Core uncertainty computation helpers
# ============================================================================

def entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute entropy of probability distributions."""
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe), axis=-1)


def _normalize_uncertainty(values: np.ndarray, vmin: float = None, vmax: float = None) -> np.ndarray:
    """Normalize uncertainty values to [0, 1] range.
    
    Args:
        values: Array of uncertainty values to normalize
        vmin: Minimum value for normalization (uses values.min() if None)
        vmax: Maximum value for normalization (uses values.max() if None)
    
    Returns:
        Normalized values in [0, 1] range
    """
    values = np.asarray(values)
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    if vmax - vmin == 0:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def it_uncertainty(probs_members: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute IT (Information-Theoretic) uncertainty decomposition.
    
    probs_members: [M, N, K] - M members, N samples, K classes
    Returns: dict with p_bar, TU (total), AU (aleatoric), EU (epistemic),
             plus normalized versions TU_norm, AU_norm, EU_norm in [0, 1]
    """
    p_bar = probs_members.mean(axis=0)
    tu = entropy(p_bar)
    au = entropy(probs_members).mean(axis=0)
    eu = tu - au
    
    # Normalize each uncertainty type independently to [0, 1]
    return {
        "p_bar": p_bar,
        "TU": tu, "AU": au, "EU": eu,
        "TU_norm": _normalize_uncertainty(tu),
        "AU_norm": _normalize_uncertainty(au),
        "EU_norm": _normalize_uncertainty(eu),
    }


def gl_uncertainty(
    mu_members: np.ndarray,
    sigma2_members: np.ndarray,
    n_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute GL (Gaussian Logit) uncertainty decomposition.
    
    mu_members, sigma2_members: [M, N, K]
    Returns: dict with p_ale, p_epi, AU (aleatoric), EU (epistemic),
             plus normalized versions AU_norm, EU_norm in [0, 1]
    """
    mu_bar = mu_members.mean(axis=0)
    sigma2_ale = sigma2_members.mean(axis=0)
    sigma2_epi = mu_members.var(axis=0)

    p_ale = sampling_softmax_np(mu_bar, sigma2_ale, n_samples=n_samples, rng=rng)
    p_epi = sampling_softmax_np(mu_bar, sigma2_epi, n_samples=n_samples, rng=rng)

    au = entropy(p_ale)
    eu = entropy(p_epi)
    
    # Normalize each uncertainty type independently to [0, 1]
    return {
        "p_ale": p_ale, "p_epi": p_epi,
        "AU": au, "EU": eu,
        "AU_norm": _normalize_uncertainty(au),
        "EU_norm": _normalize_uncertainty(eu),
    }


def _predictive_probs_it(probs_members: np.ndarray) -> np.ndarray:
    """Get predictive probabilities from IT model members."""
    return probs_members.mean(axis=0)


def _predictive_probs_gl(mu_members: np.ndarray, sigma2_members: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Get predictive probabilities from GL model members."""
    mu_bar = mu_members.mean(axis=0)
    sigma2_ale = sigma2_members.mean(axis=0)
    sigma2_epi = mu_members.var(axis=0)
    sigma2_tot = sigma2_ale + sigma2_epi
    return sampling_softmax_np(mu_bar, sigma2_tot, n_samples=n_samples)


def _evaluate_probs(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics from predicted probabilities.
    
    Filters out OOD samples (label == -1) before computing metrics.
    """
    # Filter out OOD samples (label == -1)
    valid_mask = y_true >= 0
    if not np.any(valid_mask):
        return {"accuracy": 0.0, "ece": 0.0}
    
    probs_valid = probs[valid_mask]
    y_true_valid = y_true[valid_mask]
    
    return {
        "accuracy": accuracy_from_probs(probs_valid, y_true_valid),
        "ece": expected_calibration_error(probs_valid, y_true_valid),
    }


# ============================================================================
# Training and prediction helpers
# ============================================================================

def _train_mc_dropout_it(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, cfg: Dict[str, Any]):
    """Train MC Dropout IT and return probs_members."""
    model = train_mc_dropout_it(
        X_train, y_train,
        input_dim=cfg.get("input_dim", 2),
        num_classes=cfg.get("num_classes", 3),
        p=cfg.get("dropout_p", 0.25),
        epochs=cfg.get("epochs", 300),
        lr=cfg.get("lr", 1e-3),
        batch_size=cfg.get("batch_size", 32)
    )
    return mc_dropout_predict_it(model, X_eval, mc_samples=cfg.get("mc_samples", 100))


def _train_mc_dropout_gl(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, cfg: Dict[str, Any]):
    """Train MC Dropout GL and return (mu_members, sigma2_members)."""
    model = train_mc_dropout_gl(
        X_train, y_train,
        input_dim=cfg.get("input_dim", 2),
        num_classes=cfg.get("num_classes", 3),
        p=cfg.get("dropout_p", 0.25),
        epochs=cfg.get("epochs", 300),
        lr=cfg.get("lr", 1e-3),
        batch_size=cfg.get("batch_size", 32),
        n_samples=cfg.get("gl_samples", 100)
    )
    return mc_dropout_predict_gl(model, X_eval, mc_samples=cfg.get("mc_samples", 100))


def _train_deep_ensemble_it(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, cfg: Dict[str, Any]):
    """Train Deep Ensemble IT and return probs_members."""
    ensemble = train_deep_ensemble_it(
        X_train, y_train,
        input_dim=cfg.get("input_dim", 2),
        num_classes=cfg.get("num_classes", 3),
        K=cfg.get("K", 10),
        epochs=cfg.get("epochs", 300),
        lr=cfg.get("lr", 1e-3),
        batch_size=cfg.get("batch_size", 32),
        seed=cfg.get("seed", 42)
    )
    return ensemble_predict_it(ensemble, X_eval)


def _train_deep_ensemble_gl(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, cfg: Dict[str, Any]):
    """Train Deep Ensemble GL and return (mu_members, sigma2_members)."""
    ensemble = train_deep_ensemble_gl(
        X_train, y_train,
        input_dim=cfg.get("input_dim", 2),
        num_classes=cfg.get("num_classes", 3),
        K=cfg.get("K", 10),
        epochs=cfg.get("epochs", 300),
        lr=cfg.get("lr", 1e-3),
        batch_size=cfg.get("batch_size", 32),
        seed=cfg.get("seed", 42),
        n_samples=cfg.get("gl_samples", 100)
    )
    return ensemble_predict_gl(ensemble, X_eval)


def _train_bnn_it(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, cfg: Dict[str, Any]):
    """Train BNN IT and return probs_members."""
    mcmc = train_bnn_it(
        X_train, y_train,
        hidden_width=cfg.get("hidden_width", 32),
        weight_scale=cfg.get("weight_scale", 1.0),
        warmup=cfg.get("warmup", 200),
        samples=cfg.get("samples", 200),
        chains=cfg.get("chains", 1),
        seed=cfg.get("seed", 42)
    )
    return bnn_predict_it(
        mcmc, X_eval,
        hidden_width=cfg.get("hidden_width", 32),
        weight_scale=cfg.get("weight_scale", 1.0),
        num_classes=cfg.get("num_classes", 3)
    )


def _train_bnn_gl(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, cfg: Dict[str, Any]):
    """Train BNN GL and return (mu_members, sigma2_members)."""
    mcmc = train_bnn_gl(
        X_train, y_train,
        hidden_width=cfg.get("hidden_width", 32),
        weight_scale=cfg.get("weight_scale", 1.0),
        warmup=cfg.get("warmup", 200),
        samples=cfg.get("samples", 200),
        chains=cfg.get("chains", 1),
        seed=cfg.get("seed", 42)
    )
    return bnn_predict_gl(
        mcmc, X_eval,
        hidden_width=cfg.get("hidden_width", 32),
        weight_scale=cfg.get("weight_scale", 1.0),
        num_classes=cfg.get("num_classes", 3)
    )


# ============================================================================
# Ring OOD comparison bar plot
# ============================================================================

def _plot_ring_gap_comparison(
    unc_train: Dict[str, np.ndarray],
    unc_gap: Dict[str, np.ndarray],
    au_eu_corr_train: float,
    au_eu_corr_gap: float,
    model_name: str,
    subfolder: str,
    is_it: bool = True,
):
    """Plot bar chart comparing Ring vs Gap uncertainty metrics.
    
    Args:
        unc_train: Uncertainty dict for training (ring) data
        unc_gap: Uncertainty dict for gap data
        au_eu_corr_train: AU-EU correlation for ring data
        au_eu_corr_gap: AU-EU correlation for gap data
        model_name: Name of the model
        subfolder: Subfolder for saving the plot
        is_it: True for IT models (include TU), False for GL models
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Prepare raw data
    if is_it:
        metrics_raw = ['Mean AU', 'Mean EU', 'Mean TU', 'AU-EU Corr']
        ring_values_raw = [
            unc_train['AU'].mean(),
            unc_train['EU'].mean(),
            unc_train['TU'].mean(),
            au_eu_corr_train,
        ]
        gap_values_raw = [
            unc_gap['AU'].mean(),
            unc_gap['EU'].mean(),
            unc_gap['TU'].mean(),
            au_eu_corr_gap,
        ]
        metrics_norm = ['Mean AU (norm)', 'Mean EU (norm)', 'Mean TU (norm)']
        ring_values_norm = [
            unc_train['AU_norm'].mean(),
            unc_train['EU_norm'].mean(),
            unc_train['TU_norm'].mean(),
        ]
        gap_values_norm = [
            unc_gap['AU_norm'].mean(),
            unc_gap['EU_norm'].mean(),
            unc_gap['TU_norm'].mean(),
        ]
    else:
        metrics_raw = ['Mean AU', 'Mean EU', 'AU-EU Corr']
        ring_values_raw = [
            unc_train['AU'].mean(),
            unc_train['EU'].mean(),
            au_eu_corr_train,
        ]
        gap_values_raw = [
            unc_gap['AU'].mean(),
            unc_gap['EU'].mean(),
            au_eu_corr_gap,
        ]
        metrics_norm = ['Mean AU (norm)', 'Mean EU (norm)']
        ring_values_norm = [
            unc_train['AU_norm'].mean(),
            unc_train['EU_norm'].mean(),
        ]
        gap_values_norm = [
            unc_gap['AU_norm'].mean(),
            unc_gap['EU_norm'].mean(),
        ]
    
    width = 0.35
    
    # Create 2-row subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
    
    # Top row: Raw values
    x_raw = np.arange(len(metrics_raw))
    bars1_raw = ax1.bar(x_raw - width/2, ring_values_raw, width, label='Ring (Training)', color='tab:blue', alpha=0.7)
    bars2_raw = ax1.bar(x_raw + width/2, gap_values_raw, width, label='Gap (OOD)', color='tab:red', alpha=0.7)
    
    ax1.set_ylabel('Value')
    ax1.set_title(f'{model_name} - Ring vs Gap Comparison (Raw Values)')
    ax1.set_xticks(x_raw)
    ax1.set_xticklabels(metrics_raw)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on raw bars
    for bar in bars1_raw:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    for bar in bars2_raw:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    # Bottom row: Normalized values
    x_norm = np.arange(len(metrics_norm))
    bars1_norm = ax2.bar(x_norm - width/2, ring_values_norm, width, label='Ring (Training)', color='tab:blue', alpha=0.7)
    bars2_norm = ax2.bar(x_norm + width/2, gap_values_norm, width, label='Gap (OOD)', color='tab:red', alpha=0.7)
    
    ax2.set_ylabel('Value')
    ax2.set_title(f'{model_name} - Ring vs Gap Comparison (Normalized Values)')
    ax2.set_xticks(x_norm)
    ax2.set_xticklabels(metrics_norm)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on normalized bars
    for bar in bars1_norm:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    for bar in bars2_norm:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    save_dir = Path("results") / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{model_name}_ring_gap_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================================
# Heatmap generation helper
# ============================================================================

def _plot_uncertainty_heatmaps(
    X_eval: np.ndarray,
    uncertainty: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    probs_pred: np.ndarray,
    model_name: str,
    experiment_name: str,
    subfolder: str,
    is_it: bool,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    grid_extent: tuple = None,
    grid_res: int = None,
):
    """Generate and save panel plot with all uncertainty heatmaps and misclassifications.
    
    Args:
        X_eval: Evaluation points [N, 2] for 2D heatmap
        uncertainty: Dict containing normalized uncertainty values (AU_norm, EU_norm, TU_norm)
        X_test: Test data coordinates [M, 2]
        y_test: True labels [M]
        probs_pred: Predicted probabilities [M, K]
        model_name: Name of the model (e.g., "mc_dropout_it")
        experiment_name: Name of the experiment condition (e.g., "rcd_3.0")
        subfolder: Subfolder for saving plots
        is_it: True for IT models (has TU), False for GL models
        X_train: Optional training data coordinates [L, 2] for overlay
        y_train: Optional training labels [L] for overlay coloring
        grid_extent: Tuple (x_min, x_max, y_min, y_max) for imshow extent
        grid_res: Grid resolution (e.g., 100 for 100x100)
    """
    plot_uncertainty_panel(
        X_eval=X_eval,
        uncertainty=uncertainty,
        X_test=X_test,
        y_true=y_test,
        probs_pred=probs_pred,
        model_name=model_name,
        experiment_name=experiment_name,
        subfolder=subfolder,
        is_it=is_it,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent,
        grid_res=grid_res,
    )


# ============================================================================
# Shared experiment logic
# ============================================================================

def _run_it_experiment_single(
    model_name: str,
    train_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Dict[str, Any],
    experiment_name: str,
    subfolder: str,
):
    """Run a single IT experiment: train, predict on grid, compute uncertainties, save."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} (IT) - {experiment_name}")
    print(f"{'='*60}")
    
    # Create 100x100 grid for visualization
    grid_res = 100
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    # Predict on grid for visualization
    probs_members = train_fn(X_train, y_train, grid, cfg)
    unc = it_uncertainty(probs_members)
    probs_pred = _predictive_probs_it(probs_members)
    
    # Metrics on training data (for reference)
    probs_members_train = train_fn(X_train, y_train, X_train, cfg)
    probs_pred_train = _predictive_probs_it(probs_members_train)
    metrics = _evaluate_probs(probs_pred_train, y_train)
    
    print(f"  Train Accuracy: {metrics['accuracy']:.4f}, ECE: {metrics['ece']:.4f}")
    print(f"  Mean TU: {unc['TU'].mean():.4f} (norm: {unc['TU_norm'].mean():.4f}), "
          f"AU: {unc['AU'].mean():.4f} (norm: {unc['AU_norm'].mean():.4f}), "
          f"EU: {unc['EU'].mean():.4f} (norm: {unc['EU_norm'].mean():.4f})")
    
    save_classification_outputs(
        {
            "probs_members": probs_members,
            "y_true": y_train,
            "x_eval": grid,
            "metrics": np.array(list(metrics.values()), dtype=np.float32),
        },
        model_name=model_name,
        experiment_name=experiment_name,
        subfolder=subfolder,
    )
    
    # Grid extent for imshow
    grid_extent = (x_min, x_max, y_min, y_max)
    
    # Generate and save panel plot with uncertainty heatmaps
    _plot_uncertainty_heatmaps(
        X_eval=grid,
        uncertainty=unc,
        X_test=X_train,
        y_test=y_train,
        probs_pred=probs_pred_train,
        model_name=model_name,
        experiment_name=experiment_name,
        subfolder=f"{subfolder}/heatmaps",
        is_it=True,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent,
        grid_res=grid_res,
    )
    
    # Compute AU-EU correlation
    au_eu_corr = np.corrcoef(unc["AU_norm"], unc["EU_norm"])[0, 1]
    
    return {
        "uncertainty": unc, 
        "metrics": metrics, 
        "probs_members": probs_members,
        "au_eu_correlation": au_eu_corr,
    }


def _run_gl_experiment_single(
    model_name: str,
    train_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Dict[str, Any],
    experiment_name: str,
    subfolder: str,
):
    """Run a single GL experiment: train, predict on grid, compute uncertainties, save."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} (GL) - {experiment_name}")
    print(f"{'='*60}")
    
    # Create 100x100 grid for visualization
    grid_res = 100
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    # Predict on grid for visualization
    mu_members, sigma2_members = train_fn(X_train, y_train, grid, cfg)
    unc = gl_uncertainty(mu_members, sigma2_members, n_samples=cfg.get("gl_samples", 100))
    probs_pred = _predictive_probs_gl(mu_members, sigma2_members, n_samples=cfg.get("gl_samples", 100))
    
    # Metrics on training data (for reference)
    mu_members_train, sigma2_members_train = train_fn(X_train, y_train, X_train, cfg)
    probs_pred_train = _predictive_probs_gl(mu_members_train, sigma2_members_train, n_samples=cfg.get("gl_samples", 100))
    metrics = _evaluate_probs(probs_pred_train, y_train)
    
    print(f"  Train Accuracy: {metrics['accuracy']:.4f}, ECE: {metrics['ece']:.4f}")
    print(f"  Mean AU: {unc['AU'].mean():.4f} (norm: {unc['AU_norm'].mean():.4f}), "
          f"EU: {unc['EU'].mean():.4f} (norm: {unc['EU_norm'].mean():.4f})")
    
    save_classification_outputs(
        {
            "mu_members": mu_members,
            "sigma2_members": sigma2_members,
            "y_true": y_train,
            "x_eval": grid,
            "metrics": np.array(list(metrics.values()), dtype=np.float32),
        },
        model_name=model_name,
        experiment_name=experiment_name,
        subfolder=subfolder,
    )
    
    # Grid extent for imshow
    grid_extent = (x_min, x_max, y_min, y_max)
    
    # Generate and save panel plot with uncertainty heatmaps
    _plot_uncertainty_heatmaps(
        X_eval=grid,
        uncertainty=unc,
        X_test=X_train,
        y_test=y_train,
        probs_pred=probs_pred_train,
        model_name=model_name,
        experiment_name=experiment_name,
        subfolder=f"{subfolder}/heatmaps",
        is_it=False,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent,
        grid_res=grid_res,
    )
    
    # Compute AU-EU correlation
    au_eu_corr = np.corrcoef(unc["AU_norm"], unc["EU_norm"])[0, 1]
    
    return {
        "uncertainty": unc, 
        "metrics": metrics, 
        "mu_members": mu_members, 
        "sigma2_members": sigma2_members,
        "au_eu_correlation": au_eu_corr,
    }


def _save_sweep_summary(
    results: Dict[Any, Dict[str, Any]],
    condition_name: str,
    model_name: str,
    is_it: bool,
    subfolder: str,
):
    """Save summary statistics and plots for a sweep experiment.
    
    Plots normalized uncertainties (in [0, 1] range) for consistency with regression experiments.
    """
    x_values = list(results.keys())
    acc = [results[v]["metrics"]["accuracy"] for v in x_values]
    ece = [results[v]["metrics"]["ece"] for v in x_values]
    
    # Extract AU-EU correlation for each condition
    au_eu_corr = [results[v]["au_eu_correlation"] for v in x_values]
    
    if is_it:
        # Raw values for statistics
        tu = [results[v]["uncertainty"]["TU"].mean() for v in x_values]
        au = [results[v]["uncertainty"]["AU"].mean() for v in x_values]
        eu = [results[v]["uncertainty"]["EU"].mean() for v in x_values]
        # Normalized values for plotting
        tu_norm = [results[v]["uncertainty"]["TU_norm"].mean() for v in x_values]
        au_norm = [results[v]["uncertainty"]["AU_norm"].mean() for v in x_values]
        eu_norm = [results[v]["uncertainty"]["EU_norm"].mean() for v in x_values]
        
        # Plot normalized uncertainties
        plot_metric_curves(x_values, {"TU_norm": tu_norm, "AU_norm": au_norm, "EU_norm": eu_norm}, 
                          f"{model_name} IT uncertainty (normalized) vs {condition_name}", condition_name, subfolder)
        summary = {
            condition_name: x_values, "Accuracy": acc, "ECE": ece,
            "TU": tu, "AU": au, "EU": eu,
            "TU_norm": tu_norm, "AU_norm": au_norm, "EU_norm": eu_norm,
            "AU_EU_corr": au_eu_corr,
        }
    else:
        # Raw values for statistics
        au = [results[v]["uncertainty"]["AU"].mean() for v in x_values]
        eu = [results[v]["uncertainty"]["EU"].mean() for v in x_values]
        # Normalized values for plotting
        au_norm = [results[v]["uncertainty"]["AU_norm"].mean() for v in x_values]
        eu_norm = [results[v]["uncertainty"]["EU_norm"].mean() for v in x_values]
        
        # Plot normalized uncertainties
        plot_metric_curves(x_values, {"AU_GL_norm": au_norm, "EU_GL_norm": eu_norm}, 
                          f"{model_name} GL uncertainty (normalized) vs {condition_name}", condition_name, subfolder)
        summary = {
            condition_name: x_values, "Accuracy": acc, "ECE": ece,
            "AU_GL": au, "EU_GL": eu,
            "AU_GL_norm": au_norm, "EU_GL_norm": eu_norm,
            "AU_EU_corr": au_eu_corr,
        }
    
    # Plot metrics
    plot_metric_curves(x_values, {"Accuracy": acc, "ECE": ece}, 
                      f"{model_name} metrics vs {condition_name}", condition_name, subfolder)
    
    # Plot AU-EU correlation across conditions
    plot_metric_curves(x_values, {"AU_EU_corr": au_eu_corr}, 
                      f"{model_name} AU-EU correlation vs {condition_name}", condition_name, subfolder)
    
    save_statistics(summary, f"{model_name}_{condition_name}_summary", subfolder=subfolder)
    
    return summary


# ============================================================================
# SAMPLE SIZE EXPERIMENTS - Per-model functions
# ============================================================================

def run_mc_dropout_it_sample_size_experiment(
    base_cfg: Dict[str, Any],
    sample_sizes: List[int],
    seed: int = 42,
    **model_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """
    Run sample size experiment for MC Dropout IT.
    
    Args:
        base_cfg: Base config for simulate_dataset (will be copied and N_train modified)
        sample_sizes: List of training set sizes to test
        seed: Random seed
        **model_kwargs: Model hyperparameters (epochs, lr, dropout_p, mc_samples, etc.)
    
    Returns:
        Dict mapping sample_size -> results dict
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for n_train in sample_sizes:
        cfg_local = dict(cfg)
        cfg_local["N_train"] = n_train
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="mc_dropout_it",
            train_fn=_train_mc_dropout_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"sample_size_{n_train}",
            subfolder="classification/sample_size/mc_dropout_it",
        )
        result["meta"] = meta
        results[n_train] = result
    
    _save_sweep_summary(results, "N_train", "mc_dropout_it", is_it=True,
                       subfolder="classification/sample_size/mc_dropout_it")
    return results


def run_mc_dropout_gl_sample_size_experiment(
    base_cfg: Dict[str, Any],
    sample_sizes: List[int],
    seed: int = 42,
    **model_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """Run sample size experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for n_train in sample_sizes:
        cfg_local = dict(cfg)
        cfg_local["N_train"] = n_train
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="mc_dropout_gl",
            train_fn=_train_mc_dropout_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"sample_size_{n_train}",
            subfolder="classification/sample_size/mc_dropout_gl",
        )
        result["meta"] = meta
        results[n_train] = result
    
    _save_sweep_summary(results, "N_train", "mc_dropout_gl", is_it=False,
                       subfolder="classification/sample_size/mc_dropout_gl")
    return results


def run_deep_ensemble_it_sample_size_experiment(
    base_cfg: Dict[str, Any],
    sample_sizes: List[int],
    seed: int = 42,
    **model_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """Run sample size experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for n_train in sample_sizes:
        cfg_local = dict(cfg)
        cfg_local["N_train"] = n_train
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="deep_ensemble_it",
            train_fn=_train_deep_ensemble_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"sample_size_{n_train}",
            subfolder="classification/sample_size/deep_ensemble_it",
        )
        result["meta"] = meta
        results[n_train] = result
    
    _save_sweep_summary(results, "N_train", "deep_ensemble_it", is_it=True,
                       subfolder="classification/sample_size/deep_ensemble_it")
    return results


def run_deep_ensemble_gl_sample_size_experiment(
    base_cfg: Dict[str, Any],
    sample_sizes: List[int],
    seed: int = 42,
    **model_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """Run sample size experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for n_train in sample_sizes:
        cfg_local = dict(cfg)
        cfg_local["N_train"] = n_train
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="deep_ensemble_gl",
            train_fn=_train_deep_ensemble_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"sample_size_{n_train}",
            subfolder="classification/sample_size/deep_ensemble_gl",
        )
        result["meta"] = meta
        results[n_train] = result
    
    _save_sweep_summary(results, "N_train", "deep_ensemble_gl", is_it=False,
                       subfolder="classification/sample_size/deep_ensemble_gl")
    return results


def run_bnn_it_sample_size_experiment(
    base_cfg: Dict[str, Any],
    sample_sizes: List[int],
    seed: int = 42,
    **model_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """Run sample size experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for n_train in sample_sizes:
        cfg_local = dict(cfg)
        cfg_local["N_train"] = n_train
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="bnn_it",
            train_fn=_train_bnn_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"sample_size_{n_train}",
            subfolder="classification/sample_size/bnn_it",
        )
        result["meta"] = meta
        results[n_train] = result
    
    _save_sweep_summary(results, "N_train", "bnn_it", is_it=True,
                       subfolder="classification/sample_size/bnn_it")
    return results


def run_bnn_gl_sample_size_experiment(
    base_cfg: Dict[str, Any],
    sample_sizes: List[int],
    seed: int = 42,
    **model_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """Run sample size experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for n_train in sample_sizes:
        cfg_local = dict(cfg)
        cfg_local["N_train"] = n_train
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="bnn_gl",
            train_fn=_train_bnn_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"sample_size_{n_train}",
            subfolder="classification/sample_size/bnn_gl",
        )
        result["meta"] = meta
        results[n_train] = result
    
    _save_sweep_summary(results, "N_train", "bnn_gl", is_it=False,
                       subfolder="classification/sample_size/bnn_gl")
    return results


# ============================================================================
# OOD EXPERIMENTS - Per-model functions
# ============================================================================

def run_mc_dropout_it_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run OOD experiment for MC Dropout IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    result = _run_it_experiment_single(
        model_name="mc_dropout_it",
        train_fn=_train_mc_dropout_it,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cfg=cfg,
        experiment_name="ood",
        subfolder="classification/ood/mc_dropout_it",
    )
    result["meta"] = meta
    
    # Compute OOD detection AUC if OOD mask exists
    ood_mask = meta.get("ood_mask_test")
    if ood_mask is not None and np.any(ood_mask):
        eu_scores = result["uncertainty"]["EU"]
        auc = roc_auc_score_manual(eu_scores, ood_mask.astype(int))
        result["ood_auc_eu"] = auc
        print(f"  OOD Detection AUC (EU): {auc:.4f}")
    
    return result


def run_mc_dropout_gl_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run OOD experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    result = _run_gl_experiment_single(
        model_name="mc_dropout_gl",
        train_fn=_train_mc_dropout_gl,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cfg=cfg,
        experiment_name="ood",
        subfolder="classification/ood/mc_dropout_gl",
    )
    result["meta"] = meta
    
    ood_mask = meta.get("ood_mask_test")
    if ood_mask is not None and np.any(ood_mask):
        eu_scores = result["uncertainty"]["EU"]
        auc = roc_auc_score_manual(eu_scores, ood_mask.astype(int))
        result["ood_auc_eu"] = auc
        print(f"  OOD Detection AUC (EU): {auc:.4f}")
    
    return result


def run_deep_ensemble_it_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run OOD experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    result = _run_it_experiment_single(
        model_name="deep_ensemble_it",
        train_fn=_train_deep_ensemble_it,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cfg=cfg,
        experiment_name="ood",
        subfolder="classification/ood/deep_ensemble_it",
    )
    result["meta"] = meta
    
    ood_mask = meta.get("ood_mask_test")
    if ood_mask is not None and np.any(ood_mask):
        eu_scores = result["uncertainty"]["EU"]
        auc = roc_auc_score_manual(eu_scores, ood_mask.astype(int))
        result["ood_auc_eu"] = auc
        print(f"  OOD Detection AUC (EU): {auc:.4f}")
    
    return result


def run_deep_ensemble_gl_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run OOD experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    result = _run_gl_experiment_single(
        model_name="deep_ensemble_gl",
        train_fn=_train_deep_ensemble_gl,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cfg=cfg,
        experiment_name="ood",
        subfolder="classification/ood/deep_ensemble_gl",
    )
    result["meta"] = meta
    
    ood_mask = meta.get("ood_mask_test")
    if ood_mask is not None and np.any(ood_mask):
        eu_scores = result["uncertainty"]["EU"]
        auc = roc_auc_score_manual(eu_scores, ood_mask.astype(int))
        result["ood_auc_eu"] = auc
        print(f"  OOD Detection AUC (EU): {auc:.4f}")
    
    return result


def run_bnn_it_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run OOD experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    result = _run_it_experiment_single(
        model_name="bnn_it",
        train_fn=_train_bnn_it,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cfg=cfg,
        experiment_name="ood",
        subfolder="classification/ood/bnn_it",
    )
    result["meta"] = meta
    
    ood_mask = meta.get("ood_mask_test")
    if ood_mask is not None and np.any(ood_mask):
        eu_scores = result["uncertainty"]["EU"]
        auc = roc_auc_score_manual(eu_scores, ood_mask.astype(int))
        result["ood_auc_eu"] = auc
        print(f"  OOD Detection AUC (EU): {auc:.4f}")
    
    return result


def run_bnn_gl_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run OOD experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    result = _run_gl_experiment_single(
        model_name="bnn_gl",
        train_fn=_train_bnn_gl,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cfg=cfg,
        experiment_name="ood",
        subfolder="classification/ood/bnn_gl",
    )
    result["meta"] = meta
    
    ood_mask = meta.get("ood_mask_test")
    if ood_mask is not None and np.any(ood_mask):
        eu_scores = result["uncertainty"]["EU"]
        auc = roc_auc_score_manual(eu_scores, ood_mask.astype(int))
        result["ood_auc_eu"] = auc
        print(f"  OOD Detection AUC (EU): {auc:.4f}")
    
    return result


# ============================================================================
# UNDERSAMPLING EXPERIMENTS - Per-model functions
# ============================================================================

def run_mc_dropout_it_undersampling_experiment(
    base_cfg: Dict[str, Any],
    rho_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """
    Run undersampling experiment for MC Dropout IT.
    
    Args:
        base_cfg: Base config (should include undersampling settings template)
        rho_values: List of undersampling rho values to test
        seed: Random seed
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rho in rho_values:
        cfg_local = dict(cfg)
        cfg_local["undersampling"] = {
            "boundary_band": {"enabled": True, "d0": 0.15, "rho": rho},
        }
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="mc_dropout_it",
            train_fn=_train_mc_dropout_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"undersampling_rho_{rho}",
            subfolder="classification/undersampling/mc_dropout_it",
        )
        result["meta"] = meta
        results[rho] = result
    
    _save_sweep_summary(results, "rho", "mc_dropout_it", is_it=True,
                       subfolder="classification/undersampling/mc_dropout_it")
    return results


def run_mc_dropout_gl_undersampling_experiment(
    base_cfg: Dict[str, Any],
    rho_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run undersampling experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rho in rho_values:
        cfg_local = dict(cfg)
        cfg_local["undersampling"] = {
            "boundary_band": {"enabled": True, "d0": 0.15, "rho": rho},
        }
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="mc_dropout_gl",
            train_fn=_train_mc_dropout_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"undersampling_rho_{rho}",
            subfolder="classification/undersampling/mc_dropout_gl",
        )
        result["meta"] = meta
        results[rho] = result
    
    _save_sweep_summary(results, "rho", "mc_dropout_gl", is_it=False,
                       subfolder="classification/undersampling/mc_dropout_gl")
    return results


def run_deep_ensemble_it_undersampling_experiment(
    base_cfg: Dict[str, Any],
    rho_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run undersampling experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rho in rho_values:
        cfg_local = dict(cfg)
        cfg_local["undersampling"] = {
            "boundary_band": {"enabled": True, "d0": 0.15, "rho": rho},
        }
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="deep_ensemble_it",
            train_fn=_train_deep_ensemble_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"undersampling_rho_{rho}",
            subfolder="classification/undersampling/deep_ensemble_it",
        )
        result["meta"] = meta
        results[rho] = result
    
    _save_sweep_summary(results, "rho", "deep_ensemble_it", is_it=True,
                       subfolder="classification/undersampling/deep_ensemble_it")
    return results


def run_deep_ensemble_gl_undersampling_experiment(
    base_cfg: Dict[str, Any],
    rho_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run undersampling experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rho in rho_values:
        cfg_local = dict(cfg)
        cfg_local["undersampling"] = {
            "boundary_band": {"enabled": True, "d0": 0.15, "rho": rho},
        }
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="deep_ensemble_gl",
            train_fn=_train_deep_ensemble_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"undersampling_rho_{rho}",
            subfolder="classification/undersampling/deep_ensemble_gl",
        )
        result["meta"] = meta
        results[rho] = result
    
    _save_sweep_summary(results, "rho", "deep_ensemble_gl", is_it=False,
                       subfolder="classification/undersampling/deep_ensemble_gl")
    return results


def run_bnn_it_undersampling_experiment(
    base_cfg: Dict[str, Any],
    rho_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run undersampling experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rho in rho_values:
        cfg_local = dict(cfg)
        cfg_local["undersampling"] = {
            "boundary_band": {"enabled": True, "d0": 0.15, "rho": rho},
        }
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="bnn_it",
            train_fn=_train_bnn_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"undersampling_rho_{rho}",
            subfolder="classification/undersampling/bnn_it",
        )
        result["meta"] = meta
        results[rho] = result
    
    _save_sweep_summary(results, "rho", "bnn_it", is_it=True,
                       subfolder="classification/undersampling/bnn_it")
    return results


def run_bnn_gl_undersampling_experiment(
    base_cfg: Dict[str, Any],
    rho_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run undersampling experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rho in rho_values:
        cfg_local = dict(cfg)
        cfg_local["undersampling"] = {
            "boundary_band": {"enabled": True, "d0": 0.15, "rho": rho},
        }
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="bnn_gl",
            train_fn=_train_bnn_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"undersampling_rho_{rho}",
            subfolder="classification/undersampling/bnn_gl",
        )
        result["meta"] = meta
        results[rho] = result
    
    _save_sweep_summary(results, "rho", "bnn_gl", is_it=False,
                       subfolder="classification/undersampling/bnn_gl")
    return results


# ============================================================================
# RCD (RELATIVE CLASS DISTANCE) EXPERIMENTS - Per-model functions
# ============================================================================

def run_mc_dropout_it_rcd_experiment(
    base_cfg: Dict[str, Any],
    rcd_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """
    Run relative class distance experiment for MC Dropout IT.
    
    Args:
        base_cfg: Base config for simulate_dataset
        rcd_values: List of RCD values to test (d_between / sigma_within)
        seed: Random seed
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rcd in rcd_values:
        cfg_local = dict(cfg)
        cfg_local["rcd"] = rcd
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="mc_dropout_it",
            train_fn=_train_mc_dropout_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"rcd_{rcd}",
            subfolder="classification/rcd/mc_dropout_it",
        )
        result["meta"] = meta
        results[rcd] = result
    
    _save_sweep_summary(results, "rcd", "mc_dropout_it", is_it=True,
                       subfolder="classification/rcd/mc_dropout_it")
    return results


def run_mc_dropout_gl_rcd_experiment(
    base_cfg: Dict[str, Any],
    rcd_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run relative class distance experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rcd in rcd_values:
        cfg_local = dict(cfg)
        cfg_local["rcd"] = rcd
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="mc_dropout_gl",
            train_fn=_train_mc_dropout_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"rcd_{rcd}",
            subfolder="classification/rcd/mc_dropout_gl",
        )
        result["meta"] = meta
        results[rcd] = result
    
    _save_sweep_summary(results, "rcd", "mc_dropout_gl", is_it=False,
                       subfolder="classification/rcd/mc_dropout_gl")
    return results


def run_deep_ensemble_it_rcd_experiment(
    base_cfg: Dict[str, Any],
    rcd_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run relative class distance experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rcd in rcd_values:
        cfg_local = dict(cfg)
        cfg_local["rcd"] = rcd
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="deep_ensemble_it",
            train_fn=_train_deep_ensemble_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"rcd_{rcd}",
            subfolder="classification/rcd/deep_ensemble_it",
        )
        result["meta"] = meta
        results[rcd] = result
    
    _save_sweep_summary(results, "rcd", "deep_ensemble_it", is_it=True,
                       subfolder="classification/rcd/deep_ensemble_it")
    return results


def run_deep_ensemble_gl_rcd_experiment(
    base_cfg: Dict[str, Any],
    rcd_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run relative class distance experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rcd in rcd_values:
        cfg_local = dict(cfg)
        cfg_local["rcd"] = rcd
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="deep_ensemble_gl",
            train_fn=_train_deep_ensemble_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"rcd_{rcd}",
            subfolder="classification/rcd/deep_ensemble_gl",
        )
        result["meta"] = meta
        results[rcd] = result
    
    _save_sweep_summary(results, "rcd", "deep_ensemble_gl", is_it=False,
                       subfolder="classification/rcd/deep_ensemble_gl")
    return results


def run_bnn_it_rcd_experiment(
    base_cfg: Dict[str, Any],
    rcd_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run relative class distance experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rcd in rcd_values:
        cfg_local = dict(cfg)
        cfg_local["rcd"] = rcd
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="bnn_it",
            train_fn=_train_bnn_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"rcd_{rcd}",
            subfolder="classification/rcd/bnn_it",
        )
        result["meta"] = meta
        results[rcd] = result
    
    _save_sweep_summary(results, "rcd", "bnn_it", is_it=True,
                       subfolder="classification/rcd/bnn_it")
    return results


def run_bnn_gl_rcd_experiment(
    base_cfg: Dict[str, Any],
    rcd_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run relative class distance experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for rcd in rcd_values:
        cfg_local = dict(cfg)
        cfg_local["rcd"] = rcd
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="bnn_gl",
            train_fn=_train_bnn_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"rcd_{rcd}",
            subfolder="classification/rcd/bnn_gl",
        )
        result["meta"] = meta
        results[rcd] = result
    
    _save_sweep_summary(results, "rcd", "bnn_gl", is_it=False,
                       subfolder="classification/rcd/bnn_gl")
    return results


# ============================================================================
# LABEL NOISE EXPERIMENTS
# ============================================================================

def run_mc_dropout_it_label_noise_experiment(
    base_cfg: Dict[str, Any],
    eta_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """
    Run label noise experiment for MC Dropout IT.
    
    Args:
        base_cfg: Base config for simulate_dataset
        eta_values: List of label noise rates to test (fraction of labels to flip)
        seed: Random seed
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for eta in eta_values:
        cfg_local = dict(cfg)
        cfg_local["eta"] = eta
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="mc_dropout_it",
            train_fn=_train_mc_dropout_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"eta_{eta}",
            subfolder="classification/label_noise/mc_dropout_it",
        )
        result["meta"] = meta
        results[eta] = result
    
    _save_sweep_summary(results, "eta", "mc_dropout_it", is_it=True,
                       subfolder="classification/label_noise/mc_dropout_it")
    return results


def run_mc_dropout_gl_label_noise_experiment(
    base_cfg: Dict[str, Any],
    eta_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run label noise experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for eta in eta_values:
        cfg_local = dict(cfg)
        cfg_local["eta"] = eta
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="mc_dropout_gl",
            train_fn=_train_mc_dropout_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"eta_{eta}",
            subfolder="classification/label_noise/mc_dropout_gl",
        )
        result["meta"] = meta
        results[eta] = result
    
    _save_sweep_summary(results, "eta", "mc_dropout_gl", is_it=False,
                       subfolder="classification/label_noise/mc_dropout_gl")
    return results


def run_deep_ensemble_it_label_noise_experiment(
    base_cfg: Dict[str, Any],
    eta_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run label noise experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for eta in eta_values:
        cfg_local = dict(cfg)
        cfg_local["eta"] = eta
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="deep_ensemble_it",
            train_fn=_train_deep_ensemble_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"eta_{eta}",
            subfolder="classification/label_noise/deep_ensemble_it",
        )
        result["meta"] = meta
        results[eta] = result
    
    _save_sweep_summary(results, "eta", "deep_ensemble_it", is_it=True,
                       subfolder="classification/label_noise/deep_ensemble_it")
    return results


def run_deep_ensemble_gl_label_noise_experiment(
    base_cfg: Dict[str, Any],
    eta_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run label noise experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for eta in eta_values:
        cfg_local = dict(cfg)
        cfg_local["eta"] = eta
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="deep_ensemble_gl",
            train_fn=_train_deep_ensemble_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"eta_{eta}",
            subfolder="classification/label_noise/deep_ensemble_gl",
        )
        result["meta"] = meta
        results[eta] = result
    
    _save_sweep_summary(results, "eta", "deep_ensemble_gl", is_it=False,
                       subfolder="classification/label_noise/deep_ensemble_gl")
    return results


def run_bnn_it_label_noise_experiment(
    base_cfg: Dict[str, Any],
    eta_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run label noise experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for eta in eta_values:
        cfg_local = dict(cfg)
        cfg_local["eta"] = eta
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_it_experiment_single(
            model_name="bnn_it",
            train_fn=_train_bnn_it,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"eta_{eta}",
            subfolder="classification/label_noise/bnn_it",
        )
        result["meta"] = meta
        results[eta] = result
    
    _save_sweep_summary(results, "eta", "bnn_it", is_it=True,
                       subfolder="classification/label_noise/bnn_it")
    return results


def run_bnn_gl_label_noise_experiment(
    base_cfg: Dict[str, Any],
    eta_values: List[float],
    seed: int = 42,
    **model_kwargs,
) -> Dict[float, Dict[str, Any]]:
    """Run label noise experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    results = {}
    
    for eta in eta_values:
        cfg_local = dict(cfg)
        cfg_local["eta"] = eta
        X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg_local)
        
        result = _run_gl_experiment_single(
            model_name="bnn_gl",
            train_fn=_train_bnn_gl,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cfg=cfg_local,
            experiment_name=f"eta_{eta}",
            subfolder="classification/label_noise/bnn_gl",
        )
        result["meta"] = meta
        results[eta] = result
    
    _save_sweep_summary(results, "eta", "bnn_gl", is_it=False,
                       subfolder="classification/label_noise/bnn_gl")
    return results


# ============================================================================
# HEATMAP UTILITIES
# ============================================================================

def compute_uncertainty_heatmap(
    base_cfg: Dict[str, Any],
    model_name: str,
    tag: str,
    grid_res: int = 60,
    subfolder: str = "classification/heatmaps",
    seed: int = 42,
    **model_kwargs,
):
    """
    Compute and plot normalized uncertainty heatmaps for a given model.
    
    Args:
        base_cfg: Base config for data generation
        model_name: One of mc_dropout_it, mc_dropout_gl, deep_ensemble_it, etc.
        tag: Tag for the plot title (e.g., "baseline", "small_N")
        grid_res: Grid resolution for heatmap
        subfolder: Where to save plots
        seed: Random seed
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    x = np.linspace(-1.5, 1.5, grid_res)
    y = np.linspace(-1.5, 1.5, grid_res)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    X_train, y_train, X_test, y_test, meta = simulate_dataset(cfg)
    
    model_subfolder = f"{subfolder}/{model_name}"
    
    if model_name.endswith("_it"):
        train_fn = {
            "mc_dropout_it": _train_mc_dropout_it,
            "deep_ensemble_it": _train_deep_ensemble_it,
            "bnn_it": _train_bnn_it,
        }[model_name]
        probs_members = train_fn(X_train, y_train, grid, cfg)
        unc = it_uncertainty(probs_members)
        # Plot normalized heatmaps with fixed [0, 1] colorbar
        plot_uncertainty_heatmap(grid, unc["AU_norm"], f"{model_name} AU ({tag})", model_subfolder, normalized=True)
        plot_uncertainty_heatmap(grid, unc["EU_norm"], f"{model_name} EU ({tag})", model_subfolder, normalized=True)
        plot_uncertainty_heatmap(grid, unc["TU_norm"], f"{model_name} TU ({tag})", model_subfolder, normalized=True)
    else:
        train_fn = {
            "mc_dropout_gl": _train_mc_dropout_gl,
            "deep_ensemble_gl": _train_deep_ensemble_gl,
            "bnn_gl": _train_bnn_gl,
        }[model_name]
        mu_members, sigma2_members = train_fn(X_train, y_train, grid, cfg)
        unc = gl_uncertainty(mu_members, sigma2_members, n_samples=cfg.get("gl_samples", 100))
        # Plot normalized heatmaps with fixed [0, 1] colorbar
        plot_uncertainty_heatmap(grid, unc["AU_norm"], f"{model_name} AU_GL ({tag})", model_subfolder, normalized=True)
        plot_uncertainty_heatmap(grid, unc["EU_norm"], f"{model_name} EU_GL ({tag})", model_subfolder, normalized=True)


# ============================================================================
# LEGACY COMPATIBILITY - Keep old generic functions for backward compatibility
# ============================================================================

def run_sample_size_experiment(cfg: Dict[str, Any], model_type: str, sample_sizes: List[int]) -> Dict[str, Any]:
    """Legacy: Run sample size experiment for any model type."""
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_sample_size_experiment(cfg, sample_sizes)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_sample_size_experiment(cfg, sample_sizes)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_sample_size_experiment(cfg, sample_sizes)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_sample_size_experiment(cfg, sample_sizes)
    elif model_type == "bnn_it":
        return run_bnn_it_sample_size_experiment(cfg, sample_sizes)
    elif model_type == "bnn_gl":
        return run_bnn_gl_sample_size_experiment(cfg, sample_sizes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_ood_experiment(cfg: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Legacy: Run OOD experiment for any model type."""
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_ood_experiment(cfg)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_ood_experiment(cfg)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_ood_experiment(cfg)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_ood_experiment(cfg)
    elif model_type == "bnn_it":
        return run_bnn_it_ood_experiment(cfg)
    elif model_type == "bnn_gl":
        return run_bnn_gl_ood_experiment(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_undersampling_experiment(cfg: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Legacy: Run undersampling experiment for any model type."""
    # For legacy compatibility, run with default rho values
    rho_values = [0.0, 0.3, 0.6]
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_undersampling_experiment(cfg, rho_values)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_undersampling_experiment(cfg, rho_values)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_undersampling_experiment(cfg, rho_values)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_undersampling_experiment(cfg, rho_values)
    elif model_type == "bnn_it":
        return run_bnn_it_undersampling_experiment(cfg, rho_values)
    elif model_type == "bnn_gl":
        return run_bnn_gl_undersampling_experiment(cfg, rho_values)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_rcd_experiment(cfg: Dict[str, Any], model_type: str, rcd_values: List[float]) -> Dict[str, Any]:
    """Legacy: Run RCD (relative class distance) experiment for any model type."""
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_rcd_experiment(cfg, rcd_values)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_rcd_experiment(cfg, rcd_values)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_rcd_experiment(cfg, rcd_values)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_rcd_experiment(cfg, rcd_values)
    elif model_type == "bnn_it":
        return run_bnn_it_rcd_experiment(cfg, rcd_values)
    elif model_type == "bnn_gl":
        return run_bnn_gl_rcd_experiment(cfg, rcd_values)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_label_noise_experiment(cfg: Dict[str, Any], model_type: str, eta_values: List[float]) -> Dict[str, Any]:
    """Legacy: Run label noise experiment for any model type."""
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_label_noise_experiment(cfg, eta_values)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_label_noise_experiment(cfg, eta_values)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_label_noise_experiment(cfg, eta_values)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_label_noise_experiment(cfg, eta_values)
    elif model_type == "bnn_it":
        return run_bnn_it_label_noise_experiment(cfg, eta_values)
    elif model_type == "bnn_gl":
        return run_bnn_gl_label_noise_experiment(cfg, eta_values)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# ROTATION OOD EXPERIMENTS - Per-model functions
# ============================================================================

def _run_rotation_ood_it_experiment(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_train_ood: np.ndarray,
    y_train_ood: np.ndarray,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    subfolder: str,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for an IT model with heatmap visualization.
    
    Trains ONE model on ID data, then predicts on both ID and OOD data.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (IT) - Rotation OOD Experiment")
    print(f"{'='*60}")
    
    # Create grids for visualization - one for ID region, one for OOD region
    grid_res = 100
    
    # ID grid (covers original training data region)
    x_min_id, x_max_id = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min_id, y_max_id = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx_id, yy_id = np.meshgrid(
        np.linspace(x_min_id, x_max_id, grid_res),
        np.linspace(y_min_id, y_max_id, grid_res)
    )
    grid_id = np.stack([xx_id.ravel(), yy_id.ravel()], axis=1).astype(np.float32)
    
    # OOD grid (covers rotated training data region)
    x_min_ood, x_max_ood = X_train_ood[:, 0].min() - 0.5, X_train_ood[:, 0].max() + 0.5
    y_min_ood, y_max_ood = X_train_ood[:, 1].min() - 0.5, X_train_ood[:, 1].max() + 0.5
    xx_ood, yy_ood = np.meshgrid(
        np.linspace(x_min_ood, x_max_ood, grid_res),
        np.linspace(y_min_ood, y_max_ood, grid_res)
    )
    grid_ood = np.stack([xx_ood.ravel(), yy_ood.ravel()], axis=1).astype(np.float32)
    
    # Train model ONCE on ID training data, then predict on all inputs
    if model_name == "mc_dropout_it":
        model = train_mc_dropout_it(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            p=cfg.get("dropout_p", 0.25),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32)
        )
        mc_samples = cfg.get("mc_samples", 100)
        probs_members_grid_id = mc_dropout_predict_it(model, grid_id, mc_samples=mc_samples)
        probs_members_grid_ood = mc_dropout_predict_it(model, grid_ood, mc_samples=mc_samples)
        probs_members_train = mc_dropout_predict_it(model, X_train, mc_samples=mc_samples)
        probs_members_train_ood = mc_dropout_predict_it(model, X_train_ood, mc_samples=mc_samples)
        
    elif model_name == "deep_ensemble_it":
        ensemble = train_deep_ensemble_it(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            K=cfg.get("K", 10),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32),
            seed=cfg.get("seed", 42)
        )
        probs_members_grid_id = ensemble_predict_it(ensemble, grid_id)
        probs_members_grid_ood = ensemble_predict_it(ensemble, grid_ood)
        probs_members_train = ensemble_predict_it(ensemble, X_train)
        probs_members_train_ood = ensemble_predict_it(ensemble, X_train_ood)
        
    elif model_name == "bnn_it":
        mcmc = train_bnn_it(
            X_train, y_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            warmup=cfg.get("warmup", 200),
            samples=cfg.get("samples", 200),
            chains=cfg.get("chains", 1),
            seed=cfg.get("seed", 42)
        )
        probs_members_grid_id = bnn_predict_it(
            mcmc, grid_id,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        probs_members_grid_ood = bnn_predict_it(
            mcmc, grid_ood,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        probs_members_train = bnn_predict_it(
            mcmc, X_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        probs_members_train_ood = bnn_predict_it(
            mcmc, X_train_ood,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
    else:
        raise ValueError(f"Unknown IT model: {model_name}")
    
    # Compute uncertainties on grids
    unc_grid_id = it_uncertainty(probs_members_grid_id)
    unc_grid_ood = it_uncertainty(probs_members_grid_ood)
    
    # Compute uncertainties on training points
    unc_train = it_uncertainty(probs_members_train)
    unc_train_ood = it_uncertainty(probs_members_train_ood)
    
    # Get predictive probabilities
    probs_pred_train = _predictive_probs_it(probs_members_train)
    probs_pred_train_ood = _predictive_probs_it(probs_members_train_ood)
    
    # Compute metrics
    metrics_id = _evaluate_probs(probs_pred_train, y_train)
    metrics_ood = _evaluate_probs(probs_pred_train_ood, y_train_ood)
    
    print(f"  Train Accuracy (ID): {metrics_id['accuracy']:.4f}, ECE: {metrics_id['ece']:.4f}")
    print(f"  Train Accuracy (OOD): {metrics_ood['accuracy']:.4f}, ECE: {metrics_ood['ece']:.4f}")
    print(f"  Mean TU: ID={unc_train['TU'].mean():.4f} (norm: {unc_train['TU_norm'].mean():.4f}), "
          f"OOD={unc_train_ood['TU'].mean():.4f} (norm: {unc_train_ood['TU_norm'].mean():.4f})")
    print(f"  Mean AU: ID={unc_train['AU'].mean():.4f} (norm: {unc_train['AU_norm'].mean():.4f}), "
          f"OOD={unc_train_ood['AU'].mean():.4f} (norm: {unc_train_ood['AU_norm'].mean():.4f})")
    print(f"  Mean EU: ID={unc_train['EU'].mean():.4f} (norm: {unc_train['EU_norm'].mean():.4f}), "
          f"OOD={unc_train_ood['EU'].mean():.4f} (norm: {unc_train_ood['EU_norm'].mean():.4f})")
    
    # Save outputs
    save_classification_outputs(
        {
            "probs_members_train": probs_members_train,
            "probs_members_train_ood": probs_members_train_ood,
            "y_train": y_train,
            "x_train": X_train,
            "x_train_ood": X_train_ood,
            "grid_id": grid_id,
            "grid_ood": grid_ood,
            "metrics_id": np.array(list(metrics_id.values()), dtype=np.float32),
            "metrics_ood": np.array(list(metrics_ood.values()), dtype=np.float32),
        },
        model_name=model_name,
        experiment_name="rotation_ood",
        subfolder=subfolder,
    )
    
    # Generate heatmaps for ID grid
    grid_extent_id = (x_min_id, x_max_id, y_min_id, y_max_id)
    _plot_uncertainty_heatmaps(
        X_eval=grid_id,
        uncertainty=unc_grid_id,
        X_test=X_train,
        y_test=y_train,
        probs_pred=probs_pred_train,
        model_name=model_name,
        experiment_name="rotation_ood_ID",
        subfolder=f"{subfolder}/heatmaps",
        is_it=True,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent_id,
        grid_res=grid_res,
    )
    
    # Generate heatmaps for OOD grid
    grid_extent_ood = (x_min_ood, x_max_ood, y_min_ood, y_max_ood)
    _plot_uncertainty_heatmaps(
        X_eval=grid_ood,
        uncertainty=unc_grid_ood,
        X_test=X_train_ood,
        y_test=y_train_ood,
        probs_pred=probs_pred_train_ood,
        model_name=model_name,
        experiment_name="rotation_ood_OOD",
        subfolder=f"{subfolder}/heatmaps",
        is_it=True,
        X_train=X_train_ood,
        y_train=y_train_ood,
        grid_extent=grid_extent_ood,
        grid_res=grid_res,
    )
    
    # Compute AU-EU correlation
    au_eu_corr_id = np.corrcoef(unc_train["AU_norm"], unc_train["EU_norm"])[0, 1]
    au_eu_corr_ood = np.corrcoef(unc_train_ood["AU_norm"], unc_train_ood["EU_norm"])[0, 1]
    
    # Save summary statistics
    summary = {
        "model": model_name,
        "acc_id": metrics_id["accuracy"],
        "acc_ood": metrics_ood["accuracy"],
        "ece_id": metrics_id["ece"],
        "ece_ood": metrics_ood["ece"],
        "mean_tu_id": float(unc_train["TU"].mean()),
        "mean_tu_ood": float(unc_train_ood["TU"].mean()),
        "mean_au_id": float(unc_train["AU"].mean()),
        "mean_au_ood": float(unc_train_ood["AU"].mean()),
        "mean_eu_id": float(unc_train["EU"].mean()),
        "mean_eu_ood": float(unc_train_ood["EU"].mean()),
        "au_eu_corr_id": au_eu_corr_id,
        "au_eu_corr_ood": au_eu_corr_ood,
        "rotation_angle": meta.get("rotation_angle", 45),
    }
    save_statistics([summary], f"{model_name}_rotation_ood_summary", subfolder=subfolder)
    
    return {
        "uncertainty_id": unc_train,
        "uncertainty_ood": unc_train_ood,
        "uncertainty_grid_id": unc_grid_id,
        "uncertainty_grid_ood": unc_grid_ood,
        "metrics_id": metrics_id,
        "metrics_ood": metrics_ood,
        "probs_pred_id": probs_pred_train,
        "probs_pred_ood": probs_pred_train_ood,
        "au_eu_correlation_id": au_eu_corr_id,
        "au_eu_correlation_ood": au_eu_corr_ood,
        "meta": meta,
    }


def _run_rotation_ood_gl_experiment(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_train_ood: np.ndarray,
    y_train_ood: np.ndarray,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    subfolder: str,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for a GL model with heatmap visualization.
    
    Trains ONE model on ID data, then predicts on both ID and OOD data.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (GL) - Rotation OOD Experiment")
    print(f"{'='*60}")
    
    # Create grids for visualization - one for ID region, one for OOD region
    grid_res = 100
    
    # ID grid (covers original training data region)
    x_min_id, x_max_id = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min_id, y_max_id = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx_id, yy_id = np.meshgrid(
        np.linspace(x_min_id, x_max_id, grid_res),
        np.linspace(y_min_id, y_max_id, grid_res)
    )
    grid_id = np.stack([xx_id.ravel(), yy_id.ravel()], axis=1).astype(np.float32)
    
    # OOD grid (covers rotated training data region)
    x_min_ood, x_max_ood = X_train_ood[:, 0].min() - 0.5, X_train_ood[:, 0].max() + 0.5
    y_min_ood, y_max_ood = X_train_ood[:, 1].min() - 0.5, X_train_ood[:, 1].max() + 0.5
    xx_ood, yy_ood = np.meshgrid(
        np.linspace(x_min_ood, x_max_ood, grid_res),
        np.linspace(y_min_ood, y_max_ood, grid_res)
    )
    grid_ood = np.stack([xx_ood.ravel(), yy_ood.ravel()], axis=1).astype(np.float32)
    
    gl_samples = cfg.get("gl_samples", 100)
    
    # Train model ONCE on ID training data, then predict on all inputs
    if model_name == "mc_dropout_gl":
        model = train_mc_dropout_gl(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            p=cfg.get("dropout_p", 0.25),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32),
            n_samples=gl_samples
        )
        mc_samples = cfg.get("mc_samples", 100)
        mu_members_grid_id, sigma2_members_grid_id = mc_dropout_predict_gl(model, grid_id, mc_samples=mc_samples)
        mu_members_grid_ood, sigma2_members_grid_ood = mc_dropout_predict_gl(model, grid_ood, mc_samples=mc_samples)
        mu_members_train, sigma2_members_train = mc_dropout_predict_gl(model, X_train, mc_samples=mc_samples)
        mu_members_train_ood, sigma2_members_train_ood = mc_dropout_predict_gl(model, X_train_ood, mc_samples=mc_samples)
        
    elif model_name == "deep_ensemble_gl":
        ensemble = train_deep_ensemble_gl(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            K=cfg.get("K", 10),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32),
            seed=cfg.get("seed", 42),
            n_samples=gl_samples
        )
        mu_members_grid_id, sigma2_members_grid_id = ensemble_predict_gl(ensemble, grid_id)
        mu_members_grid_ood, sigma2_members_grid_ood = ensemble_predict_gl(ensemble, grid_ood)
        mu_members_train, sigma2_members_train = ensemble_predict_gl(ensemble, X_train)
        mu_members_train_ood, sigma2_members_train_ood = ensemble_predict_gl(ensemble, X_train_ood)
        
    elif model_name == "bnn_gl":
        mcmc = train_bnn_gl(
            X_train, y_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            warmup=cfg.get("warmup", 200),
            samples=cfg.get("samples", 200),
            chains=cfg.get("chains", 1),
            seed=cfg.get("seed", 42)
        )
        mu_members_grid_id, sigma2_members_grid_id = bnn_predict_gl(
            mcmc, grid_id,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        mu_members_grid_ood, sigma2_members_grid_ood = bnn_predict_gl(
            mcmc, grid_ood,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        mu_members_train, sigma2_members_train = bnn_predict_gl(
            mcmc, X_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        mu_members_train_ood, sigma2_members_train_ood = bnn_predict_gl(
            mcmc, X_train_ood,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
    else:
        raise ValueError(f"Unknown GL model: {model_name}")
    
    # Compute uncertainties on grids
    unc_grid_id = gl_uncertainty(mu_members_grid_id, sigma2_members_grid_id, n_samples=gl_samples)
    unc_grid_ood = gl_uncertainty(mu_members_grid_ood, sigma2_members_grid_ood, n_samples=gl_samples)
    
    # Compute uncertainties on training points
    unc_train = gl_uncertainty(mu_members_train, sigma2_members_train, n_samples=gl_samples)
    unc_train_ood = gl_uncertainty(mu_members_train_ood, sigma2_members_train_ood, n_samples=gl_samples)
    
    # Get predictive probabilities
    probs_pred_train = _predictive_probs_gl(mu_members_train, sigma2_members_train, n_samples=gl_samples)
    probs_pred_train_ood = _predictive_probs_gl(mu_members_train_ood, sigma2_members_train_ood, n_samples=gl_samples)
    
    # Compute metrics
    metrics_id = _evaluate_probs(probs_pred_train, y_train)
    metrics_ood = _evaluate_probs(probs_pred_train_ood, y_train_ood)
    
    print(f"  Train Accuracy (ID): {metrics_id['accuracy']:.4f}, ECE: {metrics_id['ece']:.4f}")
    print(f"  Train Accuracy (OOD): {metrics_ood['accuracy']:.4f}, ECE: {metrics_ood['ece']:.4f}")
    print(f"  Mean AU: ID={unc_train['AU'].mean():.4f} (norm: {unc_train['AU_norm'].mean():.4f}), "
          f"OOD={unc_train_ood['AU'].mean():.4f} (norm: {unc_train_ood['AU_norm'].mean():.4f})")
    print(f"  Mean EU: ID={unc_train['EU'].mean():.4f} (norm: {unc_train['EU_norm'].mean():.4f}), "
          f"OOD={unc_train_ood['EU'].mean():.4f} (norm: {unc_train_ood['EU_norm'].mean():.4f})")
    
    # Save outputs
    save_classification_outputs(
        {
            "mu_members_train": mu_members_train,
            "sigma2_members_train": sigma2_members_train,
            "mu_members_train_ood": mu_members_train_ood,
            "sigma2_members_train_ood": sigma2_members_train_ood,
            "y_train": y_train,
            "x_train": X_train,
            "x_train_ood": X_train_ood,
            "grid_id": grid_id,
            "grid_ood": grid_ood,
            "metrics_id": np.array(list(metrics_id.values()), dtype=np.float32),
            "metrics_ood": np.array(list(metrics_ood.values()), dtype=np.float32),
        },
        model_name=model_name,
        experiment_name="rotation_ood",
        subfolder=subfolder,
    )
    
    # Generate heatmaps for ID grid
    grid_extent_id = (x_min_id, x_max_id, y_min_id, y_max_id)
    _plot_uncertainty_heatmaps(
        X_eval=grid_id,
        uncertainty=unc_grid_id,
        X_test=X_train,
        y_test=y_train,
        probs_pred=probs_pred_train,
        model_name=model_name,
        experiment_name="rotation_ood_ID",
        subfolder=f"{subfolder}/heatmaps",
        is_it=False,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent_id,
        grid_res=grid_res,
    )
    
    # Generate heatmaps for OOD grid
    grid_extent_ood = (x_min_ood, x_max_ood, y_min_ood, y_max_ood)
    _plot_uncertainty_heatmaps(
        X_eval=grid_ood,
        uncertainty=unc_grid_ood,
        X_test=X_train_ood,
        y_test=y_train_ood,
        probs_pred=probs_pred_train_ood,
        model_name=model_name,
        experiment_name="rotation_ood_OOD",
        subfolder=f"{subfolder}/heatmaps",
        is_it=False,
        X_train=X_train_ood,
        y_train=y_train_ood,
        grid_extent=grid_extent_ood,
        grid_res=grid_res,
    )
    
    # Compute AU-EU correlation
    au_eu_corr_id = np.corrcoef(unc_train["AU_norm"], unc_train["EU_norm"])[0, 1]
    au_eu_corr_ood = np.corrcoef(unc_train_ood["AU_norm"], unc_train_ood["EU_norm"])[0, 1]
    
    # Save summary statistics
    summary = {
        "model": model_name,
        "acc_id": metrics_id["accuracy"],
        "acc_ood": metrics_ood["accuracy"],
        "ece_id": metrics_id["ece"],
        "ece_ood": metrics_ood["ece"],
        "mean_au_id": float(unc_train["AU"].mean()),
        "mean_au_ood": float(unc_train_ood["AU"].mean()),
        "mean_eu_id": float(unc_train["EU"].mean()),
        "mean_eu_ood": float(unc_train_ood["EU"].mean()),
        "au_eu_corr_id": au_eu_corr_id,
        "au_eu_corr_ood": au_eu_corr_ood,
        "rotation_angle": meta.get("rotation_angle", 45),
    }
    save_statistics([summary], f"{model_name}_rotation_ood_summary", subfolder=subfolder)
    
    return {
        "uncertainty_id": unc_train,
        "uncertainty_ood": unc_train_ood,
        "uncertainty_grid_id": unc_grid_id,
        "uncertainty_grid_ood": unc_grid_ood,
        "metrics_id": metrics_id,
        "metrics_ood": metrics_ood,
        "probs_pred_id": probs_pred_train,
        "probs_pred_ood": probs_pred_train_ood,
        "au_eu_correlation_id": au_eu_corr_id,
        "au_eu_correlation_ood": au_eu_corr_ood,
        "meta": meta,
    }


def run_mc_dropout_it_rotation_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Run rotation OOD experiment for MC Dropout IT.
    
    Args:
        base_cfg: Config for simulate_rotation_ood_dataset
        seed: Random seed
        **model_kwargs: Model hyperparameters
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_train_ood, y_train_ood, meta = simulate_rotation_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_rotation_ood_it_experiment(
        model_name="mc_dropout_it",
        X_train=X_train, y_train=y_train,
        X_train_ood=X_train_ood, y_train_ood=y_train_ood,
        cfg=cfg,
        meta=meta,
        subfolder="classification/rotation_ood/mc_dropout_it",
    )


def run_mc_dropout_gl_rotation_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_train_ood, y_train_ood, meta = simulate_rotation_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_rotation_ood_gl_experiment(
        model_name="mc_dropout_gl",
        X_train=X_train, y_train=y_train,
        X_train_ood=X_train_ood, y_train_ood=y_train_ood,
        cfg=cfg,
        meta=meta,
        subfolder="classification/rotation_ood/mc_dropout_gl",
    )


def run_deep_ensemble_it_rotation_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_train_ood, y_train_ood, meta = simulate_rotation_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_rotation_ood_it_experiment(
        model_name="deep_ensemble_it",
        X_train=X_train, y_train=y_train,
        X_train_ood=X_train_ood, y_train_ood=y_train_ood,
        cfg=cfg,
        meta=meta,
        subfolder="classification/rotation_ood/deep_ensemble_it",
    )


def run_deep_ensemble_gl_rotation_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_train_ood, y_train_ood, meta = simulate_rotation_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_rotation_ood_gl_experiment(
        model_name="deep_ensemble_gl",
        X_train=X_train, y_train=y_train,
        X_train_ood=X_train_ood, y_train_ood=y_train_ood,
        cfg=cfg,
        meta=meta,
        subfolder="classification/rotation_ood/deep_ensemble_gl",
    )


def run_bnn_it_rotation_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_train_ood, y_train_ood, meta = simulate_rotation_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_rotation_ood_it_experiment(
        model_name="bnn_it",
        X_train=X_train, y_train=y_train,
        X_train_ood=X_train_ood, y_train_ood=y_train_ood,
        cfg=cfg,
        meta=meta,
        subfolder="classification/rotation_ood/bnn_it",
    )


def run_bnn_gl_rotation_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run rotation OOD experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_train_ood, y_train_ood, meta = simulate_rotation_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_rotation_ood_gl_experiment(
        model_name="bnn_gl",
        X_train=X_train, y_train=y_train,
        X_train_ood=X_train_ood, y_train_ood=y_train_ood,
        cfg=cfg,
        meta=meta,
        subfolder="classification/rotation_ood/bnn_gl",
    )


def run_rotation_ood_experiment(cfg: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Legacy: Run rotation OOD experiment for any model type."""
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_rotation_ood_experiment(cfg)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_rotation_ood_experiment(cfg)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_rotation_ood_experiment(cfg)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_rotation_ood_experiment(cfg)
    elif model_type == "bnn_it":
        return run_bnn_it_rotation_ood_experiment(cfg)
    elif model_type == "bnn_gl":
        return run_bnn_gl_rotation_ood_experiment(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# RING OOD EXPERIMENTS - Concentric rings with gap region
# ============================================================================

def _run_ring_ood_it_experiment(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_gap: np.ndarray,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    subfolder: str,
) -> Dict[str, Any]:
    """Run ring OOD experiment for an IT model with heatmap visualization.
    
    Trains ONE model on ring data, then predicts on both rings and gap region.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (IT) - Ring OOD Experiment")
    print(f"{'='*60}")
    
    # Create grid for visualization covering full region
    grid_res = 100
    outer_r_max = meta["outer_r_max"]
    margin = 0.5
    x_min, x_max = -outer_r_max - margin, outer_r_max + margin
    y_min, y_max = -outer_r_max - margin, outer_r_max + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    # Train model ONCE on ring training data, then predict on all inputs
    if model_name == "mc_dropout_it":
        model = train_mc_dropout_it(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            p=cfg.get("dropout_p", 0.25),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32)
        )
        mc_samples = cfg.get("mc_samples", 100)
        probs_members_grid = mc_dropout_predict_it(model, grid, mc_samples=mc_samples)
        probs_members_train = mc_dropout_predict_it(model, X_train, mc_samples=mc_samples)
        probs_members_gap = mc_dropout_predict_it(model, X_gap, mc_samples=mc_samples)
        
    elif model_name == "deep_ensemble_it":
        ensemble = train_deep_ensemble_it(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            K=cfg.get("K", 10),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32),
            seed=cfg.get("seed", 42)
        )
        probs_members_grid = ensemble_predict_it(ensemble, grid)
        probs_members_train = ensemble_predict_it(ensemble, X_train)
        probs_members_gap = ensemble_predict_it(ensemble, X_gap)
        
    elif model_name == "bnn_it":
        mcmc = train_bnn_it(
            X_train, y_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            warmup=cfg.get("warmup", 200),
            samples=cfg.get("samples", 200),
            chains=cfg.get("chains", 1),
            seed=cfg.get("seed", 42)
        )
        probs_members_grid = bnn_predict_it(
            mcmc, grid,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        probs_members_train = bnn_predict_it(
            mcmc, X_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        probs_members_gap = bnn_predict_it(
            mcmc, X_gap,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
    else:
        raise ValueError(f"Unknown IT model: {model_name}")
    
    # Compute uncertainties
    unc_grid = it_uncertainty(probs_members_grid)
    unc_train = it_uncertainty(probs_members_train)
    unc_gap = it_uncertainty(probs_members_gap)
    
    # Get predictive probabilities
    probs_pred_train = _predictive_probs_it(probs_members_train)
    
    # Compute metrics on training data (rings)
    metrics_train = _evaluate_probs(probs_pred_train, y_train)
    
    print(f"  Ring Accuracy: {metrics_train['accuracy']:.4f}, ECE: {metrics_train['ece']:.4f}")
    print(f"  Ring  - Mean TU: {unc_train['TU'].mean():.4f} (norm: {unc_train['TU_norm'].mean():.4f}), "
          f"AU: {unc_train['AU'].mean():.4f} (norm: {unc_train['AU_norm'].mean():.4f}), "
          f"EU: {unc_train['EU'].mean():.4f} (norm: {unc_train['EU_norm'].mean():.4f})")
    print(f"  Gap   - Mean TU: {unc_gap['TU'].mean():.4f} (norm: {unc_gap['TU_norm'].mean():.4f}), "
          f"AU: {unc_gap['AU'].mean():.4f} (norm: {unc_gap['AU_norm'].mean():.4f}), "
          f"EU: {unc_gap['EU'].mean():.4f} (norm: {unc_gap['EU_norm'].mean():.4f})")
    
    # Save outputs
    save_classification_outputs(
        {
            "probs_members_train": probs_members_train,
            "probs_members_gap": probs_members_gap,
            "y_train": y_train,
            "x_train": X_train,
            "x_gap": X_gap,
            "grid": grid,
            "metrics_train": np.array(list(metrics_train.values()), dtype=np.float32),
        },
        model_name=model_name,
        experiment_name="ring_ood",
        subfolder=subfolder,
    )
    
    # Generate heatmap for full region with training points
    grid_extent = (x_min, x_max, y_min, y_max)
    _plot_uncertainty_heatmaps(
        X_eval=grid,
        uncertainty=unc_grid,
        X_test=X_train,
        y_test=y_train,
        probs_pred=probs_pred_train,
        model_name=model_name,
        experiment_name="ring_ood",
        subfolder=f"{subfolder}/heatmaps",
        is_it=True,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent,
        grid_res=grid_res,
    )
    
    # Generate heatmap with gap points overlaid
    probs_pred_gap = _predictive_probs_it(probs_members_gap)
    _plot_uncertainty_heatmaps(
        X_eval=grid,
        uncertainty=unc_grid,
        X_test=X_gap,
        y_test=np.zeros(len(X_gap), dtype=np.int64),
        probs_pred=probs_pred_gap,
        model_name=model_name,
        experiment_name="ring_ood_gap",
        subfolder=f"{subfolder}/heatmaps",
        is_it=True,
        X_train=X_gap,
        y_train=np.zeros(len(X_gap), dtype=np.int64),
        grid_extent=grid_extent,
        grid_res=grid_res,
    )
    
    # Compute AU-EU correlation
    au_eu_corr_train = np.corrcoef(unc_train["AU_norm"], unc_train["EU_norm"])[0, 1]
    au_eu_corr_gap = np.corrcoef(unc_gap["AU_norm"], unc_gap["EU_norm"])[0, 1]
    
    # Generate comparison bar plot
    _plot_ring_gap_comparison(
        unc_train, unc_gap,
        au_eu_corr_train, au_eu_corr_gap,
        model_name, subfolder,
        is_it=True,
    )
    
    # Save summary statistics
    summary = {
        "model": model_name,
        "acc_train": metrics_train["accuracy"],
        "ece_train": metrics_train["ece"],
        "mean_tu_train": float(unc_train["TU"].mean()),
        "mean_tu_gap": float(unc_gap["TU"].mean()),
        "mean_au_train": float(unc_train["AU"].mean()),
        "mean_au_gap": float(unc_gap["AU"].mean()),
        "mean_eu_train": float(unc_train["EU"].mean()),
        "mean_eu_gap": float(unc_gap["EU"].mean()),
        "au_eu_corr_train": au_eu_corr_train,
        "au_eu_corr_gap": au_eu_corr_gap,
        "inner_r_min": meta["inner_r_min"],
        "inner_r_max": meta["inner_r_max"],
        "outer_r_min": meta["outer_r_min"],
        "outer_r_max": meta["outer_r_max"],
    }
    save_statistics([summary], f"{model_name}_ring_ood_summary", subfolder=subfolder)
    
    return {
        "uncertainty_train": unc_train,
        "uncertainty_gap": unc_gap,
        "uncertainty_grid": unc_grid,
        "metrics_train": metrics_train,
        "probs_pred_train": probs_pred_train,
        "au_eu_correlation_train": au_eu_corr_train,
        "au_eu_correlation_gap": au_eu_corr_gap,
        "meta": meta,
    }


def _run_ring_ood_gl_experiment(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_gap: np.ndarray,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    subfolder: str,
) -> Dict[str, Any]:
    """Run ring OOD experiment for a GL model with heatmap visualization.
    
    Trains ONE model on ring data, then predicts on both rings and gap region.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} (GL) - Ring OOD Experiment")
    print(f"{'='*60}")
    
    # Create grid for visualization covering full region
    grid_res = 100
    outer_r_max = meta["outer_r_max"]
    margin = 0.5
    x_min, x_max = -outer_r_max - margin, outer_r_max + margin
    y_min, y_max = -outer_r_max - margin, outer_r_max + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    gl_samples = cfg.get("gl_samples", 100)
    
    # Train model ONCE on ring training data, then predict on all inputs
    if model_name == "mc_dropout_gl":
        model = train_mc_dropout_gl(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            p=cfg.get("dropout_p", 0.25),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32),
            n_samples=gl_samples
        )
        mc_samples = cfg.get("mc_samples", 100)
        mu_members_grid, sigma2_members_grid = mc_dropout_predict_gl(model, grid, mc_samples=mc_samples)
        mu_members_train, sigma2_members_train = mc_dropout_predict_gl(model, X_train, mc_samples=mc_samples)
        mu_members_gap, sigma2_members_gap = mc_dropout_predict_gl(model, X_gap, mc_samples=mc_samples)
        
    elif model_name == "deep_ensemble_gl":
        ensemble = train_deep_ensemble_gl(
            X_train, y_train,
            input_dim=cfg.get("input_dim", 2),
            num_classes=cfg.get("num_classes", 2),
            K=cfg.get("K", 10),
            epochs=cfg.get("epochs", 300),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 32),
            seed=cfg.get("seed", 42),
            n_samples=gl_samples
        )
        mu_members_grid, sigma2_members_grid = ensemble_predict_gl(ensemble, grid)
        mu_members_train, sigma2_members_train = ensemble_predict_gl(ensemble, X_train)
        mu_members_gap, sigma2_members_gap = ensemble_predict_gl(ensemble, X_gap)
        
    elif model_name == "bnn_gl":
        mcmc = train_bnn_gl(
            X_train, y_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            warmup=cfg.get("warmup", 200),
            samples=cfg.get("samples", 200),
            chains=cfg.get("chains", 1),
            seed=cfg.get("seed", 42)
        )
        mu_members_grid, sigma2_members_grid = bnn_predict_gl(
            mcmc, grid,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        mu_members_train, sigma2_members_train = bnn_predict_gl(
            mcmc, X_train,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
        mu_members_gap, sigma2_members_gap = bnn_predict_gl(
            mcmc, X_gap,
            hidden_width=cfg.get("hidden_width", 32),
            weight_scale=cfg.get("weight_scale", 1.0),
            num_classes=cfg.get("num_classes", 2)
        )
    else:
        raise ValueError(f"Unknown GL model: {model_name}")
    
    # Compute uncertainties
    unc_grid = gl_uncertainty(mu_members_grid, sigma2_members_grid, n_samples=gl_samples)
    unc_train = gl_uncertainty(mu_members_train, sigma2_members_train, n_samples=gl_samples)
    unc_gap = gl_uncertainty(mu_members_gap, sigma2_members_gap, n_samples=gl_samples)
    
    # Get predictive probabilities
    probs_pred_train = _predictive_probs_gl(mu_members_train, sigma2_members_train, n_samples=gl_samples)
    
    # Compute metrics on training data (rings)
    metrics_train = _evaluate_probs(probs_pred_train, y_train)
    
    print(f"  Ring Accuracy: {metrics_train['accuracy']:.4f}, ECE: {metrics_train['ece']:.4f}")
    print(f"  Ring  - Mean AU: {unc_train['AU'].mean():.4f} (norm: {unc_train['AU_norm'].mean():.4f}), "
          f"EU: {unc_train['EU'].mean():.4f} (norm: {unc_train['EU_norm'].mean():.4f})")
    print(f"  Gap   - Mean AU: {unc_gap['AU'].mean():.4f} (norm: {unc_gap['AU_norm'].mean():.4f}), "
          f"EU: {unc_gap['EU'].mean():.4f} (norm: {unc_gap['EU_norm'].mean():.4f})")
    
    # Save outputs
    save_classification_outputs(
        {
            "mu_members_train": mu_members_train,
            "sigma2_members_train": sigma2_members_train,
            "mu_members_gap": mu_members_gap,
            "sigma2_members_gap": sigma2_members_gap,
            "y_train": y_train,
            "x_train": X_train,
            "x_gap": X_gap,
            "grid": grid,
            "metrics_train": np.array(list(metrics_train.values()), dtype=np.float32),
        },
        model_name=model_name,
        experiment_name="ring_ood",
        subfolder=subfolder,
    )
    
    # Generate heatmap for full region with training points
    grid_extent = (x_min, x_max, y_min, y_max)
    _plot_uncertainty_heatmaps(
        X_eval=grid,
        uncertainty=unc_grid,
        X_test=X_train,
        y_test=y_train,
        probs_pred=probs_pred_train,
        model_name=model_name,
        experiment_name="ring_ood",
        subfolder=f"{subfolder}/heatmaps",
        is_it=False,
        X_train=X_train,
        y_train=y_train,
        grid_extent=grid_extent,
        grid_res=grid_res,
    )
    
    # Generate heatmap with gap points overlaid
    probs_pred_gap = _predictive_probs_gl(mu_members_gap, sigma2_members_gap, n_samples=gl_samples)
    _plot_uncertainty_heatmaps(
        X_eval=grid,
        uncertainty=unc_grid,
        X_test=X_gap,
        y_test=np.zeros(len(X_gap), dtype=np.int64),
        probs_pred=probs_pred_gap,
        model_name=model_name,
        experiment_name="ring_ood_gap",
        subfolder=f"{subfolder}/heatmaps",
        is_it=False,
        X_train=X_gap,
        y_train=np.zeros(len(X_gap), dtype=np.int64),
        grid_extent=grid_extent,
        grid_res=grid_res,
    )
    
    # Compute AU-EU correlation
    au_eu_corr_train = np.corrcoef(unc_train["AU_norm"], unc_train["EU_norm"])[0, 1]
    au_eu_corr_gap = np.corrcoef(unc_gap["AU_norm"], unc_gap["EU_norm"])[0, 1]
    
    # Generate comparison bar plot
    _plot_ring_gap_comparison(
        unc_train, unc_gap,
        au_eu_corr_train, au_eu_corr_gap,
        model_name, subfolder,
        is_it=False,
    )
    
    # Save summary statistics
    summary = {
        "model": model_name,
        "acc_train": metrics_train["accuracy"],
        "ece_train": metrics_train["ece"],
        "mean_au_train": float(unc_train["AU"].mean()),
        "mean_au_gap": float(unc_gap["AU"].mean()),
        "mean_eu_train": float(unc_train["EU"].mean()),
        "mean_eu_gap": float(unc_gap["EU"].mean()),
        "au_eu_corr_train": au_eu_corr_train,
        "au_eu_corr_gap": au_eu_corr_gap,
        "inner_r_min": meta["inner_r_min"],
        "inner_r_max": meta["inner_r_max"],
        "outer_r_min": meta["outer_r_min"],
        "outer_r_max": meta["outer_r_max"],
    }
    save_statistics([summary], f"{model_name}_ring_ood_summary", subfolder=subfolder)
    
    return {
        "uncertainty_train": unc_train,
        "uncertainty_gap": unc_gap,
        "uncertainty_grid": unc_grid,
        "metrics_train": metrics_train,
        "probs_pred_train": probs_pred_train,
        "au_eu_correlation_train": au_eu_corr_train,
        "au_eu_correlation_gap": au_eu_corr_gap,
        "meta": meta,
    }


# ============================================================================
# Ring OOD - Per-model wrapper functions
# ============================================================================

def run_mc_dropout_it_ring_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Run ring OOD experiment for MC Dropout IT.
    
    Args:
        base_cfg: Config for simulate_ring_ood_dataset
        seed: Random seed
        **model_kwargs: Model hyperparameters
    """
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_gap, meta = simulate_ring_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_ring_ood_it_experiment(
        model_name="mc_dropout_it",
        X_train=X_train, y_train=y_train,
        X_gap=X_gap,
        cfg=cfg,
        meta=meta,
        subfolder="classification/ring_ood/mc_dropout_it",
    )


def run_mc_dropout_gl_ring_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run ring OOD experiment for MC Dropout GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_gap, meta = simulate_ring_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_ring_ood_gl_experiment(
        model_name="mc_dropout_gl",
        X_train=X_train, y_train=y_train,
        X_gap=X_gap,
        cfg=cfg,
        meta=meta,
        subfolder="classification/ring_ood/mc_dropout_gl",
    )


def run_deep_ensemble_it_ring_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run ring OOD experiment for Deep Ensemble IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_gap, meta = simulate_ring_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_ring_ood_it_experiment(
        model_name="deep_ensemble_it",
        X_train=X_train, y_train=y_train,
        X_gap=X_gap,
        cfg=cfg,
        meta=meta,
        subfolder="classification/ring_ood/deep_ensemble_it",
    )


def run_deep_ensemble_gl_ring_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run ring OOD experiment for Deep Ensemble GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_gap, meta = simulate_ring_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_ring_ood_gl_experiment(
        model_name="deep_ensemble_gl",
        X_train=X_train, y_train=y_train,
        X_gap=X_gap,
        cfg=cfg,
        meta=meta,
        subfolder="classification/ring_ood/deep_ensemble_gl",
    )


def run_bnn_it_ring_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run ring OOD experiment for BNN IT."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_gap, meta = simulate_ring_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_ring_ood_it_experiment(
        model_name="bnn_it",
        X_train=X_train, y_train=y_train,
        X_gap=X_gap,
        cfg=cfg,
        meta=meta,
        subfolder="classification/ring_ood/bnn_it",
    )


def run_bnn_gl_ring_ood_experiment(
    base_cfg: Dict[str, Any],
    seed: int = 42,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run ring OOD experiment for BNN GL."""
    np.random.seed(seed)
    cfg = {**base_cfg, **model_kwargs, "seed": seed}
    
    X_train, y_train, X_gap, meta = simulate_ring_ood_dataset(cfg)
    cfg["num_classes"] = meta["num_classes"]
    cfg["input_dim"] = 2
    
    return _run_ring_ood_gl_experiment(
        model_name="bnn_gl",
        X_train=X_train, y_train=y_train,
        X_gap=X_gap,
        cfg=cfg,
        meta=meta,
        subfolder="classification/ring_ood/bnn_gl",
    )


def run_ring_ood_experiment(cfg: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Legacy: Run ring OOD experiment for any model type."""
    if model_type == "mc_dropout_it":
        return run_mc_dropout_it_ring_ood_experiment(cfg)
    elif model_type == "mc_dropout_gl":
        return run_mc_dropout_gl_ring_ood_experiment(cfg)
    elif model_type == "deep_ensemble_it":
        return run_deep_ensemble_it_ring_ood_experiment(cfg)
    elif model_type == "deep_ensemble_gl":
        return run_deep_ensemble_gl_ring_ood_experiment(cfg)
    elif model_type == "bnn_it":
        return run_bnn_it_ring_ood_experiment(cfg)
    elif model_type == "bnn_gl":
        return run_bnn_gl_ring_ood_experiment(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
