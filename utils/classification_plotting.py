"""
Plotting utilities for classification uncertainty experiments.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from utils.results_save import save_plot


def plot_metric_curves(
    x_values: List[float],
    metrics: Dict[str, List[float]],
    title: str,
    xlabel: str,
    subfolder: str,
    ylabel: str | None = None,
):
    """Plot metric curves over a sweep variable.
    
    Args:
        x_values: X-axis values (sweep variable)
        metrics: Dictionary mapping metric names to lists of values
        title: Plot title
        xlabel: X-axis label
        subfolder: Subfolder for saving the plot
        ylabel: Y-axis label (auto-detected if None based on metric names)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in metrics.items():
        ax.plot(x_values, values, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    
    # Auto-detect ylabel if not specified
    if ylabel is None:
        # Check if all metrics are normalized (end with _norm)
        all_normalized = all(name.endswith("_norm") for name in metrics.keys())
        if all_normalized:
            ylabel = "Normalized Uncertainty [0, 1]"
        else:
            ylabel = "Value"
    ax.set_ylabel(ylabel)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    save_plot(fig, title, subfolder=subfolder)
    plt.show()
    plt.close(fig)


def plot_uncertainty_heatmap(
    grid_xy: np.ndarray,
    values: np.ndarray,
    title: str,
    subfolder: str,
    vmin: float | None = None,
    vmax: float | None = None,
    normalized: bool = False,
):
    """Plot uncertainty values as a 2D heatmap.
    
    Args:
        grid_xy: Grid coordinates [N, 2]
        values: Uncertainty values to plot [N]
        title: Plot title
        subfolder: Subfolder for saving the plot
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        normalized: If True, fix colorbar to [0, 1] range
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    x = grid_xy[:, 0]
    y = grid_xy[:, 1]
    
    # For normalized heatmaps, fix the color range to [0, 1]
    if normalized:
        vmin = 0.0
        vmax = 1.0
    
    sc = ax.scatter(x, y, c=values, s=12, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    if normalized:
        cbar.set_label("Normalized Uncertainty")
    plt.tight_layout()
    save_plot(fig, title, subfolder=subfolder)
    plt.show()
    plt.close(fig)


def plot_misclassifications(
    X_test: np.ndarray,
    y_true: np.ndarray,
    probs_pred: np.ndarray,
    title: str,
    subfolder: str,
):
    """Plot test data with misclassified points highlighted.
    
    Args:
        X_test: Test data coordinates [N, 2]
        y_true: True labels [N] (OOD samples have label -1)
        probs_pred: Predicted probabilities [N, K]
        title: Plot title
        subfolder: Subfolder for saving
    """
    # Filter out OOD samples (label == -1) for evaluation
    valid_mask = y_true >= 0
    if not np.any(valid_mask):
        # No valid samples to plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"{title}\nNo valid samples (all OOD)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.tight_layout()
        save_plot(fig, title, subfolder=subfolder)
        plt.show()
        plt.close(fig)
        return
    
    X_test_valid = X_test[valid_mask]
    y_true_valid = y_true[valid_mask]
    probs_pred_valid = probs_pred[valid_mask]
    
    y_pred = np.argmax(probs_pred_valid, axis=1)
    correct = y_pred == y_true_valid
    incorrect = ~correct
    
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    # Plot correctly classified points by true class
    for cls in np.unique(y_true_valid):
        mask = (y_true_valid == cls) & correct
        ax.scatter(X_test_valid[mask, 0], X_test_valid[mask, 1], 
                  s=20, color=colors[int(cls)], alpha=0.7, label=f"Class {cls}")
    
    # Highlight misclassified points with red circles
    if np.any(incorrect):
        ax.scatter(X_test_valid[incorrect, 0], X_test_valid[incorrect, 1], 
                  s=80, facecolors='none', edgecolors='red', linewidth=2, 
                  marker='o', label="Misclassified")
    
    accuracy = correct.mean()
    ax.set_title(f"{title}\nAccuracy: {accuracy:.2%}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, title, subfolder=subfolder)
    plt.show()
    plt.close(fig)


def plot_uncertainty_panel(
    X_eval: np.ndarray,
    uncertainty: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_true: np.ndarray,
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
    """Generate a panel plot with all uncertainty heatmaps and misclassification.
    
    Args:
        X_eval: Evaluation grid coordinates [N, 2]
        uncertainty: Dict containing normalized uncertainty values (AU_norm, EU_norm, TU_norm)
        X_test: Test data coordinates [M, 2]
        y_true: True labels [M] (OOD samples have label -1)
        probs_pred: Predicted probabilities [M, K]
        model_name: Name of the model (e.g., "mc_dropout_it")
        experiment_name: Name of the experiment condition (e.g., "rcd_3.0")
        subfolder: Subfolder for saving plots
        is_it: True for IT models (has TU, 2x2 layout), False for GL models (1x3 layout)
        X_train: Optional training data coordinates [L, 2] for overlay
        y_train: Optional training labels [L] for overlay coloring
        grid_extent: Tuple (x_min, x_max, y_min, y_max) for imshow extent
        grid_res: Grid resolution (e.g., 100 for 100x100)
    """
    # Filter out OOD samples (label == -1) for accuracy calculation
    valid_mask = y_true >= 0
    if np.any(valid_mask):
        y_pred = np.argmax(probs_pred[valid_mask], axis=1)
        y_true_valid = y_true[valid_mask]
        correct = y_pred == y_true_valid
        incorrect = ~correct
        accuracy = correct.mean()
        X_test_valid = X_test[valid_mask]
        y_true_valid_for_plot = y_true_valid
    else:
        # No valid samples
        accuracy = 0.0
        correct = np.array([], dtype=bool)
        incorrect = np.array([], dtype=bool)
        X_test_valid = X_test
        y_true_valid_for_plot = y_true
    
    colors = ["tab:blue", "tab:orange", "tab:green"]
    
    # Use grid-based imshow if grid_extent and grid_res are provided
    use_imshow = grid_extent is not None and grid_res is not None
    
    if is_it:
        # IT models: 2x2 grid with AU, EU, TU, Misclassification
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axs = axes.flatten()
        
        if use_imshow:
            x_min, x_max, y_min, y_max = grid_extent
            extent = [x_min, x_max, y_min, y_max]
            
            # Reshape uncertainty to 2D grid
            AU_grid = uncertainty["AU_norm"].reshape(grid_res, grid_res)
            EU_grid = uncertainty["EU_norm"].reshape(grid_res, grid_res)
            TU_grid = uncertainty["TU_norm"].reshape(grid_res, grid_res)
            
            # Panel 1: AU
            im0 = axs[0].imshow(AU_grid, extent=extent, origin='lower', 
                               cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            axs[0].set_title(f"AU (Aleatoric)")
            axs[0].set_xlabel("x1")
            axs[0].set_ylabel("x2")
            plt.colorbar(im0, ax=axs[0], shrink=0.8, label="Normalized")
            # Overlay training data points with edge-only markers
            if X_train is not None and y_train is not None:
                for cls in range(int(y_train.max()) + 1):
                    mask = y_train == cls
                    axs[0].scatter(X_train[mask, 0], X_train[mask, 1],
                                  facecolors='none', edgecolors=colors[cls],
                                  s=20, linewidths=1.0, alpha=0.4)
            
            # Panel 2: EU
            im1 = axs[1].imshow(EU_grid, extent=extent, origin='lower', 
                               cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            axs[1].set_title(f"EU (Epistemic)")
            axs[1].set_xlabel("x1")
            axs[1].set_ylabel("x2")
            plt.colorbar(im1, ax=axs[1], shrink=0.8, label="Normalized")
            # Overlay training data points with edge-only markers
            if X_train is not None and y_train is not None:
                for cls in range(int(y_train.max()) + 1):
                    mask = y_train == cls
                    axs[1].scatter(X_train[mask, 0], X_train[mask, 1],
                                  facecolors='none', edgecolors=colors[cls],
                                  s=20, linewidths=1.0, alpha=0.4)
            
            # Panel 3: TU
            im2 = axs[2].imshow(TU_grid, extent=extent, origin='lower', 
                               cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            axs[2].set_title(f"TU (Total)")
            axs[2].set_xlabel("x1")
            axs[2].set_ylabel("x2")
            plt.colorbar(im2, ax=axs[2], shrink=0.8, label="Normalized")
            # Overlay training data points with edge-only markers
            if X_train is not None and y_train is not None:
                for cls in range(int(y_train.max()) + 1):
                    mask = y_train == cls
                    axs[2].scatter(X_train[mask, 0], X_train[mask, 1],
                                  facecolors='none', edgecolors=colors[cls],
                                  s=20, linewidths=1.0, alpha=0.4)
        else:
            # Fallback to scatter if no grid info
            sc0 = axs[0].scatter(X_eval[:, 0], X_eval[:, 1], c=uncertainty["AU_norm"], 
                                 s=12, cmap="viridis", vmin=0.0, vmax=1.0)
            axs[0].set_title(f"AU (Aleatoric)")
            axs[0].set_xlabel("x1")
            axs[0].set_ylabel("x2")
            axs[0].set_aspect("equal", adjustable="box")
            plt.colorbar(sc0, ax=axs[0], shrink=0.8, label="Normalized")
            
            sc1 = axs[1].scatter(X_eval[:, 0], X_eval[:, 1], c=uncertainty["EU_norm"], 
                                 s=12, cmap="viridis", vmin=0.0, vmax=1.0)
            axs[1].set_title(f"EU (Epistemic)")
            axs[1].set_xlabel("x1")
            axs[1].set_ylabel("x2")
            axs[1].set_aspect("equal", adjustable="box")
            plt.colorbar(sc1, ax=axs[1], shrink=0.8, label="Normalized")
            
            sc2 = axs[2].scatter(X_eval[:, 0], X_eval[:, 1], c=uncertainty["TU_norm"], 
                                 s=12, cmap="viridis", vmin=0.0, vmax=1.0)
            axs[2].set_title(f"TU (Total)")
            axs[2].set_xlabel("x1")
            axs[2].set_ylabel("x2")
            axs[2].set_aspect("equal", adjustable="box")
            plt.colorbar(sc2, ax=axs[2], shrink=0.8, label="Normalized")
        
        # Panel 4: Training Data
        if X_train is not None and y_train is not None:
            for cls in range(int(y_train.max()) + 1):
                mask = y_train == cls
                axs[3].scatter(X_train[mask, 0], X_train[mask, 1], 
                              c=colors[cls], s=20, alpha=0.7, label=f"Class {cls}")
        axs[3].set_title("Training Data")
        axs[3].set_xlabel("x1")
        axs[3].set_ylabel("x2")
        axs[3].set_aspect("equal", adjustable="box")
        axs[3].legend(loc="best", fontsize=8)
        axs[3].grid(True, alpha=0.3)
        
    else:
        # GL models: 1x3 row with AU_GL, EU_GL, Training Data
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        if use_imshow:
            x_min, x_max, y_min, y_max = grid_extent
            extent = [x_min, x_max, y_min, y_max]
            
            # Reshape uncertainty to 2D grid
            AU_grid = uncertainty["AU_norm"].reshape(grid_res, grid_res)
            EU_grid = uncertainty["EU_norm"].reshape(grid_res, grid_res)
            
            # Panel 1: AU_GL
            im0 = axs[0].imshow(AU_grid, extent=extent, origin='lower', 
                               cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            axs[0].set_title(f"AU_GL (Aleatoric)")
            axs[0].set_xlabel("x1")
            axs[0].set_ylabel("x2")
            plt.colorbar(im0, ax=axs[0], shrink=0.8, label="Normalized")
            # Overlay training data points with edge-only markers
            if X_train is not None and y_train is not None:
                for cls in range(int(y_train.max()) + 1):
                    mask = y_train == cls
                    axs[0].scatter(X_train[mask, 0], X_train[mask, 1],
                                  facecolors='none', edgecolors=colors[cls],
                                  s=20, linewidths=1.0, alpha=0.4)
            
            # Panel 2: EU_GL
            im1 = axs[1].imshow(EU_grid, extent=extent, origin='lower', 
                               cmap='viridis', vmin=0.0, vmax=1.0, aspect='equal')
            axs[1].set_title(f"EU_GL (Epistemic)")
            axs[1].set_xlabel("x1")
            axs[1].set_ylabel("x2")
            plt.colorbar(im1, ax=axs[1], shrink=0.8, label="Normalized")
            # Overlay training data points with edge-only markers
            if X_train is not None and y_train is not None:
                for cls in range(int(y_train.max()) + 1):
                    mask = y_train == cls
                    axs[1].scatter(X_train[mask, 0], X_train[mask, 1],
                                  facecolors='none', edgecolors=colors[cls],
                                  s=20, linewidths=1.0, alpha=0.4)
        else:
            # Fallback to scatter if no grid info
            sc0 = axs[0].scatter(X_eval[:, 0], X_eval[:, 1], c=uncertainty["AU_norm"], 
                                 s=12, cmap="viridis", vmin=0.0, vmax=1.0)
            axs[0].set_title(f"AU_GL (Aleatoric)")
            axs[0].set_xlabel("x1")
            axs[0].set_ylabel("x2")
            axs[0].set_aspect("equal", adjustable="box")
            plt.colorbar(sc0, ax=axs[0], shrink=0.8, label="Normalized")
            
            sc1 = axs[1].scatter(X_eval[:, 0], X_eval[:, 1], c=uncertainty["EU_norm"], 
                                 s=12, cmap="viridis", vmin=0.0, vmax=1.0)
            axs[1].set_title(f"EU_GL (Epistemic)")
            axs[1].set_xlabel("x1")
            axs[1].set_ylabel("x2")
            axs[1].set_aspect("equal", adjustable="box")
            plt.colorbar(sc1, ax=axs[1], shrink=0.8, label="Normalized")
        
        # Panel 3: Training Data
        if X_train is not None and y_train is not None:
            for cls in range(int(y_train.max()) + 1):
                mask = y_train == cls
                axs[2].scatter(X_train[mask, 0], X_train[mask, 1], 
                              c=colors[cls], s=20, alpha=0.7, label=f"Class {cls}")
        axs[2].set_title("Training Data")
        axs[2].set_xlabel("x1")
        axs[2].set_ylabel("x2")
        axs[2].set_aspect("equal", adjustable="box")
        axs[2].legend(loc="best", fontsize=8)
        axs[2].grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f"{model_name} - {experiment_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save panel plot
    title = f"{model_name} panel {experiment_name}"
    save_plot(fig, title, subfolder=subfolder)
    plt.show()
    plt.close(fig)


def plot_au_eu_correlation(
    au: np.ndarray,
    eu: np.ndarray,
    model_name: str,
    experiment_name: str,
    subfolder: str,
):
    """Plot AU vs EU scatter with regression line and correlation coefficient.
    
    Args:
        au: Aleatoric uncertainty values [N]
        eu: Epistemic uncertainty values [N]
        model_name: Name of the model
        experiment_name: Name of the experiment condition
        subfolder: Subfolder for saving plots
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Scatter plot
    ax.scatter(au, eu, alpha=0.3, s=10, c='steelblue')
    
    # Compute correlation
    corr = np.corrcoef(au, eu)[0, 1]
    
    # Regression line
    if len(au) > 1:
        slope, intercept = np.polyfit(au, eu, 1)
        x_line = np.linspace(au.min(), au.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')
    
    ax.set_xlabel("AU (Aleatoric Uncertainty)", fontsize=12)
    ax.set_ylabel("EU (Epistemic Uncertainty)", fontsize=12)
    ax.set_title(f"AU vs EU Correlation\nr = {corr:.3f}", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Equal aspect for better visualization
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    
    # Save correlation plot
    title = f"{model_name} AU_EU_correlation {experiment_name}"
    save_plot(fig, title, subfolder=subfolder)
    plt.show()
    plt.close(fig)
