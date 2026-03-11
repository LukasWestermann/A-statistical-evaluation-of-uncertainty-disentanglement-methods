"""
Single-panel plots for Deep Ensemble sin/heteroscedastic OOD: predictive mean with one band each.
- One plot: mean +/- aleatoric variance band.
- One plot: mean +/- epistemic variance band.
Loads from results/ood/outputs/ood/heteroscedastic/sin/ (latest *Deep_Ensemble*raw_outputs*.npz).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

OOD_RANGES = [(10, 15)]


def _build_ood_mask(x_grid, ood_ranges):
    x_flat = np.asarray(x_grid).ravel()
    ood_mask = np.zeros(len(x_flat), dtype=bool)
    for ood_start, ood_end in ood_ranges:
        ood_mask |= (x_flat >= ood_start) & (x_flat <= ood_end)
    return ood_mask


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


def main():
    search_dir = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
    save_dir = project_root / "results" / "ood" / "plots"
    if not search_dir.exists():
        print("Directory does not exist:", search_dir)
        return

    npz_files = sorted(search_dir.glob("*Deep_Ensemble*raw_outputs*.npz"))
    if not npz_files:
        print("No Deep Ensemble npz found under", search_dir)
        return

    npz_path = npz_files[-1]
    print("Loading:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    mu_samples = np.asarray(data["mu_samples"])
    sigma2_samples = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    y_grid_clean = np.asarray(data["y_grid_clean"])
    mu_samples, sigma2_samples = _ensure_samples_first(mu_samples, sigma2_samples, x_grid)

    x_train = data["x_train_subset"] if "x_train_subset" in data else None
    y_train = data["y_train_subset"] if "y_train_subset" in data else None
    if x_train is not None:
        x_train = np.asarray(x_train)
    if y_train is not None:
        y_train = np.asarray(y_train)

    mu_pred = np.mean(mu_samples, axis=0).squeeze()
    ale_var = np.mean(sigma2_samples, axis=0).squeeze()
    epi_var = np.var(mu_samples, axis=0).squeeze()

    ood_mask = _build_ood_mask(x_grid, OOD_RANGES)
    id_mask = ~ood_mask
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean.ravel()

    boundary_x = []
    if np.any(ood_mask):
        transitions = np.where(np.diff(ood_mask.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = list(x[transitions + 1])

    x_train_flat = None
    y_train_flat = None
    if x_train is not None and y_train is not None:
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train.ravel()
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train.ravel()

    def shade_ood(ax):
        for ood_start, ood_end in OOD_RANGES:
            ax.axvspan(ood_start, ood_end, alpha=0.35, color="lightgrey", zorder=0)

    def add_common(ax):
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(
                x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none"
            )
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax.plot(x[id_mask], y_clean_flat[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y_clean_flat[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
            ax.scatter(
                x[ood_mask], y_clean_flat[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5
            )
        ax.set_xlabel("x", fontsize=12, fontweight="bold")
        ax.set_ylabel("y", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
        ax.tick_params(labelsize=10)

    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- Plot 1: Aleatoric band (mean +/- ale_var) -----
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(x, mu_pred, "b-", linewidth=2.5, label="Predictive mean", zorder=5)
    ax1.fill_between(
        x,
        mu_pred - ale_var,
        mu_pred + ale_var,
        alpha=0.35,
        color="#06A77D",
        label="±aleatoric variance",
        zorder=1,
    )
    add_common(ax1)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.set_title(
        "Deep Ensemble - Sin (heteroscedastic) OOD - Predictive mean ± aleatoric variance",
        fontsize=14, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    path_ale = save_dir / "panel_Deep_Ensemble_sin_hetero_ood_aleatoric_band.png"
    fig1.savefig(path_ale, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved:", path_ale)

    # ----- Plot 2: Epistemic band (mean +/- epi_var) -----
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(x, mu_pred, "b-", linewidth=2.5, label="Predictive mean", zorder=5)
    ax2.fill_between(
        x,
        mu_pred - epi_var,
        mu_pred + epi_var,
        alpha=0.35,
        color="#F18F01",
        label="±epistemic variance",
        zorder=1,
    )
    add_common(ax2)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.set_title(
        "Deep Ensemble - Sin (heteroscedastic) OOD - Predictive mean ± epistemic variance",
        fontsize=14, fontweight="bold", pad=10,
    )
    plt.tight_layout()
    path_epi = save_dir / "panel_Deep_Ensemble_sin_hetero_ood_epistemic_band.png"
    fig2.savefig(path_epi, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved:", path_epi)


if __name__ == "__main__":
    main()
