"""
One-off script: 2x2 panel (Au / Eu variance bands) for OOD, linear, heteroscedastic,
using existing raw_outputs .npz files (no retraining).

Layout:
- 2 rows x 2 columns = 4 models:
    [ BAMLSS        | MC Dropout   ]
    [ BNN           | Deep Ensemble]

Each subplot shows:
- training data
- true function
- predictive mean
- ± sqrt(ale_var) band (aleatoric, Au)
- ± sqrt(epi_var) band (epistemic, Eu)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# Same OOD range as used in other OOD plotting scripts
OOD_RANGES = [(10, 15)]

# Models in the order requested: BAMLSS, MC Dropout, BNN, Deep Ensemble
MODELS = [
    ("*BAMLSS*", "BAMLSS"),
    ("*MC_Dropout*", "MC Dropout"),
    ("*BNN*", "BNN"),
    ("*Deep_Ensemble*", "Deep Ensemble"),
]


def _build_ood_mask(x_grid, ood_ranges):
    x_flat = np.asarray(x_grid).ravel()
    ood_mask = np.zeros(len(x_flat), dtype=bool)
    for ood_start, ood_end in ood_ranges:
        ood_mask |= (x_flat >= ood_start) & (x_flat <= ood_end)
    return ood_mask


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    """Ensure samples dimension is first: [S, N]."""
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        # [N, S] -> [S, N]
        mu = mu.T
        sig = sig.T
    return mu, sig


def load_model_data(npz_path: Path):
    """Load npz and return dict with variance quantities needed for Au/Eu bands."""
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

    # Predictive mean and variance decomposition
    mu_pred = np.mean(mu_samples, axis=0).squeeze()
    ale_var = np.mean(sigma2_samples, axis=0).squeeze()
    epi_var = np.var(mu_samples, axis=0).squeeze()

    ood_mask = _build_ood_mask(x_grid, OOD_RANGES)
    id_mask = ~ood_mask

    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean.ravel()

    x_train_flat = None
    y_train_flat = None
    if x_train is not None and y_train is not None:
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train.ravel()
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train.ravel()

    return {
        "x": x,
        "y_clean_flat": y_clean_flat,
        "mu_pred": mu_pred,
        "ale_var": ale_var,
        "epi_var": epi_var,
        "ood_mask": ood_mask,
        "id_mask": id_mask,
        "x_train_flat": x_train_flat,
        "y_train_flat": y_train_flat,
    }


def _shade_ood(ax):
    for ood_start, ood_end in OOD_RANGES:
        ax.axvspan(ood_start, ood_end, alpha=0.3, color="lightgrey", zorder=0)


def _plot_single_model(ax, data, title):
    """Plot Au and Eu bands for one model on a single axes."""
    _shade_ood(ax)

    if data["x_train_flat"] is not None and data["y_train_flat"] is not None:
        ax.scatter(
            data["x_train_flat"],
            data["y_train_flat"],
            alpha=0.15,
            s=15,
            color="#2E86AB",
            zorder=3,
            edgecolors="none",
        )

    x = data["x"]
    y_clean = data["y_clean_flat"]
    id_mask, ood_mask = data["id_mask"], data["ood_mask"]

    # True function
    ax.plot(
        x[id_mask],
        y_clean[id_mask],
        "r--",
        linewidth=2,
        alpha=0.9,
        label="True function",
        zorder=4,
    )
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
        ax.scatter(
            x[ood_mask],
            y_clean[ood_mask],
            s=20,
            color="red",
            alpha=0.4,
            marker="x",
            zorder=6,
            linewidths=1.5,
        )

    mu = data["mu_pred"]
    ale_std = np.sqrt(data["ale_var"])
    epi_std = np.sqrt(data["epi_var"])

    # Predictive mean
    ax.plot(x, mu, "k-", linewidth=2, label="Predictive mean", zorder=5)

    # Aleatoric band (Au)
    ax.fill_between(
        x,
        mu - ale_std,
        mu + ale_std,
        alpha=0.35,
        color="#06A77D",
        label="±σ(aleatoric)",
        zorder=1,
    )

    # Epistemic band (Eu)
    ax.fill_between(
        x,
        mu - epi_std,
        mu + epi_std,
        alpha=0.35,
        color="#F18F01",
        label="±σ(epistemic)",
        zorder=2,
    )

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=9)


def create_2x2_ood_linear_hetero_panel(condition_data_list, display_names, save_path: Path):
    """2x2 panel (one subplot per model) with Au and Eu variance bands."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    # Map models to positions:
    # 0: BAMLSS, 1: MC Dropout, 2: BNN, 3: Deep Ensemble
    # -> (0,0), (0,1), (1,0), (1,1)
    for idx, (ax, data, name) in enumerate(
        zip(axes.flatten(), condition_data_list, display_names)
    ):
        if data is None:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
            continue
        _plot_single_model(ax, data, name)

    # Shared labels and legend
    for ax in axes[1, :]:
        ax.set_xlabel("x", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("y", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("y", fontsize=11, fontweight="bold")

    # Single legend in top-left axes
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=8, framealpha=0.9, loc="upper left")

    fig.suptitle(
        "OOD — Linear, heteroscedastic — Au & Eu variance bands",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def main():
    # Only linear, heteroscedastic condition
    func_type = "linear"
    noise_type = "heteroscedastic"

    outputs_base = project_root / "results" / "ood" / "outputs" / "ood"
    search_dir = outputs_base / noise_type / func_type
    if not search_dir.exists():
        print("Outputs directory does not exist:", search_dir)
        return

    print("Searching raw_outputs under:", search_dir)
    condition_data_list = []
    display_names = []

    for pattern, display_name in MODELS:
        npz_files = sorted(search_dir.glob(f"{pattern}raw_outputs*.npz"))
        if not npz_files:
            print("  No npz for", display_name, "under", search_dir, "- leaving subplot empty.")
            condition_data_list.append(None)
            display_names.append(display_name)
            continue
        npz_path = npz_files[-1]
        print("  Using", npz_path, "for", display_name)
        try:
            data = load_model_data(npz_path)
            condition_data_list.append(data)
        except Exception as e:
            print("  Error loading", npz_path, ":", e)
            condition_data_list.append(None)
        display_names.append(display_name)

    if all(d is None for d in condition_data_list):
        print("No data available for linear, heteroscedastic OOD – nothing to plot.")
        return

    save_dir = project_root / "results" / "ood" / "plots"
    save_path = save_dir / "panel_ood_2x2_linear_heteroscedastic_Au_Eu_variance.png"
    create_2x2_ood_linear_hetero_panel(condition_data_list, display_names, save_path)


if __name__ == "__main__":
    main()

