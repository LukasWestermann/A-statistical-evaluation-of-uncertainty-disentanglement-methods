"""
2x4 panel plots (2 rows AU/EU x 4 models) for sample_size: sin, heteroscedastic, 100%.
Generates variance (mean +/- std bands) and entropy (line plots) figures.
Loads npz from results/sample_size/outputs/sample_size/<noise_type>/<func_type>/.
No OOD regions; titles indicate "Sample size 100%".
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical

OUTPUTS_BASE = project_root / "results" / "sample_size" / "outputs" / "sample_size"
FUNC_TYPE = "sin"
NOISE_TYPE = "heteroscedastic"
PCT = 100

# No OOD shading
OOD_RANGES = []

MODELS = [
    ("*Deep_Ensemble*pct100*raw_outputs*", "Deep Ensemble"),
    ("*MC_Dropout*pct100*raw_outputs*", "MC Dropout"),
    ("*BNN*pct100*raw_outputs*", "BNN"),
    ("*BAMLSS*pct100*raw_outputs*", "BAMLSS"),
]


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


def load_model_data(npz_path):
    """Load npz and return dict with variance and entropy quantities. No OOD: all points ID."""
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

    ent = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
    ale_entropy = np.asarray(ent["aleatoric"]).squeeze()
    epi_entropy = np.asarray(ent["epistemic"]).squeeze()

    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    n = len(x)
    ood_mask = np.zeros(n, dtype=bool)
    id_mask = np.ones(n, dtype=bool)
    boundary_x = []

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
        "ale_entropy": ale_entropy,
        "epi_entropy": epi_entropy,
        "ood_mask": ood_mask,
        "id_mask": id_mask,
        "boundary_x": boundary_x,
        "x_train_flat": x_train_flat,
        "y_train_flat": y_train_flat,
    }


def _shade_ood(ax):
    for ood_start, ood_end in OOD_RANGES:
        ax.axvspan(ood_start, ood_end, alpha=0.35, color="lightgrey", zorder=0)


def _add_common_variance(ax, data):
    _shade_ood(ax)
    if data["x_train_flat"] is not None and data["y_train_flat"] is not None:
        ax.scatter(
            data["x_train_flat"], data["y_train_flat"],
            alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none"
        )
    for bx in data["boundary_x"]:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    x, y_clean = data["x"], data["y_clean_flat"]
    id_mask, ood_mask = data["id_mask"], data["ood_mask"]
    ax.plot(x[id_mask], y_clean[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
        ax.scatter(x[ood_mask], y_clean[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=9)


def create_2x4_variance_panel(condition_data_list, display_names, save_path):
    """2 rows (AU, EU) x 4 cols (models). Variance = mean +/- sqrt(var) bands."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    for row, (band_key_ale, band_key_epi, color, label) in enumerate([
        ("ale_var", None, "#06A77D", "±σ(aleatoric)"),
        (None, "epi_var", "#F18F01", "±σ(epistemic)"),
    ]):
        for col in range(4):
            ax = axes[row, col]
            if col >= len(condition_data_list) or condition_data_list[col] is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                ax.set_ylabel("y", fontsize=10)
                if row == 0:
                    ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
                continue
            data = condition_data_list[col]
            x = data["x"]
            mu_pred = data["mu_pred"]
            var = data[band_key_ale] if band_key_ale else data[band_key_epi]
            ax.plot(x, mu_pred, "b-", linewidth=2, label="Predictive mean", zorder=5)
            ax.fill_between(x, mu_pred - np.sqrt(var), mu_pred + np.sqrt(var),
                            alpha=0.35, color=color, label=label, zorder=1)
            _add_common_variance(ax, data)
            ax.set_ylabel("y", fontsize=10)
            if row == 0:
                ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    for col in range(4):
        axes[1, col].set_xlabel("x", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("y\n(Aleatoric)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("y\n(Epistemic)", fontsize=10, fontweight="bold")
    fig.suptitle("Sample size 100% — Sinusoidal, heteroscedastic — Variance (std bands)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def _add_common_entropy(ax, ax_twin, data, entropy_color):
    _shade_ood(ax)
    if data["x_train_flat"] is not None and data["y_train_flat"] is not None:
        ax.scatter(data["x_train_flat"], data["y_train_flat"], alpha=0.1, s=10, color="blue", zorder=3)
    for bx in data["boundary_x"]:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    x, y_clean = data["x"], data["y_clean_flat"]
    id_mask, ood_mask = data["id_mask"], data["ood_mask"]
    ax.plot(x[id_mask], data["mu_pred"][id_mask], "b-", linewidth=1.2, alpha=0.5, label="Predictive mean")
    ax.plot(x[ood_mask], data["mu_pred"][ood_mask], "b-", linewidth=1.2, alpha=0.5)
    ax.plot(x[id_mask], y_clean[id_mask], "r-", linewidth=1.5, alpha=0.8, label="True function")
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean[ood_mask], "r-", linewidth=1.5, alpha=0.8)
        ax.scatter(x[ood_mask], y_clean[ood_mask], s=8, color="red", alpha=0.3, marker="x", zorder=4)
    ax_twin.tick_params(axis="y", labelcolor=entropy_color)
    ax.grid(True, alpha=0.3)


def create_2x4_entropy_panel(condition_data_list, display_names, save_path):
    """2 rows (AU, EU) x 4 cols (models). Entropy line plots (twin axis)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    for row, (ent_key, color, label) in enumerate([
        ("ale_entropy", "green", "Aleatoric entropy (nats)"),
        ("epi_entropy", "#C41E3A", "Epistemic entropy (nats)"),
    ]):
        for col in range(4):
            ax = axes[row, col]
            if col >= len(condition_data_list) or condition_data_list[col] is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                if row == 0:
                    ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
                continue
            data = condition_data_list[col]
            ax_twin = ax.twinx()
            x = data["x"]
            ent = data[ent_key]
            id_mask, ood_mask = data["id_mask"], data["ood_mask"]
            ax_twin.plot(x[id_mask], ent[id_mask], "-", color=color, linewidth=2, label=label)
            ax_twin.plot(x[ood_mask], ent[ood_mask], "-", color=color, linewidth=2, alpha=0.7)
            _add_common_entropy(ax, ax_twin, data, color)
            ax.set_ylabel("y", fontsize=10)
            ax_twin.set_ylabel("Entropy (nats)", fontsize=9, color=color)
            if row == 0:
                ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
            ax.legend(loc="upper left", fontsize=8)
            ax_twin.legend(loc="upper right", fontsize=8)

    for col in range(4):
        axes[1, col].set_xlabel("x", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("y\n(Aleatoric)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("y\n(Epistemic)", fontsize=10, fontweight="bold")
    fig.suptitle("Sample size 100% — Sinusoidal, heteroscedastic — Entropy (line plots)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def main():
    search_dir = OUTPUTS_BASE / NOISE_TYPE / FUNC_TYPE
    save_dir = project_root / "results" / "sample_size" / "plots"
    display_names = [m[1] for m in MODELS]

    if not search_dir.exists():
        print("Directory missing:", search_dir)
        print("Run the sample_size experiment once (e.g. from Experiments or notebook) so save_model_outputs writes npz files.")
        return

    condition_data_list = []
    for pattern, display_name in MODELS:
        npz_files = sorted(search_dir.glob(f"{pattern}.npz"))
        if not npz_files:
            print("  No npz for", display_name, "under", search_dir, "- leaving column empty.")
            condition_data_list.append(None)
            continue
        npz_path = npz_files[-1]
        try:
            data = load_model_data(npz_path)
            condition_data_list.append(data)
        except Exception as e:
            print("  Error loading", npz_path, ":", e)
            condition_data_list.append(None)

    if all(d is None for d in condition_data_list):
        print("No data for sin/heteroscedastic/100% - skipping.")
        print("Run the sample_size experiment once so save_model_outputs writes npz files under", search_dir)
        return

    save_path_var = save_dir / "panel_sample_size_2x4_sin_hetero_100_variance.png"
    save_path_ent = save_dir / "panel_sample_size_2x4_sin_hetero_100_entropy.png"
    create_2x4_variance_panel(condition_data_list, display_names, save_path_var)
    create_2x4_entropy_panel(condition_data_list, display_names, save_path_ent)
    print("Done.")


if __name__ == "__main__":
    main()
