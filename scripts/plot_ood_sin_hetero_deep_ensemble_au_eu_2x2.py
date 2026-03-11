"""
One-off script: 2x2 panel for sin/heteroscedastic OOD (AU & EU only) per model.
Left column = variance std; right column = entropy line plots.
Loads from existing saved npz under results/ood/outputs/ood/heteroscedastic/sin/.
Generates panels for Deep_Ensemble, MC_Dropout, BNN, and BAMLSS.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project root for imports and paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical

# OOD convention for this plot
TRAIN_RANGE = (-5, 10)
OOD_RANGES = [(10, 15)]

# Models to process: (glob pattern in filename, display name for title)
MODELS = [
    ("*Deep_Ensemble*", "Deep Ensemble"),
    ("*MC_Dropout*", "MC Dropout"),
    ("*BNN*", "BNN"),
    ("*BAMLSS*", "BAMLSS"),
]


def _build_ood_mask(x_grid, ood_ranges):
    """Build boolean OOD mask from x_grid."""
    x_flat = np.asarray(x_grid).ravel()
    ood_mask = np.zeros(len(x_flat), dtype=bool)
    for ood_start, ood_end in ood_ranges:
        ood_mask |= (x_flat >= ood_start) & (x_flat <= ood_end)
    return ood_mask


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    """Ensure shape [S, N] (samples x grid). BAMLSS may save [N, S]."""
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


def create_panel_for_model(npz_path, display_name, save_path):
    """Load npz, compute variance and entropy, build 2x2 panel, save."""
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

    mu_pred = np.mean(mu_samples, axis=0)
    ale_var = np.mean(sigma2_samples, axis=0)
    epi_var = np.var(mu_samples, axis=0)

    ent = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
    ale_entropy = ent["aleatoric"].squeeze()
    epi_entropy = ent["epistemic"].squeeze()

    mu_pred = np.asarray(mu_pred).squeeze()
    ale_var = np.asarray(ale_var).squeeze()
    epi_var = np.asarray(epi_var).squeeze()

    ood_mask = _build_ood_mask(x_grid, OOD_RANGES)
    id_mask = ~ood_mask

    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean.ravel()

    boundary_x = []
    if np.any(ood_mask):
        transitions = np.where(np.diff(ood_mask.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = list(x[transitions + 1])

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

    def shade_ood_regions(ax):
        """Draw shaded OOD regions behind other content."""
        for ood_start, ood_end in OOD_RANGES:
            ax.axvspan(ood_start, ood_end, alpha=0.35, color="lightgrey", zorder=0)

    x_train_flat = None
    y_train_flat = None
    if x_train is not None and y_train is not None:
        x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train.ravel()
        y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train.ravel()

    # ----- Left column: Variance std (AU, EU only) -----
    # [0,0] Variance - Aleatoric
    ax = axes[0, 0]
    shade_ood_regions(ax)
    if x_train_flat is not None and y_train_flat is not None:
        ax.scatter(
            x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none"
        )
    for bx in boundary_x:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax.plot(x, mu_pred, "b-", linewidth=2.5, label="Predictive mean", zorder=5)
    ax.fill_between(
        x,
        mu_pred - np.sqrt(ale_var),
        mu_pred + np.sqrt(ale_var),
        alpha=0.35,
        color="#06A77D",
        label="±σ(aleatoric)",
        zorder=1,
    )
    ax.plot(x[id_mask], y_clean_flat[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean_flat[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
        ax.scatter(
            x[ood_mask], y_clean_flat[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5
        )
    ax.set_ylabel("y", fontsize=12, fontweight="bold")
    ax.set_title("Variance - Aleatoric", fontweight="bold", fontsize=13, pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)

    # [1,0] Variance - Epistemic
    ax = axes[1, 0]
    shade_ood_regions(ax)
    if x_train_flat is not None and y_train_flat is not None:
        ax.scatter(
            x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none"
        )
    for bx in boundary_x:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax.plot(x, mu_pred, "b-", linewidth=2.5, label="Predictive mean", zorder=5)
    ax.fill_between(
        x,
        mu_pred - np.sqrt(epi_var),
        mu_pred + np.sqrt(epi_var),
        alpha=0.35,
        color="#F18F01",
        label="±σ(epistemic)",
        zorder=1,
    )
    ax.plot(x[id_mask], y_clean_flat[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean_flat[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
        ax.scatter(
            x[ood_mask], y_clean_flat[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5
        )
    ax.set_xlabel("x", fontsize=12, fontweight="bold")
    ax.set_ylabel("y", fontsize=12, fontweight="bold")
    ax.set_title("Variance - Epistemic", fontweight="bold", fontsize=13, pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)

    # ----- Right column: Entropy line plots (AU, EU only) -----
    # [0,1] Entropy - Aleatoric (line plot)
    ax = axes[0, 1]
    shade_ood_regions(ax)
    if x_train_flat is not None and y_train_flat is not None:
        ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color="blue", label="Training data", zorder=3)
    for bx in boundary_x:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax_twin = ax.twinx()
    ax.plot(x[id_mask], mu_pred[id_mask], "b-", linewidth=1.2, label="Predictive mean", alpha=0.5)
    ax.plot(x[ood_mask], mu_pred[ood_mask], "b-", linewidth=1.2, alpha=0.5)
    ax_twin.plot(x[id_mask], ale_entropy[id_mask], "g-", linewidth=2, label="Aleatoric entropy (nats)")
    ax_twin.plot(x[ood_mask], ale_entropy[ood_mask], "g-", linewidth=2, alpha=0.7)
    ax.plot(x[id_mask], y_clean_flat[id_mask], "r-", linewidth=1.5, alpha=0.8, label="True function")
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean_flat[ood_mask], "r-", linewidth=1.5, alpha=0.8)
        ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=8, color="red", alpha=0.3, marker="x", zorder=4)
    ax.set_ylabel("y / Predictive mean", fontsize=11)
    ax_twin.set_ylabel("Entropy (nats)", fontsize=11, color="green")
    ax_twin.tick_params(axis="y", labelcolor="green")
    ax.set_title("Entropy - Aleatoric", fontweight="bold", fontsize=13, pad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax_twin.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # [1,1] Entropy - Epistemic (line plot)
    ax = axes[1, 1]
    shade_ood_regions(ax)
    if x_train_flat is not None and y_train_flat is not None:
        ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color="blue", label="Training data", zorder=3)
    for bx in boundary_x:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    ax_twin = ax.twinx()
    ax.plot(x[id_mask], mu_pred[id_mask], "b-", linewidth=1.2, label="Predictive mean", alpha=0.5)
    ax.plot(x[ood_mask], mu_pred[ood_mask], "b-", linewidth=1.2, alpha=0.5)
    ax_twin.plot(x[id_mask], epi_entropy[id_mask], "r-", linewidth=2, label="Epistemic entropy (nats)")
    ax_twin.plot(x[ood_mask], epi_entropy[ood_mask], "r-", linewidth=2, alpha=0.7)
    ax.plot(x[id_mask], y_clean_flat[id_mask], "r-", linewidth=1.5, alpha=0.8, label="True function")
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean_flat[ood_mask], "r-", linewidth=1.5, alpha=0.8)
        ax.scatter(x[ood_mask], y_clean_flat[ood_mask], s=8, color="red", alpha=0.3, marker="x", zorder=4)
    ax.set_xlabel("x", fontsize=12, fontweight="bold")
    ax.set_ylabel("y / Predictive mean", fontsize=11)
    ax_twin.set_ylabel("Entropy (nats)", fontsize=11, color="red")
    ax_twin.tick_params(axis="y", labelcolor="red")
    ax.set_title("Entropy - Epistemic", fontweight="bold", fontsize=13, pad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax_twin.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{display_name} - Sin Function (heteroscedastic) - AU & EU only",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def main():
    outputs_base = project_root / "results" / "ood" / "outputs" / "ood"
    search_dir = outputs_base / "heteroscedastic" / "sin"
    save_dir = project_root / "results" / "ood" / "plots"
    if not search_dir.exists():
        print("Directory does not exist:", search_dir)
        print("Run the OOD experiment once (e.g. from Experiments/OOD.ipynb) to generate npz files.")
        return

    for pattern, display_name in MODELS:
        npz_files = sorted(search_dir.glob(f"{pattern}raw_outputs*.npz"))
        if not npz_files:
            print("No npz found for", display_name, "under", search_dir, "- skipping.")
            continue
        npz_path = npz_files[-1]
        print("Loading:", npz_path)
        model_key = display_name.replace(" ", "_")
        save_path = save_dir / f"panel_{model_key}_sin_heteroscedastic_au_eu_2x2.png"
        create_panel_for_model(npz_path, display_name, save_path)


if __name__ == "__main__":
    main()
