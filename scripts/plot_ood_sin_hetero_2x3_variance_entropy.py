"""
2×3 panels for sin/heteroscedastic OOD: MC Dropout, BAMLSS, BNN.
- One 2×3 panel: Variance (row 0 = AU, row 1 = EU; columns = MC Dropout | BAMLSS | BNN).
- One 2×3 panel: Entropy (same layout).
Loads from results/ood/outputs/ood/heteroscedastic/sin/.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical

OOD_RANGES = [(10, 15)]

# Order: (glob pattern, display name)
MODELS_2X3 = [
    ("*MC_Dropout*", "MC Dropout"),
    ("*BAMLSS*", "BAMLSS"),
    ("*BNN*", "BNN"),
]


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


def load_model_data(npz_path):
    """Load one npz and return dict with x, y_clean_flat, mu_pred, ale_var, epi_var, ale_entropy, epi_entropy, ood_mask, id_mask, boundary_x, x_train_flat, y_train_flat."""
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
    ale_entropy = ent["aleatoric"].squeeze()
    epi_entropy = ent["epistemic"].squeeze()

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


def main():
    search_dir = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
    save_dir = project_root / "results" / "ood" / "plots"
    if not search_dir.exists():
        print("Directory does not exist:", search_dir)
        return

    # Load data for each model
    models_data = []
    for pattern, display_name in MODELS_2X3:
        npz_files = sorted(search_dir.glob(f"{pattern}raw_outputs*.npz"))
        if not npz_files:
            print("No npz found for", display_name, "- skipping 2x3 panels.")
            return
        npz_path = npz_files[-1]
        print("Loading:", npz_path)
        models_data.append((display_name, load_model_data(npz_path)))

    def shade_ood(ax):
        for ood_start, ood_end in OOD_RANGES:
            ax.axvspan(ood_start, ood_end, alpha=0.35, color="lightgrey", zorder=0)

    # ----- Variance 2×3: row 0 = AU, row 1 = EU; cols = MC Dropout, BAMLSS, BNN -----
    fig_var, axes_var = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for col, (display_name, d) in enumerate(models_data):
        x, y = d["x"], d["y_clean_flat"]
        ood_mask, id_mask = d["ood_mask"], d["id_mask"]
        boundary_x = d["boundary_x"]
        x_train_flat, y_train_flat = d["x_train_flat"], d["y_train_flat"]

        # Row 0: Aleatoric variance
        ax = axes_var[0, col]
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none")
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax.plot(x, d["mu_pred"], "b-", linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(
            x, d["mu_pred"] - np.sqrt(d["ale_var"]), d["mu_pred"] + np.sqrt(d["ale_var"]),
            alpha=0.35, color="#06A77D", label="±σ(aleatoric)", zorder=1,
        )
        ax.plot(x[id_mask], y[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
            ax.scatter(x[ood_mask], y[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
        ax.set_ylabel("y", fontsize=11, fontweight="bold")
        ax.set_title(f"{display_name}\nVariance - Aleatoric", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

        # Row 1: Epistemic variance
        ax = axes_var[1, col]
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none")
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax.plot(x, d["mu_pred"], "b-", linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(
            x, d["mu_pred"] - np.sqrt(d["epi_var"]), d["mu_pred"] + np.sqrt(d["epi_var"]),
            alpha=0.35, color="#F18F01", label="±σ(epistemic)", zorder=1,
        )
        ax.plot(x[id_mask], y[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
            ax.scatter(x[ood_mask], y[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
        ax.set_xlabel("x", fontsize=11, fontweight="bold")
        ax.set_ylabel("y", fontsize=11, fontweight="bold")
        ax.set_title(f"{display_name}\nVariance - Epistemic", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

    fig_var.suptitle(
        "Variance (AU & EU) - Sin heteroscedastic - MC Dropout, BAMLSS, BNN",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir.mkdir(parents=True, exist_ok=True)
    path_var = save_dir / "panel_variance_MC_Dropout_BAMLSS_BNN_sin_hetero_2x3.png"
    fig_var.savefig(path_var, dpi=150, bbox_inches="tight")
    plt.close(fig_var)
    print("Saved:", path_var)

    # ----- Variance 2×3 with BAMLSS normalized (MC Dropout & BNN: ±σ; BAMLSS: normalized variance bands as in consolidated plots) -----
    def _normalize_01(v):
        v = np.asarray(v)
        lo, hi = v.min(), v.max()
        if hi <= lo:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    SCALE_FACTOR = 0.3  # same as utils/plotting normalized variance plots

    fig_var_norm, axes_var_norm = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for col, (display_name, d) in enumerate(models_data):
        x, y = d["x"], d["y_clean_flat"]
        ood_mask, id_mask = d["ood_mask"], d["id_mask"]
        boundary_x = d["boundary_x"]
        x_train_flat, y_train_flat = d["x_train_flat"], d["y_train_flat"]

        if display_name == "BAMLSS":
            # Same recipe as plot_uncertainties_ood_normalized: normalize std to [0,1], scale by y_range * scale_factor
            std_ale = np.sqrt(d["ale_var"])
            std_epi = np.sqrt(d["epi_var"])
            std_ale_norm = _normalize_01(std_ale)
            std_epi_norm = _normalize_01(std_epi)
            y_range = float(np.ptp(y))
            ale_var_plot = std_ale_norm * y_range * SCALE_FACTOR
            epi_var_plot = std_epi_norm * y_range * SCALE_FACTOR
            ale_label, epi_label = "±norm(aleatoric)", "±norm(epistemic)"
        else:
            ale_var_plot = np.sqrt(d["ale_var"])
            epi_var_plot = np.sqrt(d["epi_var"])
            ale_label, epi_label = "±σ(aleatoric)", "±σ(epistemic)"

        # Row 0: Aleatoric variance
        ax = axes_var_norm[0, col]
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none")
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax.plot(x, d["mu_pred"], "b-", linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(
            x, d["mu_pred"] - ale_var_plot, d["mu_pred"] + ale_var_plot,
            alpha=0.35, color="#06A77D", label=ale_label, zorder=1,
        )
        ax.plot(x[id_mask], y[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
            ax.scatter(x[ood_mask], y[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
        ax.set_ylabel("y", fontsize=11, fontweight="bold")
        ax.set_title(f"{display_name}\nVariance - Aleatoric", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

        # Row 1: Epistemic variance
        ax = axes_var_norm[1, col]
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none")
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax.plot(x, d["mu_pred"], "b-", linewidth=2.5, label="Predictive mean", zorder=5)
        ax.fill_between(
            x, d["mu_pred"] - epi_var_plot, d["mu_pred"] + epi_var_plot,
            alpha=0.35, color="#F18F01", label=epi_label, zorder=1,
        )
        ax.plot(x[id_mask], y[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
            ax.scatter(x[ood_mask], y[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
        ax.set_xlabel("x", fontsize=11, fontweight="bold")
        ax.set_ylabel("y", fontsize=11, fontweight="bold")
        ax.set_title(f"{display_name}\nVariance - Epistemic", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

    fig_var_norm.suptitle(
        "Variance (AU & EU) - Sin heteroscedastic - MC Dropout, BAMLSS (norm.), BNN",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_var_norm = save_dir / "panel_variance_MC_Dropout_BAMLSS_BNN_sin_hetero_2x3_BAMLSS_normalized.png"
    fig_var_norm.savefig(path_var_norm, dpi=150, bbox_inches="tight")
    plt.close(fig_var_norm)
    print("Saved:", path_var_norm)

    # ----- Entropy 2×3: same layout -----
    fig_ent, axes_ent = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for col, (display_name, d) in enumerate(models_data):
        x, y = d["x"], d["y_clean_flat"]
        ood_mask, id_mask = d["ood_mask"], d["id_mask"]
        boundary_x = d["boundary_x"]
        x_train_flat, y_train_flat = d["x_train_flat"], d["y_train_flat"]

        # Row 0: Aleatoric entropy
        ax = axes_ent[0, col]
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color="blue", label="Training data", zorder=3)
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax_twin = ax.twinx()
        ax.plot(x[id_mask], d["mu_pred"][id_mask], "b-", linewidth=1.2, label="Predictive mean", alpha=0.5)
        ax.plot(x[ood_mask], d["mu_pred"][ood_mask], "b-", linewidth=1.2, alpha=0.5)
        ax_twin.plot(x[id_mask], d["ale_entropy"][id_mask], "g-", linewidth=2, label="Aleatoric entropy (nats)")
        ax_twin.plot(x[ood_mask], d["ale_entropy"][ood_mask], "g-", linewidth=2, alpha=0.7)
        ax.plot(x[id_mask], y[id_mask], "r-", linewidth=1.5, alpha=0.8, label="True function")
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y[ood_mask], "r-", linewidth=1.5, alpha=0.8)
            ax.scatter(x[ood_mask], y[ood_mask], s=8, color="red", alpha=0.3, marker="x", zorder=4)
        ax.set_ylabel("y / Predictive mean", fontsize=10)
        ax_twin.set_ylabel("Entropy (nats)", fontsize=10, color="green")
        ax_twin.tick_params(axis="y", labelcolor="green")
        ax.set_title(f"{display_name}\nEntropy - Aleatoric", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8)
        ax_twin.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1: Epistemic entropy
        ax = axes_ent[1, col]
        shade_ood(ax)
        if x_train_flat is not None and y_train_flat is not None:
            ax.scatter(x_train_flat, y_train_flat, alpha=0.1, s=10, color="blue", label="Training data", zorder=3)
        for bx in boundary_x:
            ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax_twin = ax.twinx()
        ax.plot(x[id_mask], d["mu_pred"][id_mask], "b-", linewidth=1.2, label="Predictive mean", alpha=0.5)
        ax.plot(x[ood_mask], d["mu_pred"][ood_mask], "b-", linewidth=1.2, alpha=0.5)
        ax_twin.plot(x[id_mask], d["epi_entropy"][id_mask], "r-", linewidth=2, label="Epistemic entropy (nats)")
        ax_twin.plot(x[ood_mask], d["epi_entropy"][ood_mask], "r-", linewidth=2, alpha=0.7)
        ax.plot(x[id_mask], y[id_mask], "r-", linewidth=1.5, alpha=0.8, label="True function")
        if np.any(ood_mask):
            ax.plot(x[ood_mask], y[ood_mask], "r-", linewidth=1.5, alpha=0.8)
            ax.scatter(x[ood_mask], y[ood_mask], s=8, color="red", alpha=0.3, marker="x", zorder=4)
        ax.set_xlabel("x", fontsize=11, fontweight="bold")
        ax.set_ylabel("y / Predictive mean", fontsize=10)
        ax_twin.set_ylabel("Entropy (nats)", fontsize=10, color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")
        ax.set_title(f"{display_name}\nEntropy - Epistemic", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8)
        ax_twin.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig_ent.suptitle(
        "Entropy (AU & EU) - Sin heteroscedastic - MC Dropout, BAMLSS, BNN",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_ent = save_dir / "panel_entropy_MC_Dropout_BAMLSS_BNN_sin_hetero_2x3.png"
    fig_ent.savefig(path_ent, dpi=150, bbox_inches="tight")
    plt.close(fig_ent)
    print("Saved:", path_ent)


if __name__ == "__main__":
    main()
