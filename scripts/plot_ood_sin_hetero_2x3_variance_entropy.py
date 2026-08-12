"""
Panels for sin/heteroscedastic OOD, from among MC Dropout, BAMLSS, BNN. Models
missing an npz file are simply omitted (grid gets fewer columns), only skipping
entirely if none of the three are available.
- One panel: raw variance (row 0 = Total, row 1 = AU, row 2 = EU; columns = whichever
  of MC Dropout / BAMLSS / BNN have data). Y-axis fixed to (-20, 31).
- One panel: same layout, but BAMLSS's bands are normalized (std -> [0,1] -> scaled
  by y_range * 0.3) since its raw variance is visually unusable at this script's
  scale; its Total row is the sum of the separately-normalized AU/EU stds, scaled
  once, matching utils/plotting.py's plot_uncertainties_ood_normalized convention.
- One panel: entropy (row 0 = AU, row 1 = EU, no Total row -- entropy decomposition
  doesn't have as settled a total=ale+epi identity). Y-axis fixed on the primary
  (response) scale only; the entropy (nats) twin axis stays autoscaled.
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
    tot_var = ale_var + epi_var
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
        "tot_var": tot_var,
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

    # Load data for each model that has it; models without data are simply omitted
    # (grid gets fewer columns) rather than aborting the whole run.
    models_data = []
    for pattern, display_name in MODELS_2X3:
        npz_files = sorted(search_dir.glob(f"{pattern}raw_outputs*.npz"))
        if not npz_files:
            print("No npz found for", display_name, "- omitting column.")
            continue
        npz_path = npz_files[-1]
        print("Loading:", npz_path)
        models_data.append((display_name, load_model_data(npz_path)))

    if not models_data:
        print("No models available - skipping all panels.")
        return
    n_cols = len(models_data)
    model_names_title = ", ".join(name for name, _ in models_data)

    def shade_ood(ax):
        for ood_start, ood_end in OOD_RANGES:
            ax.axvspan(ood_start, ood_end, alpha=0.35, color="lightgrey", zorder=0)

    VAR_YLIM = (-20, 31)  # hardcoded: this script is sin/heteroscedastic only

    # ----- Variance 3×N: row 0 = Total, row 1 = AU, row 2 = EU; cols = available models -----
    fig_var, axes_var = plt.subplots(3, n_cols, figsize=(6 * n_cols, 15), sharex=True, squeeze=False)
    for col, (display_name, d) in enumerate(models_data):
        x, y = d["x"], d["y_clean_flat"]
        ood_mask, id_mask = d["ood_mask"], d["id_mask"]
        boundary_x = d["boundary_x"]
        x_train_flat, y_train_flat = d["x_train_flat"], d["y_train_flat"]

        for row, (var_key, color, band_label, row_label) in enumerate([
            ("tot_var", "#2E86AB", "±σ(total)", "Total"),
            ("ale_var", "#06A77D", "±σ(aleatoric)", "Aleatoric"),
            ("epi_var", "#F18F01", "±σ(epistemic)", "Epistemic"),
        ]):
            ax = axes_var[row, col]
            shade_ood(ax)
            if x_train_flat is not None and y_train_flat is not None:
                ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none")
            for bx in boundary_x:
                ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
            ax.plot(x, d["mu_pred"], "b-", linewidth=2.5, label="Predictive mean", zorder=5)
            ax.fill_between(
                x, d["mu_pred"] - np.sqrt(d[var_key]), d["mu_pred"] + np.sqrt(d[var_key]),
                alpha=0.35, color=color, label=band_label, zorder=1,
            )
            ax.plot(x[id_mask], y[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
            if row == 2:
                ax.set_xlabel("x", fontsize=11, fontweight="bold")
            ax.set_ylabel("y", fontsize=11, fontweight="bold")
            ax.set_ylim(*VAR_YLIM)
            ax.set_title(f"{display_name}\nVariance - {row_label}", fontweight="bold", fontsize=12, pad=8)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

    fig_var.suptitle(
        f"Variance (Total, AU & EU) - Sin heteroscedastic - {model_names_title}",
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

    fig_var_norm, axes_var_norm = plt.subplots(3, n_cols, figsize=(6 * n_cols, 15), sharex=True, squeeze=False)
    for col, (display_name, d) in enumerate(models_data):
        x, y = d["x"], d["y_clean_flat"]
        ood_mask, id_mask = d["ood_mask"], d["id_mask"]
        boundary_x = d["boundary_x"]
        x_train_flat, y_train_flat = d["x_train_flat"], d["y_train_flat"]

        if display_name == "BAMLSS":
            # Same recipe as plot_uncertainties_ood_normalized: normalize std to [0,1], scale by y_range * scale_factor.
            # Total is the SUM of the separately-normalized ale/epi stds (not a renormalization
            # of ale_var+epi_var as a whole), matching utils/plotting.py's convention.
            std_ale = np.sqrt(d["ale_var"])
            std_epi = np.sqrt(d["epi_var"])
            std_ale_norm = _normalize_01(std_ale)
            std_epi_norm = _normalize_01(std_epi)
            std_tot_norm = std_ale_norm + std_epi_norm
            y_range = float(np.ptp(y))
            tot_var_plot = std_tot_norm * y_range * SCALE_FACTOR
            ale_var_plot = std_ale_norm * y_range * SCALE_FACTOR
            epi_var_plot = std_epi_norm * y_range * SCALE_FACTOR
            tot_label, ale_label, epi_label = "±norm(total)", "±norm(aleatoric)", "±norm(epistemic)"
        else:
            tot_var_plot = np.sqrt(d["tot_var"])
            ale_var_plot = np.sqrt(d["ale_var"])
            epi_var_plot = np.sqrt(d["epi_var"])
            tot_label, ale_label, epi_label = "±σ(total)", "±σ(aleatoric)", "±σ(epistemic)"

        for row, (var_plot, color, band_label, row_label) in enumerate([
            (tot_var_plot, "#2E86AB", tot_label, "Total"),
            (ale_var_plot, "#06A77D", ale_label, "Aleatoric"),
            (epi_var_plot, "#F18F01", epi_label, "Epistemic"),
        ]):
            ax = axes_var_norm[row, col]
            shade_ood(ax)
            if x_train_flat is not None and y_train_flat is not None:
                ax.scatter(x_train_flat, y_train_flat, alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none")
            for bx in boundary_x:
                ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
            ax.plot(x, d["mu_pred"], "b-", linewidth=2.5, label="Predictive mean", zorder=5)
            ax.fill_between(
                x, d["mu_pred"] - var_plot, d["mu_pred"] + var_plot,
                alpha=0.35, color=color, label=band_label, zorder=1,
            )
            ax.plot(x[id_mask], y[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
            if np.any(ood_mask):
                ax.plot(x[ood_mask], y[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
                ax.scatter(x[ood_mask], y[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
            if row == 2:
                ax.set_xlabel("x", fontsize=11, fontweight="bold")
            ax.set_ylabel("y", fontsize=11, fontweight="bold")
            ax.set_ylim(*VAR_YLIM)
            ax.set_title(f"{display_name}\nVariance - {row_label}", fontweight="bold", fontsize=12, pad=8)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

    fig_var_norm.suptitle(
        f"Variance (Total, AU & EU) - Sin heteroscedastic - {model_names_title} (BAMLSS norm.)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_var_norm = save_dir / "panel_variance_MC_Dropout_BAMLSS_BNN_sin_hetero_2x3_BAMLSS_normalized.png"
    fig_var_norm.savefig(path_var_norm, dpi=150, bbox_inches="tight")
    plt.close(fig_var_norm)
    print("Saved:", path_var_norm)

    # ----- Entropy 2×3: same layout -----
    fig_ent, axes_ent = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), sharex=True, squeeze=False)
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
        ax.set_ylim(*VAR_YLIM)
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
        ax.set_ylim(*VAR_YLIM)
        ax_twin.set_ylabel("Entropy (nats)", fontsize=10, color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")
        ax.set_title(f"{display_name}\nEntropy - Epistemic", fontweight="bold", fontsize=12, pad=8)
        ax.legend(loc="upper left", fontsize=8)
        ax_twin.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig_ent.suptitle(
        f"Entropy (AU & EU) - Sin heteroscedastic - {model_names_title}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_ent = save_dir / "panel_entropy_MC_Dropout_BAMLSS_BNN_sin_hetero_2x3.png"
    fig_ent.savefig(path_ent, dpi=150, bbox_inches="tight")
    plt.close(fig_ent)
    print("Saved:", path_ent)


if __name__ == "__main__":
    main()
