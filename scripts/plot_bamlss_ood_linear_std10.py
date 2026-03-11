"""
One-off script: BAMLSS OOD variance plot (linear and sinusoidal) with standard deviation scaled by 10.
Reuses saved model outputs from results/ood/outputs; no refit.
Produces the same 3-panel layout as plot_uncertainties_ood but bands = 10 * std.
Supports linear (hetero/homo) and sin (hetero/homo), e.g. sin homo.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Default OOD convention from ood_experiments
TRAIN_RANGE = (-5, 10)
OOD_RANGES = [(30, 40)]
STD_SCALE = 10
X_MAX = 12.5  # Cut off x-axis at this value (hide region beyond)


def _recompute_from_samples(mu_samples, sigma2_samples):
    """Recompute mu_pred, ale_var, epi_var, tot_var from [S, N] arrays (BAMLSS convention)."""
    mu_pred = np.mean(mu_samples, axis=0)
    ale_var = np.mean(sigma2_samples, axis=0)
    epi_var = np.var(mu_samples, axis=0)
    tot_var = ale_var + epi_var
    return mu_pred, ale_var, epi_var, tot_var


def _build_ood_mask(x_grid, train_range, ood_ranges):
    """Reconstruct ood_mask from x_grid using same convention as ood_experiments."""
    x_flat = x_grid.ravel()
    ood_mask = np.zeros(len(x_flat), dtype=bool)
    for ood_start, ood_end in ood_ranges:
        ood_mask |= (x_flat >= ood_start) & (x_flat <= ood_end)
    return ood_mask


def _plot_and_save_std_scaled(
    x_train, y_train, x_grid, y_clean,
    mu_pred, ale_var, epi_var, tot_var,
    ood_mask, noise_type, save_dir,
    func_display_name="Linear", title_suffix="(std x 10)",
    epi_only=False,
):
    """Build 3-panel OOD variance figure. If epi_only: only epistemic band = STD_SCALE*std; else all bands scaled."""
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid
    mu_pred = np.asarray(mu_pred).squeeze()
    ale_var = np.asarray(ale_var).squeeze()
    epi_var = np.asarray(epi_var).squeeze()
    tot_var = np.asarray(tot_var).squeeze()
    y_clean_flat = y_clean[:, 0] if y_clean.ndim > 1 else y_clean

    # Restrict to x <= X_MAX for both grid and training data (calculations and plot based only on this range)
    crop = x <= X_MAX
    x = x[crop]
    mu_pred = mu_pred[crop]
    ale_var = ale_var[crop]
    epi_var = epi_var[crop]
    tot_var = tot_var[crop]
    y_clean_flat = y_clean_flat[crop]
    ood_mask = ood_mask[crop]

    x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train.ravel()
    y_train_flat = y_train[:, 0] if y_train.ndim > 1 else y_train.ravel()
    train_crop = x_train_flat <= X_MAX
    x_train = x_train_flat[train_crop].reshape(-1, 1) if x_train.ndim > 1 else x_train_flat[train_crop]
    y_train = y_train_flat[train_crop].reshape(-1, 1) if y_train.ndim > 1 else y_train_flat[train_crop]

    id_mask = ~ood_mask
    ood_mask_bool = ood_mask

    # Band widths: epi_only => only epistemic scaled by STD_SCALE; else all scaled
    if epi_only:
        band_tot = np.sqrt(tot_var)
        band_ale = np.sqrt(ale_var)
        band_epi = STD_SCALE * np.sqrt(epi_var)
        label_tot, label_ale, label_epi = "±σ(total)", "±σ(aleatoric)", f"±{STD_SCALE}σ(epistemic)"
    else:
        band_tot = STD_SCALE * np.sqrt(tot_var)
        band_ale = STD_SCALE * np.sqrt(ale_var)
        band_epi = STD_SCALE * np.sqrt(epi_var)
        label_tot = f"±{STD_SCALE}σ(total)"
        label_ale = f"±{STD_SCALE}σ(aleatoric)"
        label_epi = f"±{STD_SCALE}σ(epistemic)"

    boundary_x = []
    if np.any(ood_mask_bool):
        transitions = np.where(np.diff(ood_mask_bool.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = x[transitions + 1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # Plot 1: Total
    axes[0].scatter(
        x_train[:, 0] if x_train.ndim > 1 else x_train,
        y_train[:, 0] if y_train.ndim > 1 else y_train,
        alpha=0.1, s=10, color="blue", label="Training data", zorder=3,
    )
    for bx in boundary_x:
        axes[0].axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    axes[0].plot(x[id_mask], mu_pred[id_mask], "b-", linewidth=1.2, label="Predictive mean")
    axes[0].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], "b-", linewidth=1.2)
    axes[0].fill_between(
        x[id_mask], mu_pred[id_mask] - band_tot[id_mask], mu_pred[id_mask] + band_tot[id_mask],
        alpha=0.3, color="blue", label=label_tot,
    )
    axes[0].fill_between(
        x[ood_mask_bool], mu_pred[ood_mask_bool] - band_tot[ood_mask_bool],
        mu_pred[ood_mask_bool] + band_tot[ood_mask_bool],
        alpha=0.3, color="lightblue",
    )
    axes[0].plot(x[id_mask], y_clean_flat[id_mask], "r-", linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[0].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], "r-", linewidth=1.5, alpha=0.8)
        axes[0].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=8, color="red", alpha=0.3, marker="x", zorder=4)
    axes[0].set_ylabel("y")
    axes[0].set_title(f"BAMLSS - {func_display_name} - OOD - Variance {title_suffix} ({noise_type.capitalize()}): Predictive Mean + Total Uncertainty")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Aleatoric
    axes[1].scatter(
        x_train[:, 0] if x_train.ndim > 1 else x_train,
        y_train[:, 0] if y_train.ndim > 1 else y_train,
        alpha=0.1, s=10, color="blue", label="Training data", zorder=3,
    )
    for bx in boundary_x:
        axes[1].axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    axes[1].plot(x[id_mask], mu_pred[id_mask], "b-", linewidth=1.2, label="Predictive mean")
    axes[1].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], "b-", linewidth=1.2)
    axes[1].fill_between(
        x[id_mask], mu_pred[id_mask] - band_ale[id_mask], mu_pred[id_mask] + band_ale[id_mask],
        alpha=0.3, color="green", label=label_ale,
    )
    axes[1].fill_between(
        x[ood_mask_bool], mu_pred[ood_mask_bool] - band_ale[ood_mask_bool],
        mu_pred[ood_mask_bool] + band_ale[ood_mask_bool],
        alpha=0.3, color="lightgreen",
    )
    axes[1].plot(x[id_mask], y_clean_flat[id_mask], "r-", linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[1].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], "r-", linewidth=1.5, alpha=0.8)
        axes[1].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color="red", alpha=0.4, marker="x", zorder=4)
    axes[1].set_ylabel("y")
    axes[1].set_title(f"BAMLSS - {func_display_name} - OOD - Variance {title_suffix} ({noise_type.capitalize()}): Predictive Mean + Aleatoric Uncertainty")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Epistemic
    axes[2].scatter(
        x_train[:, 0] if x_train.ndim > 1 else x_train,
        y_train[:, 0] if y_train.ndim > 1 else y_train,
        alpha=0.1, s=10, color="blue", label="Training data", zorder=3,
    )
    for bx in boundary_x:
        axes[2].axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    axes[2].plot(x[id_mask], mu_pred[id_mask], "b-", linewidth=1.2, label="Predictive mean")
    axes[2].plot(x[ood_mask_bool], mu_pred[ood_mask_bool], "b-", linewidth=1.2)
    axes[2].fill_between(
        x[id_mask], mu_pred[id_mask] - band_epi[id_mask], mu_pred[id_mask] + band_epi[id_mask],
        alpha=0.3, color="red", label=label_epi,
    )
    axes[2].fill_between(
        x[ood_mask_bool], mu_pred[ood_mask_bool] - band_epi[ood_mask_bool],
        mu_pred[ood_mask_bool] + band_epi[ood_mask_bool],
        alpha=0.3, color="coral",
    )
    axes[2].plot(x[id_mask], y_clean_flat[id_mask], "r-", linewidth=1.5, alpha=0.8)
    if np.any(ood_mask_bool):
        axes[2].plot(x[ood_mask_bool], y_clean_flat[ood_mask_bool], "r-", linewidth=1.5, alpha=0.8)
        axes[2].scatter(x[ood_mask_bool], y_clean_flat[ood_mask_bool], s=15, color="red", alpha=0.4, marker="x", zorder=4)
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("x")
    axes[2].set_title(f"BAMLSS - {func_display_name} - OOD - Variance {title_suffix} ({noise_type.capitalize()}): Predictive Mean + Epistemic Uncertainty")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(right=X_MAX)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    if epi_only:
        filename = f"BAMLSS_-_{func_display_name}_-_OOD_-_Variance_std_x_10_epi_only.png"
    else:
        filename = f"BAMLSS_-_{func_display_name}_-_OOD_-_Variance_std_x_10.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


# func_type folder name -> display name for titles and filename
FUNC_DISPLAY_NAMES = {"linear": "Linear", "sin": "Sinusoidal"}


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    outputs_base = project_root / "results" / "ood" / "outputs" / "ood"
    plots_base = project_root / "results" / "ood" / "plots"

    # Process both linear and sin (e.g. sin homo = homoscedastic/sin). One npz per (noise_type, func_folder).
    npz_globs = [
        ("linear", "**/linear/*BAMLSS*raw_outputs*.npz"),
        ("sin", "**/sin/*BAMLSS*raw_outputs*.npz"),
    ]
    seen = set()  # (noise_type, func_folder)
    npz_files_with_func = []
    for func_folder, pattern in npz_globs:
        for p in sorted(outputs_base.glob(pattern)):
            parts = p.relative_to(outputs_base).parts
            noise_type = parts[0] if len(parts) >= 2 else "heteroscedastic"
            key = (noise_type, func_folder)
            if key not in seen:
                seen.add(key)
                npz_files_with_func.append((p, func_folder))

    if not npz_files_with_func:
        print("No BAMLSS OOD raw_outputs .npz files found for linear or sin.")
        print("Expected under: results/ood/outputs/ood/{noise_type}/linear/ or .../sin/")
        print("Run the BAMLSS OOD experiment once (e.g. from Experiments/OOD.ipynb) to generate them.")
        return

    for npz_path, func_folder in npz_files_with_func:
        # Infer noise_type from path: .../ood/heteroscedastic/linear/... or .../ood/homoscedastic/sin/...
        parts = npz_path.relative_to(outputs_base).parts
        noise_type = parts[0] if len(parts) >= 2 else "heteroscedastic"
        func_display_name = FUNC_DISPLAY_NAMES.get(func_folder, func_folder.capitalize())

        data = np.load(npz_path, allow_pickle=True)
        mu_samples = data["mu_samples"]
        sigma2_samples = data["sigma2_samples"]
        x_grid = data["x_grid"]
        y_grid_clean = data["y_grid_clean"]
        x_train = data["x_train_subset"]
        y_train = data["y_train_subset"]

        # BAMLSS saves [S, N]; if we have [N, S], transpose to [S, N]
        n_grid = len(x_grid.ravel())
        if mu_samples.shape[0] == n_grid and mu_samples.shape[1] != n_grid:
            mu_samples = mu_samples.T
            sigma2_samples = sigma2_samples.T

        mu_pred, ale_var, epi_var, tot_var = _recompute_from_samples(mu_samples, sigma2_samples)
        ood_mask = _build_ood_mask(x_grid, TRAIN_RANGE, OOD_RANGES)

        save_dir = plots_base / "uncertainties_ood" / noise_type / func_folder
        _plot_and_save_std_scaled(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var,
            ood_mask, noise_type, save_dir,
            func_display_name=func_display_name,
            epi_only=False,
        )
        _plot_and_save_std_scaled(
            x_train, y_train, x_grid, y_grid_clean,
            mu_pred, ale_var, epi_var, tot_var,
            ood_mask, noise_type, save_dir,
            func_display_name=func_display_name,
            epi_only=True,
            title_suffix="(std x 10, epistemic only)",
        )


if __name__ == "__main__":
    main()
