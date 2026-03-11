"""
Recompute AU and EU from the full formulas (AU, EU with Var(mu)/sigma-bar^2, TU = AU+EU)
using saved npz. Loads one npz, ensures (samples, grid), plots and saves.
Usage: python scripts/recompute_entropy_formula_from_npz.py [path/to/file.npz]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical_with_epistemic_var

DEFAULT_NPZ_DIR = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
DEFAULT_GLOB = "*BAMLSS*raw_outputs*.npz"
OOD_RANGES = [(10, 15)]


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    """Ensure mu and sigma2 have shape (S, N) = (samples, grid)."""
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


def plot_entropy_lines(x, aleatoric, epistemic, total, save_path=None, title=None, ood_ranges=None):
    """
    Plot entropy line curves as three separate figures: aleatoric, epistemic, total vs x.
    Optionally shade OOD regions. If save_path is set, saves three files with
    _aleatoric, _epistemic, _total before the extension.
    """
    if ood_ranges is None:
        ood_ranges = OOD_RANGES
    base_title = title or "Entropy from npz (formula: AU, EU with Var(mu)/sigma-bar^2)"
    curves = [
        (aleatoric, "green", "Aleatoric (nats)", "Aleatoric"),
        (epistemic, "#C41E3A", "Epistemic (nats)", "Epistemic"),
        (total, "blue", "Total (nats)", "Total"),
    ]
    suffixes = ["aleatoric", "epistemic", "total"]
    saved = []
    for (y, color, ylabel, sub_title), suffix in zip(curves, suffixes):
        fig, ax = plt.subplots(figsize=(10, 5))
        for ood_start, ood_end in ood_ranges:
            ax.axvspan(ood_start, ood_end, alpha=0.35, color="lightgrey", zorder=0, label="OOD")
        ax.plot(x, y, color=color, linewidth=1.5, label=ylabel)
        ax.set_xlabel("x")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title(f"{base_title} — {sub_title}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            stem = save_path.stem
            ext = save_path.suffix
            out = save_path.parent / f"{stem}_{suffix}{ext}"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            saved.append(out)
        plt.close(fig)
    return saved


def main():
    if len(sys.argv) >= 2:
        npz_path = Path(sys.argv[1])
        if not npz_path.is_file():
            print("File not found:", npz_path)
            sys.exit(1)
    else:
        search_dir = DEFAULT_NPZ_DIR
        if not search_dir.exists():
            print("Default directory not found:", search_dir)
            sys.exit(1)
        npz_files = sorted(search_dir.glob(DEFAULT_GLOB))
        if not npz_files:
            print("No npz found matching", DEFAULT_GLOB, "in", search_dir)
            sys.exit(1)
        npz_path = npz_files[-1]

    print("Loading:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    mu_samples = np.asarray(data["mu_samples"])
    sigma2_samples = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    mu_samples, sigma2_samples = _ensure_samples_first(mu_samples, sigma2_samples, x_grid)

    ent = entropy_uncertainty_analytical_with_epistemic_var(mu_samples, sigma2_samples)
    ale = np.asarray(ent["aleatoric"]).squeeze()
    epi = np.asarray(ent["epistemic"]).squeeze()
    tot = np.asarray(ent["total"]).squeeze()

    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    save_dir = project_root / "results" / "ood" / "plots"
    out_path = save_dir / f"entropy_recomputed_formula_{npz_path.stem}.png"
    saved = plot_entropy_lines(x, ale, epi, tot, save_path=out_path)
    for p in saved:
        print("Saved plot:", p)
    print("Done.")


if __name__ == "__main__":
    main()
