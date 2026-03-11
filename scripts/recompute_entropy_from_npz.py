"""
Recompute entropy-based uncertainties from a saved npz and check (samples, grid) convention.
Loads mu_samples, sigma2_samples, x_grid; ensures first axis = samples, second = grid;
calls entropy_uncertainty_analytical; reports shapes and whether epistemic entropy is ever negative.
Optionally compares with wrong convention to show it can yield negative EU.
Usage: python scripts/recompute_entropy_from_npz.py [path/to/file.npz]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical

DEFAULT_NPZ_DIR = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
DEFAULT_GLOB = "*BAMLSS*raw_outputs*.npz"


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    """Ensure mu and sigma2 have shape (S, N) = (samples, grid)."""
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


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
    n_grid = np.asarray(x_grid).ravel().shape[0]

    print("\n--- Shapes before _ensure_samples_first ---")
    print("  mu_samples:", mu_samples.shape)
    print("  sigma2_samples:", sigma2_samples.shape)
    print("  n_grid (from x_grid):", n_grid)

    mu_before = mu_samples.copy()
    sig_before = sigma2_samples.copy()
    mu_samples, sigma2_samples = _ensure_samples_first(mu_samples, sigma2_samples, x_grid)

    print("\n--- Shapes after _ensure_samples_first (expected: samples x grid) ---")
    print("  mu_samples:", mu_samples.shape)
    print("  sigma2_samples:", sigma2_samples.shape)
    nsamples = mu_samples.shape[0]
    print("  nsamples (first axis):", nsamples)
    print("  n_grid (second axis):", mu_samples.shape[1])

    # Recompute entropy with correct convention
    ent = entropy_uncertainty_analytical(mu_samples, sigma2_samples)
    ale = np.asarray(ent["aleatoric"]).squeeze()
    epi = np.asarray(ent["epistemic"]).squeeze()
    tot = np.asarray(ent["total"]).squeeze()

    print("\n--- Entropy (correct convention: first axis = samples) ---")
    print("  aleatoric  min/max:", ale.min(), ale.max())
    print("  epistemic  min/max:", epi.min(), epi.max())
    print("  total      min/max:", tot.min(), tot.max())
    any_neg = np.any(epi < 0)
    print("  Any epistemic < 0?", any_neg)
    if any_neg:
        n_neg = np.sum(epi < 0)
        print("  Count epistemic < 0:", n_neg, "out of", epi.size)

    # Optional: wrong convention (transpose to N x S) and show epistemic can go negative
    if mu_before.shape[0] != mu_before.shape[1]:
        wrong_mu = mu_samples.T
        wrong_sig = sigma2_samples.T
        ent_wrong = entropy_uncertainty_analytical(wrong_mu, wrong_sig)
        epi_wrong = np.asarray(ent_wrong["epistemic"]).squeeze()
        print("\n--- Entropy with WRONG convention (first axis = grid) ---")
        print("  epistemic min/max:", epi_wrong.min(), epi_wrong.max())
        print("  Any epistemic < 0?", np.any(epi_wrong < 0))

    # Plot entropy with epistemic clamped to zero
    epi_clamped = np.maximum(epi, 0.0)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(10, 15, alpha=0.35, color="lightgrey", zorder=0, label="OOD")
    ax.plot(x, ale, color="green", linewidth=1.5, label="Aleatoric (nats)")
    ax.plot(x, epi_clamped, color="#C41E3A", linewidth=1.5, label="Epistemic clamped to 0 (nats)")
    ax.plot(x, tot, color="blue", linewidth=1.5, label="Total (nats)")
    ax.set_xlabel("x")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Entropy from npz (epistemic clamped to 0)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_dir = project_root / "results" / "ood" / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"entropy_recomputed_clamped_{npz_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
