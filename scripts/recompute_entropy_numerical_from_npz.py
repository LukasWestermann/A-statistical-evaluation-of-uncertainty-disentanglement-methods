"""
Recompute entropy-based uncertainties from a saved npz using the numerical (Monte Carlo mixture) path.
Loads mu_samples, sigma2_samples, x_grid; ensures first axis = samples, second = grid;
calls entropy_uncertainty_numerical; reports shapes and whether epistemic entropy is ever negative.
Optionally compares with wrong convention using analytical entropy only (fast; axis bug is the same).

Usage:
  python scripts/recompute_entropy_numerical_from_npz.py [path/to/file.npz]
  python scripts/recompute_entropy_numerical_from_npz.py --model mc_dropout
  python scripts/recompute_entropy_numerical_from_npz.py --model all --n-samples 5000 --seed 42
  python scripts/recompute_entropy_numerical_from_npz.py path/to/file.npz --grid-chunk-size 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical, entropy_uncertainty_numerical

DEFAULT_NPZ_DIR = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"

MODEL_GLOBS: dict[str, str] = {
    "bamlss": "*BAMLSS*raw_outputs*.npz",
    "mc_dropout": "*MC_Dropout*raw_outputs*.npz",
    "deep_ensemble": "*Deep_Ensemble*raw_outputs*.npz",
    "bnn": "*BNN*raw_outputs*.npz",
}

MODEL_ORDER: Sequence[str] = ("bamlss", "mc_dropout", "deep_ensemble", "bnn")


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    """Ensure mu and sigma2 have shape (S, N) = (samples, grid)."""
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


def _latest_npz(search_dir: Path, pattern: str) -> Path | None:
    files = sorted(search_dir.glob(pattern))
    return files[-1] if files else None


def resolve_npz_paths(search_dir: Path, model: str) -> List[Path]:
    if model == "all":
        out: List[Path] = []
        for key in MODEL_ORDER:
            p = _latest_npz(search_dir, MODEL_GLOBS[key])
            if p is None:
                print(f"[warn] No file for model preset {key!r} matching {MODEL_GLOBS[key]!r} in {search_dir}")
            else:
                out.append(p)
        return out
    if model not in MODEL_GLOBS:
        raise ValueError(f"Unknown model {model!r}")
    p = _latest_npz(search_dir, MODEL_GLOBS[model])
    return [p] if p is not None else []


def process_one_npz(
    npz_path: Path,
    *,
    n_samples: int,
    seed: int,
    grid_chunk_size: int | None,
    save_dir: Path,
) -> None:
    print("Loading:", npz_path)
    print("  n_samples:", n_samples, " seed:", seed, " grid_chunk_size:", grid_chunk_size)
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
    mu_samples, sigma2_samples = _ensure_samples_first(mu_samples, sigma2_samples, x_grid)

    print("\n--- Shapes after _ensure_samples_first (expected: samples x grid) ---")
    print("  mu_samples:", mu_samples.shape)
    print("  sigma2_samples:", sigma2_samples.shape)
    nsamples = mu_samples.shape[0]
    print("  nsamples (first axis):", nsamples)
    print("  n_grid (second axis):", mu_samples.shape[1])

    ent = entropy_uncertainty_numerical(
        mu_samples,
        sigma2_samples,
        n_samples=n_samples,
        seed=seed,
        grid_chunk_size=grid_chunk_size,
    )
    ale = np.asarray(ent["aleatoric"]).squeeze()
    epi = np.asarray(ent["epistemic"]).squeeze()
    tot = np.asarray(ent["total"]).squeeze()

    print("\n--- Entropy numerical / MC mixture (correct convention: first axis = samples) ---")
    print("  aleatoric  min/max:", ale.min(), ale.max())
    print("  epistemic  min/max:", epi.min(), epi.max())
    print("  total      min/max:", tot.min(), tot.max())
    any_neg = np.any(epi < 0)
    print("  Any epistemic < 0?", any_neg)
    if any_neg:
        n_neg = int(np.sum(epi < 0))
        print("  Count epistemic < 0:", n_neg, "out of", epi.size)

    if mu_before.shape[0] != mu_before.shape[1]:
        wrong_mu = mu_samples.T
        wrong_sig = sigma2_samples.T
        ent_wrong = entropy_uncertainty_analytical(wrong_mu, wrong_sig)
        epi_wrong = np.asarray(ent_wrong["epistemic"]).squeeze()
        print("\n--- Entropy with WRONG convention (first axis = grid); analytical reference ---")
        print("  epistemic min/max:", epi_wrong.min(), epi_wrong.max())
        print("  Any epistemic < 0?", np.any(epi_wrong < 0))

    epi_clamped = np.maximum(epi, 0.0)
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(10, 15, alpha=0.35, color="lightgrey", zorder=0, label="OOD")
    ax.plot(x, ale, color="green", linewidth=1.5, label="Aleatoric (nats)")
    ax.plot(x, epi_clamped, color="#C41E3A", linewidth=1.5, label="Epistemic clamped to 0 (nats)")
    ax.plot(x, tot, color="blue", linewidth=1.5, label="Total (nats)")
    ax.set_xlabel("x")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Entropy")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"entropy_recomputed_numerical_clamped_{npz_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute AU/EU/total entropy from npz using Monte Carlo mixture entropy (numerical)."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default=None,
        type=Path,
        help="Path to .npz (if set, --model is ignored for input)",
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_ORDER, "all"],
        default="bamlss",
        help="Which model glob under --search-dir when npz_path is omitted (default: bamlss). "
        "Use 'all' for latest BAMLSS, MC Dropout, Deep Ensemble, and BNN files.",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help=f"Directory to glob (default: {DEFAULT_NPZ_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: results/ood/plots)",
    )
    parser.add_argument("--n-samples", type=int, default=5000, help="MC samples for mixture entropy (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--grid-chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Grid columns per block for log p(y) (default: auto)",
    )
    args = parser.parse_args()

    save_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "ood" / "plots"

    if args.npz_path is not None:
        npz_path = args.npz_path
        if not npz_path.is_file():
            print("File not found:", npz_path)
            sys.exit(1)
        process_one_npz(
            npz_path,
            n_samples=args.n_samples,
            seed=args.seed,
            grid_chunk_size=args.grid_chunk_size,
            save_dir=save_dir,
        )
        print("\nDone.")
        return

    search_dir = Path(args.search_dir) if args.search_dir else DEFAULT_NPZ_DIR
    if not search_dir.is_dir():
        print("Directory not found:", search_dir)
        sys.exit(1)

    paths = resolve_npz_paths(search_dir, args.model)
    if not paths:
        print("No npz matched.", f"model={args.model!r}, dir={search_dir}")
        sys.exit(1)

    for p in paths:
        process_one_npz(
            p,
            n_samples=args.n_samples,
            seed=args.seed,
            grid_chunk_size=args.grid_chunk_size,
            save_dir=save_dir,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
