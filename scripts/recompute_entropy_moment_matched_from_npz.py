"""
Recompute AU and EU from moment-matched TU: TU = H[N(0, E[sigma^2] + Var(mu))],
AU = mean member Gaussian entropies, EU = TU - AU.

Loads saved raw_outputs .npz; ensures (samples, grid) layout; plots three panels.

Usage:
    python scripts/recompute_entropy_moment_matched_from_npz.py [path/to/file.npz]
    python scripts/recompute_entropy_moment_matched_from_npz.py --model mc_dropout
    python scripts/recompute_entropy_moment_matched_from_npz.py --model all
    python scripts/recompute_entropy_moment_matched_from_npz.py --model bnn --search-dir path/to/ood/outputs/ood/homoscedastic/linear

Outputs under results/ood/plots/ with prefix entropy_recomputed_moment_matched_ (compare with
entropy_recomputed_formula_* from recompute_entropy_formula_from_npz.py).
For OVB files (ovb_outputs*.npz), omitted entropy is recomputed from mu_samples/sigma2_samples.
If full-model keys are present, a second set of *_full_* plots is saved from
mu_samples_full/sigma2_samples_full on X_full.
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

from utils.entropy_uncertainty import entropy_uncertainty_analytical_moment_matched

DEFAULT_NPZ_DIR = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
OOD_RANGES = [(10, 15)]

# Same stem tags as utils.knn_entropy_regression.MODEL_RESOLVERS (one glob per model).
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


def plot_entropy_lines(x, aleatoric, epistemic, total, save_path=None, title=None, ood_ranges=None):
    """Save three figures: aleatoric, epistemic, total vs x (optional OOD shading)."""
    if ood_ranges is None:
        ood_ranges = OOD_RANGES
    base_title = title or "Entropy"
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


def _latest_npz(search_dir: Path, pattern: str) -> Path | None:
    files = sorted(search_dir.glob(pattern))
    return files[-1] if files else None


def resolve_npz_paths(search_dir: Path, model: str) -> List[Path]:
    """Resolve one or more npz paths from search_dir and model preset."""
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


def process_one_npz(npz_path: Path, save_dir: Path) -> None:
    print("Loading:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    mu_samples = np.asarray(data["mu_samples"])
    sigma2_samples = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    mu_samples, sigma2_samples = _ensure_samples_first(mu_samples, sigma2_samples, x_grid)

    ent = entropy_uncertainty_analytical_moment_matched(mu_samples, sigma2_samples)
    ale = np.asarray(ent["aleatoric"]).squeeze()
    epi = np.asarray(ent["epistemic"]).squeeze()
    tot = np.asarray(ent["total"]).squeeze()

    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    out_path = save_dir / f"entropy_recomputed_moment_matched_{npz_path.stem}.png"
    saved = plot_entropy_lines(x, ale, epi, tot, save_path=out_path)
    for p in saved:
        print("Saved plot:", p)

    is_ovb = "ovb_outputs" in npz_path.name.lower()
    if not is_ovb:
        return

    full_keys = {"mu_samples_full", "sigma2_samples_full", "X_full"}
    if not full_keys.issubset(set(data.files)):
        print("OVB full-model keys missing; skipped full-model recomputation for:", npz_path.name)
        return

    mu_samples_full = np.asarray(data["mu_samples_full"])
    sigma2_samples_full = np.asarray(data["sigma2_samples_full"])
    x_full = np.asarray(data["X_full"])
    x_grid_full = x_full[:, :1] if x_full.ndim > 1 else x_full.reshape(-1, 1)
    mu_samples_full, sigma2_samples_full = _ensure_samples_first(mu_samples_full, sigma2_samples_full, x_grid_full)
    ent_full = entropy_uncertainty_analytical_moment_matched(mu_samples_full, sigma2_samples_full)
    ale_full = np.asarray(ent_full["aleatoric"]).squeeze()
    epi_full = np.asarray(ent_full["epistemic"]).squeeze()
    tot_full = np.asarray(ent_full["total"]).squeeze()
    x_full_line = _x_line(x_grid_full)
    out_full_path = save_dir / f"entropy_recomputed_moment_matched_full_{npz_path.stem}.png"
    saved_full = plot_entropy_lines(x_full_line, ale_full, epi_full, tot_full, save_path=out_full_path)
    for p in saved_full:
        print("Saved full-model plot:", p)


def main():
    parser = argparse.ArgumentParser(
        description="Moment-matched analytical entropy (AU/EU/TU) from raw_outputs .npz"
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default=None,
        type=Path,
        help="Path to a single raw_outputs .npz (if set, --model is ignored for input)",
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_ORDER, "all"],
        default="bamlss",
        help="Which model glob to use under --search-dir when npz_path is omitted (default: bamlss). "
        "Use 'all' to process latest file for each of BAMLSS, MC Dropout, Deep Ensemble, BNN.",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help=f"Directory to glob for npz (default: {DEFAULT_NPZ_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: results/ood/plots under project root)",
    )
    args = parser.parse_args()

    save_dir = Path(args.out_dir) if args.out_dir else project_root / "results" / "ood" / "plots"

    if args.npz_path is not None:
        npz_path = args.npz_path
        if not npz_path.is_file():
            print("File not found:", npz_path)
            sys.exit(1)
        process_one_npz(npz_path, save_dir)
        print("Done.")
        return

    search_dir = Path(args.search_dir) if args.search_dir else DEFAULT_NPZ_DIR
    if not search_dir.is_dir():
        print("Directory not found:", search_dir)
        sys.exit(1)

    paths = resolve_npz_paths(search_dir, args.model)
    if not paths:
        print(
            "No npz matched.",
            f"model={args.model!r}, dir={search_dir}",
        )
        sys.exit(1)

    for p in paths:
        process_one_npz(p, save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
