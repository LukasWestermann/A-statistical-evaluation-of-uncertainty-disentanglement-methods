"""
Export raw (recomputed) AU(x)/EU(x)/TU(x)/mu_pred(x)/y_true(x) plus dual
ground-truth columns for every saved OVB .npz file into one combined tidy CSV.

Read-only post-processing over existing results/ovb/<model_dir>/... .npz files --
does not touch the run/save pipeline and does not modify any .npz. OVB's DGP is
structurally different from the other 4 experiments (Y = f(X) + beta2*Z + epsilon,
an omitted-variable-bias design, not a single-covariate sigma(x) model), so this is
a separate script rather than folded into scripts/export_raw_au_eu_csv.py -- see
that script's docstring for the shared 4-experiment case.

Every OVB npz stores BOTH the omitted (X-only) model's outputs (mu_samples,
sigma2_samples, always present) AND, when available, the full (X+Z) model's
outputs (mu_samples_full, sigma2_samples_full, X_full) -- this script exports
BOTH, one row per (npz file, grid/training point, which_model in
{"omitted","full"}), so the omitted-variable-bias effect is directly visible by
comparing which_model=="omitted" vs "full" rows in the same file.

Ground truth: the omitted model's TRUE conditional variance target is NOT just the
intrinsic noise sigma(x)^2 -- since it never sees Z, its residual also carries
Z's conditional variance given X, i.e.
    Var(Y|X) = beta2^2 * Var(Z|X) + sigma(x)^2 = beta2^2*(1-rho^2) + sigma(x)^2
(Z|X is Gaussian with residual variance (1-rho^2), from the DGP's
X = rho*Z + sqrt(1-rho^2)*nu construction -- verified directly against
utils.ovb_experiments.generate_ovb_data / Experiments/OVB_regression.ipynb's
generate_toy_regression_ovb). The full (X+Z) model's true target genuinely is just
sigma(x)^2, since conditioning on Z as well removes that extra term. Two columns
are always emitted so nothing is asserted only in a docstring:
    sigma2_true_intrinsic       = sigma(x)^2                          (same for both which_model rows)
    sigma2_true_omitted_target  = sigma(x)^2 + beta2^2*(1-rho^2)       (omitted rows); == intrinsic for full rows

mu_pred/AU/EU/TU (variance-based) are recomputed fresh from mu_samples/sigma2_samples
(and, for "full" rows, mu_samples_full/sigma2_samples_full) -- never taken from the
npz's own stored mu_pred/ale_var/epi_var/tot_var fields, consistent with this
repo's recompute-never-reformat convention. AU_entropy/EU_entropy/TU_entropy
(entropy-based) are likewise recomputed, via
utils.entropy_uncertainty.entropy_uncertainty_by_method(..., 'moment_matched') --
never stored raw in the npz either, and 'moment_matched' matches every
run_*_ovb_*_experiment's default entropy_method. "omitted" rows use
x_grid/y_grid_clean (stored directly). "full" rows use X_full (the full model's own
training-point locations, a different point count than x_grid) with y_true computed
analytically via the same f(x) formula (0.7x+0.5 linear / x*sin(x)+x sin) rather
than assumed -- skipped for a file if X_full wasn't saved. No entropy-scale ground
truth is emitted (entropy AU/EU are in nats, not the sigma2_true_* columns' variance
units).

seed/date: seed is read from npz metadata (None/NaN for legacy pre-pilot files).
date is NOT stored as npz metadata for OVB (only baked into the filename) --
parsed from the filename's trailing YYYYMMDD token instead.

This script is the raw AU/EU/TU export only -- it does NOT compute CRPS/NLL/oracle
scoring (see scripts/export_raw_au_eu_csv.py's docstring for why: score_bundle_pointwise's
closed-form mixture term is O(M^2 * N) in memory, and large-M files here get OOM-killed).
For the OVB CRPS/NLL diagnostic (KL_mean/KL_spread), use utils/ovb_experiments.py's own
run-time scoring, or call utils.analytic_scores.score_bundle_pointwise directly on one
saved file.

Usage:
    python scripts/export_ovb_raw_au_eu_csv.py
    python scripts/export_ovb_raw_au_eu_csv.py --out results/ovb/csv --filename my_export
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (  # noqa: E402
    collect_raw_npz_files,
    ensure_samples_first,
    _npz_scalar_str,
    _npz_scalar_float,
)
from utils.mixture_metrics import get_dgp  # noqa: E402
from utils.entropy_uncertainty import entropy_uncertainty_by_method  # noqa: E402
import utils.predictive_eval_io as pio  # noqa: E402

RESULTS_ROOT = project_root / "results"
ENTROPY_METHOD = "moment_matched"  # matches every run_*_ovb_*_experiment's default


def _f_clean(x: np.ndarray, func_type: str) -> np.ndarray:
    """Same verified DGP mean formula used everywhere in this repo (identical
    across OOD/Undersampling/Sample-size/Noise-level/OVB notebooks)."""
    x = np.asarray(x, dtype=float)
    if func_type == "linear":
        return 0.7 * x + 0.5
    return x * np.sin(x) + x


def _parse_date_from_stem(stem: str):
    m = re.search(r"(\d{8})", stem)
    return m.group(1) if m else None


def _member_stats(mu_samples: np.ndarray, sigma2_samples: np.ndarray, n_points: int):
    mu, sig = ensure_samples_first(mu_samples, sigma2_samples, np.zeros(n_points))
    mu_pred = np.mean(mu, axis=0).ravel()
    au = np.mean(sig, axis=0).ravel()
    eu = np.var(mu, axis=0).ravel()
    entropy = entropy_uncertainty_by_method(mu, sig, ENTROPY_METHOD)
    au_entropy = np.asarray(entropy["aleatoric"]).ravel()
    eu_entropy = np.asarray(entropy["epistemic"]).ravel()
    tu_entropy = np.asarray(entropy["total"]).ravel()
    return mu_pred, au, au + eu, eu, au_entropy, eu_entropy, tu_entropy


def _rows_for_npz(npz_path: Path, search_dir: Path) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    func_type = _npz_scalar_str(d, "func_type")
    noise_type = _npz_scalar_str(d, "noise_type")
    rho = _npz_scalar_float(d, "rho")
    beta2 = _npz_scalar_float(d, "beta2")
    model_name = _npz_scalar_str(d, "model_name")
    seed = _npz_scalar_float(d, "seed")
    date = _parse_date_from_stem(npz_path.stem)
    source_file = str(npz_path.relative_to(search_dir))

    sigma_fn = get_dgp(func_type, noise_type).sigma_fn
    omitted_inflation = (beta2 ** 2) * (1.0 - rho ** 2) if (beta2 is not None and rho is not None) else np.nan

    blocks = []

    # ----- omitted (X-only) model: always present, on x_grid -----
    x_grid = np.asarray(d["x_grid"]).ravel()
    y_true_omitted = np.asarray(d["y_grid_clean"]).ravel()
    mu_pred, au, tu, eu, au_ent, eu_ent, tu_ent = _member_stats(d["mu_samples"], d["sigma2_samples"], len(x_grid))
    sigma2_true_intrinsic = sigma_fn(x_grid) ** 2
    sigma2_true_omitted_target = sigma2_true_intrinsic + omitted_inflation
    n = len(x_grid)
    blocks.append(pd.DataFrame({
        "model_name": [model_name] * n, "noise_type": [noise_type] * n, "func_type": [func_type] * n,
        "rho": [rho] * n, "beta2": [beta2] * n, "seed": [seed] * n, "date": [date] * n,
        "x": x_grid, "y_true": y_true_omitted, "mu_pred": mu_pred, "AU": au, "EU": eu, "TU": tu,
        "AU_entropy": au_ent, "EU_entropy": eu_ent, "TU_entropy": tu_ent,
        "sigma2_true_intrinsic": sigma2_true_intrinsic,
        "sigma2_true_omitted_target": sigma2_true_omitted_target,
        "which_model": ["omitted"] * n, "source_file": [source_file] * n,
    }))

    # ----- full (X+Z) model: only if saved, on its own training-point X_full -----
    if "mu_samples_full" in d.files and "sigma2_samples_full" in d.files and "X_full" in d.files:
        # X_full is [n_train, 2] = (X, Z) stacked column-wise -- only the X column
        # (index 0) is the x-axis; Z isn't plotted/compared against here.
        x_full = np.asarray(d["X_full"])
        x_full = x_full[:, 0] if x_full.ndim == 2 else x_full.ravel()
        y_true_full = _f_clean(x_full, func_type)
        mu_pred_f, au_f, tu_f, eu_f, au_ent_f, eu_ent_f, tu_ent_f = _member_stats(
            d["mu_samples_full"], d["sigma2_samples_full"], len(x_full)
        )
        sigma2_true_intrinsic_f = sigma_fn(x_full) ** 2
        nf = len(x_full)
        blocks.append(pd.DataFrame({
            "model_name": [model_name] * nf, "noise_type": [noise_type] * nf, "func_type": [func_type] * nf,
            "rho": [rho] * nf, "beta2": [beta2] * nf, "seed": [seed] * nf, "date": [date] * nf,
            "x": x_full, "y_true": y_true_full, "mu_pred": mu_pred_f, "AU": au_f, "EU": eu_f, "TU": tu_f,
            "AU_entropy": au_ent_f, "EU_entropy": eu_ent_f, "TU_entropy": tu_ent_f,
            "sigma2_true_intrinsic": sigma2_true_intrinsic_f,
            "sigma2_true_omitted_target": sigma2_true_intrinsic_f,  # full model: no omitted-variable inflation
            "which_model": ["full"] * nf, "source_file": [source_file] * nf,
        }))

    return pd.concat(blocks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-dir", type=Path, default=RESULTS_ROOT / "ovb",
                         help="Root directory to search for ovb_outputs*.npz files (default: results/ovb).")
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "ovb" / "csv",
                         help="Output directory for the combined CSV.")
    parser.add_argument("--filename", type=str, default="ovb_raw_au_eu",
                         help="Base filename (without .csv) for the combined CSV.")
    args = parser.parse_args()

    npz_files = sorted(set(args.search_dir.rglob("ovb_outputs*.npz")))
    if not npz_files:
        print(f"No ovb_outputs npz files found under {args.search_dir}")
        return

    print(f"Found {len(npz_files)} npz files under {args.search_dir}")
    blocks = []
    for p in npz_files:
        try:
            blocks.append(_rows_for_npz(p, args.search_dir))
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    if not blocks:
        print("Nothing loaded successfully -- no CSV written.")
        return

    combined = pd.concat(blocks, ignore_index=True)
    n_full = int((combined["which_model"] == "full").sum())
    print(f"Combined: {len(combined)} rows across {len(blocks)} files "
          f"({len(combined) - n_full} omitted-model rows, {n_full} full-model rows)")

    pio.csv_dir = args.out
    pio.save_tidy_csv(combined, args.filename)


if __name__ == "__main__":
    main()
