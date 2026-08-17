"""
Export raw (recomputed) AU(x)/EU(x)/TU(x) -- BOTH the variance-based and the
entropy-based decomposition -- plus mu_pred(x)/y_true(x)/sigma2_true(x), for every
saved .npz file of one experiment into a single combined tidy CSV.

Covers OOD, Undersampling, Sample-size, and Noise-level (OVB has a genuinely
different row shape -- see scripts/export_ovb_raw_au_eu_csv.py). Read-only
post-processing over existing results/<experiment>/outputs/... .npz files -- does
not touch the run/save pipeline and does not modify any .npz. Nothing entropy-based
is ever stored raw in these npz files (only mu_samples/sigma2_samples are) -- the
run pipeline computes entropy values transiently for plots/summary stats and
discards them, so AU_entropy/EU_entropy/TU_entropy here are recomputed fresh via
utils.entropy_uncertainty.entropy_uncertainty_by_method(mu_samples, sigma2_samples,
'moment_matched') -- the same function AND method every run function in this repo
already defaults to (checked every run_*_experiment signature across all 5
experiments; no notebook call overrides it), so this reproduces exactly what the
original run would have computed, just not thrown away. It's a closed-form
vectorized formula (no Monte Carlo), ~0.2ms per file.

The variance-based AU/EU/mu_pred are recomputed fresh from mu_samples/sigma2_samples
(never stored raw anywhere else in this repo -- see tables/README.md's "recompute,
never reformat" convention), using the exact same formula as
scripts/make_paper_tables.py:
    AU(x)      = mean(sigma2_samples, axis=members)
    EU(x)      = var(mu_samples, axis=members)
    TU(x)      = AU(x) + EU(x)                          (law of total variance)
    mu_pred(x) = mean(mu_samples, axis=members)
y_true(x) is y_grid_clean straight from the npz (the deterministic true function
value -- there is no *noisy* y at grid points, only at the separate, differently-
shaped x_train_subset/y_train_subset, which don't fit this per-grid-point tidy
layout and aren't exported here). sigma2_true(x) is the ground-truth noise
variance: for OOD/Undersampling/Sample-size, the fixed-scale DGP formula
(utils.mixture_metrics.get_dgp); for Noise-level, the tau-parametrized formula
(tau read from each file's own metadata) -- both verified against the same
notebooks scripts/make_paper_tables.py's sigma_true_baseline/sigma_true_noise_sweep
were verified against. sigma2_true is a variance-decomposition ground truth only --
no entropy-scale ground truth is emitted (entropy AU/EU are in nats, not the
variance/sigma^2 units sigma2_true is in, so they aren't directly comparable).

Also includes, per grid point, the same analytic exact-expected-score columns the fixed
utils/*_experiments.py call sites now use (utils.analytic_scores.score_bundle_pointwise),
scored against the TRUE distribution N(y_true, sigma2_true) -- not the clean-target bug
this replaced: CRPS_mixture/NLL_mixture (primary, full predictive distribution),
CRPS_gaussian/NLL_gaussian (secondary, moment-matched single Gaussian, labeled), Oracle_CRPS/
Oracle_NLL (exact floor under the true distribution), IQD (excess CRPS over the oracle),
KL/KL_mean/KL_spread (excess NLL over the oracle, split into mean-shift and
spread-mismatch components). These are recomputed fresh from mu_samples/sigma2_samples,
same as AU/EU -- nothing scoring-related is stored raw in the npz either.

One row per (npz file, grid point). Includes the seed column when present (seed-
replication pilot) -- None/NaN for legacy single-run files that predate it, never
fabricated. Also includes each model's saved hyperparameters (dropout_p/mc_samples
for MC Dropout, n_nets for Deep Ensemble) and the experiment's own swept knob
(pct for sample-size, tau for noise-level) as explicit columns where applicable.

Usage:
    python scripts/export_raw_au_eu_csv.py --experiment samplesize
    python scripts/export_raw_au_eu_csv.py --experiment ood
    python scripts/export_raw_au_eu_csv.py --experiment undersampling
    python scripts/export_raw_au_eu_csv.py --experiment noise_level
    python scripts/export_raw_au_eu_csv.py --experiment samplesize --out results/sample_size/csv --filename my_export
"""
from __future__ import annotations

import argparse
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
from utils.metrics import compute_predictive_aggregation  # noqa: E402
from utils.analytic_scores import score_bundle_pointwise  # noqa: E402
import utils.predictive_eval_io as pio  # noqa: E402

ENTROPY_METHOD = "moment_matched"  # matches every run_*_experiment's default across all 5 experiments

RESULTS_ROOT = project_root / "results"


def _sigma_true_baseline(x: np.ndarray, func_type: str, noise_type: str, d) -> np.ndarray:
    """OOD/Undersampling/Sample-size: fixed noise scale, no tau in the npz. Same
    already-verified formula as scripts/make_paper_tables.py's sigma_true_baseline
    (utils.mixture_metrics.get_dgp), squared for the variance."""
    return get_dgp(func_type, noise_type).sigma_fn(x) ** 2


def _sigma_true_noise_sweep(x: np.ndarray, func_type: str, noise_type: str, d) -> np.ndarray:
    """Noise-level: sigma depends on tau, read from this file's own metadata. Same
    already-verified formula as scripts/make_paper_tables.py's
    sigma_true_noise_sweep (verified against Experiments/Noise_Level.ipynb):
        homoscedastic:   sigma(x) = tau
        heteroscedastic: sigma(x) = |tau * sin(0.5x + 5)|
    """
    tau = _npz_scalar_float(d, "tau")
    x = np.asarray(x, dtype=float)
    if noise_type == "homoscedastic":
        sigma = np.full_like(x, float(tau))
    else:
        sigma = np.abs(float(tau) * np.sin(0.5 * x + 5))
    return sigma ** 2


# One entry per experiment covered by this script. `results_subdir` matches the
# results/<x>/ directory name on disk (note "sample_size" has an underscore,
# unlike the --experiment flag value "samplesize" -- kept as "samplesize" for the
# flag/default filename to match the pre-existing, already-verified sample-size
# export exactly).
EXPERIMENTS = {
    "ood": {
        "results_subdir": "ood",
        "sigma_true_fn": _sigma_true_baseline,
        "extra_meta_keys": [],
    },
    "undersampling": {
        "results_subdir": "undersampling",
        "sigma_true_fn": _sigma_true_baseline,
        "extra_meta_keys": [],
    },
    "samplesize": {
        "results_subdir": "sample_size",
        "sigma_true_fn": _sigma_true_baseline,
        "extra_meta_keys": ["pct"],
    },
    "noise_level": {
        "results_subdir": "noise_level",
        "sigma_true_fn": _sigma_true_noise_sweep,
        "extra_meta_keys": ["tau"],
    },
}


def _row_block_for_npz(npz_path: Path, search_dir: Path, sigma_true_fn, extra_meta_keys) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    mu, sig = ensure_samples_first(d["mu_samples"], d["sigma2_samples"], d["x_grid"])
    x = np.asarray(d["x_grid"]).ravel()
    y_true = np.asarray(d["y_grid_clean"]).ravel()
    mu_pred = np.mean(mu, axis=0).ravel()
    au = np.mean(sig, axis=0).ravel()
    eu = np.var(mu, axis=0).ravel()
    tu = au + eu

    entropy = entropy_uncertainty_by_method(mu, sig, ENTROPY_METHOD)
    au_entropy = np.asarray(entropy["aleatoric"]).ravel()
    eu_entropy = np.asarray(entropy["epistemic"]).ravel()
    tu_entropy = np.asarray(entropy["total"]).ravel()

    func_type = _npz_scalar_str(d, "func_type")
    noise_type = _npz_scalar_str(d, "noise_type")
    sigma2_true = sigma_true_fn(x, func_type, noise_type, d)
    sigma_true = np.sqrt(sigma2_true)

    mu_star, sigma2_star = compute_predictive_aggregation(mu, sig)
    scores = score_bundle_pointwise(mu, sig, mu_star, sigma2_star, y_true, sigma_true)

    n = len(x)
    row = {
        "model_name": [_npz_scalar_str(d, "model_name")] * n,
        "noise_type": [noise_type] * n,
        "func_type": [func_type] * n,
    }
    for key in extra_meta_keys:
        row[key] = [_npz_scalar_float(d, key)] * n
    row.update({
        "seed": [_npz_scalar_float(d, "seed")] * n,  # None/NaN for legacy pre-pilot files
        "dropout_p": [_npz_scalar_float(d, "dropout_p")] * n,  # MC Dropout only
        "mc_samples": [_npz_scalar_float(d, "mc_samples")] * n,  # MC Dropout only
        "n_nets": [_npz_scalar_float(d, "n_nets")] * n,  # Deep Ensemble only
        "date": [_npz_scalar_str(d, "date")] * n,
        "x": x,
        "y_true": y_true,
        "mu_pred": mu_pred,
        "AU": au,
        "EU": eu,
        "TU": tu,
        "AU_entropy": au_entropy,
        "EU_entropy": eu_entropy,
        "TU_entropy": tu_entropy,
        "sigma2_true": sigma2_true,
        "CRPS_mixture": np.asarray(scores["crps_mixture"]).ravel(),
        "NLL_mixture": np.asarray(scores["nll_mixture"]).ravel(),
        "CRPS_gaussian": np.asarray(scores["crps_gaussian"]).ravel(),
        "NLL_gaussian": np.asarray(scores["nll_gaussian"]).ravel(),
        "Oracle_CRPS": np.asarray(scores["oracle_crps"]).ravel(),
        "Oracle_NLL": np.asarray(scores["oracle_nll"]).ravel(),
        "IQD": np.asarray(scores["iqd"]).ravel(),
        "KL": np.asarray(scores["kl"]).ravel(),
        "KL_mean": np.asarray(scores["kl_mean"]).ravel(),
        "KL_spread": np.asarray(scores["kl_spread"]).ravel(),
        # Path relative to search_dir, not just the bare filename -- the same
        # date+model+knob filename can legitimately recur under every DGP
        # subdirectory (noise_type/func_type aren't part of the filename itself,
        # only the directory), so the bare name alone is ambiguous.
        "source_file": [str(npz_path.relative_to(search_dir))] * n,
    })
    return pd.DataFrame(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True,
                         help="Which experiment's raw outputs to export.")
    parser.add_argument(
        "--search-dir", type=Path, default=None,
        help="Root directory to search for *raw_outputs*.npz files "
             "(default: results/<experiment>/outputs/<experiment>).",
    )
    parser.add_argument("--out", type=Path, default=None,
                         help="Output directory for the combined CSV (default: results/<experiment>/csv).")
    parser.add_argument("--filename", type=str, default=None,
                         help="Base filename (without .csv) for the combined CSV "
                              "(default: '<experiment>_raw_au_eu', or 'samplesize_raw_au_eu' for --experiment samplesize).")
    args = parser.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    results_subdir = cfg["results_subdir"]
    search_dir = args.search_dir or (RESULTS_ROOT / results_subdir / "outputs" / results_subdir)
    out_dir = args.out or (RESULTS_ROOT / results_subdir / "csv")
    filename = args.filename or f"{args.experiment}_raw_au_eu"

    npz_files = collect_raw_npz_files(search_dir)
    if not npz_files:
        print(f"No raw_outputs npz files found under {search_dir}")
        return

    print(f"Found {len(npz_files)} npz files under {search_dir}")
    blocks = []
    for p in npz_files:
        try:
            blocks.append(_row_block_for_npz(p, search_dir, cfg["sigma_true_fn"], cfg["extra_meta_keys"]))
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    if not blocks:
        print("Nothing loaded successfully -- no CSV written.")
        return

    combined = pd.concat(blocks, ignore_index=True)
    print(f"Combined: {len(combined)} rows across {len(blocks)} files")

    pio.csv_dir = out_dir
    pio.save_tidy_csv(combined, filename)


if __name__ == "__main__":
    main()
