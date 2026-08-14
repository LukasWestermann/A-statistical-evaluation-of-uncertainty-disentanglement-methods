"""
Export raw (recomputed) AU(x)/EU(x)/TU(x)/mu_pred(x)/y_true(x)/sigma2_true(x) for
every saved sample-size .npz file into one combined tidy CSV.

Read-only post-processing over existing results/sample_size/outputs/sample_size/
*.npz files -- does not touch the run/save pipeline and does not modify any .npz.
AU/EU/mu_pred are recomputed fresh from mu_samples/sigma2_samples (never stored raw
anywhere else in this repo -- see tables/README.md's "recompute, never reformat"
convention), using the exact same formula as scripts/make_paper_tables.py:
    AU(x)      = mean(sigma2_samples, axis=members)
    EU(x)      = var(mu_samples, axis=members)
    TU(x)      = AU(x) + EU(x)                          (law of total variance)
    mu_pred(x) = mean(mu_samples, axis=members)
y_true(x) is y_grid_clean straight from the npz (the deterministic true function
value -- there is no *noisy* y at grid points, only at the separate, differently-
shaped x_train_subset/y_train_subset; see the module docstring's note on that).
sigma2_true(x) is the ground-truth noise variance, via the same verified DGP
formula scripts/make_paper_tables.py uses (utils.mixture_metrics.get_dgp) -- lets a
reader check AU's calibration directly from this CSV, no other script needed.

One row per (npz file, grid point). Includes the seed column when present (sample-
size seed-replication pilot) -- None/NaN for legacy single-run files that predate
it, never fabricated. Also includes each model's saved hyperparameters
(dropout_p/mc_samples for MC Dropout, n_nets for Deep Ensemble) as explicit
columns -- None for models/files where that field isn't applicable/saved.

Note: x_train_subset/y_train_subset (the actual training points used for that run)
are also stored in every npz, but at a different length (n_train, not n_grid) --
they don't fit this per-grid-point tidy layout. Ask for a separate training-points
CSV if you want those exported too.

Usage:
    python scripts/export_samplesize_raw_au_eu_csv.py
    python scripts/export_samplesize_raw_au_eu_csv.py --out results/sample_size/csv --filename my_export
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
import utils.predictive_eval_io as pio  # noqa: E402

RESULTS_ROOT = project_root / "results"


def _row_block_for_npz(npz_path: Path, search_dir: Path) -> pd.DataFrame:
    d = np.load(npz_path, allow_pickle=True)
    mu, sig = ensure_samples_first(d["mu_samples"], d["sigma2_samples"], d["x_grid"])
    x = np.asarray(d["x_grid"]).ravel()
    y_true = np.asarray(d["y_grid_clean"]).ravel()
    mu_pred = np.mean(mu, axis=0).ravel()
    au = np.mean(sig, axis=0).ravel()
    eu = np.var(mu, axis=0).ravel()
    tu = au + eu

    func_type = _npz_scalar_str(d, "func_type")
    noise_type = _npz_scalar_str(d, "noise_type")
    sigma2_true = get_dgp(func_type, noise_type).sigma_fn(x) ** 2

    n = len(x)
    return pd.DataFrame({
        "model_name": [_npz_scalar_str(d, "model_name")] * n,
        "noise_type": [noise_type] * n,
        "func_type": [func_type] * n,
        "pct": [_npz_scalar_float(d, "pct")] * n,
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
        "sigma2_true": sigma2_true,
        # Path relative to search_dir, not just the bare filename -- the same
        # date+model+pct filename can legitimately recur under every DGP
        # subdirectory (noise_type/func_type aren't part of the filename itself,
        # only the directory), so the bare name alone is ambiguous.
        "source_file": [str(npz_path.relative_to(search_dir))] * n,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--search-dir", type=Path,
        default=RESULTS_ROOT / "sample_size" / "outputs" / "sample_size",
        help="Root directory to search for *raw_outputs*.npz files (default: sample-size outputs).",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_ROOT / "sample_size" / "csv",
                         help="Output directory for the combined CSV.")
    parser.add_argument("--filename", type=str, default="samplesize_raw_au_eu",
                         help="Base filename (without .csv) for the combined CSV.")
    args = parser.parse_args()

    npz_files = collect_raw_npz_files(args.search_dir)
    if not npz_files:
        print(f"No raw_outputs npz files found under {args.search_dir}")
        return

    print(f"Found {len(npz_files)} npz files under {args.search_dir}")
    blocks = []
    for p in npz_files:
        try:
            blocks.append(_row_block_for_npz(p, args.search_dir))
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    if not blocks:
        print("Nothing loaded successfully -- no CSV written.")
        return

    combined = pd.concat(blocks, ignore_index=True)
    print(f"Combined: {len(combined)} rows across {len(blocks)} files")

    pio.csv_dir = args.out
    pio.save_tidy_csv(combined, args.filename)


if __name__ == "__main__":
    main()
