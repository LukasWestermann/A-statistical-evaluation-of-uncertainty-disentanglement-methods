"""
Recompute moment-matched epistemic entropy (EU) for BAMLSS OOD raw_outputs .npz files
and report absolute values |EU| (summaries per file + optional long CSV).

Usage (from repo root):

    python scripts/recompute_bamlss_ood_moment_matched_abs_eu.py

    python scripts/recompute_bamlss_ood_moment_matched_abs_eu.py --ood-root results/ood

    python scripts/recompute_bamlss_ood_moment_matched_abs_eu.py --long-csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
DEFAULT_OOD_ROOT = project_root / "results" / "ood"
DEFAULT_SUMMARY_CSV = DEFAULT_OOD_ROOT / "statistics" / "bamlss_ood_moment_matched_abs_eu_summary.csv"
DEFAULT_LONG_CSV = DEFAULT_OOD_ROOT / "statistics" / "bamlss_ood_moment_matched_abs_eu_per_grid.csv"

sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical_moment_matched


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = np.asarray(x_grid).ravel().shape[0]
    if mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        mu = mu.T
        sig = sig.T
    return mu, sig


def _x_line(x_grid) -> np.ndarray:
    xg = np.asarray(x_grid)
    return xg[:, 0] if xg.ndim > 1 else xg.ravel()


def _bamlss_npz_paths(ood_root: Path) -> list[Path]:
    if not ood_root.is_dir():
        return []
    return sorted(ood_root.rglob("*BAMLSS*raw_outputs*.npz"))


def main() -> None:
    p = argparse.ArgumentParser(description="BAMLSS OOD: moment-matched |EU| from raw_outputs .npz")
    p.add_argument(
        "--ood-root",
        type=Path,
        default=DEFAULT_OOD_ROOT,
        help="Root to search recursively (default: results/ood)",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="eps for moment-matched Gaussian entropy (default 1e-10)",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Where to write per-file summary CSV",
    )
    p.add_argument(
        "--long-csv",
        action="store_true",
        help="Also write one row per grid point (can be large)",
    )
    p.add_argument(
        "--long-csv-path",
        type=Path,
        default=DEFAULT_LONG_CSV,
        help="Path for long-format CSV when --long-csv is set",
    )
    args = p.parse_args()

    ood_root = args.ood_root.resolve()
    paths = _bamlss_npz_paths(ood_root)
    if not paths:
        sys.exit(f"No *BAMLSS*raw_outputs*.npz under {ood_root}")

    summary_rows: list[dict] = []
    long_rows: list[dict] = []

    for npz_path in paths:
        rel = npz_path.relative_to(project_root) if npz_path.is_relative_to(project_root) else npz_path
        data = np.load(npz_path, allow_pickle=True)
        mu = np.asarray(data["mu_samples"])
        sig = np.asarray(data["sigma2_samples"])
        x_grid = np.asarray(data["x_grid"])
        mu, sig = _ensure_samples_first(mu, sig, x_grid)
        ent = entropy_uncertainty_analytical_moment_matched(mu, sig, eps=args.eps)
        au = np.asarray(ent["aleatoric"]).squeeze()
        eu = np.asarray(ent["epistemic"]).squeeze()
        tu = np.asarray(ent["total"]).squeeze()
        x = _x_line(x_grid)
        eu_abs = np.abs(eu)
        resid = np.abs(eu - (tu - au))
        n = eu.size
        summary_rows.append(
            {
                "npz_relpath": str(rel).replace("\\", "/"),
                "n_grid": n,
                "mean_abs_EU": float(np.mean(eu_abs)),
                "median_abs_EU": float(np.median(eu_abs)),
                "max_abs_EU": float(np.max(eu_abs)),
                "p95_abs_EU": float(np.percentile(eu_abs, 95)),
                "mean_signed_EU": float(np.mean(eu)),
                "frac_EU_negative": float(np.mean(eu < 0)),
                "max_check_abs_EU_minus_TU_minus_AU": float(np.max(resid)),
            }
        )
        if args.long_csv:
            stem = npz_path.stem
            for i in range(n):
                long_rows.append(
                    {
                        "npz_stem": stem,
                        "npz_relpath": str(rel).replace("\\", "/"),
                        "i": i,
                        "x": float(x[i]),
                        "abs_epistemic_entropy": float(eu_abs[i]),
                        "signed_epistemic_entropy": float(eu[i]),
                        "TU": float(tu[i]),
                        "AU": float(au[i]),
                    }
                )

    # Print compact table (absolute values focus)
    print(f"BAMLSS OOD moment-matched |EU| under {ood_root} ({len(paths)} files)\n")
    hdr = f"{'npz (relative)':<70} {'mean|EU|':>12} {'median|EU|':>12} {'max|EU|':>12}"
    print(hdr)
    print("-" * len(hdr))
    for row in summary_rows:
        short = row["npz_relpath"]
        if len(short) > 67:
            short = "..." + short[-64:]
        print(
            f"{short:<70} {row['mean_abs_EU']:12.6e} {row['median_abs_EU']:12.6e} {row['max_abs_EU']:12.6e}"
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote summary CSV: {args.summary_csv}")

    if args.long_csv and long_rows:
        args.long_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.long_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
            w.writeheader()
            w.writerows(long_rows)
        print(f"Wrote long CSV: {args.long_csv_path} ({len(long_rows)} rows)")


if __name__ == "__main__":
    main()
