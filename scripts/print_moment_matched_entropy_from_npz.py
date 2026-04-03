"""
Print moment-matched entropy (TU, AU, EU) from a saved raw_outputs .npz.

EU = TU - AU by definition. Also prints |TU|, |AU|, |EU| for quick comparison across
machines. Note: taking abs(EU) is for inspection only; negative EU usually means
cancellation / numerics, not a separate "folded" definition.

Usage
-----

Single file:

    python scripts/print_moment_matched_entropy_from_npz.py results/ood/outputs/ood/heteroscedastic/sin/20260206_BAMLSS_raw_outputs.npz

Scan all ``*raw_outputs*.npz`` under ``results/ood`` (repo default):

    python scripts/print_moment_matched_entropy_from_npz.py --scan-ood

Same, but only BAMLSS files:

    python scripts/print_moment_matched_entropy_from_npz.py --scan-ood --contains BAMLSS

Custom root:

    python scripts/print_moment_matched_entropy_from_npz.py --scan-ood --ood-root results/ood/outputs/ood/homoscedastic/sin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
DEFAULT_OOD_ROOT = project_root / "results" / "ood"
sys.path.insert(0, str(project_root))

from utils.entropy_uncertainty import entropy_uncertainty_analytical_moment_matched


def _ensure_samples_first(mu_samples, sigma2_samples, x_grid):
    """Ensure mu and sigma2 have shape (S, N) = (samples, grid)."""
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


def print_entropy_report(npz_path: Path, *, n_show: int, eps: float) -> None:
    """Load one npz and print TU/AU/EU table + summary."""
    npz_path = npz_path.resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    mu_samples = np.asarray(data["mu_samples"])
    sigma2_samples = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    mu_samples, sigma2_samples = _ensure_samples_first(mu_samples, sigma2_samples, x_grid)

    ent = entropy_uncertainty_analytical_moment_matched(
        mu_samples, sigma2_samples, eps=eps
    )
    au = np.asarray(ent["aleatoric"]).squeeze()
    eu = np.asarray(ent["epistemic"]).squeeze()
    tu = np.asarray(ent["total"]).squeeze()
    x = _x_line(x_grid)

    n = tu.shape[0]
    diff = tu - au
    residual = eu - diff

    print(f"npz: {npz_path}")
    print(f"mu_samples shape (S, N) = {mu_samples.shape}, sigma2_samples = {sigma2_samples.shape}, N_grid = {n}")
    print()
    print("Moment-matched (nats): TU = total entropy, AU = aleatoric, EU = epistemic = TU - AU")
    print("|TU|, |AU|, |EU| are absolute values (EU negative => often float cancellation).")
    print()

    n_show_clamped = max(1, min(n_show, n))
    idxs = np.linspace(0, n - 1, num=n_show_clamped, dtype=int)

    hdr = (
        f"{'i':>6} {'x':>10} "
        f"{'TU':>14} {'AU':>14} {'EU':>14} "
        f"{'|TU|':>14} {'|AU|':>14} {'|EU|':>14} "
        f"{'TU-AU':>14} {'|resid|':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i in idxs:
        print(
            f"{i:6d} {x[i]:10.5f} "
            f"{tu[i]:14.8f} {au[i]:14.8f} {eu[i]:14.8e} "
            f"{abs(tu[i]):14.8f} {abs(au[i]):14.8f} {abs(eu[i]):14.8e} "
            f"{diff[i]:14.8e} {abs(residual[i]):12.2e}"
        )

    print()
    print("Summary over all grid points:")
    print(f"  EU: min={eu.min():.8e}  max={eu.max():.8e}  mean={eu.mean():.8e}  median={np.median(eu):.8e}")
    print(f"  |EU|: min={np.min(np.abs(eu)):.8e}  max={np.max(np.abs(eu)):.8e}  mean={np.mean(np.abs(eu)):.8e}")
    print(f"  negative EU count: {np.sum(eu < 0)} / {n}")
    print(f"  max |EU - (TU-AU)|: {np.max(np.abs(residual)):.8e}")
    print(f"  TU range: [{tu.min():.8f}, {tu.max():.8f}]  AU range: [{au.min():.8f}, {au.max():.8f}]")


def _collect_scan_paths(ood_root: Path, contains: str | None) -> list[Path]:
    if not ood_root.is_dir():
        return []
    paths = sorted(ood_root.rglob("*raw_outputs*.npz"))
    if contains:
        sub = contains
        paths = [p for p in paths if sub in p.name]
    return paths


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print TU/AU/EU (moment-matched) from raw_outputs .npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Default OOD scan root: {DEFAULT_OOD_ROOT}",
    )
    p.add_argument(
        "npz_path",
        type=Path,
        nargs="?",
        default=None,
        help="Single raw_outputs .npz (omit if using --scan-ood)",
    )
    p.add_argument(
        "--scan-ood",
        action="store_true",
        help=f"Process all *raw_outputs*.npz under --ood-root (default: {DEFAULT_OOD_ROOT})",
    )
    p.add_argument(
        "--ood-root",
        type=Path,
        default=DEFAULT_OOD_ROOT,
        help="Root directory for --scan-ood (default: results/ood under repo)",
    )
    p.add_argument(
        "--contains",
        type=str,
        default=None,
        help="When scanning: keep only files whose name contains this substring (e.g. BAMLSS)",
    )
    p.add_argument(
        "--n-show",
        type=int,
        default=10,
        help="Number of grid indices to print (evenly spaced; default 10)",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="eps passed to moment-matched Gaussian entropy (default 1e-10)",
    )
    args = p.parse_args()

    if args.npz_path is not None and args.scan_ood:
        sys.exit("Use either a single npz_path or --scan-ood, not both.")

    if args.npz_path is not None:
        try:
            print_entropy_report(args.npz_path, n_show=args.n_show, eps=args.eps)
        except FileNotFoundError as e:
            sys.exit(f"File not found: {e}")
        return

    if not args.scan_ood:
        p.print_help()
        print(
            "\nProvide a path to one .npz, or use --scan-ood to process files under results/ood.",
            file=sys.stderr,
        )
        sys.exit(2)

    ood_root = args.ood_root.resolve()
    paths = _collect_scan_paths(ood_root, args.contains)
    if not paths:
        filt = f" (filter --contains {args.contains!r})" if args.contains else ""
        sys.exit(f"No *raw_outputs*.npz found under {ood_root}{filt}")

    print(f"Scanning {len(paths)} file(s) under {ood_root}\n")
    sep = "\n" + "=" * 88 + "\n"
    for i, path in enumerate(paths):
        if i:
            print(sep)
        try:
            print_entropy_report(path, n_show=args.n_show, eps=args.eps)
        except Exception as ex:
            print(f"[error] {path}: {ex}", file=sys.stderr)


if __name__ == "__main__":
    main()
