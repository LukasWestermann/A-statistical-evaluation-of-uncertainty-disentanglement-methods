"""
1×4 OVB summary panels: normalized AU / EU / Total vs Correlation X–Z (β₂ fixed) or vs β₂ (ρ fixed).

- **Variance:** newest ``*_ovb_{rho|beta2}_stats_*.xlsx`` per model under ``results/ovb/<model>/...``.
- **Entropy:** all ``*.xlsx`` under ``<entropy_stats_root>/ovb/<noise>/<func>/`` (recomputed entropy statistics).

Y-axis for normalized uncertainties: ``[0, 1.1]`` (OVB panels only; other summaries may use ``[0, 1.5]``).

Total (TU) lines are drawn as **normalized AU + normalized EU** at plot time (not read from ``mean_tot_*`` columns).

Writes 16 PNGs under ``<out>/ovb/summary_panels/{correlation_xz|beta2}/``.

Usage (from project root)::

    python scripts/plot_ovb_sweep_summary_panels.py
    python scripts/plot_ovb_sweep_summary_panels.py --fixed-beta2 1.0 --fixed-rho 0.7
    python scripts/plot_ovb_sweep_summary_panels.py --stats-date-suffix 20260406
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils import results_save
from utils.ovb_sweep_summary_loaders import (
    load_stats_by_model_entropy,
    load_stats_by_model_variance,
)
from utils.regression_summary_panels import (
    FuncType,
    MeasureType,
    NoiseType,
    create_ovb_parameter_sweep_panel_au_eu_tu_only,
    get_ovb_sweep_uncertainty_columns,
)
from utils.results_save import save_plot

# OVB 1×4 panels: tighter y-range than sample-size / noise-level summaries (1.5).
OVB_SUMMARY_UNCERTAINTY_YLIM = (0.0, 1.1)


def _apply_tot_as_au_plus_eu(
    stats_by_model: Dict[str, pd.DataFrame],
    measure: MeasureType,
) -> Dict[str, pd.DataFrame]:
    """
    At plot time, set normalized total to AU + EU column-wise (variance or entropy).
    Ignores any stale ``mean_tot_*`` / ``Avg_Total_*`` values from older Excel exports.
    """
    au_col, eu_col, tot_col = get_ovb_sweep_uncertainty_columns(measure)
    out: Dict[str, pd.DataFrame] = {}
    for name, df in stats_by_model.items():
        if df is None or df.empty:
            out[name] = df if df is not None else pd.DataFrame()
            continue
        d = df.copy()
        if au_col in d.columns and eu_col in d.columns:
            d[tot_col] = d[au_col].astype(float) + d[eu_col].astype(float)
        out[name] = d
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OVB summary panels: Correlation X–Z vs omitted Z (β₂); AU, EU, Total."
    )
    p.add_argument(
        "--ovb-root",
        type=Path,
        default=project_root / "results" / "ovb",
        help="Root containing deep_ensemble/, mcdropout/, bnn/, bamlss/",
    )
    p.add_argument(
        "--entropy-stats-root",
        type=Path,
        default=project_root / "results" / "entropy_recomputed_moment_matched_batch" / "statistics",
        help="Directory that contains ovb/<noise>/<func>/*.xlsx",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=project_root / "results",
        help="Base results directory for plots (see save_plot subfolders).",
    )
    p.add_argument("--fixed-beta2", type=float, default=1.0, help="Fixed β₂ when varying Corr(X, Z).")
    p.add_argument(
        "--fixed-rho",
        type=float,
        default=0.75,
        help="Target ρ for β₂-varying panels (matches typical OVB β₂ experiments). If no rows match, "
        "a single ρ in the stats file is used; mixed entropy tables pick the ρ with the most distinct β₂.",
    )
    p.add_argument(
        "--stats-date-suffix",
        type=str,
        default=None,
        help="If set, only variance workbooks whose filename contains this token (e.g. 20260406) are used.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_save.plots_dir = args.out

    combinations: list[tuple[FuncType, NoiseType]] = []
    for func_type in ("linear", "sin"):
        for noise_type in ("homoscedastic", "heteroscedastic"):
            combinations.append((func_type, noise_type))

    measures: tuple[MeasureType, ...] = ("variance", "entropy")
    desc_rho = f"(β₂={args.fixed_beta2:g})"
    desc_beta = f"(ρ={args.fixed_rho:g})"

    for func_type, noise_type in combinations:
        for measure in measures:
            if measure == "variance":
                stats_rho = load_stats_by_model_variance(
                    args.ovb_root,
                    func_type,
                    noise_type,
                    "rho",
                    args.fixed_beta2,
                    args.fixed_rho,
                    stats_date_suffix=args.stats_date_suffix,
                )
                stats_beta = load_stats_by_model_variance(
                    args.ovb_root,
                    func_type,
                    noise_type,
                    "beta2",
                    args.fixed_beta2,
                    args.fixed_rho,
                    stats_date_suffix=args.stats_date_suffix,
                )
            else:
                stats_rho = load_stats_by_model_entropy(
                    args.entropy_stats_root,
                    func_type,
                    noise_type,
                    "rho",
                    args.fixed_beta2,
                    args.fixed_rho,
                )
                stats_beta = load_stats_by_model_entropy(
                    args.entropy_stats_root,
                    func_type,
                    noise_type,
                    "beta2",
                    args.fixed_beta2,
                    args.fixed_rho,
                )

            stats_rho = _apply_tot_as_au_plus_eu(stats_rho, measure)
            stats_beta = _apply_tot_as_au_plus_eu(stats_beta, measure)

            fig_rho = create_ovb_parameter_sweep_panel_au_eu_tu_only(
                stats_rho,
                func_type,
                noise_type,
                measure,
                "rho",
                desc_rho,
                uncertainty_ylim=OVB_SUMMARY_UNCERTAINTY_YLIM,
            )
            name_rho = f"ovb_correlation_xz_{func_type}_{noise_type}_{measure}"
            save_plot(fig_rho, name_rho, subfolder="ovb/summary_panels/correlation_xz")
            plt.close(fig_rho)

            fig_beta = create_ovb_parameter_sweep_panel_au_eu_tu_only(
                stats_beta,
                func_type,
                noise_type,
                measure,
                "beta2",
                desc_beta,
                uncertainty_ylim=OVB_SUMMARY_UNCERTAINTY_YLIM,
            )
            name_beta = f"ovb_beta2_{func_type}_{noise_type}_{measure}"
            save_plot(fig_beta, name_beta, subfolder="ovb/summary_panels/beta2")
            plt.close(fig_beta)


if __name__ == "__main__":
    main()
