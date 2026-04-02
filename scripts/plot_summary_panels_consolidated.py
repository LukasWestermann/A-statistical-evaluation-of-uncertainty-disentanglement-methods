"""
Build sample-size and noise-level summary line panels in one output tree.

- **Variance**: CSV statistics (same as ``plot_*_summary_4x2.py``).
- **Entropy**: **only** Excel workbooks from the moment-matched batch
  (``recompute_entropy_moment_matched_batch_from_npz``). Combinations with no
  moment-matched stats are skipped (no CSV fallback).

Normalized AU/EU/(Total) rows use y-axis ``[0, 1.5]`` (``NORMALIZED_UNCERTAINTY_YLIM``).
Correlation subplots stay on approximately ``[-1, 1]``.

Default layout (flat)::

    results/summary_panels_consolidated/
        sample_size/*.png
        noise_level/*.png

Filenames embed ``{func_type}_{noise_type}`` (e.g. ``sin_heteroscedastic``).

Usage::

    python scripts/plot_summary_panels_consolidated.py
    python scripts/plot_summary_panels_consolidated.py --out results/my_summaries
    python scripts/plot_summary_panels_consolidated.py --moment-matched-stats path/to/statistics
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

_pss = runpy.run_path(
    str(project_root / "scripts" / "plot_sample_size_summary_4x2.py"),
    run_name="<load_sample_size_stats>",
)
_pnl = runpy.run_path(
    str(project_root / "scripts" / "plot_noise_level_summary_4x2.py"),
    run_name="<load_noise_level_stats>",
)

from utils.moment_matched_summary_loaders import (
    load_noise_level_entropy_moment_matched,
    load_sample_size_entropy_moment_matched,
)
from utils.regression_summary_panels import (
    NORMALIZED_UNCERTAINTY_YLIM,
    create_noise_level_summary_panel,
    create_noise_level_summary_panel_au_eu_only,
    create_noise_level_summary_panel_corr_only,
    create_sample_size_summary_panel,
    create_sample_size_summary_panel_au_eu_only,
    create_sample_size_summary_panel_corr_only,
)

load_sample_size_stats = _pss["load_sample_size_stats"]
load_noise_level_stats = _pnl["load_noise_level_stats"]


def main() -> None:
    default_mm = project_root / "results" / "entropy_recomputed_moment_matched_batch" / "statistics"
    parser = argparse.ArgumentParser(
        description=(
            "Consolidated summaries: variance from CSV; entropy only from moment-matched Excel. "
            "Flat sample_size/ and noise_level/ folders."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "results" / "summary_panels_consolidated",
        help="Output root (contains sample_size/ and noise_level/ only)",
    )
    parser.add_argument(
        "--moment-matched-stats",
        type=Path,
        default=default_mm,
        help="Root folder with moment-matched batch statistics/ tree",
    )
    parser.add_argument(
        "--noise-distribution",
        default="normal",
        help="Noise-level moment-matched workbook suffix (e.g. normal, laplace)",
    )
    args = parser.parse_args()
    out_root: Path = args.out.resolve()
    mm_root: Path = args.moment_matched_stats.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    sub_ss = out_root / "sample_size"
    sub_nl = out_root / "noise_level"
    sub_ss.mkdir(parents=True, exist_ok=True)
    sub_nl.mkdir(parents=True, exist_ok=True)

    ylim = NORMALIZED_UNCERTAINTY_YLIM
    dist = args.noise_distribution

    for func_type in ("linear", "sin"):
        for noise_type in ("homoscedastic", "heteroscedastic"):
            stats_var = load_sample_size_stats(
                func_type,  # type: ignore[arg-type]
                noise_type,  # type: ignore[arg-type]
                "variance",  # type: ignore[arg-type]
            )
            if stats_var:
                for fig_fn, creator, extra_kw in [
                    (
                        f"sample_size_summary_4x2_{func_type}_{noise_type}_variance.png",
                        create_sample_size_summary_panel,
                        {"uncertainty_ylim": ylim},
                    ),
                    (
                        f"sample_size_summary_1x4_au_eu_{func_type}_{noise_type}_variance.png",
                        create_sample_size_summary_panel_au_eu_only,
                        {"uncertainty_ylim": ylim},
                    ),
                    (
                        f"sample_size_summary_1x4_corr_{func_type}_{noise_type}_variance.png",
                        create_sample_size_summary_panel_corr_only,
                        {},
                    ),
                ]:
                    fig = creator(
                        stats_var,
                        func_type,  # type: ignore[arg-type]
                        noise_type,  # type: ignore[arg-type]
                        "variance",  # type: ignore[arg-type]
                        **extra_kw,
                    )
                    p = sub_ss / fig_fn
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print("Saved", p, "(variance, CSV)")
            else:
                print(f"  [skip] sample_size variance {func_type}/{noise_type}: no CSV stats")

            stats_ent_mm = load_sample_size_entropy_moment_matched(mm_root, func_type, noise_type)
            if not stats_ent_mm:
                print(
                    f"  [skip] sample_size entropy {func_type}/{noise_type}: "
                    "no moment-matched workbooks (entropy requires moment-matched stats)"
                )
                continue

            for fig_fn, creator, extra_kw in [
                (
                    f"sample_size_summary_4x2_{func_type}_{noise_type}_entropy.png",
                    create_sample_size_summary_panel,
                    {"uncertainty_ylim": ylim},
                ),
                (
                    f"sample_size_summary_1x4_au_eu_{func_type}_{noise_type}_entropy.png",
                    create_sample_size_summary_panel_au_eu_only,
                    {"uncertainty_ylim": ylim},
                ),
                (
                    f"sample_size_summary_1x4_corr_{func_type}_{noise_type}_entropy.png",
                    create_sample_size_summary_panel_corr_only,
                    {},
                ),
            ]:
                fig = creator(
                    stats_ent_mm,
                    func_type,  # type: ignore[arg-type]
                    noise_type,  # type: ignore[arg-type]
                    "entropy",  # type: ignore[arg-type]
                    **extra_kw,
                )
                p = sub_ss / fig_fn
                fig.savefig(p, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print("Saved", p, "(entropy, moment_matched_xlsx)")

    for func_type in ("linear", "sin"):
        for noise_type in ("homoscedastic", "heteroscedastic"):
            stats_var = load_noise_level_stats(
                func_type,  # type: ignore[arg-type]
                noise_type,  # type: ignore[arg-type]
                "variance",  # type: ignore[arg-type]
                distribution="normal",
            )
            if stats_var:
                for fig_fn, creator, extra_kw in [
                    (
                        f"noise_level_summary_4x2_{func_type}_{noise_type}_variance.png",
                        create_noise_level_summary_panel,
                        {"uncertainty_ylim": ylim},
                    ),
                    (
                        f"noise_level_summary_1x4_au_eu_{func_type}_{noise_type}_variance.png",
                        create_noise_level_summary_panel_au_eu_only,
                        {"uncertainty_ylim": ylim},
                    ),
                    (
                        f"noise_level_summary_1x4_corr_{func_type}_{noise_type}_variance.png",
                        create_noise_level_summary_panel_corr_only,
                        {},
                    ),
                ]:
                    fig = creator(
                        stats_var,
                        func_type,  # type: ignore[arg-type]
                        noise_type,  # type: ignore[arg-type]
                        "variance",  # type: ignore[arg-type]
                        **extra_kw,
                    )
                    p = sub_nl / fig_fn
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print("Saved", p, "(variance, CSV)")
            else:
                print(f"  [skip] noise_level variance {func_type}/{noise_type}: no CSV stats")

            stats_ent_mm = load_noise_level_entropy_moment_matched(
                mm_root, func_type, noise_type, distribution=dist
            )
            if not stats_ent_mm:
                print(
                    f"  [skip] noise_level entropy {func_type}/{noise_type}: "
                    "no moment-matched workbooks (entropy requires moment-matched stats)"
                )
                continue

            for fig_fn, creator, extra_kw in [
                (
                    f"noise_level_summary_4x2_{func_type}_{noise_type}_entropy.png",
                    create_noise_level_summary_panel,
                    {"uncertainty_ylim": ylim},
                ),
                (
                    f"noise_level_summary_1x4_au_eu_{func_type}_{noise_type}_entropy.png",
                    create_noise_level_summary_panel_au_eu_only,
                    {"uncertainty_ylim": ylim},
                ),
                (
                    f"noise_level_summary_1x4_corr_{func_type}_{noise_type}_entropy.png",
                    create_noise_level_summary_panel_corr_only,
                    {},
                ),
            ]:
                fig = creator(
                    stats_ent_mm,
                    func_type,  # type: ignore[arg-type]
                    noise_type,  # type: ignore[arg-type]
                    "entropy",  # type: ignore[arg-type]
                    **extra_kw,
                )
                p = sub_nl / fig_fn
                fig.savefig(p, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print("Saved", p, "(entropy, moment_matched_xlsx)")

    print("Done. Output root:", out_root)
    print("Moment-matched stats root:", mm_root)


if __name__ == "__main__":
    main()
