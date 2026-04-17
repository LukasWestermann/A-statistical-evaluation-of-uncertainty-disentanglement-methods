"""
Two 4×4 baseline figures (variance-only and entropy-only) from pct100 ``*raw_outputs*.npz``.

Loads the latest pct100 files per model and condition from the sample-size experiment and uses
``compute_moment_matched_grid_result(..., ood_ranges=[])`` — same entropy definition as
``recompute_entropy_moment_matched_batch_from_npz.py`` for the sample_size experiment.

Usage (from project root):

    python scripts/plot_baseline_pct100_variance_entropy_overview.py
    python scripts/plot_baseline_pct100_variance_entropy_overview.py \\
        --out-variance results/thesis_figures/baseline_pct100_variance_4x4.png \\
        --out-entropy results/thesis_figures/baseline_pct100_entropy_4x4.png
    python scripts/plot_baseline_pct100_variance_entropy_overview.py --strict

With ``--strict``, exit with code 1 if any of the 16 NPZs is missing. Default is to warn and leave
that subplot blank.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (
    CONDITIONS_4,
    MODEL_RESOLVERS,
    compute_moment_matched_grid_result,
    function_display,
    resolve_latest_npz_at_pct,
)


def _noise_display(noise_type: str) -> str:
    return "homoscedastic" if noise_type == "homoscedastic" else "heteroscedastic"


def _plot_variance_panel(ax, res) -> None:
    x = res.x
    mu = np.asarray(res.mu_pred).ravel()
    ale_var = np.asarray(res.ale_var).ravel()
    epi_var = np.asarray(res.epi_var).ravel()
    tot_var = ale_var + epi_var
    sig_tot = np.sqrt(np.maximum(tot_var, 0.0))
    y_clean = np.asarray(res.y_clean_flat).ravel()

    if res.x_train_flat is not None and res.y_train_flat is not None:
        ax.scatter(
            res.x_train_flat,
            res.y_train_flat,
            alpha=0.12,
            s=8,
            color="C0",
            zorder=2,
        )

    ax.fill_between(
        x,
        mu - sig_tot,
        mu + sig_tot,
        alpha=0.28,
        color="C0",
        label="±σ(total)",
        zorder=1,
    )
    ax.plot(x, mu, color="C0", linewidth=1.4, label="Predictive mean", zorder=3)
    ax.plot(x, y_clean, color="C3", linestyle="--", linewidth=1.0, alpha=0.85, label="Clean function")

    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


def _plot_entropy_panel(ax, res) -> None:
    x = res.x
    ale_e = np.asarray(res.ale_entropy).ravel()
    epi_e = np.asarray(res.epi_entropy).ravel()
    tot_e = np.asarray(res.tot_entropy).ravel()

    if res.x_train_flat is not None and res.y_train_flat is not None:
        ax.scatter(
            res.x_train_flat,
            res.y_train_flat,
            alpha=0.12,
            s=8,
            color="gray",
            zorder=1,
            label="Training data",
        )

    ax.plot(x, ale_e, color="tab:green", linewidth=1.2, label="Aleatoric entropy")
    ax.plot(x, epi_e, color="tab:red", linewidth=1.2, label="Epistemic entropy")
    ax.plot(x, tot_e, color="tab:blue", linewidth=1.4, label="Total entropy")
    ax.set_ylabel("Entropy (nats)")
    ax.grid(True, alpha=0.3)


def _make_figure(
    n_models: int,
    n_conds: int,
) -> Tuple[Any, Any]:
    fig, axes = plt.subplots(
        n_models,
        n_conds,
        figsize=(18, 16),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    return fig, axes


def main() -> int:
    p = argparse.ArgumentParser(description="Two 4×4 baselines: variance-only and entropy-only.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root (default: parent of scripts/).",
    )
    p.add_argument("--pct", type=float, default=100.0, help="Training percentage in filenames (default: 100).")
    p.add_argument(
        "--out-variance",
        type=Path,
        default=None,
        help="Variance figure PNG (default: results/thesis_figures/baseline_pct100_variance_4x4.png).",
    )
    p.add_argument(
        "--out-entropy",
        type=Path,
        default=None,
        help="Entropy figure PNG (default: results/thesis_figures/baseline_pct100_entropy_4x4.png).",
    )
    p.add_argument("--grid-stride", type=int, default=1)
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected NPZ is missing.",
    )
    args = p.parse_args()

    root = args.project_root.resolve()
    out_v = args.out_variance
    out_e = args.out_entropy
    if out_v is None:
        out_v = root / "results" / "thesis_figures" / "baseline_pct100_variance_4x4.png"
    if out_e is None:
        out_e = root / "results" / "thesis_figures" / "baseline_pct100_entropy_4x4.png"
    out_v = out_v.resolve()
    out_e = out_e.resolve()
    out_v.parent.mkdir(parents=True, exist_ok=True)
    out_e.parent.mkdir(parents=True, exist_ok=True)

    base = root / "results" / "sample_size" / "outputs" / "sample_size"

    n_models = len(MODEL_RESOLVERS)
    n_conds = len(CONDITIONS_4)
    fig_v, axes_v = _make_figure(n_models, n_conds)
    fig_e, axes_e = _make_figure(n_models, n_conds)

    missing: list[str] = []
    ref_ax_v: Optional[Any] = None
    ref_ax_e: Optional[Any] = None

    for mi, (model_disp, model_tag, _) in enumerate(MODEL_RESOLVERS):
        for ci, (func_type, noise_type) in enumerate(CONDITIONS_4):
            ax_v = axes_v[mi, ci]
            ax_e = axes_e[mi, ci]
            search_dir = base / noise_type / func_type
            npz_path = resolve_latest_npz_at_pct(search_dir, model_tag, args.pct)
            if npz_path is None:
                msg = f"No pct{args.pct:g} raw_outputs for {model_tag} under {search_dir}"
                missing.append(msg)
                ax_v.set_axis_off()
                ax_e.set_axis_off()
                ax_v.text(0.5, 0.5, "Missing data", ha="center", va="center", transform=ax_v.transAxes, fontsize=10)
                ax_e.text(0.5, 0.5, "Missing data", ha="center", va="center", transform=ax_e.transAxes, fontsize=10)
                continue
            try:
                res = compute_moment_matched_grid_result(npz_path, [], args.grid_stride, args.eps)
            except Exception as e:
                msg = f"{npz_path}: {e}"
                missing.append(msg)
                warnings.warn(msg)
                ax_v.set_axis_off()
                ax_e.set_axis_off()
                ax_v.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax_v.transAxes, fontsize=10)
                ax_e.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax_e.transAxes, fontsize=10)
                continue

            _plot_variance_panel(ax_v, res)
            _plot_entropy_panel(ax_e, res)
            if ref_ax_v is None:
                ref_ax_v = ax_v
            if ref_ax_e is None:
                ref_ax_e = ax_e

            title = f"{function_display(func_type)} — {_noise_display(noise_type)}"
            if mi == 0:
                ax_v.set_title(title, fontsize=10)
                ax_e.set_title(title, fontsize=10)
            if mi == n_models - 1:
                ax_v.set_xlabel("x")
                ax_e.set_xlabel("x")

        ax_left_v = axes_v[mi, 0]
        ax_left_e = axes_e[mi, 0]
        for ax_left in (ax_left_v, ax_left_e):
            ax_left.annotate(
                model_disp,
                xy=(-0.18, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="medium",
            )

    if ref_ax_v is not None:
        h, lab = ref_ax_v.get_legend_handles_labels()
        fig_v.legend(
            h,
            lab,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize=8,
            frameon=False,
        )
    if ref_ax_e is not None:
        h, lab = ref_ax_e.get_legend_handles_labels()
        fig_e.legend(
            h,
            lab,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize=8,
            frameon=False,
        )

    if missing:
        for m in missing:
            print(f"WARNING: {m}", file=sys.stderr)
        if args.strict:
            print("ERROR: --strict set; aborting.", file=sys.stderr)
            return 1

    fig_v.savefig(out_v, dpi=args.dpi, bbox_inches="tight")
    fig_e.savefig(out_e, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_v}")
    print(f"Saved: {out_e}")
    plt.close(fig_v)
    plt.close(fig_e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
