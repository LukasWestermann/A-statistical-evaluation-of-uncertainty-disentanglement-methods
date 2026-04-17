#!/usr/bin/env python3
"""
Line plots of mean spatial AU_norm / EU_norm vs N_train from classification
sample-size *_outputs.npz.

Uses the same uncertainty definitions as the experiment heatmaps
(``it_uncertainty`` / ``gl_uncertainty``).  **AU and EU use fixed colors**
within each track.  Writes **two PNGs per model**: IT (entropy-based) and
GL (variance-based), not a single combined figure.

BNN typically only has npz for a subset of N (default 100 and 1000); use
``--bnn-sample-sizes`` for that while ``--sample-sizes`` applies to other models.

Example:
  python scripts/plot_classification_sample_size_au_eu_lines_from_npz.py \\
    --models mc_dropout deep_ensemble bnn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.classification_experiments import gl_uncertainty, it_uncertainty  # noqa: E402

COLOR_AU = "#1f77b4"
COLOR_EU = "#ff7f0e"

DEFAULT_OUTPUTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "sample_size"
    / "outputs"
    / "classification"
    / "sample_size"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "sample_size"
    / "plots"
    / "classification"
    / "sample_size"
    / "au_eu_lines_from_npz"
)


def _npz_path(outputs_root: Path, variant: str, n_train: int) -> Path:
    return outputs_root / variant / f"{variant}_sample_size_{n_train}_outputs.npz"


def _mean_norm_from_it(path: Path) -> tuple[float, float]:
    d = np.load(path)
    u = it_uncertainty(d["probs_members"])
    return float(np.mean(u["AU_norm"])), float(np.mean(u["EU_norm"]))


def _mean_norm_from_gl(path: Path, gl_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    d = np.load(path)
    u = gl_uncertainty(d["mu_members"], d["sigma2_members"], n_samples=gl_samples, rng=rng)
    return float(np.mean(u["AU_norm"])), float(np.mean(u["EU_norm"]))


def _sizes_for_model(model: str, sample_sizes: list[int], bnn_sample_sizes: list[int]) -> list[int]:
    if model == "bnn":
        return sorted(bnn_sample_sizes)
    return sorted(sample_sizes)


def plot_sample_size_lines(
    model: str,
    sizes: list[int],
    outputs_root: Path,
    out_dir: Path,
    gl_samples: int,
    seed: int,
    dpi: int,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    it_key = f"{model}_it"
    gl_key = f"{model}_gl"

    ns_it, au_it, eu_it = [], [], []
    ns_gl, au_gl, eu_gl = [], [], []

    for n in sizes:
        p_it = _npz_path(outputs_root, it_key, n)
        if p_it.is_file():
            a, e = _mean_norm_from_it(p_it)
            ns_it.append(n)
            au_it.append(a)
            eu_it.append(e)
        p_gl = _npz_path(outputs_root, gl_key, n)
        if p_gl.is_file():
            a, e = _mean_norm_from_gl(p_gl, gl_samples=gl_samples, rng=rng)
            ns_gl.append(n)
            au_gl.append(a)
            eu_gl.append(e)

    if not ns_it and not ns_gl:
        print(f"WARNING: no npz for model={model}; skipping.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    display = model.replace("_", " ").title()
    written: list[Path] = []

    if ns_it:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(ns_it, au_it, "o-", color=COLOR_AU, lw=2, ms=7, label="AU")
        ax.plot(ns_it, eu_it, "o-", color=COLOR_EU, lw=2, ms=7, label="EU")
        ax.set_xlabel(r"$N_{\mathrm{train}}$")
        ax.set_ylabel("Mean normalized uncertainty on grid [0, 1]")
        ax.set_title(f"{display} — IT (entropy-based) AU / EU vs sample size")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p_it_out = out_dir / f"{model}_sample_size_au_eu_lines_it.png"
        fig.savefig(p_it_out, bbox_inches="tight")
        plt.close(fig)
        written.append(p_it_out)
        print(f"Wrote {p_it_out}")

    if ns_gl:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(ns_gl, au_gl, "o-", color=COLOR_AU, lw=2, ms=7, label="AU")
        ax.plot(ns_gl, eu_gl, "o-", color=COLOR_EU, lw=2, ms=7, label="EU")
        ax.set_xlabel(r"$N_{\mathrm{train}}$")
        ax.set_ylabel("Mean normalized uncertainty on grid [0, 1]")
        ax.set_title(f"{display} — GL (variance-based) AU / EU vs sample size")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p_gl_out = out_dir / f"{model}_sample_size_au_eu_lines_gl.png"
        fig.savefig(p_gl_out, bbox_inches="tight")
        plt.close(fig)
        written.append(p_gl_out)
        print(f"Wrote {p_gl_out}")

    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--models", nargs="+", default=["mc_dropout", "deep_ensemble", "bnn"])
    p.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="N_train list for non-BNN models",
    )
    p.add_argument(
        "--bnn-sample-sizes",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="N_train list for BNN (matches Classification_Sample_Size.ipynb)",
    )
    p.add_argument("--gl-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    for m in args.models:
        sizes = _sizes_for_model(m, list(args.sample_sizes), list(args.bnn_sample_sizes))
        plot_sample_size_lines(
            m,
            sizes,
            args.outputs_root,
            args.out_dir,
            gl_samples=args.gl_samples,
            seed=args.seed,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
