#!/usr/bin/env python3
"""
Line plots of mean spatial AU_norm / EU_norm vs label-noise rate eta from
classification label-noise *_outputs.npz.

Same IT/GL uncertainty definitions and **fixed AU/EU colors** as the sample-size
script.  Writes **two PNGs per model**: IT (entropy-based) and GL (variance-based).

BNN typically uses fewer eta values (default 0.0 and 0.6); use
``--bnn-eta-values`` vs ``--eta-values`` for other models.

Example:
  python scripts/plot_classification_label_noise_au_eu_lines_from_npz.py \\
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
    / "label_noise"
    / "outputs"
    / "classification"
    / "label_noise"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classification"
    / "label_noise"
    / "plots"
    / "classification"
    / "label_noise"
    / "au_eu_lines_from_npz"
)


def _npz_path(outputs_root: Path, variant: str, eta: float) -> Path:
    # Matches experiment_name=f"eta_{eta}" in classification_experiments.py
    return outputs_root / variant / f"{variant}_eta_{eta}_outputs.npz"


def _mean_norm_from_it(path: Path) -> tuple[float, float]:
    d = np.load(path)
    u = it_uncertainty(d["probs_members"])
    return float(np.mean(u["AU_norm"])), float(np.mean(u["EU_norm"]))


def _mean_norm_from_gl(path: Path, gl_samples: int, rng: np.random.Generator) -> tuple[float, float]:
    d = np.load(path)
    u = gl_uncertainty(d["mu_members"], d["sigma2_members"], n_samples=gl_samples, rng=rng)
    return float(np.mean(u["AU_norm"])), float(np.mean(u["EU_norm"]))


def _etas_for_model(model: str, eta_values: list[float], bnn_eta_values: list[float]) -> list[float]:
    if model == "bnn":
        return sorted(float(x) for x in bnn_eta_values)
    return sorted(float(x) for x in eta_values)


def plot_label_noise_lines(
    model: str,
    etas: list[float],
    outputs_root: Path,
    out_dir: Path,
    gl_samples: int,
    seed: int,
    dpi: int,
) -> list[Path]:
    rng = np.random.default_rng(seed)
    it_key = f"{model}_it"
    gl_key = f"{model}_gl"

    xs_it, au_it, eu_it = [], [], []
    xs_gl, au_gl, eu_gl = [], [], []

    for eta in etas:
        p_it = _npz_path(outputs_root, it_key, eta)
        if p_it.is_file():
            a, e = _mean_norm_from_it(p_it)
            xs_it.append(eta)
            au_it.append(a)
            eu_it.append(e)
        p_gl = _npz_path(outputs_root, gl_key, eta)
        if p_gl.is_file():
            a, e = _mean_norm_from_gl(p_gl, gl_samples=gl_samples, rng=rng)
            xs_gl.append(eta)
            au_gl.append(a)
            eu_gl.append(e)

    if not xs_it and not xs_gl:
        print(f"WARNING: no npz for model={model}; skipping.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    display = model.replace("_", " ").title()
    written: list[Path] = []

    if xs_it:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(xs_it, au_it, "o-", color=COLOR_AU, lw=2, ms=7, label="AU")
        ax.plot(xs_it, eu_it, "o-", color=COLOR_EU, lw=2, ms=7, label="EU")
        ax.set_xlabel(r"Label noise rate $\eta$")
        ax.set_ylabel("Mean normalized uncertainty on grid [0, 1]")
        ax.set_title(f"{display} — IT (entropy-based) AU / EU vs $\\eta$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p_it_out = out_dir / f"{model}_label_noise_au_eu_lines_it.png"
        fig.savefig(p_it_out, bbox_inches="tight")
        plt.close(fig)
        written.append(p_it_out)
        print(f"Wrote {p_it_out}")

    if xs_gl:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(xs_gl, au_gl, "o-", color=COLOR_AU, lw=2, ms=7, label="AU")
        ax.plot(xs_gl, eu_gl, "o-", color=COLOR_EU, lw=2, ms=7, label="EU")
        ax.set_xlabel(r"Label noise rate $\eta$")
        ax.set_ylabel("Mean normalized uncertainty on grid [0, 1]")
        ax.set_title(f"{display} — GL (variance-based) AU / EU vs $\\eta$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p_gl_out = out_dir / f"{model}_label_noise_au_eu_lines_gl.png"
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
        "--eta-values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.6],
        help="Eta sweep for non-BNN models (matches Classification_Label_Noise.ipynb)",
    )
    p.add_argument(
        "--bnn-eta-values",
        type=float,
        nargs="+",
        default=[0.0, 0.6],
        help="Eta sweep for BNN (matches eta_values_bnn in notebook)",
    )
    p.add_argument("--gl-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    for m in args.models:
        etas = _etas_for_model(m, list(args.eta_values), list(args.bnn_eta_values))
        plot_label_noise_lines(
            m,
            etas,
            args.outputs_root,
            args.out_dir,
            gl_samples=args.gl_samples,
            seed=args.seed,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
