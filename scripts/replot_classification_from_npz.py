#!/usr/bin/env python3
"""
Replot classification heatmap panels and sweep line plots from saved *_outputs.npz files.

Example (MC Dropout GL RCD outputs):
  python scripts/replot_classification_from_npz.py \\
    --inputs results/classification/rcd/outputs/classification/rcd/mc_dropout_gl \\
    --out results/classification/rcd/replots/mc_dropout_gl \\
    --gl-samples 100

Notes:
  - Training point coordinates (X_train) are not stored in npz; heatmaps are reproduced
    without scatter overlays unless you use --resimulate (requires matching base_cfg).
  - For sweep curves, the script parses rcd / eta / sample_size from filenames like
    *_rcd_3_outputs.npz, *_eta_0.1_outputs.npz, *_sample_size_500_outputs.npz.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.classification_plotting as classification_plotting

classification_plotting.plt.show = lambda *args, **kwargs: None  # batch mode

import utils.results_save as results_save_module
from utils.classification_data import simulate_dataset
from utils.classification_experiments import (
    _predictive_probs_gl,
    _predictive_probs_it,
    gl_uncertainty,
    it_uncertainty,
)
from utils.classification_plotting import plot_metric_curves, plot_uncertainty_panel


def _noop_show(*_a, **_k):
    return None


plt.show = _noop_show


STEM_SWEEP_RE = re.compile(
    r"^(?P<model>.+)_(?P<key>rcd|eta|sample_size|rho)_(?P<val>[\d.]+)$",
    re.IGNORECASE,
)


def parse_sweep_from_stem(stem: str) -> Optional[Tuple[str, float]]:
    """Return (x_axis_label, numeric_value) if stem matches *_{rcd|eta|sample_size|rho}_* ."""
    m = STEM_SWEEP_RE.match(stem)
    if not m:
        return None
    key = m.group("key").lower()
    val = float(m.group("val"))
    # Match notebook / _save_sweep_summary x-axis names
    if key == "sample_size":
        return "N_train", val
    return key, val


def parse_model_and_experiment(stem: str) -> Tuple[str, str]:
    """mc_dropout_gl_rcd_3 -> (mc_dropout_gl, rcd_3)."""
    m = STEM_SWEEP_RE.match(stem)
    if m:
        return m.group("model"), f"{m.group('key').lower()}_{m.group('val')}"
    return stem, stem


def infer_grid_res(n_points: int) -> int:
    r = int(round(np.sqrt(n_points)))
    if r * r != n_points:
        raise ValueError(
            f"x_eval has {n_points} points; expected a square grid (n = r^2). "
            "Pass --grid-res explicitly if needed."
        )
    return r


def extent_from_x_eval(x_eval: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(x_eval[:, 0].min()),
        float(x_eval[:, 0].max()),
        float(x_eval[:, 1].min()),
        float(x_eval[:, 1].max()),
    )


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def is_gl(d: Dict[str, np.ndarray]) -> bool:
    return "mu_members" in d and "sigma2_members" in d


def is_it(d: Dict[str, np.ndarray]) -> bool:
    return "probs_members" in d


def experiment_stem(path: Path) -> str:
    """mc_dropout_gl_rcd_3_outputs.npz -> mc_dropout_gl_rcd_3"""
    name = path.name
    if name.endswith("_outputs.npz"):
        return name[: -len("_outputs.npz")]
    return path.stem


def replot_single(
    npz_path: Path,
    out_plots: Path,
    gl_samples: int,
    model_name: Optional[str],
    grid_res: Optional[int],
    resimulate_cfg: Optional[Dict[str, Any]],
) -> None:
    d = load_npz(npz_path)
    stem = experiment_stem(npz_path)
    inferred_model, inferred_exp = parse_model_and_experiment(stem)
    mname = model_name or inferred_model
    experiment_name = inferred_exp if inferred_exp != stem else stem

    x_eval = np.asarray(d["x_eval"], dtype=np.float32)
    n = x_eval.shape[0]
    gres = grid_res or infer_grid_res(n)
    extent = extent_from_x_eval(x_eval)

    X_train = None
    y_train = None
    if resimulate_cfg is not None:
        cfg = dict(resimulate_cfg)
        if "rcd" in experiment_name and "rcd_" in stem:
            m = re.search(r"rcd_([\d.]+)", stem)
            if m:
                cfg["rcd"] = float(m.group(1))
        if "eta" in stem:
            m = re.search(r"eta_([\d.]+)", stem)
            if m:
                cfg["eta"] = float(m.group(1))
        if "sample_size" in stem:
            m = re.search(r"sample_size_(\d+)", stem)
            if m:
                cfg["N_train"] = int(m.group(1))
        cfg.setdefault("seed", cfg.get("seed", 42))
        X_train, y_train, _, _, _ = simulate_dataset(cfg)

    results_save_module.plots_dir = out_plots
    subfolder = "panels"

    if is_gl(d):
        mu = np.asarray(d["mu_members"])
        sig = np.asarray(d["sigma2_members"])
        unc = gl_uncertainty(mu, sig, n_samples=gl_samples)
        probs_grid = _predictive_probs_gl(mu, sig, n_samples=gl_samples)
        y_grid = np.full(x_eval.shape[0], -1, dtype=np.int64)
        plot_uncertainty_panel(
            X_eval=x_eval,
            uncertainty=unc,
            X_test=x_eval,
            y_true=y_grid,
            probs_pred=probs_grid,
            model_name=mname,
            experiment_name=experiment_name,
            subfolder=subfolder,
            is_it=False,
            X_train=X_train,
            y_train=y_train,
            grid_extent=extent,
            grid_res=gres,
        )
    elif is_it(d):
        probs_m = np.asarray(d["probs_members"])
        unc = it_uncertainty(probs_m)
        probs_grid = _predictive_probs_it(probs_m)
        y_grid = np.full(x_eval.shape[0], -1, dtype=np.int64)
        plot_uncertainty_panel(
            X_eval=x_eval,
            uncertainty=unc,
            X_test=x_eval,
            y_true=y_grid,
            probs_pred=probs_grid,
            model_name=mname,
            experiment_name=experiment_name,
            subfolder=subfolder,
            is_it=True,
            X_train=X_train,
            y_train=y_train,
            grid_extent=extent,
            grid_res=gres,
        )
    else:
        raise ValueError(f"Unrecognized npz layout (no probs_members or mu/sigma): {npz_path}")


def collect_npz_paths(inputs: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(p.glob("*.npz")))
        elif p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(Path().glob(str(p))))
    out = [x for x in paths if x.name.endswith("_outputs.npz")]
    # Stable unique by resolved path (avoids duplicate Windows paths)
    seen = set()
    unique: List[Path] = []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return sorted(unique, key=lambda x: x.name)


def replot_sweep(
    paths: List[Path],
    out_plots: Path,
    gl_samples: int,
    condition_label: Optional[str],
    model_name: Optional[str],
) -> None:
    entries: List[Tuple[float, Dict[str, np.ndarray], str]] = []
    for p in paths:
        stem = experiment_stem(p)
        parsed = parse_sweep_from_stem(stem)
        if parsed is None:
            continue
        _, xval = parsed
        entries.append((xval, load_npz(p), stem))

    if not entries:
        print("No files matched sweep pattern *_rcd_* / *_eta_* / *_sample_size_* / *_rho_*; skip sweep plots.")
        return

    entries.sort(key=lambda t: t[0])
    x_values = [t[0] for t in entries]

    first_stem = entries[0][2]
    inferred_model, _ = parse_model_and_experiment(first_stem)
    mname = model_name or inferred_model
    cond_key = condition_label or parse_sweep_from_stem(first_stem)[0]

    acc, ece = [], []
    au_eu_corr = []
    if is_gl(entries[0][1]):
        au_n, eu_n = [], []
        for _, d, _ in entries:
            unc = gl_uncertainty(
                np.asarray(d["mu_members"]),
                np.asarray(d["sigma2_members"]),
                n_samples=gl_samples,
            )
            au_n.append(float(unc["AU_norm"].mean()))
            eu_n.append(float(unc["EU_norm"].mean()))
            c = np.corrcoef(unc["AU_norm"], unc["EU_norm"])[0, 1]
            au_eu_corr.append(float(c) if np.isfinite(c) else float("nan"))
            met = np.asarray(d["metrics"], dtype=np.float64).ravel()
            acc.append(float(met[0]))
            ece.append(float(met[1]))
        metrics_unc = {"AU_GL_norm": au_n, "EU_GL_norm": eu_n}
        t_unc = f"{mname} GL uncertainty (normalized) vs {cond_key} (replot)"
    elif is_it(entries[0][1]):
        tu_n, au_n, eu_n = [], [], []
        for _, d, _ in entries:
            unc = it_uncertainty(np.asarray(d["probs_members"]))
            tu_n.append(float(unc["TU_norm"].mean()))
            au_n.append(float(unc["AU_norm"].mean()))
            eu_n.append(float(unc["EU_norm"].mean()))
            c = np.corrcoef(unc["AU_norm"], unc["EU_norm"])[0, 1]
            au_eu_corr.append(float(c) if np.isfinite(c) else float("nan"))
            met = np.asarray(d["metrics"], dtype=np.float64).ravel()
            acc.append(float(met[0]))
            ece.append(float(met[1]))
        metrics_unc = {"TU_norm": tu_n, "AU_norm": au_n, "EU_norm": eu_n}
        t_unc = f"{mname} IT uncertainty (normalized) vs {cond_key} (replot)"
    else:
        raise ValueError("First file is neither IT nor GL layout.")

    results_save_module.plots_dir = out_plots
    sub = "sweep"
    plot_metric_curves(x_values, metrics_unc, t_unc, cond_key, subfolder=sub)
    plot_metric_curves(
        x_values,
        {"Accuracy": acc, "ECE": ece},
        f"{mname} metrics vs {cond_key} (replot)",
        cond_key,
        subfolder=sub,
    )
    plot_metric_curves(
        x_values,
        {"AU_EU_corr": au_eu_corr},
        f"{mname} AU-EU correlation vs {cond_key} (replot)",
        cond_key,
        subfolder=sub,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Npz file(s) and/or directories containing *_outputs.npz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (plots saved under <out>/plots/...)",
    )
    ap.add_argument("--gl-samples", type=int, default=100, help="gl_uncertainty / predictive GL sampling (default 100)")
    ap.add_argument("--grid-res", type=int, default=None, help="Override sqrt(N) grid resolution")
    ap.add_argument("--model-name", type=str, default=None, help="Override model name in titles")
    ap.add_argument(
        "--no-panels",
        action="store_true",
        help="Skip per-file heatmap panels",
    )
    ap.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip aggregated sweep line plots",
    )
    ap.add_argument(
        "--resimulate",
        type=Path,
        default=None,
        help="Optional JSON file with simulate_dataset() config to recover X_train for overlays",
    )
    args = ap.parse_args()

    paths = collect_npz_paths(args.inputs)
    if not paths:
        print("No *_outputs.npz files found.", file=sys.stderr)
        sys.exit(1)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    resim_cfg = None
    if args.resimulate is not None:
        with open(args.resimulate, encoding="utf-8") as f:
            resim_cfg = json.load(f)

    if not args.no_panels:
        for p in paths:
            print(f"Panel: {p.name}")
            replot_single(
                p,
                plots_dir,
                gl_samples=args.gl_samples,
                model_name=args.model_name,
                grid_res=args.grid_res,
                resimulate_cfg=resim_cfg,
            )

    if not args.no_sweep:
        replot_sweep(
            paths,
            plots_dir,
            gl_samples=args.gl_samples,
            condition_label=None,
            model_name=args.model_name,
        )

    print(f"Done. Plots under: {plots_dir}")


if __name__ == "__main__":
    main()
