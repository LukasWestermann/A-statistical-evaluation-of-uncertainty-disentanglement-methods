"""
Build 4×N heatmap panels from saved classification sample-size NPZ outputs.

Rows (default):
  1) GL variance-based EU — sum/mean over classes of var(mu_members, axis=0)
  2) GL variance-based AU — sum/mean over classes of mean(sigma2_members, axis=0)
  3) Entropy EU — it_uncertainty (default) or gl_uncertainty (--entropy-source gl)
  4) Entropy AU — same source as row 3

Normalization matches training panels: min–max per quantity; use --normalize-scope
row-global to pool bounds across sample-size columns within each row.

Expects files:
  {outputs_root}/{model}_it/{model}_it_sample_size_{N}_outputs.npz
  {outputs_root}/{model}_gl/{model}_gl_sample_size_{N}_outputs.npz
(with probs_members / mu_members, sigma2_members, x_eval).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project root (parent of scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.classification_experiments import (  # noqa: E402
    _normalize_uncertainty,
    gl_uncertainty,
    it_uncertainty,
)

ROW_LABELS_VAR_EU = "Var EU (GL logits)"
ROW_LABELS_VAR_AU = "Var AU (GL logits)"
ROW_LABELS_ENT_EU = "Entropy EU"
ROW_LABELS_ENT_AU = "Entropy AU"


def _infer_grid_res(x_eval: np.ndarray) -> int:
    n = x_eval.shape[0]
    r = int(round(n**0.5))
    if r * r != n:
        raise ValueError(
            f"x_eval length {n} is not a square grid (sqrt ~ {r}); "
            "expected N = grid_res**2."
        )
    return r


def _grid_extent(x_eval: np.ndarray) -> tuple[float, float, float, float]:
    x0, x1 = float(x_eval[:, 0].min()), float(x_eval[:, 0].max())
    y0, y1 = float(x_eval[:, 1].min()), float(x_eval[:, 1].max())
    return (x0, x1, y0, y1)


def _gl_variance_scalars(
    mu_members: np.ndarray,
    sigma2_members: np.ndarray,
    agg: str,
) -> tuple[np.ndarray, np.ndarray]:
    sigma2_epi = mu_members.var(axis=0)
    sigma2_ale = sigma2_members.mean(axis=0)
    if agg == "sum":
        v_eu = sigma2_epi.sum(axis=-1)
        v_au = sigma2_ale.sum(axis=-1)
    elif agg == "mean":
        v_eu = sigma2_epi.mean(axis=-1)
        v_au = sigma2_ale.mean(axis=-1)
    else:
        raise ValueError(f"agg must be 'sum' or 'mean', got {agg!r}")
    return v_eu.astype(np.float64), v_au.astype(np.float64)


def _entropy_raw_it(probs_members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = it_uncertainty(probs_members)
    return np.asarray(u["EU"], dtype=np.float64), np.asarray(u["AU"], dtype=np.float64)


def _entropy_raw_gl(
    mu_members: np.ndarray,
    sigma2_members: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    u = gl_uncertainty(
        mu_members, sigma2_members, n_samples=n_samples, rng=rng
    )
    return np.asarray(u["EU"], dtype=np.float64), np.asarray(u["AU"], dtype=np.float64)


def _npz_path(outputs_root: Path, model_variant: str, n_train: int) -> Path:
    # e.g. mc_dropout_it / mc_dropout_it_sample_size_100_outputs.npz
    sub = outputs_root / model_variant
    fname = f"{model_variant}_sample_size_{n_train}_outputs.npz"
    return sub / fname


def _load_gl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["mu_members"], d["sigma2_members"], d["x_eval"]


def _load_it(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["probs_members"], d["x_eval"]


def _apply_row_global_norm(
    raw_per_col: list[np.ndarray],
) -> list[np.ndarray]:
    stacked = np.concatenate(raw_per_col, axis=0)
    vmin, vmax = float(stacked.min()), float(stacked.max())
    return [_normalize_uncertainty(a, vmin=vmin, vmax=vmax) for a in raw_per_col]


def _plot_model_grid(
    model: str,
    sample_sizes: list[int],
    outputs_root: Path,
    out_dir: Path,
    entropy_source: str,
    gl_samples: int,
    var_agg: str,
    normalize_scope: str,
    dpi: int,
    strict: bool,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    valid_ns: list[int] = []
    for n in sample_sizes:
        p_gl = _npz_path(outputs_root, f"{model}_gl", n)
        if not p_gl.is_file():
            msg = f"Missing GL outputs: {p_gl}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"WARNING: {msg} — skipping N={n}")
            continue
        if entropy_source == "it":
            p_it = _npz_path(outputs_root, f"{model}_it", n)
            if not p_it.is_file():
                msg = f"Missing IT outputs (required for entropy-source=it): {p_it}"
                if strict:
                    raise FileNotFoundError(msg)
                print(f"WARNING: {msg} — skipping N={n}")
                continue
        valid_ns.append(n)

    if not valid_ns:
        print(f"No data for model {model!r}; skipping figure.")
        return

    var_eu_cols: list[np.ndarray] = []
    var_au_cols: list[np.ndarray] = []
    ent_eu_cols: list[np.ndarray] = []
    ent_au_cols: list[np.ndarray] = []
    extents: list[tuple[float, float, float, float]] = []
    grid_res: int | None = None

    for n in valid_ns:
        p_gl = _npz_path(outputs_root, f"{model}_gl", n)
        mu, sig2, x_eval = _load_gl(p_gl)
        extents.append(_grid_extent(x_eval))
        g = _infer_grid_res(x_eval)
        if grid_res is None:
            grid_res = g
        elif g != grid_res:
            raise ValueError(f"Grid resolution mismatch at N={n}")

        veu, vau = _gl_variance_scalars(mu, sig2, var_agg)
        var_eu_cols.append(veu.ravel())
        var_au_cols.append(vau.ravel())

        if entropy_source == "it":
            p_it = _npz_path(outputs_root, f"{model}_it", n)
            probs, x_it = _load_it(p_it)
            if x_it.shape != x_eval.shape or not np.allclose(x_it, x_eval):
                print(
                    f"WARNING: x_eval mismatch IT vs GL for N={n}; using GL grid shape only."
                )
            eu, au = _entropy_raw_it(probs)
        else:
            eu, au = _entropy_raw_gl(mu, sig2, n_samples=gl_samples, rng=rng)

        ent_eu_cols.append(eu.ravel())
        ent_au_cols.append(au.ravel())

    # Normalize
    rows_raw = [var_eu_cols, var_au_cols, ent_eu_cols, ent_au_cols]
    if normalize_scope == "cell":
        normed_rows = [
            [_normalize_uncertainty(a) for a in row]
            for row in rows_raw
        ]
    elif normalize_scope == "row-global":
        normed_rows = [_apply_row_global_norm(row) for row in rows_raw]
    else:
        raise ValueError(normalize_scope)

    n_plot = len(var_eu_cols)
    fig, axes = plt.subplots(4, n_plot, figsize=(3.2 * n_plot, 11), squeeze=False)

    row_labels = [
        ROW_LABELS_VAR_EU,
        ROW_LABELS_VAR_AU,
        ROW_LABELS_ENT_EU + f" ({entropy_source.upper()})",
        ROW_LABELS_ENT_AU + f" ({entropy_source.upper()})",
    ]

    assert grid_res is not None

    for j in range(n_plot):
        ex = extents[j]
        for i, row_norm in enumerate(normed_rows):
            grid = row_norm[j].reshape(grid_res, grid_res)
            ax = axes[i, j]
            im = ax.imshow(
                grid,
                extent=[ex[0], ex[1], ex[2], ex[3]],
                origin="lower",
                aspect="equal",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
            if i == 0:
                ax.set_title(f"N = {valid_ns[j]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{model}: sample-size uncertainty grid "
        f"(var agg={var_agg}, norm={normalize_scope})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_sample_size_uncertainty_grid.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    default_root = (
        _PROJECT_ROOT
        / "results"
        / "classification"
        / "sample_size"
        / "outputs"
        / "classification"
        / "sample_size"
    )
    parser = argparse.ArgumentParser(
        description="4×N heatmaps from saved classification sample-size NPZ outputs."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_root,
        help="Directory containing <model>_it/ and <model>_gl/ subfolders",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mc_dropout", "deep_ensemble", "bnn"],
        help="Base model names (without _it/_gl suffix)",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "results"
        / "classification"
        / "sample_size"
        / "plots"
        / "classification"
        / "sample_size"
        / "uncertainty_grids",
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--entropy-source",
        choices=("it", "gl"),
        default="it",
    )
    parser.add_argument("--gl-samples", type=int, default=100)
    parser.add_argument("--var-agg", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--normalize-scope",
        choices=("cell", "row-global"),
        default="cell",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested NPZ is missing",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for model in args.models:
        _plot_model_grid(
            model=model,
            sample_sizes=list(args.sample_sizes),
            outputs_root=args.outputs_root.resolve(),
            out_dir=args.out_dir.resolve(),
            entropy_source=args.entropy_source,
            gl_samples=args.gl_samples,
            var_agg=args.var_agg,
            normalize_scope=args.normalize_scope,
            dpi=args.dpi,
            strict=args.strict,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
