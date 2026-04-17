#!/usr/bin/env python3
"""
Rebuild sample-size uncertainty heatmap panels from saved *_outputs.npz only.

Matches experiment heatmaps (gl_uncertainty / it_uncertainty normalized fields), laid out
as rows × columns (one column per N_train).

Training points for overlays are regenerated with utils.classification_data.simulate_dataset
using the same defaults as Experiments/Classification_Sample_Size.ipynb (no experiment rerun).
Optional --sim-cfg JSON merges on top of those defaults (e.g. different rcd, blob_sigma).

  python scripts/replot_sample_size_panels_from_npz.py \\
    --variant gl --models mc_dropout \\
    --sample-sizes 100 1000

Use --variant both for one figure per model: GL block on top, IT block below (e.g. 4×4 grid
with default four sample sizes and --rows au-eu).

For model bnn, --bnn-sample-sizes applies (default 100 1000) instead of --sample-sizes, matching
the reduced BNN sweep in Classification_Sample_Size.ipynb.

Use --no-overlay to skip point scatter.

GL panels always show AU_norm and EU_norm only. For IT, use --rows au-eu (default) for the same
two rows, or --rows full to include TU_norm as a third row. IT two-row figures are saved as
*_it_au_eu.png so they do not overwrite three-row *_it.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.classification_data import simulate_dataset  # noqa: E402
from utils.classification_experiments import (  # noqa: E402
    _normalize_uncertainty,
    gl_uncertainty,
    it_uncertainty,
)

# Keys consumed by simulate_dataset — aligned with Classification_Sample_Size.ipynb base_cfg.
# rcd omitted here → simulate_dataset default rcd=3.0; add via --sim-cfg if your notebook differed.
DEFAULT_SAMPLE_SIZE_SIMULATE_CFG: dict[str, Any] = {
    "N_test": 500,
    "num_classes": 3,
    "blob_sigma": 0.25,
    "tau": 0.2,
    "eta": 0.0,
    "sigma_in": 0.0,
}

_TRAIN_COLORS = ["tab:blue", "tab:orange", "tab:green"]

# Figure typography
_COL_TITLE_FONTSIZE = 14
_ROW_LABEL_FONTSIZE = 10
_SUPTITLE_FONTSIZE = 15

_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "bnn": "BNN",
}


def _model_display_name(model_id: str) -> str:
    if model_id in _MODEL_DISPLAY_NAMES:
        return _MODEL_DISPLAY_NAMES[model_id]
    return model_id.replace("_", " ").title()


def _suptitle_for_model(model_id: str) -> str:
    return f"{_model_display_name(model_id)} Sample Size experiment"


def _infer_grid_res(x_eval: np.ndarray) -> int:
    n = x_eval.shape[0]
    r = int(round(n**0.5))
    if r * r != n:
        raise ValueError(
            f"x_eval length {n} is not a square grid (sqrt ~ {r}); expected N = grid_res**2."
        )
    return r


def _grid_extent(x_eval: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(x_eval[:, 0].min()),
        float(x_eval[:, 0].max()),
        float(x_eval[:, 1].min()),
        float(x_eval[:, 1].max()),
    )


def _npz_path(outputs_root: Path, model_variant: str, n_train: int) -> Path:
    sub = outputs_root / model_variant
    fname = f"{model_variant}_sample_size_{n_train}_outputs.npz"
    return sub / fname


def _load_gl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["mu_members"], d["sigma2_members"], d["x_eval"]


def _load_it(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["probs_members"], d["x_eval"]


def _apply_row_global_norm(raw_per_col: list[np.ndarray]) -> list[np.ndarray]:
    stacked = np.concatenate(raw_per_col, axis=0)
    vmin, vmax = float(stacked.min()), float(stacked.max())
    return [_normalize_uncertainty(a, vmin=vmin, vmax=vmax) for a in raw_per_col]


def _overlay_train_scatter(ax: plt.Axes, X_train: np.ndarray, y_train: np.ndarray) -> None:
    ymax = int(np.max(y_train))
    for cls in range(ymax + 1):
        mask = y_train == cls
        if not np.any(mask):
            continue
        c = _TRAIN_COLORS[cls % len(_TRAIN_COLORS)]
        ax.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            facecolors="none",
            edgecolors=c,
            s=20,
            linewidths=1.0,
            alpha=0.4,
        )


def _train_xy_per_n(
    data_cfg: dict[str, Any],
    n_list: list[int],
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Regenerate training sets with simulate_dataset (same process as notebooks)."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in n_list:
        cfg = {**data_cfg, "N_train": int(n), "seed": int(seed)}
        X_train, y_train, _, _, _ = simulate_dataset(cfg)
        out[n] = (X_train, y_train)
    return out


def _variant_keys(variant: str, rows: str) -> tuple[list[str], list[str]]:
    if variant == "gl":
        return (["AU", "EU"], ["AU_norm", "EU_norm"])
    if rows == "au-eu":
        return (["AU", "EU"], ["AU_norm", "EU_norm"])
    if rows == "full":
        return (["TU", "AU", "EU"], ["TU_norm", "AU_norm", "EU_norm"])
    raise ValueError(f"rows must be full or au-eu, got {rows!r}")


def _row_display_labels(variant: str, rows: str) -> list[str]:
    """Left-axis labels (GL = variance-based decomposition, IT = entropy-based)."""
    if variant == "gl":
        return ["AU (variance-based)", "EU (variance-based)"]
    if rows == "au-eu":
        return ["AU (entropy-based)", "EU (entropy-based)"]
    if rows == "full":
        return [
            "Total uncertainty (entropy-based)",
            "AU (entropy-based)",
            "EU (entropy-based)",
        ]
    raise ValueError(f"rows must be full or au-eu, got {rows!r}")


def _load_normed_panel_data(
    model: str,
    variant: str,
    ns: list[int],
    outputs_root: Path,
    normalize_scope: str,
    gl_samples: int,
    rng: np.random.Generator,
    rows: str,
) -> tuple[list[list[np.ndarray]], list[str], list[tuple[float, float, float, float]], int] | None:
    """Load and normalize grids for fixed ordered sample sizes ns; None if any npz missing."""
    if not ns:
        return None
    suffix = f"{model}_gl" if variant == "gl" else f"{model}_it"
    for n in ns:
        if not _npz_path(outputs_root, suffix, n).is_file():
            return None

    raw_keys, norm_keys = _variant_keys(variant, rows)
    n_r = len(norm_keys)
    raw_per_row: list[list[np.ndarray]] = [[] for _ in range(n_r)]
    norm_cell_per_row: list[list[np.ndarray]] = [[] for _ in range(n_r)]
    extents: list[tuple[float, float, float, float]] = []
    grid_res: int | None = None

    for n in ns:
        path = _npz_path(outputs_root, suffix, n)
        if variant == "gl":
            mu, sig2, x_eval = _load_gl(path)
            u = gl_uncertainty(mu, sig2, n_samples=gl_samples, rng=rng)
        else:
            probs, x_eval = _load_it(path)
            u = it_uncertainty(probs)

        extents.append(_grid_extent(x_eval))
        g = _infer_grid_res(x_eval)
        if grid_res is None:
            grid_res = g
        elif g != grid_res:
            raise ValueError(f"Grid resolution mismatch at N={n}")

        for i, rk in enumerate(raw_keys):
            raw_per_row[i].append(np.asarray(u[rk], dtype=np.float64).ravel())
        for i, nk in enumerate(norm_keys):
            norm_cell_per_row[i].append(np.asarray(u[nk], dtype=np.float64).ravel())

    if normalize_scope == "cell":
        normed_rows = norm_cell_per_row
    elif normalize_scope == "row-global":
        normed_rows = [_apply_row_global_norm(row) for row in raw_per_row]
    else:
        raise ValueError(f"normalize_scope must be cell or row-global, got {normalize_scope!r}")

    assert grid_res is not None
    row_labels = _row_display_labels(variant, rows)
    return normed_rows, row_labels, extents, grid_res


def _draw_axes_grid(
    axes: np.ndarray,
    normed_rows: list[list[np.ndarray]],
    row_labels: list[str],
    extents: list[tuple[float, float, float, float]],
    grid_res: int,
    valid_ns: list[int],
    train_xy: dict[int, tuple[np.ndarray, np.ndarray]] | None,
    fig: plt.Figure,
    show_column_titles: bool,
) -> None:
    n_rows, n_plot = len(normed_rows), len(valid_ns)
    for j in range(n_plot):
        ex = extents[j]
        n_j = valid_ns[j]
        X_tr = y_tr = None
        if train_xy is not None and n_j in train_xy:
            X_tr, y_tr = train_xy[n_j]

        for i in range(n_rows):
            ax = axes[i, j]
            grid = normed_rows[i][j].reshape(grid_res, grid_res)
            im = ax.imshow(
                grid,
                extent=[ex[0], ex[1], ex[2], ex[3]],
                origin="lower",
                aspect="equal",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            if X_tr is not None and y_tr is not None:
                _overlay_train_scatter(ax, X_tr, y_tr)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=_ROW_LABEL_FONTSIZE)
            if show_column_titles and i == 0:
                ax.set_title(f"N = {n_j}", fontsize=_COL_TITLE_FONTSIZE)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_panels_for_model(
    model: str,
    variant: str,
    sample_sizes: list[int],
    outputs_root: Path,
    out_dir: Path,
    normalize_scope: str,
    gl_samples: int,
    dpi: int,
    strict: bool,
    seed: int,
    data_cfg: dict[str, Any],
    overlay_points: bool,
    rows: str,
) -> None:
    rng = np.random.default_rng(seed)
    suffix = f"{model}_gl" if variant == "gl" else f"{model}_it"

    valid_ns: list[int] = []
    for n in sample_sizes:
        p = _npz_path(outputs_root, suffix, n)
        if not p.is_file():
            msg = f"Missing outputs: {p}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"WARNING: {msg} — skipping N={n}")
            continue
        valid_ns.append(n)

    if not valid_ns:
        print(f"No data for model {model!r} ({variant}); skipping.")
        return

    built = _load_normed_panel_data(
        model, variant, valid_ns, outputs_root, normalize_scope, gl_samples, rng, rows
    )
    assert built is not None
    normed_rows, row_labels, extents, grid_res = built

    train_xy: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
    if overlay_points:
        train_xy = _train_xy_per_n(data_cfg, valid_ns, seed)

    n_rows = len(row_labels)
    n_plot = len(valid_ns)
    fig_h = 4.0 + 3.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_plot, figsize=(3.2 * n_plot, fig_h), squeeze=False)
    _draw_axes_grid(
        axes, normed_rows, row_labels, extents, grid_res, valid_ns, train_xy, fig, True
    )

    fig.suptitle(_suptitle_for_model(model), fontsize=_SUPTITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    it_suffix = "_au_eu" if variant == "it" and rows == "au-eu" else ""
    out_path = out_dir / f"{model}_sample_size_panels_from_npz_{variant}{it_suffix}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_stacked_gl_it_for_model(
    model: str,
    sample_sizes: list[int],
    outputs_root: Path,
    out_dir: Path,
    normalize_scope: str,
    gl_samples: int,
    dpi: int,
    strict: bool,
    seed: int,
    data_cfg: dict[str, Any],
    overlay_points: bool,
    rows: str,
) -> None:
    """Single figure: GL rows on top, IT rows below; columns = common sample sizes."""
    rng = np.random.default_rng(seed)

    if strict:
        for n in sample_sizes:
            pg = _npz_path(outputs_root, f"{model}_gl", n)
            pi = _npz_path(outputs_root, f"{model}_it", n)
            if not pg.is_file():
                raise FileNotFoundError(f"Missing outputs: {pg}")
            if not pi.is_file():
                raise FileNotFoundError(f"Missing outputs: {pi}")
        common_ns = list(sample_sizes)
    else:
        common_ns = []
        for n in sample_sizes:
            pg = _npz_path(outputs_root, f"{model}_gl", n)
            pi = _npz_path(outputs_root, f"{model}_it", n)
            if pg.is_file() and pi.is_file():
                common_ns.append(n)
            else:
                if not pg.is_file():
                    print(f"WARNING: Missing GL outputs: {pg} — dropping N={n} from stacked plot")
                if not pi.is_file():
                    print(f"WARNING: Missing IT outputs: {pi} — dropping N={n} from stacked plot")

    if not common_ns:
        print(f"No paired GL+IT data for model {model!r}; skipping stacked plot.")
        return

    gl_built = _load_normed_panel_data(
        model, "gl", common_ns, outputs_root, normalize_scope, gl_samples, rng, rows
    )
    it_built = _load_normed_panel_data(
        model, "it", common_ns, outputs_root, normalize_scope, gl_samples, rng, rows
    )
    assert gl_built is not None and it_built is not None

    gl_normed, gl_labels, extents, grid_res = gl_built
    it_normed, it_labels, it_extents, it_grid = it_built
    if it_grid != grid_res:
        raise ValueError("GL vs IT grid resolution mismatch in stacked plot.")
    for j, (e1, e2) in enumerate(zip(extents, it_extents)):
        if e1 != e2:
            raise ValueError(f"x_eval extent mismatch GL vs IT at column {j}")

    n_gl, n_it = len(gl_labels), len(it_labels)
    n_plot = len(common_ns)
    total_rows = n_gl + n_it
    fig_h = 4.0 + 3.0 * total_rows
    fig, axes = plt.subplots(total_rows, n_plot, figsize=(3.2 * n_plot, fig_h), squeeze=False)

    train_xy: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
    if overlay_points:
        train_xy = _train_xy_per_n(data_cfg, common_ns, seed)

    _draw_axes_grid(
        axes[:n_gl, :],
        gl_normed,
        gl_labels,
        extents,
        grid_res,
        common_ns,
        train_xy,
        fig,
        show_column_titles=True,
    )
    _draw_axes_grid(
        axes[n_gl:, :],
        it_normed,
        it_labels,
        extents,
        grid_res,
        common_ns,
        train_xy,
        fig,
        show_column_titles=False,
    )

    fig.suptitle(_suptitle_for_model(model), fontsize=_SUPTITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    it_suffix = "_au_eu" if rows == "au-eu" else ""
    out_path = out_dir / f"{model}_sample_size_panels_from_npz_gl_it{it_suffix}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    default_root = (
        PROJECT_ROOT
        / "results"
        / "classification"
        / "sample_size"
        / "outputs"
        / "classification"
        / "sample_size"
    )
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_root,
        help="Root with subfolders <model>_gl/ and <model>_it/",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mc_dropout", "deep_ensemble", "bnn"],
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000],
        help="Column N_train values for mc_dropout, deep_ensemble, etc.",
    )
    parser.add_argument(
        "--bnn-sample-sizes",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="Used only when 'bnn' is in --models (default: 100 1000).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "classification"
        / "sample_size"
        / "plots"
        / "classification"
        / "sample_size"
        / "replots_from_npz",
    )
    parser.add_argument(
        "--variant",
        choices=("gl", "it", "both"),
        required=True,
        help="gl, it, or both (stacked GL on top + IT below, one PNG per model).",
    )
    parser.add_argument(
        "--rows",
        choices=("full", "au-eu"),
        default="au-eu",
        help="IT: full = TU+AU+EU rows; au-eu = AU+EU only (GL is always AU+EU). Default: au-eu.",
    )
    parser.add_argument(
        "--normalize-scope",
        choices=("cell", "row-global"),
        default="cell",
        help="cell: per-N *_norm from utils (matches notebooks). row-global: pool raw across N.",
    )
    parser.add_argument("--gl-samples", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed passed to simulate_dataset for overlays (match notebook seed).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested npz is missing",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not draw training points (default: regenerate via simulate_dataset).",
    )
    parser.add_argument(
        "--sim-cfg",
        type=Path,
        default=None,
        help="Optional JSON dict merged into default simulate_dataset cfg (e.g. {\"rcd\": 5.0}).",
    )
    args = parser.parse_args()

    data_cfg = dict(DEFAULT_SAMPLE_SIZE_SIMULATE_CFG)
    if args.sim_cfg is not None:
        with open(args.sim_cfg, encoding="utf-8") as f:
            data_cfg.update(json.load(f))

    out_dir = args.out_dir.resolve()
    root = args.outputs_root.resolve()

    def sizes_for(m: str) -> list[int]:
        return list(args.bnn_sample_sizes) if m == "bnn" else list(args.sample_sizes)

    for model in args.models:
        sample_sizes = sizes_for(model)
        if args.variant == "both":
            plot_stacked_gl_it_for_model(
                model=model,
                sample_sizes=sample_sizes,
                outputs_root=root,
                out_dir=out_dir,
                normalize_scope=args.normalize_scope,
                gl_samples=args.gl_samples,
                dpi=args.dpi,
                strict=args.strict,
                seed=args.seed,
                data_cfg=data_cfg,
                overlay_points=not args.no_overlay,
                rows=args.rows,
            )
        else:
            plot_panels_for_model(
                model=model,
                variant=args.variant,
                sample_sizes=sample_sizes,
                outputs_root=root,
                out_dir=out_dir,
                normalize_scope=args.normalize_scope,
                gl_samples=args.gl_samples,
                dpi=args.dpi,
                strict=args.strict,
                seed=args.seed,
                data_cfg=data_cfg,
                overlay_points=not args.no_overlay,
                rows=args.rows,
            )


if __name__ == "__main__":
    main()
