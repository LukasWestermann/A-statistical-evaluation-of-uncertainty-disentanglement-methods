"""
Qualitative sample-size baseline panels at a fixed training percentage (default 100%).

For each (function type × noise type): loads the latest ``*pct<k>*raw_outputs*.npz`` per model from
``results/sample_size/outputs/sample_size/<noise_type>/<func_type>/``. Uncertainties are computed with
``compute_moment_matched_grid_result`` (same moment-matched analytical entropy as
``recompute_entropy_moment_matched_batch_from_npz.py`` / ``plot_baseline_pct100_variance_entropy_overview.py``).
The script writes:

- 2×4 variance (aleatoric / epistemic ±σ bands)
- 2×4 entropy (aleatoric / epistemic lines, twin y-axis)
- 1×4 variance aleatoric only
- 1×4 variance epistemic only

Figure suptitles use ``function_display``, noise, and plot kind only (no "sample size" prefix).

Example::

    python scripts/plot_sample_size_2x4_sin_hetero_100.py
    python scripts/plot_sample_size_2x4_sin_hetero_100.py --func sin --noise heteroscedastic
    python scripts/plot_sample_size_2x4_sin_hetero_100.py --pct 100 --out-root results/sample_size/plots/pct100_baseline
    python scripts/plot_sample_size_2x4_sin_hetero_100.py --grid-stride 1 --eps 1e-10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (
    compute_moment_matched_grid_result,
    create_1x4_variance_panel,
    create_2x4_entropy_panel,
    create_2x4_variance_panel,
)

FuncType = Literal["linear", "sin"]
NoiseType = Literal["homoscedastic", "heteroscedastic"]

OUTPUTS_BASE = project_root / "results" / "sample_size" / "outputs" / "sample_size"
DEFAULT_OUT_ROOT = project_root / "results" / "sample_size" / "plots" / "pct100_baseline"

# No OOD shading (in-distribution qualitative plots)
OOD_RANGES: List[Tuple[float, float]] = []


def _pct_token(pct: float) -> str:
    if abs(pct - round(pct)) < 1e-9:
        return str(int(round(pct)))
    s = f"{pct:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def model_globs(pct: float) -> Sequence[Tuple[str, str]]:
    tok = _pct_token(pct)
    return (
        (f"*Deep_Ensemble*pct{tok}*raw_outputs*", "Deep Ensemble"),
        (f"*MC_Dropout*pct{tok}*raw_outputs*", "MC Dropout"),
        (f"*BNN*pct{tok}*raw_outputs*", "BNN"),
        (f"*BAMLSS*pct{tok}*raw_outputs*", "BAMLSS"),
    )


def load_model_data(
    npz_path: Path,
    ood_ranges: Sequence[Tuple[float, float]],
    grid_stride: int,
    eps: float,
) -> dict:
    """Moment-matched grid result as panel dict (entropy matches batch recomputation pipeline)."""
    res = compute_moment_matched_grid_result(npz_path, ood_ranges, grid_stride, eps)
    return {
        "x": res.x,
        "y_clean_flat": res.y_clean_flat,
        "mu_pred": res.mu_pred,
        "ale_var": res.ale_var,
        "epi_var": res.epi_var,
        "ale_entropy": res.ale_entropy,
        "epi_entropy": res.epi_entropy,
        "ood_mask": res.ood_mask,
        "id_mask": res.id_mask,
        "boundary_x": res.boundary_x,
        "x_train_flat": res.x_train_flat,
        "y_train_flat": res.y_train_flat,
    }


def load_condition_list(
    search_dir: Path,
    models: Sequence[Tuple[str, str]],
    ood_ranges: Sequence[Tuple[float, float]],
    grid_stride: int,
    eps: float,
) -> List[Optional[dict]]:
    out: List[Optional[dict]] = []
    for pattern, _display_name in models:
        npz_files = sorted(search_dir.glob(f"{pattern}.npz"))
        if not npz_files:
            out.append(None)
            continue
        npz_path = npz_files[-1]
        try:
            out.append(load_model_data(npz_path, ood_ranges, grid_stride, eps))
        except Exception as e:
            print(f"  Error loading {npz_path}: {e}")
            out.append(None)
    return out


def process_one_condition(
    func_type: FuncType,
    noise_type: NoiseType,
    pct: float,
    out_dir: Path,
    models: Sequence[Tuple[str, str]],
    display_names: Sequence[str],
    grid_stride: int,
    eps: float,
) -> bool:
    search_dir = OUTPUTS_BASE / noise_type / func_type
    if not search_dir.exists():
        print(f"  [skip] missing directory: {search_dir}")
        return False

    condition_data_list = load_condition_list(search_dir, models, OOD_RANGES, grid_stride, eps)
    if all(d is None for d in condition_data_list):
        print(f"  [skip] no pct{_pct_token(pct)} npz under {search_dir}")
        return False

    sub = out_dir / noise_type / func_type
    sub.mkdir(parents=True, exist_ok=True)
    tok = _pct_token(pct)

    stem = f"pct{tok}_{func_type}_{noise_type}"
    p_2x4_v = sub / f"panel_2x4_variance_{stem}.png"
    p_2x4_e = sub / f"panel_2x4_entropy_{stem}.png"
    p_1x4_ale = sub / f"panel_1x4_variance_aleatoric_{stem}.png"
    p_1x4_epi = sub / f"panel_1x4_variance_epistemic_{stem}.png"

    # Baseline: no experiment knob in the suptitle (empty string)
    experiment_title = ""

    create_2x4_variance_panel(
        condition_data_list,
        display_names,
        func_type,
        noise_type,
        p_2x4_v,
        experiment_title,
        OOD_RANGES,
    )
    print(f"  Saved: {p_2x4_v}")

    create_2x4_entropy_panel(
        condition_data_list,
        display_names,
        func_type,
        noise_type,
        p_2x4_e,
        experiment_title,
        OOD_RANGES,
        entropy_subtitle="Entropy (line plots)",
    )
    print(f"  Saved: {p_2x4_e}")

    create_1x4_variance_panel(
        condition_data_list,
        display_names,
        func_type,
        noise_type,
        p_1x4_ale,
        experiment_title,
        OOD_RANGES,
        "aleatoric",
    )
    print(f"  Saved: {p_1x4_ale}")

    create_1x4_variance_panel(
        condition_data_list,
        display_names,
        func_type,
        noise_type,
        p_1x4_epi,
        experiment_title,
        OOD_RANGES,
        "epistemic",
    )
    print(f"  Saved: {p_1x4_epi}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pct100 qualitative 2×4 / 1×4 panels for all or selected conditions.")
    p.add_argument(
        "--pct",
        type=float,
        default=100.0,
        help="Training percentage token in filenames (default: 100).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUT_ROOT}).",
    )
    p.add_argument(
        "--func",
        choices=("linear", "sin"),
        action="append",
        default=None,
        help="Restrict to this function type (repeatable). Default: all.",
    )
    p.add_argument(
        "--noise",
        choices=("homoscedastic", "heteroscedastic"),
        action="append",
        default=None,
        help="Restrict to this noise type (repeatable). Default: all.",
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=1,
        help="Subsample grid points (1 = full grid; same as baseline overview script).",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="Numerical floor for moment-matched entropy (default: 1e-10).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pct = float(args.pct)
    models = model_globs(pct)
    display_names = [m[1] for m in models]

    funcs: Tuple[FuncType, ...] = ("linear", "sin")
    noises: Tuple[NoiseType, ...] = ("homoscedastic", "heteroscedastic")
    if args.func:
        funcs = tuple(dict.fromkeys(args.func))  # type: ignore[assignment]
    if args.noise:
        noises = tuple(dict.fromkeys(args.noise))  # type: ignore[assignment]

    out_root: Path = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    gs = int(args.grid_stride)
    eps = float(args.eps)
    print(
        f"Outputs under {out_root} (pct={_pct_token(pct)}); "
        f"moment-matched entropy (grid_stride={gs}, eps={eps}); "
        f"reading from {OUTPUTS_BASE}/<noise>/<func>/"
    )
    n_ok = 0
    for noise_type in noises:
        for func_type in funcs:
            print(f"\n=== {func_type}, {noise_type} ===")
            if process_one_condition(
                func_type,
                noise_type,
                pct,
                out_root,
                models,
                display_names,
                gs,
                eps,
            ):
                n_ok += 1

    print(f"\nDone. Wrote panels for {n_ok} condition(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
