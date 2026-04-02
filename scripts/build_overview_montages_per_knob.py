"""
Build per-knob appendix-style overviews: vertical montage of moment-matched 2×4 variance
(bands) and 2×4 entropy (lines) for each training % (sample size) or each τ (noise level).

Reads raw_outputs .npz from results/sample_size/outputs/sample_size and
results/noise_level/outputs/noise_level (per distribution subfolder).

Default output:
  results/summary_panels_consolidated/overview_per_knob/
    sample_size/{noise}/{func}/overview_pct{pct}_{func}_{noise}.png
    noise_level/{noise}/{func}/{distribution}/overview_tau{tau}_{func}_{noise}.png
    baseline/{noise}/{func}/overview_{func}_{noise}.png  (experiment ``baseline``: 100% training data,
      same montage as pct=100 but suptitle is only ``Linear, heteroscedastic — …`` with no "Sample size" prefix)

Example:
  python scripts/build_overview_montages_per_knob.py
  python scripts/build_overview_montages_per_knob.py --experiments sample_size --dpi 200
  python scripts/build_overview_montages_per_knob.py --experiments baseline
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image

from utils.knn_entropy_regression import (
    CONDITIONS_4,
    MODEL_RESOLVERS,
    compute_moment_matched_grid_result,
    create_2x4_entropy_panel,
    create_2x4_variance_panel,
    discover_pcts_in_dir,
    discover_taus_in_dir,
    knn_result_to_panel_dict,
    resolve_latest_npz_at_pct,
    resolve_latest_npz_at_tau,
)


def _canonical_model_tags() -> List[str]:
    return [m[1] for m in MODEL_RESOLVERS]


def parse_model_tags_filter(arg: Optional[str]) -> Optional[Set[str]]:
    if arg is None or not str(arg).strip():
        return None
    canonical = _canonical_model_tags()
    allowed: Set[str] = set()
    for tok in str(arg).split(","):
        t = tok.strip().replace(" ", "_")
        if not t:
            continue
        match = next((c for c in canonical if c.lower() == t.lower()), None)
        if match is None:
            raise ValueError(
                f"Unknown model tag {tok!r}. Use comma-separated stem tags: {', '.join(canonical)}"
            )
        allowed.add(match)
    return allowed if allowed else None


def _tau_filename_token(tau: float) -> str:
    if abs(tau - round(tau)) < 1e-9:
        return str(int(round(tau)))
    return str(tau).replace(".", "p")


def _vertical_montage(
    top: Image.Image,
    bottom: Image.Image,
    gap_px: int = 16,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    w = max(top.width, bottom.width)
    h = top.height + gap_px + bottom.height
    out = Image.new("RGB", (w, h), background)
    out.paste(top, ((w - top.width) // 2, 0))
    out.paste(bottom, ((w - bottom.width) // 2, top.height + gap_px))
    return out


def _resize_to_width(im: Image.Image, target_w: int) -> Image.Image:
    if im.width == target_w:
        return im
    scale = target_w / im.width
    new_h = max(1, int(round(im.height * scale)))
    return im.resize((target_w, new_h), Image.Resampling.LANCZOS)


def _build_condition_list(
    search_dir: Path,
    grid_stride: int,
    eps: float,
    ood_ranges: Sequence[Tuple[float, float]],
    model_tags_filter: Optional[Set[str]],
    resolve_one,
) -> List:
    display_names = [m[0] for m in MODEL_RESOLVERS]
    out: List = []
    for _disp, tag, _globs in MODEL_RESOLVERS:
        if model_tags_filter is not None and tag not in model_tags_filter:
            out.append(None)
            continue
        p = resolve_one(search_dir, tag)
        if p is None:
            out.append(None)
            continue
        try:
            res = compute_moment_matched_grid_result(p, list(ood_ranges), grid_stride, eps)
            out.append(knn_result_to_panel_dict(res))
        except Exception as e:
            print(f"  skip {tag}: {e}")
            out.append(None)
    return out


def _render_panels_to_buffers(
    condition_data_list: List,
    func_type: str,
    noise_type: str,
    experiment_title: str,
    ood_ranges: Sequence[Tuple[float, float]],
    dpi: int,
) -> Tuple[io.BytesIO, io.BytesIO]:
    names = [m[0] for m in MODEL_RESOLVERS]
    buf_v = io.BytesIO()
    buf_e = io.BytesIO()
    create_2x4_variance_panel(
        condition_data_list,
        names,
        func_type,
        noise_type,
        buf_v,
        experiment_title,
        ood_ranges,
        dpi=dpi,
    )
    create_2x4_entropy_panel(
        condition_data_list,
        names,
        func_type,
        noise_type,
        buf_e,
        experiment_title,
        ood_ranges,
        entropy_subtitle="Entropy",
        dpi=dpi,
    )
    buf_v.seek(0)
    buf_e.seek(0)
    return buf_v, buf_e


def _montage_from_buffers(buf_v: io.BytesIO, buf_e: io.BytesIO) -> Image.Image:
    im_v = Image.open(buf_v).convert("RGB")
    im_e = Image.open(buf_e).convert("RGB")
    im_e = _resize_to_width(im_e, im_v.width)
    return _vertical_montage(im_v, im_e)


def run_sample_size(
    out_root: Path,
    grid_stride: int,
    eps: float,
    model_tags_filter: Optional[Set[str]],
    dpi: int,
    save_components: bool,
) -> None:
    base = project_root / "results" / "sample_size" / "outputs" / "sample_size"
    title_short = "Sample size"
    ood_ranges: Sequence[Tuple[float, float]] = []

    for func_type, noise_type in CONDITIONS_4:
        search_dir = base / noise_type / func_type
        pcts = discover_pcts_in_dir(search_dir)
        if not pcts:
            print(f"[sample_size] no npz with pct under {search_dir}")
            continue
        for pct in pcts:
            cond = _build_condition_list(
                search_dir,
                grid_stride,
                eps,
                ood_ranges,
                model_tags_filter,
                lambda sd, tag: resolve_latest_npz_at_pct(sd, tag, pct),
            )
            if all(d is None for d in cond):
                print(f"[sample_size] skip {func_type}/{noise_type} pct={pct}: no models")
                continue
            pct_token = _tau_filename_token(pct)  # same int/float token rules
            sub = out_root / "sample_size" / noise_type / func_type
            sub.mkdir(parents=True, exist_ok=True)
            out_path = sub / f"overview_pct{pct_token}_{func_type}_{noise_type}.png"
            buf_v, buf_e = _render_panels_to_buffers(
                cond, func_type, noise_type, title_short, ood_ranges, dpi
            )
            montage = _montage_from_buffers(buf_v, buf_e)
            montage.save(out_path)
            print("Saved", out_path)
            if save_components:
                comp = sub / "components" / f"pct{pct_token}"
                comp.mkdir(parents=True, exist_ok=True)
                buf_v.seek(0)
                buf_e.seek(0)
                Image.open(buf_v).save(comp / "variance_2x4.png")
                Image.open(buf_e).save(comp / "entropy_2x4.png")


def run_sample_size_baseline(
    out_root: Path,
    grid_stride: int,
    eps: float,
    model_tags_filter: Optional[Set[str]],
    dpi: int,
    save_components: bool,
    baseline_pct: float,
) -> None:
    """Moment-matched variance+entropy montage at fixed training fraction; neutral titles (no sample-size knob)."""
    base = project_root / "results" / "sample_size" / "outputs" / "sample_size"
    experiment_title = ""
    ood_ranges: Sequence[Tuple[float, float]] = []

    for func_type, noise_type in CONDITIONS_4:
        search_dir = base / noise_type / func_type
        pcts = discover_pcts_in_dir(search_dir)
        if not pcts:
            print(f"[baseline] no npz with pct under {search_dir}")
            continue
        if not any(abs(float(p) - float(baseline_pct)) < 1e-6 for p in pcts):
            print(
                f"[baseline] skip {func_type}/{noise_type}: pct={baseline_pct} not in {pcts}"
            )
            continue
        cond = _build_condition_list(
            search_dir,
            grid_stride,
            eps,
            ood_ranges,
            model_tags_filter,
            lambda sd, tag: resolve_latest_npz_at_pct(sd, tag, baseline_pct),
        )
        if all(d is None for d in cond):
            print(f"[baseline] skip {func_type}/{noise_type} pct={baseline_pct}: no models")
            continue
        pct_token = _tau_filename_token(baseline_pct)
        sub = out_root / "baseline" / noise_type / func_type
        sub.mkdir(parents=True, exist_ok=True)
        out_path = sub / f"overview_{func_type}_{noise_type}.png"
        buf_v, buf_e = _render_panels_to_buffers(
            cond, func_type, noise_type, experiment_title, ood_ranges, dpi
        )
        montage = _montage_from_buffers(buf_v, buf_e)
        montage.save(out_path)
        print("Saved", out_path)
        if save_components:
            comp = sub / "components" / f"pct{pct_token}"
            comp.mkdir(parents=True, exist_ok=True)
            buf_v.seek(0)
            buf_e.seek(0)
            Image.open(buf_v).save(comp / "variance_2x4.png")
            Image.open(buf_e).save(comp / "entropy_2x4.png")


def run_noise_level(
    out_root: Path,
    grid_stride: int,
    eps: float,
    distribution: str,
    model_tags_filter: Optional[Set[str]],
    dpi: int,
    save_components: bool,
) -> None:
    base = project_root / "results" / "noise_level" / "outputs" / "noise_level"
    title_short = "Noise level"
    ood_ranges: Sequence[Tuple[float, float]] = []

    for func_type, noise_type in CONDITIONS_4:
        search_dir = base / noise_type / func_type / distribution
        taus = discover_taus_in_dir(search_dir)
        if not taus:
            print(f"[noise_level] no npz with tau under {search_dir}")
            continue
        for tau in taus:
            cond = _build_condition_list(
                search_dir,
                grid_stride,
                eps,
                ood_ranges,
                model_tags_filter,
                lambda sd, tag: resolve_latest_npz_at_tau(sd, tag, tau),
            )
            if all(d is None for d in cond):
                print(f"[noise_level] skip {func_type}/{noise_type} tau={tau}: no models")
                continue
            tau_token = _tau_filename_token(tau)
            sub = out_root / "noise_level" / noise_type / func_type / distribution
            sub.mkdir(parents=True, exist_ok=True)
            out_path = sub / f"overview_tau{tau_token}_{func_type}_{noise_type}.png"
            buf_v, buf_e = _render_panels_to_buffers(
                cond, func_type, noise_type, title_short, ood_ranges, dpi
            )
            montage = _montage_from_buffers(buf_v, buf_e)
            montage.save(out_path)
            print("Saved", out_path)
            if save_components:
                comp = sub / "components" / f"tau{tau_token}"
                comp.mkdir(parents=True, exist_ok=True)
                buf_v.seek(0)
                buf_e.seek(0)
                Image.open(buf_v).save(comp / "variance_2x4.png")
                Image.open(buf_e).save(comp / "entropy_2x4.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-knob variance+entropy overview montages.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "results" / "summary_panels_consolidated" / "overview_per_knob",
        help="Output root directory",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="sample_size,noise_level",
        help="Comma-separated: sample_size, noise_level, baseline",
    )
    parser.add_argument(
        "--baseline-pct",
        type=float,
        default=100.0,
        help="Training %% used for experiment baseline (default 100)",
    )
    parser.add_argument("--grid-stride", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--distribution",
        type=str,
        default="normal",
        help="Noise-level subdirectory (e.g. normal)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Also save standalone variance_2x4.png and entropy_2x4.png under components/",
    )
    parser.add_argument(
        "--model-tags",
        type=str,
        default=None,
        help="Optional comma-separated model stem tags (Deep_Ensemble, MC_Dropout, BNN, BAMLSS)",
    )
    args = parser.parse_args()
    model_tags_filter = parse_model_tags_filter(args.model_tags)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    exp_set = {e.strip().lower() for e in str(args.experiments).split(",") if e.strip()}
    if "sample_size" in exp_set:
        run_sample_size(out_root, args.grid_stride, args.eps, model_tags_filter, args.dpi, args.save_components)
    if "noise_level" in exp_set:
        run_noise_level(
            out_root,
            args.grid_stride,
            args.eps,
            args.distribution,
            model_tags_filter,
            args.dpi,
            args.save_components,
        )
    if "baseline" in exp_set:
        run_sample_size_baseline(
            out_root,
            args.grid_stride,
            args.eps,
            model_tags_filter,
            args.dpi,
            args.save_components,
            args.baseline_pct,
        )


if __name__ == "__main__":
    main()
