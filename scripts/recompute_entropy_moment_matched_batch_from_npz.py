"""
Recompute AU/EU via moment-matched analytical TU (E[sigma^2]+Var(mu) -> Gaussian entropy)
from saved regression raw_outputs .npz.

Batch mirrors recompute_entropy_numerical_batch_from_npz (stats, normalized regions, 2x4 panels).

Single file:
    python scripts/recompute_entropy_moment_matched_batch_from_npz.py [path/to/file.npz]

Batch:
    python scripts/recompute_entropy_moment_matched_batch_from_npz.py --batch --experiments ood,sample_size

Outputs default to results/entropy_recomputed_moment_matched_batch/ (statistics/ + plots/ + tables/).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.regression_summary_panels import (
    NORMALIZED_UNCERTAINTY_YLIM,
    create_noise_level_summary_panel,
    create_noise_level_summary_panel_au_eu_only,
    create_noise_level_summary_panel_corr_only,
    create_sample_size_summary_panel,
    create_sample_size_summary_panel_au_eu_only,
    create_sample_size_summary_panel_corr_only,
)
from utils.moment_matched_2x4_entropy_panels import (
    create_noise_level_moment_matched_entropy_panel_1x4_au_eu,
    create_noise_level_moment_matched_entropy_panel_1x4_corr,
    create_noise_level_moment_matched_entropy_panel_2x4,
    create_sample_size_moment_matched_entropy_panel_1x4_au_eu,
    create_sample_size_moment_matched_entropy_panel_1x4_corr,
    create_sample_size_moment_matched_entropy_panel_2x4,
)
from utils.knn_entropy_regression import (
    CONDITIONS_4,
    DEFAULT_OOD_RANGES,
    MODEL_RESOLVERS,
    build_noise_level_entropy_dataframe,
    build_sample_size_entropy_dataframe,
    build_undersampling_approx_dataframe,
    collect_raw_npz_files,
    compute_moment_matched_grid_result,
    create_1x4_variance_panel,
    create_2x4_entropy_panel,
    create_2x4_variance_panel,
    dataframe_ood_knn_three_regions,
    function_display,
    is_ovb_or_non_raw_path,
    knn_result_to_panel_dict,
    model_key_from_stem,
    model_prefix_for_filename,
    normalized_entropy_stats_ood_regions,
    plot_entropy_lines,
    read_npz_metadata,
    resolve_latest_npz,
    save_stats_excel,
)

DEFAULT_NPZ_DIR = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
DEFAULT_GLOB = "*BAMLSS*raw_outputs*.npz"

NOTE_OOD = (
    "NLL/CRPS/Spearman not in raw_outputs npz; moment-matched TU "
    "(Gaussian with Var = E[sigma^2]+Var(mu)), AU = mean member entropies"
)
NOTE_SAMPLE_SIZE = "from raw_outputs moment-matched analytical entropy"
NOTE_NOISE_LEVEL = "from raw_outputs moment-matched analytical entropy"
NOTE_UNDERSAMPLING = (
    "undersampling npz single grid; not split by sampling regions; moment-matched analytical entropy"
)

TAG_TO_DISPLAY = {tag: disp for disp, tag, _ in MODEL_RESOLVERS}


def _default_out_root() -> Path:
    return project_root / "results" / "entropy_recomputed_moment_matched_batch"


def _canonical_model_tags() -> List[str]:
    return [m[1] for m in MODEL_RESOLVERS]


def parse_model_tags_filter(arg: Optional[str]) -> Optional[set[str]]:
    if arg is None or not str(arg).strip():
        return None
    canonical = _canonical_model_tags()
    allowed: set[str] = set()
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


def run_single_npz(
    npz_path: Path,
    out_root: Path,
    grid_stride: int,
    ood_ranges: Sequence[Tuple[float, float]],
    eps: float,
):
    print("Loading:", npz_path)
    print("  grid_stride:", grid_stride, " eps:", eps)
    res = compute_moment_matched_grid_result(npz_path, ood_ranges, grid_stride, eps)
    print("  mu_samples shape (M, N):", res.mu_samples.shape)
    plot_dir = out_root / "plots" / "single"
    out_path = plot_dir / f"entropy_recomputed_moment_matched_{npz_path.stem}.png"
    title = "Entropy"
    saved = plot_entropy_lines(
        res.x, res.ale_entropy, res.epi_entropy, res.tot_entropy, out_path, title, ood_ranges
    )
    for p in saved:
        print("Saved plot:", p)


def run_batch(
    out_root: Path,
    experiments: Sequence[str],
    grid_stride: int,
    ood_ranges: Sequence[Tuple[float, float]],
    per_npz_plots: bool,
    eps: float,
    model_tags_filter: Optional[set[str]] = None,
):
    import numpy as np

    def _model_ok(mk: Optional[str]) -> bool:
        if not mk:
            return False
        if model_tags_filter is None:
            return True
        return mk in model_tags_filter

    date = datetime.now().strftime("%Y%m%d")
    stats_root = out_root / "statistics"
    plots_root = out_root / "plots"
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    aggregate: Dict[str, List[dict]] = {k: [] for k in ("ood", "sample_size", "noise_level", "undersampling")}

    exp_set = {e.strip().lower() for e in experiments}

    configs = []
    if "ood" in exp_set:
        configs.append(("ood", project_root / "results" / "ood" / "outputs" / "ood", list(ood_ranges), "OOD"))
    if "sample_size" in exp_set:
        configs.append(("sample_size", project_root / "results" / "sample_size" / "outputs" / "sample_size", [], "Sample size"))
    if "noise_level" in exp_set:
        configs.append(("noise_level", project_root / "results" / "noise_level" / "outputs" / "noise_level", [], "Noise level"))
    if "undersampling" in exp_set:
        configs.append(("undersampling", project_root / "results" / "undersampling" / "outputs" / "undersampling", [], "Undersampling"))

    print(
        "Batch moment-matched entropy: analytical (fast). Use --grid-stride if needed.",
        flush=True,
    )
    if model_tags_filter is not None:
        print("Model filter:", ", ".join(sorted(model_tags_filter)), flush=True)

    for key, base, panel_ood_ranges, title_short in configs:
        for func_type, noise_type in CONDITIONS_4:
            search_dir = base / noise_type / func_type
            if not search_dir.exists():
                continue

            all_npz = collect_raw_npz_files(search_dir)
            ranges_for_entropy = list(ood_ranges) if key == "ood" else list(panel_ood_ranges)
            ranges_for_panels = ranges_for_entropy

            if key == "ood":
                for _disp, model_tag, globs in MODEL_RESOLVERS:
                    if not _model_ok(model_tag):
                        continue
                    npz_path = resolve_latest_npz(search_dir, globs)
                    if npz_path is None:
                        continue
                    mk = model_key_from_stem(npz_path.stem)
                    if mk != model_tag:
                        print(f"[{key}] warn: latest npz stem tag {mk!r} != resolver {model_tag!r} for {npz_path.name}")
                    try:
                        res = compute_moment_matched_grid_result(
                            npz_path, ood_ranges, grid_stride, eps
                        )
                    except Exception as e:
                        print(f"[{key}] skip {npz_path.name}: {e}")
                        continue
                    meta = res.meta
                    model_name = meta.get("model_name") or mk
                    stats = normalized_entropy_stats_ood_regions(
                        res.ale_entropy, res.epi_entropy, res.tot_entropy,
                        res.mu_pred, res.y_clean_flat, res.id_mask, res.ood_mask,
                    )
                    df = dataframe_ood_knn_three_regions(
                        stats,
                        function_display(func_type),
                        noise_type,
                        func_type,
                        model_name,
                        date,
                        meta.get("dropout_p"),
                        meta.get("mc_samples"),
                        meta.get("n_nets"),
                        recompute_note=NOTE_OOD,
                    )
                    df.insert(0, "npz_stem", npz_path.stem)
                    rel = str(npz_path.relative_to(project_root)) if npz_path.is_relative_to(project_root) else str(npz_path)
                    df.insert(1, "npz_relpath", rel)
                    for _, row in df.iterrows():
                        aggregate["ood"].append(row.to_dict())
                    mp = model_prefix_for_filename(
                        str(model_name),
                        meta.get("dropout_p"),
                        meta.get("mc_samples"),
                        meta.get("n_nets"),
                    )
                    out_sub = stats_root / key / noise_type / func_type
                    save_stats_excel(df, out_sub, f"{date}_{mp}_moment_matched_entropy_{sanitize_stem(npz_path.stem)}")
                    if per_npz_plots:
                        pdir = plots_root / key / noise_type / func_type / "per_npz"
                        plot_entropy_lines(
                            res.x, res.ale_entropy, res.epi_entropy, res.tot_entropy,
                            pdir / f"moment_matched_{npz_path.stem}.png",
                            "Entropy",
                            ood_ranges,
                        )

            elif key == "sample_size":
                by_model: Dict[str, List[Path]] = defaultdict(list)
                for npz_path in all_npz:
                    mk = model_key_from_stem(npz_path.stem)
                    if mk:
                        by_model[mk].append(npz_path)
                stats_by_model_summary: Dict[str, pd.DataFrame] = {}
                for model_tag, paths in by_model.items():
                    if not _model_ok(model_tag):
                        continue
                    pct_to_ent: Dict = {}
                    pct_to_mse: Dict[float, float] = {}
                    meta_agg: Dict = {}
                    for npz_path in sorted(paths):
                        try:
                            res = compute_moment_matched_grid_result(
                                npz_path, [], grid_stride, eps
                            )
                        except Exception as e:
                            print(f"[{key}] skip {npz_path.name}: {e}")
                            continue
                        meta = res.meta
                        meta_agg.update({k: v for k, v in meta.items() if v is not None})
                        p = meta.get("pct")
                        if p is None:
                            print(f"[{key}] skip {npz_path.name}: no pct in npz/filename")
                            continue
                        pct_to_ent[float(p)] = (res.ale_entropy, res.epi_entropy, res.tot_entropy)
                        pct_to_mse[float(p)] = float(np.mean((res.mu_pred - res.y_clean_flat) ** 2))
                    if not pct_to_ent:
                        continue
                    model_name = str(meta_agg.get("model_name") or model_tag)
                    df = build_sample_size_entropy_dataframe(
                        pct_to_ent,
                        pct_to_mse,
                        function_display(func_type),
                        noise_type,
                        func_type,
                        model_name,
                        date,
                        meta_agg.get("dropout_p"),
                        meta_agg.get("mc_samples"),
                        meta_agg.get("n_nets"),
                        recompute_note=NOTE_SAMPLE_SIZE,
                    )
                    for _, row in df.iterrows():
                        aggregate["sample_size"].append(row.to_dict())
                    mp = model_prefix_for_filename(
                        model_name,
                        meta_agg.get("dropout_p"),
                        meta_agg.get("mc_samples"),
                        meta_agg.get("n_nets"),
                    )
                    out_sub = stats_root / key / noise_type / func_type
                    save_stats_excel(df, out_sub, f"{date}_{mp}_moment_matched_entropy_sample_size")
                    disp = TAG_TO_DISPLAY.get(model_tag)
                    if disp is not None:
                        stats_by_model_summary[disp] = df

                if stats_by_model_summary:
                    summ_dir = plots_root / key / "summary_panels" / noise_type / func_type
                    summ_dir.mkdir(parents=True, exist_ok=True)
                    fig = create_sample_size_summary_panel(
                        stats_by_model_summary,
                        func_type,
                        noise_type,
                        "entropy",
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out_png = summ_dir / (
                        f"sample_size_summary_4x2_{func_type}_{noise_type}_entropy_moment_matched.png"
                    )
                    fig.savefig(out_png, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved summary panel: {out_png}")

                    fig_1x4_au = create_sample_size_summary_panel_au_eu_only(
                        stats_by_model_summary,
                        func_type,
                        noise_type,
                        "entropy",
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out_1x4_au = summ_dir / (
                        f"sample_size_summary_1x4_au_eu_{func_type}_{noise_type}_entropy_moment_matched.png"
                    )
                    fig_1x4_au.savefig(out_1x4_au, dpi=300, bbox_inches="tight")
                    plt.close(fig_1x4_au)
                    print(f"Saved 1x4 AU/EU panel: {out_1x4_au}")

                    fig_1x4_corr = create_sample_size_summary_panel_corr_only(
                        stats_by_model_summary, func_type, noise_type, "entropy"
                    )
                    out_1x4_corr = summ_dir / (
                        f"sample_size_summary_1x4_corr_{func_type}_{noise_type}_entropy_moment_matched.png"
                    )
                    fig_1x4_corr.savefig(out_1x4_corr, dpi=300, bbox_inches="tight")
                    plt.close(fig_1x4_corr)
                    print(f"Saved 1x4 correlation panel: {out_1x4_corr}")

                    panel_dir = plots_root / key
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    fig2 = create_sample_size_moment_matched_entropy_panel_2x4(
                        stats_by_model_summary,
                        func_type,
                        noise_type,
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out2 = panel_dir / (
                        f"sample_size_moment_matched_2x4_{func_type}_{noise_type}_entropy.png"
                    )
                    fig2.savefig(out2, dpi=300, bbox_inches="tight")
                    plt.close(fig2)
                    print(f"Saved 2x4 TU/AU/EU panel: {out2}")

                    fig2_au = create_sample_size_moment_matched_entropy_panel_1x4_au_eu(
                        stats_by_model_summary,
                        func_type,
                        noise_type,
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out2_au = panel_dir / (
                        f"sample_size_moment_matched_1x4_au_eu_{func_type}_{noise_type}_entropy.png"
                    )
                    fig2_au.savefig(out2_au, dpi=300, bbox_inches="tight")
                    plt.close(fig2_au)
                    print(f"Saved 1x4 moment-matched AU/EU/TU panel: {out2_au}")

                    fig2_corr = create_sample_size_moment_matched_entropy_panel_1x4_corr(
                        stats_by_model_summary, func_type, noise_type
                    )
                    out2_corr = panel_dir / (
                        f"sample_size_moment_matched_1x4_corr_{func_type}_{noise_type}_entropy.png"
                    )
                    fig2_corr.savefig(out2_corr, dpi=300, bbox_inches="tight")
                    plt.close(fig2_corr)
                    print(f"Saved 1x4 moment-matched correlation panel: {out2_corr}")

            elif key == "noise_level":
                by_grp: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
                for npz_path in all_npz:
                    mk = model_key_from_stem(npz_path.stem)
                    if not mk:
                        continue
                    data = np.load(npz_path, allow_pickle=True)
                    meta0 = read_npz_metadata(npz_path, data)
                    dist = str(meta0.get("distribution") or "normal")
                    by_grp[(mk, dist)].append(npz_path)
                stats_by_dist_model: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
                for (model_tag, dist), paths in by_grp.items():
                    if not _model_ok(model_tag):
                        continue
                    tau_to_ent: Dict = {}
                    tau_to_mse: Dict[float, float] = {}
                    meta_last: Dict = {}
                    for npz_path in sorted(paths):
                        try:
                            res = compute_moment_matched_grid_result(
                                npz_path, [], grid_stride, eps
                            )
                        except Exception as e:
                            print(f"[{key}] skip {npz_path.name}: {e}")
                            continue
                        meta_last = res.meta
                        t = meta_last.get("tau")
                        if t is None:
                            print(f"[{key}] skip {npz_path.name}: no tau")
                            continue
                        tau_to_ent[float(t)] = (res.ale_entropy, res.epi_entropy, res.tot_entropy)
                        tau_to_mse[float(t)] = float(np.mean((res.mu_pred - res.y_clean_flat) ** 2))
                    if not tau_to_ent:
                        continue
                    model_name = str(meta_last.get("model_name") or model_tag)
                    df = build_noise_level_entropy_dataframe(
                        tau_to_ent,
                        tau_to_mse,
                        dist,
                        function_display(func_type),
                        noise_type,
                        func_type,
                        model_name,
                        date,
                        meta_last.get("dropout_p"),
                        meta_last.get("mc_samples"),
                        meta_last.get("n_nets"),
                        recompute_note=NOTE_NOISE_LEVEL,
                    )
                    for _, row in df.iterrows():
                        aggregate["noise_level"].append(row.to_dict())
                    mp = model_prefix_for_filename(
                        model_name,
                        meta_last.get("dropout_p"),
                        meta_last.get("mc_samples"),
                        meta_last.get("n_nets"),
                    )
                    out_sub = stats_root / key / noise_type / func_type
                    save_stats_excel(df, out_sub, f"{date}_{mp}_moment_matched_entropy_noise_{dist}")
                    disp = TAG_TO_DISPLAY.get(model_tag)
                    if disp is not None:
                        stats_by_dist_model[dist][disp] = df

                for dist, smap in stats_by_dist_model.items():
                    if not smap:
                        continue
                    summ_dir = plots_root / key / "summary_panels" / noise_type / func_type
                    summ_dir.mkdir(parents=True, exist_ok=True)
                    fig = create_noise_level_summary_panel(
                        smap,
                        func_type,
                        noise_type,
                        "entropy",
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out_png = summ_dir / (
                        f"noise_level_summary_4x2_{func_type}_{noise_type}_{dist}_entropy_moment_matched.png"
                    )
                    fig.savefig(out_png, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved 4x2 summary: {out_png}")

                    fig_nl_au = create_noise_level_summary_panel_au_eu_only(
                        smap,
                        func_type,
                        noise_type,
                        "entropy",
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out_nl_au = summ_dir / (
                        f"noise_level_summary_1x4_au_eu_{func_type}_{noise_type}_{dist}_entropy_moment_matched.png"
                    )
                    fig_nl_au.savefig(out_nl_au, dpi=300, bbox_inches="tight")
                    plt.close(fig_nl_au)
                    print(f"Saved 1x4 AU/EU panel: {out_nl_au}")

                    fig_nl_corr = create_noise_level_summary_panel_corr_only(
                        smap, func_type, noise_type, "entropy"
                    )
                    out_nl_corr = summ_dir / (
                        f"noise_level_summary_1x4_corr_{func_type}_{noise_type}_{dist}_entropy_moment_matched.png"
                    )
                    fig_nl_corr.savefig(out_nl_corr, dpi=300, bbox_inches="tight")
                    plt.close(fig_nl_corr)
                    print(f"Saved 1x4 correlation panel: {out_nl_corr}")

                    panel_dir = plots_root / key
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    fig2 = create_noise_level_moment_matched_entropy_panel_2x4(
                        smap,
                        func_type,
                        noise_type,
                        distribution=dist,
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out2 = panel_dir / (
                        f"noise_level_moment_matched_2x4_{func_type}_{noise_type}_{dist}_entropy.png"
                    )
                    fig2.savefig(out2, dpi=300, bbox_inches="tight")
                    plt.close(fig2)
                    print(f"Saved 2x4 TU/AU/EU panel: {out2}")

                    fig2_nl_au = create_noise_level_moment_matched_entropy_panel_1x4_au_eu(
                        smap,
                        func_type,
                        noise_type,
                        distribution=dist,
                        uncertainty_ylim=NORMALIZED_UNCERTAINTY_YLIM,
                    )
                    out2_nl_au = panel_dir / (
                        f"noise_level_moment_matched_1x4_au_eu_{func_type}_{noise_type}_{dist}_entropy.png"
                    )
                    fig2_nl_au.savefig(out2_nl_au, dpi=300, bbox_inches="tight")
                    plt.close(fig2_nl_au)
                    print(f"Saved 1x4 moment-matched AU/EU/TU panel: {out2_nl_au}")

                    fig2_nl_corr = create_noise_level_moment_matched_entropy_panel_1x4_corr(
                        smap, func_type, noise_type, distribution=dist
                    )
                    out2_nl_corr = panel_dir / (
                        f"noise_level_moment_matched_1x4_corr_{func_type}_{noise_type}_{dist}_entropy.png"
                    )
                    fig2_nl_corr.savefig(out2_nl_corr, dpi=300, bbox_inches="tight")
                    plt.close(fig2_nl_corr)
                    print(f"Saved 1x4 moment-matched correlation panel: {out2_nl_corr}")

            elif key == "undersampling":
                for _disp, model_tag, globs in MODEL_RESOLVERS:
                    if not _model_ok(model_tag):
                        continue
                    npz_path = resolve_latest_npz(search_dir, globs)
                    if npz_path is None:
                        continue
                    mk = model_key_from_stem(npz_path.stem)
                    try:
                        res = compute_moment_matched_grid_result(
                            npz_path, [], grid_stride, eps
                        )
                    except Exception as e:
                        print(f"[{key}] skip {npz_path.name}: {e}")
                        continue
                    meta = res.meta
                    model_name = str(meta.get("model_name") or mk)
                    df = build_undersampling_approx_dataframe(
                        res.ale_entropy, res.epi_entropy, res.tot_entropy,
                        res.mu_pred, res.y_clean_flat,
                        function_display(func_type),
                        noise_type, func_type, model_name, date,
                        meta.get("dropout_p"), meta.get("mc_samples"), meta.get("n_nets"),
                        recompute_note=NOTE_UNDERSAMPLING,
                    )
                    df.insert(0, "npz_stem", npz_path.stem)
                    for _, row in df.iterrows():
                        aggregate["undersampling"].append(row.to_dict())
                    mp = model_prefix_for_filename(
                        model_name,
                        meta.get("dropout_p"),
                        meta.get("mc_samples"),
                        meta.get("n_nets"),
                    )
                    out_sub = stats_root / key / noise_type / func_type
                    save_stats_excel(df, out_sub, f"{date}_{mp}_moment_matched_entropy_{sanitize_stem(npz_path.stem)}")

            display_names = [m[0] for m in MODEL_RESOLVERS]
            condition_data_list: List = []
            for _disp, _tag, globs in MODEL_RESOLVERS:
                if not _model_ok(_tag):
                    condition_data_list.append(None)
                    continue
                p = resolve_latest_npz(search_dir, globs)
                if p is None:
                    condition_data_list.append(None)
                    continue
                try:
                    res = compute_moment_matched_grid_result(
                        p, ranges_for_entropy, grid_stride, eps
                    )
                    condition_data_list.append(knn_result_to_panel_dict(res))
                except Exception as e:
                    print(f"[{key}] panel skip {_tag}: {e}")
                    condition_data_list.append(None)

            if all(d is None for d in condition_data_list):
                continue
            slug = f"{func_type}_{noise_type}"
            pdir = plots_root / key
            create_2x4_variance_panel(
                condition_data_list, display_names, func_type, noise_type,
                pdir / f"panel_{key}_moment_matched_2x4_{slug}_variance.png",
                title_short,
                ranges_for_panels,
            )
            create_1x4_variance_panel(
                condition_data_list, display_names, func_type, noise_type,
                pdir / f"panel_{key}_moment_matched_1x4_{slug}_variance_aleatoric.png",
                title_short,
                ranges_for_panels,
                "aleatoric",
            )
            create_1x4_variance_panel(
                condition_data_list, display_names, func_type, noise_type,
                pdir / f"panel_{key}_moment_matched_1x4_{slug}_variance_epistemic.png",
                title_short,
                ranges_for_panels,
                "epistemic",
            )
            create_2x4_entropy_panel(
                condition_data_list, display_names, func_type, noise_type,
                pdir / f"panel_{key}_moment_matched_2x4_{slug}_entropy.png",
                title_short,
                ranges_for_panels,
                entropy_subtitle="Entropy",
            )

    agg_path = tables_dir / f"moment_matched_entropy_aggregate_{date}.xlsx"
    any_rows = any(aggregate[k] for k in aggregate)
    if any_rows:
        with pd.ExcelWriter(agg_path, engine="openpyxl") as writer:
            for sheet, rows in aggregate.items():
                if not rows:
                    continue
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet[:31], index=False)
        print("Wrote aggregate workbook:", agg_path)
    else:
        print("No statistics rows generated; skipping aggregate workbook.")


def sanitize_stem(stem: str, max_len: int = 80) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return s[:max_len]


def main():
    parser = argparse.ArgumentParser(
        description="Moment-matched analytical entropy from raw_outputs .npz (batch or single file)"
    )
    parser.add_argument("npz_path", nargs="?", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument(
        "--experiments",
        default="ood,sample_size,noise_level,undersampling",
        help="Comma-separated experiment keys",
    )
    parser.add_argument("--grid-stride", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-10, help="Variance floor for Gaussian entropy")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--ood-range",
        action="append",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="OOD x-interval (repeatable). Default: 10 15",
    )
    parser.add_argument("--per-npz-plots", action="store_true", help="OOD only: save line plots per npz")
    parser.add_argument(
        "--models",
        default=None,
        metavar="TAGS",
        help=f"Comma-separated stem tags to include: {', '.join(_canonical_model_tags())}",
    )
    args = parser.parse_args()
    out_root = Path(args.out_root) if args.out_root else _default_out_root()

    ood_ranges = [tuple(pair) for pair in args.ood_range] if args.ood_range else list(DEFAULT_OOD_RANGES)

    try:
        model_tags_filter = parse_model_tags_filter(args.models)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    if args.batch:
        exps = [x.strip() for x in args.experiments.split(",") if x.strip()]
        run_batch(
            out_root,
            exps,
            args.grid_stride,
            ood_ranges,
            args.per_npz_plots,
            args.eps,
            model_tags_filter=model_tags_filter,
        )
        print("Done (batch).")
        return

    if args.npz_path:
        npz_path = Path(args.npz_path)
        if not npz_path.is_file():
            print("File not found:", npz_path)
            sys.exit(1)
        if is_ovb_or_non_raw_path(npz_path):
            print("Refusing OVB or non-raw_outputs path:", npz_path)
            sys.exit(1)
    else:
        search_dir = DEFAULT_NPZ_DIR
        if not search_dir.exists():
            print("Default directory not found:", search_dir)
            sys.exit(1)
        npz_files = sorted(search_dir.glob(DEFAULT_GLOB))
        if not npz_files:
            print("No npz found matching", DEFAULT_GLOB)
            sys.exit(1)
        npz_path = npz_files[-1]

    run_single_npz(npz_path, out_root, args.grid_stride, ood_ranges, args.eps)
    print("Done.")


if __name__ == "__main__":
    main()
