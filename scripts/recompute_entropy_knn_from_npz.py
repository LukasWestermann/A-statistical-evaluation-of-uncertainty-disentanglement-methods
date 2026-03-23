"""
Recompute AU/EU via k-NN (Kozachenko–Leonenko) from saved regression raw_outputs .npz.

Single file:
    python scripts/recompute_entropy_knn_from_npz.py [path/to/file.npz] [--L 5000] [--k 3] [--seed 0]

Batch (OOD, sample_size, noise_level, undersampling — all four models; excludes OVB):
    python scripts/recompute_entropy_knn_from_npz.py --batch

Outputs default to results/entropy_recomputed_knn/ (statistics/ + plots/ + tables/).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (
    CONDITIONS_4,
    DEFAULT_OOD_RANGES,
    MODEL_RESOLVERS,
    build_noise_level_entropy_dataframe,
    build_sample_size_entropy_dataframe,
    build_undersampling_approx_dataframe,
    collect_raw_npz_files,
    compute_knn_grid_result,
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
    rng_for_npz,
    save_stats_excel,
)

DEFAULT_NPZ_DIR = project_root / "results" / "ood" / "outputs" / "ood" / "heteroscedastic" / "sin"
DEFAULT_GLOB = "*BAMLSS*raw_outputs*.npz"


def _default_out_root() -> Path:
    return project_root / "results" / "entropy_recomputed_knn"


def run_single_npz(
    npz_path: Path,
    L: int,
    k_nn: int,
    seed: int,
    out_root: Path,
    grid_stride: int,
    ood_ranges: Sequence[Tuple[float, float]],
):
    import numpy as np
    from utils.knn_entropy_regression import ensure_samples_first
    from utils.knn_entropy import entropy_uncertainty_knn_gaussian_mixture

    print("Loading:", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    mu = np.asarray(data["mu_samples"])
    sig = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    mu, sig = ensure_samples_first(mu, sig, x_grid)
    if grid_stride > 1:
        mu = mu[:, ::grid_stride]
        sig = sig[:, ::grid_stride]
        x_grid = x_grid[::grid_stride]

    rng = rng_for_npz(seed, npz_path)
    print("  mu_samples shape (M, N):", mu.shape)
    print("  L =", L, ", k_nn =", k_nn, ", seed =", seed)
    ent = entropy_uncertainty_knn_gaussian_mixture(mu, sig, L=L, k_nn=k_nn, rng=rng)
    ale = np.asarray(ent["aleatoric"]).squeeze()
    epi = np.asarray(ent["epistemic"]).squeeze()
    tot = np.asarray(ent["total"]).squeeze()
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    plot_dir = out_root / "plots" / "single"
    out_path = plot_dir / f"entropy_recomputed_knn_L{L}_k{k_nn}_{npz_path.stem}.png"
    title = f"k-NN entropy (Eq. 6), L={L}, k={k_nn}"
    saved = plot_entropy_lines(x, ale, epi, tot, out_path, title, ood_ranges)
    for p in saved:
        print("Saved plot:", p)


def run_batch(
    out_root: Path,
    L: int,
    k_nn: int,
    seed: int,
    experiments: Sequence[str],
    grid_stride: int,
    ood_ranges: Sequence[Tuple[float, float]],
    per_npz_plots: bool,
):
    import numpy as np

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
        "Batch k-NN: cost scales with (# members × # grid points). "
        "Use --grid-stride and smaller --L if needed.",
        flush=True,
    )

    for key, base, panel_ood_ranges, title_short in configs:
        for func_type, noise_type in CONDITIONS_4:
            search_dir = base / noise_type / func_type
            if not search_dir.exists():
                continue

            all_npz = collect_raw_npz_files(search_dir)
            ranges_for_knn = list(ood_ranges) if key == "ood" else list(panel_ood_ranges)
            ranges_for_panels = ranges_for_knn

            # --- Per-experiment stats ---
            if key == "ood":
                for npz_path in all_npz:
                    mk = model_key_from_stem(npz_path.stem)
                    if not mk:
                        continue
                    try:
                        res = compute_knn_grid_result(npz_path, L, k_nn, seed, ood_ranges, grid_stride)
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
                    save_stats_excel(df, out_sub, f"{date}_{mp}_knn_entropy_{sanitize_stem(npz_path.stem)}")
                    if per_npz_plots:
                        pdir = plots_root / key / noise_type / func_type / "per_npz"
                        plot_entropy_lines(
                            res.x, res.ale_entropy, res.epi_entropy, res.tot_entropy,
                            pdir / f"knn_L{L}_k{k_nn}_{npz_path.stem}.png",
                            f"k-NN L={L} k={k_nn}",
                            ood_ranges,
                        )

            elif key == "sample_size":
                by_model: Dict[str, List[Path]] = defaultdict(list)
                for npz_path in all_npz:
                    mk = model_key_from_stem(npz_path.stem)
                    if mk:
                        by_model[mk].append(npz_path)
                for model_tag, paths in by_model.items():
                    pct_to_ent: Dict = {}
                    pct_to_mse: Dict[float, float] = {}
                    meta_agg: Dict = {}
                    for npz_path in paths:
                        try:
                            res = compute_knn_grid_result(npz_path, L, k_nn, seed, [], grid_stride)
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
                    save_stats_excel(df, out_sub, f"{date}_{mp}_knn_entropy_sample_size")

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
                for (model_tag, dist), paths in by_grp.items():
                    tau_to_ent: Dict = {}
                    tau_to_mse: Dict[float, float] = {}
                    meta_last: Dict = {}
                    for npz_path in paths:
                        try:
                            res = compute_knn_grid_result(npz_path, L, k_nn, seed, [], grid_stride)
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
                    save_stats_excel(df, out_sub, f"{date}_{mp}_knn_entropy_noise_{dist}")

            elif key == "undersampling":
                for npz_path in all_npz:
                    mk = model_key_from_stem(npz_path.stem)
                    if not mk:
                        continue
                    try:
                        res = compute_knn_grid_result(npz_path, L, k_nn, seed, [], grid_stride)
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
                    save_stats_excel(df, out_sub, f"{date}_{mp}_knn_entropy_{sanitize_stem(npz_path.stem)}")

            # --- 2x4 panels (variance + k-NN entropy) ---
            display_names = [m[0] for m in MODEL_RESOLVERS]
            condition_data_list: List = []
            for _disp, _tag, globs in MODEL_RESOLVERS:
                p = resolve_latest_npz(search_dir, globs)
                if p is None:
                    condition_data_list.append(None)
                    continue
                try:
                    res = compute_knn_grid_result(p, L, k_nn, seed, ranges_for_knn, grid_stride)
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
                pdir / f"panel_{key}_knn_2x4_{slug}_variance.png",
                title_short,
                ranges_for_panels,
            )
            create_2x4_entropy_panel(
                condition_data_list, display_names, func_type, noise_type,
                pdir / f"panel_{key}_knn_2x4_{slug}_entropy.png",
                title_short,
                ranges_for_panels,
            )

    agg_path = tables_dir / f"knn_entropy_aggregate_{date}.xlsx"
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
    parser = argparse.ArgumentParser(description="k-NN entropy from raw_outputs .npz (batch or single file)")
    parser.add_argument("npz_path", nargs="?", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument(
        "--experiments",
        default="ood,sample_size,noise_level,undersampling",
        help="Comma-separated experiment keys",
    )
    parser.add_argument("--L", type=int, default=5000)
    parser.add_argument("--k", type=int, default=3, dest="k_nn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-stride", type=int, default=1)
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
    args = parser.parse_args()
    out_root = Path(args.out_root) if args.out_root else _default_out_root()

    ood_ranges = [tuple(pair) for pair in args.ood_range] if args.ood_range else list(DEFAULT_OOD_RANGES)

    if args.batch:
        exps = [x.strip() for x in args.experiments.split(",") if x.strip()]
        run_batch(
            out_root, args.L, args.k_nn, args.seed, exps, args.grid_stride, ood_ranges, args.per_npz_plots,
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
        import numpy as np
        npz_files = sorted(search_dir.glob(DEFAULT_GLOB))
        if not npz_files:
            print("No npz found matching", DEFAULT_GLOB)
            sys.exit(1)
        npz_path = npz_files[-1]

    run_single_npz(npz_path, args.L, args.k_nn, args.seed, out_root, args.grid_stride, ood_ranges)
    print("Done.")


if __name__ == "__main__":
    main()
