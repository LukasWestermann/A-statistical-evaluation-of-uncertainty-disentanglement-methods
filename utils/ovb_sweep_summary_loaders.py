"""
Load OVB parameter statistics for 1×4 summary line plots (variance from legacy Excel; entropy from batch Excel).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from utils.regression_summary_panels import MODEL_KEYS, FuncType, NoiseType

SweepParam = Literal["rho", "beta2"]

# results/ovb/<subdir>/<noise>/<func>/
OVB_MODEL_SUBDIRS: Dict[str, str] = {
    "Deep Ensemble": "deep_ensemble",
    "MC Dropout": "mcdropout",
    "BNN": "bnn",
    "BAMLSS": "bamlss",
}


def _variance_stats_glob(model_name: str, sweep: SweepParam) -> str:
    if model_name == "MC Dropout":
        return f"ovb_{sweep}_stats_*.xlsx"
    if model_name == "Deep Ensemble":
        return f"deep_ensemble_ovb_{sweep}_stats_*.xlsx"
    if model_name == "BNN":
        return f"bnn_ovb_{sweep}_stats_*.xlsx"
    if model_name == "BAMLSS":
        return f"bamlss_ovb_{sweep}_stats_*.xlsx"
    raise ValueError(f"Unknown model_name: {model_name}")


def _pick_latest_excel(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_ovb_variance_sweep_frame(
    ovb_root: Path,
    func_type: FuncType,
    noise_type: NoiseType,
    model_name: str,
    sweep: SweepParam,
    stats_date_suffix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the newest matching ``*_ovb_{rho|beta2}_stats_*.xlsx`` for one model and condition.
    If ``stats_date_suffix`` is set (e.g. ``20260406``), only files whose stem ends with that date are considered.
    """
    sub = OVB_MODEL_SUBDIRS[model_name]
    directory = ovb_root / sub / noise_type / func_type
    pattern = _variance_stats_glob(model_name, sweep)
    if stats_date_suffix:
        # Restrict to files containing the date token (before .xlsx)
        if not directory.is_dir():
            return pd.DataFrame()
        cands = [
            p
            for p in directory.glob(pattern)
            if stats_date_suffix in p.stem
        ]
        matches = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        path = matches[0] if matches else None
    else:
        path = _pick_latest_excel(directory, pattern)
    if path is None:
        return pd.DataFrame()
    return pd.read_excel(path)


def _pick_rho_for_beta2_sweep(
    df: pd.DataFrame,
    fixed_rho: float,
    rtol: float,
    atol: float,
) -> Optional[float]:
    """
    If ``fixed_rho`` matches no rows, infer ρ for a β₂-sweep slice:

    - **Single ρ in table** (typical ``ovb_beta2_stats``): use that value (e.g. if CLI ``fixed_rho``
      does not match the file).
    - **Mixed tables** (e.g. concatenated moment-matched workbooks): use the ρ that has the largest
      number of distinct β₂ values (the actual β₂ sweep block).
    """
    rho_col = df["rho"].astype(float)
    mask = np.isclose(rho_col, fixed_rho, rtol=rtol, atol=atol)
    if mask.any():
        return fixed_rho
    ur = np.unique(rho_col.values)
    if len(ur) == 1:
        return float(ur[0])
    best_rho: Optional[float] = None
    best_n = 0
    for rv in ur:
        sub = df.loc[np.isclose(rho_col, float(rv), rtol=rtol, atol=atol)]
        n_beta = int(sub["beta2"].nunique())
        if n_beta > best_n:
            best_n = n_beta
            best_rho = float(rv)
    if best_n > 1 and best_rho is not None:
        return best_rho
    return None


def filter_sort_variance_sweep(
    df: pd.DataFrame,
    sweep: SweepParam,
    fixed_beta2: float,
    fixed_rho: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Keep rows for rho sweep (fixed beta2) or beta2 sweep (fixed rho); sort by varying parameter."""
    if df.empty or "rho" not in df.columns or "beta2" not in df.columns:
        return pd.DataFrame()
    if sweep == "rho":
        mask = np.isclose(df["beta2"].astype(float), fixed_beta2, rtol=rtol, atol=atol)
        out = df.loc[mask].copy()
        out = out.sort_values("rho", kind="mergesort")
        # One row per ρ (concatenated batch Excel can repeat the same run)
        out = out.assign(_xkey=out["rho"].astype(float).round(8)).drop_duplicates(
            subset=["_xkey"], keep="last"
        )
        out = out.drop(columns=["_xkey"])
    else:
        rho_col = df["rho"].astype(float)
        rho_use = _pick_rho_for_beta2_sweep(df, fixed_rho, rtol=rtol, atol=atol)
        if rho_use is None:
            return pd.DataFrame()
        mask = np.isclose(rho_col, rho_use, rtol=rtol, atol=atol)
        out = df.loc[mask].copy()
        out = out.sort_values("beta2", kind="mergesort")
        out = out.assign(_xkey=out["beta2"].astype(float).round(8)).drop_duplicates(
            subset=["_xkey"], keep="last"
        )
        out = out.drop(columns=["_xkey"])
    return out


def _display_name_from_model_field(raw: object) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    for display, key in MODEL_KEYS.items():
        if s == display or s == key or s.replace(" ", "_") == key:
            return display
    return None


def _display_name_from_filename(path: Path) -> Optional[str]:
    stem = path.stem
    for display, key in MODEL_KEYS.items():
        if key in stem:
            return display
    compact = stem.replace("_", "").upper()
    for display, key in MODEL_KEYS.items():
        ku = key.replace("_", "").upper()
        if ku and ku in compact:
            return display
    return None


def load_ovb_entropy_concat_for_condition(
    entropy_stats_root: Path,
    func_type: FuncType,
    noise_type: NoiseType,
) -> pd.DataFrame:
    """Concatenate all moment-matched OVB statistics workbooks for one (noise, func)."""
    directory = entropy_stats_root / "ovb" / noise_type / func_type
    if not directory.is_dir():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.xlsx")):
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        df["_source_xlsx"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    names: list[Optional[str]] = []
    for _, row in combined.iterrows():
        raw = row["model_name"] if "model_name" in combined.columns else None
        disp = _display_name_from_model_field(raw)
        if disp is None and "npz_relpath" in combined.columns:
            disp = _display_name_from_filename(Path(str(row["npz_relpath"])))
        if disp is None:
            disp = _display_name_from_filename(Path(str(row.get("_source_xlsx", ""))))
        names.append(disp)
    combined["_display_model"] = names
    return combined


def filter_entropy_rows_for_model(
    combined: pd.DataFrame,
    model_name: str,
    sweep: SweepParam,
    fixed_beta2: float,
    fixed_rho: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Rows for one display model, filtered to one sweep slice, sorted."""
    if combined.empty:
        return pd.DataFrame()
    m = combined["_display_model"] == model_name
    df = combined.loc[m].copy()
    if df.empty:
        return pd.DataFrame()
    return filter_sort_variance_sweep(df, sweep, fixed_beta2, fixed_rho, rtol=rtol, atol=atol)


def load_stats_by_model_variance(
    ovb_root: Path,
    func_type: FuncType,
    noise_type: NoiseType,
    sweep: SweepParam,
    fixed_beta2: float,
    fixed_rho: float,
    stats_date_suffix: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Per-model DataFrames ready for plotting (columns rho/beta2 + mean_*_var_norm)."""
    order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]
    out: Dict[str, pd.DataFrame] = {}
    for name in order:
        df = load_ovb_variance_sweep_frame(
            ovb_root, func_type, noise_type, name, sweep, stats_date_suffix=stats_date_suffix
        )
        out[name] = filter_sort_variance_sweep(df, sweep, fixed_beta2, fixed_rho)
    return out


def load_stats_by_model_entropy(
    entropy_stats_root: Path,
    func_type: FuncType,
    noise_type: NoiseType,
    sweep: SweepParam,
    fixed_beta2: float,
    fixed_rho: float,
) -> Dict[str, pd.DataFrame]:
    combined = load_ovb_entropy_concat_for_condition(entropy_stats_root, func_type, noise_type)
    order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]
    out: Dict[str, pd.DataFrame] = {}
    for name in order:
        out[name] = filter_entropy_rows_for_model(
            combined, name, sweep, fixed_beta2, fixed_rho
        )
    return out
