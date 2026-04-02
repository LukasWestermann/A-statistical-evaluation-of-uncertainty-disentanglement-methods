"""
Load per-model DataFrames from moment-matched batch Excel outputs for summary panels.

Expected layout (from ``recompute_entropy_moment_matched_batch_from_npz``):

    {stats_root}/sample_size/{noise_type}/{func_type}/*moment_matched_entropy_sample_size*.xlsx
    {stats_root}/noise_level/{noise_type}/{func_type}/*moment_matched_entropy_noise_{dist}*.xlsx
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from utils.regression_summary_panels import MODEL_KEYS

DISPLAY_NAMES = list(MODEL_KEYS.keys())


def _display_for_model_name(raw: str) -> Optional[str]:
    """Map ``model_name`` cell (or stem tag) to panel display name."""
    s = str(raw).strip()
    s_ = s.replace(" ", "_")
    for disp, key in MODEL_KEYS.items():
        if s == disp or s == key or s_ == key:
            return disp
    return None


def _pick_latest_per_display(paths: list[Path]) -> Dict[str, Path]:
    """Sort by name (date prefix); last file wins per display model."""
    by_disp: Dict[str, Path] = {}
    for p in sorted(paths):
        try:
            df = pd.read_excel(p, engine="openpyxl")
        except Exception:
            continue
        if df.empty:
            continue
        disp = _display_for_model_name(df["model_name"].iloc[0]) if "model_name" in df.columns else None
        if disp is None:
            continue
        by_disp[disp] = p
    return by_disp


def load_sample_size_entropy_moment_matched(
    stats_root: Path,
    func_type: str,
    noise_type: str,
) -> Dict[str, pd.DataFrame]:
    """
    Build ``stats_by_model`` dict for ``measure="entropy"`` summary panel functions.
    Returns {} if directory missing or no readable workbooks.
    """
    d = stats_root / "sample_size" / noise_type / func_type
    if not d.is_dir():
        return {}
    paths = [p for p in d.glob("*moment_matched_entropy_sample_size*.xlsx") if p.is_file()]
    if not paths:
        return {}
    latest = _pick_latest_per_display(paths)
    out: Dict[str, pd.DataFrame] = {}
    for disp in DISPLAY_NAMES:
        p = latest.get(disp)
        if p is None:
            continue
        try:
            df = pd.read_excel(p, engine="openpyxl")
        except Exception:
            continue
        if df.empty or "Percentage" not in df.columns:
            continue
        out[disp] = df
    return out


def load_noise_level_entropy_moment_matched(
    stats_root: Path,
    func_type: str,
    noise_type: str,
    distribution: str = "normal",
) -> Dict[str, pd.DataFrame]:
    """Same as sample-size loader, for τ sweeps and a given noise distribution."""
    d = stats_root / "noise_level" / noise_type / func_type
    if not d.is_dir():
        return {}
    needle = f"moment_matched_entropy_noise_{distribution}"
    paths = [p for p in d.glob("*.xlsx") if needle in p.name and p.is_file()]
    if not paths:
        return {}
    latest = _pick_latest_per_display(paths)
    out: Dict[str, pd.DataFrame] = {}
    for disp in DISPLAY_NAMES:
        p = latest.get(disp)
        if p is None:
            continue
        try:
            df = pd.read_excel(p, engine="openpyxl")
        except Exception:
            continue
        if df.empty or "Tau" not in df.columns:
            continue
        out[disp] = df
    return out
