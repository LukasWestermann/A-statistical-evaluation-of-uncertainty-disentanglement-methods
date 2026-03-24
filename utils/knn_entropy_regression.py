"""
K-NN entropy recomputation from saved regression raw_outputs .npz files.

Mirrors experiment normalization for entropy statistics (OOD, sample_size, noise_level)
and supports 2x4 panel plotting (variance from samples; entropy from k-NN).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.entropy_uncertainty import entropy_uncertainty_numerical
from utils.knn_entropy import entropy_uncertainty_knn_gaussian_mixture
from utils.results_save import sanitize_filename

CONDITIONS_4: List[Tuple[str, str]] = [
    ("linear", "homoscedastic"),
    ("linear", "heteroscedastic"),
    ("sin", "homoscedastic"),
    ("sin", "heteroscedastic"),
]

# (display_name, stem_tag, glob_candidates newest-last after sort)
MODEL_RESOLVERS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Deep Ensemble", "Deep_Ensemble", ("*Deep_Ensemble*pct100*raw_outputs*.npz", "*Deep_Ensemble*raw_outputs*.npz")),
    ("MC Dropout", "MC_Dropout", ("*MC_Dropout*pct100*raw_outputs*.npz", "*MC_Dropout*raw_outputs*.npz")),
    ("BNN", "BNN", ("*BNN*pct100*raw_outputs*.npz", "*BNN*raw_outputs*.npz")),
    ("BAMLSS", "BAMLSS", ("*BAMLSS*pct100*raw_outputs*.npz", "*BAMLSS*raw_outputs*.npz")),
]

DEFAULT_OOD_RANGES: List[Tuple[float, float]] = [(10.0, 15.0)]


def is_ovb_or_non_raw_path(path: Path) -> bool:
    """True if path should be skipped (OVB outputs or not raw_outputs)."""
    p = str(path).replace("\\", "/").lower()
    if "ovb_outputs_" in path.name.lower():
        return True
    if "/ovb/" in p or "\\ovb\\" in str(path).lower():
        return True
    if "raw_outputs" not in path.name:
        return True
    return False


def ensure_samples_first(mu_samples, sigma2_samples, x_grid) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu_samples)
    sig = np.asarray(sigma2_samples)
    n_grid = int(np.asarray(x_grid).ravel().shape[0])
    if mu.ndim == 2 and mu.shape[0] == n_grid and mu.shape[1] != n_grid:
        return mu.T, sig.T
    return mu, sig


def build_ood_mask(x_grid: np.ndarray, ood_ranges: Sequence[Tuple[float, float]]) -> np.ndarray:
    x_flat = np.asarray(x_grid).ravel()
    ood_mask = np.zeros(len(x_flat), dtype=bool)
    for lo, hi in ood_ranges:
        ood_mask |= (x_flat >= lo) & (x_flat <= hi)
    return ood_mask


def rng_for_npz(base_seed: int, npz_path: Path) -> np.random.Generator:
    h = hash(str(npz_path.resolve())) % (2**31)
    return np.random.default_rng(np.random.SeedSequence([base_seed, h]))


def int_seed_for_npz(base_seed: int, npz_path: Path) -> int:
    """Deterministic int seed for APIs that take int (e.g. MC mixture entropy)."""
    h = hash(str(npz_path.resolve())) % (2**31)
    return int((base_seed * 1_000_003 + h) % (2**31))


def function_display(func_type: str) -> str:
    return "Linear" if func_type == "linear" else "Sinusoidal"


def model_key_from_stem(stem: str) -> Optional[str]:
    for _disp, tag, _globs in MODEL_RESOLVERS:
        if tag in stem:
            return tag
    return None


def _npz_scalar_str(data: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    if key not in data.files:
        return default
    try:
        v = np.asarray(data[key]).ravel()[0]
        if hasattr(v, "item"):
            return str(v.item())
        return str(v)
    except Exception:
        return default


def _npz_scalar_float(data: Any, key: str) -> Optional[float]:
    if key not in data.files:
        return None
    try:
        return float(np.asarray(data[key]).ravel()[0])
    except Exception:
        return None


def parse_pct_from_stem(stem: str) -> Optional[float]:
    m = re.search(r"pct(\d+(?:\.\d+)?)", stem, re.I)
    if m:
        return float(m.group(1))
    return None


def parse_tau_from_stem(stem: str) -> Optional[float]:
    m = re.search(r"tau(\d+(?:\.\d+)?)", stem, re.I)
    if m:
        return float(m.group(1))
    return None


def read_npz_metadata(npz_path: Path, data: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "model_name": _npz_scalar_str(data, "model_name"),
        "pct": _npz_scalar_float(data, "pct"),
        "tau": _npz_scalar_float(data, "tau"),
        "distribution": _npz_scalar_str(data, "distribution", "normal"),
        "dropout_p": _npz_scalar_float(data, "dropout_p"),
        "mc_samples": _npz_scalar_float(data, "mc_samples"),
        "n_nets": _npz_scalar_float(data, "n_nets"),
    }
    if meta["pct"] is None:
        meta["pct"] = parse_pct_from_stem(npz_path.stem)
    if meta["tau"] is None:
        meta["tau"] = parse_tau_from_stem(npz_path.stem)
    return meta


def model_prefix_for_filename(model_name: str, dropout_p: Optional[float], mc_samples: Optional[float], n_nets: Optional[float]) -> str:
    if not model_name:
        return ""
    param_parts: List[str] = []
    if model_name == "MC_Dropout":
        if dropout_p is not None:
            param_parts.append(f"p{dropout_p}")
        if mc_samples is not None:
            param_parts.append(f"M{int(mc_samples)}")
    elif model_name == "Deep_Ensemble":
        if n_nets is not None:
            param_parts.append(f"K{int(n_nets)}")
    if param_parts:
        return f"{model_name}_{'_'.join(param_parts)}"
    return model_name


def apply_stride(
    mu: np.ndarray,
    sig: np.ndarray,
    x_grid: np.ndarray,
    y_clean: np.ndarray,
    grid_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid_stride <= 1:
        return mu, sig, x_grid, y_clean
    return (
        mu[:, ::grid_stride],
        sig[:, ::grid_stride],
        np.asarray(x_grid)[::grid_stride],
        np.asarray(y_clean)[::grid_stride],
    )


@dataclass
class KnnGridResult:
    mu_samples: np.ndarray
    sigma2_samples: np.ndarray
    x_grid: np.ndarray
    y_grid_clean: np.ndarray
    mu_pred: np.ndarray
    ale_var: np.ndarray
    epi_var: np.ndarray
    ale_entropy: np.ndarray
    epi_entropy: np.ndarray
    tot_entropy: np.ndarray
    ood_mask: np.ndarray
    id_mask: np.ndarray
    x: np.ndarray
    y_clean_flat: np.ndarray
    boundary_x: List[float]
    x_train_flat: Optional[np.ndarray]
    y_train_flat: Optional[np.ndarray]
    meta: Dict[str, Any]


def compute_knn_grid_result(
    npz_path: Path,
    L: int,
    k_nn: int,
    base_seed: int,
    ood_ranges: Sequence[Tuple[float, float]],
    grid_stride: int = 1,
) -> KnnGridResult:
    data = np.load(npz_path, allow_pickle=True)
    mu_samples = np.asarray(data["mu_samples"])
    sigma2_samples = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    y_grid_clean = np.asarray(data["y_grid_clean"])
    mu_samples, sigma2_samples = ensure_samples_first(mu_samples, sigma2_samples, x_grid)
    mu_samples, sigma2_samples, x_grid, y_grid_clean = apply_stride(
        mu_samples, sigma2_samples, x_grid, y_grid_clean, grid_stride
    )

    meta = read_npz_metadata(npz_path, data)
    rng = rng_for_npz(base_seed, npz_path)
    ent = entropy_uncertainty_knn_gaussian_mixture(mu_samples, sigma2_samples, L=L, k_nn=k_nn, rng=rng)
    ale_entropy = np.asarray(ent["aleatoric"]).squeeze()
    epi_entropy = np.asarray(ent["epistemic"]).squeeze()
    tot_entropy = np.asarray(ent["total"]).squeeze()

    mu_pred = np.mean(mu_samples, axis=0).squeeze()
    ale_var = np.mean(sigma2_samples, axis=0).squeeze()
    epi_var = np.var(mu_samples, axis=0).squeeze()

    ood_mask = build_ood_mask(x_grid, ood_ranges)
    id_mask = ~ood_mask
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean.ravel()

    boundary_x: List[float] = []
    if np.any(ood_mask):
        transitions = np.where(np.diff(ood_mask.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = list(x[transitions + 1])

    x_train_flat = y_train_flat = None
    if "x_train_subset" in data.files and "y_train_subset" in data.files:
        xt = np.asarray(data["x_train_subset"])
        yt = np.asarray(data["y_train_subset"])
        x_train_flat = xt[:, 0] if xt.ndim > 1 else xt.ravel()
        y_train_flat = yt[:, 0] if yt.ndim > 1 else yt.ravel()

    return KnnGridResult(
        mu_samples=mu_samples,
        sigma2_samples=sigma2_samples,
        x_grid=x_grid,
        y_grid_clean=y_grid_clean,
        mu_pred=mu_pred,
        ale_var=ale_var,
        epi_var=epi_var,
        ale_entropy=ale_entropy,
        epi_entropy=epi_entropy,
        tot_entropy=tot_entropy,
        ood_mask=ood_mask,
        id_mask=id_mask,
        x=x,
        y_clean_flat=y_clean_flat,
        boundary_x=boundary_x,
        x_train_flat=x_train_flat,
        y_train_flat=y_train_flat,
        meta=meta,
    )


def compute_numerical_grid_result(
    npz_path: Path,
    n_samples: int,
    base_seed: int,
    ood_ranges: Sequence[Tuple[float, float]],
    grid_stride: int = 1,
    grid_chunk_size: Optional[int] = None,
) -> KnnGridResult:
    """Same contract as compute_knn_grid_result; entropy from MC mixture (numerical)."""
    data = np.load(npz_path, allow_pickle=True)
    mu_samples = np.asarray(data["mu_samples"])
    sigma2_samples = np.asarray(data["sigma2_samples"])
    x_grid = np.asarray(data["x_grid"])
    y_grid_clean = np.asarray(data["y_grid_clean"])
    mu_samples, sigma2_samples = ensure_samples_first(mu_samples, sigma2_samples, x_grid)
    mu_samples, sigma2_samples, x_grid, y_grid_clean = apply_stride(
        mu_samples, sigma2_samples, x_grid, y_grid_clean, grid_stride
    )

    meta = read_npz_metadata(npz_path, data)
    seed = int_seed_for_npz(base_seed, npz_path)
    ent = entropy_uncertainty_numerical(
        mu_samples,
        sigma2_samples,
        n_samples=n_samples,
        seed=seed,
        grid_chunk_size=grid_chunk_size,
    )
    ale_entropy = np.asarray(ent["aleatoric"]).squeeze()
    epi_entropy = np.asarray(ent["epistemic"]).squeeze()
    tot_entropy = np.asarray(ent["total"]).squeeze()

    mu_pred = np.mean(mu_samples, axis=0).squeeze()
    ale_var = np.mean(sigma2_samples, axis=0).squeeze()
    epi_var = np.var(mu_samples, axis=0).squeeze()

    ood_mask = build_ood_mask(x_grid, ood_ranges)
    id_mask = ~ood_mask
    x = x_grid[:, 0] if x_grid.ndim > 1 else x_grid.ravel()
    y_clean_flat = y_grid_clean[:, 0] if y_grid_clean.ndim > 1 else y_grid_clean.ravel()

    boundary_x: List[float] = []
    if np.any(ood_mask):
        transitions = np.where(np.diff(ood_mask.astype(int)) != 0)[0]
        if len(transitions) > 0:
            boundary_x = list(x[transitions + 1])

    x_train_flat = y_train_flat = None
    if "x_train_subset" in data.files and "y_train_subset" in data.files:
        xt = np.asarray(data["x_train_subset"])
        yt = np.asarray(data["y_train_subset"])
        x_train_flat = xt[:, 0] if xt.ndim > 1 else xt.ravel()
        y_train_flat = yt[:, 0] if yt.ndim > 1 else yt.ravel()

    return KnnGridResult(
        mu_samples=mu_samples,
        sigma2_samples=sigma2_samples,
        x_grid=x_grid,
        y_grid_clean=y_grid_clean,
        mu_pred=mu_pred,
        ale_var=ale_var,
        epi_var=epi_var,
        ale_entropy=ale_entropy,
        epi_entropy=epi_entropy,
        tot_entropy=tot_entropy,
        ood_mask=ood_mask,
        id_mask=id_mask,
        x=x,
        y_clean_flat=y_clean_flat,
        boundary_x=boundary_x,
        x_train_flat=x_train_flat,
        y_train_flat=y_train_flat,
        meta=meta,
    )


def knn_result_to_panel_dict(res: KnnGridResult) -> Dict[str, Any]:
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


def region_mse(mu_pred: np.ndarray, y_true: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean((mu_pred[mask] - y_true[mask]) ** 2))


def normalized_entropy_stats_ood_regions(
    ale: np.ndarray,
    epi: np.ndarray,
    tot: np.ndarray,
    mu_pred: np.ndarray,
    y_true: np.ndarray,
    id_mask: np.ndarray,
    ood_mask: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Same normalization as compute_and_save_statistics_entropy_ood (per-region summaries)."""

    def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    all_ale = ale
    all_epi = epi
    ale_min, ale_max = float(all_ale.min()), float(all_ale.max())
    epi_min, epi_max = float(all_epi.min()), float(all_epi.max())

    regions = [
        ("ID", id_mask),
        ("OOD", ood_mask),
        ("Combined", np.ones_like(id_mask, dtype=bool)),
    ]
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in regions:
        ale_vals = ale[mask]
        epi_vals = epi[mask]
        ale_norm = normalize(ale_vals, ale_min, ale_max)
        epi_norm = normalize(epi_vals, epi_min, epi_max)
        tot_norm = ale_norm + epi_norm
        corr = np.corrcoef(epi_vals, ale_vals)[0, 1] if ale_vals.size > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0
        out[name] = {
            "Avg_Aleatoric_Entropy_norm": float(np.mean(ale_norm)),
            "Avg_Epistemic_Entropy_norm": float(np.mean(epi_norm)),
            "Avg_Total_Entropy_norm": float(np.mean(tot_norm)),
            "Correlation_Epi_Ale": float(corr),
            "MSE": region_mse(mu_pred, y_true, mask),
        }
    return out


def dataframe_ood_knn_three_regions(
    stats_by_region: Dict[str, Dict[str, float]],
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    date: str,
    dropout_p: Optional[float],
    mc_samples: Optional[float],
    n_nets: Optional[float],
    recompute_note: str = "NLL/CRPS/Spearman not in raw_outputs npz",
) -> pd.DataFrame:
    """Rows match save_summary_statistics_entropy_ood columns (one row per region)."""
    rows = []
    for region in ("ID", "OOD", "Combined"):
        s = stats_by_region[region]
        row = {
            "Region": region,
            "Avg_Aleatoric_Entropy_norm": s["Avg_Aleatoric_Entropy_norm"],
            "Avg_Epistemic_Entropy_norm": s["Avg_Epistemic_Entropy_norm"],
            "Avg_Total_Entropy_norm": s["Avg_Total_Entropy_norm"],
            "Correlation_Epi_Ale": s["Correlation_Epi_Ale"],
            "MSE": s["MSE"],
            "NLL": np.nan,
            "CRPS": np.nan,
            "Spearman_Aleatoric": np.nan,
            "Spearman_Epistemic": np.nan,
            "knn_recompute_note": recompute_note,
            "function_name": function_name,
            "noise_type": noise_type,
            "func_type": func_type,
            "model_name": model_name,
            "date": date,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_sample_size_entropy_dataframe(
    pct_to_ent: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pct_to_mse: Dict[float, float],
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    date: str,
    dropout_p: Optional[float],
    mc_samples: Optional[float],
    n_nets: Optional[float],
    recompute_note: str = "from raw_outputs k-NN",
) -> pd.DataFrame:
    """Mirror compute_and_save_statistics_entropy normalization across percentages."""
    percentages = sorted(pct_to_ent.keys())
    if not percentages:
        return pd.DataFrame()

    def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    all_ale = np.concatenate([pct_to_ent[p][0].ravel() for p in percentages])
    all_epi = np.concatenate([pct_to_ent[p][1].ravel() for p in percentages])
    ale_min, ale_max = float(all_ale.min()), float(all_ale.max())
    epi_min, epi_max = float(all_epi.min()), float(all_epi.max())

    rows = []
    for pct in percentages:
        ale, epi, tot = pct_to_ent[pct]
        ale_v = ale.ravel()
        epi_v = epi.ravel()
        ale_norm = normalize(ale_v, ale_min, ale_max)
        epi_norm = normalize(epi_v, epi_min, epi_max)
        tot_norm = ale_norm + epi_norm
        corr = np.corrcoef(epi_v, ale_v)[0, 1] if ale_v.size > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0
        rows.append({
            "Percentage": pct,
            "Avg_Aleatoric_Entropy_norm": float(np.mean(ale_norm)),
            "Avg_Epistemic_Entropy_norm": float(np.mean(epi_norm)),
            "Avg_Total_Entropy_norm": float(np.mean(tot_norm)),
            "Correlation_Epi_Ale": float(corr),
            "MSE": pct_to_mse.get(pct, float("nan")),
            "knn_recompute_note": "from raw_outputs k-NN",
            "model_name": model_name,
            "date": date,
        })
    return pd.DataFrame(rows)


def build_noise_level_entropy_dataframe(
    tau_to_ent: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    tau_to_mse: Dict[float, float],
    distribution: str,
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    date: str,
    dropout_p: Optional[float],
    mc_samples: Optional[float],
    n_nets: Optional[float],
    recompute_note: str = "from raw_outputs k-NN",
) -> pd.DataFrame:
    taus = sorted(tau_to_ent.keys())
    if not taus:
        return pd.DataFrame()

    def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    all_ale = np.concatenate([tau_to_ent[t][0].ravel() for t in taus])
    all_epi = np.concatenate([tau_to_ent[t][1].ravel() for t in taus])
    ale_min, ale_max = float(all_ale.min()), float(all_ale.max())
    epi_min, epi_max = float(all_epi.min()), float(all_epi.max())

    rows = []
    for tau in taus:
        ale, epi, tot = tau_to_ent[tau]
        ale_v = ale.ravel()
        epi_v = epi.ravel()
        ale_norm = normalize(ale_v, ale_min, ale_max)
        epi_norm = normalize(epi_v, epi_min, epi_max)
        tot_norm = ale_norm + epi_norm
        corr = np.corrcoef(epi_v, ale_v)[0, 1] if ale_v.size > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0
        rows.append({
            "Tau": tau,
            "Distribution": distribution,
            "Avg_Aleatoric_Entropy_norm": float(np.mean(ale_norm)),
            "Avg_Epistemic_Entropy_norm": float(np.mean(epi_norm)),
            "Avg_Total_Entropy_norm": float(np.mean(tot_norm)),
            "Correlation_Epi_Ale": float(corr),
            "MSE": tau_to_mse.get(tau, float("nan")),
            "knn_recompute_note": "from raw_outputs k-NN",
            "model_name": model_name,
            "date": date,
        })
    return pd.DataFrame(rows)


def build_undersampling_approx_dataframe(
    ale: np.ndarray,
    epi: np.ndarray,
    tot: np.ndarray,
    mu_pred: np.ndarray,
    y_true: np.ndarray,
    function_name: str,
    noise_type: str,
    func_type: str,
    model_name: str,
    date: str,
    dropout_p: Optional[float],
    mc_samples: Optional[float],
    n_nets: Optional[float],
    recompute_note: str = "undersampling npz has single grid; not split by sampling regions",
) -> pd.DataFrame:
    """Single full-grid 'region' — undersampling multi-region not in npz."""
    ale_min, ale_max = float(ale.min()), float(ale.max())
    epi_min, epi_max = float(epi.min()), float(epi.max())

    def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    ale_norm = normalize(ale, ale_min, ale_max)
    epi_norm = normalize(epi, epi_min, epi_max)
    tot_norm = ale_norm + epi_norm
    corr = np.corrcoef(epi, ale)[0, 1] if ale.size > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    mse = float(np.mean((mu_pred - y_true) ** 2))
    return pd.DataFrame([{
        "Region": "full_grid_approx",
        "Avg_Aleatoric_Entropy_norm": float(np.mean(ale_norm)),
        "Avg_Epistemic_Entropy_norm": float(np.mean(epi_norm)),
        "Avg_Total_Entropy_norm": float(np.mean(tot_norm)),
        "Correlation_Epi_Ale": float(corr),
        "MSE": mse,
        "regions_approximated": True,
        "knn_recompute_note": recompute_note,
        "model_name": model_name,
        "date": date,
    }])


def resolve_latest_npz(search_dir: Path, globs: Tuple[str, ...]) -> Optional[Path]:
    found: List[Path] = []
    for g in globs:
        found.extend(search_dir.glob(g))
    if not found:
        for g in globs:
            found.extend(search_dir.rglob(g))
    found = [p for p in found if not is_ovb_or_non_raw_path(p)]
    if not found:
        return None
    return sorted({p.resolve() for p in found})[-1]


def collect_raw_npz_files(search_dir: Path) -> List[Path]:
    if not search_dir.exists():
        return []
    out = []
    for p in search_dir.rglob("*raw_outputs*.npz"):
        if not is_ovb_or_non_raw_path(p):
            out.append(p.resolve())
    return sorted(set(out))


def save_stats_excel(df: pd.DataFrame, out_dir: Path, filename_stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sanitize_filename(filename_stem)}.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# --- Plotting (2x4 panels, configurable OOD ranges) ---

def shade_ood(ax, ood_ranges: Sequence[Tuple[float, float]]) -> None:
    for lo, hi in ood_ranges:
        ax.axvspan(lo, hi, alpha=0.35, color="lightgrey", zorder=0)


def add_common_variance(ax, data: Dict[str, Any], ood_ranges: Sequence[Tuple[float, float]]) -> None:
    shade_ood(ax, ood_ranges)
    if data.get("x_train_flat") is not None and data.get("y_train_flat") is not None:
        ax.scatter(
            data["x_train_flat"], data["y_train_flat"],
            alpha=0.15, s=15, color="#2E86AB", zorder=3, edgecolors="none",
        )
    for bx in data["boundary_x"]:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    x, y_clean = data["x"], data["y_clean_flat"]
    id_mask, ood_mask = data["id_mask"], data["ood_mask"]
    ax.plot(x[id_mask], y_clean[id_mask], "r--", linewidth=2, alpha=0.9, label="True function", zorder=4)
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean[ood_mask], "r--", linewidth=2, alpha=0.9, zorder=4)
        ax.scatter(x[ood_mask], y_clean[ood_mask], s=20, color="red", alpha=0.4, marker="x", zorder=6, linewidths=1.5)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=9)


def create_2x4_variance_panel(
    condition_data_list: List[Optional[Dict[str, Any]]],
    display_names: Sequence[str],
    func_type: str,
    noise_type: str,
    save_path: Path,
    experiment_title: str,
    ood_ranges: Sequence[Tuple[float, float]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    func_title = function_display(func_type)
    noise_title = "homoscedastic" if noise_type == "homoscedastic" else "heteroscedastic"

    for row, (band_key_ale, band_key_epi, color, label) in enumerate([
        ("ale_var", None, "#06A77D", "±σ(aleatoric)"),
        (None, "epi_var", "#F18F01", "±σ(epistemic)"),
    ]):
        for col in range(4):
            ax = axes[row, col]
            if col >= len(condition_data_list) or condition_data_list[col] is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                ax.set_ylabel("y", fontsize=10)
                if row == 0:
                    ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
                continue
            data = condition_data_list[col]
            x = data["x"]
            mu_pred = data["mu_pred"]
            var = data[band_key_ale] if band_key_ale else data[band_key_epi]
            ax.plot(x, mu_pred, "b-", linewidth=2, label="Predictive mean", zorder=5)
            ax.fill_between(x, mu_pred - np.sqrt(var), mu_pred + np.sqrt(var), alpha=0.35, color=color, label=label, zorder=1)
            add_common_variance(ax, data, ood_ranges)
            ax.set_ylabel("y", fontsize=10)
            if row == 0:
                ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    for col in range(4):
        axes[1, col].set_xlabel("x", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("y\n(Aleatoric)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("y\n(Epistemic)", fontsize=10, fontweight="bold")
    fig.suptitle(f"{experiment_title} — {func_title}, {noise_title} — Variance (std bands)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_common_entropy(ax, ax_twin, data: Dict[str, Any], entropy_color: str, ood_ranges: Sequence[Tuple[float, float]]) -> None:
    shade_ood(ax, ood_ranges)
    if data.get("x_train_flat") is not None and data.get("y_train_flat") is not None:
        ax.scatter(data["x_train_flat"], data["y_train_flat"], alpha=0.1, s=10, color="blue", zorder=3)
    for bx in data["boundary_x"]:
        ax.axvline(x=bx, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
    x, y_clean = data["x"], data["y_clean_flat"]
    id_mask, ood_mask = data["id_mask"], data["ood_mask"]
    ax.plot(x[id_mask], data["mu_pred"][id_mask], "b-", linewidth=1.2, alpha=0.5, label="Predictive mean")
    ax.plot(x[ood_mask], data["mu_pred"][ood_mask], "b-", linewidth=1.2, alpha=0.5)
    ax.plot(x[id_mask], y_clean[id_mask], "r-", linewidth=1.5, alpha=0.8, label="True function")
    if np.any(ood_mask):
        ax.plot(x[ood_mask], y_clean[ood_mask], "r-", linewidth=1.5, alpha=0.8)
        ax.scatter(x[ood_mask], y_clean[ood_mask], s=8, color="red", alpha=0.3, marker="x", zorder=4)
    ax_twin.tick_params(axis="y", labelcolor=entropy_color)
    ax.grid(True, alpha=0.3)


def create_2x4_entropy_panel(
    condition_data_list: List[Optional[Dict[str, Any]]],
    display_names: Sequence[str],
    func_type: str,
    noise_type: str,
    save_path: Path,
    experiment_title: str,
    ood_ranges: Sequence[Tuple[float, float]],
    entropy_subtitle: str = "Entropy k-NN (line plots)",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    func_title = function_display(func_type)
    noise_title = "homoscedastic" if noise_type == "homoscedastic" else "heteroscedastic"

    for row, (ent_key, color, label) in enumerate([
        ("ale_entropy", "green", "Aleatoric entropy (nats)"),
        ("epi_entropy", "#C41E3A", "Epistemic entropy (nats)"),
    ]):
        for col in range(4):
            ax = axes[row, col]
            if col >= len(condition_data_list) or condition_data_list[col] is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                if row == 0:
                    ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
                continue
            data = condition_data_list[col]
            ax_twin = ax.twinx()
            x = data["x"]
            ent = data[ent_key]
            id_mask, ood_mask = data["id_mask"], data["ood_mask"]
            ax_twin.plot(x[id_mask], ent[id_mask], "-", color=color, linewidth=2, label=label)
            ax_twin.plot(x[ood_mask], ent[ood_mask], "-", color=color, linewidth=2, alpha=0.7)
            add_common_entropy(ax, ax_twin, data, color, ood_ranges)
            ax.set_ylabel("y", fontsize=10)
            ax_twin.set_ylabel("Entropy (nats)", fontsize=9, color=color)
            if row == 0:
                ax.set_title(display_names[col], fontweight="bold", fontsize=11, pad=6)
            ax.legend(loc="upper left", fontsize=8)
            ax_twin.legend(loc="upper right", fontsize=8)

    for col in range(4):
        axes[1, col].set_xlabel("x", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("y\n(Aleatoric)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("y\n(Epistemic)", fontsize=10, fontweight="bold")
    fig.suptitle(f"{experiment_title} — {func_title}, {noise_title} — {entropy_subtitle}", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_lines(
    x: np.ndarray,
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    total: np.ndarray,
    save_path: Optional[Path],
    title: Optional[str],
    ood_ranges: Sequence[Tuple[float, float]],
) -> List[Path]:
    import matplotlib.pyplot as plt

    base_title = title or "Entropy from npz (k-NN / Eq. 6)"
    curves = [
        (aleatoric, "green", "Aleatoric (nats)", "Aleatoric"),
        (epistemic, "#C41E3A", "Epistemic (nats)", "Epistemic"),
        (total, "blue", "Total (nats)", "Total"),
    ]
    suffixes = ["aleatoric", "epistemic", "total"]
    saved: List[Path] = []
    for (y, color, ylabel, sub_title), suffix in zip(curves, suffixes):
        fig, ax = plt.subplots(figsize=(10, 5))
        for lo, hi in ood_ranges:
            ax.axvspan(lo, hi, alpha=0.35, color="lightgrey", zorder=0, label="OOD")
        ax.plot(x, y, color=color, linewidth=1.5, label=ylabel)
        ax.set_xlabel("x")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title(f"{base_title} — {sub_title}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            stem, ext = save_path.stem, save_path.suffix
            out = save_path.parent / f"{stem}_{suffix}{ext}"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            saved.append(out)
        plt.close(fig)
    return saved
