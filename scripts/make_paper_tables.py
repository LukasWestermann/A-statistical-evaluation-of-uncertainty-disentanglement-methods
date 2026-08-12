"""
Build NeurIPS-style LaTeX result tables for the EIML3 workshop paper from raw
simulation .npz outputs.

Reads raw per-grid-point ``mu_samples``/``sigma2_samples`` and RECOMPUTES every
number -- never reformats a pre-aggregated summary file. AU (aleatoric) =
mean(sigma2_samples, axis=0); EU (epistemic) = var(mu_samples, axis=0) --
variance-based decomposition only (law of total variance: TU = AU + EU, TU is not
reported per the task scope).

Scope:
    MODELS         -- (npz model tag, display name) pairs actually used for a run.
                       Choose via --models (comma-separated tags, e.g.
                       "Deep_Ensemble,BAMLSS,MC_Dropout,BNN"); defaults to
                       "Deep_Ensemble,BAMLSS" if omitted. Any of the 4 known models
                       (see CANDIDATE_MODELS) can be included in any combination and
                       order -- a model/cell with no saved data simply renders as
                       "---", never an error or a fabricated value. Note: as of this
                       writing MC Dropout has zero saved results anywhere in the
                       repo and BNN has exactly one file (OOD, homoscedastic/linear)
                       -- selecting them is fully supported, just mostly "---" until
                       more runs exist. Region tables (OOD/Undersampling) scale to
                       any number of models for free (models are rows); sweep
                       tables (Noise-level/OVB) grow by 3 columns per model and may
                       exceed the NeurIPS single-column width budget at 4 models --
                       a heuristic warning is printed when that happens (see
                       estimate_sweep_table_width_pt), not auto-shrunk.
    DECOMPOSITION  -- fixed at "variance"; entropy-based columns are out of scope.

Metric definitions (must match the thesis):
    Mean AU / EU:  arithmetic mean of the pointwise estimate over the relevant
                   region (OOD/undersampling) or the relevant knob value's own grid
                   (sample-size/noise/OVB).
    Min-max normalisation, PER uncertainty type (AU, EU normalised separately), PER
    model, PER experiment:
        u~(x_i) = (u(x_i) - min_j u(x_j)) / (max_j u(x_j) - min_j u(x_j))
    with min/max taken over the FULL POOLED SCOPE of the experiment for that
    (model, DGP) -- OOD: the union of ID and OOD grid points (one grid, already
    spans both); undersampling: the one grid (spans both regions); sample-size:
    all training shares pooled; noise: all tau pooled; OVB: all knob values in that
    sweep pooled. Normalisation happens FIRST over the pooled scope, THEN the
    result is aggregated (mean) within each region/knob subset -- reversing this
    order silently changes every number.
    rho:  Pearson correlation between pointwise AU and EU, computed on the SAME
          points used for the corresponding AU/EU mean, using RAW (non-normalised)
          values -- min-max normalisation is monotone-affine and does not change
          Pearson r, so computing on the raw scale is equivalent and simpler.

Seed dimension: NONE exists anywhere in the raw data (confirmed by direct
inspection of every npz across all 5 experiments -- save_model_outputs has no seed
parameter, and no npz has a seed key). Every table therefore reports plain values,
never mean +/- sd. Where two dated files exist for the same (model, DGP, knob) cell
(re-runs on different days, not independent seeds -- confirmed byte-identical where
checked), the latest date is used.

Missing data -> "---", always. Never fabricated, interpolated, or silently dropped.

Also emits 9 companion tables (OOD/Undersampling/Sample-size/Noise-level only, no
OVB) alongside the 6 tables above: absolute-scale (raw, non-normalised) AU/EU
mean+-sd and epistemic share (tab_*_absolute.tex), and AU-vs-true-noise-variance
diagnostics -- bias/slope/Pearson/Spearman of AU(x) vs. sigma^2(x)
(tab_*_autruth.tex) -- plus a small noise-only across-sweep AU-vs-tau correlation
table (tab_noise_au_sweep_corr.tex). See tables/README.md's "Companion tables"
section for full detail (sigma^2(x) reconstruction, homoscedastic '---' handling,
number-format rationale).

Usage:
    python scripts/make_paper_tables.py --out tables/
    python scripts/make_paper_tables.py --out tables/ --models Deep_Ensemble,BAMLSS,MC_Dropout,BNN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (  # noqa: E402
    build_ood_mask,
    ensure_samples_first,
    resolve_latest_npz,
    resolve_latest_npz_at_pct,
    resolve_latest_npz_at_tau,
)

# ============================== Scope (flip here, or override --models at the CLI) ==============================

# All models this script knows how to load (npz tag -> display name). Which subset
# actually gets used for a given run is chosen via --models (see build_arg_parser);
# MODELS below is just the default when --models is omitted.
CANDIDATE_MODELS: List[Tuple[str, str]] = [
    ("Deep_Ensemble", "Deep Ensemble"),
    ("MC_Dropout", "MC Dropout"),
    ("BNN", "BNN"),
    ("BAMLSS", "BAMLSS"),
]
CANDIDATE_MODEL_TAGS = [tag for tag, _ in CANDIDATE_MODELS]
CANDIDATE_MODEL_DISPLAY = dict(CANDIDATE_MODELS)

MODELS: List[Tuple[str, str]] = [
    ("Deep_Ensemble", "Deep Ensemble"),
    ("BAMLSS", "BAMLSS"),
]  # default scope; overwritten in main() from --models
DECOMPOSITION = "variance"  # only option currently implemented

DGPS: List[Tuple[str, str, str]] = [
    # (func_type, noise_type, short label -- thesis ordering, never reordered)
    ("linear", "homoscedastic", "Lin-Hom"),
    ("linear", "heteroscedastic", "Lin-Het"),
    ("sin", "homoscedastic", "Sin-Hom"),
    ("sin", "heteroscedastic", "Sin-Het"),
]

OOD_RANGES = [(10.0, 15.0)]  # matches DEFAULT_OOD_RANGES / train_range=(-5,10)

# Empirically verified against the actual saved x_train_subset point counts in
# every results/undersampling/*.npz file (551/6/441 out of 998, matching a
# proportional-density model almost exactly) -- NOT the same as the stale default
# in utils/undersampling_experiments.py (dead code, never actually used for the
# saved runs) nor scripts/export_undersampling_overview_latex.py's
# DEFAULT_SAMPLING_REGIONS (also does not match; see tables/README.md).
UNDERSAMPLING_REGIONS: List[Tuple[Tuple[float, float], float]] = [
    ((-5, 0), 1.0),
    ((0, 6), 0.01),
    ((6, 10), 1.0),
]

NOISE_TAUS: List[float] = [0.5, 1.0, 2.0, 2.5, 5.0]  # all values found on disk

SAMPLE_SIZE_PCTS: List[float] = [5.0, 10.0, 25.0, 50.0, 100.0]  # all values found on disk

OVB_BETA2_VALUES: List[float] = [0.0, 0.5, 1.0, 2.0]
OVB_FIXED_RHO = 0.75
OVB_RHO_VALUES: List[float] = [0.0, 0.25, 0.75, 1.0]
OVB_FIXED_BETA2 = 1.0
OVB_DIR_FOR_MODEL = {
    "Deep_Ensemble": "deep_ensemble",
    "MC_Dropout": "mc_dropout",
    "BNN": "bnn",
    "BAMLSS": "bamlss",
}  # mc_dropout/bnn subdirectories don't exist on disk yet -- resolve_ovb_npz
   # no-ops gracefully via `if not search_dir.exists(): return None`, no crash.

RESULTS_ROOT = project_root / "results"


# ============================== Metric pipeline ==============================

def _normalize(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def _rho(ale_raw: np.ndarray, epi_raw: np.ndarray) -> float:
    if ale_raw.size < 2 or np.std(ale_raw) == 0 or np.std(epi_raw) == 0:
        return 0.0
    r = pearsonr(ale_raw, epi_raw)[0]
    return 0.0 if np.isnan(r) else float(r)


def normalize_pool_aggregate(pooled_ale: np.ndarray, pooled_epi: np.ndarray,
                              subset_masks: Dict[str, np.ndarray]) -> Dict[str, Optional[dict]]:
    """Normalize AU/EU separately over the pooled scope, THEN aggregate (mean) +
    compute rho (on raw values) within each named subset. Returns
    {subset_name: {'AU':.., 'EU':.., 'rho':..} or None (empty subset)}."""
    ale_min, ale_max = float(pooled_ale.min()), float(pooled_ale.max())
    epi_min, epi_max = float(pooled_epi.min()), float(pooled_epi.max())
    ale_n = _normalize(pooled_ale, ale_min, ale_max)
    epi_n = _normalize(pooled_epi, epi_min, epi_max)

    out: Dict[str, Optional[dict]] = {}
    for name, mask in subset_masks.items():
        if not np.any(mask):
            out[name] = None
            continue
        out[name] = {
            "AU": float(np.mean(ale_n[mask])),
            "EU": float(np.mean(epi_n[mask])),
            "rho": _rho(pooled_ale[mask], pooled_epi[mask]),
        }
    return out


def load_ale_epi(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x, ale_var, epi_var), all 1D, at every x_grid point."""
    d = np.load(npz_path, allow_pickle=True)
    mu, sig = ensure_samples_first(d["mu_samples"], d["sigma2_samples"], d["x_grid"])
    ale = np.mean(sig, axis=0).ravel()
    epi = np.var(mu, axis=0).ravel()
    x = np.asarray(d["x_grid"]).ravel()
    return x, ale, epi


# ============================== Per-experiment cell computation ==============================

def search_dir_standard(experiment: str, noise_type: str, func_type: str, extra: str = "") -> Path:
    return RESULTS_ROOT / experiment / "outputs" / experiment / noise_type / func_type / extra


def compute_region_table(experiment: str, region_masker) -> Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]]:
    """Shared logic for OOD / Undersampling: one file per (model, DGP), pooled
    scope = that single grid, split into two named regions by `region_masker(x)`
    -> {region_name: bool_mask}. Returns {(func_type,noise_type): {model_tag: stats_by_region}}."""
    out: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        search_dir = search_dir_standard(experiment, noise_type, func_type)
        per_model: Dict[str, Dict[str, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            npz_path = resolve_latest_npz(search_dir, (f"*{model_tag}*raw_outputs*.npz",)) if search_dir.exists() else None
            if npz_path is None:
                per_model[model_tag] = {"A": None, "B": None}
                continue
            x, ale, epi = load_ale_epi(npz_path)
            masks = region_masker(x)
            per_model[model_tag] = normalize_pool_aggregate(ale, epi, masks)
        out[(func_type, noise_type)] = per_model
    return out


def ood_region_masker(x: np.ndarray) -> Dict[str, np.ndarray]:
    ood_mask = build_ood_mask(x, OOD_RANGES)
    return {"A": ~ood_mask, "B": ood_mask}  # A=ID, B=OOD


def undersampling_region_masker(x: np.ndarray) -> Dict[str, np.ndarray]:
    under = np.zeros_like(x, dtype=bool)
    well = np.zeros_like(x, dtype=bool)
    for (lo, hi), density in UNDERSAMPLING_REGIONS:
        inside = (x >= lo) & (x <= hi)
        if density < 0.5:
            under |= inside
        else:
            well |= inside
    return {"A": under, "B": well}  # A=under-sampled, B=well-sampled


def compute_noise_table() -> Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]]:
    """{(func_type,noise_type): {model_tag: {tau: stats or None}}}, pooled over
    whichever tau values actually have data for that (model, DGP)."""
    out: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        search_dir = search_dir_standard("noise_level", noise_type, func_type, extra="normal")
        per_model: Dict[str, Dict[float, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            raw_by_tau: Dict[float, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
            for tau in NOISE_TAUS:
                npz_path = resolve_latest_npz_at_tau(search_dir, model_tag, tau) if search_dir.exists() else None
                if npz_path is None:
                    raw_by_tau[tau] = None
                    continue
                _x, ale, epi = load_ale_epi(npz_path)
                raw_by_tau[tau] = (ale, epi)
            present = {t: v for t, v in raw_by_tau.items() if v is not None}
            if not present:
                per_model[model_tag] = {t: None for t in NOISE_TAUS}
                continue
            pooled_ale = np.concatenate([v[0] for v in present.values()])
            pooled_epi = np.concatenate([v[1] for v in present.values()])
            ale_min, ale_max = float(pooled_ale.min()), float(pooled_ale.max())
            epi_min, epi_max = float(pooled_epi.min()), float(pooled_epi.max())
            result: Dict[float, Optional[dict]] = {}
            for t in NOISE_TAUS:
                if raw_by_tau[t] is None:
                    result[t] = None
                    continue
                ale, epi = raw_by_tau[t]
                ale_n = _normalize(ale, ale_min, ale_max)
                epi_n = _normalize(epi, epi_min, epi_max)
                result[t] = {"AU": float(np.mean(ale_n)), "EU": float(np.mean(epi_n)), "rho": _rho(ale, epi)}
            per_model[model_tag] = result
        out[(func_type, noise_type)] = per_model
    return out


def compute_samplesize_table() -> Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]]:
    """{(func_type,noise_type): {model_tag: {pct: stats or None}}}, pooled over
    whichever training-share (pct) values actually have data for that (model, DGP).
    Same shape/logic as compute_noise_table, swapping tau for pct."""
    out: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        search_dir = search_dir_standard("sample_size", noise_type, func_type)
        per_model: Dict[str, Dict[float, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            raw_by_pct: Dict[float, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
            for pct in SAMPLE_SIZE_PCTS:
                npz_path = resolve_latest_npz_at_pct(search_dir, model_tag, pct) if search_dir.exists() else None
                if npz_path is None:
                    raw_by_pct[pct] = None
                    continue
                _x, ale, epi = load_ale_epi(npz_path)
                raw_by_pct[pct] = (ale, epi)
            present = {p: v for p, v in raw_by_pct.items() if v is not None}
            if not present:
                per_model[model_tag] = {p: None for p in SAMPLE_SIZE_PCTS}
                continue
            pooled_ale = np.concatenate([v[0] for v in present.values()])
            pooled_epi = np.concatenate([v[1] for v in present.values()])
            ale_min, ale_max = float(pooled_ale.min()), float(pooled_ale.max())
            epi_min, epi_max = float(pooled_epi.min()), float(pooled_epi.max())
            result: Dict[float, Optional[dict]] = {}
            for p in SAMPLE_SIZE_PCTS:
                if raw_by_pct[p] is None:
                    result[p] = None
                    continue
                ale, epi = raw_by_pct[p]
                ale_n = _normalize(ale, ale_min, ale_max)
                epi_n = _normalize(epi, epi_min, epi_max)
                result[p] = {"AU": float(np.mean(ale_n)), "EU": float(np.mean(epi_n)), "rho": _rho(ale, epi)}
            per_model[model_tag] = result
        out[(func_type, noise_type)] = per_model
    return out


def resolve_ovb_npz(model_tag: str, noise_type: str, func_type: str,
                     beta2: Optional[float] = None, rho: Optional[float] = None,
                     tol: float = 1e-6) -> Optional[Path]:
    search_dir = RESULTS_ROOT / "ovb" / OVB_DIR_FOR_MODEL[model_tag] / noise_type / func_type
    if not search_dir.exists():
        return None
    candidates = []
    for p in sorted(search_dir.glob(f"ovb_outputs_{model_tag}_*.npz")):
        d = np.load(p, allow_pickle=True)
        if "beta2" not in d.files or "rho" not in d.files:
            continue
        b2 = float(np.asarray(d["beta2"]).ravel()[0])
        r = float(np.asarray(d["rho"]).ravel()[0])
        if beta2 is not None and abs(b2 - beta2) > tol:
            continue
        if rho is not None and abs(r - rho) > tol:
            continue
        candidates.append(p)
    return sorted(candidates)[-1] if candidates else None


def compute_ovb_sweep_table(sweep: str) -> Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]]:
    """sweep='beta2' (fixed rho=OVB_FIXED_RHO) or 'rho' (fixed beta2=OVB_FIXED_BETA2).
    Uses the 'omitted' (X-only) model output -- see tables/README.md for why."""
    values = OVB_BETA2_VALUES if sweep == "beta2" else OVB_RHO_VALUES
    fixed_kw = {"rho": OVB_FIXED_RHO} if sweep == "beta2" else {"beta2": OVB_FIXED_BETA2}

    out: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        per_model: Dict[str, Dict[float, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            raw_by_val: Dict[float, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
            for v in values:
                kw = dict(fixed_kw)
                kw[sweep] = v
                npz_path = resolve_ovb_npz(model_tag, noise_type, func_type, **kw)
                if npz_path is None:
                    raw_by_val[v] = None
                    continue
                _x, ale, epi = load_ale_epi(npz_path)
                raw_by_val[v] = (ale, epi)
            present = {k: val for k, val in raw_by_val.items() if val is not None}
            if not present:
                per_model[model_tag] = {v: None for v in values}
                continue
            pooled_ale = np.concatenate([val[0] for val in present.values()])
            pooled_epi = np.concatenate([val[1] for val in present.values()])
            ale_min, ale_max = float(pooled_ale.min()), float(pooled_ale.max())
            epi_min, epi_max = float(pooled_epi.min()), float(pooled_epi.max())
            result: Dict[float, Optional[dict]] = {}
            for v in values:
                if raw_by_val[v] is None:
                    result[v] = None
                    continue
                ale, epi = raw_by_val[v]
                ale_n = _normalize(ale, ale_min, ale_max)
                epi_n = _normalize(epi, epi_min, epi_max)
                result[v] = {"AU": float(np.mean(ale_n)), "EU": float(np.mean(epi_n)), "rho": _rho(ale, epi)}
            per_model[model_tag] = result
        out[(func_type, noise_type)] = per_model
    return out


# ============================== Companion tables: absolute-scale + AU-vs-truth ==============================
#
# Everything below computes RAW (non-pooled-normalised) absolute-scale AU/EU
# summaries and diagnostics of AU against the true noise variance sigma^2(x), for
# OOD/Undersampling/Sample-size/Noise-level only (OVB excluded -- not requested).
# These are entirely separate outputs (tab_*_absolute.tex, tab_*_autruth.tex,
# tab_noise_au_sweep_corr.tex) that do not touch compute_region_table/
# compute_noise_table/compute_samplesize_table/render_region_table/
# render_sweep_table or their output files in any way.

HOMOSCEDASTIC_VAR_TOL = 1e-8  # variance of sigma^2(x) below this on a region/grid => treated as constant
SHARE_DENOM_EPS = 1e-10       # |AU(x)+EU(x)| below this => point excluded from the epistemic-share mean
MIN_SWEEP_POINTS_FOR_CORR = 3  # need >= this many tau points with data for the noise across-sweep correlation


def sigma_true_baseline(x: np.ndarray, func_type: str, noise_type: str) -> np.ndarray:
    """True noise sd for OOD / Undersampling / Sample-size (no swept noise scale) --
    verified directly against Experiments/OOD.ipynb, "Sample Size.ipynb" and
    Experiments/Undersampling.ipynb's generate_toy_regression, all three identical:
        homoscedastic:   sigma(x) = 1.0
        heteroscedastic: sigma(x) = |2.5 * sin(0.5x + 5)|
    Reuses utils.mixture_metrics.DGP_TABLE (built/tested earlier for
    eval_predictive_baseline.py) rather than re-deriving the formula here."""
    from utils.mixture_metrics import get_dgp
    return get_dgp(func_type, noise_type).sigma_fn(np.asarray(x, dtype=float))


def sigma_true_noise_sweep(x: np.ndarray, noise_type: str, tau: float) -> np.ndarray:
    """True noise sd for the Noise-level sweep -- verified directly against
    Experiments/Noise_Level.ipynb's generate_toy_regression:
        homoscedastic:   sigma(x) = tau
        heteroscedastic: sigma(x) = |tau * sin(0.5x + 5)|
    Consistent with sigma_true_baseline at tau=1.0 (homoscedastic) / tau=2.5
    (heteroscedastic)."""
    x = np.asarray(x, dtype=float)
    if noise_type == "homoscedastic":
        return np.full_like(x, float(tau))
    return np.abs(float(tau) * np.sin(0.5 * x + 5))


def _is_homoscedastic_on_grid(sigma2_true: np.ndarray) -> bool:
    """Detected via the variance of sigma^2(x) itself, never by hardcoding which
    DGP names are homoscedastic -- a literal constant array has variance exactly
    0.0 in floating point (same repeated scalar squared), safely below
    HOMOSCEDASTIC_VAR_TOL and safely far from the smallest real heteroscedastic
    variance (order 1+)."""
    return float(np.var(sigma2_true)) < HOMOSCEDASTIC_VAR_TOL


def absolute_region_stats(au: np.ndarray, eu: np.ndarray) -> dict:
    """Raw mean/sd (population, ddof=0 -- deterministic grid evaluations, not a
    sample) for AU and EU, plus the epistemic share averaged POINTWISE:
    mean_x[EU(x)/(AU(x)+EU(x))], not mean(EU)/(mean(AU)+mean(EU)) -- the two differ
    whenever the ratio varies across x (true in OOD/undersampling). Points where
    |AU(x)+EU(x)| < SHARE_DENOM_EPS are excluded from the share mean and counted in
    n_share_excluded (not silently dropped -- the count is reported upstream).
    Also includes rho(AU,EU) -- the SAME quantity already shown in the original
    (normalised) tables, via the same _rho() helper on the same raw arrays; min-max
    normalisation is monotone-affine and doesn't change Pearson r, so this is
    numerically identical to the corresponding cell in tab_*.tex, just surfaced
    here too so it doesn't require cross-referencing the other table."""
    denom = au + eu
    stable = np.abs(denom) >= SHARE_DENOM_EPS
    n_excluded = int((~stable).sum())
    share_vals = eu[stable] / denom[stable] if np.any(stable) else np.array([])
    return {
        "mean_au": float(np.mean(au)), "sd_au": float(np.std(au, ddof=0)),
        "mean_eu": float(np.mean(eu)), "sd_eu": float(np.std(eu, ddof=0)),
        "share": float(np.mean(share_vals)) if share_vals.size else float("nan"),
        "rho": _rho(au, eu),
        "n_points": int(au.size), "n_share_excluded": n_excluded,
    }


def truth_region_stats(au: np.ndarray, sigma2_true: np.ndarray) -> dict:
    """Bias = mean_x[AU(x) - sigma^2(x)] -- always computed, homoscedastic
    included (correlation is shift/scale-invariant and would miss a uniformly
    compressed AU; bias catches it). Slope/intercept of the OLS fit
    AU(x) ~ sigma^2(x), and Pearson + Spearman correlation, are None ('--' at
    render time) when sigma^2(x) is constant on this region (zero-variance
    regressor -- detected via _is_homoscedastic_on_grid, not a hardcoded DGP
    list)."""
    bias = float(np.mean(au - sigma2_true))
    if _is_homoscedastic_on_grid(sigma2_true):
        return {"bias": bias, "slope": None, "intercept": None, "pearson": None,
                "spearman": None, "homoscedastic": True}
    lr = linregress(sigma2_true, au)
    rho_s = spearmanr(sigma2_true, au).correlation
    return {
        "bias": bias, "slope": float(lr.slope), "intercept": float(lr.intercept),
        "pearson": float(lr.rvalue),
        "spearman": float(rho_s) if not np.isnan(rho_s) else None,
        "homoscedastic": False,
    }


def compute_region_companions(experiment: str, region_masker
                               ) -> Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]]:
    """OOD / Undersampling companions: {(func_type,noise_type): {model_tag:
    {'A':{...absolute+truth} or None, 'B':...}}}. Re-resolves the same latest npz
    as compute_region_table (via the same resolve_latest_npz call) but keeps AU/EU
    on the raw scale and adds the sigma^2_true(x) comparison; does not call or
    modify compute_region_table."""
    out: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        search_dir = search_dir_standard(experiment, noise_type, func_type)
        per_model: Dict[str, Dict[str, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            npz_path = resolve_latest_npz(search_dir, (f"*{model_tag}*raw_outputs*.npz",)) if search_dir.exists() else None
            if npz_path is None:
                per_model[model_tag] = {"A": None, "B": None}
                continue
            x, ale, epi = load_ale_epi(npz_path)
            masks = region_masker(x)
            sigma2_true = sigma_true_baseline(x, func_type, noise_type) ** 2
            cell: Dict[str, Optional[dict]] = {}
            for name, mask in masks.items():
                if not np.any(mask):
                    cell[name] = None
                    continue
                cell[name] = {
                    **absolute_region_stats(ale[mask], epi[mask]),
                    **truth_region_stats(ale[mask], sigma2_true[mask]),
                }
            per_model[model_tag] = cell
        out[(func_type, noise_type)] = per_model
    return out


def compute_samplesize_companions() -> Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]]:
    """{(func_type,noise_type): {model_tag: {pct: {...absolute+truth} or None}}} --
    one region per swept training share (the whole grid at that pct), matching
    tab_samplesize.tex's row structure (no ID/OOD-style sub-split)."""
    out: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        search_dir = search_dir_standard("sample_size", noise_type, func_type)
        per_model: Dict[str, Dict[float, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            result: Dict[float, Optional[dict]] = {}
            for pct in SAMPLE_SIZE_PCTS:
                npz_path = resolve_latest_npz_at_pct(search_dir, model_tag, pct) if search_dir.exists() else None
                if npz_path is None:
                    result[pct] = None
                    continue
                x, ale, epi = load_ale_epi(npz_path)
                sigma2_true = sigma_true_baseline(x, func_type, noise_type) ** 2
                result[pct] = {**absolute_region_stats(ale, epi), **truth_region_stats(ale, sigma2_true)}
            per_model[model_tag] = result
        out[(func_type, noise_type)] = per_model
    return out


def compute_noise_companions() -> Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]]:
    """{(func_type,noise_type): {model_tag: {tau: {...absolute+truth} or None}}} --
    one region per swept tau (the whole grid at that tau)."""
    out: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]] = {}
    for func_type, noise_type, _label in DGPS:
        search_dir = search_dir_standard("noise_level", noise_type, func_type, extra="normal")
        per_model: Dict[str, Dict[float, Optional[dict]]] = {}
        for model_tag, _display in MODELS:
            result: Dict[float, Optional[dict]] = {}
            for tau in NOISE_TAUS:
                npz_path = resolve_latest_npz_at_tau(search_dir, model_tag, tau) if search_dir.exists() else None
                if npz_path is None:
                    result[tau] = None
                    continue
                x, ale, epi = load_ale_epi(npz_path)
                sigma2_true = sigma_true_noise_sweep(x, noise_type, tau) ** 2
                result[tau] = {**absolute_region_stats(ale, epi), **truth_region_stats(ale, sigma2_true)}
            per_model[model_tag] = result
        out[(func_type, noise_type)] = per_model
    return out


def compute_noise_au_sweep_corr(noise_companions: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]]
                                 ) -> Dict[Tuple[str, str], Dict[str, Optional[dict]]]:
    """Per (DGP, model): correlate the per-tau mean AU values against tau itself
    (the swept 'true noise level') ACROSS the sweep -- distinct from
    truth_region_stats's within-grid AU-vs-sigma^2(x) correlation, and meaningful
    in homoscedastic settings too since tau varies even when sigma^2(x) does not.
    None ('--' at render time) when fewer than MIN_SWEEP_POINTS_FOR_CORR tau values
    have data, or either series is constant over the available points."""
    out: Dict[Tuple[str, str], Dict[str, Optional[dict]]] = {}
    for func_type, noise_type, _label in DGPS:
        per_model: Dict[str, Optional[dict]] = {}
        for model_tag, _display in MODELS:
            per_tau = noise_companions[(func_type, noise_type)][model_tag]
            taus_present = sorted(t for t in NOISE_TAUS if per_tau[t] is not None)
            if len(taus_present) < MIN_SWEEP_POINTS_FOR_CORR:
                per_model[model_tag] = None
                continue
            tau_arr = np.array(taus_present, dtype=float)
            au_arr = np.array([per_tau[t]["mean_au"] for t in taus_present], dtype=float)
            if np.var(tau_arr) < HOMOSCEDASTIC_VAR_TOL or np.var(au_arr) < HOMOSCEDASTIC_VAR_TOL:
                per_model[model_tag] = None
                continue
            lr = linregress(tau_arr, au_arr)
            rho_s = spearmanr(tau_arr, au_arr).correlation
            per_model[model_tag] = {
                "pearson": float(lr.rvalue),
                "spearman": float(rho_s) if not np.isnan(rho_s) else None,
                "n_points": len(taus_present),
            }
        out[(func_type, noise_type)] = per_model
    return out


# ============================== LaTeX rendering ==============================

def fmt(x: Optional[float], nd: int = 4) -> str:
    return "---" if x is None else f"{x:.{nd}f}"


def _sentence_case(s: str) -> str:
    """Capitalize only the first character -- unlike str, this does not
    lowercase the rest of the string (which would mangle AU/EU/OOD/ID acronyms)."""
    return (s[:1].upper() + s[1:]) if s else s


def _cell(stats: Optional[dict], key: str) -> str:
    return "---" if stats is None else fmt(stats[key])


# ---- Companion-table number formatting: per-column fixed-vs-scientific choice ----
#
# Real magnitudes span an enormous range once BAMLSS's already-known OOD
# uncertainty explosion is included (mean AU as large as ~1e19-1e21 in
# heteroscedastic OOD/undersampling cells; EU as small as ~1e-28 in some
# near-degenerate undersampling cells) -- no single fixed-decimal precision can
# render both ends legibly, so each column picks fixed-point OR scientific
# notation based on its own realized value range (never mixed within one column).

def _latex_sci(v: float, sig: int = 3) -> str:
    if v == 0:
        return "0"
    s = f"{v:.{sig - 1}e}"
    mantissa, exp = s.split("e")
    return rf"{mantissa}\times10^{{{int(exp)}}}"


def choose_column_format(values: List[Optional[float]], min_decimals: int = 4,
                          max_decimals: int = 6) -> Tuple[str, int]:
    """('fixed', nd) if some nd in [min_decimals,max_decimals] shows the column's
    smallest nonzero magnitude as nonzero AND keeps the largest magnitude under
    1e4 digits; else ('sci', 3) -- 3 significant figures in scientific notation."""
    finite = [abs(v) for v in values if v is not None and np.isfinite(v) and v != 0]
    if not finite:
        return ("fixed", min_decimals)
    min_mag, max_mag = min(finite), max(finite)
    for nd in range(min_decimals, max_decimals + 1):
        if round(min_mag, nd) != 0 and max_mag < 1e4:
            return ("fixed", nd)
    return ("sci", 3)


def _fmt_num(v: float, spec: Tuple[str, int]) -> str:
    kind, p = spec
    return f"{v:.{p}f}" if kind == "fixed" else _latex_sci(v, p)


def fmt_meansd_cell(mean_v: Optional[float], sd_v: Optional[float], spec: Tuple[str, int]) -> str:
    """'$mean \\pm sd$' -- always math-mode since \\pm requires it, regardless of
    fixed vs. scientific notation."""
    if mean_v is None or sd_v is None:
        return "---"
    return f"${_fmt_num(mean_v, spec)} \\pm {_fmt_num(sd_v, spec)}$"


def fmt_scalar_cell(v: Optional[float], spec: Tuple[str, int]) -> str:
    if v is None:
        return "---"
    n = _fmt_num(v, spec)
    return f"${n}$" if spec[0] == "sci" else n


def render_region_table(data: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]],
                         label: str, caption: str,
                         region_a_name: str, region_b_name: str) -> str:
    n_cols = 7
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
        r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
        rf"Scenario/Model & AU$_{{\text{{{region_a_name}}}}}$ & AU$_{{\text{{{region_b_name}}}}}$ "
        rf"& EU$_{{\text{{{region_a_name}}}}}$ & EU$_{{\text{{{region_b_name}}}}}$ "
        rf"& $\rho_{{\text{{{region_a_name}}}}}$ & $\rho_{{\text{{{region_b_name}}}}}$ \\",
        r"\midrule",
    ]
    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for model_tag, display in MODELS:
            s = per_model[model_tag]
            row = [
                display,
                _cell(s["A"], "AU"), _cell(s["B"], "AU"),
                _cell(s["A"], "EU"), _cell(s["B"], "EU"),
                _cell(s["A"], "rho"), _cell(s["B"], "rho"),
            ]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


NEURIPS_SINGLE_COLUMN_WIDTH_PT = 396.0  # 5.5in text block
_PT_PER_CHAR = 4.5   # rough glyph width at \small
_NUMERIC_CELL_CHARS = 7  # e.g. "-0.9391"
_TABCOLSEP_PT = 4.0  # matches \setlength{\tabcolsep}{4pt} used by every table here


def estimate_sweep_table_width_pt(n_models: int, knob_col_chars: int = 4) -> float:
    """Heuristic only (no LaTeX toolchain available in this environment to compile
    and confirm) -- content width (knob column + 3 numeric columns per model) plus
    \\tabcolsep spacing on both sides of every column. Consistently estimates the
    2-model case (7 cols) well under budget, 3 models (10 cols) tight-but-fits, and
    4 models (13 cols) clearly over NEURIPS_SINGLE_COLUMN_WIDTH_PT."""
    n_cols = 1 + 3 * n_models
    content_pt = (knob_col_chars + _NUMERIC_CELL_CHARS * 3 * n_models) * _PT_PER_CHAR
    spacing_pt = n_cols * 2 * _TABCOLSEP_PT
    return content_pt + spacing_pt


def render_sweep_table(data: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]],
                        values: Sequence[float], label: str, caption: str,
                        knob_symbol: str, value_fmt=lambda v: f"{v:g}") -> str:
    n_models = len(MODELS)
    n_cols = 1 + 3 * n_models
    est_width = estimate_sweep_table_width_pt(n_models)
    if est_width > NEURIPS_SINGLE_COLUMN_WIDTH_PT:
        print(
            f"WARNING: tab_{label}.tex has {n_cols} columns ({n_models} models x 3 "
            f"+ 1 knob column); estimated width ~{est_width:.0f}pt exceeds the "
            f"{NEURIPS_SINGLE_COLUMN_WIDTH_PT:.0f}pt (5.5in) single-column budget at "
            f"\\small. This is a heuristic character-count estimate, not a compiled "
            f"check (no LaTeX toolchain in this environment) -- please verify in "
            f"Overleaf. Not auto-shrinking below \\small; consider fewer models, "
            f"splitting the table, or \\resizebox."
        )
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
        r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
    ]
    # Header row 1: model group spans + cmidrules
    header1_cells = [""]
    cmidrules = []
    col = 2
    for _tag, display in MODELS:
        header1_cells.append(rf"\multicolumn{{3}}{{c}}{{{display}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col+2}}}")
        col += 3
    lines.append(" & ".join(header1_cells) + r" \\")
    lines.append(" ".join(cmidrules))
    header2_cells = [knob_symbol] + ["AU", "EU", r"$\rho$"] * n_models
    lines.append(" & ".join(header2_cells) + r" \\")
    lines.append(r"\midrule")

    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for v in values:
            row = [value_fmt(v)]
            for model_tag, _display in MODELS:
                s = per_model[model_tag][v]
                row += [_cell(s, "AU"), _cell(s, "EU"), _cell(s, "rho")]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ============================== Companion-table rendering ==============================
#
# Standalone tabular bodies only (no \begin{table}/\caption/\label -- those live in
# the manuscript). Row order, \multicolumn{N}{l}{\textit{DGP}} headings and
# \addlinespace separators mirror render_region_table/render_sweep_table exactly so
# companions can be read against the originals by row position. Column order is
# metric-major (all of metric 1 across regions/models, then metric 2, ...),
# matching render_region_table's AU_A,AU_B,EU_A,EU_B,rho_A,rho_B convention.
# Per-metric number format (fixed vs. scientific) is decided once per metric,
# pooling both regions (or all swept knob values) together, rather than per exact
# rendered column -- avoids AU_ID and AU_OOD silently switching notations for what
# is conceptually the same reading.

def render_region_absolute_table(data: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]],
                                  region_a_name: str, region_b_name: str) -> str:
    au_spec = choose_column_format([
        v for per_model in data.values() for cell in per_model.values()
        for name in ("A", "B") if cell[name] is not None
        for v in (cell[name]["mean_au"], cell[name]["sd_au"])
    ])
    eu_spec = choose_column_format([
        v for per_model in data.values() for cell in per_model.values()
        for name in ("A", "B") if cell[name] is not None
        for v in (cell[name]["mean_eu"], cell[name]["sd_eu"])
    ])
    n_cols = 9
    lines = [
        r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
        rf"Scenario/Model & AU$_{{\text{{{region_a_name}}}}}$ & AU$_{{\text{{{region_b_name}}}}}$ "
        rf"& EU$_{{\text{{{region_a_name}}}}}$ & EU$_{{\text{{{region_b_name}}}}}$ "
        rf"& $\rho_{{\text{{{region_a_name}}}}}$ & $\rho_{{\text{{{region_b_name}}}}}$ "
        rf"& Share$_{{\text{{{region_a_name}}}}}$ & Share$_{{\text{{{region_b_name}}}}}$ \\",
        r"\midrule",
    ]
    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for model_tag, display in MODELS:
            a, b = per_model[model_tag]["A"], per_model[model_tag]["B"]
            row = [
                display,
                fmt_meansd_cell(a["mean_au"] if a else None, a["sd_au"] if a else None, au_spec),
                fmt_meansd_cell(b["mean_au"] if b else None, b["sd_au"] if b else None, au_spec),
                fmt_meansd_cell(a["mean_eu"] if a else None, a["sd_eu"] if a else None, eu_spec),
                fmt_meansd_cell(b["mean_eu"] if b else None, b["sd_eu"] if b else None, eu_spec),
                fmt(a["rho"], 4) if a else "---",
                fmt(b["rho"], 4) if b else "---",
                fmt(a["share"], 3) if a else "---",
                fmt(b["share"], 3) if b else "---",
            ]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_region_autruth_table(data: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]],
                                 region_a_name: str, region_b_name: str) -> str:
    bias_spec = choose_column_format([
        cell[name]["bias"] for per_model in data.values() for cell in per_model.values()
        for name in ("A", "B") if cell[name] is not None
    ])
    # Slope is unbounded (unlike Pearson/Spearman, which are always in [-1,1] by
    # definition) -- when AU explodes (BAMLSS OOD/heteroscedastic pathology, see
    # the absolute-scale table), regressing it against a bounded sigma^2(x) yields
    # an astronomically large slope. Fixed 3-decimals would render that as a
    # 20+ digit garbage string, so slope gets the same fixed-vs-scientific choice
    # as bias/mean/sd, deviating from the literal "3 decimals" instruction only in
    # this pathological case -- flagged in the run summary.
    slope_spec = choose_column_format([
        cell[name]["slope"] for per_model in data.values() for cell in per_model.values()
        for name in ("A", "B") if cell[name] is not None and cell[name]["slope"] is not None
    ], min_decimals=3, max_decimals=3)
    n_cols = 9
    lines = [
        r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
        rf"Scenario/Model & Bias$_{{\text{{{region_a_name}}}}}$ & Bias$_{{\text{{{region_b_name}}}}}$ "
        rf"& Slope$_{{\text{{{region_a_name}}}}}$ & Slope$_{{\text{{{region_b_name}}}}}$ "
        rf"& Pearson$_{{\text{{{region_a_name}}}}}$ & Pearson$_{{\text{{{region_b_name}}}}}$ "
        rf"& Spearman$_{{\text{{{region_a_name}}}}}$ & Spearman$_{{\text{{{region_b_name}}}}}$ \\",
        r"\midrule",
    ]
    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for model_tag, display in MODELS:
            a, b = per_model[model_tag]["A"], per_model[model_tag]["B"]
            row = [
                display,
                fmt_scalar_cell(a["bias"] if a else None, bias_spec),
                fmt_scalar_cell(b["bias"] if b else None, bias_spec),
                fmt_scalar_cell(a["slope"] if a else None, slope_spec),
                fmt_scalar_cell(b["slope"] if b else None, slope_spec),
                fmt(a["pearson"], 3) if a and a["pearson"] is not None else "---",
                fmt(b["pearson"], 3) if b and b["pearson"] is not None else "---",
                fmt(a["spearman"], 3) if a and a["spearman"] is not None else "---",
                fmt(b["spearman"], 3) if b and b["spearman"] is not None else "---",
            ]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_sweep_absolute_table(data: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]],
                                 values: Sequence[float], knob_symbol: str,
                                 value_fmt=lambda v: f"{v:g}") -> str:
    n_models = len(MODELS)
    n_cols = 1 + 4 * n_models
    au_spec = choose_column_format([
        s[k] for per_model in data.values() for series in per_model.values()
        for s in series.values() if s is not None for k in ("mean_au", "sd_au")
    ])
    eu_spec = choose_column_format([
        s[k] for per_model in data.values() for series in per_model.values()
        for s in series.values() if s is not None for k in ("mean_eu", "sd_eu")
    ])
    lines = [r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}", r"\toprule"]
    header1_cells, cmidrules, col = [""], [], 2
    for _tag, display in MODELS:
        header1_cells.append(rf"\multicolumn{{4}}{{c}}{{{display}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + 3}}}")
        col += 4
    lines.append(" & ".join(header1_cells) + r" \\")
    lines.append(" ".join(cmidrules))
    lines.append(" & ".join([knob_symbol] + ["AU", "EU", r"$\rho$", "Share"] * n_models) + r" \\")
    lines.append(r"\midrule")
    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for v in values:
            row = [value_fmt(v)]
            for model_tag, _display in MODELS:
                s = per_model[model_tag][v]
                if s is None:
                    row += ["---", "---", "---", "---"]
                else:
                    row += [fmt_meansd_cell(s["mean_au"], s["sd_au"], au_spec),
                            fmt_meansd_cell(s["mean_eu"], s["sd_eu"], eu_spec),
                            fmt(s["rho"], 4),
                            fmt(s["share"], 3)]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_sweep_autruth_table(data: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]],
                                values: Sequence[float], knob_symbol: str,
                                value_fmt=lambda v: f"{v:g}") -> str:
    n_models = len(MODELS)
    n_cols = 1 + 4 * n_models
    bias_spec = choose_column_format([
        s["bias"] for per_model in data.values() for series in per_model.values()
        for s in series.values() if s is not None
    ])
    slope_spec = choose_column_format([
        s["slope"] for per_model in data.values() for series in per_model.values()
        for s in series.values() if s is not None and s["slope"] is not None
    ], min_decimals=3, max_decimals=3)
    lines = [r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}", r"\toprule"]
    header1_cells, cmidrules, col = [""], [], 2
    for _tag, display in MODELS:
        header1_cells.append(rf"\multicolumn{{4}}{{c}}{{{display}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + 3}}}")
        col += 4
    lines.append(" & ".join(header1_cells) + r" \\")
    lines.append(" ".join(cmidrules))
    lines.append(" & ".join([knob_symbol] + ["Bias", "Slope", "Pearson", "Spearman"] * n_models) + r" \\")
    lines.append(r"\midrule")
    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for v in values:
            row = [value_fmt(v)]
            for model_tag, _display in MODELS:
                s = per_model[model_tag][v]
                if s is None:
                    row += ["---", "---", "---", "---"]
                else:
                    row += [fmt_scalar_cell(s["bias"], bias_spec),
                            fmt_scalar_cell(s["slope"], slope_spec),
                            fmt(s["pearson"], 3) if s["pearson"] is not None else "---",
                            fmt(s["spearman"], 3) if s["spearman"] is not None else "---"]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_noise_au_sweep_corr_table(data: Dict[Tuple[str, str], Dict[str, Optional[dict]]]) -> str:
    n_cols = 3
    lines = [
        r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
        r"Scenario/Model & Pearson & Spearman \\",
        r"\midrule",
    ]
    for i, (func_type, noise_type, dgp_label) in enumerate(DGPS):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{dgp_label}}}}} \\")
        per_model = data[(func_type, noise_type)]
        for model_tag, display in MODELS:
            s = per_model[model_tag]
            row = [
                display,
                fmt(s["pearson"], 3) if s is not None else "---",
                fmt(s["spearman"], 3) if s is not None and s["spearman"] is not None else "---",
            ]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ============================== Companion sanity checks (Step 5) ==============================

def _sanity_region_companion(name: str, data: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[dict]]]]) -> None:
    print(f"\n[{name}] grid-point counts / diagnostics:")
    for func_type, noise_type, dgp_label in DGPS:
        per_model = data[(func_type, noise_type)]
        for model_tag, cell in per_model.items():
            for region_name in ("A", "B"):
                s = cell[region_name]
                if s is None:
                    continue
                if not np.isnan(s["share"]):
                    assert -1e-9 <= s["share"] <= 1.0 + 1e-9, (
                        f"{name} {dgp_label}/{model_tag}/{region_name}: share {s['share']} out of [0,1]")
                for ck in ("pearson", "spearman"):
                    v = s[ck]
                    if v is not None:
                        assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9, (
                            f"{name} {dgp_label}/{model_tag}/{region_name}: {ck} {v} out of [-1,1]")
                flag = " FLAG: mean_EU within one sd of zero" if s["mean_eu"] <= s["sd_eu"] else ""
                print(f"  {dgp_label}/{model_tag}/{region_name}: n={s['n_points']} "
                      f"share_excluded={s['n_share_excluded']}{flag}")


def _sanity_sweep_companion(name: str, data: Dict[Tuple[str, str], Dict[str, Dict[float, Optional[dict]]]],
                             values: Sequence[float]) -> None:
    print(f"\n[{name}] grid-point counts / diagnostics:")
    for func_type, noise_type, dgp_label in DGPS:
        per_model = data[(func_type, noise_type)]
        for model_tag, series in per_model.items():
            for v in values:
                s = series[v]
                if s is None:
                    continue
                if not np.isnan(s["share"]):
                    assert -1e-9 <= s["share"] <= 1.0 + 1e-9, (
                        f"{name} {dgp_label}/{model_tag}/{v}: share {s['share']} out of [0,1]")
                for ck in ("pearson", "spearman"):
                    cv = s[ck]
                    if cv is not None:
                        assert -1.0 - 1e-9 <= cv <= 1.0 + 1e-9, (
                            f"{name} {dgp_label}/{model_tag}/{v}: {ck} {cv} out of [-1,1]")
                flag = " FLAG: mean_EU within one sd of zero" if s["mean_eu"] <= s["sd_eu"] else ""
                print(f"  {dgp_label}/{model_tag}/{knob_label(name)}={v}: n={s['n_points']} "
                      f"share_excluded={s['n_share_excluded']}{flag}")


def knob_label(name: str) -> str:
    return "pct" if "samplesize" in name else "tau"


def print_sigma2_variance_diagnostics() -> None:
    print("\nVariance of sigma^2(x) per DGP (baseline formula used by OOD/Undersampling/Sample-size):")
    for func_type, noise_type, dgp_label in DGPS:
        search_dir = search_dir_standard("ood", noise_type, func_type)
        npz_path = None
        for model_tag, _display in MODELS:
            npz_path = resolve_latest_npz(search_dir, (f"*{model_tag}*raw_outputs*.npz",)) if search_dir.exists() else None
            if npz_path is not None:
                break
        if npz_path is None:
            print(f"  {dgp_label}: no file found to evaluate on")
            continue
        x, _ale, _epi = load_ale_epi(npz_path)
        var = float(np.var(sigma_true_baseline(x, func_type, noise_type) ** 2))
        kind = "HOMOSCEDASTIC" if var < HOMOSCEDASTIC_VAR_TOL else "heteroscedastic"
        print(f"  {dgp_label}: var(sigma^2(x))={var:.6g} -> {kind}")

    print("\nVariance of sigma^2(x) per (DGP, tau) for the Noise-level sweep:")
    for func_type, noise_type, dgp_label in DGPS:
        search_dir = search_dir_standard("noise_level", noise_type, func_type, extra="normal")
        for tau in NOISE_TAUS:
            npz_path = None
            for model_tag, _display in MODELS:
                p = resolve_latest_npz_at_tau(search_dir, model_tag, tau) if search_dir.exists() else None
                if p is not None:
                    npz_path = p
                    break
            if npz_path is None:
                continue
            x, _ale, _epi = load_ale_epi(npz_path)
            var = float(np.var(sigma_true_noise_sweep(x, noise_type, tau) ** 2))
            kind = "HOMOSCEDASTIC" if var < HOMOSCEDASTIC_VAR_TOL else "heteroscedastic"
            print(f"  {dgp_label}, tau={tau}: var(sigma^2(x))={var:.6g} -> {kind}")


# ============================== Caption helpers (data-driven headline text) ==============================

def region_trend_phrase(data, region_a_name, region_b_name) -> str:
    """Very small data-driven summary used to seed the caption -- describes the
    average AU/EU change from region A to region B, averaged across present cells."""
    au_deltas, eu_deltas = [], []
    for per_model in data.values():
        for stats in per_model.values():
            a, b = stats.get("A"), stats.get("B")
            if a is not None and b is not None:
                au_deltas.append(b["AU"] - a["AU"])
                eu_deltas.append(b["EU"] - a["EU"])
    if not au_deltas:
        return "insufficient data to characterise the trend"
    au_mean, eu_mean = float(np.mean(au_deltas)), float(np.mean(eu_deltas))
    au_dir = "higher" if au_mean > 0 else "lower"
    eu_dir = "higher" if eu_mean > 0 else "lower"
    return (f"AU is on average {abs(au_mean):.4f} {au_dir} and EU is on average "
            f"{abs(eu_mean):.4f} {eu_dir} in {region_b_name} than {region_a_name}")


def sweep_trend_phrase(data, values) -> str:
    """Data-driven summary of AU/EU trend across a knob sweep, averaged across
    (DGP, model) cells that have both endpoints present."""
    au_deltas, eu_deltas = [], []
    for per_model in data.values():
        for series in per_model.values():
            present = [(v, series[v]) for v in values if series[v] is not None]
            if len(present) < 2:
                continue
            (_, first), (_, last) = present[0], present[-1]
            au_deltas.append(last["AU"] - first["AU"])
            eu_deltas.append(last["EU"] - first["EU"])
    if not au_deltas:
        return "insufficient data to characterise the trend"
    au_mean, eu_mean = float(np.mean(au_deltas)), float(np.mean(eu_deltas))
    au_desc = "rises" if au_mean > 0.01 else ("falls" if au_mean < -0.01 else "stays roughly flat")
    eu_desc = "rises" if eu_mean > 0.01 else ("falls" if eu_mean < -0.01 else "stays roughly flat")
    return f"AU {au_desc} (\\ensuremath{{\\Delta={au_mean:+.4f}}}) while EU {eu_desc} (\\ensuremath{{\\Delta={eu_mean:+.4f}}}) from first to last swept value"


# ============================== Main ==============================

def main():
    global MODELS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=project_root / "tables")
    parser.add_argument(
        "--models", type=str, default="Deep_Ensemble,BAMLSS",
        help=(
            "Comma-separated npz model tags to include, in order (becomes row/column "
            f"order). One or more of: {','.join(CANDIDATE_MODEL_TAGS)}. Models with no "
            "saved data for a given cell render as '---', never an error. "
            "Default matches the original 2-model scope."
        ),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    requested_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    unknown = [t for t in requested_tags if t not in CANDIDATE_MODEL_TAGS]
    if unknown:
        raise SystemExit(
            f"Unknown model tag(s) in --models: {unknown}. "
            f"Must be one or more of: {CANDIDATE_MODEL_TAGS}"
        )
    if not requested_tags:
        raise SystemExit("--models must name at least one model.")
    MODELS = [(tag, CANDIDATE_MODEL_DISPLAY[tag]) for tag in requested_tags]
    print(f"Models in scope for this run: {[d for _, d in MODELS]}")

    print("Computing OOD table...")
    ood_data = compute_region_table("ood", ood_region_masker)
    ood_trend = region_trend_phrase(ood_data, "ID", "OOD")
    ood_caption = (
        "Aleatoric (AU) and epistemic (EU) uncertainty, min--max normalised per model "
        "over the pooled in-distribution (ID, $x\\in[-5,10]$) and out-of-distribution "
        "(OOD, $x\\in(10,15]$) grid, with the Pearson correlation $\\rho$(AU,EU) computed "
        "on the raw (unnormalised) values within each region. " + _sentence_case(ood_trend) + "."
    )
    (args.out / "tab_ood.tex").write_text(
        render_region_table(ood_data, "ood", ood_caption, "ID", "OOD"))

    print("Computing Undersampling table...")
    us_data = compute_region_table("undersampling", undersampling_region_masker)
    us_trend = region_trend_phrase(us_data, "under-sampled", "well-sampled")
    us_caption = (
        "AU and EU, min--max normalised per model over the pooled grid, split into the "
        "interior under-sampled band ($x\\in(0,6)$, density factor 0.01 relative to "
        "$x\\in[-5,0]\\cup[6,10]$ at density 1) versus the well-sampled remainder. "
        + _sentence_case(us_trend) + "."
    )
    (args.out / "tab_undersampling.tex").write_text(
        render_region_table(us_data, "undersampling", us_caption, "under", "well"))

    print("Computing Sample-size table...")
    samplesize_data = compute_samplesize_table()
    samplesize_trend = sweep_trend_phrase(samplesize_data, SAMPLE_SIZE_PCTS)
    samplesize_caption = (
        "AU and EU vs.\\ training-data share, min--max normalised per model over all "
        "training shares pooled per (model, DGP), then aggregated per share. "
        + _sentence_case(samplesize_trend) + "."
    )
    (args.out / "tab_samplesize.tex").write_text(
        render_sweep_table(samplesize_data, SAMPLE_SIZE_PCTS, "samplesize", samplesize_caption,
                            r"Train \%"))

    print("Computing Noise-level table...")
    noise_data = compute_noise_table()
    noise_trend = sweep_trend_phrase(noise_data, NOISE_TAUS)
    noise_caption = (
        "AU and EU vs.\\ noise scale $\\tau$, min--max normalised per model over all "
        "$\\tau$ values pooled per (model, DGP), then aggregated per $\\tau$. "
        + _sentence_case(noise_trend) + ". BAMLSS is only available at "
        "$\\tau\\in\\{0.5,5\\}$ for this run."
    )
    (args.out / "tab_noise.tex").write_text(
        render_sweep_table(noise_data, NOISE_TAUS, "noise", noise_caption, r"$\tau$"))

    print("Computing OVB beta2 table...")
    ovb_beta2_data = compute_ovb_sweep_table("beta2")
    ovb_beta2_trend = sweep_trend_phrase(ovb_beta2_data, OVB_BETA2_VALUES)
    ovb_caption = (
        f"AU and EU (omitted, X-only model) vs.\\ structural-bias coefficient "
        f"$\\beta_2$ at fixed $\\rho_{{XZ}}={OVB_FIXED_RHO}$, min--max normalised per "
        f"model over all swept $\\beta_2$ values pooled. " + _sentence_case(ovb_beta2_trend) + "."
    )
    (args.out / "tab_ovb.tex").write_text(
        render_sweep_table(ovb_beta2_data, OVB_BETA2_VALUES, "ovb", ovb_caption, r"$\beta_2$"))

    print("Computing OVB rho table...")
    ovb_rho_data = compute_ovb_sweep_table("rho")
    ovb_rho_trend = sweep_trend_phrase(ovb_rho_data, OVB_RHO_VALUES)
    ovb_rho_caption = (
        f"AU and EU (omitted, X-only model) vs.\\ the X--Z correlation "
        f"$\\rho_{{XZ}}$ at fixed $\\beta_2={OVB_FIXED_BETA2}$, min--max normalised per "
        f"model over all swept $\\rho_{{XZ}}$ values pooled. " + _sentence_case(ovb_rho_trend) + "."
    )
    (args.out / "tab_ovb_rho.tex").write_text(
        render_sweep_table(ovb_rho_data, OVB_RHO_VALUES, "ovb-rho", ovb_rho_caption, r"$\rho_{XZ}$"))

    print(f"\nWrote 6 tables under {args.out}")

    # ---- Companion tables: absolute-scale + AU-vs-true-noise-variance diagnostics ----
    print("\nComputing companion tables (absolute-scale + AU-vs-truth diagnostics)...")
    ood_comp = compute_region_companions("ood", ood_region_masker)
    (args.out / "tab_ood_absolute.tex").write_text(render_region_absolute_table(ood_comp, "ID", "OOD"))
    (args.out / "tab_ood_autruth.tex").write_text(render_region_autruth_table(ood_comp, "ID", "OOD"))

    us_comp = compute_region_companions("undersampling", undersampling_region_masker)
    (args.out / "tab_undersampling_absolute.tex").write_text(render_region_absolute_table(us_comp, "under", "well"))
    (args.out / "tab_undersampling_autruth.tex").write_text(render_region_autruth_table(us_comp, "under", "well"))

    samplesize_comp = compute_samplesize_companions()
    (args.out / "tab_samplesize_absolute.tex").write_text(
        render_sweep_absolute_table(samplesize_comp, SAMPLE_SIZE_PCTS, r"Train \%"))
    (args.out / "tab_samplesize_autruth.tex").write_text(
        render_sweep_autruth_table(samplesize_comp, SAMPLE_SIZE_PCTS, r"Train \%"))

    noise_comp = compute_noise_companions()
    (args.out / "tab_noise_absolute.tex").write_text(
        render_sweep_absolute_table(noise_comp, NOISE_TAUS, r"$\tau$"))
    (args.out / "tab_noise_autruth.tex").write_text(
        render_sweep_autruth_table(noise_comp, NOISE_TAUS, r"$\tau$"))

    noise_sweep_corr = compute_noise_au_sweep_corr(noise_comp)
    (args.out / "tab_noise_au_sweep_corr.tex").write_text(render_noise_au_sweep_corr_table(noise_sweep_corr))

    print(f"Wrote 9 companion tables under {args.out}")

    # ---- Step 5: sanity output ----
    print("\n" + "=" * 70)
    print("COMPANION-TABLE SANITY CHECKS")
    print("=" * 70)
    _sanity_region_companion("ood_absolute/autruth", ood_comp)
    _sanity_region_companion("undersampling_absolute/autruth", us_comp)
    _sanity_sweep_companion("samplesize_absolute/autruth", samplesize_comp, SAMPLE_SIZE_PCTS)
    _sanity_sweep_companion("noise_absolute/autruth", noise_comp, NOISE_TAUS)
    print("\n[noise_au_sweep_corr] per-DGP/model results:")
    for func_type, noise_type, dgp_label in DGPS:
        for model_tag, _display in MODELS:
            s = noise_sweep_corr[(func_type, noise_type)][model_tag]
            if s is None:
                print(f"  {dgp_label}/{model_tag}: --- (fewer than {MIN_SWEEP_POINTS_FOR_CORR} tau points, "
                      f"or a constant series)")
            else:
                assert -1.0 - 1e-9 <= s["pearson"] <= 1.0 + 1e-9
                if s["spearman"] is not None:
                    assert -1.0 - 1e-9 <= s["spearman"] <= 1.0 + 1e-9
                print(f"  {dgp_label}/{model_tag}: n_tau={s['n_points']} pearson={s['pearson']:.3f} "
                      f"spearman={s['spearman'] if s['spearman'] is None else round(s['spearman'], 3)}")
    print_sigma2_variance_diagnostics()

    # ---- Headline summary (deliverable #4) ----
    print("\n" + "=" * 70)
    print("HEADLINE NUMBERS")
    print("=" * 70)

    print("\nOVB beta2-sweep AU/EU ranges (both models, all DGPs):")
    for model_tag, display in MODELS:
        aus, eus = [], []
        for per_model in ovb_beta2_data.values():
            for v, s in per_model[model_tag].items():
                if s is not None:
                    aus.append(s["AU"]); eus.append(s["EU"])
        if aus:
            print(f"  {display}: AU in [{min(aus):.4f}, {max(aus):.4f}], EU in [{min(eus):.4f}, {max(eus):.4f}]")
        else:
            print(f"  {display}: no data")

    print("\nID -> OOD rho shifts (per DGP, per model):")
    for func_type, noise_type, dgp_label in DGPS:
        for model_tag, display in MODELS:
            s = ood_data[(func_type, noise_type)][model_tag]
            a, b = s["A"], s["B"]
            if a is not None and b is not None:
                print(f"  {dgp_label} / {display}: rho_ID={a['rho']:.4f} -> rho_OOD={b['rho']:.4f} "
                      f"(delta={b['rho']-a['rho']:+.4f})")
            else:
                print(f"  {dgp_label} / {display}: --- (missing data)")


if __name__ == "__main__":
    main()
