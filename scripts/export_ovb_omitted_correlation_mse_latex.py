r"""
Export two LaTeX tables for **OVB omitted** (1D) model runs.

1. **Correlations:** $\rho_{\mathrm{var}} =$ ``au_eu_corr_var`` from legacy OVB sweep Excel under
   ``results/ovb/<model>/...``; $\rho_{\mathrm{ent}} =$ ``Correlation_Epi_Ale`` from moment-matched
   batch workbooks (concatenated under ``<stats_mm>/ovb/<noise>/<func>/``).

2. **MSE:** $\mathrm{MSE}_{\mathrm{var}}$ from ``mse`` in the same legacy Excel;
   $\mathrm{MSE}_{\mathrm{mm}}$ from ``MSE`` in the moment-matched rows.

Uses the same sweep slicing as ``plot_ovb_sweep_summary_panels`` / ``ovb_sweep_summary_loaders``:
for each (function $\times$ noise) we emit a **$\rho$ sweep** block (fixed $\beta_2$) and a **$\beta_2$ sweep**
block (fixed $\rho$, with the loader's fallback when rows do not match ``fixed_rho`` exactly).

Usage::

    python scripts/export_ovb_omitted_correlation_mse_latex.py
    python scripts/export_ovb_omitted_correlation_mse_latex.py --fixed-beta2 1.0 --fixed-rho 0.75 \\
        --ovb-root results/ovb --stats-mm results/entropy_recomputed_moment_matched_batch/statistics
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.ovb_sweep_summary_loaders import (
    load_stats_by_model_entropy,
    load_stats_by_model_variance,
)
from utils.regression_summary_panels import MODEL_KEYS

MODEL_ORDER = list(MODEL_KEYS.keys())
SweepParam = Literal["rho", "beta2"]

CONDITIONS: List[Tuple[str, str]] = [
    ("linear", "homoscedastic"),
    ("linear", "heteroscedastic"),
    ("sin", "homoscedastic"),
    ("sin", "heteroscedastic"),
]

COND_LABEL = {
    ("linear", "homoscedastic"): "Linear, homoscedastic",
    ("linear", "heteroscedastic"): "Linear, heteroscedastic",
    ("sin", "homoscedastic"): "Sinusoidal, homoscedastic",
    ("sin", "heteroscedastic"): "Sinusoidal, heteroscedastic",
}

MODEL_HEADER = {
    "Deep Ensemble": "DeepEns",
    "MC Dropout": "MCDrop",
    "BNN": "BNN",
    "BAMLSS": "BAMLSS",
}


def _collect_varying_values(
    var_by: Dict[str, pd.DataFrame],
    mm_by: Dict[str, pd.DataFrame],
    sweep: SweepParam,
) -> List[float]:
    col: str = "rho" if sweep == "rho" else "beta2"
    s: Set[float] = set()
    for d in (*var_by.values(), *mm_by.values()):
        if d is None or d.empty or col not in d.columns:
            continue
        for x in pd.to_numeric(d[col], errors="coerce").dropna().unique():
            s.add(float(x))
    return sorted(s)


def _row_mask(df: pd.DataFrame, sweep: SweepParam, value: float) -> pd.Series:
    col = "rho" if sweep == "rho" else "beta2"
    return np.isclose(df[col].astype(float), float(value), rtol=1e-5, atol=1e-8)


def _float_at_sweep(
    df: Optional[pd.DataFrame],
    sweep: SweepParam,
    value: float,
    columns: Tuple[str, ...],
) -> Optional[float]:
    if df is None or df.empty:
        return None
    col = "rho" if sweep == "rho" else "beta2"
    if col not in df.columns:
        return None
    sub = df.loc[_row_mask(df, sweep, value)]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    for c in columns:
        if c in sub.columns:
            v = float(row[c])
            if not math.isnan(v):
                return v
    return None


def _corr_var_at(df: Optional[pd.DataFrame], sweep: SweepParam, value: float) -> Optional[float]:
    return _float_at_sweep(df, sweep, value, ("au_eu_corr_var",))


def _corr_ent_at(df: Optional[pd.DataFrame], sweep: SweepParam, value: float) -> Optional[float]:
    return _float_at_sweep(df, sweep, value, ("Correlation_Epi_Ale",))


def _mse_var_at(df: Optional[pd.DataFrame], sweep: SweepParam, value: float) -> Optional[float]:
    return _float_at_sweep(df, sweep, value, ("mse", "MSE"))


def _mse_mm_at(df: Optional[pd.DataFrame], sweep: SweepParam, value: float) -> Optional[float]:
    return _float_at_sweep(df, sweep, value, ("MSE", "mse"))


def _fmt_rho(x: Optional[float]) -> str:
    if x is None:
        return "---"
    return f"{x:.4f}"


def _fmt_mse(x: Optional[float]) -> str:
    if x is None:
        return "---"
    if math.isnan(x):
        return "---"
    ax = abs(x)
    if ax != 0 and (ax >= 1e4 or ax < 1e-3):
        return f"{x:.4e}"
    return f"{x:.6g}"


def _fmt_param_value(sweep: SweepParam, value: float) -> str:
    if sweep == "rho":
        return f"{value:g}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    s = f"{value:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _table_header_corr(nmodel: int) -> List[str]:
    colspec = "l" + "rr" * nmodel
    lines = [
        rf"  \begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        r"    \toprule",
    ]
    head1 = [r"\multicolumn{2}{c}{" + MODEL_HEADER[m] + r"}" for m in MODEL_ORDER]
    lines.append("    Param. & " + " & ".join(head1) + r" \\")
    lo = 2
    cmid = []
    for _ in MODEL_ORDER:
        hi = lo + 1
        cmid.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
        lo = hi + 1
    lines.append("    " + " ".join(cmid))
    head2 = []
    for _ in MODEL_ORDER:
        head2.extend([r"$\rho_{\mathrm{var}}$", r"$\rho_{\mathrm{ent}}$"])
    lines.append("    & " + " & ".join(head2) + r" \\")
    lines.append(r"    \midrule")
    return lines


def _table_header_mse(nmodel: int) -> List[str]:
    colspec = "l" + "rr" * nmodel
    lines = [
        rf"  \begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        r"    \toprule",
    ]
    head1 = [r"\multicolumn{2}{c}{" + MODEL_HEADER[m] + r"}" for m in MODEL_ORDER]
    lines.append("    Param. & " + " & ".join(head1) + r" \\")
    lo = 2
    cmid = []
    for _ in MODEL_ORDER:
        hi = lo + 1
        cmid.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
        lo = hi + 1
    lines.append("    " + " ".join(cmid))
    head2 = []
    for _ in MODEL_ORDER:
        head2.extend([r"$\mathrm{MSE}_{\mathrm{var}}$", r"$\mathrm{MSE}_{\mathrm{mm}}$"])
    lines.append("    & " + " & ".join(head2) + r" \\")
    lines.append(r"    \midrule")
    return lines


def build_latex_correlation(
    ovb_root: Path,
    mm_stats_root: Path,
    fixed_beta2: float,
    fixed_rho: float,
    stats_date_suffix: Optional[str],
) -> str:
    nmodel = len(MODEL_ORDER)
    ncol = 1 + 2 * nmodel

    lines: List[str] = [
        "% Auto-generated by scripts/export_ovb_omitted_correlation_mse_latex.py",
        r"% OVB omitted model: $\rho_{\mathrm{var}}$ from legacy sweep Excel; $\rho_{\mathrm{ent}}$ moment-matched.",
        r"% Preamble: \usepackage{booktabs}",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{OVB experiment (omitted-variable model on the 1D evaluation grid): Pearson correlation between epistemic and aleatoric uncertainty. "
        r"$\rho_{\mathrm{var}}$ from variance-based statistics in legacy OVB sweep workbooks; $\rho_{\mathrm{ent}}$ from moment-matched analytical entropy (batch Excel).}",
        r"  \label{tab:ovb-omitted-corr-variance-moment-entropy}",
    ]
    lines.extend(_table_header_corr(nmodel))

    for func_type, noise_type in CONDITIONS:
        scen = COND_LABEL[(func_type, noise_type)]
        lines.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l}}{{\textbf{{{scen}}}}} \\")

        for sweep in ("rho", "beta2"):
            if sweep == "rho":
                sub = rf"\textit{{Varying $\rho$ (fixed $\beta_2 = {fixed_beta2:g}$)}}"
            else:
                sub = rf"\textit{{Varying $\beta_2$ (fixed $\rho$ slice; default target $\rho = {fixed_rho:g}$)}}"
            lines.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l}}{{{sub}}} \\")

            var_by = load_stats_by_model_variance(
                ovb_root,
                func_type,  # type: ignore[arg-type]
                noise_type,  # type: ignore[arg-type]
                sweep,  # type: ignore[arg-type]
                fixed_beta2,
                fixed_rho,
                stats_date_suffix=stats_date_suffix,
            )
            mm_by = load_stats_by_model_entropy(
                mm_stats_root,
                func_type,  # type: ignore[arg-type]
                noise_type,  # type: ignore[arg-type]
                sweep,  # type: ignore[arg-type]
                fixed_beta2,
                fixed_rho,
            )
            vals = _collect_varying_values(var_by, mm_by, sweep)
            if not vals:
                lines.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l@{{}}}}{{No data.}} \\")
            else:
                for v in vals:
                    pv = _fmt_param_value(sweep, v)
                    cells_v = [_fmt_rho(_corr_var_at(var_by.get(m), sweep, v)) for m in MODEL_ORDER]
                    cells_e = [_fmt_rho(_corr_ent_at(mm_by.get(m), sweep, v)) for m in MODEL_ORDER]
                    interleaved: List[str] = []
                    for a, b in zip(cells_v, cells_e):
                        interleaved.extend([a, b])
                    lines.append(f"    {pv} & " + " & ".join(interleaved) + r" \\")

            lines.append(r"    \addlinespace")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def build_latex_mse(
    ovb_root: Path,
    mm_stats_root: Path,
    fixed_beta2: float,
    fixed_rho: float,
    stats_date_suffix: Optional[str],
) -> str:
    nmodel = len(MODEL_ORDER)
    ncol = 1 + 2 * nmodel

    lines: List[str] = [
        "% Second table: MSE (legacy OVB vs moment-matched)",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{OVB experiment (omitted model): mean squared error of the predictive mean on the evaluation grid. "
        r"$\mathrm{MSE}_{\mathrm{var}}$ from legacy sweep workbooks; $\mathrm{MSE}_{\mathrm{mm}}$ from moment-matched batch.}",
        r"  \label{tab:ovb-omitted-mse-variance-moment}",
    ]
    lines.extend(_table_header_mse(nmodel))

    for func_type, noise_type in CONDITIONS:
        scen = COND_LABEL[(func_type, noise_type)]
        lines.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l}}{{\textbf{{{scen}}}}} \\")

        for sweep in ("rho", "beta2"):
            if sweep == "rho":
                sub = rf"\textit{{Varying $\rho$ (fixed $\beta_2 = {fixed_beta2:g}$)}}"
            else:
                sub = rf"\textit{{Varying $\beta_2$ (target $\rho = {fixed_rho:g}$)}}"
            lines.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l}}{{{sub}}} \\")

            var_by = load_stats_by_model_variance(
                ovb_root,
                func_type,  # type: ignore[arg-type]
                noise_type,  # type: ignore[arg-type]
                sweep,  # type: ignore[arg-type]
                fixed_beta2,
                fixed_rho,
                stats_date_suffix=stats_date_suffix,
            )
            mm_by = load_stats_by_model_entropy(
                mm_stats_root,
                func_type,  # type: ignore[arg-type]
                noise_type,  # type: ignore[arg-type]
                sweep,  # type: ignore[arg-type]
                fixed_beta2,
                fixed_rho,
            )
            vals = _collect_varying_values(var_by, mm_by, sweep)
            if not vals:
                lines.append(rf"    \multicolumn{{{ncol}}}{{@{{}}l@{{}}}}{{No data.}} \\")
            else:
                for v in vals:
                    pv = _fmt_param_value(sweep, v)
                    cells_v = [_fmt_mse(_mse_var_at(var_by.get(m), sweep, v)) for m in MODEL_ORDER]
                    cells_mm = [_fmt_mse(_mse_mm_at(mm_by.get(m), sweep, v)) for m in MODEL_ORDER]
                    interleaved = []
                    for a, b in zip(cells_v, cells_mm):
                        interleaved.extend([a, b])
                    lines.append(f"    {pv} & " + " & ".join(interleaved) + r" \\")

            lines.append(r"    \addlinespace")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ovb-root",
        type=Path,
        default=project_root / "results" / "ovb",
    )
    parser.add_argument(
        "--stats-mm",
        type=Path,
        default=project_root
        / "results"
        / "entropy_recomputed_moment_matched_batch"
        / "statistics",
    )
    parser.add_argument("--fixed-beta2", type=float, default=1.0)
    parser.add_argument("--fixed-rho", type=float, default=0.75)
    parser.add_argument("--stats-date-suffix", type=str, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "results" / "ovb" / "tables" / "ovb_omitted_correlation_mse.tex",
    )
    args = parser.parse_args()
    ovb_root = args.ovb_root.resolve()
    mm_root = args.stats_mm.resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = str(args.stats_date_suffix).strip() if args.stats_date_suffix else None
    tex = (
        build_latex_correlation(
            ovb_root,
            mm_root,
            float(args.fixed_beta2),
            float(args.fixed_rho),
            suffix,
        )
        + build_latex_mse(
            ovb_root,
            mm_root,
            float(args.fixed_beta2),
            float(args.fixed_rho),
            suffix,
        )
    )
    out_path.write_text(tex, encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
