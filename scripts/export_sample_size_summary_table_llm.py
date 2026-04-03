"""
Export sample-size summary tables for variance vs moment-matched entropy (AU, EU, rho, MSE).

**Variance** rows come from ``results/sample_size/statistics`` CSVs (same as
``plot_sample_size_summary_4x2`` / consolidated variance panels): normalized
spatial averages of aleatoric/epistemic **variance-derived** uncertainties,
``Correlation_Epi_Ale``, ``MSE``.

**Entropy (moment-matched)** rows come from moment-matched batch Excel under
``entropy_recomputed_moment_matched_batch/statistics`` (same as
``load_sample_size_entropy_moment_matched``): min--max normalized AU/EU/TU on
the grid, ``Correlation_Epi_Ale``, ``MSE``.

Writes Markdown (readable for Claude) and CSV (one row per scenario × pct × model).

Also writes baseline (default **100%** training) **LaTeX** tables unless ``--no-tex``:

- ``--out-tex-corr``: $\rho$ for variance vs **moment-matched batch** entropy (from
  ``entropy_recomputed_moment_matched_batch/statistics/.../moment_matched_entropy_sample_size*.xlsx``).
- ``out-tex-mse``: single MSE table (same value for both decompositions).

With ``--figure-out``, saves two small-multiples PNGs: normalized **AU** and **Pearson ρ**
(variance solid vs moment-matched entropy dashed).

Usage::

    python scripts/export_sample_size_summary_table_llm.py
    python scripts/export_sample_size_summary_table_llm.py --figure-out results/sample_size/plots/sample_size_AU_var_vs_ent_mm_overview.png
"""
from __future__ import annotations

import argparse
import math
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.moment_matched_summary_loaders import load_sample_size_entropy_moment_matched
from utils.regression_summary_panels import MODEL_KEYS

_pss = runpy.run_path(
    str(project_root / "scripts" / "plot_sample_size_summary_4x2.py"),
    run_name="<load_sample_size_stats>",
)
load_sample_size_stats = _pss["load_sample_size_stats"]

MODEL_ORDER = list(MODEL_KEYS.keys())

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


def _collect_percentages(
    var_by: Dict[str, pd.DataFrame],
    mm_by: Dict[str, pd.DataFrame],
) -> List[float]:
    s: Set[float] = set()
    for d in (*var_by.values(), *mm_by.values()):
        if d is None or d.empty or "Percentage" not in d.columns:
            continue
        for x in pd.to_numeric(d["Percentage"], errors="coerce").dropna().unique():
            s.add(float(x))
    return sorted(s)


def _row_at_pct(df: Optional[pd.DataFrame], pct: float) -> Optional[pd.Series]:
    if df is None or df.empty or "Percentage" not in df.columns:
        return None
    sub = df[pd.to_numeric(df["Percentage"], errors="coerce").sub(pct).abs() < 1e-6]
    if sub.empty:
        return None
    return sub.iloc[0]


def _f(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fmt_md(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def build_rows(
    mm_stats_root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for func_type, noise_type in CONDITIONS:
        scen = COND_LABEL[(func_type, noise_type)]
        var_by = load_sample_size_stats(func_type, noise_type, "variance")  # type: ignore[arg-type]
        mm_by = load_sample_size_entropy_moment_matched(mm_stats_root, func_type, noise_type)
        pcts = _collect_percentages(var_by, mm_by)
        for pct in pcts:
            for model in MODEL_ORDER:
                rv = _row_at_pct(var_by.get(model), pct)
                re = _row_at_pct(mm_by.get(model), pct)
                row: Dict[str, Any] = {
                    "scenario": scen,
                    "function": func_type,
                    "noise": noise_type,
                    "training_pct": int(pct) if abs(pct - round(pct)) < 1e-9 else pct,
                    "model": model,
                    "AU_var_norm": _f(rv["Avg_Aleatoric_norm"]) if rv is not None else None,
                    "EU_var_norm": _f(rv["Avg_Epistemic_norm"]) if rv is not None else None,
                    "TU_var_norm": _f(rv["Avg_Total_norm"]) if rv is not None and "Avg_Total_norm" in rv.index else None,
                    "rho_var": _f(rv["Correlation_Epi_Ale"]) if rv is not None else None,
                    "MSE_var": _f(rv["MSE"]) if rv is not None else None,
                    "AU_ent_mm_norm": _f(re["Avg_Aleatoric_Entropy_norm"]) if re is not None else None,
                    "EU_ent_mm_norm": _f(re["Avg_Epistemic_Entropy_norm"]) if re is not None else None,
                    "TU_ent_mm_norm": _f(re["Avg_Total_Entropy_norm"]) if re is not None else None,
                    "rho_ent_mm": _f(re["Correlation_Epi_Ale"]) if re is not None else None,
                    "MSE_ent_mm": _f(re["MSE"]) if re is not None else None,
                }
                rows.append(row)
    return rows


def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def save_au_overview_figure(df: pd.DataFrame, path: Path) -> None:
    """2×2 panels: one per scenario; lines = models; solid = AU (variance), dashed = AU (moment-matched entropy)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(MODEL_ORDER)))
    for ax, (func_type, noise_type), color_cycle in zip(
        axes.flat,
        CONDITIONS,
        [colors] * 4,
    ):
        scen = COND_LABEL[(func_type, noise_type)]
        sub = df[(df["function"] == func_type) & (df["noise"] == noise_type)]
        if sub.empty:
            ax.set_title(scen)
            continue
        pcts = sorted({float(x) for x in sub["training_pct"].unique()})
        for mi, model in enumerate(MODEL_ORDER):
            c = color_cycle[mi]
            xv, yv = [], []
            xe, ye = [], []
            sub_m = sub[sub["model"] == model]
            for p in pcts:
                r = sub_m[np.isclose(sub_m["training_pct"].astype(float), p, rtol=0, atol=1e-6)]
                if r.empty:
                    continue
                row = r.iloc[0]
                av = row["AU_var_norm"]
                ae = row["AU_ent_mm_norm"]
                if pd.notna(av):
                    xv.append(p)
                    yv.append(float(av))
                if pd.notna(ae):
                    xe.append(p)
                    ye.append(float(ae))
            if yv:
                ax.plot(xv, yv, "-", color=c, linewidth=1.8, label=f"{model} (var.)")
            if ye:
                ax.plot(xe, ye, "--", color=c, linewidth=1.4, alpha=0.85, label=f"{model} (ent. mm)")
        ax.set_title(scen, fontsize=10)
        ax.set_ylim(0, 1.55)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training %")
    axes[0, 0].set_ylabel("Normalized avg. aleatoric uncertainty")
    axes[1, 0].set_ylabel("Normalized avg. aleatoric uncertainty")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Sample size: AU from variance summaries (solid) vs moment-matched entropy (dashed)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rho_overview_figure(df: pd.DataFrame, path: Path) -> None:
    """Same layout as AU figure; y = Pearson ρ (epistemic vs aleatoric) on the grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(MODEL_ORDER)))
    for ax, (func_type, noise_type), color_cycle in zip(axes.flat, CONDITIONS, [colors] * 4):
        scen = COND_LABEL[(func_type, noise_type)]
        sub = df[(df["function"] == func_type) & (df["noise"] == noise_type)]
        if sub.empty:
            ax.set_title(scen)
            continue
        pcts = sorted({float(x) for x in sub["training_pct"].unique()})
        for mi, model in enumerate(MODEL_ORDER):
            c = color_cycle[mi]
            xv, yv = [], []
            xe, ye = [], []
            sub_m = sub[sub["model"] == model]
            for p in pcts:
                r = sub_m[np.isclose(sub_m["training_pct"].astype(float), p, rtol=0, atol=1e-6)]
                if r.empty:
                    continue
                row = r.iloc[0]
                rv = row["rho_var"]
                re = row["rho_ent_mm"]
                if pd.notna(rv):
                    xv.append(p)
                    yv.append(float(rv))
                if pd.notna(re):
                    xe.append(p)
                    ye.append(float(re))
            if yv:
                ax.plot(xv, yv, "-", color=c, linewidth=1.8, label=f"{model} (var.)")
            if ye:
                ax.plot(xe, ye, "--", color=c, linewidth=1.4, alpha=0.85, label=f"{model} (ent. mm)")
        ax.set_title(scen, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training %")
    axes[0, 0].set_ylabel(r"$\rho$ (epistemic vs aleatoric)")
    axes[1, 0].set_ylabel(r"$\rho$ (epistemic vs aleatoric)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        r"Sample size: $\rho$ from variance summaries (solid) vs moment-matched entropy (dashed)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _latex_num(x: Any, nd: int = 3) -> str:
    v = _f(x)
    if v is None:
        return "---"
    return f"{v:.{nd}f}"


def write_baseline_latex_tables(
    df: pd.DataFrame,
    out_corr: Path,
    out_mse: Path,
    training_pct: float = 100.0,
    nd: int = 3,
) -> None:
    """
    Two LaTeX fragments for thesis: (1) Pearson rho variance vs moment-matched batch entropy;
    (2) single MSE table (identical for both decompositions when both present).

    Expects columns from ``build_rows`` / ``rows_to_dataframe`` including
    ``function``, ``noise``, ``model``, ``training_pct``, ``rho_var``, ``rho_ent_mm``,
    ``MSE_var`` (and optionally ``MSE_ent_mm``).
    """
    sub = df[np.isclose(pd.to_numeric(df["training_pct"], errors="coerce"), training_pct, rtol=0, atol=1e-6)]
    lines_corr = [
        "% Requires \\usepackage{booktabs}",
        "% Entropy rho/MSE from entropy_recomputed_moment_matched_batch/statistics "
        "(*moment_matched_entropy_sample_size*.xlsx)",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{Sample-size baseline ({training_pct:g}\,\% training): Pearson $\rho$ "
        rf"(epistemic vs aleatoric). Variance from \texttt{{results/sample\_size/statistics}}; "
        rf"entropy from moment-matched batch recomputation.}}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{Sinusoidal} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Model} & Homo. & Hetero. & Homo. & Hetero. \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Variance}} \\",
    ]
    for model in MODEL_ORDER:
        cells = []
        for func_type, noise_type in CONDITIONS:
            row = sub[(sub["function"] == func_type) & (sub["noise"] == noise_type) & (sub["model"] == model)]
            rho = row["rho_var"].iloc[0] if len(row) else None
            cells.append(_latex_num(rho, nd))
        lines_corr.append(
            model.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\"
        )
    lines_corr.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{l}{\textit{Entropy (moment-matched batch)}} \\",
        ]
    )
    for model in MODEL_ORDER:
        cells = []
        for func_type, noise_type in CONDITIONS:
            row = sub[(sub["function"] == func_type) & (sub["noise"] == noise_type) & (sub["model"] == model)]
            rho = row["rho_ent_mm"].iloc[0] if len(row) else None
            cells.append(_latex_num(rho, nd))
        lines_corr.append(
            model.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\"
        )
    lines_corr.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    lines_mse = [
        "% Requires \\usepackage{booktabs}",
        "% MSE identical for variance and moment-matched entropy (same predictive mean vs true curve).",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{Sample-size baseline ({training_pct:g}\,\% training): MSE of predictive mean vs.\ true $f(x)$ on the grid.}}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{Sinusoidal} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"& Homo. & Hetero. & Homo. & Hetero. \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = []
        for func_type, noise_type in CONDITIONS:
            row = sub[(sub["function"] == func_type) & (sub["noise"] == noise_type) & (sub["model"] == model)]
            mse = row["MSE_var"].iloc[0] if len(row) else None
            cells.append(_latex_num(mse, nd))
        lines_mse.append(
            model.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\"
        )
    lines_mse.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    out_corr = Path(out_corr).resolve()
    out_mse = Path(out_mse).resolve()
    out_corr.parent.mkdir(parents=True, exist_ok=True)
    out_mse.parent.mkdir(parents=True, exist_ok=True)
    out_corr.write_text("\n".join(lines_corr), encoding="utf-8")
    out_mse.write_text("\n".join(lines_mse), encoding="utf-8")


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "# Sample size experiment — summary metrics (variance vs moment-matched entropy)",
        "",
        "Auto-generated. Use for documentation or LLM context.",
        "",
        "## Data sources",
        "",
        "- **Variance columns** (`*_var_*`): from `results/sample_size/statistics/.../*.csv` "
        "without `_entropy` in the filename (same as thesis variance summary panels). "
        "`AU`/`EU`/`TU` are **min–max normalized** spatial averages of aleatoric/epistemic/total "
        "uncertainty from **variance**; `rho` = Pearson correlation epistemic vs aleatoric along the grid; "
        "`MSE` = mean squared error vs true curve.",
        "",
        "- **Entropy columns** (`*_ent_mm_*`): from **moment-matched analytical entropy** recomputation "
        "(`entropy_recomputed_moment_matched_batch/statistics/.../*moment_matched_entropy_sample_size.xlsx`). "
        "Same normalization and correlation definition as consolidated entropy figures.",
        "",
        "---",
        "",
    ]

    current_scen: Optional[str] = None
    for r in rows:
        if r["scenario"] != current_scen:
            current_scen = r["scenario"]
            lines.append(f"## {current_scen}")
            lines.append("")
            lines.append(
                "| % | Model | AU_var | EU_var | TU_var | ρ_var | MSE_var | "
                "AU_ent_mm | EU_ent_mm | TU_ent_mm | ρ_ent_mm | MSE_ent_mm |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        pct = r["training_pct"]
        pct_s = str(int(pct)) if isinstance(pct, float) and pct == int(pct) else str(pct)
        lines.append(
            f"| {pct_s} | {r['model']} | "
            f"{_fmt_md(r['AU_var_norm'])} | {_fmt_md(r['EU_var_norm'])} | {_fmt_md(r['TU_var_norm'])} | "
            f"{_fmt_md(r['rho_var'])} | {_fmt_md(r['MSE_var'])} | "
            f"{_fmt_md(r['AU_ent_mm_norm'])} | {_fmt_md(r['EU_ent_mm_norm'])} | {_fmt_md(r['TU_ent_mm_norm'])} | "
            f"{_fmt_md(r['rho_ent_mm'])} | {_fmt_md(r['MSE_ent_mm'])} |"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-mm",
        type=Path,
        default=project_root
        / "results"
        / "entropy_recomputed_moment_matched_batch"
        / "statistics",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=project_root
        / "results"
        / "sample_size"
        / "tables"
        / "sample_size_summary_variance_vs_moment_entropy.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=project_root
        / "results"
        / "sample_size"
        / "tables"
        / "sample_size_summary_variance_vs_moment_entropy.csv",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        nargs="?",
        const=project_root
        / "results"
        / "sample_size"
        / "plots"
        / "sample_size_AU_var_vs_ent_mm_overview.png",
        default=None,
        help="Save AU line-plot overview PNG. Pass a path, or use flag with no value for default path.",
    )
    parser.add_argument(
        "--out-tex-corr",
        type=Path,
        default=project_root
        / "results"
        / "sample_size"
        / "tables"
        / "sample_size_baseline_100_corr_var_vs_mm_entropy.tex",
        help="LaTeX: rho table (variance + moment-matched batch entropy).",
    )
    parser.add_argument(
        "--out-tex-mse",
        type=Path,
        default=project_root
        / "results"
        / "sample_size"
        / "tables"
        / "sample_size_baseline_100_mse.tex",
        help="LaTeX: single MSE table (same for both decompositions).",
    )
    parser.add_argument(
        "--baseline-pct",
        type=float,
        default=100.0,
        help="Training %% row to export for LaTeX tables (default 100).",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Skip writing baseline LaTeX fragments.",
    )
    args = parser.parse_args()
    mm_root = Path(args.stats_mm).resolve()
    rows = build_rows(mm_root)
    df = rows_to_dataframe(rows)

    out_md = Path(args.out_md).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(build_markdown(rows), encoding="utf-8")
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print("Wrote", out_md)
    print("Wrote", out_csv, f"({len(df)} rows)")

    if not args.no_tex:
        write_baseline_latex_tables(
            df,
            Path(args.out_tex_corr),
            Path(args.out_tex_mse),
            training_pct=args.baseline_pct,
        )
        print("Wrote", Path(args.out_tex_corr).resolve())
        print("Wrote", Path(args.out_tex_mse).resolve())
    if args.figure_out:
        fp = Path(args.figure_out).resolve()
        save_au_overview_figure(df, fp)
        print("Wrote", fp)
        fp_rho = fp.parent / "sample_size_rho_var_vs_ent_mm_overview.png"
        save_rho_overview_figure(df, fp_rho)
        print("Wrote", fp_rho)


if __name__ == "__main__":
    main()
