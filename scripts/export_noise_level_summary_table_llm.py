"""
Export noise-level summary tables for variance vs moment-matched entropy (AU, EU, rho, MSE).

**Variance** rows come from ``results/noise_level/statistics`` CSVs (same as
``plot_noise_level_summary_4x2`` / consolidated variance panels), for a chosen
noise **distribution** subdirectory (default ``normal``): normalized spatial
averages from **variance**, ``Correlation_Epi_Ale``, ``MSE``.

**Entropy (moment-matched)** rows come from moment-matched batch Excel
(``*moment_matched_entropy_noise_{distribution}.xlsx`` under
``entropy_recomputed_moment_matched_batch/statistics/noise_level/...``).

Writes Markdown and CSV (one row per scenario × τ × model).

With ``--figure-out``, saves AU and ρ overview PNGs (variance solid vs moment-matched entropy dashed).

Usage::

    python scripts/export_noise_level_summary_table_llm.py
    python scripts/export_noise_level_summary_table_llm.py --figure-out
    python scripts/export_noise_level_summary_table_llm.py --distribution normal --figure-out
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

from utils.moment_matched_summary_loaders import load_noise_level_entropy_moment_matched
from utils.regression_summary_panels import MODEL_KEYS

_pnl = runpy.run_path(
    str(project_root / "scripts" / "plot_noise_level_summary_4x2.py"),
    run_name="<load_noise_level_stats>",
)
load_noise_level_stats = _pnl["load_noise_level_stats"]

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


def _collect_taus(
    var_by: Dict[str, pd.DataFrame],
    mm_by: Dict[str, pd.DataFrame],
) -> List[float]:
    s: Set[float] = set()
    for d in (*var_by.values(), *mm_by.values()):
        if d is None or d.empty or "Tau" not in d.columns:
            continue
        for x in pd.to_numeric(d["Tau"], errors="coerce").dropna().unique():
            s.add(float(x))
    return sorted(s)


def _row_at_tau(df: Optional[pd.DataFrame], tau: float) -> Optional[pd.Series]:
    if df is None or df.empty or "Tau" not in df.columns:
        return None
    sub = df[np.isclose(pd.to_numeric(df["Tau"], errors="coerce"), tau, rtol=0, atol=1e-5)]
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


def _fmt_tau_cell(tau: float) -> str:
    if abs(tau - round(tau)) < 1e-6:
        return str(int(round(tau)))
    return str(tau).rstrip("0").rstrip(".") if "." in str(tau) else str(tau)


def build_rows(mm_stats_root: Path, distribution: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for func_type, noise_type in CONDITIONS:
        scen = COND_LABEL[(func_type, noise_type)]
        var_by = load_noise_level_stats(
            func_type,  # type: ignore[arg-type]
            noise_type,  # type: ignore[arg-type]
            "variance",  # type: ignore[arg-type]
            distribution=distribution,
        )
        mm_by = load_noise_level_entropy_moment_matched(
            mm_stats_root, func_type, noise_type, distribution=distribution
        )
        taus = _collect_taus(var_by, mm_by)
        for tau in taus:
            for model in MODEL_ORDER:
                rv = _row_at_tau(var_by.get(model), tau)
                re = _row_at_tau(mm_by.get(model), tau)
                row: Dict[str, Any] = {
                    "scenario": scen,
                    "function": func_type,
                    "noise": noise_type,
                    "distribution": distribution,
                    "tau": float(tau),
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(MODEL_ORDER)))
    for ax, (func_type, noise_type), color_cycle in zip(axes.flat, CONDITIONS, [colors] * 4):
        scen = COND_LABEL[(func_type, noise_type)]
        sub = df[(df["function"] == func_type) & (df["noise"] == noise_type)]
        if sub.empty:
            ax.set_title(scen)
            continue
        taus = sorted({float(x) for x in sub["tau"].unique()})
        for mi, model in enumerate(MODEL_ORDER):
            c = color_cycle[mi]
            xv, yv = [], []
            xe, ye = [], []
            sub_m = sub[sub["model"] == model]
            for t in taus:
                r = sub_m[np.isclose(sub_m["tau"].astype(float), t, rtol=0, atol=1e-5)]
                if r.empty:
                    continue
                row = r.iloc[0]
                av = row["AU_var_norm"]
                ae = row["AU_ent_mm_norm"]
                if pd.notna(av):
                    xv.append(t)
                    yv.append(float(av))
                if pd.notna(ae):
                    xe.append(t)
                    ye.append(float(ae))
            if yv:
                ax.plot(xv, yv, "-", color=c, linewidth=1.8, label=f"{model} (var.)")
            if ye:
                ax.plot(xe, ye, "--", color=c, linewidth=1.4, alpha=0.85, label=f"{model} (ent. mm)")
        ax.set_title(scen, fontsize=10)
        ax.set_ylim(0, 1.55)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$\tau$ (noise scale)")
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
        "Noise level: AU from variance summaries (solid) vs moment-matched entropy (dashed)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rho_overview_figure(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(MODEL_ORDER)))
    for ax, (func_type, noise_type), color_cycle in zip(axes.flat, CONDITIONS, [colors] * 4):
        scen = COND_LABEL[(func_type, noise_type)]
        sub = df[(df["function"] == func_type) & (df["noise"] == noise_type)]
        if sub.empty:
            ax.set_title(scen)
            continue
        taus = sorted({float(x) for x in sub["tau"].unique()})
        for mi, model in enumerate(MODEL_ORDER):
            c = color_cycle[mi]
            xv, yv = [], []
            xe, ye = [], []
            sub_m = sub[sub["model"] == model]
            for t in taus:
                r = sub_m[np.isclose(sub_m["tau"].astype(float), t, rtol=0, atol=1e-5)]
                if r.empty:
                    continue
                row = r.iloc[0]
                rv = row["rho_var"]
                re = row["rho_ent_mm"]
                if pd.notna(rv):
                    xv.append(t)
                    yv.append(float(rv))
                if pd.notna(re):
                    xe.append(t)
                    ye.append(float(re))
            if yv:
                ax.plot(xv, yv, "-", color=c, linewidth=1.8, label=f"{model} (var.)")
            if ye:
                ax.plot(xe, ye, "--", color=c, linewidth=1.4, alpha=0.85, label=f"{model} (ent. mm)")
        ax.set_title(scen, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle=":")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$\tau$ (noise scale)")
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
        r"Noise level: $\rho$ from variance summaries (solid) vs moment-matched entropy (dashed)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_markdown(rows: List[Dict[str, Any]], distribution: str) -> str:
    lines: List[str] = [
        "# Noise level experiment — summary metrics (variance vs moment-matched entropy)",
        "",
        f"Distribution: **{distribution}**. Auto-generated for documentation or LLM context.",
        "",
        "## Data sources",
        "",
        "- **Variance columns** (`*_var_*`): from `results/noise_level/statistics/.../{distribution}/` "
        "CSV files without `_entropy` in the filename (same as thesis noise-level variance summary panels).",
        "",
        "- **Entropy columns** (`*_ent_mm_*`): from **moment-matched analytical entropy** "
        f"(`*moment_matched_entropy_noise_{distribution}.xlsx`).",
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
                "| τ | Model | AU_var | EU_var | TU_var | ρ_var | MSE_var | "
                "AU_ent_mm | EU_ent_mm | TU_ent_mm | ρ_ent_mm | MSE_ent_mm |"
            )
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        tau_s = _fmt_tau_cell(float(r["tau"]))
        lines.append(
            f"| {tau_s} | {r['model']} | "
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
        "--distribution",
        type=str,
        default="normal",
        help="Noise distribution subfolder / moment-matched workbook suffix",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=project_root
        / "results"
        / "noise_level"
        / "tables"
        / "noise_level_summary_variance_vs_moment_entropy.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=project_root
        / "results"
        / "noise_level"
        / "tables"
        / "noise_level_summary_variance_vs_moment_entropy.csv",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        nargs="?",
        const=project_root
        / "results"
        / "noise_level"
        / "plots"
        / "noise_level_AU_var_vs_ent_mm_overview.png",
        default=None,
        help="Save AU / rho overview PNGs. Flag only → default path for AU plot.",
    )
    args = parser.parse_args()
    dist = str(args.distribution).strip() or "normal"
    mm_root = Path(args.stats_mm).resolve()
    rows = build_rows(mm_root, dist)
    df = rows_to_dataframe(rows)

    out_md = Path(args.out_md).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(build_markdown(rows, dist), encoding="utf-8")
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print("Wrote", out_md)
    print("Wrote", out_csv, f"({len(df)} rows)")
    if args.figure_out:
        fp = Path(args.figure_out).resolve()
        save_au_overview_figure(df, fp)
        print("Wrote", fp)
        fp_rho = fp.parent / "noise_level_rho_var_vs_ent_mm_overview.png"
        save_rho_overview_figure(df, fp_rho)
        print("Wrote", fp_rho)


if __name__ == "__main__":
    main()
