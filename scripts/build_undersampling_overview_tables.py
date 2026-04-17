"""
Build undersampling overview LaTeX tables from Excel summaries under
``results/undersampling/statistics/undersampling/{noise}/{func}/``.

For each of four conditions (linear/sin × homo/hetero), produces:
  * One wide table: Entropy (norm.) | Variance (norm.), each with Undersampled
    and Well-sampled blocks of AU, EU, Pearson Corr, MSE.
  * One MSE-only table: same scenarios, Undersampled vs Well-sampled columns.

Writes:
  ``results/undersampling/tables/appendix_undersampling_tables_generated.tex``

Requires: \\usepackage{booktabs}
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent

STATS_BASE = project_root / "results" / "undersampling" / "statistics" / "undersampling"
OUT_TEX = project_root / "results" / "undersampling" / "tables" / "appendix_undersampling_tables_generated.tex"

CONDITIONS: List[Tuple[str, str, str, str]] = [
    ("linear", "homoscedastic", "Linear", "homoscedastic"),
    ("linear", "heteroscedastic", "Linear", "heteroscedastic"),
    ("sin", "homoscedastic", "Sinusoidal", "homoscedastic"),
    ("sin", "heteroscedastic", "Sinusoidal", "heteroscedastic"),
]

MODEL_ORDER = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]


def _model_from_filename(name: str) -> Optional[str]:
    """Map Excel filename to display model name (first match wins)."""
    if "Deep_Ensemble" in name:
        return "Deep Ensemble"
    if "MC_Dropout" in name:
        return "MC Dropout"
    if "BAMLSS" in name:
        return "BAMLSS"
    if "BNN" in name:
        return "BNN"
    return None


def _func_title(func_type: str) -> str:
    return "Linear" if func_type == "linear" else "Sinusoidal"


def _glob_suffix(
    region: str,
    func_title: str,
    noise_type: str,
    metric: str,
) -> str:
    """Expected end of filename after date_modelprefix_."""
    return f"{region}_uncertainties_summary_{func_title}_{noise_type}_{metric}.xlsx"


def _latest_per_model(cond_dir: Path, region: str, metric: str) -> Dict[str, Path]:
    """Latest Excel (by full filename sort descending) per model for given region/metric."""
    func_title = _func_title(cond_dir.name)
    noise_type = cond_dir.parent.name
    suffix = _glob_suffix(region, func_title, noise_type, metric)
    files = [p for p in cond_dir.glob(f"*_{suffix}") if p.is_file()]
    by_model: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        m = _model_from_filename(p.name)
        if m:
            by_model[m].append(p)
    out: Dict[str, Path] = {}
    for m, lst in by_model.items():
        lst.sort(key=lambda x: x.name, reverse=True)
        out[m] = lst[0]
    return out


def _round_val(x, ndec: int = 4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "---"
    try:
        return round(float(x), ndec)
    except (TypeError, ValueError):
        return "---"


def _read_row(path: Path, measure: str) -> Optional[pd.Series]:
    if not path.is_file():
        return None
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df.iloc[0]


def _pack_measure(row: Optional[pd.Series], measure: str) -> Tuple:
    if row is None:
        return ("---", "---", "---", "---")
    if measure == "variance":
        au, eu = "Avg_Aleatoric_norm", "Avg_Epistemic_norm"
    else:
        au, eu = "Avg_Aleatoric_Entropy_norm", "Avg_Epistemic_Entropy_norm"
    corr, mse = "Correlation_Epi_Ale", "MSE"
    return (
        _round_val(row[au]) if au in row else "---",
        _round_val(row[eu]) if eu in row else "---",
        _round_val(row[corr]) if corr in row else "---",
        _round_val(row[mse]) if mse in row else "---",
    )


def _tex_escape_model(name: str) -> str:
    return name.replace("_", r"\_")


def build_combined_table(
    func_type: str,
    noise_type: str,
    func_title: str,
    noise_title: str,
) -> str:
    cond_dir = STATS_BASE / noise_type / func_type
    if not cond_dir.is_dir():
        return f"% No directory: {cond_dir}\n"

    ent_u = _latest_per_model(cond_dir, "Undersampled", "entropy")
    ent_w = _latest_per_model(cond_dir, "Well_sampled", "entropy")
    var_u = _latest_per_model(cond_dir, "Undersampled", "variance")
    var_w = _latest_per_model(cond_dir, "Well_sampled", "variance")

    lines: List[str] = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{Undersampling summary (normalized AU/EU, Pearson correlation $\\rho$, MSE): "
        f"{func_title}, {noise_title}. "
        f"Left: entropy-based; right: variance-based. "
        f"\\emph{{us}} = spatially undersampled band(s); \\emph{{ws}} = well-sampled band(s).}}",
        f"  \\label{{tab:undersampling-{func_type}-{noise_type}-combined}}",
        "  \\begin{tabular}{l*{16}{c}}",
        "  \\toprule",
        "  \\multicolumn{1}{c}{}"
        " & \\multicolumn{8}{c}{Entropy (normalized)}"
        " & \\multicolumn{8}{c}{Variance (normalized)} \\\\",
        "  \\cmidrule(lr){2-9}\\cmidrule(lr){10-17}",
        "  Model & \\multicolumn{4}{c}{\\emph{us}} & \\multicolumn{4}{c}{\\emph{ws}}"
        " & \\multicolumn{4}{c}{\\emph{us}} & \\multicolumn{4}{c}{\\emph{ws}} \\\\",
        "  \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\\cmidrule(lr){14-17}",
        "  & AU & EU & $\\rho$ & MSE & AU & EU & $\\rho$ & MSE"
        " & AU & EU & $\\rho$ & MSE & AU & EU & $\\rho$ & MSE \\\\",
        "  \\midrule",
    ]

    for model in MODEL_ORDER:
        eu = ent_u.get(model)
        ew = ent_w.get(model)
        vu = var_u.get(model)
        vw = var_w.get(model)
        reu = _read_row(eu, "entropy") if eu else None
        rew = _read_row(ew, "entropy") if ew else None
        rvu = _read_row(vu, "variance") if vu else None
        rvw = _read_row(vw, "variance") if vw else None
        p_eu = _pack_measure(reu, "entropy")
        p_ew = _pack_measure(rew, "entropy")
        p_vu = _pack_measure(rvu, "variance")
        p_vw = _pack_measure(rvw, "variance")
        row = (
            f"  {_tex_escape_model(model)}"
            f" & {p_eu[0]} & {p_eu[1]} & {p_eu[2]} & {p_eu[3]}"
            f" & {p_ew[0]} & {p_ew[1]} & {p_ew[2]} & {p_ew[3]}"
            f" & {p_vu[0]} & {p_vu[1]} & {p_vu[2]} & {p_vu[3]}"
            f" & {p_vw[0]} & {p_vw[1]} & {p_vw[2]} & {p_vw[3]} \\\\"
        )
        lines.append(row)

    lines.extend(
        [
            "  \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_mse_table(
    func_type: str,
    noise_type: str,
    func_title: str,
    noise_title: str,
) -> str:
    """MSE only from variance Undersampled / Well_sampled (same MSE as entropy files)."""
    cond_dir = STATS_BASE / noise_type / func_type
    if not cond_dir.is_dir():
        return f"% No directory: {cond_dir}\n"

    var_u = _latest_per_model(cond_dir, "Undersampled", "variance")
    var_w = _latest_per_model(cond_dir, "Well_sampled", "variance")

    lines: List[str] = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{Undersampling MSE: {func_title}, {noise_title}. "
        f"\\emph{{us}} = undersampled; \\emph{{ws}} = well-sampled (same regions as in uncertainty tables).}}",
        f"  \\label{{tab:undersampling-{func_type}-{noise_type}-mse}}",
        "  \\begin{tabular}{lcc}",
        "  \\toprule",
        "  Model & MSE (\\emph{us}) & MSE (\\emph{ws}) \\\\",
        "  \\midrule",
    ]
    for model in MODEL_ORDER:
        ru = _read_row(var_u.get(model), "variance") if var_u.get(model) else None
        rw = _read_row(var_w.get(model), "variance") if var_w.get(model) else None
        mu = _round_val(ru["MSE"]) if ru is not None and "MSE" in ru else "---"
        mw = _round_val(rw["MSE"]) if rw is not None and "MSE" in rw else "---"
        lines.append(f"  {_tex_escape_model(model)} & {mu} & {mw} \\\\")

    lines.extend(
        [
            "  \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("This script requires openpyxl: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    if not STATS_BASE.is_dir():
        print("Statistics directory not found:", STATS_BASE, file=sys.stderr)
        sys.exit(1)

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = [
        "% Auto-generated by scripts/build_undersampling_overview_tables.py",
        "% Undersampling: Entropy | Variance combined tables + MSE tables.",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]

    for func_type, noise_type, func_title, noise_title in CONDITIONS:
        parts.append(f"% --- {func_title}, {noise_title} ---")
        parts.append("\\subsection{" + func_title + ", " + noise_title + "}")
        parts.append(build_combined_table(func_type, noise_type, func_title, noise_title))
        parts.append(build_mse_table(func_type, noise_type, func_title, noise_title))
        parts.append("")

    OUT_TEX.write_text("\n".join(parts), encoding="utf-8")
    print("Wrote:", OUT_TEX)


if __name__ == "__main__":
    main()
