"""
Build noise-level overview tables from results/noise_level/statistics and write LaTeX.

Reads:
- Combined Excel: statistics/noise_level/noise_level_{noise_type}_{distribution}_combined.xlsx
- Or per-model files: statistics/{noise_type}/{func_type}/{distribution}/*.xlsx (variance)
  and statistics/{noise_type}/{func_type}/*_entropy*.xlsx (entropy)

Output: results/noise_level/noise_level_tables_generated.tex (for \\input from appendix/main).
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_BASE = PROJECT_ROOT / "results" / "noise_level" / "statistics"
OUT_TEX = PROJECT_ROOT / "results" / "noise_level" / "noise_level_tables_generated.tex"
DISTRIBUTION = "normal"

CONDITIONS = [
    ("linear", "homoscedastic", "Linear", "Homoscedastic"),
    ("linear", "heteroscedastic", "Linear", "Heteroscedastic"),
    ("sin", "homoscedastic", "Sinusoidal", "Homoscedastic"),
    ("sin", "heteroscedastic", "Sinusoidal", "Heteroscedastic"),
]
MODEL_ORDER = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]

MODEL_KEYS = {
    "Deep Ensemble": "Deep_Ensemble",
    "MC Dropout": "MC_Dropout",
    "BNN": "BNN",
    "BAMLSS": "BAMLSS",
}


def _round_val(x, ndec=3):
    if pd.isna(x):
        return "---"
    try:
        return str(round(float(x), ndec))
    except (TypeError, ValueError):
        return str(x)


def _load_from_combined(noise_type: str, func_type: str, distribution: str):
    """Load variance and entropy DataFrames from combined Excel. Returns (variance_df dict, entropy_df dict) per model."""
    combined_dir = STATS_BASE / "noise_level"
    fname = combined_dir / f"noise_level_{noise_type}_{distribution}_combined.xlsx"
    if not fname.exists():
        return {}, {}

    variance_by_model = {}
    entropy_by_model = {}
    try:
        xl = pd.ExcelFile(fname)
    except Exception as e:
        print(f"  Warning: could not read {fname}: {e}")
        return {}, {}

    for sheet in xl.sheet_names:
        model_display = None
        for disp, key in MODEL_KEYS.items():
            if key in sheet and f"_{func_type}" in sheet:
                model_display = disp
                break
        if model_display is None:
            continue

        try:
            df = pd.read_excel(xl, sheet_name=sheet)
        except Exception:
            continue
        if "Tau" not in df.columns or df.empty:
            continue

        # Variance columns (no _entropy suffix)
        au_var = "Avg_Aleatoric_norm"
        eu_var = "Avg_Epistemic_norm"
        if au_var in df.columns and eu_var in df.columns:
            variance_by_model[model_display] = df.copy()

        # Entropy columns: either plain or with _entropy suffix (when merged)
        au_ent = "Avg_Aleatoric_Entropy_norm"
        eu_ent = "Avg_Epistemic_Entropy_norm"
        au_ent_suf = "Avg_Aleatoric_Entropy_norm_entropy"
        eu_ent_suf = "Avg_Epistemic_Entropy_norm_entropy"
        if au_ent_suf in df.columns:
            au_ent, eu_ent = au_ent_suf, eu_ent_suf
        if au_ent in df.columns and eu_ent in df.columns:
            entropy_by_model[model_display] = df.copy()

    return variance_by_model, entropy_by_model


def _load_per_model_files(noise_type: str, func_type: str, distribution: str, measure: str):
    """Load one DataFrame per model from statistics/{noise_type}/{func_type}/ or .../distribution/."""
    func_name = "Linear" if func_type == "linear" else "Sinusoidal"
    base = STATS_BASE / noise_type / func_type
    if measure == "variance":
        search_dir = base / distribution if (base / distribution).exists() else base
        pattern = f"*uncertainties_summary_{func_name}_{noise_type}_{distribution}.xlsx"
    else:
        search_dir = base
        pattern = f"*uncertainties_summary_{func_name}_{noise_type}_{distribution}_entropy.xlsx"
    if not search_dir.exists():
        return {}
    files = list(search_dir.glob(pattern))
    if not files:
        files = list(search_dir.glob("*.xlsx")) + list(search_dir.glob("*.csv"))
    out = {}
    for f in sorted(files):
        for disp, key in MODEL_KEYS.items():
            if key not in f.stem:
                continue
            if (measure == "entropy") != ("_entropy" in f.stem or "entropy" in f.name.lower()):
                continue
            try:
                df = pd.read_excel(f) if f.suffix.lower() == ".xlsx" else pd.read_csv(f)
            except Exception:
                continue
            if "Tau" not in df.columns:
                continue
            out[disp] = df
            break
    return out


def load_stats(noise_type: str, func_type: str, distribution: str):
    """Return (variance_by_model, entropy_by_model) each dict model_display -> DataFrame."""
    var_d, ent_d = _load_from_combined(noise_type, func_type, distribution)
    if not var_d:
        var_d = _load_per_model_files(noise_type, func_type, distribution, "variance")
    if not ent_d:
        ent_d = _load_per_model_files(noise_type, func_type, distribution, "entropy")
    return var_d, ent_d


def _infer_entropy_columns(entropy_by_model):
    """Return (au_col, eu_col) for entropy DataFrames."""
    if not entropy_by_model:
        return "Avg_Aleatoric_Entropy_norm", "Avg_Epistemic_Entropy_norm"
    df = next(iter(entropy_by_model.values()))
    if "Avg_Aleatoric_Entropy_norm_entropy" in df.columns:
        return "Avg_Aleatoric_Entropy_norm_entropy", "Avg_Epistemic_Entropy_norm_entropy"
    return "Avg_Aleatoric_Entropy_norm", "Avg_Epistemic_Entropy_norm"


def build_table_latex(func_type: str, noise_type: str, func_title: str, noise_title: str,
                      measure: str, variance_by_model: dict, entropy_by_model: dict) -> str:
    """One table per (func_type, noise_type, measure): rows = tau, columns = 4 models, cells = AU / EU."""
    if measure == "variance":
        data_by_model = variance_by_model
        au_col, eu_col = "Avg_Aleatoric_norm", "Avg_Epistemic_norm"
    else:
        data_by_model = entropy_by_model
        au_col, eu_col = _infer_entropy_columns(entropy_by_model)

    if not data_by_model:
        return f"% No data for {func_type} {noise_type} {measure}\n"

    # Collect all tau values (union from all models)
    taus = []
    for df in data_by_model.values():
        for t in df["Tau"].dropna().unique():
            t = float(t) if not isinstance(t, (int, float)) else t
            if t not in taus:
                taus.append(t)
    taus = sorted(taus)

    metric_cap = "Variance" if measure == "variance" else "Entropy"
    caption = f"Noise level regression — {func_title}, {noise_title} — {metric_cap} decomposition (normalized AU / EU)."
    label = f"tab:noise-{func_type}-{noise_type}-{measure}"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lcccccccc}",
        "  \\toprule",
        "  $\\tau$ & \\multicolumn{2}{c}{Deep Ensemble} & \\multicolumn{2}{c}{MC Dropout} & \\multicolumn{2}{c}{BNN} & \\multicolumn{2}{c}{BAMLSS} \\\\",
        "  & AU & EU & AU & EU & AU & EU & AU & EU \\\\",
        "  \\midrule",
    ]

    for tau in taus:
        row = [str(tau)]
        for model in MODEL_ORDER:
            df = data_by_model.get(model)
            if df is None:
                row.extend(["---", "---"])
                continue
            tau_vals = pd.to_numeric(df["Tau"], errors="coerce")
            r = df[np.isclose(tau_vals, float(tau), rtol=1e-9)]
            if r.empty:
                row.extend(["---", "---"])
            else:
                a = _round_val(r[au_col].iloc[0]) if au_col in r.columns else "---"
                e = _round_val(r[eu_col].iloc[0]) if eu_col in r.columns else "---"
                row.extend([a, e])
        lines.append("  " + " & ".join(row) + " \\\\")

    lines.extend([
        "  \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_corr_mse_latex(func_type: str, noise_type: str, func_title: str, noise_title: str,
                        measure: str, variance_by_model: dict, entropy_by_model: dict) -> str:
    """One small table: Corr and MSE per model per tau."""
    if measure == "variance":
        data_by_model = variance_by_model
    else:
        data_by_model = entropy_by_model
    corr_col = "Correlation_Epi_Ale"
    mse_col = "MSE"
    if not data_by_model:
        return ""
    df0 = next(iter(data_by_model.values()))
    if corr_col not in df0.columns or mse_col not in df0.columns:
        return ""

    taus = sorted(df0["Tau"].dropna().unique(), key=lambda x: float(x))
    metric_cap = "Variance" if measure == "variance" else "Entropy"
    caption = f"Noise level — Corr(EU,AU) and MSE — {func_title}, {noise_title}, {metric_cap}."
    label = f"tab:noise-{func_type}-{noise_type}-{measure}-corr-mse"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lcccccccc}",
        "  \\toprule",
        "  $\\tau$ & \\multicolumn{2}{c}{Deep Ensemble} & \\multicolumn{2}{c}{MC Dropout} & \\multicolumn{2}{c}{BNN} & \\multicolumn{2}{c}{BAMLSS} \\\\",
        "  & Corr & MSE & Corr & MSE & Corr & MSE & Corr & MSE \\\\",
        "  \\midrule",
    ]
    for tau in taus:
        row = [str(tau)]
        for model in MODEL_ORDER:
            df = data_by_model.get(model)
            if df is None:
                row.extend(["---", "---"])
                continue
            tau_vals = pd.to_numeric(df["Tau"], errors="coerce")
            r = df[np.isclose(tau_vals, float(tau), rtol=1e-9)]
            if r.empty:
                row.extend(["---", "---"])
            else:
                c = _round_val(r[corr_col].iloc[0]) if corr_col in r.columns else "---"
                m = _round_val(r[mse_col].iloc[0], ndec=4) if mse_col in r.columns else "---"
                row.extend([c, m])
        lines.append("  " + " & ".join(row) + " \\\\")
    lines.extend(["  \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main():
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "% Auto-generated by scripts/build_noise_level_latex_tables.py",
        "% Noise-level regression: aggregate stats from results/noise_level/statistics",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]
    for func_type, noise_type, func_title, noise_title in CONDITIONS:
        var_by_model, ent_by_model = load_stats(noise_type, func_type, DISTRIBUTION)
        parts.append(f"% --- {func_title}, {noise_title} ---")
        parts.append("\\subsection{" + func_title + ", " + noise_title + "}")
        for measure in ("variance", "entropy"):
            parts.append(build_table_latex(
                func_type, noise_type, func_title, noise_title, measure,
                var_by_model, ent_by_model
            ))
            corr_mse = build_corr_mse_latex(
                func_type, noise_type, func_title, noise_title, measure,
                var_by_model, ent_by_model
            )
            if corr_mse.strip():
                parts.append(corr_mse)
        parts.append("")

    tex = "\n".join(parts)
    OUT_TEX.write_text(tex, encoding="utf-8")
    print("Wrote:", OUT_TEX)


if __name__ == "__main__":
    main()
