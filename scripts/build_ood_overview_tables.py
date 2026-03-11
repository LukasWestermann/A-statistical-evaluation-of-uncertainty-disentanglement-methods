"""
Build OOD overview tables from ID/OOD summary CSVs and write LaTeX to
results/ood/appendix_ood_tables_generated.tex (for \\input from appendix_ood_figures_and_tables.tex).
Optionally save aggregated CSVs under results/ood/statistics/.
"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
stats_base = project_root / "results" / "ood" / "statistics" / "ood"
out_tex = project_root / "results" / "ood" / "appendix_ood_tables_generated.tex"

CONDITIONS = [
    ("linear", "homoscedastic", "Linear", "homoscedastic"),
    ("linear", "heteroscedastic", "Linear", "heteroscedastic"),
    ("sin", "homoscedastic", "Sinusoidal", "homoscedastic"),
    ("sin", "heteroscedastic", "Sinusoidal", "heteroscedastic"),
]

MODEL_ORDER = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]


def _model_from_filename(name):
    """Extract display model name from filename stem (no extension). E.g. 20260107_Deep_Ensemble_K20_ID_... -> Deep Ensemble."""
    if "Deep_Ensemble" in name:
        return "Deep Ensemble"
    if "MC_Dropout" in name:
        return "MC Dropout"
    if "BNN" in name and "BAMLSS" not in name:
        return "BNN"
    if "BAMLSS" in name:
        return "BAMLSS"
    return None


def _load_region_df(cond_dir, region, metric):
    """Load one CSV for region (ID or OOD) and metric (variance or entropy). Returns (model_name -> row dict)."""
    # pattern: *_{region}_uncertainties_summary_{Function}_{noise}_{metric}.csv
    # cond_dir = stats_base / noise_type / func_type  -> .name = func_type, .parent.name = noise_type
    func_title = "Linear" if cond_dir.name == "linear" else "Sinusoidal"
    noise = cond_dir.parent.name  # homoscedastic or heteroscedastic
    suffix = f"{region}_uncertainties_summary_{func_title}_{noise}_{metric}.csv"
    files = list(cond_dir.glob(f"*_{suffix}"))
    if not files:
        return {}
    # Use latest by date prefix (YYYYMMDD)
    files_sorted = sorted(files, key=lambda p: p.name[:8], reverse=True)
    result = {}
    for f in files_sorted:
        model = _model_from_filename(f.stem)
        if model is None or model in result:
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            row = df.iloc[0]
            result[model] = row
        except Exception:
            continue
    return result


def _round_val(x, ndec=4):
    if pd.isna(x):
        return "---"
    try:
        return round(float(x), ndec)
    except (TypeError, ValueError):
        return str(x)


def _build_overview_table(func_type, noise_type, func_title, noise_title, metric, save_csv=False):
    """Build one overview table (variance or entropy) for one condition. Returns LaTeX string."""
    cond_dir = stats_base / noise_type / func_type
    if not cond_dir.exists():
        return f"% No data for {func_type} {noise_type} {metric}\n"

    id_data = _load_region_df(cond_dir, "ID", metric)
    ood_data = _load_region_df(cond_dir, "OOD", metric)

    if metric == "variance":
        ale_col, epi_col = "Avg_Aleatoric_Variance", "Avg_Epistemic_Variance"
    else:
        ale_col, epi_col = "Avg_Aleatoric_Entropy_norm", "Avg_Epistemic_Entropy_norm"
    corr_col, mse_col = "Correlation_Epi_Ale", "MSE"

    rows = []
    for model in MODEL_ORDER:
        id_row = id_data.get(model)
        ood_row = ood_data.get(model)
        id_ale = _round_val(id_row[ale_col]) if id_row is not None and ale_col in id_row else "---"
        id_epi = _round_val(id_row[epi_col]) if id_row is not None and epi_col in id_row else "---"
        id_corr = _round_val(id_row[corr_col]) if id_row is not None and corr_col in id_row else "---"
        id_mse = _round_val(id_row[mse_col]) if id_row is not None and mse_col in id_row else "---"
        ood_ale = _round_val(ood_row[ale_col]) if ood_row is not None and ale_col in ood_row else "---"
        ood_epi = _round_val(ood_row[epi_col]) if ood_row is not None and epi_col in ood_row else "---"
        ood_corr = _round_val(ood_row[corr_col]) if ood_row is not None and corr_col in ood_row else "---"
        ood_mse = _round_val(ood_row[mse_col]) if ood_row is not None and mse_col in ood_row else "---"
        rows.append((model, id_ale, id_epi, id_corr, id_mse, ood_ale, ood_epi, ood_corr, ood_mse))

    if save_csv:
        csv_dir = project_root / "results" / "ood" / "statistics"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"overview_{metric}_{func_type}_{noise_type}.csv"
        df = pd.DataFrame(rows, columns=["Model", "ID_Avg_Ale", "ID_Avg_Epi", "ID_Corr", "ID_MSE", "OOD_Avg_Ale", "OOD_Avg_Epi", "OOD_Corr", "OOD_MSE"])
        df.to_csv(csv_path, index=False)
        print("Wrote CSV:", csv_path)

    metric_cap = "Variance" if metric == "variance" else "Entropy"
    caption = f"OOD summary statistics ({metric_cap}): {func_title}, {noise_title}. ID vs OOD regions."
    label = f"tab:ood-{func_type}-{noise_type}-{metric}"

    lines = [
        f"\\begin{{table}}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lcccccccc}",
        "  \\toprule",
        "  Model & \\multicolumn{4}{c}{ID} & \\multicolumn{4}{c}{OOD} \\\\",
        "  & Avg\\_Ale & Avg\\_Epi & Corr & MSE & Avg\\_Ale & Avg\\_Epi & Corr & MSE \\\\",
        "  \\midrule",
    ]
    for model, id_ale, id_epi, id_corr, id_mse, ood_ale, ood_epi, ood_corr, ood_mse in rows:
        lines.append(f"  {model} & {id_ale} & {id_epi} & {id_corr} & {id_mse} & {ood_ale} & {ood_epi} & {ood_corr} & {ood_mse} \\\\")
    lines.extend([
        "  \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def main():
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "% Auto-generated by scripts/build_ood_overview_tables.py",
        "% Summary statistics (variance and entropy) split by ID and OOD, including correlations.",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]
    for func_type, noise_type, func_title, noise_title in CONDITIONS:
        parts.append(f"% --- {func_title}, {noise_title} ---")
        parts.append("\\subsection{" + func_title + ", " + noise_title + "}")
        parts.append(_build_overview_table(func_type, noise_type, func_title, noise_title, "variance", save_csv=True))
        parts.append(_build_overview_table(func_type, noise_type, func_title, noise_title, "entropy", save_csv=True))
        parts.append("")

    out_tex.write_text("\n".join(parts), encoding="utf-8")
    print("Wrote:", out_tex)


if __name__ == "__main__":
    main()
