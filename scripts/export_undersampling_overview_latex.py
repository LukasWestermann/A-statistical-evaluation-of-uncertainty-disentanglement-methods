"""
Export undersampling overview tables: variance (Undersampled vs Well_sampled) +
moment-matched analytical entropy on the same spatial split.

**Variance** numbers are read from ``results/undersampling/statistics/{noise}/{func}/``
Excel files produced by ``save_summary_statistics_undersampling`` for regions
``Undersampled`` and ``Well_sampled`` (same min--max normalization as the experiment).

**Entropy** uses ``entropy_uncertainty_analytical_moment_matched`` on the latest
``raw_outputs`` .npz per model, then ``moment_matched_entropy_density_split_metrics``
with the same ``sampling_regions`` definition as the notebook (configurable).

Writes (by default under ``results/undersampling/tables/``):
  - Markdown / space-separated lines (easy to paste into an LLM)
  - LaTeX ``booktabs`` fragment

Usage::

    python scripts/export_undersampling_overview_latex.py
    python scripts/export_undersampling_overview_latex.py --stdout-only
    python scripts/export_undersampling_overview_latex.py --out-md path/to/overview.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (
    MODEL_RESOLVERS,
    compute_moment_matched_grid_result,
    moment_matched_entropy_density_split_metrics,
    resolve_latest_npz,
)
from utils.regression_summary_panels import FUNC_NAME_BY_TYPE, MODEL_KEYS

# Default matches ``Experiments/Undersampling.ipynb`` (non-uniform middle band).
DEFAULT_SAMPLING_REGIONS: List[Tuple[Tuple[float, float], float]] = [
    ((-5, 4), 1.0),
    ((4, 8), 0.05),
    ((8, 10), 1.0),
]

CONDITIONS: List[Tuple[str, str]] = [
    ("linear", "homoscedastic"),
    ("linear", "heteroscedastic"),
    ("sin", "homoscedastic"),
    ("sin", "heteroscedastic"),
]

MODEL_ORDER = list(MODEL_KEYS.keys())


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fmt_plain(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _fmt_tex(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "---"
    return f"{x:.{nd}f}"


def _pick_latest_variance_row(
    stats_dir: Path,
    model_substr: str,
    region_token: str,
    func_human: str,
    noise_type: str,
) -> Optional[pd.Series]:
    """Latest Excel row for ``Undersampled`` / ``Well_sampled`` variance summary."""
    if not stats_dir.is_dir():
        return None
    pat = f"*{region_token}*uncertainties_summary*{func_human}*{noise_type}*variance.xlsx"
    cands = [p for p in stats_dir.glob(pat) if model_substr in p.name]
    if not cands:
        return None
    p = sorted(cands)[-1]
    try:
        df = pd.read_excel(p, engine="openpyxl")
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[0]


def load_variance_split(
    stats_dir: Path,
    model_display: str,
    func_type: str,
    noise_type: str,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    func_human = FUNC_NAME_BY_TYPE[func_type]  # type: ignore[index]
    key = MODEL_KEYS[model_display]
    ru = _pick_latest_variance_row(stats_dir, key, "Undersampled", func_human, noise_type)
    rw = _pick_latest_variance_row(stats_dir, key, "Well_sampled", func_human, noise_type)

    def pack(row: Optional[pd.Series]) -> Optional[Dict[str, float]]:
        if row is None:
            return None
        return {
            "AU": _f(row.get("Avg_Aleatoric_norm")),
            "EU": _f(row.get("Avg_Epistemic_norm")),
            "rho": _f(row.get("Correlation_Epi_Ale")),
            "MSE": _f(row.get("MSE")),
        }

    return pack(ru), pack(rw)


def load_entropy_split_from_npz(
    npz_path: Path,
    sampling_regions: Sequence[Tuple[Tuple[float, float], float]],
    grid_stride: int,
    eps: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    res = compute_moment_matched_grid_result(npz_path, [], grid_stride, eps)
    u, w = moment_matched_entropy_density_split_metrics(res, sampling_regions)
    return u, w


def build_scenario_rows(
    stats_root: Path,
    npz_root: Path,
    sampling_regions: Sequence[Tuple[Tuple[float, float], float]],
    grid_stride: int,
    eps: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for func_type, noise_type in CONDITIONS:
        # Notebook sets stats_dir = results/undersampling/statistics; save_statistics uses
        # subfolder ``undersampling/{noise}/{func}`` under that.
        sdir = stats_root / "undersampling" / noise_type / func_type
        nzdir = npz_root / noise_type / func_type
        scen = f"{FUNC_NAME_BY_TYPE[func_type]}, {noise_type}"  # type: ignore[index]
        for model_display in MODEL_ORDER:
            vu, vw = load_variance_split(sdir, model_display, func_type, noise_type)
            _disp, tag, globs = next(
                (m for m in MODEL_RESOLVERS if m[0] == model_display),
                (None, None, ()),
            )
            npz_path = resolve_latest_npz(nzdir, globs) if tag else None
            eu = ew = None
            if npz_path is not None:
                try:
                    raw_u, raw_w = load_entropy_split_from_npz(
                        npz_path, sampling_regions, grid_stride, eps
                    )
                    eu = {
                        "AU": raw_u.get("Avg_Aleatoric_Entropy_norm"),
                        "EU": raw_u.get("Avg_Epistemic_Entropy_norm"),
                        "rho": raw_u.get("Correlation_Epi_Ale"),
                        "MSE": raw_u.get("MSE"),
                    }
                    ew = {
                        "AU": raw_w.get("Avg_Aleatoric_Entropy_norm"),
                        "EU": raw_w.get("Avg_Epistemic_Entropy_norm"),
                        "rho": raw_w.get("Correlation_Epi_Ale"),
                        "MSE": raw_w.get("MSE"),
                    }
                except Exception:
                    eu = ew = None

            rows.append(
                {
                    "scenario": scen,
                    "function": func_type,
                    "noise": noise_type,
                    "model": model_display,
                    "var_undersampled": vu,
                    "var_well": vw,
                    "ent_undersampled": eu,
                    "ent_well": ew,
                    "npz": str(npz_path) if npz_path else "",
                }
            )
    return rows


def plain_header_line() -> str:
    return (
        "Model "
        "AU_us AU_ns EU_us EU_ns rho_us rho_ns "
        "AU_us AU_ns EU_us EU_ns rho_us rho_ns"
    )


def row_plain_line(model: str, vu, vw, eu, ew) -> str:
    def six_pack(a, b):
        if a is None:
            a = {}
        if b is None:
            b = {}
        return [
            _fmt_plain(a.get("AU")),
            _fmt_plain(b.get("AU")),
            _fmt_plain(a.get("EU")),
            _fmt_plain(b.get("EU")),
            _fmt_plain(a.get("rho")),
            _fmt_plain(b.get("rho")),
        ]

    v = six_pack(vu, vw)
    e = six_pack(eu, ew)
    return model + " " + " ".join(v + e)


def latex_table_for_scenario(scenario: str, block_rows: List[Dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{Undersampling overview --- {scenario} (variance vs moment-matched entropy).}}",
        r"\begin{tabular}{l*{12}{r}}",
        r"\toprule",
        r" & \multicolumn{6}{c}{Variance} & \multicolumn{6}{c}{Entropy (moment-matched)} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        r"Model & $\overline{\mathrm{AU}}_{\mathrm{us}}$ & $\overline{\mathrm{AU}}_{\mathrm{ns}}$"
        r" & $\overline{\mathrm{EU}}_{\mathrm{us}}$ & $\overline{\mathrm{EU}}_{\mathrm{ns}}$"
        r" & $\rho_{\mathrm{us}}$ & $\rho_{\mathrm{ns}}$"
        r" & $\overline{\mathrm{AU}}_{\mathrm{us}}$ & $\overline{\mathrm{AU}}_{\mathrm{ns}}$"
        r" & $\overline{\mathrm{EU}}_{\mathrm{us}}$ & $\overline{\mathrm{EU}}_{\mathrm{ns}}$"
        r" & $\rho_{\mathrm{us}}$ & $\rho_{\mathrm{ns}}$ \\",
        r"\midrule",
    ]
    for r in block_rows:
        vu, vw = r["var_undersampled"], r["var_well"]
        eu, ew = r["ent_undersampled"], r["ent_well"]

        def six_tex(a, b):
            if a is None:
                a = {}
            if b is None:
                b = {}
            return [
                _fmt_tex(a.get("AU")),
                _fmt_tex(b.get("AU")),
                _fmt_tex(a.get("EU")),
                _fmt_tex(b.get("EU")),
                _fmt_tex(a.get("rho")),
                _fmt_tex(b.get("rho")),
            ]

        v = six_tex(vu, vw)
        e = six_tex(eu, ew)
        name = r["model"].replace("_", r"\_")
        lines.append(
            name + " & " + " & ".join(v + e) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    _tables = project_root / "results" / "undersampling" / "tables"
    p = argparse.ArgumentParser(description="Undersampling overview: variance + moment-matched entropy (LaTeX / MD).")
    p.add_argument(
        "--stats-root",
        type=Path,
        default=project_root / "results" / "undersampling" / "statistics",
        help="Notebook ``statistics`` folder (contains ``undersampling/{noise}/{func}/`` Excel files)",
    )
    p.add_argument(
        "--npz-root",
        type=Path,
        default=project_root / "results" / "undersampling" / "outputs" / "undersampling",
        help="Root with raw_outputs .npz per condition",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=_tables / "overview_undersampling.md",
        help="Write Markdown + plain lines to this path",
    )
    p.add_argument(
        "--out-tex",
        type=Path,
        default=_tables / "overview_undersampling.tex",
        help="Write concatenated LaTeX tables to this path",
    )
    p.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print markdown to stdout; do not write files",
    )
    p.add_argument("--grid-stride", type=int, default=1)
    p.add_argument("--eps", type=float, default=1e-12)
    args = p.parse_args()

    rows = build_scenario_rows(
        Path(args.stats_root).resolve(),
        Path(args.npz_root).resolve(),
        DEFAULT_SAMPLING_REGIONS,
        args.grid_stride,
        args.eps,
    )

    by_scen: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_scen.setdefault(r["scenario"], []).append(r)

    md_chunks: List[str] = [
        "# Undersampling overview tables",
        "",
        "Columns: **us** = spatially undersampled band(s) (density factor < 0.5); "
        "**ns** = normal / well-sampled band(s) (density ≥ 0.5). "
        "First six numeric columns are variance-based; last six are moment-matched entropy.",
        "",
        f"Default sampling regions: `{DEFAULT_SAMPLING_REGIONS}`.",
        "",
    ]

    tex_chunks: List[str] = []

    for scen in sorted(by_scen.keys()):
        block = by_scen[scen]
        md_chunks.append(f"## {scen}")
        md_chunks.append("")
        md_chunks.append("```text")
        md_chunks.append(plain_header_line())
        for r in block:
            md_chunks.append(
                row_plain_line(
                    r["model"],
                    r["var_undersampled"],
                    r["var_well"],
                    r["ent_undersampled"],
                    r["ent_well"],
                )
            )
        md_chunks.append("```")
        md_chunks.append("")
        tex_chunks.append(latex_table_for_scenario(scen, block))
        tex_chunks.append("\n\n")

    text = "\n".join(md_chunks)
    tex_body = "\n".join(tex_chunks).strip() + "\n"

    if args.stdout_only:
        print(text)
        return

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    print("Wrote", out_md)

    out_tex = Path(args.out_tex).resolve()
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    tex_preamble = "% Requires \\usepackage{booktabs} in the preamble.\n\n"
    out_tex.write_text(tex_preamble + tex_body, encoding="utf-8")
    print("Wrote", out_tex)


if __name__ == "__main__":
    main()
