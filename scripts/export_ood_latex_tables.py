"""Print LaTeX tables for OOD experiments: moment-matched entropy + variance from raw_outputs npz.

Uses the latest ``*raw_outputs*.npz`` per model under ``results/ood/outputs/ood/{noise}/{func}/``,
``DEFAULT_OOD_RANGES`` from ``utils.knn_entropy_regression``, and the same regional normalization
as ``normalized_entropy_stats_ood_regions`` (entropy) and the variance analogue (mean $\sigma^2$,
$\mathrm{Var}(\mu)$).

Usage::

    python scripts/export_ood_latex_tables.py
    python scripts/export_ood_latex_tables.py > results/ood/tables/ood_summary.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (  # noqa: E402
    CONDITIONS_4,
    DEFAULT_OOD_RANGES,
    MODEL_RESOLVERS,
    compute_moment_matched_grid_result,
    normalized_entropy_stats_ood_regions,
    resolve_latest_npz,
)

MODEL_ORDER = [m[0] for m in MODEL_RESOLVERS]

# Keys match ``CONDITIONS_4``: (func_type, noise_type)
COND_TITLE = {
    ("linear", "homoscedastic"): "Linear, homoscedastic",
    ("linear", "heteroscedastic"): "Linear, heteroscedastic",
    ("sin", "homoscedastic"): "Sinusoidal, homoscedastic",
    ("sin", "heteroscedastic"): "Sinusoidal, heteroscedastic",
}


def var_stats_region(ale, epi, id_mask, ood_mask):
    def normalize(values, vmin, vmax):
        if vmax - vmin == 0:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    ale_min, ale_max = float(ale.min()), float(ale.max())
    epi_min, epi_max = float(epi.min()), float(epi.max())
    out = {}
    for name, mask in [("ID", id_mask), ("OOD", ood_mask)]:
        av, ev = ale[mask], epi[mask]
        an = normalize(av, ale_min, ale_max)
        en = normalize(ev, epi_min, epi_max)
        corr = np.corrcoef(ev, av)[0, 1] if av.size > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0
        out[name] = (
            float(np.mean(an)),
            float(np.mean(en)),
            float(corr),
        )
    return out


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def collect():
    base = project_root / "results" / "ood" / "outputs" / "ood"
    if not base.is_dir():
        raise FileNotFoundError(f"Missing OOD npz root: {base}")
    out = {}
    for func_type, noise_type in CONDITIONS_4:
        key = (func_type, noise_type)
        out[key] = {}
        d = base / noise_type / func_type
        for disp, _tag, globs in MODEL_RESOLVERS:
            p = resolve_latest_npz(d, globs)
            if p is None:
                out[key][disp] = None
                continue
            res = compute_moment_matched_grid_result(
                p, DEFAULT_OOD_RANGES, grid_stride=1, eps=1e-10
            )
            ent = normalized_entropy_stats_ood_regions(
                res.ale_entropy,
                res.epi_entropy,
                res.tot_entropy,
                res.mu_pred,
                res.y_clean_flat,
                res.id_mask,
                res.ood_mask,
            )
            vs = var_stats_region(
                res.ale_var, res.epi_var, res.id_mask, res.ood_mask
            )
            out[key][disp] = {"ent": ent, "var": vs, "npz": p.name}
    n_ok = sum(1 for sub in out.values() for c in sub.values() if c is not None)
    if n_ok == 0:
        raise RuntimeError(
            f"No OOD npz resolved under {base}. "
            f"project_root={project_root!s} exists={base.is_dir()}"
        )
    return out


def latex_block(func_type: str, noise_type: str, data: dict) -> str:
    title = COND_TITLE[(func_type, noise_type)]
    lines = [
        rf"\noindent\textbf{{{title}}}\par\medskip",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{lcccccccccccc}",
        r"\toprule",
        r" & \multicolumn{6}{c}{Entropy-based (moment-matched)} "
        r"& \multicolumn{6}{c}{Variance-based} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        r"Model & AU$_I$ & AU$_O$ & EU$_I$ & EU$_O$ & Corr$_I$ & Corr$_O$ "
        r"& AU$_I$ & AU$_O$ & EU$_I$ & EU$_O$ & Corr$_I$ & Corr$_O$ \\",
        r"\midrule",
    ]
    for disp in MODEL_ORDER:
        cell = data.get(disp)
        if cell is None:
            row = disp + " & " + " & ".join(["---"] * 12) + r" \\"
            lines.append(row)
            continue
        ent, var_ = cell["ent"], cell["var"]
        ei, eo = ent["ID"], ent["OOD"]
        vi, vo = var_["ID"], var_["OOD"]
        vals = [
            fmt3(ei["Avg_Aleatoric_Entropy_norm"]),
            fmt3(eo["Avg_Aleatoric_Entropy_norm"]),
            fmt3(ei["Avg_Epistemic_Entropy_norm"]),
            fmt3(eo["Avg_Epistemic_Entropy_norm"]),
            fmt3(ei["Correlation_Epi_Ale"]),
            fmt3(eo["Correlation_Epi_Ale"]),
            fmt3(vi[0]),
            fmt3(vo[0]),
            fmt3(vi[1]),
            fmt3(vo[1]),
            fmt3(vi[2]),
            fmt3(vo[2]),
        ]
        lines.append(disp + " & " + " & ".join(vals) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\caption{OOD experiment: normalized mean aleatoric/epistemic uncertainty and "
            r"Pearson correlation (epistemic vs.\ aleatoric) in ID and OOD regions. "
            r"Entropy uses moment-matched Gaussian entropy with AU/EU min--max pooled over the full grid; "
            r"variance uses the same pooling on mean $\sigma^2$ and $\mathrm{Var}(\mu)$.}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rows = collect()
    for func_type, noise_type in CONDITIONS_4:
        print(latex_block(func_type, noise_type, rows[(func_type, noise_type)]))


if __name__ == "__main__":
    main()
