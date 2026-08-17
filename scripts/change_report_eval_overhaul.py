"""
Read-only post-processing: recompute BOTH the OLD (clean-target, reproducing the pre-fix
bug) and NEW (analytic exact-expected) CRPS/NLL directly from existing saved .npz files,
for every experiment, and write a before/after comparison to reports/eval_overhaul.md
(+ a raw CSV of every recomputed cell). Does not retrain anything, does not touch the
run/save pipeline, does not modify any .npz -- mirrors the read-only conventions of
scripts/export_raw_au_eu_csv.py / scripts/export_ovb_raw_au_eu_csv.py.

OLD score (the bug): moment-matched Gaussian (mu_star, sigma2_star) scored via
utils.metrics.compute_gaussian_nll/compute_crps_gaussian against y_grid_clean -- the
clean, noise-free true mean. This is exactly what every utils/*_experiments.py call site
did before the fix.

NEW score: the full predictive mixture (primary) plus the moment-matched Gaussian
(secondary, labeled), scored via utils.analytic_scores.score_bundle as the exact expected
CRPS/NLL against the TRUE distribution N(mu_true, sigma_true^2) -- plus the oracle floor,
IQD, KL, KL_mean, KL_spread.

Each npz's own true sigma is computed the SAME way the fixed call sites now compute it:
utils.mixture_metrics.get_dgp(...).sigma_fn for OOD/Undersampling/Sample-size,
utils.mixture_metrics.sigma_true_noise_level(..., tau) for Noise-level (tau read from the
npz's own metadata), and the omitted-variable-inflated formula for OVB.

Scope note: this aggregates one scalar per npz file (i.e. per model/DGP/knob/seed cell)
over the FULL x_grid ("Combined" region) -- it does not re-derive OOD's ID/OOD split or
Undersampling's per-density-region masks (those live only inside the notebook-driven run
functions, not in the saved npz metadata), so this report is a whole-grid comparison, not
a per-sub-region one. That's still sufficient to show the direction and magnitude of the
bug's effect and to detect any ranking changes.

Usage:
    python scripts/change_report_eval_overhaul.py
    python scripts/change_report_eval_overhaul.py --out reports/eval_overhaul.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.knn_entropy_regression import (  # noqa: E402
    collect_raw_npz_files,
    collect_ovb_npz_files,
    ensure_samples_first,
    _npz_scalar_str,
    _npz_scalar_float,
)
from utils.metrics import (  # noqa: E402
    compute_predictive_aggregation,
    compute_gaussian_nll,
    compute_crps_gaussian,
)
from utils.mixture_metrics import get_dgp, sigma_true_noise_level, normalize_mixture_arrays  # noqa: E402
from utils.analytic_scores import score_bundle  # noqa: E402

RESULTS_ROOT = project_root / "results"
REPORTS_ROOT = project_root / "reports"


# ============================== sigma_true sources (single source of truth, matches
# ============================== the fixed call sites in utils/*_experiments.py) ==========

def _sigma_true_baseline(x, func_type, noise_type, d):
    return get_dgp(func_type, noise_type).sigma_fn(x)


def _sigma_true_noise_sweep(x, func_type, noise_type, d):
    tau = _npz_scalar_float(d, "tau")
    return sigma_true_noise_level(x, noise_type, tau)


EXPERIMENTS = {
    "ood": dict(results_subdir="ood", sigma_true_fn=_sigma_true_baseline, sweep_key=None),
    "undersampling": dict(results_subdir="undersampling", sigma_true_fn=_sigma_true_baseline, sweep_key=None),
    "samplesize": dict(results_subdir="sample_size", sigma_true_fn=_sigma_true_baseline, sweep_key="pct"),
    "noise_level": dict(results_subdir="noise_level", sigma_true_fn=_sigma_true_noise_sweep, sweep_key="tau"),
}


# ============================== per-npz row builders ==============================

def _row_for_npz(npz_path: Path, experiment: str, cfg: dict) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    mu, sig = ensure_samples_first(d["mu_samples"], d["sigma2_samples"], d["x_grid"])
    x = np.asarray(d["x_grid"]).ravel()
    y_grid_clean_flat = np.asarray(d["y_grid_clean"]).ravel()
    func_type = _npz_scalar_str(d, "func_type")
    noise_type = _npz_scalar_str(d, "noise_type")
    model_name = _npz_scalar_str(d, "model_name")
    seed = _npz_scalar_float(d, "seed")

    sigma_true = np.asarray(cfg["sigma_true_fn"](x, func_type, noise_type, d), dtype=float)

    mu_star, sigma2_star = compute_predictive_aggregation(mu, sig)

    # OLD (the bug): moment-matched Gaussian scored against the CLEAN target.
    old_nll = float(compute_gaussian_nll(y_grid_clean_flat, mu_star, sigma2_star))
    old_crps = float(compute_crps_gaussian(y_grid_clean_flat, mu_star, sigma2_star))

    # NEW: exact expected scores against the TRUE distribution.
    mu_n, sig_n = normalize_mixture_arrays(mu, sig, n_expected=len(x))
    scores = score_bundle(mu_n, sig_n, mu_star, sigma2_star, y_grid_clean_flat, sigma_true)

    row = dict(
        experiment=experiment, model_name=model_name, func_type=func_type,
        noise_type=noise_type, seed=seed,
        old_nll=old_nll, old_crps=old_crps,
        new_nll=scores["nll_mixture"], new_crps=scores["crps_mixture"],
        new_nll_gaussian=scores["nll_gaussian"], new_crps_gaussian=scores["crps_gaussian"],
        oracle_nll=scores["oracle_nll"], oracle_crps=scores["oracle_crps"],
        iqd=scores["iqd"], kl=scores["kl"], kl_mean=scores["kl_mean"], kl_spread=scores["kl_spread"],
        source_file=str(npz_path.relative_to(RESULTS_ROOT)),
    )
    if cfg["sweep_key"]:
        row[cfg["sweep_key"]] = _npz_scalar_float(d, cfg["sweep_key"])
    return row


def _rows_for_ovb_npz(npz_path: Path) -> list:
    d = np.load(npz_path, allow_pickle=True)
    func_type = _npz_scalar_str(d, "func_type")
    noise_type = _npz_scalar_str(d, "noise_type")
    model_name = _npz_scalar_str(d, "model_name")
    seed = _npz_scalar_float(d, "seed")
    rho = _npz_scalar_float(d, "rho")
    beta2 = _npz_scalar_float(d, "beta2")

    dgp = get_dgp(func_type, noise_type)
    rows = []

    # Omitted model (X only), evaluated on x_grid, target inflated by the omitted variable.
    x_grid = np.asarray(d["x_grid"]).ravel()
    y_grid_clean_flat = np.asarray(d["y_grid_clean"]).ravel()
    mu, sig = ensure_samples_first(d["mu_samples"], d["sigma2_samples"], x_grid)
    sigma_intrinsic = dgp.sigma_fn(x_grid)
    sigma_true_omitted = np.sqrt(sigma_intrinsic ** 2 + (beta2 ** 2) * (1.0 - rho ** 2))
    mu_star, sigma2_star = compute_predictive_aggregation(mu, sig)
    mu_n, sig_n = normalize_mixture_arrays(mu, sig, n_expected=len(x_grid))
    scores_om = score_bundle(mu_n, sig_n, mu_star, sigma2_star, y_grid_clean_flat, sigma_true_omitted)
    rows.append(dict(
        experiment="ovb", model_name=model_name, func_type=func_type, noise_type=noise_type,
        seed=seed, rho=rho, beta2=beta2, which_model="omitted",
        old_nll=None, old_crps=None,  # OVB never had a prior (buggy) baseline to compare against
        new_nll=scores_om["nll_mixture"], new_crps=scores_om["crps_mixture"],
        new_nll_gaussian=scores_om["nll_gaussian"], new_crps_gaussian=scores_om["crps_gaussian"],
        oracle_nll=scores_om["oracle_nll"], oracle_crps=scores_om["oracle_crps"],
        iqd=scores_om["iqd"], kl=scores_om["kl"], kl_mean=scores_om["kl_mean"], kl_spread=scores_om["kl_spread"],
        source_file=str(npz_path.relative_to(RESULTS_ROOT)),
    ))

    # Full model (X, Z), evaluated at X's own training points -- X_full[:,0] == X.
    if "mu_samples_full" in d.files and "sigma2_samples_full" in d.files and "X_full" in d.files:
        x_full = np.asarray(d["X_full"])
        x_full_flat = x_full[:, 0] if x_full.ndim == 2 else x_full.ravel()
        mu_true_full = dgp.mean_fn(x_full_flat)
        sigma_true_full = dgp.sigma_fn(x_full_flat)
        mu_f, sig_f = ensure_samples_first(d["mu_samples_full"], d["sigma2_samples_full"], x_full_flat)
        mu_star_f, sigma2_star_f = compute_predictive_aggregation(mu_f, sig_f)
        mu_f_n, sig_f_n = normalize_mixture_arrays(mu_f, sig_f, n_expected=len(x_full_flat))
        scores_full = score_bundle(mu_f_n, sig_f_n, mu_star_f, sigma2_star_f, mu_true_full, sigma_true_full)
        rows.append(dict(
            experiment="ovb", model_name=model_name, func_type=func_type, noise_type=noise_type,
            seed=seed, rho=rho, beta2=beta2, which_model="full",
            old_nll=None, old_crps=None,
            new_nll=scores_full["nll_mixture"], new_crps=scores_full["crps_mixture"],
            new_nll_gaussian=scores_full["nll_gaussian"], new_crps_gaussian=scores_full["crps_gaussian"],
            oracle_nll=scores_full["oracle_nll"], oracle_crps=scores_full["oracle_crps"],
            iqd=scores_full["iqd"], kl=scores_full["kl"], kl_mean=scores_full["kl_mean"], kl_spread=scores_full["kl_spread"],
            source_file=str(npz_path.relative_to(RESULTS_ROOT)),
        ))
    return rows


# ============================== collection ==============================

def _collect_all() -> pd.DataFrame:
    rows = []
    for experiment, cfg in EXPERIMENTS.items():
        search_dir = RESULTS_ROOT / cfg["results_subdir"] / "outputs" / cfg["results_subdir"]
        npz_files = collect_raw_npz_files(search_dir)
        print(f"[{experiment}] {len(npz_files)} npz files under {search_dir}")
        for p in npz_files:
            try:
                rows.append(_row_for_npz(p, experiment, cfg))
            except Exception as e:
                print(f"  SKIP {p.name}: {e}")

    ovb_dir = RESULTS_ROOT / "ovb"
    ovb_files = collect_ovb_npz_files(ovb_dir)
    print(f"[ovb] {len(ovb_files)} npz files under {ovb_dir}")
    for p in ovb_files:
        try:
            rows.extend(_rows_for_ovb_npz(p))
        except Exception as e:
            print(f"  SKIP {p.name}: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ============================== report generation ==============================

def _fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _group_cols(experiment: str) -> list:
    cols = ["model_name", "func_type", "noise_type"]
    if experiment == "samplesize":
        cols.append("pct")
    elif experiment == "noise_level":
        cols.append("tau")
    return cols


def _write_report(df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("# CRPS/NLL evaluation overhaul: before/after change report\n")
    lines.append(
        "OLD = moment-matched Gaussian scored against the clean/noise-free `y_grid_clean` "
        "(the pre-fix bug -- rewards overconfidence, breaks the proper-scoring-rule "
        "guarantee). NEW = exact analytic expected CRPS/NLL of the full predictive "
        "mixture against the known true distribution `N(mu_true, sigma_true^2)`, plus the "
        "exact oracle floor. Recomputed directly from existing saved `.npz` outputs "
        "(no retraining). Numbers are averaged over the full x-grid (\"Combined\" region), "
        "not broken out by OOD id/ood or undersampling density sub-regions.\n"
    )

    non_ovb = df[df["experiment"] != "ovb"]
    ranking_flips = []

    for experiment in ["ood", "undersampling", "samplesize", "noise_level"]:
        sub = non_ovb[non_ovb["experiment"] == experiment]
        if sub.empty:
            lines.append(f"\n## {experiment}\n\nNo saved npz outputs found -- nothing to compare.\n")
            continue

        lines.append(f"\n## {experiment}\n")
        gcols = _group_cols(experiment)
        agg = sub.groupby(gcols, dropna=False).agg(
            old_crps=("old_crps", "mean"), new_crps=("new_crps", "mean"),
            old_nll=("old_nll", "mean"), new_nll=("new_nll", "mean"),
            oracle_crps=("oracle_crps", "mean"), oracle_nll=("oracle_nll", "mean"),
            iqd=("iqd", "mean"), kl=("kl", "mean"), n=("old_crps", "size"),
        ).reset_index()

        lines.append("| " + " | ".join(gcols) + " | old_CRPS | new_CRPS | Δ_CRPS (%) | old_NLL | new_NLL | Δ_NLL (%) | oracle_CRPS | oracle_NLL | n |")
        lines.append("|" + "---|" * (len(gcols) + 9))
        for _, r in agg.iterrows():
            d_crps_pct = 100 * (r["new_crps"] - r["old_crps"]) / abs(r["old_crps"]) if r["old_crps"] else float("nan")
            d_nll_pct = 100 * (r["new_nll"] - r["old_nll"]) / abs(r["old_nll"]) if r["old_nll"] else float("nan")
            row_vals = [str(r[c]) for c in gcols]
            row_vals += [_fmt(r["old_crps"]), _fmt(r["new_crps"]), _fmt(d_crps_pct, 1),
                         _fmt(r["old_nll"]), _fmt(r["new_nll"]), _fmt(d_nll_pct, 1),
                         _fmt(r["oracle_crps"]), _fmt(r["oracle_nll"]), str(int(r["n"]))]
            lines.append("| " + " | ".join(row_vals) + " |")

        # Ranking-change detection: within each (func_type, noise_type[, knob]) scenario,
        # does the model ORDER by CRPS (best-to-worst) differ between old and new scoring?
        rank_group_cols = [c for c in gcols if c != "model_name"]
        if rank_group_cols and "model_name" in gcols:
            for key, grp in agg.groupby(rank_group_cols, dropna=False):
                if grp["model_name"].nunique() < 2:
                    continue
                old_order = list(grp.sort_values("old_crps")["model_name"])
                new_order = list(grp.sort_values("new_crps")["model_name"])
                if old_order != new_order:
                    key_str = key if isinstance(key, tuple) else (key,)
                    scenario = dict(zip(rank_group_cols, key_str))
                    ranking_flips.append(dict(experiment=experiment, scenario=scenario,
                                               old_order=old_order, new_order=new_order))

    lines.append("\n## OVB (new capability -- no prior baseline existed)\n")
    ovb = df[df["experiment"] == "ovb"]
    if ovb.empty:
        lines.append("No saved OVB npz outputs found.\n")
    else:
        lines.append(
            "OVB never computed CRPS/NLL before this work -- there is no OLD value to "
            "compare against. Below: the NEW analytic scores, with the KL divergence split "
            "into KL_mean (mean-shift, i.e. the omitted-variable bias signal) and KL_spread "
            "(variance-mismatch). Expectation: KL_spread stays small for the omitted model "
            "(it widens sigma to absorb the bias) while KL_mean carries most of the excess "
            "KL, growing with |beta2| and shrinking as rho -> 1.\n"
        )
        agg = ovb.groupby(["model_name", "func_type", "noise_type", "which_model", "rho", "beta2"], dropna=False).agg(
            new_crps=("new_crps", "mean"), new_nll=("new_nll", "mean"),
            oracle_crps=("oracle_crps", "mean"), oracle_nll=("oracle_nll", "mean"),
            kl_mean=("kl_mean", "mean"), kl_spread=("kl_spread", "mean"), n=("new_crps", "size"),
        ).reset_index()
        lines.append("| model_name | func_type | noise_type | which_model | rho | beta2 | new_CRPS | new_NLL | oracle_CRPS | oracle_NLL | KL_mean | KL_spread | n |")
        lines.append("|" + "---|" * 13)
        for _, r in agg.iterrows():
            lines.append("| " + " | ".join([
                str(r["model_name"]), str(r["func_type"]), str(r["noise_type"]), str(r["which_model"]),
                _fmt(r["rho"], 2), _fmt(r["beta2"], 2), _fmt(r["new_crps"]), _fmt(r["new_nll"]),
                _fmt(r["oracle_crps"]), _fmt(r["oracle_nll"]), _fmt(r["kl_mean"]), _fmt(r["kl_spread"]),
                str(int(r["n"])),
            ]) + " |")

    lines.append("\n## Ranking changes\n")
    if not ranking_flips:
        lines.append("No CRPS-based method ranking changes detected in any (experiment, scenario) group.\n")
    else:
        for flip in ranking_flips:
            lines.append(
                f"- **{flip['experiment']}** {flip['scenario']}: old ranking "
                f"{' < '.join(flip['old_order'])} -> new ranking {' < '.join(flip['new_order'])}"
            )

    lines.append("\n## Summary\n")
    if non_ovb.empty:
        lines.append("No non-OVB npz outputs were found on disk, so no before/after comparison could be computed.\n")
    else:
        mean_old_crps = non_ovb["old_crps"].mean()
        mean_new_crps = non_ovb["new_crps"].mean()
        mean_old_nll = non_ovb["old_nll"].mean()
        mean_new_nll = non_ovb["new_nll"].mean()
        lines.append(
            f"Across {len(non_ovb)} recomputed (model, DGP, knob, seed) cells, the OLD "
            f"clean-target CRPS averaged {mean_old_crps:.4f} vs. the NEW analytic CRPS at "
            f"{mean_new_crps:.4f} ({100*(mean_new_crps-mean_old_crps)/abs(mean_old_crps):+.1f}%); "
            f"OLD NLL averaged {mean_old_nll:.4f} vs. NEW NLL at {mean_new_nll:.4f} "
            f"({100*(mean_new_nll-mean_old_nll)/abs(mean_old_nll):+.1f}%). "
            + ("No method ranking changed under the fix, so relative conclusions about "
               "which method performs best/worst on predictive accuracy are unaffected -- "
               "only the absolute NLL/CRPS numbers were wrong."
               if not ranking_flips else
               f"{len(ranking_flips)} (experiment, scenario) group(s) had their CRPS-based "
               "method ranking change under the fix (see above) -- any prior conclusion "
               "about relative method ordering in those specific scenarios should be treated "
               "as an artifact of the clean-target bug, not a real result.")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPORTS_ROOT / "eval_overhaul.md")
    parser.add_argument("--csv-out", type=Path, default=RESULTS_ROOT / "eval_overhaul" / "csv" / "eval_overhaul_raw.csv")
    args = parser.parse_args()

    df = _collect_all()
    if df.empty:
        print("No npz files found anywhere -- nothing to report.")
        return

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f"Wrote {args.csv_out} ({len(df)} rows)")

    _write_report(df, args.out)


if __name__ == "__main__":
    main()
