"""
Score the overall predictive distribution -- RMSE, closed-form mixture CRPS, exact
mixture NLL, PIT, marginal + stratified coverage, sharpness, and an oracle floor from
the true DGP -- for MC Dropout, Deep Ensemble, and BAMLSS on the BASELINE scenario
(100% of training data, standard noise settings) across the 4 canonical toy-regression
DGPs. Also runs, for each method, a component-count sensitivity sweep (Deep Ensemble's
K, MC Dropout's M, BAMLSS's nsamples) by subsampling the already-trained/predicted
members -- no retraining -- plus an MC Dropout dropout-probability (p) sweep, which
DOES retrain once per (p, replicate) since p is a training hyperparameter.

Trains every method FRESH -- does not load the existing results/*/outputs/*.npz files.
Does, however, WRITE its own raw_outputs.npz per (method, DGP, seed) baseline cell, via
utils.results_save.save_model_outputs -- same convention as
ood/sample_size/undersampling/noise_level_experiments.py -- so metrics can be recomputed
later without retraining. Not written for the component-count/dropout-p sweep replicates
(those are cheap to regenerate from the baseline npz or a retrain, and saving one per
sweep replicate would be a lot of near-duplicate files).
Math lives in utils/mixture_metrics.py; this script just wires up training + I/O.

Usage:
    python scripts/eval_predictive_baseline.py                                   # quick smoke default
    python scripts/eval_predictive_baseline.py MC_Dropout:linear:homoscedastic   # single scenario
    python scripts/eval_predictive_baseline.py MC_Dropout:linear:homoscedastic --smoke  # + fast hyperparams
    python scripts/eval_predictive_baseline.py --batch                          # full 3-method x 4-DGP sweep
    python scripts/eval_predictive_baseline.py --batch --methods Deep_Ensemble,MC_Dropout
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Models.MC_Dropout import MCDropoutRegressor, train_model, mc_dropout_predict
from Models.Deep_Ensemble import train_ensemble_deep, ensemble_predict_deep
from Models.BAMLSS import bamlss_predict
from utils import mixture_metrics as mm
from utils import predictive_eval_io as pio
from utils import predictive_eval_plotting as pep
import utils.results_save as results_save_module

ALL_METHODS = ["MC_Dropout", "Deep_Ensemble", "BAMLSS"]
ALL_DGPS = [("linear", "homoscedastic"), ("linear", "heteroscedastic"),
            ("sin", "homoscedastic"), ("sin", "heteroscedastic")]


# ============================== Train-and-predict, one per method ==============================
# Each returns raw (mu_samples, sigma2_samples) at x_test, shape [M, N_test]. Reseeds the
# global np/torch RNG right before training -- correctness-critical for reproducibility
# regardless of run order, since dropout masks, DataLoader shuffling, and Deep Ensemble's
# own per-member seeding all touch global RNG state.

def _train_predict_mc_dropout(x_train, y_train, x_test, seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.mc_dropout_batch_size, shuffle=True)
    model = MCDropoutRegressor(p=args.mc_dropout_p)
    train_model(model, loader, epochs=args.mc_dropout_epochs, lr=args.mc_dropout_lr,
                loss_type='beta_nll', beta=args.mc_dropout_beta)
    _, _, _, _, (mu_samples, sigma2_samples) = mc_dropout_predict(
        model, x_test, M=args.mc_dropout_m, return_raw_arrays=True)
    return mu_samples, sigma2_samples


def _train_predict_deep_ensemble(x_train, y_train, x_test, seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    ensemble = train_ensemble_deep(x_train, y_train, batch_size=args.ensemble_batch_size,
                                    K=args.ensemble_k, loss_type='beta_nll',
                                    beta=args.ensemble_beta, epochs=args.ensemble_epochs)
    _, _, _, _, (mu_samples, sigma2_samples) = ensemble_predict_deep(
        ensemble, x_test, return_raw_arrays=True)
    return mu_samples, sigma2_samples


def _train_predict_bamlss(x_train, y_train, x_test, seed, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, _, _, _, (mu_samples, sigma2_samples) = bamlss_predict(
        x_train, y_train, x_test, n_iter=args.bamlss_n_iter, burnin=args.bamlss_burnin,
        thin=args.bamlss_thin, nsamples=args.bamlss_nsamples, return_raw_arrays=True)
    return mu_samples, sigma2_samples


TRAIN_PREDICT_FNS = {
    "MC_Dropout": _train_predict_mc_dropout,
    "Deep_Ensemble": _train_predict_deep_ensemble,
    "BAMLSS": _train_predict_bamlss,
}


# ============================== Tidy-row helpers ==============================

def _scalar_rows(scalar_dict, **tags):
    return [dict(metric=k, value=v, **tags) for k, v in scalar_dict.items()]


def _pit_rows(x_test, y_test, pit, **tags):
    return [dict(x_test=float(x), y_test=float(y), pit_value=float(p), **tags)
            for x, y, p in zip(np.asarray(x_test).ravel(), np.asarray(y_test).ravel(), pit)]


def _stratified_rows(strat_list, **tags):
    return [dict(**row, **tags) for row in strat_list]


# ============================== Component-count sweeps (no retraining) ==============================
# Deep Ensemble's K members, MC Dropout's M forward passes, and BAMLSS's nsamples
# posterior draws are all, at heart, "how many mixture components did we use" -- so one
# subsample_members-based sweep covers all three: fit/train once at the max size (already
# done for the baseline row), then evaluate nested random subsets of the SAME already-
# computed raw predictions. No retraining for any of the three.

# Per-method grid of component counts to sweep, keyed by the args attribute holding it.
SIZE_SWEEP_GRID_ARG = {
    "Deep_Ensemble": "ensemble_size_grid",
    "MC_Dropout": "mc_dropout_m_grid",
    "BAMLSS": "bamlss_nsamples_grid",
}


def _component_size_sweep(method, mu_all, sigma2_all, grid, R, seed_base, dgp, x_test, y_test,
                           mu_true, sigma_true, args, extra_tags=None):
    extra_tags = extra_tags or {}
    K = mu_all.shape[0]
    eff_grid = sorted({m for m in grid if m <= K})
    print(f"-- {method} component-size sweep (pool={K}, grid={eff_grid}, R={R}) --")
    rows = []
    for M in eff_grid:
        sweep_seed = seed_base + M
        for r, idx, mu_sub, sigma2_sub in mm.subsample_members(mu_all, sigma2_all, M, R, seed=sweep_seed):
            result = mm.evaluate_mixture(x_test, y_test, mu_sub, sigma2_sub, mu_true, sigma_true,
                                          coverage_levels=args.coverage_levels, n_bins=args.n_bins,
                                          include_stratified=False)
            tags = dict(method=method, dgp=dgp, scenario='ensemble_size_sweep',
                        ensemble_size=M, replicate=r, seed=sweep_seed, **extra_tags)
            rows += _scalar_rows(result['scalar'], **tags)
    return rows


# ============================== MC Dropout dropout-probability sweep (retrains) ==============
# Unlike M, p is baked into the trained weights -- there is no way to get different-p
# predictions out of one trained model, so this genuinely retrains once per (p, replicate).

def run_mc_dropout_p_sweep(x_train, y_train, x_test, y_test, mu_true, sigma_true, dgp, args):
    print(f"-- MC_Dropout dropout-p sweep (grid={args.mc_dropout_p_grid}, R={args.mc_dropout_p_sweep_r}) --")
    rows = []
    for p in args.mc_dropout_p_grid:
        for rep in range(args.mc_dropout_p_sweep_r):
            seed = args.mc_dropout_p_sweep_seed + rep
            np.random.seed(seed)
            torch.manual_seed(seed)
            ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            loader = DataLoader(ds, batch_size=args.mc_dropout_batch_size, shuffle=True)
            model = MCDropoutRegressor(p=p)
            train_model(model, loader, epochs=args.mc_dropout_epochs, lr=args.mc_dropout_lr,
                        loss_type='beta_nll', beta=args.mc_dropout_beta)
            _, _, _, _, (mu_s, sigma2_s) = mc_dropout_predict(
                model, x_test, M=args.mc_dropout_m, return_raw_arrays=True)
            mu_s, sigma2_s = mm.normalize_mixture_arrays(mu_s, sigma2_s, n_expected=len(x_test))

            result = mm.evaluate_mixture(x_test, y_test, mu_s, sigma2_s, mu_true, sigma_true,
                                          coverage_levels=args.coverage_levels, n_bins=args.n_bins,
                                          include_stratified=False)
            tags = dict(method='MC_Dropout', dgp=dgp, scenario='mc_dropout_p_sweep',
                        ensemble_size=mu_s.shape[0], replicate=rep, seed=seed, dropout_p=p)
            rows += _scalar_rows(result['scalar'], **tags)
    return rows


# ============================== Per-DGP evaluation ==============================

RAW_OUTPUT_EXTRA_KWARGS = {
    "MC_Dropout": lambda args: {"dropout_p": args.mc_dropout_p, "mc_samples": args.mc_dropout_m},
    "Deep_Ensemble": lambda args: {"n_nets": args.ensemble_k},
    "BAMLSS": lambda args: {},
}


def run_dgp(func_type, noise_type, methods, args, date, include_p_sweep=True):
    dgp = mm.dgp_name(func_type, noise_type)
    print(f"\n{'=' * 70}\nDGP: {dgp}\n{'=' * 70}")

    scalar_rows, pit_rows, strat_rows = [], [], []

    x_train, y_train, _ = mm.generate_training_data(
        func_type, noise_type, n_train=args.n_train, train_range=tuple(args.train_range), seed=args.seed)
    x_test, y_test, mu_true, sigma_true = mm.make_test_set(
        func_type, noise_type, x_range=tuple(args.train_range), n_test=args.n_test, test_seed=args.test_seed)

    oracle = mm.oracle_metrics(mu_true, sigma_true)
    scalar_rows += _scalar_rows(
        {'crps': oracle['oracle_crps'], 'nll': oracle['oracle_nll']},
        method='oracle', dgp=dgp, scenario='baseline', ensemble_size=None, replicate=0, seed=args.seed)

    raw_by_method = {}  # method -> full-size (mu_samples, sigma2_samples), for the sweeps below

    for method in methods:
        print(f"-- {method} --")
        mu_samples, sigma2_samples = TRAIN_PREDICT_FNS[method](x_train, y_train, x_test, args.seed, args)
        mu_samples, sigma2_samples = mm.normalize_mixture_arrays(mu_samples, sigma2_samples, n_expected=len(x_test))
        raw_by_method[method] = (mu_samples, sigma2_samples)

        # Persist the raw per-member predictions -- same save_model_outputs convention as
        # the other experiment scripts -- so this baseline cell's mixture can be rescored
        # later (new metrics, sanity checks) without retraining.
        results_save_module.save_model_outputs(
            mu_samples=mu_samples, sigma2_samples=sigma2_samples,
            x_grid=x_test, y_grid_clean=mu_true,
            x_train_subset=x_train, y_train_subset=y_train,
            model_name=method, noise_type=noise_type, func_type=func_type,
            subfolder='predictive_eval', seed=args.seed, date=date,
            **RAW_OUTPUT_EXTRA_KWARGS[method](args))

        # MC_Dropout rows carry the dropout_p used, so the fixed-p baseline/size-sweep
        # rows are traceable alongside the varying-p sweep below (NaN for other methods).
        extra = {'dropout_p': args.mc_dropout_p} if method == 'MC_Dropout' else {}
        result = mm.evaluate_mixture(x_test, y_test, mu_samples, sigma2_samples, mu_true, sigma_true,
                                      coverage_levels=args.coverage_levels, n_bins=args.n_bins,
                                      include_stratified=True)
        tags = dict(method=method, dgp=dgp, scenario='baseline',
                    ensemble_size=mu_samples.shape[0], replicate=0, seed=args.seed, **extra)
        scalar_rows += _scalar_rows(result['scalar'], **tags)
        pit_rows += _pit_rows(x_test, y_test, result['pit'], **tags)
        strat_rows += _stratified_rows(result['stratified'], **tags)
        print(f"   rmse={result['scalar']['rmse']:.4f}  crps={result['scalar']['crps']:.4f}  "
              f"nll={result['scalar']['nll']:.4f}  coverage_0.9={result['scalar']['coverage_0.9']:.3f}")

    for method, (mu_all, sigma2_all) in raw_by_method.items():
        grid = getattr(args, SIZE_SWEEP_GRID_ARG[method])
        extra = {'dropout_p': args.mc_dropout_p} if method == 'MC_Dropout' else {}
        scalar_rows += _component_size_sweep(method, mu_all, sigma2_all, grid, args.ensemble_size_r,
                                              args.ensemble_size_seed, dgp, x_test, y_test,
                                              mu_true, sigma_true, args, extra_tags=extra)

    if 'MC_Dropout' in methods and include_p_sweep:
        # Genuinely retrains per (p, replicate) -- deliberately NOT repeated per outer seed
        # (see main()'s seed loop) to avoid 5x'ing an already-expensive retraining sweep.
        # Its own replicate variability comes from args.mc_dropout_p_sweep_seed + rep,
        # independent of the outer seed loop's purpose (baseline/component-count
        # seed-sensitivity). Documented limitation: the p-sweep's variability estimate is
        # therefore seed-locked, not pooled across the 5 outer seeds like everything else.
        scalar_rows += run_mc_dropout_p_sweep(x_train, y_train, x_test, y_test, mu_true, sigma_true, dgp, args)

    return scalar_rows, pit_rows, strat_rows


# ============================== CLI ==============================

def apply_smoke_overrides(args):
    """Shrink all model hyperparameters to fast/tiny values for plumbing verification
    only -- results at these settings are not statistically meaningful."""
    args.n_train = 200
    args.n_test = 60
    args.mc_dropout_epochs = 50
    args.mc_dropout_m = 20
    args.ensemble_epochs = 50
    args.ensemble_k = 5
    args.bamlss_n_iter = 600
    args.bamlss_burnin = 100
    args.bamlss_thin = 5
    args.bamlss_nsamples = 50
    args.ensemble_size_r = 3
    args.ensemble_size_grid = "2,3,5"
    args.mc_dropout_m_grid = "5,10,20"
    args.mc_dropout_p_grid = "0.25,0.5"
    args.mc_dropout_p_sweep_r = 1
    args.bamlss_nsamples_grid = "10,25,50"


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", nargs="?", default=None,
                    help="Single-cell run: 'method:func_type:noise_type', "
                         "e.g. 'MC_Dropout:linear:homoscedastic'.")
    p.add_argument("--batch", action="store_true", help="Full sweep over --methods x --dgps.")
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--dgps", default=",".join(f"{f}:{n}" for f, n in ALL_DGPS))
    p.add_argument("--smoke", action="store_true",
                    help="Shrink all model hyperparameters for a fast plumbing check "
                         "(not statistically meaningful).")

    p.add_argument("--seed", type=int, default=42,
                    help="Single training/model seed for a --scenario or default smoke run. "
                         "Ignored (overridden by --seeds) under --batch.")
    p.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated outer seeds to repeat the full sweep over, "
                         "matching this repo's 5-seed replication convention used elsewhere "
                         "(training/model seeding only -- --test-seed stays fixed across all "
                         "of them, since the test set is deliberately seed-independent). "
                         "Defaults to '42,43,44,45,46' under --batch, or just [--seed] "
                         "otherwise, so quick manual/smoke invocations aren't silently 5x'd.")
    p.add_argument("--test-seed", type=int, default=20260809)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--train-range", type=float, nargs=2, default=[-5, 10])
    p.add_argument("--n-test", type=int, default=300)

    p.add_argument("--mc-dropout-p", type=float, default=0.25)
    p.add_argument("--mc-dropout-beta", type=float, default=0.5)
    p.add_argument("--mc-dropout-epochs", type=int, default=500)
    p.add_argument("--mc-dropout-lr", type=float, default=1e-3)
    p.add_argument("--mc-dropout-batch-size", type=int, default=32)
    p.add_argument("--mc-dropout-m", type=int, default=100)
    p.add_argument("--mc-dropout-m-grid", type=str, default="5,10,20,50,100",
                    help="Component-count sweep for MC Dropout's M (prediction-time only, "
                         "no retraining -- subsampled from the baseline M forward passes).")
    p.add_argument("--mc-dropout-p-grid", type=str, default="0.1,0.25,0.4,0.5",
                    help="Dropout-probability sweep. Unlike M, p is a training hyperparameter, "
                         "so this retrains once per (p, replicate).")
    p.add_argument("--mc-dropout-p-sweep-r", type=int, default=1,
                    help="Retraining repeats per p value (>1 costs a full retrain each time).")
    p.add_argument("--mc-dropout-p-sweep-seed", type=int, default=888)

    p.add_argument("--ensemble-k", type=int, default=20)
    p.add_argument("--ensemble-beta", type=float, default=0.5)
    p.add_argument("--ensemble-epochs", type=int, default=500)
    p.add_argument("--ensemble-batch-size", type=int, default=32)

    p.add_argument("--bamlss-n-iter", type=int, default=12000)
    p.add_argument("--bamlss-burnin", type=int, default=2000)
    p.add_argument("--bamlss-thin", type=int, default=10)
    p.add_argument("--bamlss-nsamples", type=int, default=1000)
    p.add_argument("--bamlss-nsamples-grid", type=str, default="10,50,100,250,500,1000",
                    help="Component-count sweep for BAMLSS's nsamples (prediction-time only, "
                         "no re-fitting -- subsampled from the baseline posterior draws).")

    p.add_argument("--coverage-levels", type=str, default="0.5,0.8,0.9,0.95")
    p.add_argument("--n-bins", type=int, default=10)

    p.add_argument("--ensemble-size-grid", type=str, default="2,3,5,10,20",
                    help="Component-count sweep for Deep Ensemble's K.")
    p.add_argument("--ensemble-size-r", type=int, default=20,
                    help="Random-subset replicates, shared by all three component-count sweeps "
                         "(Deep Ensemble K, MC Dropout M, BAMLSS nsamples).")
    p.add_argument("--ensemble-size-seed", type=int, default=777)

    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--figures", dest="figures", action="store_true", default=True)
    p.add_argument("--no-figures", dest="figures", action="store_false")
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.smoke:
        apply_smoke_overrides(args)

    if args.scenario:
        method, func_type, noise_type = args.scenario.split(":")
        if method not in TRAIN_PREDICT_FNS:
            raise ValueError(f"Unknown method '{method}'. Must be one of {list(TRAIN_PREDICT_FNS)}")
        methods = [method]
        dgps = [(func_type, noise_type)]
    elif args.batch:
        methods = args.methods.split(",")
        dgps = [tuple(x.split(":")) for x in args.dgps.split(",")]
    else:
        print("No scenario/--batch given: running a quick single-cell smoke default "
              "(MC_Dropout, linear-homoscedastic). Pass --batch for the full sweep, or a "
              "positional 'method:func_type:noise_type' for a specific cell.")
        methods = ["MC_Dropout"]
        dgps = [("linear", "homoscedastic")]
        if not args.smoke:
            apply_smoke_overrides(args)

    args.coverage_levels = [float(x) for x in args.coverage_levels.split(",")]
    args.ensemble_size_grid = [int(x) for x in args.ensemble_size_grid.split(",")]
    args.mc_dropout_m_grid = [int(x) for x in args.mc_dropout_m_grid.split(",")]
    args.mc_dropout_p_grid = [float(x) for x in args.mc_dropout_p_grid.split(",")]
    args.bamlss_nsamples_grid = [int(x) for x in args.bamlss_nsamples_grid.split(",")]

    # 5-seed replication, matching this repo's convention elsewhere -- default only kicks in
    # for --batch, so a quick --scenario/smoke invocation isn't silently 5x'd. --test-seed
    # stays a single fixed value across every outer seed (the test set is deliberately
    # seed-independent, see mm.make_test_set's docstring).
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif args.batch:
        seeds = [42, 43, 44, 45, 46]
    else:
        seeds = [args.seed]

    out_root = args.out_root or (project_root / "results" / "predictive_eval")
    results_save_module.plots_dir = out_root / "plots"
    pio.csv_dir = out_root / "csv"
    date = datetime.now().strftime('%Y%m%d')

    all_scalar, all_pit, all_strat = [], [], []
    for i, seed in enumerate(seeds):
        args.seed = seed
        print(f"\n{'#' * 70}\nOuter seed {seed} ({i + 1}/{len(seeds)})\n{'#' * 70}")
        for func_type, noise_type in dgps:
            scalar_rows, pit_rows, strat_rows = run_dgp(func_type, noise_type, methods, args, date,
                                                          include_p_sweep=(i == 0))
            all_scalar += scalar_rows
            all_pit += pit_rows
            all_strat += strat_rows

    scalar_df = pd.DataFrame(all_scalar)
    pit_df = pd.DataFrame(all_pit)
    strat_df = pd.DataFrame(all_strat)

    pio.save_tidy_csv(scalar_df, f"{date}_scalar_metrics")
    pio.save_tidy_csv(pit_df, f"{date}_pit_values")
    pio.save_tidy_csv(strat_df, f"{date}_stratified_coverage")

    if args.figures and len(scalar_df):
        pep.plot_reliability_curve(scalar_df, subfolder="", filename=f"{date}_reliability_curve")
        pep.plot_sharpness_distribution(scalar_df, subfolder="", filename=f"{date}_sharpness_distribution")
        if (scalar_df['scenario'] == 'ensemble_size_sweep').any():
            pep.plot_crps_coverage_vs_ensemble_size(scalar_df, subfolder="",
                                                     filename=f"{date}_crps_coverage_vs_ensemble_size")
        if (scalar_df['scenario'] == 'mc_dropout_p_sweep').any():
            pep.plot_mc_dropout_p_sweep(scalar_df, subfolder="", filename=f"{date}_mc_dropout_p_sweep")
        if len(pit_df):
            pep.plot_pit_histograms(pit_df, subfolder="", filename=f"{date}_pit_histograms")

    print(f"\nDone. scalar={len(scalar_df)} rows, pit={len(pit_df)} rows, stratified={len(strat_df)} rows.")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
