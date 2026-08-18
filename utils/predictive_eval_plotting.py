"""
Figures for the predictive-distribution evaluation script: reliability curve, sharpness
distribution, CRPS/coverage vs Deep Ensemble size, and PIT histograms.

Follows utils.panel_plotting's faceted-figure idiom (n_cols = min(3, n_facets),
plt.subplots(rows, cols, figsize=(6*cols, 4*rows), sharex=True, squeeze=False), unused
axes hidden via fig.delaxes) and reuses utils.results_save.save_plot so output PNGs
match the rest of the repo (dpi=300, bbox_inches='tight'). Per-method colors reuse the
existing model_colors convention from utils.dashboards.py.

Dispersion on the two sweep figures is ACROSS SEEDS: the eval script runs exactly one
independently-seeded cell per (grid value, seed), so each error bar is the mean +/- std
over the 5 outer seeds and the dimmed points behind it are those individual seeds.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from utils.predictive_eval_aggregate import seed_mean_std
from utils.results_save import save_plot

METHOD_COLORS = {
    'MC_Dropout': '#1f77b4',
    'Deep_Ensemble': '#ff7f0e',
    'BNN': '#2ca02c',
    'BAMLSS': '#d62728',
}
ORACLE_COLOR = '#555555'


def _facet_grid(n_facets, sharex=True, sharey=False, panel_w=6, panel_h=4):
    """Same layout formula as utils.panel_plotting: n_cols=min(3,n_facets), n_rows=ceil.
    Returns (fig, axes_flat) with unused trailing axes already removed."""
    n_cols = min(3, n_facets)
    n_rows = math.ceil(n_facets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows),
                              sharex=sharex, sharey=sharey, squeeze=False)
    axes = axes.flatten()
    for i in range(n_facets, len(axes)):
        fig.delaxes(axes[i])
    return fig, axes[:n_facets]


def _method_color(method):
    return METHOD_COLORS.get(method, '#9467bd')


def plot_reliability_curve(scalar_df, subfolder="predictive_eval/plots", filename="reliability_curve"):
    """Nominal vs empirical coverage, one line per method, diagonal reference, faceted by DGP."""
    baseline = scalar_df[scalar_df['scenario'] == 'baseline']
    cov = baseline[baseline['metric'].str.startswith('coverage_')].copy()
    cov['nominal'] = cov['metric'].str.replace('coverage_', '', regex=False).astype(float)

    dgps = sorted(cov['dgp'].unique())
    fig, axes = _facet_grid(len(dgps), sharex=True, sharey=True)

    for ax, dgp in zip(axes, dgps):
        sub = cov[cov['dgp'] == dgp]
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1, alpha=0.6, label='Ideal')
        for method in sorted(sub['method'].unique()):
            m = sub[sub['method'] == method].sort_values('nominal')
            ax.plot(m['nominal'], m['value'], marker='o', linewidth=1.8,
                     color=_method_color(method), label=method)
        ax.set_title(dgp, fontsize=11)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle("Reliability: nominal vs empirical coverage (baseline)", y=1.02)
    fig.tight_layout()
    path = save_plot(fig, filename, subfolder=subfolder)
    plt.close(fig)
    return path


def plot_sharpness_distribution(scalar_df, subfolder="predictive_eval/plots", filename="sharpness_distribution"):
    """Grid: rows=DGP, cols=sharpness metric (mean_width_90, median_width_90, mean_pred_std);
    grouped bar per method."""
    baseline = scalar_df[scalar_df['scenario'] == 'baseline']
    metrics = ['mean_width_90', 'median_width_90', 'mean_pred_std']
    dgps = sorted(baseline['dgp'].unique())

    fig, axes = plt.subplots(len(dgps), len(metrics),
                              figsize=(5 * len(metrics), 3.5 * len(dgps)), squeeze=False)

    for i, dgp in enumerate(dgps):
        d = baseline[baseline['dgp'] == dgp]
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            m = d[d['metric'] == metric].sort_values('method')
            colors = [_method_color(x) for x in m['method']]
            ax.bar(m['method'], m['value'], color=colors)
            ax.set_title(f"{dgp}\n{metric}" if j == 0 else metric, fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle("Sharpness (baseline)", y=1.01)
    fig.tight_layout()
    path = save_plot(fig, filename, subfolder=subfolder)
    plt.close(fig)
    return path


def plot_crps_coverage_vs_ensemble_size(scalar_df, subfolder="predictive_eval/plots",
                                         filename="crps_coverage_vs_ensemble_size"):
    """All methods with a component-count sweep (scenario='ensemble_size_sweep' -- Deep
    Ensemble's K, MC Dropout's M, and BAMLSS's nsamples are all, at heart, 'how many
    mixture components did we use'). 2 stacked rows (CRPS, coverage_0.9) x facets=DGP.
    Per facet: x=component count, one mean +/- std (across the outer seeds) errorbar line
    per method with the individual per-seed points dimmed behind it; CRPS row gets the
    oracle CRPS as a dashed horizontal reference; coverage row gets a dashed line at the
    nominal level (0.9) itself."""
    sweep = scalar_df[scalar_df['scenario'] == 'ensemble_size_sweep']
    oracle = scalar_df[(scalar_df['method'] == 'oracle') & (scalar_df['metric'] == 'crps')]
    # From `sweep`, not the full frame: a DGP with no sweep rows would otherwise get an
    # empty column pair.
    dgps = sorted(sweep['dgp'].unique())
    methods = sorted(sweep['method'].unique())

    fig, axes = plt.subplots(2, len(dgps), figsize=(5 * len(dgps), 8), sharex='col', squeeze=False)

    for col, dgp in enumerate(dgps):
        ax_crps, ax_cov = axes[0][col], axes[1][col]

        for method in methods:
            color = _method_color(method)
            for ax, metric in ((ax_crps, 'crps'), (ax_cov, 'coverage_0.9')):
                d = sweep[(sweep['dgp'] == dgp) & (sweep['metric'] == metric)
                          & (sweep['method'] == method)]
                if not len(d):
                    continue
                ax.scatter(d['ensemble_size'], d['value'], alpha=0.25, s=12,
                           color=color, zorder=1)
                agg = seed_mean_std(d, 'ensemble_size')
                # std is NaN for a single-seed run; draw those as bare markers.
                ax.errorbar(agg['ensemble_size'], agg['mean'], yerr=agg['std'].fillna(0.0),
                            color=color, linewidth=2, marker='o', capsize=3,
                            zorder=3, label=method)

        oracle_d = oracle[oracle['dgp'] == dgp]
        if len(oracle_d):
            ax_crps.axhline(oracle_d['value'].iloc[0], linestyle='--', color=ORACLE_COLOR, label='Oracle CRPS')
        ax_crps.set_title(dgp, fontsize=11)
        ax_crps.set_ylabel("CRPS")
        ax_crps.grid(True, alpha=0.3)
        ax_crps.legend(fontsize=7)

        ax_cov.axhline(0.9, linestyle='--', color=ORACLE_COLOR, label='Nominal 0.9')
        ax_cov.set_xlabel("Number of mixture components")
        ax_cov.set_ylabel("Coverage @ 90%")
        ax_cov.set_ylim(0, 1)
        ax_cov.grid(True, alpha=0.3)
        ax_cov.legend(fontsize=7)

    fig.suptitle("CRPS and coverage-at-90 vs number of mixture components", y=1.02)
    fig.tight_layout()
    path = save_plot(fig, filename, subfolder=subfolder)
    plt.close(fig)
    return path


def plot_mc_dropout_p_sweep(scalar_df, subfolder="predictive_eval/plots",
                             filename="mc_dropout_p_sweep"):
    """MC Dropout only (dropout_p is a training hyperparameter, so this is a separate
    scenario from the component-count sweeps above). 2 stacked rows (CRPS, coverage_0.9)
    x facets=DGP; x=dropout_p, per-seed points behind a mean +/- std (across seeds)
    errorbar line, oracle CRPS / nominal-0.9 dashed references as in
    plot_crps_coverage_vs_ensemble_size."""
    sweep = scalar_df[scalar_df['scenario'] == 'mc_dropout_p_sweep']
    oracle = scalar_df[(scalar_df['method'] == 'oracle') & (scalar_df['metric'] == 'crps')]
    dgps = sorted(sweep['dgp'].unique())
    if not dgps:
        return None
    color = _method_color('MC_Dropout')

    fig, axes = plt.subplots(2, len(dgps), figsize=(5 * len(dgps), 8), sharex='col', squeeze=False)

    for col, dgp in enumerate(dgps):
        ax_crps, ax_cov = axes[0][col], axes[1][col]

        crps_d = sweep[(sweep['dgp'] == dgp) & (sweep['metric'] == 'crps')]
        ax_crps.scatter(crps_d['dropout_p'], crps_d['value'], alpha=0.3, s=16,
                        color=color, zorder=1)
        if len(crps_d):
            agg = seed_mean_std(crps_d, 'dropout_p')
            ax_crps.errorbar(agg['dropout_p'], agg['mean'], yerr=agg['std'].fillna(0.0),
                             color=color, linewidth=2, marker='o', capsize=3, zorder=3)
        oracle_d = oracle[oracle['dgp'] == dgp]
        if len(oracle_d):
            ax_crps.axhline(oracle_d['value'].iloc[0], linestyle='--', color=ORACLE_COLOR, label='Oracle CRPS')
            ax_crps.legend(fontsize=8)
        ax_crps.set_title(dgp, fontsize=11)
        ax_crps.set_ylabel("CRPS")
        ax_crps.grid(True, alpha=0.3)

        cov_d = sweep[(sweep['dgp'] == dgp) & (sweep['metric'] == 'coverage_0.9')]
        ax_cov.scatter(cov_d['dropout_p'], cov_d['value'], alpha=0.3, s=16,
                       color=color, zorder=1)
        if len(cov_d):
            agg = seed_mean_std(cov_d, 'dropout_p')
            ax_cov.errorbar(agg['dropout_p'], agg['mean'], yerr=agg['std'].fillna(0.0),
                            color=color, linewidth=2, marker='o', capsize=3, zorder=3)
        ax_cov.axhline(0.9, linestyle='--', color=ORACLE_COLOR, label='Nominal 0.9')
        ax_cov.set_xlabel("Dropout probability (p)")
        ax_cov.set_ylabel("Coverage @ 90%")
        ax_cov.set_ylim(0, 1)
        ax_cov.grid(True, alpha=0.3)
        ax_cov.legend(fontsize=8)

    fig.suptitle("MC Dropout: CRPS and coverage-at-90 vs dropout probability", y=1.02)
    fig.tight_layout()
    path = save_plot(fig, filename, subfolder=subfolder)
    plt.close(fig)
    return path


def plot_pit_histograms(pit_df, subfolder="predictive_eval/plots", filename="pit_histograms", n_bins=10):
    """Grid rows=method, cols=DGP; PIT histogram (density) + dashed uniform-density
    reference at height 1."""
    methods = sorted(pit_df['method'].unique())
    dgps = sorted(pit_df['dgp'].unique())

    fig, axes = plt.subplots(len(methods), len(dgps),
                              figsize=(4.5 * len(dgps), 3.2 * len(methods)), squeeze=False)

    for i, method in enumerate(methods):
        for j, dgp in enumerate(dgps):
            ax = axes[i][j]
            vals = pit_df[(pit_df['method'] == method) & (pit_df['dgp'] == dgp)]['pit_value']
            ax.hist(vals, bins=n_bins, range=(0, 1), density=True,
                    color=_method_color(method), alpha=0.75, edgecolor='white')
            ax.axhline(1.0, linestyle='--', color='black', linewidth=1, alpha=0.6)
            if i == 0:
                ax.set_title(dgp, fontsize=10)
            if j == 0:
                ax.set_ylabel(method, fontsize=10)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

    fig.suptitle("PIT histograms (baseline)", y=1.01)
    fig.tight_layout()
    path = save_plot(fig, filename, subfolder=subfolder)
    plt.close(fig)
    return path
