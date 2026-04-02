"""
2×4 panels for moment-matched entropy summaries (sample size & noise level).

Columns: Deep Ensemble, MC Dropout, BNN, BAMLSS.
Row 1: normalized AU, EU, TU vs. percentage or τ.
Row 2: correlation (epistemic vs. aleatoric) vs. same x-axis.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.regression_summary_panels import FUNC_NAME_BY_TYPE

FuncType = Literal["linear", "sin"]
NoiseType = Literal["homoscedastic", "heteroscedastic"]

MODEL_ORDER = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]

AU_COL = "Avg_Aleatoric_Entropy_norm"
EU_COL = "Avg_Epistemic_Entropy_norm"
TU_COL = "Avg_Total_Entropy_norm"
CORR_COL = "Correlation_Epi_Ale"


def _annotate_series_percentage(ax, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03) -> None:
    """Small value labels next to markers (matches regression summary sample-size panels)."""
    if len(xs) == 0:
        return
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    span = max(y_max - y_min, 1e-6)
    for x, y in zip(xs, ys):
        dy = dy_scale * span
        ax.text(x + 0.8, y + dy, fmt.format(y), fontsize=9, alpha=0.9, ha="left", va="bottom")


def _annotate_series_tau(ax, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03) -> None:
    """Small value labels next to markers vs. τ (matches regression summary noise-level panels)."""
    if len(xs) == 0:
        return
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    span = max(y_max - y_min, 1e-6)
    dx = 0.02 * (float(xs[-1] - xs[0]) if len(xs) > 1 and xs[-1] != xs[0] else 1.0)
    for x, y in zip(xs, ys):
        dy = dy_scale * span
        ax.text(x + dx, y + dy, fmt.format(y), fontsize=9, alpha=0.9, ha="left", va="bottom")


def _empty_cell_axes(ax_top, ax_bot, title: str) -> None:
    ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes, fontsize=11)
    ax_bot.axis("off")
    ax_top.set_title(title, fontweight="bold", fontsize=11)


def create_sample_size_moment_matched_entropy_panel_2x4(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """Moment-matched entropy: norm. AU/EU/TU and correlation vs. training percentage."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    func_label = FUNC_NAME_BY_TYPE[func_type]

    for col, model_name in enumerate(MODEL_ORDER):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty or "Percentage" not in df.columns:
            _empty_cell_axes(ax_top, ax_bot, model_name)
            continue

        x = df["Percentage"].values.astype(float)
        ax_top.plot(x, df[AU_COL].values, "o-", color="green", markersize=5, linewidth=1.8, label="AU")
        ax_top.plot(x, df[EU_COL].values, "s-", color="orange", markersize=5, linewidth=1.8, label="EU")
        if TU_COL in df.columns:
            ax_top.plot(x, df[TU_COL].values, "^-", color="blue", markersize=5, linewidth=1.8, label="TU")
        ax_top.set_ylabel("Norm. entropy", fontsize=10)
        ax_top.set_title(model_name, fontweight="bold", fontsize=11)
        ax_top.grid(True, alpha=0.3)
        h, _ = ax_top.get_legend_handles_labels()
        if h:
            ax_top.legend(fontsize=7, loc="best")

        if CORR_COL in df.columns:
            y_c = df[CORR_COL].values
            ax_bot.plot(x, y_c, "D-", color="purple", markersize=5, linewidth=1.8, label="Corr(EU, AU)")
            ax_bot.set_ylim(-1.05, 1.05)
            ax_bot.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
            hb, _ = ax_bot.get_legend_handles_labels()
            if hb:
                ax_bot.legend(fontsize=7, loc="best")
        else:
            ax_bot.text(0.5, 0.5, "No correlation", ha="center", va="center", transform=ax_bot.transAxes, fontsize=9)

        ax_bot.set_xlabel("Training data (%)", fontsize=10)
        ax_bot.set_ylabel("Corr(EU, AU)", fontsize=10)
        ax_bot.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Norm. entropy", fontsize=10)
    axes[1, 0].set_ylabel("Corr(EU, AU)", fontsize=10)
    title = f"Sample size — {func_label} ({noise_type}) — Entropy (normalized)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if uncertainty_ylim is not None:
        for c in range(4):
            axes[0, c].set_ylim(*uncertainty_ylim)
    return fig


def create_sample_size_moment_matched_entropy_panel_1x4_au_eu(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """1×4: moment-matched norm. AU/EU/TU vs. training percentage (appendix split)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    func_label = FUNC_NAME_BY_TYPE[func_type]

    for col, model_name in enumerate(MODEL_ORDER):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty or "Percentage" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        x = df["Percentage"].values.astype(float)
        y_au = df[AU_COL].values
        y_eu = df[EU_COL].values
        ax.plot(x, y_au, "o-", color="green", markersize=5, linewidth=1.8, label="AU")
        ax.plot(x, y_eu, "s-", color="orange", markersize=5, linewidth=1.8, label="EU")
        if TU_COL in df.columns:
            y_tu = df[TU_COL].values
            ax.plot(x, y_tu, "^-", color="blue", markersize=5, linewidth=1.8, label="TU")
        _annotate_series_percentage(ax, x, y_au)
        _annotate_series_percentage(ax, x, y_eu)
        if TU_COL in df.columns:
            _annotate_series_percentage(ax, x, y_tu)
        ax.set_ylabel("Norm. entropy", fontsize=10)
        ax.set_title(model_name, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training data (%)", fontsize=10)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=7, loc="best")

    axes[0].set_ylabel("Norm. entropy", fontsize=10)
    title = f"Sample size — {func_label} ({noise_type}) — Entropy (normalized) — AU / EU / TU"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if uncertainty_ylim is not None:
        for i in range(4):
            axes[i].set_ylim(*uncertainty_ylim)
    return fig


def create_sample_size_moment_matched_entropy_panel_1x4_corr(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
):
    """1×4: Corr(EU, AU) vs. training percentage (appendix split)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.2), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    func_label = FUNC_NAME_BY_TYPE[func_type]

    for col, model_name in enumerate(MODEL_ORDER):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty or "Percentage" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        x = df["Percentage"].values.astype(float)
        if CORR_COL in df.columns:
            y_c = df[CORR_COL].values
            ax.plot(x, y_c, "D-", color="purple", markersize=5, linewidth=1.8, label="Corr(EU, AU)")
            _annotate_series_percentage(ax, x, y_c, fmt="{:.2f}", dy_scale=0.03)
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
            hb, _ = ax.get_legend_handles_labels()
            if hb:
                ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No correlation", ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_xlabel("Training data (%)", fontsize=10)
        ax.set_ylabel("Corr(EU, AU)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(model_name, fontweight="bold", fontsize=11)

    axes[0].set_ylabel("Corr(EU, AU)", fontsize=10)
    title = f"Sample size — {func_label} ({noise_type}) — Entropy (normalized) — correlation"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def create_noise_level_moment_matched_entropy_panel_1x4_au_eu(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    distribution: str = "normal",
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """1×4: moment-matched norm. AU/EU/TU vs. τ (appendix split)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    func_label = FUNC_NAME_BY_TYPE[func_type]

    for col, model_name in enumerate(MODEL_ORDER):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty or "Tau" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        x = df["Tau"].values.astype(float)
        order = np.argsort(x)
        x = x[order]
        y_au = df[AU_COL].values[order]
        y_eu = df[EU_COL].values[order]
        ax.plot(x, y_au, "o-", color="green", markersize=5, linewidth=1.8, label="AU")
        ax.plot(x, y_eu, "s-", color="orange", markersize=5, linewidth=1.8, label="EU")
        if TU_COL in df.columns:
            y_tu = df[TU_COL].values[order]
            ax.plot(x, y_tu, "^-", color="blue", markersize=5, linewidth=1.8, label="TU")
        _annotate_series_tau(ax, x, y_au)
        _annotate_series_tau(ax, x, y_eu)
        if TU_COL in df.columns:
            _annotate_series_tau(ax, x, y_tu)
        ax.set_ylabel("Norm. entropy", fontsize=10)
        ax.set_title(model_name, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Noise level (τ)", fontsize=10)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=7, loc="best")

    axes[0].set_ylabel("Norm. entropy", fontsize=10)
    title = (
        f"Noise level ({distribution}) — {func_label} ({noise_type}) — "
        "Entropy (normalized) — AU / EU / TU"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if uncertainty_ylim is not None:
        for i in range(4):
            axes[i].set_ylim(*uncertainty_ylim)
    return fig


def create_noise_level_moment_matched_entropy_panel_1x4_corr(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    distribution: str = "normal",
):
    """1×4: Corr(EU, AU) vs. τ (appendix split)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.2), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    func_label = FUNC_NAME_BY_TYPE[func_type]

    for col, model_name in enumerate(MODEL_ORDER):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty or "Tau" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        x = df["Tau"].values.astype(float)
        order = np.argsort(x)
        x = x[order]
        if CORR_COL in df.columns:
            y_c = df[CORR_COL].values[order]
            ax.plot(x, y_c, "D-", color="purple", markersize=5, linewidth=1.8, label="Corr(EU, AU)")
            _annotate_series_tau(ax, x, y_c, fmt="{:.2f}", dy_scale=0.03)
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
            hb, _ = ax.get_legend_handles_labels()
            if hb:
                ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No correlation", ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_xlabel("Noise level (τ)", fontsize=10)
        ax.set_ylabel("Corr(EU, AU)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(model_name, fontweight="bold", fontsize=11)

    axes[0].set_ylabel("Corr(EU, AU)", fontsize=10)
    title = (
        f"Noise level ({distribution}) — {func_label} ({noise_type}) — "
        "Entropy (normalized) — correlation"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def create_noise_level_moment_matched_entropy_panel_2x4(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    distribution: str = "normal",
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """Moment-matched entropy: norm. AU/EU/TU and correlation vs. noise level τ."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    func_label = FUNC_NAME_BY_TYPE[func_type]

    for col, model_name in enumerate(MODEL_ORDER):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty or "Tau" not in df.columns:
            _empty_cell_axes(ax_top, ax_bot, model_name)
            continue

        x = df["Tau"].values.astype(float)
        order = np.argsort(x)
        x = x[order]
        ax_top.plot(x, df[AU_COL].values[order], "o-", color="green", markersize=5, linewidth=1.8, label="AU")
        ax_top.plot(x, df[EU_COL].values[order], "s-", color="orange", markersize=5, linewidth=1.8, label="EU")
        if TU_COL in df.columns:
            ax_top.plot(x, df[TU_COL].values[order], "^-", color="blue", markersize=5, linewidth=1.8, label="TU")
        ax_top.set_ylabel("Norm. entropy", fontsize=10)
        ax_top.set_title(model_name, fontweight="bold", fontsize=11)
        ax_top.grid(True, alpha=0.3)
        h, _ = ax_top.get_legend_handles_labels()
        if h:
            ax_top.legend(fontsize=7, loc="best")

        if CORR_COL in df.columns:
            y_c = df[CORR_COL].values[order]
            ax_bot.plot(x, y_c, "D-", color="purple", markersize=5, linewidth=1.8, label="Corr(EU, AU)")
            ax_bot.set_ylim(-1.05, 1.05)
            ax_bot.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
            hb, _ = ax_bot.get_legend_handles_labels()
            if hb:
                ax_bot.legend(fontsize=7, loc="best")
        else:
            ax_bot.text(0.5, 0.5, "No correlation", ha="center", va="center", transform=ax_bot.transAxes, fontsize=9)

        ax_bot.set_xlabel("Noise level (τ)", fontsize=10)
        ax_bot.set_ylabel("Corr(EU, AU)", fontsize=10)
        ax_bot.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Norm. entropy", fontsize=10)
    axes[1, 0].set_ylabel("Corr(EU, AU)", fontsize=10)
    title = (
        f"Noise level ({distribution}) — {func_label} ({noise_type}) — "
        "Entropy (normalized)"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if uncertainty_ylim is not None:
        for c in range(4):
            axes[0, c].set_ylim(*uncertainty_ylim)
    return fig
