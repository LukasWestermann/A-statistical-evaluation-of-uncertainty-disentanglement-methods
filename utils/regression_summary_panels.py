"""
Shared 4×2 summary line plots (AU/EU vs. percentage or τ, plus correlation row).

Used by scripts/plot_sample_size_summary_4x2.py, scripts/plot_noise_level_summary_4x2.py,
scripts/plot_summary_panels_consolidated.py, and moment-matched batch recomputation outputs.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FuncType = Literal["linear", "sin"]
NoiseType = Literal["homoscedastic", "heteroscedastic"]
MeasureType = Literal["variance", "entropy"]

MODEL_KEYS: Dict[str, str] = {
    "Deep Ensemble": "Deep_Ensemble",
    "MC Dropout": "MC_Dropout",
    "BNN": "BNN",
    "BAMLSS": "BAMLSS",
}

FUNC_NAME_BY_TYPE: Dict[FuncType, str] = {
    "linear": "Linear",
    "sin": "Sinusoidal",
}

# Fixed y-range for normalized AU/EU/(TU) panels when cross-figure comparison is needed.
NORMALIZED_UNCERTAINTY_YLIM: Tuple[float, float] = (0.0, 1.5)
NORMALIZED_UNCERTAINTY_YLIM_01 = NORMALIZED_UNCERTAINTY_YLIM  # backward-compatible alias


def get_uncertainty_columns(
    df: pd.DataFrame,
    measure: MeasureType,
) -> Tuple[str, str, Optional[str]]:
    """Column names for AU, EU and optional total uncertainty."""
    if measure == "variance":
        au_col = "Avg_Aleatoric_norm"
        eu_col = "Avg_Epistemic_norm"
        tot_col = "Avg_Total_norm" if "Avg_Total_norm" in df.columns else None
    else:
        au_col = "Avg_Aleatoric_Entropy_norm"
        eu_col = "Avg_Epistemic_Entropy_norm"
        tot_col = (
            "Avg_Total_Entropy_norm"
            if "Avg_Total_Entropy_norm" in df.columns
            else None
        )

    return au_col, eu_col, tot_col


def create_sample_size_summary_panel(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    measure: MeasureType,
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """
    4×2 panel: models as columns; row 1 = norm. avg AU/EU vs. Percentage; row 2 = correlation.

    If ``uncertainty_ylim`` is set (e.g. ``(0, 1.5)``), row 1 uses that y-axis range on every column
    (including empty cells) so panels are visually comparable.
    """
    model_order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

    for col, model_name in enumerate(model_order):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        df = stats_by_model.get(model_name)
        if df is None or df.empty:
            ax_top.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax_top.transAxes,
                fontsize=11,
            )
            ax_bot.axis("off")
            ax_top.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        percentages = df["Percentage"].values

        au_col, eu_col, tot_col = get_uncertainty_columns(df, measure)
        corr_col = "Correlation_Epi_Ale"

        y_au = df[au_col].values
        y_eu = df[eu_col].values

        ax_top.plot(
            percentages,
            y_au,
            "o-",
            markersize=6,
            linewidth=2,
            color="green",
            label="AU",
        )
        ax_top.plot(
            percentages,
            y_eu,
            "s-",
            markersize=6,
            linewidth=2,
            color="orange",
            label="EU",
        )

        if tot_col is not None:
            y_tot = df[tot_col].values
            ax_top.plot(
                percentages,
                y_tot,
                "^-",
                markersize=6,
                linewidth=2,
                color="blue",
                label="Total",
            )

        def _annotate_series(ax, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03):
            if len(xs) == 0:
                return
            y_min, y_max = np.min(ys), np.max(ys)
            span = max(y_max - y_min, 1e-6)
            for x, y in zip(xs, ys):
                dy = dy_scale * span
                ax.text(
                    x + 0.8,
                    y + dy,
                    fmt.format(y),
                    fontsize=9,
                    alpha=0.9,
                    ha="left",
                    va="bottom",
                )

        _annotate_series(ax_top, percentages, y_au)
        _annotate_series(ax_top, percentages, y_eu)

        ax_top.set_ylabel("Norm. avg uncertainty", fontsize=9)
        ax_top.set_title(model_name, fontweight="bold", fontsize=11)
        ax_top.grid(True, alpha=0.3)

        if corr_col in df.columns:
            y_corr = df[corr_col].values
            ax_bot.plot(
                percentages,
                y_corr,
                "D-",
                markersize=6,
                linewidth=2,
                color="purple",
                label="Corr(EU, AU)",
            )
            _annotate_series(ax_bot, percentages, y_corr, fmt="{:.2f}", dy_scale=0.03)
            ax_bot.set_ylim(-1.05, 1.05)
        else:
            ax_bot.text(
                0.5,
                0.5,
                "No correlation column",
                ha="center",
                va="center",
                transform=ax_bot.transAxes,
                fontsize=9,
            )

        ax_bot.set_xlabel("Training data percentage (%)", fontsize=9)
        ax_bot.set_ylabel("Corr(EU, AU)", fontsize=9)
        ax_bot.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax_bot.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Norm. avg uncertainty", fontsize=9)
    axes[1, 0].set_ylabel("Corr(EU, AU)", fontsize=9)

    func_label = FUNC_NAME_BY_TYPE[func_type]
    measure_label = "Variance" if measure == "variance" else "Entropy (normalized)"
    title = f"Sample size — {func_label} ({noise_type}) — {measure_label}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ax in (axes[0, 0], axes[1, 0]):
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")

    if uncertainty_ylim is not None:
        for c in range(4):
            axes[0, c].set_ylim(*uncertainty_ylim)

    return fig


def create_sample_size_summary_panel_au_eu_only(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    measure: MeasureType,
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """
    1×4 panel: norm. avg AU/EU (and Total if available) vs. training percentage; one column per model.
    Companion to create_sample_size_summary_panel_corr_only for appendix figures.

    Optional ``uncertainty_ylim`` (e.g. ``(0, 1.5)``) fixes the y-axis on every subplot for comparison.
    """
    model_order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for col, model_name in enumerate(model_order):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        percentages = df["Percentage"].values
        au_col, eu_col, tot_col = get_uncertainty_columns(df, measure)
        y_au = df[au_col].values
        y_eu = df[eu_col].values

        ax.plot(percentages, y_au, "o-", markersize=6, linewidth=2, color="green", label="AU")
        ax.plot(percentages, y_eu, "s-", markersize=6, linewidth=2, color="orange", label="EU")
        if tot_col is not None:
            y_tot = df[tot_col].values
            ax.plot(percentages, y_tot, "^-", markersize=6, linewidth=2, color="blue", label="Total")

        def _annotate_series(ax_, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03):
            if len(xs) == 0:
                return
            y_min, y_max = np.min(ys), np.max(ys)
            span = max(y_max - y_min, 1e-6)
            for x, y in zip(xs, ys):
                dy = dy_scale * span
                ax_.text(x + 0.8, y + dy, fmt.format(y), fontsize=9, alpha=0.9, ha="left", va="bottom")

        _annotate_series(ax, percentages, y_au)
        _annotate_series(ax, percentages, y_eu)

        ax.set_ylabel("Norm. avg uncertainty", fontsize=9)
        ax.set_title(model_name, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training data percentage (%)", fontsize=9)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Norm. avg uncertainty", fontsize=9)
    func_label = FUNC_NAME_BY_TYPE[func_type]
    measure_label = "Variance" if measure == "variance" else "Entropy (normalized)"
    title = f"Sample size — {func_label} ({noise_type}) — {measure_label} (AU / EU)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if uncertainty_ylim is not None:
        for i in range(4):
            axes[i].set_ylim(*uncertainty_ylim)
    return fig


def create_sample_size_summary_panel_corr_only(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    measure: MeasureType,
):
    """1×4 panel: Corr(EU, AU) vs. training percentage; one column per model."""
    model_order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.2), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    corr_col = "Correlation_Epi_Ale"

    for col, model_name in enumerate(model_order):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        percentages = df["Percentage"].values

        def _annotate_series(ax_, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03):
            if len(xs) == 0:
                return
            y_min, y_max = np.min(ys), np.max(ys)
            span = max(y_max - y_min, 1e-6)
            for x, y in zip(xs, ys):
                dy = dy_scale * span
                ax_.text(x + 0.8, y + dy, fmt.format(y), fontsize=9, alpha=0.9, ha="left", va="bottom")

        if corr_col in df.columns:
            y_corr = df[corr_col].values
            ax.plot(percentages, y_corr, "D-", markersize=6, linewidth=2, color="purple", label="Corr(EU, AU)")
            _annotate_series(ax, percentages, y_corr, fmt="{:.2f}", dy_scale=0.03)
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.text(0.5, 0.5, "No correlation column", ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_xlabel("Training data percentage (%)", fontsize=9)
        ax.set_ylabel("Corr(EU, AU)", fontsize=9)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(model_name, fontweight="bold", fontsize=11)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Corr(EU, AU)", fontsize=9)
    func_label = FUNC_NAME_BY_TYPE[func_type]
    measure_label = "Variance" if measure == "variance" else "Entropy (normalized)"
    title = f"Sample size — {func_label} ({noise_type}) — {measure_label} (correlation)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def create_noise_level_summary_panel(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    measure: MeasureType,
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """
    4×2 panel: models as columns; row 1 = norm. avg AU/EU vs. Tau; row 2 = correlation.

    If ``uncertainty_ylim`` is set (e.g. ``(0, 1.5)``), row 1 uses that range on every column.
    """
    model_order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)

    for col, model_name in enumerate(model_order):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        df = stats_by_model.get(model_name)
        if df is None or df.empty:
            ax_top.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax_top.transAxes,
                fontsize=11,
            )
            ax_bot.axis("off")
            ax_top.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        taus = df["Tau"].values

        au_col, eu_col, tot_col = get_uncertainty_columns(df, measure)
        corr_col = "Correlation_Epi_Ale"

        y_au = df[au_col].values
        y_eu = df[eu_col].values

        ax_top.plot(
            taus,
            y_au,
            "o-",
            markersize=6,
            linewidth=2,
            color="green",
            label="AU",
        )
        ax_top.plot(
            taus,
            y_eu,
            "s-",
            markersize=6,
            linewidth=2,
            color="orange",
            label="EU",
        )

        if tot_col is not None:
            y_tot = df[tot_col].values
            ax_top.plot(
                taus,
                y_tot,
                "^-",
                markersize=6,
                linewidth=2,
                color="blue",
                label="Total",
            )

        def _annotate_series(ax, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03):
            if len(xs) == 0:
                return
            y_min, y_max = np.min(ys), np.max(ys)
            span = max(y_max - y_min, 1e-6)
            for x, y in zip(xs, ys):
                dy = dy_scale * span
                ax.text(
                    x + 0.02 * (xs[-1] - xs[0] if xs[-1] != xs[0] else 1.0),
                    y + dy,
                    fmt.format(y),
                    fontsize=9,
                    alpha=0.9,
                    ha="left",
                    va="bottom",
                )

        _annotate_series(ax_top, taus, y_au)
        _annotate_series(ax_top, taus, y_eu)

        ax_top.set_ylabel("Norm. avg uncertainty", fontsize=9)
        ax_top.set_title(model_name, fontweight="bold", fontsize=11)
        ax_top.grid(True, alpha=0.3)

        if corr_col in df.columns:
            y_corr = df[corr_col].values
            ax_bot.plot(
                taus,
                y_corr,
                "D-",
                markersize=6,
                linewidth=2,
                color="purple",
                label="Corr(EU, AU)",
            )
            _annotate_series(ax_bot, taus, y_corr, fmt="{:.2f}", dy_scale=0.03)
            ax_bot.set_ylim(-1.05, 1.05)
        else:
            ax_bot.text(
                0.5,
                0.5,
                "No correlation column",
                ha="center",
                va="center",
                transform=ax_bot.transAxes,
                fontsize=9,
            )

        ax_bot.set_xlabel("Noise level (τ)", fontsize=9)
        ax_bot.set_ylabel("Corr(EU, AU)", fontsize=9)
        ax_bot.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax_bot.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Norm. avg uncertainty", fontsize=9)
    axes[1, 0].set_ylabel("Corr(EU, AU)", fontsize=9)

    func_label = FUNC_NAME_BY_TYPE[func_type]
    measure_label = "Variance" if measure == "variance" else "Entropy (normalized)"
    title = f"Noise level — {func_label} ({noise_type}) — {measure_label}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ax in (axes[0, 0], axes[1, 0]):
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")

    if uncertainty_ylim is not None:
        for c in range(4):
            axes[0, c].set_ylim(*uncertainty_ylim)

    return fig


def create_noise_level_summary_panel_au_eu_only(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    measure: MeasureType,
    uncertainty_ylim: Optional[Tuple[float, float]] = None,
):
    """1×4: norm. AU/EU/Total vs. τ; one column per model. Optional ``uncertainty_ylim`` e.g. ``(0, 1.5)``."""
    model_order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for col, model_name in enumerate(model_order):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        taus = df["Tau"].values
        au_col, eu_col, tot_col = get_uncertainty_columns(df, measure)
        y_au = df[au_col].values
        y_eu = df[eu_col].values

        ax.plot(taus, y_au, "o-", markersize=6, linewidth=2, color="green", label="AU")
        ax.plot(taus, y_eu, "s-", markersize=6, linewidth=2, color="orange", label="EU")
        if tot_col is not None:
            y_tot = df[tot_col].values
            ax.plot(taus, y_tot, "^-", markersize=6, linewidth=2, color="blue", label="Total")

        def _annotate_series(ax_, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03):
            if len(xs) == 0:
                return
            y_min, y_max = np.min(ys), np.max(ys)
            span = max(y_max - y_min, 1e-6)
            for x, y in zip(xs, ys):
                dy = dy_scale * span
                ax_.text(
                    x + 0.02 * (xs[-1] - xs[0] if xs[-1] != xs[0] else 1.0),
                    y + dy,
                    fmt.format(y),
                    fontsize=9,
                    alpha=0.9,
                    ha="left",
                    va="bottom",
                )

        _annotate_series(ax, taus, y_au)
        _annotate_series(ax, taus, y_eu)

        ax.set_ylabel("Norm. avg uncertainty", fontsize=9)
        ax.set_title(model_name, fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Noise level (τ)", fontsize=9)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Norm. avg uncertainty", fontsize=9)
    func_label = FUNC_NAME_BY_TYPE[func_type]
    measure_label = "Variance" if measure == "variance" else "Entropy (normalized)"
    title = f"Noise level — {func_label} ({noise_type}) — {measure_label} (AU / EU)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if uncertainty_ylim is not None:
        for i in range(4):
            axes[i].set_ylim(*uncertainty_ylim)
    return fig


def create_noise_level_summary_panel_corr_only(
    stats_by_model: Dict[str, pd.DataFrame],
    func_type: FuncType,
    noise_type: NoiseType,
    measure: MeasureType,
):
    """1×4: Corr(EU, AU) vs. τ; one column per model."""
    model_order = ["Deep Ensemble", "MC Dropout", "BNN", "BAMLSS"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.2), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    corr_col = "Correlation_Epi_Ale"

    for col, model_name in enumerate(model_order):
        ax = axes[col]
        df = stats_by_model.get(model_name)
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(model_name, fontweight="bold", fontsize=11)
            continue

        taus = df["Tau"].values

        def _annotate_series(ax_, xs, ys, fmt: str = "{:.2f}", dy_scale: float = 0.03):
            if len(xs) == 0:
                return
            y_min, y_max = np.min(ys), np.max(ys)
            span = max(y_max - y_min, 1e-6)
            for x, y in zip(xs, ys):
                dy = dy_scale * span
                ax_.text(
                    x + 0.02 * (xs[-1] - xs[0] if xs[-1] != xs[0] else 1.0),
                    y + dy,
                    fmt.format(y),
                    fontsize=9,
                    alpha=0.9,
                    ha="left",
                    va="bottom",
                )

        if corr_col in df.columns:
            y_corr = df[corr_col].values
            ax.plot(taus, y_corr, "D-", markersize=6, linewidth=2, color="purple", label="Corr(EU, AU)")
            _annotate_series(ax, taus, y_corr, fmt="{:.2f}", dy_scale=0.03)
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.text(0.5, 0.5, "No correlation column", ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_xlabel("Noise level (τ)", fontsize=9)
        ax.set_ylabel("Corr(EU, AU)", fontsize=9)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(model_name, fontweight="bold", fontsize=11)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("Corr(EU, AU)", fontsize=9)
    func_label = FUNC_NAME_BY_TYPE[func_type]
    measure_label = "Variance" if measure == "variance" else "Entropy (normalized)"
    title = f"Noise level — {func_label} ({noise_type}) — {measure_label} (correlation)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
