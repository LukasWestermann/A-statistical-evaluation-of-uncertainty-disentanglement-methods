"""Tests for OVB sweep loaders and panel column mapping."""

from pathlib import Path

import pandas as pd
import pytest

from utils.ovb_sweep_summary_loaders import (
    filter_sort_variance_sweep,
    load_ovb_entropy_concat_for_condition,
    filter_entropy_rows_for_model,
)
from utils.regression_summary_panels import (
    get_ovb_sweep_uncertainty_columns,
    create_ovb_parameter_sweep_panel_au_eu_tu_only,
)


def test_get_ovb_sweep_uncertainty_columns():
    assert get_ovb_sweep_uncertainty_columns("variance") == (
        "mean_ale_var_norm",
        "mean_epi_var_norm",
        "mean_tot_var_norm",
    )
    assert get_ovb_sweep_uncertainty_columns("entropy") == (
        "Avg_Aleatoric_Entropy_norm",
        "Avg_Epistemic_Entropy_norm",
        "Avg_Total_Entropy_norm",
    )


def test_filter_sort_rho_sweep():
    df = pd.DataFrame(
        {
            "rho": [0.8, 0.0, 0.5],
            "beta2": [1.0, 1.0, 1.0],
            "mean_ale_var_norm": [0.3, 0.1, 0.2],
        }
    )
    out = filter_sort_variance_sweep(df, "rho", fixed_beta2=1.0, fixed_rho=0.7)
    assert list(out["rho"].values) == [0.0, 0.5, 0.8]


def test_rho_sweep_dedupes_duplicate_x():
    """Concatenated stats can repeat the same ρ; keep one row per ρ (last wins)."""
    df = pd.DataFrame(
        {
            "rho": [0.5, 0.5, 0.8],
            "beta2": [1.0, 1.0, 1.0],
            "mean_ale_var_norm": [0.1, 0.99, 0.3],
        }
    )
    out = filter_sort_variance_sweep(df, "rho", fixed_beta2=1.0, fixed_rho=0.7)
    assert list(out["rho"].values) == [0.5, 0.8]
    assert float(out.loc[out["rho"] == 0.5, "mean_ale_var_norm"].iloc[0]) == 0.99


def test_filter_sort_beta2_sweep():
    df = pd.DataFrame(
        {
            "rho": [0.7, 0.7, 0.7],
            "beta2": [2.0, 0.0, 1.0],
            "mean_ale_var_norm": [0.3, 0.1, 0.2],
        }
    )
    out = filter_sort_variance_sweep(df, "beta2", fixed_beta2=1.0, fixed_rho=0.7)
    assert list(out["beta2"].values) == [0.0, 1.0, 2.0]


def test_beta2_sweep_dedupes_duplicate_x():
    df = pd.DataFrame(
        {
            "rho": [0.75, 0.75, 0.75],
            "beta2": [1.0, 1.0, 2.0],
            "mean_ale_var_norm": [0.1, 0.88, 0.3],
        }
    )
    out = filter_sort_variance_sweep(df, "beta2", fixed_beta2=1.0, fixed_rho=0.75)
    assert list(out["beta2"].values) == [1.0, 2.0]
    assert float(out.loc[out["beta2"] == 1.0, "mean_ale_var_norm"].iloc[0]) == 0.88


def test_beta2_sweep_fallback_single_rho_when_fixed_rho_mismatch():
    """Typical ovb_beta2_stats: one ρ (e.g. 0.75) while CLI fixed_rho may still be 0.7."""
    df = pd.DataFrame(
        {
            "rho": [0.75, 0.75, 0.75, 0.75],
            "beta2": [0.0, 0.5, 1.0, 2.0],
            "mean_ale_var_norm": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = filter_sort_variance_sweep(df, "beta2", fixed_beta2=1.0, fixed_rho=0.7)
    assert len(out) == 4
    assert list(out["beta2"].values) == [0.0, 0.5, 1.0, 2.0]


def test_filter_sort_empty_when_fixed_mismatch():
    df = pd.DataFrame({"rho": [0.0], "beta2": [2.0], "mean_ale_var_norm": [0.1]})
    out = filter_sort_variance_sweep(df, "rho", fixed_beta2=1.0, fixed_rho=0.7)
    assert out.empty


def test_entropy_concat_and_filter(tmp_path: Path):
    stats_root = tmp_path / "statistics"
    sub = stats_root / "ovb" / "heteroscedastic" / "linear"
    sub.mkdir(parents=True)
    df_mc = pd.DataFrame(
        {
            "model_name": ["MC_Dropout"],
            "rho": [0.5],
            "beta2": [1.0],
            "Avg_Aleatoric_Entropy_norm": [0.1],
            "Avg_Epistemic_Entropy_norm": [0.2],
            "Avg_Total_Entropy_norm": [0.3],
        }
    )
    df_mc.to_excel(sub / "a_MC_Dropout_moment.xlsx", index=False)
    combined = load_ovb_entropy_concat_for_condition(stats_root, "linear", "heteroscedastic")
    assert len(combined) == 1
    one = filter_entropy_rows_for_model(combined, "MC Dropout", "rho", 1.0, 0.7)
    assert len(one) == 1
    assert float(one["rho"].iloc[0]) == 0.5


def test_create_ovb_panel_smoke():
    df = pd.DataFrame(
        {
            "rho": [0.0, 0.5],
            "beta2": [1.0, 1.0],
            "mean_ale_var_norm": [0.1, 0.2],
            "mean_epi_var_norm": [0.15, 0.25],
            "mean_tot_var_norm": [0.25, 0.45],
        }
    )
    stats = {
        "Deep Ensemble": df,
        "MC Dropout": df,
        "BNN": df,
        "BAMLSS": df,
    }
    fig = create_ovb_parameter_sweep_panel_au_eu_tu_only(
        stats,
        "linear",
        "heteroscedastic",
        "variance",
        "rho",
        "(β₂=1.0)",
        uncertainty_ylim=(0.0, 1.1),
    )
    try:
        assert fig is not None
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
