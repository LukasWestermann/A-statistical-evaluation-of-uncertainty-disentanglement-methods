"""Pooled-by-beta2 OVB moment-matched entropy dataframe (sample-size-style normalization)."""

import numpy as np

from utils.knn_entropy_regression import OvbMomentMatchedRecord, build_ovb_moment_matched_entropy_dataframe_pooled


def test_pooled_normalization_differs_from_per_file():
    """Two files, same beta2: global min/max from both grids; means differ from per-file norm."""
    # File A: ale in [0, 1] on 3 points
    ale_a = np.array([0.0, 0.5, 1.0])
    epi_a = np.array([0.0, 0.5, 1.0])
    # File B: ale in [2, 3] — extends global range to [0, 3] for ale when pooled
    ale_b = np.array([2.0, 2.5, 3.0])
    epi_b = np.array([2.0, 2.5, 3.0])

    mu = np.zeros(3)
    y = np.zeros(3)

    r1 = OvbMomentMatchedRecord(
        npz_stem="a",
        npz_relpath="a.npz",
        rho=0.1,
        beta2=1.0,
        ale_entropy=ale_a,
        epi_entropy=epi_a,
        mu_pred=mu,
        y_clean_flat=y,
    )
    r2 = OvbMomentMatchedRecord(
        npz_stem="b",
        npz_relpath="b.npz",
        rho=0.5,
        beta2=1.0,
        ale_entropy=ale_b,
        epi_entropy=epi_b,
        mu_pred=mu,
        y_clean_flat=y,
    )

    df = build_ovb_moment_matched_entropy_dataframe_pooled(
        [r1, r2],
        model_name="BAMLSS",
        date="20260101",
        function_name="Linear",
        noise_type="heteroscedastic",
        func_type="linear",
        dropout_p=None,
        mc_samples=None,
        n_nets=None,
    )
    assert len(df) == 2
    row_a = df[df["npz_stem"] == "a"].iloc[0]
    row_b = df[df["npz_stem"] == "b"].iloc[0]

    # Per-file only: file A would normalize ale to [0,1] with mean 0.5
    per_file_mean_a = float(np.mean((ale_a - ale_a.min()) / (ale_a.max() - ale_a.min())))
    assert abs(per_file_mean_a - 0.5) < 1e-9

    # Pooled: global ale min=0, max=3; file A points map to [0, 1/3, 2/3]
    pooled_mean_a = float(np.mean((ale_a - 0.0) / (3.0 - 0.0)))
    assert abs(row_a["Avg_Aleatoric_Entropy_norm"] - pooled_mean_a) < 1e-6
    assert abs(row_a["Avg_Aleatoric_Entropy_norm"] - per_file_mean_a) > 1e-6
    assert abs(
        row_a["Avg_Total_Entropy_norm"]
        - (row_a["Avg_Aleatoric_Entropy_norm"] + row_a["Avg_Epistemic_Entropy_norm"])
    ) < 1e-9


def test_separate_beta2_groups_independent_pools():
    """Different beta2 values never share a min-max pool (each group is its own concat)."""
    r1 = OvbMomentMatchedRecord(
        "a",
        "a.npz",
        0.0,
        1.0,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.zeros(2),
        np.zeros(2),
    )
    r2 = OvbMomentMatchedRecord(
        "b",
        "b.npz",
        0.0,
        2.0,
        np.array([10.0, 11.0]),
        np.array([10.0, 11.0]),
        np.zeros(2),
        np.zeros(2),
    )
    df = build_ovb_moment_matched_entropy_dataframe_pooled(
        [r1, r2],
        "M",
        "d",
        "Linear",
        "h",
        "linear",
        None,
        None,
        None,
    )
    assert len(df) == 2
    a = df[df["npz_stem"] == "a"].iloc[0]
    b = df[df["npz_stem"] == "b"].iloc[0]
    # One file per group: same as per-file min-max, mean 0.5 for [0,1] and [10,11].
    assert abs(a["Avg_Aleatoric_Entropy_norm"] - 0.5) < 1e-9
    assert abs(b["Avg_Aleatoric_Entropy_norm"] - 0.5) < 1e-9
    # If beta2 were ignored and one global pool were used, file "a" would map to ~[0, 1/11], mean ~1/22.
    wrong_pool_mean_a = float(np.mean((np.array([0.0, 1.0]) - 0.0) / 11.0))
    assert abs(a["Avg_Aleatoric_Entropy_norm"] - wrong_pool_mean_a) > 1e-3
