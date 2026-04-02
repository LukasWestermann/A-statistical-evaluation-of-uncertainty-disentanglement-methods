"""Tests for moment-matched entropy split by undersampling density regions."""
import numpy as np

from utils.knn_entropy_regression import KnnGridResult, moment_matched_entropy_density_split_metrics


def test_moment_matched_entropy_density_split_metrics_basic():
    xcol = np.array([[-5.0], [0.0], [6.0], [9.0]], dtype=np.float64)
    n = 4
    sampling_regions = [
        ((-5, 4), 1.0),
        ((4, 8), 0.05),
        ((8, 10), 1.0),
    ]
    res = KnnGridResult(
        mu_samples=np.zeros((3, n)),
        sigma2_samples=np.ones((3, n)),
        x_grid=xcol,
        y_grid_clean=np.zeros((n, 1)),
        mu_pred=np.zeros(n),
        ale_var=np.ones(n),
        epi_var=np.zeros(n),
        ale_entropy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        epi_entropy=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        tot_entropy=np.ones(n),
        ood_mask=np.zeros(n, dtype=bool),
        id_mask=np.ones(n, dtype=bool),
        x=xcol.ravel(),
        y_clean_flat=np.zeros(n),
        boundary_x=[],
        x_train_flat=None,
        y_train_flat=None,
        meta={},
    )
    u, w = moment_matched_entropy_density_split_metrics(res, sampling_regions)
    # Only x=6 lies in the low-density band (4, 8); min-max uses the full grid.
    assert abs(u["Avg_Aleatoric_Entropy_norm"] - 2.0 / 3.0) < 1e-9
    assert abs(u["Avg_Epistemic_Entropy_norm"] - 2.0 / 3.0) < 1e-9
    assert np.isfinite(w["Avg_Aleatoric_Entropy_norm"])
    assert np.isfinite(w["Correlation_Epi_Ale"])
