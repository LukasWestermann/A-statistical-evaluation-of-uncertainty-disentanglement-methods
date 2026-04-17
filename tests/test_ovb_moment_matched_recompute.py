import numpy as np

from utils.knn_entropy_regression import compute_moment_matched_ovb_full_training_result


def _save_ovb_like_npz(path, *, include_full: bool):
    rng = np.random.default_rng(0)
    n_train = 6
    n_grid = 5
    n_members = 4

    payload = {
        "mu_samples": rng.standard_normal((n_members, n_grid)).astype(np.float64),
        "sigma2_samples": (np.abs(rng.standard_normal((n_members, n_grid))) + 0.01).astype(np.float64),
        "x_grid": np.linspace(0.0, 1.0, n_grid).reshape(-1, 1).astype(np.float64),
        "y_grid_clean": rng.standard_normal((n_grid, 1)).astype(np.float64),
        "Y": rng.standard_normal((n_train, 1)).astype(np.float64),
    }
    if include_full:
        payload.update(
            {
                "X_full": rng.standard_normal((n_train, 2)).astype(np.float64),
                "mu_samples_full": rng.standard_normal((n_members, n_train)).astype(np.float64),
                "sigma2_samples_full": (
                    np.abs(rng.standard_normal((n_members, n_train))) + 0.01
                ).astype(np.float64),
            }
        )
    np.savez_compressed(path, **payload)


def test_ovb_full_helper_returns_none_without_full_keys(tmp_path):
    npz = tmp_path / "ovb_outputs_MC_Dropout_rho0.50_beta21.00_20260406.npz"
    _save_ovb_like_npz(npz, include_full=False)
    res = compute_moment_matched_ovb_full_training_result(npz, ood_ranges=[(10.0, 15.0)])
    assert res is None


def test_ovb_full_helper_returns_result_with_full_keys(tmp_path):
    npz = tmp_path / "ovb_outputs_BAMLSS_rho0.50_beta21.00_20260406.npz"
    _save_ovb_like_npz(npz, include_full=True)
    res = compute_moment_matched_ovb_full_training_result(npz, ood_ranges=[(10.0, 15.0)])
    assert res is not None
    assert res.mu_samples.shape == res.sigma2_samples.shape
    assert res.mu_samples.shape[1] == res.x.shape[0]
