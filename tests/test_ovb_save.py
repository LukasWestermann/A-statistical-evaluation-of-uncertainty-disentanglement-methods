"""Smoke tests for OVB raw output saver."""

import numpy as np

from utils.ovb_experiments import save_ovb_model_outputs


def _toy_ovb_arrays(n_train=5, n_grid=4, n_members=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_train, 1)).astype(np.float64)
    Z = rng.standard_normal(n_train).astype(np.float64)
    Y = rng.standard_normal((n_train, 1)).astype(np.float64)
    x_grid = np.linspace(-1, 1, n_grid).reshape(-1, 1)
    y_grid_clean = rng.standard_normal((n_grid, 1))
    mu_pred = rng.standard_normal((n_grid, 1))
    mu_samples = rng.standard_normal((n_members, n_grid))
    sigma2_samples = np.abs(rng.standard_normal((n_members, n_grid))) + 0.01
    ale_var = np.abs(rng.standard_normal((n_grid, 1))) + 0.01
    epi_var = np.abs(rng.standard_normal((n_grid, 1))) + 0.01
    tot_var = ale_var + epi_var
    ale_entropy = np.abs(rng.standard_normal((n_grid, 1))) + 0.01
    epi_entropy = np.abs(rng.standard_normal((n_grid, 1))) + 0.01
    tot_entropy = ale_entropy + epi_entropy
    return {
        'X': X,
        'Z': Z,
        'Y': Y,
        'x_grid': x_grid,
        'y_grid_clean': y_grid_clean,
        'mu_pred': mu_pred,
        'mu_samples': mu_samples,
        'sigma2_samples': sigma2_samples,
        'ale_var': ale_var,
        'epi_var': epi_var,
        'tot_var': tot_var,
        'ale_entropy': ale_entropy,
        'epi_entropy': epi_entropy,
        'tot_entropy': tot_entropy,
    }


def test_save_ovb_model_outputs_roundtrip_bnn(tmp_path):
    d = _toy_ovb_arrays()
    save_ovb_model_outputs(
        **d,
        rho=0.5,
        beta2=1.0,
        func_type='linear',
        noise_type='gaussian',
        results_dir=tmp_path,
        param_name='rho',
        model_name='BNN',
        entropy_method='analytical',
    )
    npz_files = list(tmp_path.glob('ovb_outputs_BNN_*.npz'))
    assert len(npz_files) == 1
    loaded = np.load(npz_files[0], allow_pickle=True)
    assert loaded['mu_samples'].shape == d['mu_samples'].shape
    assert loaded['sigma2_samples'].shape == d['sigma2_samples'].shape
    assert loaded['model_name'].item() == 'BNN'
    assert loaded['param_name'].item() == 'rho'


def test_save_ovb_bamlss_transposes_ns_layout(tmp_path):
    """BAMLSS (N_grid, S) is stored as (S, N_grid)."""
    d = _toy_ovb_arrays()
    n_grid = d['x_grid'].shape[0]
    n_mem = 3
    d['mu_samples'] = np.random.default_rng(1).standard_normal((n_grid, n_mem))
    d['sigma2_samples'] = np.abs(np.random.default_rng(2).standard_normal((n_grid, n_mem))) + 0.01

    save_ovb_model_outputs(
        **d,
        rho=0.25,
        beta2=0.75,
        func_type='linear',
        noise_type='gaussian',
        results_dir=tmp_path,
        param_name='beta2',
        model_name='BAMLSS',
        entropy_method='numerical',
    )
    npz_files = list(tmp_path.glob('ovb_outputs_BAMLSS_*.npz'))
    assert len(npz_files) == 1
    loaded = np.load(npz_files[0], allow_pickle=True)
    assert loaded['mu_samples'].shape == (n_mem, n_grid)
    assert loaded['sigma2_samples'].shape == (n_mem, n_grid)
