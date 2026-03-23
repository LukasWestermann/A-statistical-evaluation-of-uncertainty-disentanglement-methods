"""Smoke tests for k-NN entropy recomputation helpers."""

from pathlib import Path

import numpy as np

from utils.knn_entropy_regression import (
    build_ood_mask,
    collect_raw_npz_files,
    ensure_samples_first,
    is_ovb_or_non_raw_path,
)


def test_ensure_samples_first_transposes_ns_layout():
    n_grid = 5
    n_mem = 3
    x_grid = np.linspace(0, 1, n_grid).reshape(-1, 1)
    mu_wrong = np.random.default_rng(0).standard_normal((n_grid, n_mem))
    sig_wrong = np.abs(np.random.default_rng(1).standard_normal((n_grid, n_mem))) + 0.01
    mu, sig = ensure_samples_first(mu_wrong, sig_wrong, x_grid)
    assert mu.shape == (n_mem, n_grid)
    assert sig.shape == (n_mem, n_grid)


def test_is_ovb_excludes_ovb_outputs():
    assert is_ovb_or_non_raw_path(Path("results/x/ovb_outputs_MC_20250101.npz"))
    p = Path("results/ood/outputs/ood/homoscedastic/linear/20260101_BNN_raw_outputs.npz")
    assert not is_ovb_or_non_raw_path(p)


def test_collect_raw_npz_skips_ovb(tmp_path):
    (tmp_path / "good_raw_outputs.npz").write_bytes(b"")
    (tmp_path / "bad_ovb_outputs_x.npz").write_bytes(b"")
    found = collect_raw_npz_files(tmp_path)
    assert len(found) == 1
    assert "good_raw_outputs" in found[0].name


def test_synthetic_npz_layout_and_ood_mask(tmp_path):
    """Load tiny raw_outputs-style npz; check (S,N) convention and OOD mask length."""
    rng = np.random.default_rng(42)
    n_grid = 4
    s_mem = 2
    npz_path = tmp_path / "tiny_BNN_raw_outputs.npz"
    np.savez(
        npz_path,
        mu_samples=rng.standard_normal((s_mem, n_grid)).astype(np.float64),
        sigma2_samples=np.abs(rng.standard_normal((s_mem, n_grid))) + 0.01,
        x_grid=np.linspace(-1, 1, n_grid).reshape(-1, 1),
        y_grid_clean=rng.standard_normal((n_grid, 1)),
        model_name=np.array(["BNN"], dtype=object),
    )
    data = np.load(npz_path, allow_pickle=True)
    mu = np.asarray(data["mu_samples"])
    sig = np.asarray(data["sigma2_samples"])
    xg = np.asarray(data["x_grid"])
    mu, sig = ensure_samples_first(mu, sig, xg)
    assert mu.shape[0] == s_mem
    m = build_ood_mask(xg, [(0.0, 0.0)])
    assert m.shape == (n_grid,)
