"""Smoke tests for k-NN entropy recomputation helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from utils.knn_entropy_regression import (
    build_ood_mask,
    collect_raw_npz_files,
    compute_numerical_grid_result,
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


def test_compute_numerical_grid_result_smoke(tmp_path):
    """Tiny npz: numerical mixture entropy returns finite vectors."""
    rng = np.random.default_rng(0)
    n_grid = 8
    s_mem = 3
    npz_path = tmp_path / "tiny_BAMLSS_raw_outputs.npz"
    np.savez(
        npz_path,
        mu_samples=rng.standard_normal((s_mem, n_grid)).astype(np.float64),
        sigma2_samples=(np.abs(rng.standard_normal((s_mem, n_grid))) + 0.01).astype(np.float64),
        x_grid=np.linspace(-1, 1, n_grid).reshape(-1, 1),
        y_grid_clean=rng.standard_normal((n_grid, 1)),
        model_name=np.array(["BAMLSS"], dtype=object),
    )
    res = compute_numerical_grid_result(
        npz_path, n_samples=32, base_seed=42, ood_ranges=[(0.5, 1.0)], grid_stride=1, grid_chunk_size=4
    )
    assert res.ale_entropy.shape == (n_grid,)
    assert res.epi_entropy.shape == (n_grid,)
    assert res.tot_entropy.shape == (n_grid,)
    assert np.all(np.isfinite(res.ale_entropy))
    assert np.all(np.isfinite(res.epi_entropy))
    assert np.all(np.isfinite(res.tot_entropy))


def _load_recompute_knn_script():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "recompute_entropy_knn_from_npz.py"
    spec = importlib.util.spec_from_file_location("recompute_entropy_knn_from_npz", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_model_tags_filter():
    m = _load_recompute_knn_script()
    assert m.parse_model_tags_filter(None) is None
    assert m.parse_model_tags_filter("") is None
    assert m.parse_model_tags_filter("  ") is None
    assert m.parse_model_tags_filter("MC_Dropout") == {"MC_Dropout"}
    assert m.parse_model_tags_filter("mc_dropout, BNN") == {"MC_Dropout", "BNN"}
    with pytest.raises(ValueError, match="Unknown model"):
        m.parse_model_tags_filter("NotAModel")
