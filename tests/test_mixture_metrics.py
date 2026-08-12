"""
Unit tests for the Gaussian-mixture predictive-distribution scoring functions in
utils/mixture_metrics.py.
"""
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import kstest, norm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import mixture_metrics as mm
from utils.entropy_uncertainty import _sample_from_mixture
from utils.metrics import compute_crps_gaussian


class TestDGPTable:
    """The 4 canonical DGPs must match Experiments/*.ipynb's generate_toy_regression
    exactly -- NOT utils.metrics.compute_true_noise_variance, which is wrong for the
    homoscedastic case (see module docstring)."""

    def test_linear_mean(self):
        spec = mm.get_dgp('linear', 'homoscedastic')
        x = np.array([0.0, 5.0, -5.0])
        np.testing.assert_allclose(spec.mean_fn(x), 0.7 * x + 0.5, rtol=1e-6)

    def test_sin_mean(self):
        spec = mm.get_dgp('sin', 'homoscedastic')
        x = np.array([0.0, 5.0, -5.0])
        np.testing.assert_allclose(spec.mean_fn(x), x * np.sin(x) + x, rtol=1e-6)

    def test_homoscedastic_sigma_is_one(self):
        spec = mm.get_dgp('linear', 'homoscedastic')
        x = np.array([0.0, 5.0, -5.0])
        np.testing.assert_allclose(spec.sigma_fn(x), np.ones(3), rtol=1e-6)

    def test_heteroscedastic_sigma(self):
        spec = mm.get_dgp('linear', 'heteroscedastic')
        x = np.array([0.0, 5.0, -5.0])
        np.testing.assert_allclose(spec.sigma_fn(x), np.abs(2.5 * np.sin(0.5 * x + 5)), rtol=1e-6)

    def test_unknown_combination_raises(self):
        with pytest.raises(ValueError):
            mm.get_dgp('cubic', 'homoscedastic')

    def test_dgp_name(self):
        assert mm.dgp_name('linear', 'homoscedastic') == 'linear-homoscedastic'


class TestAbsMomentGaussian:
    def test_zero_sigma_limit(self):
        mu = np.array([-2.0, 0.0, 3.5])
        sigma = np.zeros(3)
        np.testing.assert_allclose(mm._abs_moment_gaussian(mu, sigma), np.abs(mu), rtol=1e-6)

    def test_symmetry_under_sign_flip(self):
        mu = np.array([1.5, -0.3, 4.0])
        sigma = np.array([0.5, 2.0, 1.1])
        np.testing.assert_allclose(mm._abs_moment_gaussian(mu, sigma),
                                    mm._abs_moment_gaussian(-mu, sigma), rtol=1e-6)


class TestMixtureCDF:
    def test_reduces_to_norm_cdf_for_single_component(self):
        mu = np.array([[1.5]])
        sigma2 = np.array([[2.0]])
        y = np.array([2.3])
        got = mm.mixture_cdf(y, mu, sigma2)
        expected = norm.cdf(y, 1.5, np.sqrt(2.0))
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_monotone_nondecreasing(self):
        mu = np.array([[-1.0], [2.0]])
        sigma2 = np.array([[1.0], [0.5]])
        ys = np.linspace(-10, 10, 200)
        vals = np.array([mm.mixture_cdf(np.array([y]), mu, sigma2)[0] for y in ys])
        assert np.all(np.diff(vals) >= -1e-12)

    def test_limits_approach_zero_and_one(self):
        mu = np.array([[-1.0], [2.0]])
        sigma2 = np.array([[1.0], [0.5]])
        assert mm.mixture_cdf(np.array([-1e6]), mu, sigma2)[0] < 1e-6
        assert mm.mixture_cdf(np.array([1e6]), mu, sigma2)[0] > 1 - 1e-6


class TestMixtureQuantile:
    def test_reduces_to_norm_ppf_for_single_component(self):
        mu = np.array([1.5])
        sigma2 = np.array([2.0])
        for p in [0.05, 0.25, 0.5, 0.75, 0.95]:
            got = mm.mixture_quantile(p, mu, sigma2)
            expected = norm.ppf(p, 1.5, np.sqrt(2.0))
            assert np.isclose(got, expected, rtol=1e-6)

    def test_round_trip_multicomponent(self):
        mu = np.array([-3.0, 0.5, 4.0])
        sigma2 = np.array([0.8, 1.2, 0.5])
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            y_p = mm.mixture_quantile(p, mu, sigma2)
            back = mm.mixture_cdf(np.array([y_p]), mu.reshape(-1, 1), sigma2.reshape(-1, 1))[0]
            assert np.isclose(back, p, atol=1e-6)


class TestMixtureCRPSClosedForm:
    """The core requirement: closed-form CRPS (Grimit et al. 2006) verified against
    numerical integration of the CRPS definition CRPS(F,y) = int (F(t) - 1{t>=y})^2 dt."""

    @staticmethod
    def _crps_via_quad(y, mu, sigma2):
        mu = np.asarray(mu, dtype=float)
        sigma2 = np.asarray(sigma2, dtype=float)
        sigma = np.sqrt(sigma2)
        lo = mu.min() - 10 * sigma.max()
        hi = mu.max() + 10 * sigma.max()

        def integrand(t):
            F_t = mm.mixture_cdf(np.array([t]), mu.reshape(-1, 1), sigma2.reshape(-1, 1))[0]
            step = 1.0 if t >= y else 0.0
            return (F_t - step) ** 2

        val, _ = quad(integrand, lo, hi, limit=200)
        return val

    def test_single_component_matches_compute_crps_gaussian(self):
        mu = np.array([[1.0, -1.0, 0.5]])
        sigma2 = np.array([[2.0, 0.5, 1.0]])
        y = np.array([0.2, -0.3, 1.1])
        got = mm.mixture_crps(y, mu, sigma2)
        expected = np.array([compute_crps_gaussian([y[i]], [mu[0, i]], [sigma2[0, i]]) for i in range(3)])
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    @pytest.mark.parametrize("mu,sigma2,y", [
        (np.array([-2.0, 3.0]), np.array([1.0, 1.5 ** 2]), 0.5),
        (np.array([-3.0, 0.0, 4.0]), np.array([0.8 ** 2, 1.2 ** 2, 0.5 ** 2]), 1.0),
        (np.array([-3.0, 0.0, 4.0]), np.array([0.8 ** 2, 1.2 ** 2, 0.5 ** 2]), -5.0),
    ])
    def test_multicomponent_matches_numerical_integration(self, mu, sigma2, y):
        closed_form = mm.mixture_crps(np.array([y]), mu.reshape(-1, 1), sigma2.reshape(-1, 1))[0]
        numerical = self._crps_via_quad(y, mu, sigma2)
        assert np.isclose(closed_form, numerical, atol=1e-3)


class TestMixtureNLL:
    def test_single_component_matches_direct_gaussian_nll(self):
        mu = np.array([[1.0, -2.0]])
        sigma2 = np.array([[2.0, 0.5]])
        y = np.array([0.5, -1.0])
        got = mm.mixture_nll(y, mu, sigma2)
        direct = np.mean(0.5 * np.log(2 * np.pi * sigma2[0]) + (y - mu[0]) ** 2 / (2 * sigma2[0]))
        assert np.isclose(got, direct, rtol=1e-6)

    def test_multicomponent_matches_naive_logsumexp(self):
        mu = np.array([[-1.0, 2.0], [1.0, -1.0], [0.0, 0.5]])
        sigma2 = np.array([[1.0, 0.5], [0.8, 1.2], [2.0, 0.3]])
        y = np.array([0.3, -0.2])
        got = mm.mixture_log_pdf(y, mu, sigma2)
        M = mu.shape[0]
        naive = np.array([
            np.log(np.mean([norm.pdf(y[j], mu[i, j], np.sqrt(sigma2[i, j])) for i in range(M)]))
            for j in range(len(y))
        ])
        np.testing.assert_allclose(got, naive, rtol=1e-6)


class TestPIT:
    def test_values_in_unit_interval(self):
        mu = np.array([[-1.0, 2.0], [1.0, -1.0]])
        sigma2 = np.array([[1.0, 0.5], [0.8, 1.2]])
        y = np.array([-5.0, 5.0])
        pit = mm.compute_pit(y, mu, sigma2)
        assert np.all(pit >= 0.0) and np.all(pit <= 1.0)

    def test_uniform_when_y_drawn_from_same_mixture(self):
        mu = np.array([-1.0, 2.0])
        sigma2 = np.array([1.0, 0.7])
        n = 5000
        mu_tiled = np.tile(mu[:, None], (1, n))
        sigma2_tiled = np.tile(sigma2[:, None], (1, n))
        y = _sample_from_mixture(mu_tiled, sigma2_tiled, n_samples=1, seed=123)[0]
        pit = mm.compute_pit(y, mu_tiled, sigma2_tiled)
        _, pvalue = kstest(pit, 'uniform')
        assert pvalue > 0.01


class TestCoverage:
    def test_nesting_monotonicity(self):
        # Symmetric synthetic quantile bounds around 0: level a's interval is [-a, a].
        quantile_dict = {}
        for a in (0.5, 0.8, 0.9, 0.95):
            lo_level, hi_level = (1 - a) / 2, (1 + a) / 2
            quantile_dict[lo_level] = np.array([-hi_level])
            quantile_dict[hi_level] = np.array([hi_level])
        y = np.array([0.6])
        coverage = mm.empirical_coverage(y, quantile_dict, nominal_levels=(0.5, 0.8, 0.9, 0.95))
        levels = sorted(coverage)
        values = [coverage[lv] for lv in levels]
        assert all(v2 >= v1 - 1e-12 for v1, v2 in zip(values, values[1:]))

    def test_large_sample_coverage_matches_nominal(self):
        mu = np.array([0.0, 0.0])
        sigma2 = np.array([1.0, 1.0])  # both components identical -> reduces to N(0,1)
        n = 5000
        mu_tiled = np.tile(mu[:, None], (1, n))
        sigma2_tiled = np.tile(sigma2[:, None], (1, n))
        y = _sample_from_mixture(mu_tiled, sigma2_tiled, n_samples=1, seed=7)[0]
        levels_needed = sorted({lv for a in (0.5, 0.8, 0.9, 0.95) for lv in ((1 - a) / 2, (1 + a) / 2)})
        qdict = mm.mixture_quantiles_for_levels(mu_tiled, sigma2_tiled, levels_needed)
        coverage = mm.empirical_coverage(y, qdict, nominal_levels=(0.5, 0.8, 0.9, 0.95))
        for a, c in coverage.items():
            assert abs(c - a) < 0.03


class TestSubsampleMembers:
    def test_subset_size_and_uniqueness(self):
        mu_all = np.arange(5 * 4).reshape(5, 4).astype(float)
        sigma2_all = np.ones((5, 4))
        seen = set()
        for r, idx, mu_sub, sigma2_sub in mm.subsample_members(mu_all, sigma2_all, M=3, R=4, seed=1):
            assert mu_sub.shape == (3, 4)
            key = tuple(sorted(idx.tolist()))
            assert key not in seen
            seen.add(key)

    def test_reproducible_given_same_seed(self):
        mu_all = np.arange(6 * 3).reshape(6, 3).astype(float)
        sigma2_all = np.ones((6, 3))
        idx_a = [tuple(idx.tolist()) for _, idx, _, _ in mm.subsample_members(mu_all, sigma2_all, M=2, R=3, seed=42)]
        idx_b = [tuple(idx.tolist()) for _, idx, _, _ in mm.subsample_members(mu_all, sigma2_all, M=2, R=3, seed=42)]
        assert idx_a == idx_b

    def test_m_equals_k_collapses_to_single_replicate(self):
        mu_all = np.arange(4 * 2).reshape(4, 2).astype(float)
        sigma2_all = np.ones((4, 2))
        subsets = list(mm.subsample_members(mu_all, sigma2_all, M=4, R=20, seed=1))
        assert len(subsets) == 1
        assert comb(4, 4) == 1

    def test_m_greater_than_k_raises(self):
        mu_all = np.ones((3, 2))
        sigma2_all = np.ones((3, 2))
        with pytest.raises(ValueError):
            list(mm.subsample_members(mu_all, sigma2_all, M=5, R=2, seed=1))


class TestNormalizeMixtureArrays:
    def test_noop_when_already_oriented(self):
        mu = np.ones((5, 20))
        sigma2 = np.ones((5, 20))
        mu_out, sigma2_out = mm.normalize_mixture_arrays(mu, sigma2, n_expected=20)
        assert mu_out.shape == (5, 20)

    def test_transposes_when_needed(self):
        mu = np.ones((20, 5))  # (N, S) instead of (S, N)
        sigma2 = np.ones((20, 5))
        mu_out, sigma2_out = mm.normalize_mixture_arrays(mu, sigma2, n_expected=20)
        assert mu_out.shape == (5, 20)

    def test_raises_on_unresolvable_shape(self):
        mu = np.ones((3, 4))
        sigma2 = np.ones((3, 4))
        with pytest.raises(ValueError):
            mm.normalize_mixture_arrays(mu, sigma2, n_expected=99)
