"""
Unit tests for utils/analytic_scores.py -- exact expected CRPS/NLL of a Gaussian or
Gaussian-mixture forecast against a known Gaussian truth. Every closed-form / quadrature
result is validated against a large-sample Monte Carlo estimate (draw y ~ N(mu_t, sig_t^2),
average the per-y exact score), mirroring tests/test_mixture_metrics.py's
TestMixtureCRPSClosedForm pattern (closed form validated against an independent
numerical/sampling reference).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import analytic_scores as asc
from utils import mixture_metrics as mm
from utils.metrics import compute_crps_gaussian

N_MC = 2_000_000
ATOL = 1e-3
RTOL = 2e-3  # MC noise scales with the score's own magnitude (e.g. large sig_t, or a
             # badly mismatched sig_f in the NLL case) -- combine with ATOL rather than
             # using a razor-tight absolute-only bound.


def _weighted_mixture_crps_at_y(w, mu_k, sig_k, y):
    """Independent reference: CRPS(F, y) for F = sum_k w_k N(mu_k, sig_k^2), via the same
    energy-distance identity as utils.mixture_metrics.mixture_crps but generalized to
    arbitrary (non-uniform) weights -- used only to build the MC reference below, not
    imported from utils.analytic_scores itself."""
    w = np.asarray(w, dtype=float)
    mu_k = np.asarray(mu_k, dtype=float)
    sig_k = np.asarray(sig_k, dtype=float)
    term1 = np.sum(w * mm.abs_moment_gaussian(mu_k - y, sig_k))
    pair_mu = mu_k[:, None] - mu_k[None, :]
    pair_sigma = np.sqrt(sig_k[:, None] ** 2 + sig_k[None, :] ** 2)
    pair_w = w[:, None] * w[None, :]
    term2 = 0.5 * np.sum(pair_w * mm.abs_moment_gaussian(pair_mu, pair_sigma))
    return term1 - term2


class TestAbsMomentGaussianReuse:
    def test_analytic_scores_reuses_mixture_metrics_abs_moment(self):
        assert asc._A is mm.abs_moment_gaussian
        assert mm.abs_moment_gaussian is mm._abs_moment_gaussian


class TestExpectedCRPSGaussianVsGaussian:
    @pytest.mark.parametrize("mu_f,sig_f,mu_t,sig_t,seed", [
        (0.0, 1.0, 0.0, 1.0, 1),      # forecast == truth
        (2.0, 1.0, 0.0, 1.0, 2),      # mu_f != mu_t
        (0.0, 0.2, 0.0, 3.0, 3),      # sig_f << sig_t
        (0.0, 3.0, 0.0, 0.2, 4),      # sig_f >> sig_t
    ])
    def test_matches_monte_carlo(self, mu_f, sig_f, mu_t, sig_t, seed):
        analytic = float(asc.expected_crps_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t))
        rng = np.random.default_rng(seed)
        y = rng.normal(mu_t, sig_t, size=N_MC)
        mc = float(compute_crps_gaussian(y, np.full(N_MC, mu_f), np.full(N_MC, sig_f ** 2)))
        assert np.isclose(analytic, mc, atol=ATOL, rtol=RTOL)

    def test_degenerate_forecast_equals_truth(self):
        for mu, s in [(0.0, 1.0), (-3.0, 2.5), (5.0, 0.1)]:
            got = asc.expected_crps_gaussian_vs_gaussian(mu, s, mu, s)
            assert np.isclose(got, s / np.sqrt(np.pi), rtol=1e-10)

    def test_zero_truth_variance_recovers_standard_crps(self):
        mu_f, sig_f, mu_t = 1.5, 2.0, 0.7
        analytic = float(asc.expected_crps_gaussian_vs_gaussian(mu_f, sig_f, mu_t, 0.0))
        standard = float(compute_crps_gaussian([mu_t], [mu_f], [sig_f ** 2]))
        assert np.isclose(analytic, standard, rtol=1e-6)


class TestExpectedCRPSMixtureVsGaussian:
    def test_matches_monte_carlo_unequal_weights(self):
        w = np.array([0.5, 0.3, 0.2])
        mu_k = np.array([-2.0, 0.5, 3.0])
        sig_k = np.array([0.8, 1.5, 0.5])
        mu_t, sig_t = 0.3, 1.2

        analytic = float(asc.expected_crps_mixture_vs_gaussian(
            w, mu_k[:, None], sig_k[:, None], np.array([mu_t]), np.array([sig_t]))[0])

        rng = np.random.default_rng(11)
        n = 5000  # per-draw reference formula is O(M^2) in pure Python, kept small
        y = rng.normal(mu_t, sig_t, size=n)
        mc = np.mean([_weighted_mixture_crps_at_y(w, mu_k, sig_k, yi) for yi in y])
        # Looser atol than the Gaussian-vs-Gaussian test, to account for the smaller MC sample.
        assert np.isclose(analytic, mc, atol=5e-3)

    def test_uniform_weights_default_matches_explicit(self):
        mu_k = np.array([-1.0, 0.5, 2.0])[:, None]
        sig_k = np.array([1.0, 0.8, 1.2])[:, None]
        mu_t, sig_t = np.array([0.0]), np.array([1.0])
        with_none = asc.expected_crps_mixture_vs_gaussian(None, mu_k, sig_k, mu_t, sig_t)
        with_explicit = asc.expected_crps_mixture_vs_gaussian(
            np.full(3, 1 / 3), mu_k, sig_k, mu_t, sig_t)
        np.testing.assert_allclose(with_none, with_explicit, rtol=1e-10)


class TestExpectedNLLGaussianVsGaussian:
    @pytest.mark.parametrize("mu_f,sig_f,mu_t,sig_t,seed", [
        (0.0, 1.0, 0.0, 1.0, 21),
        (2.0, 1.0, 0.0, 1.0, 22),
        (0.0, 0.3, 0.0, 2.0, 23),
        (0.0, 2.0, 0.0, 0.3, 24),
    ])
    def test_matches_monte_carlo(self, mu_f, sig_f, mu_t, sig_t, seed):
        analytic = float(asc.expected_nll_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t))
        rng = np.random.default_rng(seed)
        y = rng.normal(mu_t, sig_t, size=N_MC)
        nll_per_point = 0.5 * np.log(2 * np.pi * sig_f ** 2) + (y - mu_f) ** 2 / (2 * sig_f ** 2)
        mc = float(np.mean(nll_per_point))
        assert np.isclose(analytic, mc, atol=ATOL, rtol=RTOL)


class TestExpectedNLLMixtureQuadrature:
    def test_matches_monte_carlo(self):
        mu_k = np.array([-1.5, 0.5, 2.5])[:, None]
        sig_k = np.array([0.7, 1.3, 0.9])[:, None]
        mu_t, sig_t = np.array([0.2]), np.array([1.0])

        analytic = float(asc.expected_nll_mixture_vs_gaussian(None, mu_k, sig_k, mu_t, sig_t)[0])

        rng = np.random.default_rng(31)
        y = rng.normal(mu_t[0], sig_t[0], size=N_MC)
        mu_tiled = np.tile(mu_k, (1, N_MC))
        sig2_tiled = np.tile(sig_k ** 2, (1, N_MC))
        log_f = mm.mixture_log_pdf(y, mu_tiled, sig2_tiled)
        mc = float(np.mean(-log_f))
        assert np.isclose(analytic, mc, atol=ATOL)

    def test_quadrature_converged_at_64_nodes(self):
        mu_k = np.array([-1.5, 0.5, 2.5])[:, None]
        sig_k = np.array([0.7, 1.3, 0.9])[:, None]
        mu_t, sig_t = np.array([0.2]), np.array([1.0])
        v16 = asc.expected_nll_mixture_vs_gaussian(None, mu_k, sig_k, mu_t, sig_t, n_quad=16)
        v64 = asc.expected_nll_mixture_vs_gaussian(None, mu_k, sig_k, mu_t, sig_t, n_quad=64)
        # 16 nodes already agree with 64 nodes to within ~2e-4 here -- confirms convergence
        # (not bitwise equality, which would require an unreasonably tight bound).
        np.testing.assert_allclose(v16, v64, atol=1e-3)

    def test_weighted_not_supported(self):
        mu_k = np.array([-1.0, 1.0])[:, None]
        sig_k = np.array([1.0, 1.0])[:, None]
        with pytest.raises(NotImplementedError):
            asc.expected_nll_mixture_vs_gaussian(np.array([0.3, 0.7]), mu_k, sig_k,
                                                   np.array([0.0]), np.array([1.0]))


class TestOracleFloors:
    def test_oracle_crps_matches_forecast_equals_truth_degenerate_case(self):
        for sig_t in [0.1, 1.0, 3.0]:
            direct = asc.oracle_crps(sig_t)
            via_general = asc.expected_crps_gaussian_vs_gaussian(0.0, sig_t, 0.0, sig_t)
            assert np.isclose(direct, via_general, rtol=1e-10)

    def test_oracle_nll_matches_forecast_equals_truth_degenerate_case(self):
        for sig_t in [0.1, 1.0, 3.0]:
            direct = asc.oracle_nll(sig_t)
            via_general = asc.expected_nll_gaussian_vs_gaussian(0.0, sig_t, 0.0, sig_t)
            assert np.isclose(direct, via_general, rtol=1e-10)

    def test_oracle_nll_can_be_negative(self):
        assert asc.oracle_nll(0.05) < 0


class TestScoreDivergences:
    def test_iqd_and_kl_nonnegative_with_equality_iff_forecast_equals_truth(self):
        rng = np.random.default_rng(99)
        for _ in range(200):
            mu_f, mu_t = rng.uniform(-5, 5, size=2)
            sig_f, sig_t = rng.uniform(0.1, 5, size=2)
            crps_e = asc.expected_crps_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t)
            nll_e = asc.expected_nll_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t)
            assert asc.iqd(crps_e, sig_t) >= -1e-9
            assert asc.kl_divergence(nll_e, sig_t) >= -1e-9

        crps_eq = asc.expected_crps_gaussian_vs_gaussian(1.0, 2.0, 1.0, 2.0)
        nll_eq = asc.expected_nll_gaussian_vs_gaussian(1.0, 2.0, 1.0, 2.0)
        assert np.isclose(asc.iqd(crps_eq, 2.0), 0.0, atol=1e-9)
        assert np.isclose(asc.kl_divergence(nll_eq, 2.0), 0.0, atol=1e-9)

    def test_kl_mean_plus_kl_spread_equals_kl_divergence(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            mu_f, mu_t = rng.uniform(-5, 5, size=2)
            sig_f, sig_t = rng.uniform(0.1, 5, size=2)
            nll_e = asc.expected_nll_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t)
            kl = asc.kl_divergence(nll_e, sig_t)
            decomposed = asc.kl_mean(mu_t, sig_f, mu_f) + asc.kl_spread(sig_f, sig_t)
            assert np.isclose(kl, decomposed, rtol=1e-8)
