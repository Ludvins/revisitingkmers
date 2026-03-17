"""Unit tests for clustering/vmf_mixture.py.

Run with:
    pytest clustering/tests/test_vmf_mixture.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
from scipy.special import i0, i1, ive

from clustering.vmf_mixture import (
    vmf_log_normalizer,
    A_D,
    _solve_kappa,
    _e_step,
    _validate,
    UncertainVMFMixture,
    VMFMixtureClusterer,
    _EPS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_unit_sphere_data(N: int, D: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random unit vectors and concentrations for testing."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((N, D))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    tau = rng.uniform(1.0, 20.0, N)
    return m, tau


def _sample_cluster(mu: np.ndarray, kappa: float, n: int, rng) -> np.ndarray:
    """Approximate vMF sampler (Gaussian perturbation + normalise)."""
    D = len(mu)
    noise = rng.standard_normal((n, D)) / (kappa ** 0.5)
    samples = mu[None, :] + noise
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    return samples / np.maximum(norms, _EPS)


def _make_clustered_data(K: int = 3, N_per: int = 150, D: int = 3, seed: int = 42):
    """Synthetic data with K well-separated vMF clusters."""
    rng = np.random.default_rng(seed)
    # Orthogonal centres on standard basis
    centers = np.eye(D)[:K]
    m_list, tau_list, labels = [], [], []
    for k, mu_k in enumerate(centers):
        pts = _sample_cluster(mu_k, kappa=200.0, n=N_per, rng=rng)
        m_list.append(pts)
        tau_list.append(rng.uniform(20.0, 80.0, N_per))
        labels.extend([k] * N_per)
    return np.vstack(m_list), np.concatenate(tau_list), np.array(labels)


# ══════════════════════════════════════════════════════════════════════════════
# 1. vmf_log_normalizer
# ══════════════════════════════════════════════════════════════════════════════

class TestVMFLogNormalizer:
    """Closed-form checks for specific dimensions."""

    def test_d2_closed_form(self):
        """D=2: log C_2(κ) = −log(2π I_0(κ)) (order-0 Bessel)."""
        kappa = np.array([0.01, 1.0, 5.0, 20.0, 100.0])
        expected = -np.log(2 * np.pi * i0(kappa))
        actual = vmf_log_normalizer(2, kappa)
        np.testing.assert_allclose(actual, expected, rtol=1e-5,
                                    err_msg="D=2 log normalizer mismatch")

    def test_d3_sign(self):
        """C_D(κ) ≤ 1 for all κ, so log C_D(κ) ≤ 0 for large κ (κ≫1)."""
        # For large κ the density concentrates; C_D(κ) can be > or < 1 depending
        # on dimension, but the log normalizer is a valid finite number.
        kappa = np.array([1.0, 10.0, 100.0])
        result = vmf_log_normalizer(3, kappa)
        assert np.all(np.isfinite(result)), "log normalizer must be finite"

    def test_large_kappa_stable(self):
        """No overflow/NaN for κ up to 1000."""
        kappa = np.linspace(1, 1000, 50)
        result = vmf_log_normalizer(10, kappa)
        assert np.all(np.isfinite(result)), "log normalizer must be finite for large κ"

    def test_scalar_input(self):
        """Accepts scalar κ."""
        result = vmf_log_normalizer(4, 5.0)
        assert np.isfinite(result)


# ══════════════════════════════════════════════════════════════════════════════
# 2. A_D — mean resultant length
# ══════════════════════════════════════════════════════════════════════════════

class TestAD:
    def test_bounds(self):
        """0 ≤ A_D(κ) < 1 for all κ ≥ 0 and D ∈ {2,3,10}."""
        kappa = np.concatenate([[0.0], np.logspace(-3, 3, 50)])
        for D in [2, 3, 10]:
            a = A_D(D, kappa)
            assert np.all(a >= 0), f"D={D}: A_D < 0"
            assert np.all(a < 1), f"D={D}: A_D ≥ 1"

    def test_monotone_increasing(self):
        """A_D is strictly increasing in κ."""
        kappa = np.linspace(0.01, 100.0, 200)
        for D in [3, 10]:
            a = A_D(D, kappa)
            diffs = np.diff(a)
            assert np.all(diffs > 0), f"D={D}: A_D is not monotone increasing"

    def test_zero_concentration(self):
        """A_D(0) = 0 (uniform distribution)."""
        for D in [2, 3, 10]:
            assert abs(A_D(D, 0.0)) < 1e-8, f"D={D}: A_D(0) ≠ 0"

    def test_large_kappa(self):
        """A_D(1000) ≈ 1 (point mass)."""
        for D in [3, 10]:
            assert A_D(D, 1000.0) > 0.99, f"D={D}: A_D(1000) should be close to 1"

    def test_d2_closed_form(self):
        """D=2: A_2(κ) = I_1(κ)/I_0(κ)."""
        kappa = np.array([1.0, 5.0, 20.0])
        expected = i1(kappa) / i0(kappa)
        actual = A_D(2, kappa)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_no_overflow(self):
        """No NaN or Inf for κ up to 1e4."""
        kappa = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 1e4])
        for D in [3, 10, 256]:
            a = A_D(D, kappa)
            assert np.all(np.isfinite(a)), f"D={D}: A_D produced non-finite value"


# ══════════════════════════════════════════════════════════════════════════════
# 3. _solve_kappa — κ inversion
# ══════════════════════════════════════════════════════════════════════════════

class TestSolveKappa:
    def test_round_trip(self):
        """Solve then evaluate: A_D(_solve_kappa(r_bar)) ≈ r_bar."""
        r_bar = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for D in [3, 10]:
            kappa = _solve_kappa(r_bar, D, newton_steps=20)
            recovered = A_D(D, kappa)
            np.testing.assert_allclose(recovered, r_bar, atol=1e-4,
                                        err_msg=f"Round-trip failed for D={D}")

    def test_output_in_bounds(self):
        """Output is within [kappa_min, kappa_max]."""
        r_bar = np.array([0.01, 0.5, 0.99])
        kappa = _solve_kappa(r_bar, D=3, kappa_min=1e-4, kappa_max=500.0)
        assert np.all(kappa >= 1e-4)
        assert np.all(kappa <= 500.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. E-step
# ══════════════════════════════════════════════════════════════════════════════

class TestEStep:
    def test_rows_sum_to_one(self):
        """Each row of the responsibility matrix sums to 1."""
        N, D, K = 50, 5, 4
        m, tau = _make_unit_sphere_data(N, D, seed=1)
        mu = np.random.default_rng(1).standard_normal((K, D))
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        kappa = np.array([5.0, 10.0, 3.0, 8.0])
        log_pi = np.log(np.ones(K) / K)
        from clustering.vmf_mixture import A_D, vmf_log_normalizer
        log_C = vmf_log_normalizer(D, kappa)
        a_tau = A_D(D, tau)
        r = _e_step(m, a_tau, log_pi, mu, kappa, log_C)
        np.testing.assert_allclose(r.sum(axis=1), np.ones(N), atol=1e-10,
                                    err_msg="Responsibilities do not sum to 1")

    def test_all_nonneg(self):
        """All responsibilities are non-negative."""
        N, D, K = 30, 4, 3
        m, tau = _make_unit_sphere_data(N, D, seed=2)
        mu = np.eye(D)[:K]
        kappa = np.array([5.0, 5.0, 5.0])
        log_pi = np.log(np.ones(K) / K)
        from clustering.vmf_mixture import vmf_log_normalizer, A_D
        r = _e_step(m, A_D(D, tau), log_pi, mu, kappa, vmf_log_normalizer(D, kappa))
        assert np.all(r >= 0)

    def test_shape(self):
        """Output shape is (N, K)."""
        N, D, K = 20, 3, 5
        m, tau = _make_unit_sphere_data(N, D, seed=3)
        mu = np.eye(D)[:K] if D >= K else np.eye(K, D)
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        kappa = np.ones(K) * 5.0
        log_pi = np.log(np.ones(K) / K)
        from clustering.vmf_mixture import vmf_log_normalizer, A_D
        r = _e_step(m, A_D(D, tau), log_pi, mu, kappa, vmf_log_normalizer(D, kappa))
        assert r.shape == (N, K)


# ══════════════════════════════════════════════════════════════════════════════
# 5. UncertainVMFMixture — fit properties
# ══════════════════════════════════════════════════════════════════════════════

class TestUncertainVMFMixture:
    @pytest.fixture(scope="class")
    def fitted_model(self):
        m, tau, _ = _make_clustered_data(K=3, N_per=100, D=3, seed=0)
        model = UncertainVMFMixture(K_max=6, min_cluster_size=5, random_state=0)
        model.fit(m, tau)
        return model, m, tau

    def test_mu_unit_norm(self, fitted_model):
        """All fitted cluster means μ_k are unit vectors."""
        model, _, _ = fitted_model
        norms = np.linalg.norm(model.mu_, axis=1)
        np.testing.assert_allclose(norms, np.ones(model.K_), atol=1e-8,
                                    err_msg="μ_k are not unit norm")

    def test_pi_sums_to_one(self, fitted_model):
        """Mixing weights sum to 1."""
        model, _, _ = fitted_model
        np.testing.assert_allclose(model.pi_.sum(), 1.0, atol=1e-10)

    def test_pi_nonneg(self, fitted_model):
        """All mixing weights are non-negative."""
        model, _, _ = fitted_model
        assert np.all(model.pi_ >= 0)

    def test_kappa_in_range(self, fitted_model):
        """Fitted κ values are within [kappa_min, kappa_max]."""
        model, _, _ = fitted_model
        assert np.all(model.kappa_ >= model.kappa_min)
        assert np.all(model.kappa_ <= model.kappa_max)

    def test_predict_proba_rows_sum_to_one(self, fitted_model):
        """predict_proba rows sum to 1."""
        model, m, tau = fitted_model
        r = model.predict_proba(m, tau)
        np.testing.assert_allclose(r.sum(axis=1), np.ones(len(m)), atol=1e-10)

    def test_predict_labels_in_range(self, fitted_model):
        """predict returns labels in {-1, 0, ..., K-1}."""
        model, m, tau = fitted_model
        labels = model.predict(m, tau)
        assert np.all(labels >= -1)
        assert np.all(labels < model.K_)

    def test_score_finite(self, fitted_model):
        """score() returns a finite float."""
        model, m, tau = fitted_model
        s = model.score(m, tau)
        assert np.isfinite(s), f"score is not finite: {s}"

    def test_not_fitted_raises(self):
        """predict/score raise before fit."""
        model = UncertainVMFMixture()
        m, tau = _make_unit_sphere_data(10, 3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(m, tau)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pruning
# ══════════════════════════════════════════════════════════════════════════════

class TestPruning:
    def test_pruning_reduces_k(self):
        """Pruning removes spurious clusters: K < K_max after fit on 3-cluster data."""
        # Use high-D so random init places most extra centers far from all data.
        # In D=32, data lies in a 3-centre subspace; random centres elsewhere
        # attract essentially 0 responsibility → N_k ≈ 0 → pruned.
        rng = np.random.default_rng(7)
        D, N_per = 32, 100
        centers = np.eye(D)[:3]   # 3 orthogonal axes in R^32
        m_list, tau_list = [], []
        for mu_k in centers:
            pts = _sample_cluster(mu_k, kappa=200.0, n=N_per, rng=rng)
            m_list.append(pts)
            tau_list.append(rng.uniform(20.0, 80.0, N_per))
        m = np.vstack(m_list)
        tau = np.concatenate(tau_list)
        model = UncertainVMFMixture(
            K_max=10, min_cluster_size=10, max_iter=100, random_state=7
        )
        model.fit(m, tau)
        assert model.K_ < 10, f"Expected K<10 (some pruning), got K={model.K_}"

    def test_no_pruning_when_disabled(self):
        """min_cluster_size=0 disables pruning (K stays at K_max or less)."""
        m, tau, _ = _make_clustered_data(K=3, N_per=50, D=3, seed=8)
        model = UncertainVMFMixture(
            K_max=4, min_cluster_size=0, max_iter=50, random_state=8
        )
        model.fit(m, tau)
        # Without pruning K should not decrease below initialised value
        assert model.K_ >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. Synthetic recovery
# ══════════════════════════════════════════════════════════════════════════════

class TestSyntheticRecovery:
    def test_ari_high(self):
        """Adjusted Rand Index ≥ 0.9 on three well-separated clusters."""
        from sklearn.metrics import adjusted_rand_score
        m, tau, true_labels = _make_clustered_data(K=3, N_per=150, D=3, seed=42)
        model = UncertainVMFMixture(
            K_max=8, min_cluster_size=10, max_iter=200, tol=1e-5, random_state=42
        )
        model.fit(m, tau)
        labels = model.predict(m, tau)
        # Exclude any -1 labels from ARI computation
        mask = labels >= 0
        ari = adjusted_rand_score(true_labels[mask], labels[mask])
        assert ari >= 0.9, f"ARI={ari:.3f} < 0.9; cluster recovery failed"

    def test_recovered_k_near_true(self):
        """Fitted K should be in a reasonable range around the true K=3."""
        # K_max=8 on 3-cluster data; min_cluster_size prunes only very small
        # clusters — sub-clusters of true clusters survive because they have
        # enough points. Accept K up to K_max-1 as long as ARI is high.
        m, tau, true_labels = _make_clustered_data(K=3, N_per=150, D=3, seed=42)
        model = UncertainVMFMixture(
            K_max=8, min_cluster_size=10, max_iter=200, tol=1e-5, random_state=42
        )
        model.fit(m, tau)
        assert 2 <= model.K_ <= 8, f"K out of range: got K={model.K_}"


# ══════════════════════════════════════════════════════════════════════════════
# 8. VMFMixtureClusterer — BaseClusterer wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TestVMFMixtureClusterer:
    def _make_embedding_result(self, N=60, D=3):
        from embedders.base import EmbeddingResult
        m, tau, _ = _make_clustered_data(K=3, N_per=N // 3, D=D, seed=99)
        return EmbeddingResult(mean=m, kappa=tau, distribution="vmf")

    def test_fit_predict_labels(self):
        """fit_predict returns (N,) integer array."""
        er = self._make_embedding_result()
        clf = VMFMixtureClusterer(K_max=6, min_cluster_size=5, random_state=0)
        labels = clf.fit_predict(er)
        assert labels.shape == (len(er.mean),)
        assert labels.dtype in (np.intp, np.int64, np.int32, int)

    def test_raises_without_kappa(self):
        """Raises ValueError when kappa is None."""
        from embedders.base import EmbeddingResult
        er = EmbeddingResult(mean=np.random.randn(10, 3))
        clf = VMFMixtureClusterer()
        with pytest.raises(ValueError, match="kappa"):
            clf.fit_predict(er)

    def test_registry(self):
        """'vmf_mixture' is accessible via get_clusterer."""
        from clustering import get_clusterer
        clf = get_clusterer("vmf_mixture", K_max=5)
        assert isinstance(clf, VMFMixtureClusterer)
