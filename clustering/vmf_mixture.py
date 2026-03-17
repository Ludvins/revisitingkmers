"""Uncertain vMF mixture model for clustering directional embeddings.

Each data point i is described by a vMF prior q_i = vMF(m_i, τ_i) rather
than a point observation.  The model is a finite mixture of K vMF components
fitted by variational EM.

Technical design
----------------
**Model**
  Uncertain embedding: q_i(z) = vMF(z; m_i, τ_i),  m_i ∈ S^{D-1},  τ_i ≥ 0.
  Generative model: p(z) = Σ_k π_k · vMF(z; μ_k, κ_k).

**Objective (variational lower bound)**
  Q(θ) = Σ_i Σ_k r_{ik} · E_{z~q_i}[log π_k · vMF(z; μ_k, κ_k)]

  Key identity: E_{z~vMF(m_i,τ_i)}[z] = A_D(τ_i) · m_i,
  where A_D(κ) = I_{D/2}(κ) / I_{D/2-1}(κ).

**E-step** — EXACT
  log r_{ik} ∝ log π_k + κ_k A_D(τ_i)(μ_k^T m_i) + log C_D(κ_k)

**M-step for π_k** — EXACT
  π_k = N_k / N,  N_k = Σ_i r_{ik}

**M-step for μ_k** — EXACT
  S_k = Σ_i r_{ik} A_D(τ_i) m_i,  μ_k = S_k / ‖S_k‖

**M-step for κ_k** — GENERALIZED-EM (numerical)
  Optimality condition: A_D(κ_k) = r̄_k = (μ_k^T S_k) / N_k.
  Solved numerically: analytic Banerjee-2005 init + Newton–Raphson refinement.

**Variable-K**
  Start from K_max; after each EM step prune clusters with N_k < min_cluster_size.

**Numerical stability**
  All Bessel evaluations via scipy.special.ive (exponentially scaled) so that
  the exp(−κ) factor cancels in ratios → no overflow for large κ.

References
----------
Banerjee, A. et al. (2005). Clustering on the Unit Hypersphere using von
  Mises–Fisher Distributions. JMLR 6:1345–1382.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.special import ive, logsumexp
from scipy.cluster.vq import kmeans2

logger = logging.getLogger(__name__)

_LOG_2PI = np.log(2.0 * np.pi)
_EPS = 1e-10
_KAPPA_EPS = 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# Pure math utilities  (NumPy / SciPy only — no project imports)
# ══════════════════════════════════════════════════════════════════════════════

def vmf_log_normalizer(D: int, kappa: np.ndarray) -> np.ndarray:
    """Log normalisation constant log C_D(κ) of the vMF density.

    The vMF density on S^{D-1} is
        vMF(z; μ, κ) = C_D(κ) · exp(κ · μᵀz),
    where
        log C_D(κ) = (D/2−1) log κ − (D/2) log(2π) − log I_{D/2−1}(κ).

    Evaluated as
        log I_v(κ) = log ive(v, κ) + κ
    to avoid overflow for large κ (ive is the exponentially-scaled variant).

    Parameters
    ----------
    D : int
        Ambient dimension.
    kappa : array-like, shape (...)
        Concentration parameter(s) κ ≥ 0.

    Returns
    -------
    np.ndarray, same shape as *kappa*
        log C_D(κ).
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    v = D / 2.0 - 1.0
    log_Iv = np.log(np.maximum(ive(v, kappa), _EPS)) + kappa
    return v * np.log(np.maximum(kappa, _EPS)) - (D / 2.0) * _LOG_2PI - log_Iv


def A_D(D: int, kappa: np.ndarray) -> np.ndarray:
    """Mean resultant length A_D(κ) = I_{D/2}(κ) / I_{D/2-1}(κ).

    For z ~ vMF(μ, κ):  E[z] = A_D(κ) · μ.

    Computed as a ratio of exponentially-scaled Bessel functions so the
    exp(−κ) factors cancel, avoiding overflow for any κ.

    Properties
    ----------
    A_D(0) = 0  (uniform),  A_D(κ) → 1 as κ → ∞,  strictly increasing.

    Parameters
    ----------
    D : int
        Ambient dimension.
    kappa : array-like
        Concentration(s) κ ≥ 0.

    Returns
    -------
    np.ndarray
        A_D(κ) ∈ [0, 1), same shape as *kappa*.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    eps_mask = kappa < _KAPPA_EPS
    safe_k = np.where(eps_mask, _KAPPA_EPS, kappa)
    ratio = ive(D / 2.0, safe_k) / np.maximum(ive(D / 2.0 - 1.0, safe_k), _EPS)
    return np.clip(np.where(eps_mask, 0.0, ratio), 0.0, 1.0 - _EPS)


def _a_d_prime(D: int, kappa: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Derivative dA_D/dκ using the identity A_D'(κ) = 1−A²−(D−1)/κ·A."""
    return 1.0 - a ** 2 - (D - 1.0) / np.maximum(kappa, _KAPPA_EPS) * a


def _solve_kappa(
    r_bar: np.ndarray,
    D: int,
    newton_steps: int = 10,
    kappa_min: float = _KAPPA_EPS,
    kappa_max: float = 1000.0,
) -> np.ndarray:
    """Solve A_D(κ) = r̄ for κ via analytic initialiser + Newton–Raphson.

    The M-step optimality condition is A_D(κ_k) = r̄_k (exact), but A_D has
    no closed-form inverse.  We solve it numerically (generalized-EM step):

    1. Analytic initialiser (Banerjee et al. 2005):
           κ₀ = r̄ (D − r̄²) / (1 − r̄²).
    2. Newton–Raphson:
           κ ← κ − (A_D(κ) − r̄) / A_D'(κ),  clipped to [kappa_min, kappa_max].

    This update monotonically increases Q (proof: objective is concave in κ,
    Newton step on a concave function gives Q non-decrease).

    Parameters
    ----------
    r_bar : np.ndarray, shape (K,)
        Target mean resultant lengths ∈ (0, 1).
    D : int
    newton_steps : int
    kappa_min, kappa_max : float

    Returns
    -------
    np.ndarray, shape (K,)
    """
    r = np.clip(r_bar, _EPS, 1.0 - _EPS)
    kappa = r * (D - r ** 2) / (1.0 - r ** 2)
    kappa = np.clip(kappa, kappa_min, kappa_max)
    for _ in range(newton_steps):
        a = A_D(D, kappa)
        ap = _a_d_prime(D, kappa, a)
        kappa = kappa - (a - r) / np.where(np.abs(ap) < _EPS, _EPS, ap)
        kappa = np.clip(kappa, kappa_min, kappa_max)
    return kappa


# ══════════════════════════════════════════════════════════════════════════════
# EM steps
# ══════════════════════════════════════════════════════════════════════════════

def _e_step(
    m: np.ndarray,         # (N, D) unit vectors
    a_tau: np.ndarray,     # (N,)   A_D(τ_i)
    log_pi: np.ndarray,    # (K,)   log mixing weights
    mu: np.ndarray,        # (K, D) unit cluster means
    kappa: np.ndarray,     # (K,)   cluster concentrations
    log_C: np.ndarray,     # (K,)   vmf_log_normalizer(D, kappa)
) -> np.ndarray:
    """E-step: compute (N, K) responsibility matrix r_{ik}. (EXACT)

    log r_{ik} ∝ log π_k + κ_k · A_D(τ_i) · (μ_k^T m_i) + log C_D(κ_k)

    Exact because E_{q_i}[κ_k μ_k^T z] = κ_k A_D(τ_i)(μ_k^T m_i) by
    linearity of expectation and E_{vMF(m_i,τ_i)}[z] = A_D(τ_i) m_i.
    """
    # (N, K): κ_k A_D(τ_i)(μ_k^T m_i) + log C_D(κ_k)
    log_lik = kappa[None, :] * a_tau[:, None] * (m @ mu.T) + log_C[None, :]
    log_r = log_pi[None, :] + log_lik
    log_r -= logsumexp(log_r, axis=1, keepdims=True)
    return np.exp(log_r)


def _m_step_pi(r: np.ndarray) -> np.ndarray:
    """Exact M-step: π_k = N_k / N."""
    N_k = r.sum(axis=0)
    return np.maximum(N_k / N_k.sum(), _EPS)


def _m_step_mu(
    r: np.ndarray,       # (N, K)
    m: np.ndarray,       # (N, D)
    a_tau: np.ndarray,   # (N,)
) -> tuple[np.ndarray, np.ndarray]:
    """Exact M-step: μ_k = S_k / ‖S_k‖, returning (mu_new, S).

    S_k = Σ_i r_{ik} · A_D(τ_i) · m_i  (weighted expected-embedding sum).
    Degenerate clusters (‖S_k‖ < ε) are flagged via small norm; caller
    retains the previous μ_k for those.
    """
    # r: (N,K), a_tau[:,None]: (N,1) → product (N,K); .T → (K,N); @m → (K,D)
    S = (r * a_tau[:, None]).T @ m          # (K, D)
    norms = np.linalg.norm(S, axis=1, keepdims=True)   # (K, 1)
    mu_new = S / np.maximum(norms, _EPS)
    return mu_new, S


def _m_step_kappa(
    r: np.ndarray,
    mu: np.ndarray,
    S: np.ndarray,
    D: int,
    newton_steps: int,
    kappa_min: float,
    kappa_max: float,
) -> np.ndarray:
    """Generalized-EM M-step for κ_k.

    Optimality condition: A_D(κ_k) = r̄_k = (μ_k^T S_k) / N_k.
    Solved numerically. Empty clusters (N_k ≈ 0) yield κ → kappa_min.
    """
    N_k = r.sum(axis=0)
    r_bar = np.einsum("kd,kd->k", mu, S) / np.maximum(N_k, _EPS)
    r_bar = np.where(N_k < _EPS, 0.0, np.clip(r_bar, 0.0, 1.0 - _EPS))
    return _solve_kappa(r_bar, D, newton_steps, kappa_min, kappa_max)


def _prune(
    log_pi: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    r: np.ndarray,
    min_count: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove clusters with N_k < min_count and renormalise.

    Returns trimmed (log_pi, mu, kappa, r).
    """
    N_k = r.sum(axis=0)
    keep = N_k >= min_count
    if keep.all():
        return log_pi, mu, kappa, r
    n_pruned = int((~keep).sum())
    logger.debug("Pruning %d cluster(s); %d remain.", n_pruned, int(keep.sum()))
    log_pi = log_pi[keep] - logsumexp(log_pi[keep])
    r = r[:, keep]
    row_sums = r.sum(axis=1, keepdims=True)
    r /= np.maximum(row_sums, _EPS)
    return log_pi, mu[keep], kappa[keep], r


# ══════════════════════════════════════════════════════════════════════════════
# Public class
# ══════════════════════════════════════════════════════════════════════════════

class UncertainVMFMixture:
    """Finite vMF mixture fitted to uncertain directional embeddings via EM.

    Each data point i is described by a vMF prior q_i = vMF(m_i, τ_i).
    The model maximises the variational lower bound Q(θ) over π, μ, κ.
    Active component count is selected automatically by pruning.

    This class accepts (m, tau) arrays directly. For integration with the
    project's :class:`~embedders.base.EmbeddingResult` pipeline, use the
    :class:`VMFMixtureClusterer` wrapper.

    Parameters
    ----------
    K_max : int
        Initial (maximum) number of mixture components.
    min_cluster_size : float
        Clusters with effective count N_k = Σ_i r_{ik} below this are
        pruned after each EM iteration.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the change in expected log-likelihood.
    confidence_threshold : float
        Minimum max_k(r_{ik}) for a point to receive a label in *predict*.
        Below threshold → label -1.
    kappa_min, kappa_max : float
        Clipping bounds for concentration parameters.
    newton_steps : int
        Newton–Raphson iterations per κ update.
    random_state : int or None
        Seed for initialisation.
    """

    def __init__(
        self,
        K_max: int = 10,
        min_cluster_size: float = 5.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        confidence_threshold: float = 0.0,
        kappa_min: float = _KAPPA_EPS,
        kappa_max: float = 1000.0,
        newton_steps: int = 10,
        random_state=None,
    ):
        self.K_max = K_max
        self.min_cluster_size = min_cluster_size
        self.max_iter = max_iter
        self.tol = tol
        self.confidence_threshold = confidence_threshold
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.newton_steps = newton_steps
        self.random_state = random_state

        self.mu_: np.ndarray | None = None
        self.kappa_: np.ndarray | None = None
        self.pi_: np.ndarray | None = None
        self.K_: int | None = None
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, m: np.ndarray, tau: np.ndarray) -> "UncertainVMFMixture":
        """Fit the vMF mixture via variational EM.

        Parameters
        ----------
        m : np.ndarray, shape (N, D)
            Unit mean directions.  L2-normalised internally if needed.
        tau : np.ndarray, shape (N,)
            Concentration τ_i ≥ 0 of each point's vMF prior.
            τ_i = 0 means uniform (no directional information).

        Returns
        -------
        self
        """
        m, tau = _validate(m, tau)
        N, D = m.shape
        a_tau = A_D(D, tau)                        # (N,) — fixed during EM

        log_pi, mu, kappa = self._init_params(m, a_tau, D)

        prev_lb = -np.inf
        r = None
        for it in range(self.max_iter):
            log_C = vmf_log_normalizer(D, kappa)
            r = _e_step(m, a_tau, log_pi, mu, kappa, log_C)

            if self.min_cluster_size > 0 and r.shape[1] > 1:
                log_pi, mu, kappa, r = _prune(
                    log_pi, mu, kappa, r, self.min_cluster_size
                )

            log_pi = np.log(_m_step_pi(r))
            mu_new, S = _m_step_mu(r, m, a_tau)
            degen = np.linalg.norm(S, axis=1) < _EPS
            mu_new[degen] = mu[degen]              # keep previous μ for empty clusters
            mu = mu_new
            kappa = _m_step_kappa(
                r, mu, S, D, self.newton_steps, self.kappa_min, self.kappa_max
            )

            lb = _lower_bound(m, a_tau, log_pi, mu, kappa)
            logger.debug("iter %d | K=%d | lb=%.6f", it + 1, len(kappa), lb)
            if abs(lb - prev_lb) < self.tol:
                logger.debug("Converged at iteration %d.", it + 1)
                break
            prev_lb = lb

        self.mu_ = mu
        self.kappa_ = kappa
        self.pi_ = np.exp(log_pi)
        self.K_ = int(len(kappa))
        self.n_iter_ = it + 1
        self.lower_bound_ = prev_lb
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Soft cluster assignments r_{ik}, shape (N, K), rows sum to 1."""
        self._check_fitted()
        m, tau = _validate(m, tau)
        D = m.shape[1]
        log_C = vmf_log_normalizer(D, self.kappa_)
        return _e_step(m, A_D(D, tau), np.log(self.pi_), self.mu_, self.kappa_, log_C)

    def predict(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Hard assignment: argmax_k r_{ik}, with -1 for low-confidence points.

        Points where max_k(r_{ik}) < confidence_threshold receive label -1.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        r = self.predict_proba(m, tau)
        labels = r.argmax(axis=1).astype(np.intp)
        if self.confidence_threshold > 0.0:
            labels[r.max(axis=1) < self.confidence_threshold] = -1
        return labels

    def score(self, m: np.ndarray, tau: np.ndarray) -> float:
        """Mean variational lower bound per point (higher is better).

        Approximates (1/N) Σ_i log p(m_i, τ_i | θ).
        """
        self._check_fitted()
        m, tau = _validate(m, tau)
        D = m.shape[1]
        return _lower_bound(m, A_D(D, tau), np.log(self.pi_), self.mu_, self.kappa_)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_params(
        self, m: np.ndarray, a_tau: np.ndarray, D: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Spherical k-means initialisation.

        Runs scipy kmeans2 on unit-norm m to obtain K_max centroids, then
        solves for an initial κ_k from the mean resultant length of each
        assigned group.
        """
        rng = np.random.default_rng(self.random_state)
        K = min(self.K_max, len(m))

        m32 = m.astype(np.float32)
        idx = rng.choice(len(m), size=K, replace=False)
        init_centroids = m32[idx].copy()

        try:
            centroids, labels = kmeans2(m32, init_centroids, iter=20, minit="matrix")
        except Exception:
            labels = rng.integers(0, K, size=len(m))
            centroids = np.zeros((K, D), dtype=np.float32)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    centroids[k] = m32[mask].mean(axis=0)

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        mu = (centroids / np.maximum(norms, _EPS)).astype(np.float64)

        kappa = np.full(K, 1.0)
        for k in range(K):
            mask = labels == k
            if mask.sum() < 2:
                continue
            S_k = (a_tau[mask, None] * m[mask]).mean(axis=0)
            r_bar_k = float(np.clip(mu[k] @ S_k, _EPS, 1.0 - _EPS))
            kappa[k] = float(
                _solve_kappa(np.array([r_bar_k]), D, 5, self.kappa_min, self.kappa_max)[0]
            )

        counts = np.bincount(labels, minlength=K).astype(np.float64) + _EPS
        pi = counts / counts.sum()
        return np.log(pi), mu, kappa

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.mu_ is None:
            raise RuntimeError("Call fit() before predict/score.")


# ── Module-level helpers shared by UncertainVMFMixture and tests ──────────────

def _validate(m: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(m, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"m must be 2-D, got shape {m.shape}")
    if tau.ndim != 1 or len(tau) != len(m):
        raise ValueError(f"tau must be 1-D with len={len(m)}, got {tau.shape}")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.maximum(norms, _EPS)
    return m, np.maximum(tau, 0.0)


def _lower_bound(
    m: np.ndarray,
    a_tau: np.ndarray,
    log_pi: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
) -> float:
    """Mean variational lower bound Q(θ)/N."""
    D = m.shape[1]
    log_C = vmf_log_normalizer(D, kappa)
    r = _e_step(m, a_tau, log_pi, mu, kappa, log_C)
    ell = (kappa[None, :] * a_tau[:, None] * (m @ mu.T) + log_C[None, :])
    return float((r * (log_pi[None, :] + ell)).sum(axis=1).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Registry wrapper — integrates with clustering.base / evaluate_clustering.py
# ══════════════════════════════════════════════════════════════════════════════

class VMFMixtureClusterer:
    """BaseClusterer wrapper around UncertainVMFMixture.

    Accepts an :class:`~embedders.base.EmbeddingResult` from a PCL model
    (``distribution="vmf"``, ``kappa`` is not None) and returns integer labels.

    Registered in the clustering registry as ``"vmf_mixture"`` so it can be
    used via ``--cluster_algos vmf_mixture`` in the evaluation scripts.

    Parameters
    ----------
    K_max : int
        Maximum number of mixture components.
    min_cluster_size : float
        Prune threshold on effective cluster count.
    confidence_threshold : float
        Points with max responsibility below this receive label -1.
    **kwargs
        Forwarded to :class:`UncertainVMFMixture`.
    """

    def __init__(
        self,
        K_max: int = 10,
        min_cluster_size: float = 5.0,
        confidence_threshold: float = 0.0,
        random_state=None,
        **kwargs,
    ):
        self._mixture = UncertainVMFMixture(
            K_max=K_max,
            min_cluster_size=min_cluster_size,
            confidence_threshold=confidence_threshold,
            random_state=random_state,
            **kwargs,
        )

    def fit_predict(self, embedding_result, **kwargs) -> np.ndarray:
        """Cluster a vMF EmbeddingResult.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Must have ``kappa`` set (i.e. come from a PCL / vMF embedder).

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Integer cluster labels; -1 = unassigned.

        Raises
        ------
        ValueError
            If ``embedding_result.kappa`` is None.
        """
        if embedding_result.kappa is None:
            raise ValueError(
                "VMFMixtureClusterer requires a vMF EmbeddingResult "
                "(embedding_result.kappa must not be None). "
                "Use a PCL model or set --model_type pcl."
            )
        m = embedding_result.mean
        tau = embedding_result.kappa
        self._mixture.fit(m, tau)
        return self._mixture.predict(m, tau)

    # Expose mixture attributes for inspection
    @property
    def mixture(self) -> UncertainVMFMixture:
        return self._mixture


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic demo — run with:  python clustering/vmf_mixture.py
# ══════════════════════════════════════════════════════════════════════════════

def _sample_vmf_approx(
    mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Approximate vMF sampler: perturb mu with Gaussian noise then normalise.

    Valid for moderate to large kappa (kappa ≥ 5).  Not used in EM; only
    for generating synthetic test data.
    """
    D = len(mu)
    noise = rng.standard_normal((n, D)) / (kappa ** 0.5)
    samples = mu[None, :] + noise
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    return samples / np.maximum(norms, _EPS)


def _demo() -> None:
    """Synthetic experiment: 3 vMF clusters on S^2, K_max=10, show pruning."""
    import warnings
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(42)
    D = 3
    N_per_cluster = 200

    # Three well-separated cluster centres on S^2
    true_mu = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    cluster_kappa = [40.0, 50.0, 35.0]

    m_list, tau_list, true_labels = [], [], []
    for k, (mu_k, kappa_k) in enumerate(zip(true_mu, cluster_kappa)):
        samples = _sample_vmf_approx(mu_k, kappa_k, N_per_cluster, rng)
        m_list.append(samples)
        # Each point has its own tau drawn from a range around the cluster kappa
        tau_i = rng.uniform(0.5 * kappa_k, 1.5 * kappa_k, N_per_cluster)
        tau_list.append(tau_i)
        true_labels.extend([k] * N_per_cluster)

    m = np.vstack(m_list)
    tau = np.concatenate(tau_list)
    true_labels = np.array(true_labels)

    print(f"Data: N={len(m)}, D={D}, true K=3")
    print(f"Fitting UncertainVMFMixture with K_max=10, min_cluster_size=10 ...")

    model = UncertainVMFMixture(
        K_max=10, min_cluster_size=10, max_iter=200, tol=1e-5, random_state=42
    )
    model.fit(m, tau)

    print(f"Fitted K = {model.K_}  (started from K_max=10, pruned to {model.K_})")
    print(f"Converged in {model.n_iter_} iterations, lower bound = {model.lower_bound_:.4f}")

    labels = model.predict(m, tau)

    # Compute adjusted Rand Index manually (avoid sklearn import in core module)
    from collections import Counter
    pairs_same_true = pairs_same_pred = pairs_both = 0
    for k in range(3):
        for j in range(3):
            mask_t = true_labels == k
            mask_p = labels == j
            n_t = mask_t.sum()
            n_p = mask_p.sum()
            n_both = (mask_t & mask_p).sum()
            from math import comb
            pairs_same_true += comb(int(n_t), 2)
            pairs_same_pred += comb(int(n_p), 2)
            pairs_both += comb(int(n_both), 2)
    N_total = len(labels)
    from math import comb
    total_pairs = comb(N_total, 2)
    expected = pairs_same_true * pairs_same_pred / total_pairs
    ari = (pairs_both - expected) / (
        0.5 * (pairs_same_true + pairs_same_pred) - expected + _EPS
    )
    print(f"Adjusted Rand Index ≈ {ari:.3f}  (1.0 = perfect)")

    print("\nFitted cluster parameters:")
    for k in range(model.K_):
        print(f"  μ_{k} = {model.mu_[k].round(3)}, "
              f"κ_{k} = {model.kappa_[k]:.2f}, "
              f"π_{k} = {model.pi_[k]:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _demo()
