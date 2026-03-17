"""Gaussian Mixture Model clustering with heteroscedastic EM (Option A).

For probabilistic embeddings, implements the heteroscedastic EM from
Section 5.1: per-sample variance S_tot,i is folded into the cluster
likelihood as Sigma_ik = C_k + S_tot,i, so that high-uncertainty samples
naturally contribute less to cluster assignments and centroid updates.
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from clustering.base import BaseClusterer
from embedders.base import EmbeddingResult


class GMMClusterer(BaseClusterer):
    """GMM clustering with heteroscedastic EM for probabilistic embeddings.

    For deterministic embeddings: standard sklearn GMM on point estimates.
    For probabilistic embeddings: Option A heteroscedastic EM that folds
    per-sample embedding variance into both E-step and M-step.  Each
    cluster k has diagonal covariance c_k, and the per-sample effective
    covariance in the likelihood is diag(c_k + sigma^2_i).  Centroids are
    updated via precision-weighted averaging (Eq. 32).

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components.  Default 10.
    covariance_type : str, optional
        Sklearn covariance type for the deterministic path.  One of
        ``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``.
        Default ``"full"``.  The probabilistic path always uses diagonal.
    confidence_threshold : float, optional
        Minimum responsibility for cluster assignment (probabilistic
        path).  Samples below this get label -1.  Default 0.5.
    min_bin_size : int, optional
        Clusters smaller than this are discarded (labels set to -1).
        Default 1 (no filtering).
    random_state : int or None, optional
        Random seed for reproducibility.  Default None.
    max_iter : int, optional
        Maximum EM iterations.  Default 200.
    n_init : int, optional
        Number of random restarts (best log-likelihood wins).  Default 5.
    min_covar : float, optional
        Floor for diagonal cluster variance entries to prevent
        degeneracy.  Default 1e-6.
    **extra_kwargs
        Accepted for CLI compatibility; silently ignored.
    """

    def __init__(self, n_components: int = 10, covariance_type: str = "full",
                 confidence_threshold: float = 0.5, min_bin_size: int = 1,
                 random_state: int = None, max_iter: int = 200,
                 n_init: int = 5, min_covar: float = 1e-6, **extra_kwargs):
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError(
                f"n_components must be a positive integer, got {n_components}"
            )
        valid_cov_types = {"full", "tied", "diag", "spherical"}
        if covariance_type not in valid_cov_types:
            raise ValueError(
                f"covariance_type must be one of {valid_cov_types}, "
                f"got '{covariance_type}'"
            )
        if not (0 <= confidence_threshold <= 1):
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, "
                f"got {confidence_threshold}"
            )
        if not isinstance(min_bin_size, int) or min_bin_size <= 0:
            raise ValueError(
                f"min_bin_size must be a positive integer, got {min_bin_size}"
            )
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(
                f"max_iter must be a positive integer, got {max_iter}"
            )
        if not isinstance(n_init, int) or n_init <= 0:
            raise ValueError(
                f"n_init must be a positive integer, got {n_init}"
            )
        if min_covar <= 0:
            raise ValueError(
                f"min_covar must be positive, got {min_covar}"
            )

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.confidence_threshold = confidence_threshold
        self.min_bin_size = min_bin_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_covar = min_covar

    def fit_predict(self, embedding_result: EmbeddingResult, **kwargs) -> np.ndarray:
        """Cluster embeddings; uses heteroscedastic EM when variance is available.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding output with ``point_estimate`` and optional ``variance``.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Integer cluster labels of shape ``(N,)``.  ``-1`` = unassigned.
        """
        if not isinstance(embedding_result, EmbeddingResult):
            raise TypeError(
                f"Expected EmbeddingResult, got {type(embedding_result).__name__}"
            )
        if embedding_result.point_estimate.ndim != 2:
            raise ValueError(
                f"point_estimate must be 2D, got shape "
                f"{embedding_result.point_estimate.shape}"
            )

        features = embedding_result.point_estimate.astype(np.float64)

        if embedding_result.is_probabilistic:
            variance = embedding_result.variance.astype(np.float64)
            # Handle full covariance → extract diagonal
            if variance.ndim == 3:
                variance = np.diagonal(variance, axis1=1, axis2=2)
            labels = self._heteroscedastic_em(features, variance)
        else:
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init,
            )
            gmm.fit(features)
            labels = gmm.predict(features)

        # Discard clusters smaller than min_bin_size
        if self.min_bin_size > 1:
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                if label != -1 and count < self.min_bin_size:
                    labels[labels == label] = -1

        return labels

    # ------------------------------------------------------------------
    # Option A: heteroscedastic EM (Section 5.1)
    # ------------------------------------------------------------------

    def _heteroscedastic_em(self, means, variances):
        """Run heteroscedastic EM with multiple restarts.

        Parameters
        ----------
        means : ndarray (N, D)
            Point-estimate embeddings.
        variances : ndarray (N, D)
            Per-sample diagonal variances.

        Returns
        -------
        labels : ndarray (N,)
            Cluster labels; -1 = unassigned.
        """
        N = means.shape[0]
        rng = np.random.RandomState(self.random_state)

        best_ll = -np.inf
        best_resp = None

        for _ in range(self.n_init):
            seed = rng.randint(0, 2**31)
            result = self._em_single(means, variances, seed)
            if result["ll"] > best_ll:
                best_ll = result["ll"]
                best_resp = result["resp"]
                self.means_ = result["m_k"]
                self.covariances_ = result["c_k"]
                self.weights_ = result["pi_k"]
                self.n_iter_ = result["n_iter"]
                self.converged_ = result["converged"]

        # Assign labels with confidence thresholding on raw responsibilities
        labels = np.argmax(best_resp, axis=1)
        max_resp = best_resp[np.arange(N), labels]
        labels[max_resp < self.confidence_threshold] = -1

        return labels

    def _em_single(self, means, variances, seed):
        """Single heteroscedastic EM run.

        Parameters
        ----------
        means : ndarray (N, D)
        variances : ndarray (N, D)
        seed : int
            Random seed for KMeans initialization.

        Returns
        -------
        dict with keys: pi_k, m_k, c_k, resp, ll, n_iter, converged
        """
        N, D = means.shape
        K = self.n_components
        eps = 1e-10

        # --- Initialize with KMeans ---
        km = KMeans(n_clusters=K, random_state=seed, n_init=1)
        km_labels = km.fit_predict(means)

        m_k = km.cluster_centers_.copy()                  # (K, D)
        c_k = np.full((K, D), self.min_covar)             # (K, D)
        pi_k = np.bincount(km_labels, minlength=K).astype(np.float64)
        pi_k = np.maximum(pi_k, 1.0)
        pi_k /= pi_k.sum()

        # Within-cluster variance initialization
        for k in range(K):
            mask = km_labels == k
            if mask.sum() > 1:
                c_k[k] = np.maximum(np.var(means[mask], axis=0), self.min_covar)

        prev_ll = -np.inf
        converged = False
        n_iter = 0
        resp = None

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # --- E-step (Eq. 24, 30) ---
            log_resp = np.empty((N, K))
            for k in range(K):
                sigma_ik = c_k[k] + variances           # (N, D)
                delta = means - m_k[k]                   # (N, D)
                log_resp[:, k] = (
                    np.log(pi_k[k] + eps)
                    - 0.5 * np.sum(np.log(sigma_ik + eps), axis=1)
                    - 0.5 * np.sum(delta * delta / (sigma_ik + eps), axis=1)
                )

            # Normalize via log-sum-exp
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_norm)

            # Log-likelihood
            ll = float(np.sum(log_norm))

            # Convergence check
            if iteration > 0 and abs(ll - prev_ll) < 1e-6 * max(1.0, abs(prev_ll)):
                converged = True
                break
            prev_ll = ll

            # --- M-step ---
            n_k = resp.sum(axis=0)  # (K,)

            # Mixing weights (Eq. 27)
            pi_k = n_k / N
            pi_k = np.maximum(pi_k, eps)
            pi_k /= pi_k.sum()

            for k in range(K):
                nk = n_k[k]
                if nk < eps:
                    continue  # dead component, skip

                r = resp[:, k, np.newaxis]               # (N, 1)
                sigma_ik = c_k[k] + variances            # (N, D)
                precision = 1.0 / (sigma_ik + eps)       # (N, D)

                # Precision-weighted mean (Eq. 32)
                numerator = np.sum(r * precision * means, axis=0)   # (D,)
                denominator = np.sum(r * precision, axis=0)         # (D,)
                m_k[k] = numerator / (denominator + eps)

                # Cluster variance via moment matching
                delta_sq = (means - m_k[k]) ** 2                    # (N, D)
                observed_var = np.sum(r * delta_sq, axis=0) / (nk + eps)  # (D,)
                mean_noise = np.sum(r * variances, axis=0) / (nk + eps)   # (D,)
                c_k[k] = np.maximum(observed_var - mean_noise, self.min_covar)

        return {
            "pi_k": pi_k, "m_k": m_k, "c_k": c_k,
            "resp": resp, "ll": ll,
            "n_iter": n_iter, "converged": converged,
        }
