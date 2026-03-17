"""Heteroscedastic DPGMM clustering with automatic K discovery.

For probabilistic embeddings, implements MAP-EM with a symmetric Dirichlet
prior on mixing weights and heteroscedastic likelihood (Sigma_ik = C_k + S_i).
Components with negligible weight are pruned during EM, yielding automatic
cluster count selection.  For deterministic embeddings, falls back to sklearn's
BayesianGaussianMixture.
"""

import numpy as np
from scipy.special import logsumexp, digamma
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from clustering.base import BaseClusterer
from embedders.base import EmbeddingResult


class DPGMMClusterer(BaseClusterer):
    """Heteroscedastic DPGMM with MAP-EM and automatic K discovery.

    For probabilistic embeddings: MAP-EM with a symmetric Dirichlet prior
    on mixing weights and heteroscedastic likelihood where per-sample
    variance S_i is folded into the cluster likelihood as
    Sigma_ik = C_k + S_i.  Components whose weight falls below
    ``weight_threshold`` are pruned during EM.

    For deterministic embeddings: sklearn ``BayesianGaussianMixture``.

    After EM convergence, components with high symmetric KL overlap are
    merged to consolidate sub-clusters of the same true cluster.

    Parameters
    ----------
    max_components : int, optional
        Upper bound on the number of mixture components.  Default 50.
    alpha_prior : float or None, optional
        Symmetric Dirichlet concentration.  Values < 1 encourage sparsity
        (unused components are pushed to zero weight).  ``None`` defaults
        to ``1 / max_components`` (matching sklearn).  Default None.
    pca_dim : int or None, optional
        If set, reduce dimensionality with PCA before EM (probabilistic
        path).  Diagonal variances are projected via the diagonal
        approximation ``var_pca = variances @ (W ** 2)``.  Dramatically
        speeds up EM for high-dimensional embeddings (e.g. 256D -> 16D
        retains >99% variance).  ``None`` disables PCA.  Default 16.
    covariance_type : str, optional
        Covariance type for the sklearn deterministic fallback.  One of
        ``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``.
        Default ``"full"``.  The probabilistic path always uses diagonal.
    confidence_threshold : float, optional
        Minimum responsibility for cluster assignment (probabilistic
        path).  Samples below this get label -1.  Default 0.5.
    weight_threshold : float, optional
        Components with mixing weight below this are pruned during EM.
        Default 1e-3.
    merge_threshold : float or None, optional
        Per-dimension symmetric KL divergence threshold for merging
        components after EM.  Components with average per-dimension
        KL (using effective variance = cluster var + avg noise) below
        this are merged.  ``None`` disables merging.  Default 2.0.
    min_bin_size : int, optional
        Clusters smaller than this are discarded (labels set to -1).
        Default 1 (no filtering).
    random_state : int or None, optional
        Random seed for reproducibility.  Default None.
    max_iter : int, optional
        Maximum EM iterations.  Default 300.
    n_init : int, optional
        Number of random restarts (best MAP objective wins).  Default 5.
    min_covar : float, optional
        Floor for diagonal cluster variance entries to prevent
        degeneracy.  Default 1e-6.
    tol : float, optional
        Convergence tolerance on relative MAP objective change.
        Default 1e-6.
    verbose : bool, optional
        Print progress information during fitting.  Default True.
    het_covariance_type : str, optional
        Covariance type for the heteroscedastic EM path.  ``"diag"`` uses
        diagonal cluster covariances (fast, default); ``"full"`` uses full
        D×D matrices in the reduced PCA space (more expressive, ~2× slower).
        Default ``"diag"``.
    kappa_to_variance : bool, optional
        If True and the embedding has ``kappa`` but no ``variance``, convert
        kappa to an isotropic proxy variance ``σ² = 1/κ`` so the het EM path
        activates for PCL-style embeddings.  Default False.
    collect_diagnostics : bool, optional
        If True, store per-iteration convergence data in ``self.diagnostics_``
        after fitting (K_active, MAP objective, n_pruned per iteration) and
        merge sequence in ``self.merge_diagnostics_``.  Default False.
    algorithm : str, optional
        Optimization algorithm for the heteroscedastic path.  ``"map"``
        uses the existing MAP-EM (Dirichlet prior on weights, precision-
        weighted mean update, moment-matching covariance).  ``"vbem"``
        uses a hybrid Variational Bayes EM: VB E-step with digamma weight
        terms and per-sample mean uncertainty (trace correction), VB M-step
        for weights (Dirichlet) and means (Normal posterior), and a MAP
        fixed-point inner loop for cluster covariances (conjugacy is broken
        by heteroscedastic noise).  The ELBO replaces the MAP objective for
        convergence checking and restart selection.  Default ``"map"``.
    beta_0 : float, optional
        Prior precision for the Normal prior on component means in VB-EM.
        Small values (e.g. 1e-3) give a weakly informative prior; larger
        values shrink means toward ``m_0``.  Default 1e-3.
    m_0 : str, optional
        Prior mean for the Normal prior on component means in VB-EM.
        ``"data_mean"`` sets it to the empirical mean of the embeddings
        (recommended); ``"zeros"`` uses the origin.  Default ``"data_mean"``.
    cov_inner_steps : int, optional
        Number of fixed-point iterations for the cluster covariance update
        in VB-EM.  Each step recomputes the posterior mean precision, the
        variational mean, and the covariance given the current covariance
        estimate.  3–5 steps are typically sufficient.  Default 3.
    **extra_kwargs
        Accepted for CLI compatibility; silently ignored.
    """

    def __init__(self, max_components: int = 50, alpha_prior: float = None,
                 pca_dim: int = 16,
                 covariance_type: str = "diag",
                 confidence_threshold: float = 0.5,
                 weight_threshold: float = 1e-3,
                 merge_threshold: float = 2.0,
                 min_bin_size: int = 1,
                 random_state: int = None, max_iter: int = 300,
                 n_init: int = 5, min_covar: float = 1e-6,
                 tol: float = 1e-6, verbose: bool = True,
                 het_covariance_type: str = "diag",
                 kappa_to_variance: bool = False,
                 collect_diagnostics: bool = False,
                 algorithm: str = "map",
                 beta_0: float = 1e-3,
                 m_0: str = "data_mean",
                 cov_inner_steps: int = 3,
                 cov_step_size: float = 0.5,
                 **extra_kwargs):
        if not isinstance(max_components, int) or max_components <= 0:
            raise ValueError(
                f"max_components must be a positive integer, got {max_components}"
            )
        valid_cov = {"full", "tied", "diag", "spherical"}
        if covariance_type not in valid_cov:
            raise ValueError(
                f"covariance_type must be one of {valid_cov}, "
                f"got '{covariance_type}'"
            )
        if het_covariance_type not in {"diag", "full"}:
            raise ValueError(
                f"het_covariance_type must be 'diag' or 'full', "
                f"got '{het_covariance_type}'"
            )
        if not (0 <= confidence_threshold <= 1):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], "
                f"got {confidence_threshold}"
            )
        if not (0 < weight_threshold < 1):
            raise ValueError(
                f"weight_threshold must be in (0, 1), got {weight_threshold}"
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
            raise ValueError(f"min_covar must be positive, got {min_covar}")
        if algorithm not in {"map", "vbem"}:
            raise ValueError(f"algorithm must be 'map' or 'vbem', got '{algorithm}'")
        if beta_0 <= 0:
            raise ValueError(f"beta_0 must be positive, got {beta_0}")
        if m_0 not in {"data_mean", "zeros"}:
            raise ValueError(f"m_0 must be 'data_mean' or 'zeros', got '{m_0}'")
        if not isinstance(cov_inner_steps, int) or cov_inner_steps <= 0:
            raise ValueError(f"cov_inner_steps must be a positive integer, got {cov_inner_steps}")
        if cov_step_size <= 0:
            raise ValueError(f"cov_step_size must be positive, got {cov_step_size}")

        self.max_components = max_components
        self.alpha_prior = alpha_prior if alpha_prior is not None else 1.0 / max_components
        self.pca_dim = pca_dim
        self.covariance_type = covariance_type
        self.het_covariance_type = het_covariance_type
        self.kappa_to_variance = kappa_to_variance
        self.collect_diagnostics = collect_diagnostics
        self.confidence_threshold = confidence_threshold
        self.weight_threshold = weight_threshold
        self.merge_threshold = merge_threshold
        self.min_bin_size = min_bin_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.min_covar = min_covar
        self.tol = tol
        self.verbose = verbose
        self.algorithm = algorithm
        self.beta_0 = beta_0
        self.m_0 = m_0
        self.cov_inner_steps = cov_inner_steps
        self.cov_step_size = cov_step_size
        # Populated after fit_predict when collect_diagnostics=True
        self.diagnostics_ = None
        self.merge_diagnostics_ = None

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def fit_predict(self, embedding_result: EmbeddingResult, **kwargs) -> np.ndarray:
        """Cluster embeddings with automatic K discovery.

        Uses heteroscedastic MAP-EM when variance is available, otherwise
        falls back to sklearn's variational DPGMM.

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
        N, D = features.shape

        # kappa -> isotropic variance conversion (for PCL embeddings)
        if (self.kappa_to_variance
                and embedding_result.kappa is not None
                and embedding_result.variance is None):
            kappa = embedding_result.kappa.astype(np.float64)
            iso_var = (1.0 / (kappa + 1e-8))[:, np.newaxis] * np.ones((1, D))
            from embedders.base import EmbeddingResult as _ER
            embedding_result = _ER(mean=embedding_result.mean, variance=iso_var)
            self._log(f"kappa_to_variance: converted kappa (mean={kappa.mean():.4f}) "
                      f"to isotropic variance (mean={iso_var.mean():.6f})")

        self._log(f"fit_predict: N={N}, D={D}, probabilistic={embedding_result.is_probabilistic}")

        if embedding_result.is_probabilistic:
            variance = embedding_result.variance.astype(np.float64)
            # Handle full covariance -> extract diagonal
            if variance.ndim == 3:
                variance = np.diagonal(variance, axis1=1, axis2=2)
            self._log(f"Variance stats: min={variance.min():.6f}, "
                       f"mean={variance.mean():.6f}, max={variance.max():.6f}")

            # PCA dimensionality reduction
            if self.pca_dim is not None and self.pca_dim < D:
                pca_dim = min(self.pca_dim, N - 1, D)
                self._pca = PCA(n_components=pca_dim, random_state=self.random_state)
                features = self._pca.fit_transform(features)
                # Project diagonal variances: diag(W^T diag(s_i) W) = s_i @ W^2
                W_sq = self._pca.components_.T ** 2  # (D, d)
                variance = variance @ W_sq            # (N, d)
                ev = self._pca.explained_variance_ratio_.sum()
                self._log(f"PCA: {D}D -> {pca_dim}D (explained variance: {ev:.2%})")
            else:
                self._pca = None

            labels = self._heteroscedastic_dpem(
                features, variance,
                het_covariance_type=self.het_covariance_type,
            )
        else:
            # Apply PCA for deterministic path too (reduces cost at high K_max)
            if self.pca_dim is not None and self.pca_dim < D:
                pca_dim = min(self.pca_dim, N - 1, D)
                self._pca = PCA(n_components=pca_dim, random_state=self.random_state)
                features = self._pca.fit_transform(features)
                ev = self._pca.explained_variance_ratio_.sum()
                self._log(f"PCA: {D}D -> {pca_dim}D (explained variance: {ev:.2%})")
            else:
                self._pca = None
            self._log(f"Deterministic path: sklearn BayesianGaussianMixture "
                       f"(K_max={self.max_components}, cov={self.covariance_type})")
            dpgmm = BayesianGaussianMixture(
                n_components=self.max_components,
                covariance_type=self.covariance_type,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=self.alpha_prior,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=1,  # variational ELBO is stable; multiple restarts not needed
                verbose=2 if self.verbose else 0,
                verbose_interval=10,
            )
            dpgmm.fit(features)
            labels = dpgmm.predict(features)

            self.weights_ = dpgmm.weights_
            self.active_mask_ = dpgmm.weights_ > self.weight_threshold
            self.n_active_ = int(self.active_mask_.sum())
            self._pca = None

        # Discard tiny clusters
        if self.min_bin_size > 1:
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                if label != -1 and count < self.min_bin_size:
                    labels[labels == label] = -1

        return labels

    # ------------------------------------------------------------------
    # Heteroscedastic MAP-EM with Dirichlet prior
    # ------------------------------------------------------------------

    def _heteroscedastic_dpem(self, means, variances, het_covariance_type="diag"):
        """Run heteroscedastic MAP-EM with multiple restarts.

        After EM convergence, optionally merges components with low
        symmetric KL divergence and re-computes responsibilities.

        Parameters
        ----------
        means : ndarray (N, D)
            Point-estimate embeddings (already PCA-reduced if applicable).
        variances : ndarray (N, D)
            Per-sample diagonal variances (already projected if applicable).
        het_covariance_type : str
            ``"diag"`` or ``"full"`` cluster covariance for the EM path.

        Returns
        -------
        labels : ndarray (N,)
            Cluster labels; -1 = unassigned.
        """
        N, D = means.shape
        rng = np.random.RandomState(self.random_state)
        eps = 1e-10

        best_obj = -np.inf
        best_result = None
        all_run_diagnostics = []

        if self.algorithm == "vbem":
            _single_fn = (self._vbem_single_full if het_covariance_type == "full"
                          else self._vbem_single)
            algo_label = "VB-EM"
        else:
            _single_fn = (self._dpem_single_full if het_covariance_type == "full"
                          else self._dpem_single)
            algo_label = "MAP-EM"
        self._log(f"Algorithm: {algo_label} (het_cov={het_covariance_type})")

        for run_i in range(self.n_init):
            seed = rng.randint(0, 2**31)
            self._log(f"Run {run_i + 1}/{self.n_init} (seed={seed})")
            result = _single_fn(means, variances, seed,
                                collect_diagnostics=self.collect_diagnostics)
            k_active = int(result["active"].sum())
            self._log(f"  Run {run_i + 1} done: obj={result['map_obj']:.2f}, "
                       f"K_active={k_active}, iters={result['n_iter']}, "
                       f"converged={result['converged']}")
            if self.collect_diagnostics:
                all_run_diagnostics.append(result.get("iter_diagnostics", []))
            if result["map_obj"] > best_obj:
                best_obj = result["map_obj"]
                best_result = result

        self._log(f"Best run: obj={best_obj:.2f}, K_active={int(best_result['active'].sum())}")

        pi_k = best_result["pi_k"]
        m_k = best_result["m_k"]
        c_k = best_result["c_k"]
        active = best_result["active"]

        # --- Post-EM: merge similar components ---
        k_before_merge = int(active.sum())
        merge_diag = []
        if self.merge_threshold is not None:
            self._log(f"Merging components (threshold={self.merge_threshold:.2f}, "
                       f"K_before={k_before_merge})...")
            pi_k, m_k, c_k, active, merge_diag = self._merge_components(
                pi_k, m_k, c_k, active,
                best_result["resp"], variances,
                het_covariance_type=het_covariance_type,
            )
            self._log(f"After merging: K={int(active.sum())} "
                       f"(merged {k_before_merge - int(active.sum())})")

        # --- Final E-step to get clean responsibilities ---
        K = len(pi_k)
        log_resp = np.full((N, K), -np.inf)
        full_cov = (het_covariance_type == "full")
        for k in range(K):
            if not active[k]:
                continue
            delta = means - m_k[k]           # (N, D)
            if full_cov:
                # c_k[k] is (D, D); variances[i] adds to diagonal
                Sigma_k = c_k[k][np.newaxis] + np.einsum(
                    'nd,de->nde', variances, np.eye(variances.shape[1]))
                _, logdet = np.linalg.slogdet(Sigma_k)
                Sigma_inv = np.linalg.inv(Sigma_k)
                quad = np.einsum('ni,nij,nj->n', delta, Sigma_inv, delta)
                log_resp[:, k] = np.log(pi_k[k] + eps) - 0.5 * logdet - 0.5 * quad
            else:
                sigma_ik = c_k[k] + variances
                log_resp[:, k] = (
                    np.log(pi_k[k] + eps)
                    - 0.5 * np.sum(np.log(sigma_ik + eps), axis=1)
                    - 0.5 * np.sum(delta * delta / (sigma_ik + eps), axis=1)
                )
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_norm)

        # Store model attributes
        self.means_ = m_k
        self.covariances_ = c_k
        self.weights_ = pi_k
        self.active_mask_ = active
        self.n_active_ = int(active.sum())
        self.n_iter_ = best_result["n_iter"]
        self.converged_ = best_result["converged"]

        # Store diagnostics if requested
        if self.collect_diagnostics:
            self.diagnostics_ = {
                "all_runs": all_run_diagnostics,
                "best_run": best_result.get("iter_diagnostics", []),
            }
            self.merge_diagnostics_ = merge_diag

        # Assign labels with confidence thresholding
        masked_resp = resp.copy()
        masked_resp[:, ~active] = -1.0
        labels = np.argmax(masked_resp, axis=1)
        max_resp = resp[np.arange(N), labels]
        labels[max_resp < self.confidence_threshold] = -1

        # Remap component indices to contiguous 0, 1, 2, ...
        active_indices = np.where(active)[0]
        remap = {int(old): new for new, old in enumerate(active_indices)}
        remap[-1] = -1
        labels = np.array([remap.get(int(l), -1) for l in labels])

        n_assigned = int((labels != -1).sum())
        self._log(f"Final: K={self.n_active_}, assigned={n_assigned}/{N} "
                   f"({n_assigned / N * 100:.1f}%), conf_thresh={self.confidence_threshold:.2f}")

        return labels

    # ------------------------------------------------------------------
    # Post-EM component merging
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetric_kl_diag(mu_a, var_a, mu_b, var_b):
        """Average per-dimension symmetric KL between two diagonal Gaussians.

        KL(a||b) + KL(b||a) = 0.5 * sum_d [ var_a/var_b + var_b/var_a - 2
                                             + (mu_a-mu_b)^2 * (1/var_a + 1/var_b) ]
        Returns the total divided by D (average per dimension).
        """
        eps = 1e-10
        ratio_ab = var_a / (var_b + eps)
        ratio_ba = var_b / (var_a + eps)
        delta_sq = (mu_a - mu_b) ** 2
        inv_sum = 1.0 / (var_a + eps) + 1.0 / (var_b + eps)
        total = 0.5 * np.sum(ratio_ab + ratio_ba - 2.0 + delta_sq * inv_sum)
        return total / len(mu_a)

    def _merge_components(self, pi_k, m_k, c_k, active, resp, variances,
                          het_covariance_type="diag"):
        """Greedily merge active components with low symmetric KL.

        Uses effective variance (cluster covariance + responsibility-weighted
        average per-sample noise) so that the merge criterion reflects the
        true observation spread, not just the residual cluster covariance.

        Returns pi_k, m_k, c_k, active, merge_log where merge_log is a list
        of dicts {a, b, kl} for each merged pair (populated when
        collect_diagnostics is True on the parent object).
        """
        eps = 1e-10
        full_cov = (het_covariance_type == "full")

        # For KL, always use diagonal effective variance
        # (for full-cov case, extract diagonal of C_k)
        def _diag_c(k):
            return np.diag(c_k[k]) if full_cov else c_k[k]

        # Compute effective diagonal variance per component: diag(c_k) + avg noise
        eff_var = np.array([_diag_c(k) for k in range(len(active))])
        n_k = resp.sum(axis=0)
        for k in np.where(active)[0]:
            if n_k[k] > eps:
                avg_noise = np.sum(
                    resp[:, k, np.newaxis] * variances, axis=0
                ) / (n_k[k] + eps)
                eff_var[k] = _diag_c(k) + avg_noise

        active_idx = list(np.where(active)[0])
        merge_log = []
        merged = True

        while merged:
            merged = False
            best_kl = np.inf
            best_pair = None

            # Find closest pair using effective diagonal variance
            for i in range(len(active_idx)):
                for j in range(i + 1, len(active_idx)):
                    a, b = active_idx[i], active_idx[j]
                    kl = self._symmetric_kl_diag(
                        m_k[a], eff_var[a], m_k[b], eff_var[b]
                    )
                    if kl < best_kl:
                        best_kl = kl
                        best_pair = (a, b)

            if best_pair is not None and best_kl < self.merge_threshold:
                a, b = best_pair
                merge_log.append({"a": int(a), "b": int(b), "kl": float(best_kl)})

                # Merge b into a: weighted average
                w_a, w_b = pi_k[a], pi_k[b]
                w_total = w_a + w_b
                t = w_b / (w_total + eps)

                # Merged mean
                new_mu = (1 - t) * m_k[a] + t * m_k[b]
                delta = m_k[a] - m_k[b]

                # Merged covariance (diagonal or full)
                if full_cov:
                    delta_outer = np.outer(delta, delta)
                    new_cov = ((1 - t) * c_k[a] + t * c_k[b]
                               + t * (1 - t) * delta_outer)
                    new_cov = (new_cov + new_cov.T) / 2 + self.min_covar * np.eye(len(delta))
                    new_eff_diag = ((1 - t) * eff_var[a] + t * eff_var[b]
                                    + t * (1 - t) * delta ** 2)
                    c_k[a] = new_cov
                else:
                    new_var = ((1 - t) * c_k[a] + t * c_k[b]
                               + t * (1 - t) * delta ** 2)
                    new_eff_diag = ((1 - t) * eff_var[a] + t * eff_var[b]
                                    + t * (1 - t) * delta ** 2)
                    c_k[a] = np.maximum(new_var, self.min_covar)

                m_k[a] = new_mu
                eff_var[a] = np.maximum(new_eff_diag, self.min_covar)
                pi_k[a] = w_total

                # Deactivate b
                active[b] = False
                pi_k[b] = 0.0
                active_idx.remove(b)
                merged = True

        # Re-normalize weights
        pi_sum = pi_k[active].sum()
        if pi_sum > eps:
            pi_k[active] /= pi_sum
            pi_k[~active] = 0.0

        return pi_k, m_k, c_k, active, merge_log

    def _dpem_single(self, means, variances, seed, collect_diagnostics=False):
        """Single heteroscedastic MAP-EM run with Dirichlet prior (diagonal cov).

        Parameters
        ----------
        means : ndarray (N, D)
        variances : ndarray (N, D)
        seed : int
            Random seed for KMeans initialization.
        collect_diagnostics : bool
            If True, include ``iter_diagnostics`` in returned dict.

        Returns
        -------
        dict with keys: pi_k, m_k, c_k, active, resp, map_obj, n_iter,
            converged, and optionally iter_diagnostics.
        """
        N, D = means.shape
        K = self.max_components
        alpha = self.alpha_prior
        eps = 1e-10

        # --- Initialize with KMeans ---
        km = KMeans(n_clusters=K, random_state=seed, n_init=1)
        km_labels = km.fit_predict(means)

        m_k = km.cluster_centers_.copy()                  # (K, D)
        c_k = np.full((K, D), self.min_covar)             # (K, D)
        pi_k = np.bincount(km_labels, minlength=K).astype(np.float64)
        pi_k = np.maximum(pi_k, eps)
        pi_k /= pi_k.sum()
        active = np.ones(K, dtype=bool)

        # Within-cluster variance initialization
        for k in range(K):
            mask = km_labels == k
            if mask.sum() > 1:
                c_k[k] = np.maximum(np.var(means[mask], axis=0), self.min_covar)

        prev_obj = -np.inf
        converged = False
        n_iter = 0
        resp = None
        map_obj = -np.inf
        iter_diagnostics = [] if collect_diagnostics else None

        for iteration in range(self.max_iter):
            n_iter = iteration + 1
            # --- E-step ---
            log_resp = np.full((N, K), -np.inf)
            for k in range(K):
                if not active[k]:
                    continue
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

            # MAP objective = log-likelihood + Dirichlet log-prior
            ll = float(np.sum(log_norm))
            active_weights = pi_k[active]
            active_weights_safe = np.maximum(active_weights, eps)
            log_prior = float(np.sum((alpha - 1) * np.log(active_weights_safe)))
            map_obj = ll + log_prior

            # Convergence check
            n_active_now = int(active.sum())
            if iteration % 10 == 0 or iteration < 5:
                self._log(f"  iter {iteration:3d}: MAP={map_obj:.2f}, "
                           f"LL={ll:.2f}, K_active={n_active_now}")
            if iteration > 0 and abs(map_obj - prev_obj) < self.tol * max(1.0, abs(prev_obj)):
                converged = True
                self._log(f"  Converged at iter {iteration} "
                           f"(MAP={map_obj:.2f}, K_active={n_active_now})")
                if collect_diagnostics:
                    iter_diagnostics.append({
                        "iter": iteration, "map_obj": map_obj, "ll": ll,
                        "K_active": n_active_now, "n_pruned": 0, "converged": True,
                    })
                break
            prev_obj = map_obj

            # --- M-step ---
            n_k = resp.sum(axis=0)  # (K,)

            # Mixing weights with Dirichlet MAP estimate
            pi_k_raw = n_k + alpha - 1.0
            pi_k_raw[~active] = 0.0
            pi_k_raw = np.maximum(pi_k_raw, 0.0)

            # Ensure at least one component survives
            if pi_k_raw.sum() < eps:
                best_k = np.argmax(n_k)
                pi_k_raw[best_k] = n_k[best_k]
                active[best_k] = True

            pi_k = pi_k_raw / (pi_k_raw.sum() + eps)

            # Prune dead components
            n_pruned = 0
            for k in range(K):
                if active[k] and pi_k[k] < self.weight_threshold:
                    active[k] = False
                    pi_k[k] = 0.0
                    n_pruned += 1

            # Re-normalize after pruning
            pi_sum = pi_k.sum()
            if pi_sum > eps:
                pi_k /= pi_sum

            # Update centroids and covariances for active components
            for k in range(K):
                if not active[k] or n_k[k] < eps:
                    continue

                r = resp[:, k, np.newaxis]               # (N, 1)
                sigma_ik = c_k[k] + variances            # (N, D)
                precision = 1.0 / (sigma_ik + eps)       # (N, D)

                # Precision-weighted mean (Eq. 32)
                numerator = np.sum(r * precision * means, axis=0)   # (D,)
                denominator = np.sum(r * precision, axis=0)         # (D,)
                m_k[k] = numerator / (denominator + eps)

                # Cluster variance via moment matching
                delta_sq = (means - m_k[k]) ** 2                    # (N, D)
                observed_var = np.sum(r * delta_sq, axis=0) / (n_k[k] + eps)
                mean_noise = np.sum(r * variances, axis=0) / (n_k[k] + eps)
                c_k[k] = np.maximum(observed_var - mean_noise, self.min_covar)

            if collect_diagnostics:
                iter_diagnostics.append({
                    "iter": iteration, "map_obj": map_obj, "ll": ll,
                    "K_active": n_active_now, "n_pruned": n_pruned, "converged": False,
                })

        out = {
            "pi_k": pi_k, "m_k": m_k, "c_k": c_k,
            "active": active, "resp": resp, "map_obj": map_obj,
            "n_iter": n_iter, "converged": converged,
        }
        if collect_diagnostics:
            out["iter_diagnostics"] = iter_diagnostics
        return out

    def _dpem_single_full(self, means, variances, seed, collect_diagnostics=False):
        """Single heteroscedastic MAP-EM run with full (D×D) cluster covariance.

        Same interface as ``_dpem_single`` but ``c_k`` is stored as (K, D, D)
        full symmetric positive-definite matrices.  The sample noise is still
        diagonal (``S_i = diag(variances[i])``), so the effective covariance
        per sample-component pair is ``C_k + diag(s_i)`` computed on the fly.

        Parameters
        ----------
        means : ndarray (N, D)
        variances : ndarray (N, D)  — diagonal entries of per-sample noise
        seed : int
        collect_diagnostics : bool

        Returns
        -------
        dict with same keys as ``_dpem_single`` (c_k has shape (K, D, D)).
        """
        N, D = means.shape
        K = self.max_components
        alpha = self.alpha_prior
        eps = 1e-10

        # Pre-build (N, D, D) diagonal arrays from per-sample variances
        # variances_3d[i] = diag(variances[i])
        eye_D = np.eye(D)
        variances_3d = variances[:, :, np.newaxis] * eye_D[np.newaxis]  # (N, D, D)

        # --- Initialize with KMeans ---
        km = KMeans(n_clusters=K, random_state=seed, n_init=1)
        km_labels = km.fit_predict(means)

        m_k = km.cluster_centers_.copy()                        # (K, D)
        c_k = np.stack([self.min_covar * eye_D] * K)           # (K, D, D)
        pi_k = np.bincount(km_labels, minlength=K).astype(np.float64)
        pi_k = np.maximum(pi_k, eps)
        pi_k /= pi_k.sum()
        active = np.ones(K, dtype=bool)

        # Within-cluster full covariance initialization
        for k in range(K):
            mask = km_labels == k
            if mask.sum() > 1:
                cov_k = np.cov(means[mask].T)
                if cov_k.ndim == 0:
                    cov_k = np.eye(D) * float(cov_k)
                c_k[k] = np.maximum(cov_k, self.min_covar * eye_D)

        prev_obj = -np.inf
        converged = False
        n_iter = 0
        resp = None
        map_obj = -np.inf
        iter_diagnostics = [] if collect_diagnostics else None

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # --- E-step (full covariance) ---
            log_resp = np.full((N, K), -np.inf)
            for k in range(K):
                if not active[k]:
                    continue
                # Sigma_ik = C_k + diag(s_i)  shape (N, D, D)
                Sigma_k = c_k[k][np.newaxis] + variances_3d    # (N, D, D)
                _, logdet = np.linalg.slogdet(Sigma_k)         # (N,)
                Sigma_inv = np.linalg.inv(Sigma_k)             # (N, D, D)
                delta = means - m_k[k]                          # (N, D)
                quad = np.einsum('ni,nij,nj->n', delta, Sigma_inv, delta)
                log_resp[:, k] = np.log(pi_k[k] + eps) - 0.5 * logdet - 0.5 * quad

            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_norm)

            # MAP objective
            ll = float(np.sum(log_norm))
            active_weights_safe = np.maximum(pi_k[active], eps)
            log_prior = float(np.sum((alpha - 1) * np.log(active_weights_safe)))
            map_obj = ll + log_prior

            n_active_now = int(active.sum())
            if iteration % 10 == 0 or iteration < 5:
                self._log(f"  iter {iteration:3d}: MAP={map_obj:.2f}, "
                          f"LL={ll:.2f}, K_active={n_active_now}")
            if iteration > 0 and abs(map_obj - prev_obj) < self.tol * max(1.0, abs(prev_obj)):
                converged = True
                self._log(f"  Converged at iter {iteration} "
                          f"(MAP={map_obj:.2f}, K_active={n_active_now})")
                if collect_diagnostics:
                    iter_diagnostics.append({
                        "iter": iteration, "map_obj": map_obj, "ll": ll,
                        "K_active": n_active_now, "n_pruned": 0, "converged": True,
                    })
                break
            prev_obj = map_obj

            # --- M-step ---
            n_k = resp.sum(axis=0)

            pi_k_raw = n_k + alpha - 1.0
            pi_k_raw[~active] = 0.0
            pi_k_raw = np.maximum(pi_k_raw, 0.0)
            if pi_k_raw.sum() < eps:
                best_k = np.argmax(n_k)
                pi_k_raw[best_k] = n_k[best_k]
                active[best_k] = True
            pi_k = pi_k_raw / (pi_k_raw.sum() + eps)

            n_pruned = 0
            for k in range(K):
                if active[k] and pi_k[k] < self.weight_threshold:
                    active[k] = False
                    pi_k[k] = 0.0
                    n_pruned += 1

            pi_sum = pi_k.sum()
            if pi_sum > eps:
                pi_k /= pi_sum

            for k in range(K):
                if not active[k] or n_k[k] < eps:
                    continue

                r = resp[:, k]                                  # (N,)
                Sigma_k = c_k[k][np.newaxis] + variances_3d    # (N, D, D)
                Sigma_inv = np.linalg.inv(Sigma_k)             # (N, D, D)

                # Precision-weighted mean: m = (Σ_i r_i Σ_ik^{-1})^{-1} (Σ_i r_i Σ_ik^{-1} x_i)
                rS = r[:, np.newaxis, np.newaxis] * Sigma_inv  # (N, D, D)
                denom = rS.sum(axis=0)                          # (D, D)
                numer = np.einsum('nij,nj->i', rS, means)       # (D,)
                m_k[k] = np.linalg.solve(denom + eps * eye_D, numer)

                # Full covariance via moment matching
                delta = means - m_k[k]                          # (N, D)
                outer = np.einsum('n,ni,nj->ij', r, delta, delta) / (n_k[k] + eps)
                mean_noise = np.einsum('n,nij->ij', r, variances_3d) / (n_k[k] + eps)
                c_new = outer - mean_noise
                c_new = (c_new + c_new.T) / 2 + self.min_covar * eye_D
                # Ensure PSD via eigenvalue clipping
                eigvals, eigvecs = np.linalg.eigh(c_new)
                eigvals = np.maximum(eigvals, self.min_covar)
                c_k[k] = eigvecs @ np.diag(eigvals) @ eigvecs.T

            if collect_diagnostics:
                iter_diagnostics.append({
                    "iter": iteration, "map_obj": map_obj, "ll": ll,
                    "K_active": n_active_now, "n_pruned": n_pruned, "converged": False,
                })

        out = {
            "pi_k": pi_k, "m_k": m_k, "c_k": c_k,
            "active": active, "resp": resp, "map_obj": map_obj,
            "n_iter": n_iter, "converged": converged,
        }
        if collect_diagnostics:
            out["iter_diagnostics"] = iter_diagnostics
        return out

    # ------------------------------------------------------------------
    # Truncated stick-breaking DP-VBEM helpers and methods
    # ------------------------------------------------------------------

    @staticmethod
    def _stick_log_pi(a_k, b_k):
        """Expected log stick-breaking weights E_q[log π_k] for k=0,...,T-1.

        Uses T-1 Beta variational distributions q(β_k) = Beta(a_k, b_k).
        The last component k=T-1 has no Beta variable (β_T := 1) and receives
        the expectation of the remaining log-stick.

        Parameters
        ----------
        a_k : ndarray (T-1,)  Beta shape parameters a
        b_k : ndarray (T-1,)  Beta shape parameters b

        Returns
        -------
        E_log_pi : ndarray (T,)
        """
        ab = a_k + b_k
        E_log_beta   = digamma(a_k) - digamma(ab)   # E[log β_k],       (T-1,)
        E_log_1m     = digamma(b_k) - digamma(ab)   # E[log(1-β_k)],    (T-1,)
        # Cumulative sum of E[log(1-β_j)] for j < k, prepend 0 for k=0
        cum_log_1m   = np.concatenate([[0.0], np.cumsum(E_log_1m[:-1])])
        T = len(a_k) + 1
        E_log_pi = np.empty(T)
        E_log_pi[:T-1] = E_log_beta + cum_log_1m    # E[log β_k] + Σ_{j<k} E[log(1-β_j)]
        E_log_pi[T-1]  = np.sum(E_log_1m)           # last stick: no β_T, absorbs remainder
        return E_log_pi

    @staticmethod
    def _stick_expected_pi(a_k, b_k):
        """Expected stick-breaking weights E_q[π_k] for k=0,...,T-1.

        Exact under mean-field q(β) = ∏_k Beta(a_k, b_k):
          E[π_k] = E[β_k] · ∏_{j<k} E[1-β_j]

        Parameters
        ----------
        a_k : ndarray (T-1,)
        b_k : ndarray (T-1,)

        Returns
        -------
        E_pi : ndarray (T,)  (sums to 1 up to floating point)
        """
        ab = a_k + b_k
        E_beta    = a_k / ab                         # E[β_k],       (T-1,)
        E_1m_beta = b_k / ab                         # E[1-β_k],     (T-1,)
        # Cumulative product of E[1-β_j] for j < k, prepend 1 for k=0
        cum_prod  = np.concatenate([[1.0], np.cumprod(E_1m_beta[:-1])])
        T = len(a_k) + 1
        eps = 1e-300
        E_pi = np.empty(T)
        E_pi[:T-1] = E_beta * cum_prod
        E_pi[T-1]  = np.prod(E_1m_beta)             # last component: all remaining mass
        # Renormalize for floating-point robustness
        E_pi = np.maximum(E_pi, 0.0)
        E_pi /= E_pi.sum() + eps
        return E_pi

    def _dp_elbo(self, r, a_k, b_k, lam_k, m_k, c_k,
                 means, variances, m_0, beta_0, alpha_0):
        """True ELBO for truncated stick-breaking DP-VBEM (diagonal covariance).

        Computes E1+E2+E3+E4+H1+H2+H3 up to additive constants that are
        independent of all optimization variables (r, a_k, b_k, μ̂_k, λ̂_k, c_k).

        Dropped constants (documented explicitly):
          - -D/2 log(2π) from E1 (likelihood) and E4 (mean prior)
          - D/2 (1 + log(2π)) from H3 (Normal entropy)

        Parameters
        ----------
        r      : (N, T) responsibilities
        a_k    : (T-1,) Beta shape a
        b_k    : (T-1,) Beta shape b
        lam_k  : (T, D) posterior mean precisions
        m_k    : (T, D) posterior means
        c_k    : (T, D) cluster variances (point estimates)
        means  : (N, D) observed embeddings
        variances : (N, D) per-sample noise
        m_0    : (D,) prior mean
        beta_0 : float prior mean precision
        alpha_0 : float DP concentration

        Returns
        -------
        elbo : float
        """
        N, T = r.shape
        eps = 1e-10

        # Precompute E_q[log π_k]
        E_log_pi = self._stick_log_pi(a_k, b_k)     # (T,)

        # E1 — expected log-likelihood (up to -D/2 log(2π) constant)
        E1 = 0.0
        for k in range(T):
            r_k    = r[:, k]                         # (N,)
            sigma  = c_k[k] + variances              # (N, D)
            delta  = means - m_k[k]                  # (N, D)
            vb_c   = 1.0 / (lam_k[k] + eps)         # (D,) posterior variance
            ll_k   = (-0.5 * np.sum(np.log(sigma + eps), axis=1)
                      - 0.5 * np.sum((delta**2 + vb_c) / (sigma + eps), axis=1))
            E1 += np.dot(r_k, ll_k)

        # E2 — E_q[log p(Z | β)]
        E2 = float(np.sum(r * E_log_pi[np.newaxis, :]))

        # E3 — E_q[log p(β)] = Σ_k [log α_0 + (α_0-1) E_q[log(1-β_k)]]
        ab  = a_k + b_k
        E_log_1m = digamma(b_k) - digamma(ab)
        E3 = float(np.sum(np.log(alpha_0 + eps) + (alpha_0 - 1.0) * E_log_1m))

        # E4 — E_q[log p(M)] (up to -D/2 log(2π) constant)
        E4 = 0.0
        for k in range(T):
            diff   = m_k[k] - m_0                   # (D,)
            tr_inv = np.sum(1.0 / (lam_k[k] + eps)) # tr(Λ̂_k⁻¹)
            E4 += (0.5 * len(m_0) * np.log(beta_0 + eps)
                   - 0.5 * beta_0 * (np.dot(diff, diff) + tr_inv))

        # H1 — entropy of q(Z)
        safe_r = r[r > eps]
        H1 = -float(np.sum(safe_r * np.log(safe_r)))

        # H2 — entropy of q(β) = Σ_k H[Beta(a_k, b_k)]
        from scipy.special import betaln
        H2 = float(np.sum(
            betaln(a_k, b_k)
            - (a_k - 1.0) * digamma(a_k)
            - (b_k - 1.0) * digamma(b_k)
            + (ab - 2.0)  * digamma(ab)
        ))

        # H3 — entropy of q(M) (up to D/2(1+log(2π)) constant)
        # H[N(μ̂, Λ̂⁻¹)] = -½ Σ_d log λ̂_kd + const
        H3 = -0.5 * float(np.sum(np.log(lam_k + eps)))

        return E1 + E2 + E3 + E4 + H1 + H2 + H3

    def _vbem_single(self, means, variances, seed, collect_diagnostics=False):
        """Single truncated stick-breaking DP-VBEM run (diagonal cluster covariance).

        Variational families:
          q(β_k) = Beta(a_k, b_k)         k = 1,...,T-1  [EXACT update]
          q(μ_k) = ∏_d N(m̂_kd, λ̂_kd⁻¹)  k = 1,...,T    [EXACT update]
          q(z_i = k) = r_ik                               [EXACT update]
          C_k : point estimate (generalized M-step)        [APPROX]

        Cluster covariances are updated by damped gradient ascent on the ELBO
        in the reparameterized space c_kd = min_covar + exp(ρ_kd), which
        guarantees c_kd > min_covar without hard clipping.

        Because C_k is not part of the variational family, the ELBO is not
        guaranteed to be non-decreasing across outer iterations. Occasional
        small decreases are possible and are logged but not treated as errors.

        The ELBO is computed after all updates in each outer iteration,
        reflecting the full current state (r, a_k, b_k, μ̂_k, λ̂_k, c_k).
        Constants independent of all optimization variables are dropped (see
        ``_dp_elbo`` docstring).

        Parameters
        ----------
        means : ndarray (N, D)
        variances : ndarray (N, D)
        seed : int
        collect_diagnostics : bool

        Returns
        -------
        dict with keys: pi_k, m_k, c_k, active, resp, map_obj (= ELBO),
            n_iter, converged, and optionally iter_diagnostics.

        Remaining approximations
        ------------------------
        1. C_k is a point estimate, not variational (conjugacy broken by S_i).
        2. Gradient step for c_kd is a first-order generalized M-step, not a
           coordinate-ascent step; ELBO may decrease at inner steps.
        3. Truncation error is O(α_0^T), negligible for T >> K_true.
        """
        from scipy.special import betaln  # noqa: F401 (used in _dp_elbo)

        N, D = means.shape
        T = self.max_components          # truncation level (T components, T-1 Beta vars)
        alpha_0 = self.alpha_prior
        beta_0  = self.beta_0
        eta     = self.cov_step_size
        m_0 = means.mean(axis=0) if self.m_0 == "data_mean" else np.zeros(D)
        eps = 1e-10

        # --- Initialize with KMeans ---
        km = KMeans(n_clusters=T, random_state=seed, n_init=1)
        km_labels = km.fit_predict(means)

        m_k  = km.cluster_centers_.copy()           # (T, D) variational means
        c_k  = np.full((T, D), self.min_covar)      # (T, D) cluster variances (point est.)
        n_k_init  = np.bincount(km_labels, minlength=T).astype(np.float64)
        mean_noise = variances.mean(axis=0)          # (D,)

        for k in range(T):
            mask = km_labels == k
            if mask.sum() > 1:
                c_k[k] = np.maximum(np.var(means[mask], axis=0), self.min_covar)

        # Warm-start lam_k to avoid astronomically large VB corrections at iter 0
        lam_k = np.array([
            beta_0 + n_k_init[k] / (c_k[k] + mean_noise + eps)
            for k in range(T)
        ])  # (T, D)

        # Initialize Beta parameters from uniform assignment
        # a_k = 1 + n_k_init[:T-1]  (avoid T-1 index which has no Beta var)
        n_tail = (np.cumsum(n_k_init[::-1])[::-1])[1:]  # Σ_{l>k} n_l for k=0..T-2
        a_k = 1.0 + n_k_init[:T-1]                       # (T-1,)
        b_k = alpha_0 + n_tail                            # (T-1,)

        # ρ_kd = log(c_kd - min_covar) reparameterization (c_kd > min_covar always)
        rho_k = np.log(np.maximum(c_k - self.min_covar, eps))  # (T, D)

        prev_elbo = None           # None until first ELBO is available
        converged = False
        n_iter = 0
        resp = np.ones((N, T)) / T      # initial (overwritten at first E-step)
        elbo = -np.inf
        tiny_mass = 1e-10          # threshold below which n_k[k] is treated as zero
        iter_diagnostics = [] if collect_diagnostics else None

        for iteration in range(self.max_iter):
            n_iter = iteration + 1

            # ---- VB E-step: update q(z) (EXACT) ----
            E_log_pi = self._stick_log_pi(a_k, b_k)    # (T,)
            log_resp = np.full((N, T), -np.inf)
            for k in range(T):
                sigma_ik = c_k[k] + variances           # (N, D)
                delta    = means - m_k[k]               # (N, D)
                vb_corr  = 1.0 / (lam_k[k] + eps)      # (D,) posterior variance
                log_resp[:, k] = (
                    E_log_pi[k]
                    - 0.5 * np.sum(np.log(sigma_ik + eps), axis=1)
                    - 0.5 * np.sum((delta**2 + vb_corr) / (sigma_ik + eps), axis=1)
                )
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp     = np.exp(log_resp - log_norm)
            n_k      = resp.sum(axis=0)                 # (T,)

            # ---- VB M-step for β: EXACT stick-breaking update ----
            n_tail_new = (np.cumsum(n_k[::-1])[::-1])[1:]  # Σ_{l>k} n_l, shape (T-1,)
            a_k = 1.0 + n_k[:T-1]
            b_k = alpha_0 + n_tail_new

            # ---- VB M-step for m_k + generalized M-step for c_k ----
            for k in range(T):
                r = resp[:, k]                           # (N,)

                # Numerical stabilization: skip update for nearly empty components.
                # This is NOT part of the ideal derivation; it prevents degenerate
                # gradient steps when n_k[k] ~ 0 and the mean/covariance are
                # effectively unconstrained by the data.
                if n_k[k] < tiny_mass:
                    lam_k[k] = beta_0 * np.ones(D)
                    m_k[k]   = m_0.copy()
                    continue

                # Generalized M-step for c_kd is a PARTIAL gradient of the ELBO
                # with respect to C_k, holding q(z), q(beta), and q(m) fixed.
                # It is NOT the full total derivative of the ELBO (because m_k and
                # lam_k themselves depend on C_k through the exact M-step below).
                # This is a generalized variational M-step, not coordinate ascent.
                for _ in range(self.cov_inner_steps):
                    c_k[k]   = self.min_covar + np.exp(rho_k[k])   # c_kd > min_covar
                    sigma_ik = c_k[k] + variances                    # (N, D)

                    # Exact M-step for q(m_k), given current c_k[k]:
                    #   lambda_kd = beta_0 + sum_i r_ik / (c_kd + s_id)
                    lam_k[k] = beta_0 + np.sum(
                        r[:, np.newaxis] / (sigma_ik + eps), axis=0
                    )
                    #   mu_kd = [beta_0 * m_0d + sum_i r_ik * x_id/(c_kd+s_id)] / lambda_kd
                    numer    = beta_0 * m_0 + np.sum(
                        r[:, np.newaxis] * means / (sigma_ik + eps), axis=0
                    )
                    m_k[k]   = numer / (lam_k[k] + eps)

                    # Partial ELBO gradient w.r.t. c_kd (q(z), q(beta), q(m) held fixed):
                    #   dL/dc_kd = sum_i r_ik * [Q_id/(c_kd+s_id)^2 - 1/(c_kd+s_id)] / 2
                    #   where Q_id = (x_id - mu_kd)^2 + 1/lambda_kd  (VB uncertainty term)
                    delta    = means - m_k[k]                        # (N, D)
                    vb_corr  = 1.0 / (lam_k[k] + eps)               # (D,)
                    Q        = delta**2 + vb_corr                    # (N, D) Q_id
                    dL_dc    = 0.5 * np.sum(
                        r[:, np.newaxis] * (Q / (sigma_ik**2 + eps)
                                            - 1.0 / (sigma_ik + eps)), axis=0
                    )  # (D,)
                    # Chain rule: dL/d_rho_kd = exp(rho_kd) * dL/dc_kd
                    exp_rho  = np.exp(rho_k[k])                      # = c_kd - min_covar
                    grad_rho = exp_rho * dL_dc
                    # Numerical stabilization: rescale if any entry is too large.
                    # This is not part of the derivation; it prevents overflow.
                    max_g = np.abs(grad_rho).max()
                    if max_g > 1.0:
                        grad_rho = grad_rho / max_g
                    rho_k[k] = np.clip(
                        rho_k[k] + eta * grad_rho, -30.0, 10.0
                    )
                    c_k[k]   = self.min_covar + np.exp(rho_k[k])

            # ---- Compute true ELBO after all updates ----
            elbo = self._dp_elbo(
                resp, a_k, b_k, lam_k, m_k, c_k,
                means, variances, m_0, beta_0, alpha_0,
            )
            n_active_now = int((self._stick_expected_pi(a_k, b_k) > self.weight_threshold).sum())

            if iteration % 10 == 0 or iteration < 5:
                self._log(f"  iter {iteration:3d}: ELBO={elbo:.4f}, K_active~{n_active_now}")

            # Convergence check: abs(delta_ELBO) < tol * max(1, |prev_ELBO|).
            # Non-monotone ELBO from the generalized C_k M-step is expected and logged.
            if prev_elbo is not None:
                delta_elbo = elbo - prev_elbo
                if delta_elbo < 0:
                    self._log(f"  [note] ELBO decreased by {-delta_elbo:.4f} "
                              f"(generalized M-step; not an error)")
                if abs(delta_elbo) < self.tol * max(1.0, abs(prev_elbo)):
                    converged = True
                    self._log(f"  Converged at iter {iteration} "
                              f"(ELBO={elbo:.4f}, K_active~{n_active_now})")
                    if collect_diagnostics:
                        iter_diagnostics.append({
                            # map_obj stores the ELBO in the vbem path
                            "iter": iteration, "map_obj": elbo, "ll": 0.0,
                            "K_active": n_active_now, "n_pruned": 0, "converged": True,
                        })
                    break
            prev_elbo = elbo

            if collect_diagnostics:
                iter_diagnostics.append({
                    "iter": iteration, "map_obj": elbo, "ll": 0.0,
                    "K_active": n_active_now, "n_pruned": 0, "converged": False,
                })

        # --- Post-hoc: active components from expected stick-breaking weights ---
        # E[π_k] is EXACT under mean-field q(β) = ∏_k Beta(a_k, b_k)
        E_pi   = self._stick_expected_pi(a_k, b_k)  # (T,)
        active = E_pi > self.weight_threshold        # (T,) boolean, no pruning during EM
        pi_k   = E_pi                                # used downstream for merging

        out = {
            "pi_k": pi_k, "m_k": m_k, "c_k": c_k,
            "active": active, "resp": resp,
            # "map_obj" stores the ELBO in the vbem path (name kept for interface compatibility)
            "map_obj": elbo,
            "n_iter": n_iter, "converged": converged,
        }
        if collect_diagnostics:
            out["iter_diagnostics"] = iter_diagnostics
        return out


    def _vbem_single_full(self, means, variances, seed, collect_diagnostics=False):
        """Full D×D cluster covariance DP-VBEM — not yet implemented.

        The diagonal path (``_vbem_single``) is the recommended route.
        Full-covariance DP-VBEM requires a reparameterization of the (D×D)
        covariance (e.g., via Cholesky factor) and a corresponding matrix
        gradient step, which introduces additional numerical complexity.
        This is left as a future extension rather than implemented partially.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Full-covariance DP-VBEM is not implemented. "
            "Use het_covariance_type='diag' with algorithm='vbem'."
        )
