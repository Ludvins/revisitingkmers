import hashlib
import os
import numpy as np
from scipy.spatial import distance
from scipy.special import ive as _ive
from utils.progress import pbar


def _similarity_cache_key(features: np.ndarray, metric: str) -> str:
    """Build a hash key from features shape, content digest, and metric."""
    h = hashlib.sha256()
    h.update(f"{features.shape}|{metric}".encode())
    h.update(features.tobytes()[:8192])  # hash first 8KB for speed
    h.update(features[-1].tobytes())     # also hash last row for uniqueness
    return h.hexdigest()[:16]


def compute_pairwise_similarity(features: np.ndarray, metric: str = "l2",
                                 scalable: bool = False,
                                 cache_dir: str = None,
                                 scale: float = 1.0) -> np.ndarray:
    """Compute pairwise similarity matrix between all feature vectors.

    Parameters
    ----------
    features : np.ndarray
        (N, D) array of feature vectors.
    metric : str
        One of "dot", "l2"/"euclidean", "l1".
    scalable : bool
        If True and metric is "dot", compute in chunks to save memory.
    cache_dir : str, optional
        If provided, cache the similarity matrix to this directory.
        The cache file is keyed by a hash of the features and metric.
    scale : float
        Coefficient applied to squared distance before exponentiation.
        For L2: sim = exp(-scale * d^2). Ignored for "dot" metric.

    Returns
    -------
    np.ndarray
        (N, N) similarity matrix.
    """
    if not isinstance(features, np.ndarray):
        raise TypeError(
            f"features must be a numpy array, got {type(features).__name__}"
        )
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D, got shape {features.shape}"
        )
    valid_metrics = {"dot", "l2", "euclidean", "l1"}
    if metric not in valid_metrics:
        raise ValueError(
            f"metric must be one of {valid_metrics}, got '{metric}'"
        )

    features = features.astype(np.float32)

    # Check disk cache
    if cache_dir:
        cache_key = _similarity_cache_key(features, metric)
        cache_path = os.path.join(cache_dir, f"similarity_{cache_key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    if metric == "dot":
        if scalable:
            n = features.shape[0]
            similarities = np.zeros((n, n), dtype=np.float32)
            for i in pbar(range(0, n, 512), desc="Computing pairwise similarity",
                          unit="chunk"):
                similarities[i:i + 512, :] = features[i:i + 512, :] @ features.T
        else:
            similarities = np.dot(features, features.T)
    elif metric in ("euclidean", "l2"):
        d2 = distance.squareform(distance.pdist(features, "sqeuclidean"))
        similarities = np.exp(-scale * d2)
    elif metric == "l1":
        d = distance.squareform(distance.pdist(features, "minkowski", p=1.0))
        similarities = np.exp(-scale * d)

    # Save to disk cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, similarities)

    return similarities


def compute_similarity(features: np.ndarray, query: np.ndarray,
                        metric: str = "l2", scale: float = 1.0) -> np.ndarray:
    """Compute similarity of all feature vectors against a single query.

    Parameters
    ----------
    features : np.ndarray
        (N, D) array of feature vectors.
    query : np.ndarray
        (D,) query vector.
    metric : str
        One of "dot", "l2"/"euclidean", "l1".
    scale : float
        Coefficient applied to squared distance before exponentiation.
        For L2: sim = exp(-scale * d^2). Ignored for "dot" metric.

    Returns
    -------
    np.ndarray
        (N,) similarity scores.
    """
    if not isinstance(features, np.ndarray):
        raise TypeError(
            f"features must be a numpy array, got {type(features).__name__}"
        )
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D, got shape {features.shape}"
        )
    if not isinstance(query, np.ndarray):
        raise TypeError(
            f"query must be a numpy array, got {type(query).__name__}"
        )
    if query.ndim != 1:
        raise ValueError(
            f"query must be 1D, got shape {query.shape}"
        )
    if features.shape[1] != query.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: features has {features.shape[1]} "
            f"columns but query has {query.shape[0]} elements"
        )
    valid_metrics = {"dot", "l2", "euclidean", "l1"}
    if metric not in valid_metrics:
        raise ValueError(
            f"metric must be one of {valid_metrics}, got '{metric}'"
        )

    if metric == "dot":
        return np.dot(features, query)

    if metric in ("euclidean", "l2"):
        d2 = distance.cdist(features, query.reshape(1, -1), "sqeuclidean")
        return np.exp(-scale * d2).squeeze()

    if metric == "l1":
        d = distance.cdist(features, query.reshape(1, -1), "minkowski", p=1.0)
        return np.exp(-scale * d).squeeze()


def _probabilistic_cache_key(means: np.ndarray, variances: np.ndarray,
                              scale: float = 0.25) -> str:
    """Build a hash key from means, variances, and scale for probabilistic similarity."""
    h = hashlib.sha256()
    h.update(f"{means.shape}|probabilistic|{scale}".encode())
    h.update(means.tobytes()[:8192])
    h.update(means[-1].tobytes())
    h.update(variances.tobytes()[:8192])
    h.update(variances[-1].tobytes())
    return h.hexdigest()[:16]


def compute_pairwise_probabilistic_similarity(
    means: np.ndarray, variances: np.ndarray,
    chunk_size: int = 512, cache_dir: str = None,
    scale: float = 0.25, k_form: str = "adaptive",
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute pairwise probabilistic similarity (paper Eq. 5).

    **k_form="adaptive"** (default):
        sim(i,j) = exp( -scale * sum_d delta_d^2 / (var_i,d + var_j,d + alpha) )

    **k_form="identity"** (K = αI):
        sim(i,j) = exp( -scale * sum_d [0.5*log(v_d/α+1) + 0.5*delta_d^2/(v_d+α)] )
        where v_d = var_i,d + var_j,d.  α=1 is the original form;
        smaller α = tighter metric.

    Parameters
    ----------
    means : np.ndarray
        (N, D) array of mean embeddings.
    variances : np.ndarray
        (N, D) array of diagonal variances.
    chunk_size : int
        Block size for double-chunked computation to limit memory.
    cache_dir : str, optional
        If provided, cache the similarity matrix to this directory.
    scale : float
        Coefficient applied to the distance before exponentiation.
    k_form : str
        Kernel form: ``"adaptive"``, ``"identity"``, or ``"expected_distance"``.
    alpha : float
        For adaptive form: regularizer added to combined variance; large
        alpha pushes toward uniform (Euclidean-like) weighting (default 1.0).
        For identity form: diagonal of K = αI (default 1.0 = original form).

    Returns
    -------
    np.ndarray
        (N, N) similarity matrix.
    """
    valid_k_forms = {"adaptive", "identity", "expected_distance"}
    if k_form not in valid_k_forms:
        raise ValueError(f"k_form must be one of {valid_k_forms}, got '{k_form}'")

    means = means.astype(np.float32)
    variances = variances.astype(np.float32)
    N = means.shape[0]

    # Check disk cache
    if cache_dir:
        cache_key = _probabilistic_cache_key(means, variances, scale)
        cache_path = os.path.join(cache_dir, f"similarity_prob_{cache_key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    similarities = np.zeros((N, N), dtype=np.float32)
    n_chunks = (N + chunk_size - 1) // chunk_size

    for i in pbar(range(0, N, chunk_size),
                  desc="Computing probabilistic similarity", unit="chunk",
                  total=n_chunks):
        ei = min(i + chunk_size, N)
        for j in range(0, N, chunk_size):
            ej = min(j + chunk_size, N)
            # (Ci, 1, D) - (1, Cj, D) -> (Ci, Cj, D)
            delta = means[i:ei, None, :] - means[None, j:ej, :]
            cvar = variances[i:ei, None, :] + variances[None, j:ej, :]

            if k_form == "identity":
                scale_d = cvar + alpha
                dist = np.sum(0.5 * np.log(scale_d / alpha) + 0.5 * delta * delta / scale_d,
                              axis=2)
            elif k_form == "expected_distance":
                dist = np.sum(delta * delta, axis=2) + np.sum(cvar, axis=2)
            else:
                dist = np.sum(delta * delta / (cvar + alpha), axis=2)

            similarities[i:ei, j:ej] = np.exp(-scale * dist)

    # Save to disk cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, similarities)

    return similarities


def compute_probabilistic_similarity(
    means: np.ndarray, variances: np.ndarray,
    query_mean: np.ndarray, query_variance: np.ndarray,
    scale: float = 0.25, k_form: str = "adaptive",
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute probabilistic similarity against a single query (paper Eq. 5).

    **k_form="adaptive"**: sim = exp(-scale * sum delta^2/(var_i+var_q+α))
    **k_form="identity"** (K=αI): sim = exp(-scale * sum [0.5*log(v/α+1) + 0.5*delta^2/(v+α)])

    Parameters
    ----------
    means : np.ndarray
        (N, D) array of mean embeddings.
    variances : np.ndarray
        (N, D) array of diagonal variances.
    query_mean : np.ndarray
        (D,) query mean vector.
    query_variance : np.ndarray
        (D,) query variance vector.
    scale : float
        Coefficient applied to the distance (default 0.25).
    k_form : str
        Kernel form: ``"adaptive"``, ``"identity"``, or ``"expected_distance"``.
    alpha : float
        For identity form: diagonal of K = αI (default 1.0 = original form).

    Returns
    -------
    np.ndarray
        (N,) similarity scores.
    """
    delta = means - query_mean                       # (N, D)
    cvar = variances + query_variance                # (N, D)

    if k_form == "identity":
        scale_d = cvar + alpha
        dist = np.sum(0.5 * np.log(scale_d / alpha) + 0.5 * delta * delta / scale_d, axis=1)
    elif k_form == "expected_distance":
        dist = np.sum(delta * delta, axis=1) + np.sum(cvar, axis=1)
    else:
        dist = np.sum(delta * delta / (cvar + alpha), axis=1)

    return np.exp(-scale * dist)


# ---------------------------------------------------------------------------
# vMF (von Mises-Fisher) similarity
# ---------------------------------------------------------------------------

def _log_vmf_norm_const(kappa: np.ndarray, dim: int) -> np.ndarray:
    """Log normalization constant of the vMF distribution.

    log C_d(kappa) = (d/2 - 1) * log(kappa) - (d/2) * log(2*pi) - log I_{d/2-1}(kappa)

    Uses the exponentially-scaled Bessel function for numerical stability.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    v = dim / 2.0 - 1.0
    # ive(v, k) = I_v(k) * exp(-k), so log I_v(k) = log(ive(v,k)) + k
    log_ive = np.log(np.clip(_ive(v, kappa), 1e-300, None))
    log_bessel = log_ive + kappa
    return v * np.log(np.clip(kappa, 1e-300, None)) - (dim / 2.0) * np.log(2 * np.pi) - log_bessel


def _ppk_log_self_similarity(kappas: np.ndarray, dim: int) -> np.ndarray:
    """Log self-similarity for PPK: log K(i,i) = 2*log_C(ki) - log_C(2*ki).

    Used for Cauchy-Schwarz normalization so PPK values fall in [0, 1].
    """
    kappas = np.asarray(kappas, dtype=np.float64)
    log_c = _log_vmf_norm_const(kappas, dim)
    log_c_double = _log_vmf_norm_const(2.0 * kappas, dim)
    return 2.0 * log_c - log_c_double


def compute_pairwise_vmf_similarity(
    means: np.ndarray, kappas: np.ndarray,
    chunk_size: int = 512, cache_dir: str = None,
    scale: float = 1.0, k_form: str = "cosine",
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute pairwise similarity for vMF embeddings.

    **k_form="cosine"** (paper's approach):
        sim(i,j) = exp(-scale * (1 - mu_i^T mu_j))
        Pure direction similarity; kappa is ignored.

    **k_form="ppk"** (Product Probability Kernel):
        Cauchy-Schwarz normalized PPK:
        K_norm(i,j) = K(i,j) / sqrt(K(i,i) * K(j,j))
        sim = K_norm^scale
        Values in [0, 1] with self-similarity = 1.

    **k_form="adaptive"**:
        sim(i,j) = exp(-scale * (1 - mu_i^T mu_j) / (1/kappa_i + 1/kappa_j + alpha))
        Adapts the Gaussian adaptive formula using 1/kappa as variance proxy.

    Parameters
    ----------
    means : np.ndarray
        (N, D) array of L2-normalized mean directions.
    kappas : np.ndarray
        (N,) array of concentration parameters > 0.
    chunk_size : int
        Block size for chunked computation.
    cache_dir : str, optional
        Disk cache directory.
    scale : float
        Coefficient applied before exponentiation.
    k_form : str
        ``"cosine"``, ``"ppk"``, or ``"adaptive"``.
    alpha : float
        Regularizer for adaptive form.

    Returns
    -------
    np.ndarray
        (N, N) similarity matrix.
    """
    valid_k_forms = {"cosine", "cosine_direct", "ppk", "adaptive"}
    if k_form not in valid_k_forms:
        raise ValueError(f"k_form must be one of {valid_k_forms}, got '{k_form}'")

    means = means.astype(np.float32)
    kappas = kappas.astype(np.float64)
    N, D = means.shape

    # Disk cache
    if cache_dir:
        h = hashlib.sha256()
        h.update(f"{means.shape}|vmf|{k_form}|{scale}|{alpha}".encode())
        h.update(means.tobytes()[:8192])
        h.update(kappas.tobytes()[:4096])
        cache_key = h.hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"similarity_vmf_{cache_key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

    similarities = np.zeros((N, N), dtype=np.float32)
    n_chunks = (N + chunk_size - 1) // chunk_size

    # Precompute log norm constants and self-similarity for PPK
    if k_form == "ppk":
        log_c = _log_vmf_norm_const(kappas, D)  # (N,)
        log_self_sim = _ppk_log_self_similarity(kappas, D)  # (N,)

    for i in pbar(range(0, N, chunk_size),
                  desc="Computing vMF similarity", unit="chunk",
                  total=n_chunks):
        ei = min(i + chunk_size, N)
        for j in range(0, N, chunk_size):
            ej = min(j + chunk_size, N)

            # Cosine similarity: mu_i^T mu_j via matrix multiply
            cos_sim = means[i:ei] @ means[j:ej].T  # (Ci, Cj)

            if k_form == "cosine":
                d_cos = 1.0 - cos_sim
                similarities[i:ei, j:ej] = np.exp(-scale * d_cos)

            elif k_form == "cosine_direct":
                similarities[i:ei, j:ej] = np.clip(cos_sim, 0, 1)

            elif k_form == "ppk":
                # kappa_prod = ||kappa_i * mu_i + kappa_j * mu_j||
                # Expand: ||k_i*m_i + k_j*m_j||^2 = k_i^2 + k_j^2 + 2*k_i*k_j*(m_i^T m_j)
                ki = kappas[i:ei, None]   # (Ci, 1)
                kj = kappas[None, j:ej]   # (1, Cj)
                kappa_prod_sq = ki**2 + kj**2 + 2 * ki * kj * cos_sim
                kappa_prod = np.sqrt(np.clip(kappa_prod_sq, 1e-12, None))

                log_c_i = log_c[i:ei, None]  # (Ci, 1)
                log_c_j = log_c[None, j:ej]  # (1, Cj)
                log_c_prod = _log_vmf_norm_const(kappa_prod, D)

                log_sim = log_c_i + log_c_j - log_c_prod
                # Cauchy-Schwarz normalization: K_norm = K / sqrt(K_ii * K_jj)
                log_self_i = log_self_sim[i:ei, None]  # (Ci, 1)
                log_self_j = log_self_sim[None, j:ej]  # (1, Cj)
                log_sim_norm = log_sim - 0.5 * (log_self_i + log_self_j)
                # log_sim_norm <= 0, with 0 for identical distributions
                similarities[i:ei, j:ej] = np.exp(scale * log_sim_norm).astype(np.float32)

            elif k_form == "adaptive":
                d_cos = 1.0 - cos_sim
                inv_ki = 1.0 / np.clip(kappas[i:ei, None], 1e-6, None)
                inv_kj = 1.0 / np.clip(kappas[None, j:ej], 1e-6, None)
                denom = inv_ki + inv_kj + alpha
                similarities[i:ei, j:ej] = np.exp(-scale * d_cos / denom)

    # Disk cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, similarities)

    return similarities


def compute_vmf_similarity(
    means: np.ndarray, kappas: np.ndarray,
    query_mean: np.ndarray, query_kappa: float,
    scale: float = 1.0, k_form: str = "cosine",
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute vMF similarity of all embeddings against a single query.

    Parameters
    ----------
    means : np.ndarray
        (N, D) array of L2-normalized mean directions.
    kappas : np.ndarray
        (N,) concentration parameters.
    query_mean : np.ndarray
        (D,) query direction (L2-normalized).
    query_kappa : float
        Query concentration.
    scale : float
        Coefficient applied before exponentiation.
    k_form : str
        ``"cosine"``, ``"ppk"``, or ``"adaptive"``.
    alpha : float
        Regularizer for adaptive form.

    Returns
    -------
    np.ndarray
        (N,) similarity scores.
    """
    cos_sim = means @ query_mean  # (N,)
    D = means.shape[1]

    if k_form == "cosine":
        return np.exp(-scale * (1.0 - cos_sim))

    elif k_form == "cosine_direct":
        return np.clip(cos_sim, 0.0, 1.0)

    elif k_form == "ppk":
        ki = kappas.astype(np.float64)
        kj = float(query_kappa)
        kappa_prod_sq = ki**2 + kj**2 + 2 * ki * kj * cos_sim
        kappa_prod = np.sqrt(np.clip(kappa_prod_sq, 1e-12, None))
        log_c_i = _log_vmf_norm_const(ki, D)
        log_c_j = _log_vmf_norm_const(np.array([kj]), D)[0]
        log_c_prod = _log_vmf_norm_const(kappa_prod, D)
        log_sim = log_c_i + log_c_j - log_c_prod
        # Cauchy-Schwarz normalization: K_norm = K / sqrt(K_ii * K_jj)
        log_self_i = _ppk_log_self_similarity(ki, D)
        log_self_j = _ppk_log_self_similarity(np.array([kj]), D)[0]
        log_sim_norm = log_sim - 0.5 * (log_self_i + log_self_j)
        return np.exp(scale * log_sim_norm).astype(np.float32)

    elif k_form == "adaptive":
        d_cos = 1.0 - cos_sim
        inv_ki = 1.0 / np.clip(kappas, 1e-6, None)
        inv_kj = 1.0 / max(query_kappa, 1e-6)
        denom = inv_ki + inv_kj + alpha
        return np.exp(-scale * d_cos / denom)

    else:
        raise ValueError(f"k_form must be 'cosine', 'cosine_direct', 'ppk', or 'adaptive', got '{k_form}'")
