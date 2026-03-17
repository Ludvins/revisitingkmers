"""K-Medoids clustering via sklearn_extra with automatic k estimation.

Wraps sklearn_extra.cluster.KMedoids with precomputed distance matrices.
For probabilistic embeddings, uses Mahalanobis distance (paper Eq. 5).
The number of clusters is estimated from the similarity threshold by
counting connected components in the thresholded similarity graph.
"""

import logging
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn_extra.cluster import KMedoids

logger = logging.getLogger(__name__)
from clustering.base import BaseClusterer
from embedders.base import EmbeddingResult
from metrics.similarity import (
    compute_pairwise_similarity,
    compute_pairwise_probabilistic_similarity,
    compute_pairwise_vmf_similarity,
)


class KMedoidClusterer(BaseClusterer):
    """K-Medoids clustering with automatic k estimation from similarity threshold.

    For probabilistic embeddings (variance available), computes pairwise
    Mahalanobis distance. For deterministic embeddings, uses the specified
    metric. The number of clusters is estimated by counting connected
    components in the similarity graph thresholded at min_similarity.
    """

    def __init__(self, metric: str = "l2", min_bin_size: int = 10,
                 max_iter: int = 300, method: str = "alternate",
                 scalable: bool = False,
                 cache_dir: str = None, scale: float = 1.0,
                 k_form: str = "adaptive", alpha: float = 1.0,
                 **extra_kwargs):
        valid_metrics = {"dot", "l2", "euclidean", "l1"}
        if metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{metric}'"
            )
        valid_methods = {"pam", "alternate"}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )
        valid_k_forms = {"adaptive", "identity", "expected_distance", "cosine", "cosine_direct", "ppk"}
        if k_form not in valid_k_forms:
            raise ValueError(
                f"k_form must be one of {valid_k_forms}, got '{k_form}'"
            )
        self.metric = metric
        self.min_bin_size = min_bin_size
        self.max_iter = max_iter
        self.method = method
        self.scalable = scalable
        self.cache_dir = cache_dir
        self.scale = scale
        self.k_form = k_form
        self.alpha = alpha

    def fit_predict(self, embedding_result: EmbeddingResult,
                    min_similarity: float = 0.8, **kwargs) -> np.ndarray:
        """Cluster embeddings using PAM k-medoids with automatic k estimation.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Embedding output. When variance is set, probabilistic
            Mahalanobis similarity is used automatically.
        min_similarity : float
            Similarity threshold for k estimation and post-filtering.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Integer cluster labels of shape (N,). -1 = unassigned.
        """
        means = embedding_result.mean.astype(np.float32)

        # Step 1: Compute pairwise similarity
        if embedding_result.is_vmf:
            similarities = compute_pairwise_vmf_similarity(
                means, embedding_result.kappa,
                cache_dir=self.cache_dir,
                scale=self.scale, k_form=self.k_form,
                alpha=self.alpha,
            )
        elif embedding_result.is_gaussian:
            variances = embedding_result.variance.astype(np.float32)
            if variances.ndim == 3:
                variances = np.diagonal(variances, axis1=1, axis2=2)
            similarities = compute_pairwise_probabilistic_similarity(
                means, variances, cache_dir=self.cache_dir,
                scale=self.scale, k_form=self.k_form,
                alpha=self.alpha,
            )
        else:
            similarities = compute_pairwise_similarity(
                means, self.metric, scalable=self.scalable,
                cache_dir=self.cache_dir, scale=self.scale,
            )

        # Step 2: Estimate k from connected components of thresholded graph
        adjacency = (similarities >= min_similarity).astype(np.int8)
        np.fill_diagonal(adjacency, 0)
        n_components, component_labels = connected_components(
            adjacency, directed=False
        )
        # Filter out singleton components (noise) for k estimation
        _, comp_counts = np.unique(component_labels, return_counts=True)
        k_est = max(1, int((comp_counts >= self.min_bin_size).sum()))

        N = len(means)
        if k_est >= N:
            k_est = max(1, N // self.min_bin_size)

        # Step 3: Convert similarity to distance and run KMedoids
        dist = 1.0 - np.clip(similarities, 0, 1)
        np.fill_diagonal(dist, 0)

        logger.debug("KMedoids: k_est=%d, n=%d, method=%s", k_est, N, self.method)
        kmed = KMedoids(
            n_clusters=k_est, metric="precomputed",
            method=self.method, max_iter=self.max_iter, random_state=0,
        )
        labels = kmed.fit_predict(dist)

        # Step 4: Discard small clusters
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if count < self.min_bin_size:
                labels[labels == label] = -1

        return labels
