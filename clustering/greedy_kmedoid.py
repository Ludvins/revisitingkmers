"""Greedy K-Medoid clustering replicating the original paper's algorithm.

Iteratively selects the point with the highest total similarity as the seed
medoid, refines it, assigns nearby points, and repeats.  For probabilistic
embeddings the medoid variance is hardcoded to 1.0 (matching the original
paper's approach) so that similarity is computed as:

    sim = exp(-0.5 * sum( delta^2 / (1.0 + var_point) ))
"""

import logging
import numpy as np
from clustering.base import BaseClusterer

logger = logging.getLogger(__name__)
from embedders.base import EmbeddingResult
from metrics.similarity import (
    compute_pairwise_similarity,
    compute_pairwise_probabilistic_similarity,
    compute_pairwise_vmf_similarity,
    compute_similarity,
    compute_probabilistic_similarity,
    compute_vmf_similarity,
)


class GreedyKMedoidClusterer(BaseClusterer):
    """Greedy K-Medoid from the original paper with variance=1 medoids.

    Algorithm
    ---------
    1. Compute full N×N pairwise similarity matrix.
    2. Zero out entries below ``min_similarity``.
    3. Repeat until all points assigned or ``max_iter`` reached:
       a. Pick seed = argmax(row_sum) among unassigned points.
       b. Set medoid = (mean of seed, **variance = 1.0**).
       c. Refine ``n_refine`` times:
          - Compute medoid-to-all similarity (variance=1 for medoid).
          - Select unassigned points with similarity >= ``min_similarity``.
          - Update medoid mean = mean of selected; variance stays 1.0.
       d. Assign selected points to new cluster.
       e. Subtract their columns from row_sum.
    4. Filter clusters smaller than ``min_bin_size`` → label = -1.
    """

    def __init__(self, metric: str = "l2", min_bin_size: int = 10,
                 max_iter: int = 1000, scale: float = 1.0,
                 n_refine: int = 3,
                 k_form: str = "adaptive", alpha: float = 1.0,
                 cache_dir: str = None, scalable: bool = False):
        self.metric = metric
        self.min_bin_size = min_bin_size
        self.max_iter = max_iter
        self.scale = scale
        self.n_refine = n_refine
        self.k_form = k_form
        self.alpha = alpha
        self.cache_dir = cache_dir
        self.scalable = scalable

    # ------------------------------------------------------------------
    # Medoid-to-points similarity (variance=1 for medoid)
    # ------------------------------------------------------------------
    def _medoid_similarity(self, embedding_result, medoid_mean, medoid_var):
        """Compute similarity of all points to a single medoid."""
        means = embedding_result.mean.astype(np.float32)

        if embedding_result.is_gaussian:
            variances = embedding_result.variance.astype(np.float32)
            if variances.ndim == 3:
                variances = np.diagonal(variances, axis1=1, axis2=2)
            return compute_probabilistic_similarity(
                means, variances,
                medoid_mean.astype(np.float32).squeeze(),
                medoid_var.astype(np.float32).squeeze(),
                scale=self.scale, k_form=self.k_form, alpha=self.alpha,
            )
        elif embedding_result.is_vmf:
            return compute_vmf_similarity(
                means, embedding_result.kappa,
                medoid_mean.squeeze(),
                float(np.mean(embedding_result.kappa)),
                scale=self.scale, k_form=self.k_form, alpha=self.alpha,
            )
        else:
            return compute_similarity(
                means, medoid_mean.squeeze(),
                metric=self.metric, scale=self.scale,
            )

    # ------------------------------------------------------------------
    # fit_predict
    # ------------------------------------------------------------------
    def fit_predict(self, embedding_result: EmbeddingResult,
                    min_similarity: float = 0.8, **kwargs) -> np.ndarray:
        means = embedding_result.mean.astype(np.float32)
        N, D = means.shape

        # Step 1: Full pairwise similarity
        if embedding_result.is_vmf:
            similarities = compute_pairwise_vmf_similarity(
                means, embedding_result.kappa,
                cache_dir=self.cache_dir,
                scale=self.scale, k_form=self.k_form, alpha=self.alpha,
            )
        elif embedding_result.is_gaussian:
            variances = embedding_result.variance.astype(np.float32)
            if variances.ndim == 3:
                variances = np.diagonal(variances, axis1=1, axis2=2)
            similarities = compute_pairwise_probabilistic_similarity(
                means, variances, cache_dir=self.cache_dir,
                scale=self.scale, k_form=self.k_form, alpha=self.alpha,
            )
        else:
            similarities = compute_pairwise_similarity(
                means, self.metric, scalable=self.scalable,
                cache_dir=self.cache_dir, scale=self.scale,
            )

        # Step 2: Zero out below threshold
        similarities[similarities < min_similarity] = 0

        # Labels: -1 = unassigned
        labels = -np.ones(N, dtype=int)
        row_sum = np.sum(similarities, axis=1)

        cluster_id = 0
        for _ in range(self.max_iter):
            if not np.any(labels == -1):
                break

            # Step 3a: Pick seed
            seed = int(np.argmax(row_sum))
            if row_sum[seed] <= 0:
                break

            # Step 3b: Initialize medoid with variance=1.0
            medoid_mean = means[seed:seed + 1].copy()  # (1, D)
            medoid_var = np.ones_like(medoid_mean)       # variance = 1.0

            selected_idx = None

            # Step 3c: Refine
            for _ in range(self.n_refine):
                sim_to_medoid = self._medoid_similarity(
                    embedding_result, medoid_mean, medoid_var)

                within = sim_to_medoid >= min_similarity
                available = labels == -1
                selected_idx = np.where(within & available)[0]

                if len(selected_idx) == 0:
                    break

                # Update medoid mean; keep variance = 1.0
                medoid_mean = means[selected_idx].mean(axis=0, keepdims=True)
                medoid_var = np.ones_like(medoid_mean)

            # Step 3d: Assign
            if selected_idx is not None and len(selected_idx) > 0:
                labels[selected_idx] = cluster_id
                # Step 3e: Update row sums
                row_sum -= np.sum(similarities[:, selected_idx], axis=1)
                row_sum[selected_idx] = 0
                cluster_id += 1

        # Step 4: Filter small clusters
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if label != -1 and count < self.min_bin_size:
                labels[labels == label] = -1

        n_clusters = len(set(labels[labels != -1].tolist())) if (labels != -1).any() else 0
        logger.debug("GreedyKMedoid: %d iterations, %d clusters (after min_bin_size=%d)",
                     cluster_id, n_clusters, self.min_bin_size)

        return labels
