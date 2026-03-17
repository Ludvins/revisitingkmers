"""Abstract base class defining the clustering interface."""

import numpy as np
from abc import ABC, abstractmethod
from embedders.base import EmbeddingResult


class BaseClusterer(ABC):
    """Base class for clustering algorithms."""

    @abstractmethod
    def fit_predict(self, embedding_result: EmbeddingResult, **kwargs) -> np.ndarray:
        """Assign cluster labels to embedded data.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Output from any BaseEmbedder. Probabilistic-aware
            clusterers can use embedding_result.variance when available;
            others should use embedding_result.point_estimate.
        **kwargs
            Algorithm-specific parameters.

        Returns
        -------
        np.ndarray
            (N,) array of integer cluster labels. -1 means unassigned.
        """
        ...
