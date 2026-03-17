import numpy as np
from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    """Transforms raw input into a numeric feature vector."""

    @abstractmethod
    def extract(self, item) -> np.ndarray:
        """Extract features from a single item."""
        ...

    def extract_batch(self, items) -> np.ndarray:
        """Extract features from multiple items. Override for efficiency."""
        return np.array([self.extract(item) for item in items])

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the output feature vector."""
        ...
