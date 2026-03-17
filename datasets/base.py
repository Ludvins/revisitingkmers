from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseContrastiveDataset(torch.utils.data.Dataset, ABC):
    """Base class for contrastive learning datasets.

    Returns (anchor_features, positive_features, labels) where labels
    include both the positive pair (label=1) and negative samples (label=0).
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (left_features, right_features, labels).

        left_features: (1 + neg_samples, feature_dim) — anchor + negative left samples
        right_features: (1 + neg_samples, feature_dim) — positive + negative right samples
        labels: (1 + neg_samples,) — 1 for positive pair, 0 for negatives
        """
        ...
