import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingResult:
    """Unified embedding output for deterministic and probabilistic models.

    Supports three distribution types:
    - ``"point"``: Deterministic embedding (only ``mean`` set).
    - ``"gaussian"``: Gaussian embedding (``mean`` + ``variance``).
    - ``"vmf"``: Von Mises-Fisher embedding (``mean`` + ``kappa``).

    For backward compatibility, the distribution type is auto-detected
    from which fields are set when not explicitly specified.
    """
    mean: np.ndarray                       # (N, D) — always present
    variance: Optional[np.ndarray] = None  # (N, D) diagonal or (N, D, D) full cov — Gaussian only
    kappa: Optional[np.ndarray] = None     # (N,) concentration — vMF only
    distribution: str = "point"            # "point", "gaussian", "vmf"

    def __post_init__(self):
        # Auto-detect distribution type for backward compatibility
        if self.distribution == "point":
            if self.variance is not None:
                self.distribution = "gaussian"
            elif self.kappa is not None:
                self.distribution = "vmf"

    @property
    def is_probabilistic(self) -> bool:
        return self.distribution != "point"

    @property
    def is_gaussian(self) -> bool:
        return self.distribution == "gaussian"

    @property
    def is_vmf(self) -> bool:
        return self.distribution == "vmf"

    @property
    def point_estimate(self) -> np.ndarray:
        return self.mean


class BaseEmbedder(ABC):
    """Base class for all embedding models."""

    @abstractmethod
    def embed(self, inputs) -> EmbeddingResult:
        """Compute embeddings for inputs.

        The type of `inputs` depends on the concrete embedder:
        DNA embedders accept list[str], image embedders accept tensors, etc.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str = "cpu") -> "BaseEmbedder":
        ...

    @property
    def default_metric(self) -> str:
        """Distance metric this embedder's outputs are designed for."""
        return "l2"
