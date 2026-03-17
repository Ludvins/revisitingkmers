import numpy as np
from sklearn.preprocessing import normalize
from features.kmer import KmerFeatureExtractor
from embedders.base import BaseEmbedder, EmbeddingResult


class KmerProfileEmbedder(BaseEmbedder):
    """Uses raw (normalized) k-mer frequency profiles as embeddings.

    No learned transformation — the k-mer profile itself is the embedding.
    Supports configurable k, alphabet, and normalization.
    """

    def __init__(self, k: int = 4, alphabet: list[str] = None,
                 norm: str = "l1"):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        self._k = k
        self._norm = norm
        self._feature_extractor = KmerFeatureExtractor(
            k=k, alphabet=alphabet, normalized=False
        )

    def embed(self, sequences: list[str]) -> EmbeddingResult:
        if not isinstance(sequences, list) or not all(isinstance(s, str) for s in sequences):
            raise TypeError("sequences must be a list of strings")
        if len(sequences) == 0:
            raise ValueError("sequences must be a non-empty list")

        profiles = self._feature_extractor.extract_batch(sequences)
        profiles = normalize(profiles, norm=self._norm)
        return EmbeddingResult(mean=profiles)

    def save(self, path: str) -> None:
        """No-op: KmerProfileEmbedder has no learnable parameters to save."""

    @classmethod
    def load(cls, path: str, device: str = "cpu", k: int = 4,
             norm: str = "l1", **kwargs) -> "KmerProfileEmbedder":
        return cls(k=k, norm=norm)

    @property
    def default_metric(self) -> str:
        return "l1"
