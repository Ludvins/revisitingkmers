"""Embedder registry and disk-cached embedding helpers.

Use ``load_embedder(name, ...)`` to instantiate any registered embedder by name,
or ``get_embedding(embedder, sequences, cache_path=...)`` to embed sequences with
optional on-disk caching.  New embedders are registered via the ``@register(name)``
decorator in their respective module.
"""

import os
import numpy as np
from embedders.base import BaseEmbedder, EmbeddingResult

_REGISTRY: dict[str, type[BaseEmbedder]] = {}


def register(name: str):
    """Decorator to register an embedder class by name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def load_embedder(name: str, path: str = None, **kwargs) -> BaseEmbedder:
    """Load an embedder by registered name.

    Parameters
    ----------
    name : str
        Registered embedder name (e.g., "nonlinear", "kmerprofile").
    path : str, optional
        Path to saved model (if applicable).
    **kwargs
        Extra arguments passed to the embedder's load() method.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown embedder: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name].load(path, **kwargs)


_MEMORY_CACHE: dict[str, EmbeddingResult] = {}


def get_embedding(embedder: BaseEmbedder, sequences, cache_path: str = None) -> EmbeddingResult:
    """Compute embeddings, with optional disk + in-memory caching.

    For probabilistic embeddings, variance is cached alongside mean
    as a separate file with '_var.npy' suffix.

    When cache_path is provided, results are also kept in an in-memory
    cache so repeated requests within the same process avoid disk I/O.
    """
    # In-memory cache hit
    if cache_path and cache_path in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_path]

    # Disk cache hit
    if cache_path and os.path.exists(cache_path):
        mean = np.load(cache_path)
        var_path = cache_path.replace(".npy", "_var.npy")
        kappa_path = cache_path.replace(".npy", "_kappa.npy")
        variance = np.load(var_path) if os.path.exists(var_path) else None
        kappa = np.load(kappa_path) if os.path.exists(kappa_path) else None
        result = EmbeddingResult(mean=mean, variance=variance, kappa=kappa)
        _MEMORY_CACHE[cache_path] = result
        return result

    result = embedder.embed(sequences)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, result.mean)
        if result.variance is not None:
            np.save(cache_path.replace(".npy", "_var.npy"), result.variance)
        if result.kappa is not None:
            np.save(cache_path.replace(".npy", "_kappa.npy"), result.kappa)
        _MEMORY_CACHE[cache_path] = result

    return result


def clear_embedding_cache():
    """Clear the in-memory embedding cache."""
    _MEMORY_CACHE.clear()


# Register all built-in embedders
from embedders.nonlinear import NonLinearEmbedder
from embedders.kmer_profile import KmerProfileEmbedder
from embedders.llm import LLMEmbedder

from embedders.uncertaingen import UncertainGenEmbedder
from embedders.two_head_network import TwoHeadNetworkEmbedder
from embedders.pcl import PCLEmbedder

register("nonlinear")(NonLinearEmbedder)
register("uncertaingen")(UncertainGenEmbedder)
register("twoheadnetwork")(TwoHeadNetworkEmbedder)
register("pcl")(PCLEmbedder)
register("kmerprofile")(KmerProfileEmbedder)

# LLM-based embedders: register known models + generic "dnaberts"
register("llm")(LLMEmbedder)
register("dnaberts")(LLMEmbedder)

# Register specific LLM variants
from embedders.llm import KNOWN_LLM_MODELS
for _name, _config in KNOWN_LLM_MODELS.items():
    def _make_loader(_cfg):
        class _LLMVariant(LLMEmbedder):
            @classmethod
            def load(cls, path: str = None, device: str = "cpu", **kwargs) -> "LLMEmbedder":
                merged = {**_cfg, **kwargs}
                if path:
                    merged["model_name_or_path"] = path
                return cls(**merged)
        return _LLMVariant
    register(_name)(_make_loader(_config))
