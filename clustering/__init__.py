"""Clustering module with a registry for swappable algorithms."""

from clustering.base import BaseClusterer
from clustering.kmedoid import KMedoidClusterer
from clustering.greedy_kmedoid import GreedyKMedoidClusterer
from clustering.dpgmm import DPGMMClusterer
from clustering.vmf_mixture import VMFMixtureClusterer

_REGISTRY: dict[str, type[BaseClusterer]] = {
    "kmedoid": KMedoidClusterer,
    "greedy_kmedoid": GreedyKMedoidClusterer,
    "dpgmm": DPGMMClusterer,
    "vmf_mixture": VMFMixtureClusterer,
}


def register_clusterer(name: str):
    """Decorator to register a clustering algorithm by name.

    Parameters
    ----------
    name : str
        Key under which the clusterer class is stored in the registry.

    Returns
    -------
    callable
        Class decorator that adds the class to ``_REGISTRY``.
    """
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_clusterer(name: str, **kwargs) -> BaseClusterer:
    """Instantiate a clusterer by registered name.

    Parameters
    ----------
    name : str
        Registered clusterer name (e.g., "kmedoid").
    **kwargs
        Constructor arguments for the clusterer.

    Returns
    -------
    BaseClusterer
        An instance of the requested clustering algorithm.

    Raises
    ------
    ValueError
        If ``name`` is not found in the registry.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown clusterer: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)
