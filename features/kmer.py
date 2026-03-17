import itertools
import os
import numpy as np
from features.base import BaseFeatureExtractor
from utils.progress import pbar


class KmerFeatureExtractor(BaseFeatureExtractor):
    """Extracts normalized k-mer frequency profiles from sequences.

    Configurable alphabet (defaults to DNA: ACGT), k-mer size, and normalization.
    Consolidates the k-mer extraction logic previously duplicated across embedders.
    """

    def __init__(self, k: int = 4, alphabet: list[str] = None, normalized: bool = True):
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        if alphabet is not None:
            if not isinstance(alphabet, list) or len(alphabet) == 0:
                raise ValueError(f"alphabet must be a non-empty list of strings, got {alphabet!r}")
            if not all(isinstance(c, str) for c in alphabet):
                raise TypeError(f"All alphabet elements must be strings, got types: {[type(c).__name__ for c in alphabet]}")

        self.k = k
        self.alphabet = alphabet or ["A", "C", "G", "T"]
        self.normalized = normalized
        self._kmer2id = {
            "".join(kmer): idx
            for idx, kmer in enumerate(itertools.product(self.alphabet, repeat=k))
        }

    @property
    def feature_dim(self) -> int:
        return len(self.alphabet) ** self.k

    def extract(self, sequence: str) -> np.ndarray:
        """Extract k-mer frequency profile from a single sequence.

        Parameters
        ----------
        sequence : str
            Input sequence string.

        Returns
        -------
        np.ndarray
            Normalized (or raw) k-mer frequency vector of length ``feature_dim``.
        """
        if not isinstance(sequence, str):
            raise TypeError(f"sequence must be a string, got {type(sequence).__name__}")
        if len(sequence) == 0:
            raise ValueError("sequence must not be empty")

        profile = np.zeros(self.feature_dim)
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            if kmer in self._kmer2id:
                profile[self._kmer2id[kmer]] += 1
        if self.normalized and profile.sum() > 0:
            profile = profile / profile.sum()
        return profile

    def extract_batch(self, sequences: list[str], verbose: bool = False,
                      cache_path: str = None) -> np.ndarray:
        """Extract k-mer profiles for a batch of sequences.

        Parameters
        ----------
        sequences : list[str]
            List of input sequences.
        verbose : bool
            Show progress bar.
        cache_path : str
            If provided, cache the extracted profiles to this .npy
            file. On subsequent calls with the same cache_path, the cached
            profiles are returned instantly if the cache file exists.
        """
        if not isinstance(sequences, list) or len(sequences) == 0:
            raise ValueError(f"sequences must be a non-empty list, got {type(sequences).__name__} with length {len(sequences) if isinstance(sequences, list) else 'N/A'}")
        if not isinstance(sequences[0], str):
            raise TypeError(f"Each element of sequences must be a string, got {type(sequences[0]).__name__} for first element")

        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f" K-mer profiles loaded from cache: {cache_path}")
            return np.load(cache_path)

        profiles = np.array([self.extract(seq) for seq in
                             pbar(sequences, desc="Extracting k-mer profiles",
                                  unit="seq", disable=not verbose)])

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, profiles)
            if verbose:
                print(f" K-mer profiles cached to: {cache_path}")

        return profiles

    def extract_windowed(self, sequence: str, window_size: int = 500) -> np.ndarray:
        """Extract k-mer profiles for non-overlapping windows along a sequence.

        Parameters
        ----------
        sequence : str
            Input DNA sequence.
        window_size : int
            Number of base pairs per window.

        Returns
        -------
        np.ndarray
            Shape ``(num_windows, feature_dim)``. Sequences shorter than
            *window_size* produce a single window covering the whole sequence.
        """
        if not isinstance(sequence, str):
            raise TypeError(f"sequence must be a string, got {type(sequence).__name__}")
        if len(sequence) == 0:
            raise ValueError("sequence must not be empty")
        if window_size <= self.k:
            raise ValueError(f"window_size must be > k ({self.k}), got {window_size}")

        seq_len = len(sequence)
        if seq_len <= window_size:
            return self.extract(sequence).reshape(1, -1)

        windows = []
        for start in range(0, seq_len - window_size + 1, window_size):
            windows.append(self.extract(sequence[start:start + window_size]))

        # If sequence doesn't divide evenly, add a final window from the tail
        remainder = seq_len % window_size
        if remainder > self.k:
            windows.append(self.extract(sequence[-remainder:]))

        return np.stack(windows)

    def extract_windowed_batch(self, sequences: list[str], window_size: int = 500,
                               num_windows: int = None, verbose: bool = False,
                               cache_path: str = None) -> np.ndarray:
        """Extract windowed k-mer profiles for a batch of sequences.

        Parameters
        ----------
        sequences : list[str]
            Input sequences.
        window_size : int
            Base pairs per window.
        num_windows : int, optional
            If set, pad or truncate all outputs to exactly this many windows.
            Required when sequences have different lengths.
        verbose : bool
            Show progress bar.
        cache_path : str, optional
            Disk cache path (.npy).

        Returns
        -------
        np.ndarray
            Shape ``(N, W, feature_dim)`` where W is determined by the
            sequences or *num_windows*.
        """
        if not isinstance(sequences, list) or len(sequences) == 0:
            raise ValueError("sequences must be a non-empty list")

        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f" Windowed k-mer profiles loaded from cache: {cache_path}")
            return np.load(cache_path)

        profiles = []
        for seq in pbar(sequences, desc="Extracting windowed k-mer profiles",
                        unit="seq", disable=not verbose):
            prof = self.extract_windowed(seq, window_size=window_size)
            profiles.append(prof)

        if num_windows is not None:
            padded = []
            for prof in profiles:
                w = prof.shape[0]
                if w >= num_windows:
                    padded.append(prof[:num_windows])
                else:
                    pad = np.zeros((num_windows - w, self.feature_dim))
                    padded.append(np.concatenate([prof, pad], axis=0))
            result = np.stack(padded)
        else:
            # All profiles must have same number of windows
            result = np.stack(profiles)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, result)
            if verbose:
                print(f" Windowed k-mer profiles cached to: {cache_path}")

        return result

    @property
    def kmer2id(self) -> dict[str, int]:
        return self._kmer2id
