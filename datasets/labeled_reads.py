"""Contrastive dataset from labeled DNA sequences in TSV format.

Reads a TSV file with columns (sequence, label). Positive pairs are drawn
from sequences with the same label; negatives are drawn from the full pool.

Returns the same (left, right, labels) contract as PairedReadsDataset
for use with train_contrastive().
"""
import csv
import random
import numpy as np
import torch
from typing import Tuple, Callable
from datasets.base import BaseContrastiveDataset
from utils import filter_sequences
from utils.progress import pbar


class LabeledReadsDataset(BaseContrastiveDataset):
    """Contrastive dataset built from labeled DNA sequences.

    Parameters
    ----------
    file_path : str
        Path to a TSV file with columns ``sequence`` and ``label``.
        First row is treated as a header and skipped.
    transform_func : callable
        Function mapping a DNA string to a 1-D feature array (e.g. k-mer profile).
    neg_sample_per_pos : int
        Number of negative samples per positive pair.
    max_samples : int
        Maximum number of sequences to load (0 = all).
    min_seq_len : int
        Minimum sequence length; shorter sequences are filtered out.
    max_seq_len : int
        Truncate sequences to this length.
    min_abundance : int
        Discard classes with fewer than this many samples.
    verbose : bool
        Print progress.
    seed : int
        Random seed.
    """

    def __init__(self, file_path: str, transform_func: Callable[[str], np.ndarray],
                 neg_sample_per_pos: int = 20, max_samples: int = 0,
                 min_seq_len: int = 0, max_seq_len: int = 20000,
                 min_abundance: int = 0,
                 verbose: bool = True, seed: int = 0):
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a string, got {type(file_path).__name__}")
        if not callable(transform_func):
            raise TypeError(f"transform_func must be callable, got {type(transform_func).__name__}")
        if neg_sample_per_pos <= 0:
            raise ValueError(f"neg_sample_per_pos must be positive, got {neg_sample_per_pos}")

        self._neg_sample_per_pos = neg_sample_per_pos
        self._seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        # Read TSV
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # skip header
            rows = list(pbar(reader, desc=f"Reading {file_path}", unit="row",
                             disable=not verbose))

        sequences = [r[0][:max_seq_len] for r in rows]
        labels = [r[1] for r in rows]

        sequences, labels = filter_sequences(sequences, labels,
                                             min_seq_len=min_seq_len,
                                             min_abundance=min_abundance)

        # Subsample
        if max_samples > 0 and max_samples < len(sequences):
            indices = sorted(random.sample(range(len(sequences)), max_samples))
            sequences = [sequences[i] for i in indices]
            labels = [labels[i] for i in indices]

        # Transform sequences to feature vectors
        features_list = []
        for seq in pbar(sequences, desc="Extracting features", unit="seq",
                        disable=not verbose):
            features_list.append(transform_func(seq))

        self._features = torch.from_numpy(np.asarray(features_list)).to(torch.float)

        # Build label mapping and per-class indices (keyed by numeric label)
        label2id = {l: i for i, l in enumerate(sorted(set(labels)))}
        numeric_labels = [label2id[l] for l in labels]
        self._labels = torch.tensor(numeric_labels, dtype=torch.long)

        self._class_indices = {}
        for idx, nlabel in enumerate(numeric_labels):
            self._class_indices.setdefault(nlabel, []).append(idx)

        self._ones = torch.ones(len(self._features))

        if verbose:
            print(f"Dataset loaded: {len(self._features)} sequences, "
                  f"feature_dim={self._features.shape[1]}, "
                  f"classes={len(self._class_indices)}")

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @property
    def num_classes(self) -> int:
        return len(self._class_indices)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a contrastive sample: one positive pair plus negative samples.

        Positive: random same-class sequence.
        Negatives: random samples from the full pool.

        Returns
        -------
        left_features : torch.Tensor
            ``(1 + neg_sample_per_pos, feature_dim)`` — anchor + negative lefts.
        right_features : torch.Tensor
            ``(1 + neg_sample_per_pos, feature_dim)`` — positive + negative rights.
        labels : torch.Tensor
            ``(1 + neg_sample_per_pos,)`` — 1 for positive pair, 0 for negatives.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")

        anchor = self._features[idx]

        # Positive: random same-class sequence (distinct from anchor when possible)
        anchor_label = self._labels[idx].item()
        same_class = self._class_indices[anchor_label]
        if len(same_class) > 1:
            candidates = [i for i in same_class if i != idx]
            pos_idx = candidates[random.randint(0, len(candidates) - 1)]
        else:
            pos_idx = same_class[0]
        positive = self._features[pos_idx]

        # Negative samples from full pool
        neg_indices = torch.multinomial(
            self._ones, replacement=True,
            num_samples=2 * self._neg_sample_per_pos,
        )

        left_features = torch.cat([
            anchor.unsqueeze(0),
            self._features[neg_indices[:self._neg_sample_per_pos]]
        ])
        right_features = torch.cat([
            positive.unsqueeze(0),
            self._features[neg_indices[self._neg_sample_per_pos:]]
        ])
        labels = torch.tensor(
            [1] + [0] * self._neg_sample_per_pos, dtype=torch.float
        )

        return left_features, right_features, labels
