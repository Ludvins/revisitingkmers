import random
import numpy as np
import torch
from typing import Tuple, Callable
from datasets.base import BaseContrastiveDataset
from utils.progress import pbar, count_lines


class PairedReadsDataset(BaseContrastiveDataset):
    """Contrastive dataset from paired DNA reads in CSV format.

    Each line of the input file contains a comma-separated pair of reads:
        left_read,right_read

    Positive pairs come from actual read pairs. Negative samples are drawn
    uniformly from the entire pool of reads.
    """

    def __init__(self, file_path: str, transform_func: Callable[[str], np.ndarray],
                 neg_sample_per_pos: int = 1000, max_read_num: int = 0,
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

        lines_num = count_lines(file_path, desc="Counting lines",
                                verbose=verbose)

        if max_read_num > 0:
            chosen_lines = sorted(random.sample(range(lines_num), max_read_num))
        else:
            chosen_lines = None

        total_to_read = max_read_num if max_read_num > 0 else lines_num

        # Read and transform paired reads
        left_profiles, right_profiles = [], []
        chosen_idx = 0
        with open(file_path, "r") as f:
            progress = pbar(total=total_to_read, desc="Loading read pairs",
                            unit="pair", disable=not verbose)
            for line_idx, line in enumerate(f):
                if chosen_lines is not None:
                    if chosen_idx == len(chosen_lines):
                        break
                    if line_idx != chosen_lines[chosen_idx]:
                        continue
                    chosen_idx += 1

                left_read, right_read = line.strip().split(",")
                left_profiles.append(transform_func(left_read))
                right_profiles.append(transform_func(right_read))
                progress.update(1)
            progress.close()

        left_arr = np.asarray(left_profiles, dtype=np.float32)
        right_arr = np.asarray(right_profiles, dtype=np.float32)
        self._all_profiles = torch.from_numpy(np.concatenate([left_arr, right_arr]))

        if verbose:
            read_count = len(left_profiles)
            print(f"Dataset loaded: {read_count} read pairs from {lines_num} total lines.")

        self._ones = torch.ones((len(self._all_profiles),))

    def __len__(self) -> int:
        return len(self._all_profiles) // 2

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a contrastive sample: one positive pair plus negative samples.

        Parameters
        ----------
        idx : int
            Index of the anchor read pair.

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

        n_pairs = len(self)
        # Draw random indices for negative sampling from the full read pool
        neg_indices = torch.multinomial(
            self._ones, replacement=True, num_samples=2 * self._neg_sample_per_pos
        )

        # Stack anchor with negative left reads
        left_features = torch.cat([
            self._all_profiles[idx].unsqueeze(0),
            self._all_profiles[neg_indices[:self._neg_sample_per_pos]]
        ])
        right_features = torch.cat([
            self._all_profiles[idx + n_pairs].unsqueeze(0),
            self._all_profiles[neg_indices[self._neg_sample_per_pos:]]
        ])
        labels = torch.tensor(
            [1] + [0] * self._neg_sample_per_pos, dtype=torch.float
        )

        return left_features, right_features, labels
