"""Shared utility functions for the revisitingkmers package."""

import collections


def filter_sequences(sequences, labels, min_seq_len=0, min_abundance=0):
    """Filter sequences and labels by length and label abundance.

    Parameters
    ----------
    sequences : list[str]
        Input sequences.
    labels : list
        Corresponding labels (same length as sequences).
    min_seq_len : int
        Remove sequences shorter than this. 0 disables the filter.
    min_abundance : int
        Remove sequences whose label appears fewer than this many times.
        0 disables the filter.

    Returns
    -------
    tuple[list, list]
        Filtered (sequences, labels).
    """
    if min_seq_len > 0:
        filtered = [(s, l) for s, l in zip(sequences, labels) if len(s) >= min_seq_len]
        if filtered:
            sequences, labels = zip(*filtered)
            sequences, labels = list(sequences), list(labels)
        else:
            return [], []

    if min_abundance > 0:
        label_counts = collections.Counter(labels)
        filtered = [(s, l) for s, l in zip(sequences, labels)
                    if label_counts[l] >= min_abundance]
        if filtered:
            sequences, labels = zip(*filtered)
            sequences, labels = list(sequences), list(labels)
        else:
            return [], []

    return list(sequences), list(labels)
