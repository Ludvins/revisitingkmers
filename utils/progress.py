"""Consistent progress bar formatting across the project."""

import os
from tqdm import tqdm as _tqdm

BAR_FORMAT = (
    " {desc} \u2502{bar:30}\u2502 {n_fmt}/{total_fmt}"
    " [{elapsed}<{remaining}{postfix}]"
)

BAR_FORMAT_UNKNOWN = (
    " {desc} \u2502 {n_fmt} [{elapsed}{postfix}]"
)


def pbar(iterable=None, total=None, desc="", unit="it",
         leave=True, disable=False, unit_scale=False, **kwargs):
    """Create a styled tqdm progress bar.

    Parameters
    ----------
    iterable : iterable, optional
        Iterable to wrap.
    total : int, optional
        Total count (for manual update bars, pass iterable=None).
    desc : str
        Description text shown before the bar.
    unit : str
        Unit label for items.
    leave : bool
        Whether to keep the bar after completion.
    disable : bool
        If True, suppress the bar entirely.
    **kwargs
        Extra tqdm kwargs.
    """
    fmt = BAR_FORMAT if total is not None or iterable is not None else BAR_FORMAT_UNKNOWN
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=unit_scale,
        leave=leave,
        disable=disable,
        bar_format=fmt,
        ascii=" \u2588",
        **kwargs,
    )


CHUNK_SIZE = 1 << 20  # 1 MB


def count_lines(file_path: str, desc: str = "Counting lines",
                verbose: bool = True) -> int:
    """Count lines in a file using raw byte counting with disk caching.

    First checks for a cached `.linecount` file next to the original file.
    If the cache exists and is newer than the file, returns the cached count
    instantly. Otherwise counts by reading raw bytes in 1 MB chunks (~3x
    faster than Python text iteration) and saves the result.
    """
    cache_path = file_path + ".linecount"

    # Use cache if it exists and is newer than the data file
    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(file_path):
            with open(cache_path, "r") as f:
                count = int(f.read().strip())
            if verbose:
                print(f" {desc}: {count:,} lines (cached)")
            return count

    # Count by reading raw bytes
    file_size = os.path.getsize(file_path)
    count = 0
    with open(file_path, "rb") as f:
        progress = pbar(total=file_size, desc=desc, unit="B",
                        unit_scale=True, disable=not verbose)
        while chunk := f.read(CHUNK_SIZE):
            count += chunk.count(b"\n")
            progress.update(len(chunk))
        progress.close()

    # Cache for next time
    with open(cache_path, "w") as f:
        f.write(str(count))

    return count
