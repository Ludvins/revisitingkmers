import os
import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CheckM2BinResult:
    """Quality assessment for a single genome bin from CheckM2."""
    name: str
    completeness: float
    contamination: float
    quality_tier: str


@dataclass
class CheckM2Summary:
    """Aggregate CheckM2 quality results across all bins.

    Attributes
    ----------
    per_bin : list[CheckM2BinResult]
        Per-bin quality results.
    n_high_quality : int
        Bins with completeness >= 90 and contamination <= 5.
    n_medium_quality : int
        Bins with completeness >= 50 and contamination <= 10.
    n_low_quality : int
        Everything else.
    n_bins_total : int
        Total number of bins (excluding unassigned label -1).
    n_bins_evaluated : int
        Bins successfully evaluated by CheckM2.
    mean_completeness : float
        Mean completeness across evaluated bins.
    mean_contamination : float
        Mean contamination across evaluated bins.
    """
    per_bin: list = field(default_factory=list)
    n_high_quality: int = 0
    n_medium_quality: int = 0
    n_low_quality: int = 0
    n_bins_total: int = 0
    n_bins_evaluated: int = 0
    mean_completeness: float = 0.0
    mean_contamination: float = 0.0

    def to_dict(self) -> dict:
        return {
            "per_bin": [
                {"name": b.name, "completeness": b.completeness,
                 "contamination": b.contamination, "quality_tier": b.quality_tier}
                for b in self.per_bin
            ],
            "n_high_quality": self.n_high_quality,
            "n_medium_quality": self.n_medium_quality,
            "n_low_quality": self.n_low_quality,
            "n_bins_total": self.n_bins_total,
            "n_bins_evaluated": self.n_bins_evaluated,
            "mean_completeness": self.mean_completeness,
            "mean_contamination": self.mean_contamination,
        }


def _classify_quality(completeness: float, contamination: float) -> str:
    if completeness >= 90 and contamination <= 5:
        return "high"
    if completeness >= 50 and contamination <= 10:
        return "medium"
    return "low"


def export_bins_to_fasta(labels: np.ndarray, sequences: list,
                         output_dir: str, extension: str = ".fasta") -> list:
    """Write each predicted bin as a separate FASTA file.

    Sequences with label ``-1`` (unassigned) are skipped.

    Parameters
    ----------
    labels : np.ndarray
        (N,) integer cluster labels. ``-1`` means unassigned.
    sequences : list[str]
        N DNA sequences corresponding to the labels.
    output_dir : str
        Directory to write FASTA files into (created if needed).
    extension : str
        File extension for FASTA files (default ``.fasta``).

    Returns
    -------
    list[str]
        Paths of written FASTA files.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(labels) != len(sequences):
        raise ValueError(
            f"labels and sequences must have the same length, "
            f"got {len(labels)} and {len(sequences)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    if not extension.startswith("."):
        extension = "." + extension

    unique_labels = np.unique(labels)
    written = []

    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        bin_seqs = [sequences[i] for i in np.where(mask)[0]]
        if not bin_seqs:
            continue

        path = os.path.join(output_dir, f"bin_{lbl}{extension}")
        with open(path, "w") as f:
            for j, seq in enumerate(bin_seqs):
                f.write(f">seq_{j}\n{seq}\n")
        written.append(path)

    return written


def parse_checkm2_report(report_path: str) -> list:
    """Parse a CheckM2 ``quality_report.tsv`` file.

    Parameters
    ----------
    report_path : str
        Path to the TSV report produced by ``checkm2 predict``.

    Returns
    -------
    list[CheckM2BinResult]
        One result per bin in the report.
    """
    if not os.path.isfile(report_path):
        raise FileNotFoundError(
            f"CheckM2 report not found at {report_path}. "
            "The command may have failed silently."
        )

    results = []
    with open(report_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            comp = float(row["Completeness"])
            cont = float(row["Contamination"])
            results.append(CheckM2BinResult(
                name=row["Name"],
                completeness=comp,
                contamination=cont,
                quality_tier=_classify_quality(comp, cont),
            ))
    return results


def evaluate_checkm2(labels: np.ndarray, sequences: list,
                     threads: int = 1, extension: str = ".fasta",
                     tmpdir: str = None, keep_files: bool = False,
                     checkm2_cmd: str = "checkm2") -> CheckM2Summary:
    """Evaluate genome bins using CheckM2.

    Exports predicted bins as FASTA files, runs ``checkm2 predict``,
    and parses the quality report.

    Parameters
    ----------
    labels : np.ndarray
        (N,) integer cluster labels. ``-1`` means unassigned.
    sequences : list[str]
        N DNA sequences.
    threads : int
        Number of threads for CheckM2 (default 1).
    extension : str
        FASTA file extension (default ``.fasta``).
    tmpdir : str, optional
        Parent directory for temporary files. Uses system default if None.
    keep_files : bool
        If True, keep temporary FASTA and output files (default False).
    checkm2_cmd : str
        CheckM2 executable name or path (default ``"checkm2"``).

    Returns
    -------
    CheckM2Summary
        Per-bin and aggregate quality results.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(labels) != len(sequences):
        raise ValueError(
            f"labels and sequences must have the same length, "
            f"got {len(labels)} and {len(sequences)}"
        )

    # Count bins (excluding unassigned)
    unique_bins = [l for l in np.unique(labels) if l != -1]
    n_bins_total = len(unique_bins)

    if n_bins_total == 0:
        return CheckM2Summary()

    work_dir = tempfile.mkdtemp(dir=tmpdir, prefix="checkm2_eval_")
    bins_dir = os.path.join(work_dir, "bins")
    out_dir = os.path.join(work_dir, "output")

    try:
        written = export_bins_to_fasta(labels, sequences, bins_dir, extension)
        if not written:
            return CheckM2Summary()

        ext_arg = extension.lstrip(".")
        cmd = [
            checkm2_cmd, "predict",
            "--threads", str(threads),
            "--input", bins_dir,
            "--output-directory", out_dir,
            "-x", ext_arg,
            "--force",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"CheckM2 not found (tried '{checkm2_cmd}'). "
                "Install via: mamba install -c bioconda -c conda-forge checkm2"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"CheckM2 failed (exit {result.returncode}):\n{result.stderr}"
            )

        report_path = os.path.join(out_dir, "quality_report.tsv")
        per_bin = parse_checkm2_report(report_path)

        n_high = sum(1 for b in per_bin if b.quality_tier == "high")
        n_med = sum(1 for b in per_bin if b.quality_tier == "medium")
        n_low = sum(1 for b in per_bin if b.quality_tier == "low")
        n_eval = len(per_bin)
        mean_comp = np.mean([b.completeness for b in per_bin]) if per_bin else 0.0
        mean_cont = np.mean([b.contamination for b in per_bin]) if per_bin else 0.0

        return CheckM2Summary(
            per_bin=per_bin,
            n_high_quality=n_high,
            n_medium_quality=n_med,
            n_low_quality=n_low,
            n_bins_total=n_bins_total,
            n_bins_evaluated=n_eval,
            mean_completeness=float(mean_comp),
            mean_contamination=float(mean_cont),
        )
    finally:
        if not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
