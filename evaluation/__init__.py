"""Evaluation pipeline for metagenomics binning.

Public entry points:
- ``evaluate_binning`` (evaluation.binning): end-to-end embed → cluster → score pipeline.
- ``count_high_quality_clusters`` (evaluation.eval_utils): compute precision/recall/F1 at thresholds.
- ``load_tsv_data`` (evaluation.binning): load and filter sequences from a TSV file.
"""
