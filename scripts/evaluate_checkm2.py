"""Evaluate genome bin quality on assembled contigs using CheckM2.

Translates: notebooks/checkm2_evaluation/checkm2_clustering_eval.ipynb

Pipeline:
  1. Load assembled contigs from a FASTA file.
  2. Calibrate similarity threshold from a reference labeled dataset
     (same 70th-percentile approach as evaluate_clustering.py).
  3. Embed the contigs and cluster with each requested algorithm.
  4. For uncertainty-aware models, optionally apply cluster-then-reject
     at multiple coverage levels.
  5. Export each bin as a FASTA file, run ``checkm2 predict``, parse
     completeness / contamination, and aggregate HQ / MQ / LQ counts.
  6. Save metrics (JSON + CSV) and figures.

Outputs land inside the model's experiment folder by default (experiment-centric layout).
Pass --output_dir / --cache_dir explicitly to override.

Examples
--------
Single model — outputs auto-land in runs/nonlinear/results/checkm2/:
    python scripts/evaluate_checkm2.py \\
        --model_path runs/nonlinear/model.model \\
        --fasta_path data/Fecal/eukfilt_assembly.fasta \\
        --ref_data_dir datasets/ --ref_species reference \\
        --checkm2_bin checkm2/bin/checkm2

Sweep a model directory, 100% and 75% coverage:
    python scripts/evaluate_checkm2.py \\
        --model_dir runs/ \\
        --fasta_path data/Fecal/eukfilt_assembly.fasta \\
        --ref_data_dir datasets/ \\
        --coverage_levels 100,75 \\
        --checkm2_bin checkm2/bin/checkm2

Override output location explicitly:
    python scripts/evaluate_checkm2.py \\
        --model_path runs/nonlinear/model.model \\
        --fasta_path data/Fecal/eukfilt_assembly.fasta \\
        --ref_data_dir datasets/ \\
        --output_dir results/checkm2/ \\
        --checkm2_bin checkm2/bin/checkm2
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import argparse
import numpy as np
from pathlib import Path


# ── Model helpers ─────────────────────────────────────────────────────────────

_NAME_KEYWORDS = {
    "nonlinear": "nonlinear",
    "hinge": "nonlinear",
    "uncertaingen": "uncertaingen",
    "ug_": "uncertaingen",
    "bern": "nonlinear",
    "pcl": "pcl",
    "kmerprofile": "kmerprofile",
    "kmer_profile": "kmerprofile",
    "lla": "lla",
    "laplace": "lla",
}


def _resolve_dirs(args, results_subdir: str) -> None:
    """Set cache_dir and output_dir relative to the model when not explicitly given."""
    base = Path(args.model_path).parent if args.model_path else Path(args.model_dir)
    if args.cache_dir is None:
        args.cache_dir = str(base / "embeddings")
    if args.output_dir is None:
        args.output_dir = str(base / "results" / results_subdir)


def _infer_model_type(stem: str):
    stem_lower = stem.lower()
    for keyword, name in _NAME_KEYWORDS.items():
        if keyword in stem_lower:
            return name
    return None


def _collect_models(args):
    if args.model_path:
        stem = Path(args.model_path).parent.name
        mtype = args.model_type or _infer_model_type(stem)
        if mtype is None:
            raise ValueError(f"Cannot infer model type from '{stem}'. Pass --model_type.")
        return [(stem, args.model_path, mtype)]
    models = []
    for p in sorted(Path(args.model_dir).rglob("*.model")):
        stem = p.parent.name
        mtype = args.model_type or _infer_model_type(stem)
        if mtype is None:
            print(f"[SKIP] Cannot infer model type for {p}. Use --model_type.")
            continue
        models.append((stem, str(p), mtype))
    if not models:
        raise ValueError(f"No .model files found under {args.model_dir}")
    return models


def _load_embedder(model_path, model_type, device):
    from embedders import load_embedder
    return load_embedder(model_type, path=model_path, device=device)


def _embed(embedder, sequences, cache_path=None):
    from embedders import get_embedding
    return get_embedding(embedder, sequences, cache_path=cache_path)


# ── FASTA loading ─────────────────────────────────────────────────────────────

def _load_fasta(fasta_path: str, max_seq_len: int = 0) -> tuple:
    """Return (contig_ids, sequences) from a FASTA file."""
    contig_ids, sequences = [], []
    current_id, current_seq = None, []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(current_seq)
                    if max_seq_len > 0:
                        seq = seq[:max_seq_len]
                    sequences.append(seq)
                    contig_ids.append(current_id)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        seq = "".join(current_seq)
        if max_seq_len > 0:
            seq = seq[:max_seq_len]
        sequences.append(seq)
        contig_ids.append(current_id)
    return contig_ids, sequences


# ── Threshold calibration ─────────────────────────────────────────────────────

def _calibrate_threshold(embedder, ref_data_dir, ref_species, ref_sample,
                         metric, ref_max_seq_len, cache_path=None):
    from evaluation.binning import load_tsv_data, _THRESHOLD_PERCENTILE_IDX
    from evaluation.eval_utils import compute_class_center_medium_similarity

    ref_seqs, ref_labels = load_tsv_data(
        ref_data_dir, ref_species, ref_sample, "clustering",
        max_seq_len=ref_max_seq_len,
    )
    ref_result = _embed(embedder, ref_seqs, cache_path=cache_path)
    percentile_values, sim_scale = compute_class_center_medium_similarity(
        ref_result.point_estimate, ref_labels,
        metric=metric,
        variances=ref_result.variance,
        kappas=ref_result.kappa,
    )
    threshold = percentile_values[_THRESHOLD_PERCENTILE_IDX]
    return threshold, sim_scale


# ── Clustering ────────────────────────────────────────────────────────────────

def _run_registered_clusterer(name, result, threshold, scale, min_bin_size, seed):
    from clustering import get_clusterer
    clusterer = get_clusterer(name, min_bin_size=min_bin_size, random_state=seed)
    clusterer.scale = scale
    return clusterer.fit_predict(result, min_similarity=threshold)


def _run_kmeans(result, k, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=3)
    return km.fit_predict(result.point_estimate)


# ── Uncertainty rejection ─────────────────────────────────────────────────────

def _uncertainty_scores(result):
    if result.kappa is not None:
        return 1.0 / (result.kappa + 1e-12)
    if result.variance is not None:
        return result.variance.mean(axis=1)
    return np.zeros(len(result.mean))  # deterministic: no rejection


def _has_uncertainty(result):
    return result.variance is not None or result.kappa is not None


def _apply_coverage(labels, uncertainty, coverage_pct):
    n_keep = int(len(labels) * coverage_pct / 100)
    order = np.argsort(uncertainty)
    masked = labels.copy()
    masked[order[n_keep:]] = -1
    return masked


# ── Bin export helpers ────────────────────────────────────────────────────────

def _export_and_evaluate_checkm2(labels, full_sequences, out_dir,
                                  checkm2_cmd, threads, tmp_dir):
    """Export bins to FASTA and run CheckM2. Returns CheckM2Summary."""
    from evaluation.checkm2 import evaluate_checkm2
    os.makedirs(out_dir, exist_ok=True)
    return evaluate_checkm2(
        labels=labels,
        sequences=full_sequences,
        threads=threads,
        tmpdir=tmp_dir or out_dir,
        keep_files=True,
        checkm2_cmd=checkm2_cmd,
    )


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_quality_bars(summaries: dict, title: str, out_path: str, fig_format: str):
    """Stacked bar chart of HQ / MQ / LQ bin counts per run."""
    import matplotlib.pyplot as plt

    labels = list(summaries.keys())
    hq = [summaries[k].n_high_quality for k in labels]
    mq = [summaries[k].n_medium_quality for k in labels]
    lq = [summaries[k].n_low_quality for k in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar(x, hq, label="High quality", color="#2ca02c")
    ax.bar(x, mq, bottom=hq, label="Medium quality", color="#ff7f0e")
    ax.bar(x, lq, bottom=np.array(hq) + np.array(mq),
           label="Low quality", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Number of bins")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_path}.{fig_format}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_completeness_contamination(summary, title: str, out_path: str, fig_format: str):
    """Scatter plot of completeness vs contamination for each bin."""
    import matplotlib.pyplot as plt

    if not summary.per_bin:
        return
    comp = [b.completeness for b in summary.per_bin]
    cont = [b.contamination for b in summary.per_bin]
    tier_colors = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
    colors = [tier_colors.get(b.quality_tier, "grey") for b in summary.per_bin]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cont, comp, c=colors, alpha=0.6, edgecolors="none", s=20)
    ax.axhline(90, color="green", linestyle="--", linewidth=0.8, label="HQ threshold")
    ax.axvline(5, color="green", linestyle="--", linewidth=0.8)
    ax.axhline(50, color="orange", linestyle="--", linewidth=0.8, label="MQ threshold")
    ax.axvline(10, color="orange", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Contamination (%)")
    ax.set_ylabel("Completeness (%)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_path}.{fig_format}", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CheckM2 genome bin quality evaluation on assembled contigs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_grp = parser.add_mutually_exclusive_group(required=True)
    model_grp.add_argument("--model_path", help="Single .model file.")
    model_grp.add_argument("--model_dir", help="Directory to sweep all .model files.")
    parser.add_argument("--model_type",
                        help="Embedder name. Inferred from filename if omitted.")

    # Input data
    parser.add_argument("--fasta_path", required=True,
                        help="FASTA file of assembled contigs.")
    parser.add_argument("--output_dir", default=None,
                        help="Root output folder. "
                             "Defaults to <model_dir>/results/checkm2/ when omitted.")

    # Threshold calibration from reference data
    parser.add_argument("--ref_data_dir", required=True,
                        help="Root directory containing labeled reference TSV files.")
    parser.add_argument("--ref_species", default="reference",
                        help="Species name for threshold calibration.")
    parser.add_argument("--ref_sample", type=int, default=0,
                        help="Reference sample index for calibration.")
    parser.add_argument("--ref_max_seq_len", type=int, default=20000,
                        help="Max sequence length when loading reference data.")

    # Clustering
    parser.add_argument("--cluster_algos", default="greedy_kmedoid,kmeans,dpgmm",
                        help="Comma-separated algorithms: greedy_kmedoid,kmedoid,kmeans,dpgmm.")
    parser.add_argument("--kmeans_k", type=int, default=None,
                        help="Fixed k for KMeans. If None, estimated from reference cluster count.")
    parser.add_argument("--min_bin_size", type=int, default=5)

    # Coverage / uncertainty rejection
    parser.add_argument("--coverage_levels", default="100,75,50",
                        help="Coverage percentages (comma-separated). "
                             "Rejection only applied to uncertainty-aware models.")

    # Embedding
    parser.add_argument("--k", type=int, default=4, help="K-mer size.")
    parser.add_argument("--max_seq_len", type=int, default=20000,
                        help="Truncate contigs to this length for embedding.")
    parser.add_argument("--cache_dir",
                        help="Directory to cache embeddings (optional).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=26042024)

    # CheckM2
    parser.add_argument("--checkm2_bin", default="checkm2/bin/checkm2",
                        help="Path to the checkm2 executable.")
    parser.add_argument("--checkm2_db",
                        help="Path to DIAMOND database (passed as --database to checkm2).")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for checkm2 predict.")
    parser.add_argument("--tmp_dir",
                        help="Scratch directory for bin FASTA files (optional).")

    # Output
    parser.add_argument("--fig_format", default="png", choices=["png", "pdf"])

    args = parser.parse_args()
    _resolve_dirs(args, "checkm2")

    # Parse lists
    algos = [a.strip() for a in args.cluster_algos.split(",")]
    coverage_levels = [int(c) for c in args.coverage_levels.split(",")]

    fig_dir = os.path.join(args.output_dir, "figures")
    met_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    # Load contigs
    print(f"Loading contigs from {args.fasta_path}...")
    contig_ids, full_sequences = _load_fasta(args.fasta_path, max_seq_len=0)
    truncated_sequences = [s[:args.max_seq_len] for s in full_sequences]
    print(f"  {len(full_sequences)} contigs loaded.")

    models = _collect_models(args)
    all_results = []
    all_summaries = {}

    for model_stem, model_path, model_type in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_stem}  ({model_type})")
        print(f"{'='*60}")

        embedder = _load_embedder(model_path, model_type, args.device)
        metric = embedder.default_metric

        # Calibrate threshold
        print("Calibrating threshold from reference data...")
        ref_cache = None
        if args.cache_dir:
            ref_cache = os.path.join(
                args.cache_dir, model_stem,
                f"{args.ref_species}_clustering_{args.ref_sample}.npy"
            )
        threshold, sim_scale = _calibrate_threshold(
            embedder,
            ref_data_dir=args.ref_data_dir,
            ref_species=args.ref_species,
            ref_sample=args.ref_sample,
            metric=metric,
            ref_max_seq_len=args.ref_max_seq_len,
            cache_path=ref_cache,
        )
        print(f"  Threshold: {threshold:.4f}  Scale: {sim_scale:.6f}")

        # Estimate k for KMeans from reference cluster count if not given
        kmeans_k = args.kmeans_k
        if kmeans_k is None and "kmeans" in algos:
            from evaluation.binning import load_tsv_data
            _, ref_labels = load_tsv_data(
                args.ref_data_dir, args.ref_species, args.ref_sample, "clustering",
                max_seq_len=args.ref_max_seq_len,
            )
            kmeans_k = len(set(ref_labels.tolist()))
            print(f"  Auto k for KMeans: {kmeans_k} (from reference cluster count)")

        # Embed contigs
        contig_cache = None
        if args.cache_dir:
            contig_cache = os.path.join(args.cache_dir, model_stem, "contigs.npy")
        print("Embedding contigs...")
        contig_result = _embed(embedder, truncated_sequences, cache_path=contig_cache)
        uncertainty = _uncertainty_scores(contig_result)
        is_uncertain = _has_uncertainty(contig_result)

        model_summaries = {}

        for algo in algos:
            print(f"\n  Clustering: {algo}")

            if algo == "kmeans":
                labels_100 = _run_kmeans(contig_result, k=kmeans_k, seed=args.seed)
            else:
                labels_100 = _run_registered_clusterer(
                    algo, contig_result, threshold, sim_scale,
                    args.min_bin_size, args.seed,
                )

            active_coverage = coverage_levels if is_uncertain else [100]

            for cov in active_coverage:
                if cov == 100:
                    labels = labels_100
                else:
                    labels = _apply_coverage(labels_100, uncertainty, cov)

                n_bins = int((labels != -1).max()) + 1 if (labels != -1).any() else 0
                n_assigned = int((labels != -1).sum())
                print(f"    Coverage {cov}%: {n_assigned}/{len(labels)} assigned, "
                      f"~{n_bins} bins")

                run_key = f"{model_stem}_{algo}_cov{cov}"
                bin_out_dir = os.path.join(args.output_dir, "bins", run_key)
                checkm2_out_dir = os.path.join(args.output_dir, "checkm2", run_key)
                os.makedirs(bin_out_dir, exist_ok=True)
                os.makedirs(checkm2_out_dir, exist_ok=True)

                print(f"    Running CheckM2...")
                summary = _export_and_evaluate_checkm2(
                    labels=labels,
                    full_sequences=full_sequences,
                    out_dir=bin_out_dir,
                    checkm2_cmd=args.checkm2_bin,
                    threads=args.threads,
                    tmp_dir=args.tmp_dir,
                )

                print(f"    HQ={summary.n_high_quality}, MQ={summary.n_medium_quality}, "
                      f"LQ={summary.n_low_quality}  "
                      f"(completeness={summary.mean_completeness:.1f}%, "
                      f"contamination={summary.mean_contamination:.1f}%)")

                model_summaries[run_key] = summary

                # Save per-run scatter plot
                _plot_completeness_contamination(
                    summary,
                    title=f"{run_key}: completeness vs contamination",
                    out_path=os.path.join(fig_dir, f"scatter_{run_key}"),
                    fig_format=args.fig_format,
                )

                # Persist flat result row
                result_row = {
                    "model": model_stem,
                    "algo": algo,
                    "coverage_pct": cov,
                    "n_assigned": n_assigned,
                    "n_total": len(labels),
                    "n_bins_total": summary.n_bins_total,
                    "n_bins_evaluated": summary.n_bins_evaluated,
                    "n_high_quality": summary.n_high_quality,
                    "n_medium_quality": summary.n_medium_quality,
                    "n_low_quality": summary.n_low_quality,
                    "mean_completeness": summary.mean_completeness,
                    "mean_contamination": summary.mean_contamination,
                }
                all_results.append(result_row)
                all_summaries[run_key] = summary

        # Quality bar chart for all algo × coverage combos for this model
        _plot_quality_bars(
            model_summaries,
            title=f"Bin quality — {model_stem}",
            out_path=os.path.join(fig_dir, f"quality_bars_{model_stem}"),
            fig_format=args.fig_format,
        )

    # ── Save metrics ──────────────────────────────────────────────────────────
    json_path = os.path.join(met_dir, "checkm2_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {json_path}")

    csv_path = os.path.join(met_dir, "checkm2_summary.csv")
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"CSV saved to {csv_path}")

    import datetime
    eval_config = {
        **vars(args),
        "algos": algos,
        "coverage_levels": coverage_levels,
        "timestamp": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
    }
    eval_cfg_path = os.path.join(met_dir, "eval_config.json")
    with open(eval_cfg_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    print(f"Eval config saved to {eval_cfg_path}")

    # Global comparison bar chart (all models × algos)
    if all_summaries:
        _plot_quality_bars(
            all_summaries,
            title="Bin quality — all models",
            out_path=os.path.join(fig_dir, "quality_bars_all"),
            fig_format=args.fig_format,
        )


if __name__ == "__main__":
    main()
