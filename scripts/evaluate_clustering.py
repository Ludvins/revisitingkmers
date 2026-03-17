"""Evaluate embeddings with multiple clustering algorithms and uncertainty rejection.

Translates:
  - notebooks/clustering/clustering_evaluation*.ipynb
  - notebooks/genome_experiments/genome_experiment*.ipynb

Pipeline per embedder:
  1. Load reference data (clustering_0.tsv) → embed → calibrate similarity threshold
     (70th percentile of intra-class similarities, as in the paper).
  2. Load binning data (binning_N.tsv) → embed → cluster with each algorithm.
  3. Optionally apply uncertainty-based cluster-then-reject at multiple coverage levels.
  4. Compute precision / recall / F1 counts at thresholds 0.1-0.9.
  5. Save metrics (JSON + CSV) and figures.

Outputs land inside the model's experiment folder by default (experiment-centric layout).
Pass --output_dir / --cache_dir explicitly to override.

Examples
--------
Single model — outputs auto-land in runs/nonlinear/embeddings/ and runs/nonlinear/results/clustering/:
    python scripts/evaluate_clustering.py \\
        --model_path runs/nonlinear/model.model \\
        --data_dir datasets/ --species reference --samples 5,6

Sweep a model directory, KMeans only, 100% and 75% coverage:
    python scripts/evaluate_clustering.py \\
        --model_dir runs/ \\
        --data_dir datasets/ --species reference,marine,plant --samples 5,6 \\
        --cluster_algos kmeans \\
        --coverage_levels 100,75

Override output location explicitly:
    python scripts/evaluate_clustering.py \\
        --model_path runs/nonlinear/model.model \\
        --data_dir datasets/ --species reference --samples 5,6 \\
        --output_dir results/clustering/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import argparse
import numpy as np
from pathlib import Path


# ── Model helpers (shared with evaluate_classification.py) ───────────────────

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


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_tsv(data_dir, species, sample, task, max_seq_len, min_seq_len, min_abundance):
    from evaluation.binning import load_tsv_data
    return load_tsv_data(
        data_dir, species, sample, task,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        min_abundance=min_abundance,
    )


# ── Clustering ────────────────────────────────────────────────────────────────

def _run_registered_clusterer(name, result, threshold, scale, min_bin_size, seed):
    """Run a clusterer from the registry (kmedoid, greedy_kmedoid, dpgmm)."""
    from clustering import get_clusterer
    kwargs = {"min_bin_size": min_bin_size, "random_state": seed}
    if name == "dpgmm":
        kwargs["random_state"] = seed
    clusterer = get_clusterer(name, **kwargs)
    clusterer.scale = scale
    return clusterer.fit_predict(result, min_similarity=threshold)


def _run_kmeans(result, k, seed):
    """Run sklearn KMeans on the point estimate."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=3)
    return km.fit_predict(result.point_estimate)


# ── Uncertainty rejection ─────────────────────────────────────────────────────

def _has_uncertainty(result):
    return result.variance is not None or result.kappa is not None


def _uncertainty_scores(result):
    """Lower = more certain. Returns (N,) array."""
    if result.kappa is not None:
        return 1.0 / (result.kappa + 1e-12)
    if result.variance is not None:
        return result.variance.mean(axis=1)
    # Deterministic model: assign random scores
    return np.random.rand(len(result.mean))


def _apply_coverage(labels, uncertainty, coverage_pct):
    """Reject the (100 - coverage_pct)% most uncertain points."""
    n_keep = int(len(labels) * coverage_pct / 100)
    order = np.argsort(uncertainty)          # ascending = most certain first
    kept = order[:n_keep]
    masked = labels.copy()
    reject = order[n_keep:]
    masked[reject] = -1
    return masked


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(true_labels, pred_labels, thresholds, mode):
    from evaluation.eval_utils import count_high_quality_clusters
    return count_high_quality_clusters(true_labels, pred_labels,
                                       mode=mode, thresholds=thresholds)


# ── Threshold calibration ────────────────────────────────────────────────────

def _calibrate_threshold(embedder, ref_seqs, ref_labels, metric, cache_path=None):
    from evaluation.binning import _THRESHOLD_PERCENTILE_IDX
    from evaluation.eval_utils import compute_class_center_medium_similarity
    ref_result = _embed(embedder, ref_seqs, cache_path=cache_path)
    percentile_values, sim_scale = compute_class_center_medium_similarity(
        ref_result.point_estimate, ref_labels,
        metric=metric,
        variances=ref_result.variance,
        kappas=ref_result.kappa,
    )
    threshold = percentile_values[_THRESHOLD_PERCENTILE_IDX]
    return threshold, sim_scale, ref_result


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_f1_counts(results_by_algo, thresholds, title, out_path, fig_format):
    """Bar chart of F1 counts at each threshold, one bar group per algorithm."""
    import matplotlib.pyplot as plt

    algos = list(results_by_algo.keys())
    n_t = len(thresholds)
    x = np.arange(n_t)
    width = 0.8 / max(len(algos), 1)

    fig, ax = plt.subplots(figsize=(max(8, n_t * 1.2), 5))
    for i, algo in enumerate(algos):
        counts = results_by_algo[algo].get("f1_counts", [0] * n_t)
        ax.bar(x + i * width, counts, width, label=algo)

    ax.set_xticks(x + width * len(algos) / 2)
    ax.set_xticklabels([f">{t:.1f}" for t in thresholds], rotation=45)
    ax.set_xlabel("F1 threshold")
    ax.set_ylabel("Number of clusters with F1 > threshold")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_path}.{fig_format}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_coverage_curve(coverage_data, title, out_path, fig_format):
    """Line plot: F1>0.5 count vs coverage level for each algorithm."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, cov_results in coverage_data.items():
        coverages = sorted(cov_results.keys())
        counts = [cov_results[c].get("f1_counts", [0])[4]   # index 4 = threshold 0.5
                  for c in coverages]
        ax.plot(coverages, counts, marker="o", label=algo)

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Clusters with F1 > 0.5")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_path}.{fig_format}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_kmeans_sweep(k_values, f1_counts_by_k, threshold_idx, title, out_path, fig_format):
    """Line chart of F1>0.5 count vs k for KMeans."""
    import matplotlib.pyplot as plt

    counts = [f1_counts_by_k[k][threshold_idx] for k in k_values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, counts, marker="o")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Clusters with F1 > 0.5")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{out_path}.{fig_format}", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clustering benchmark: multiple algorithms + uncertainty rejection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_grp = parser.add_mutually_exclusive_group(required=True)
    model_grp.add_argument("--model_path", help="Single .model file.")
    model_grp.add_argument("--model_dir", help="Directory to sweep all .model files.")
    parser.add_argument("--model_type",
                        help="Embedder name. Inferred from filename if omitted.")

    # Data
    parser.add_argument("--data_dir", required=True,
                        help="Root data directory (contains species/task_N.tsv).")
    parser.add_argument("--output_dir", default=None,
                        help="Root output folder. Defaults to "
                             "<model_dir>/results/clustering/ when omitted.")
    parser.add_argument("--species", default="reference,marine,plant",
                        help="Comma-separated species names.")
    parser.add_argument("--samples", default="5,6",
                        help="Comma-separated sample indices for binning evaluation.")
    parser.add_argument("--ref_sample", type=int, default=0,
                        help="Sample index used for threshold calibration.")
    parser.add_argument("--max_seq_len", type=int, default=20000)
    parser.add_argument("--min_seq_len", type=int, default=2500)
    parser.add_argument("--min_abundance", type=int, default=10)

    # Clustering
    parser.add_argument("--cluster_algos",
                        default="greedy_kmedoid,kmedoid,kmeans,dpgmm",
                        help="Comma-separated algorithms: greedy_kmedoid,kmedoid,kmeans,dpgmm.")
    parser.add_argument("--kmeans_k_range", default="50,100,200,323,500",
                        help="K values to sweep for KMeans (comma-separated).")
    parser.add_argument("--min_bin_size", type=int, default=5,
                        help="Minimum cluster size (smaller clusters get label -1).")

    # Uncertainty rejection
    parser.add_argument("--coverage_levels", default="100,90,80,70,60,50,40,30,20",
                        help="Coverage percentages for cluster-then-reject.")
    parser.add_argument("--rejection_mode", default="discard",
                        choices=["discard", "garbage"],
                        help="How to handle unassigned points in metrics.")

    # Evaluation
    parser.add_argument("--metric",
                        help="Similarity metric (l1/l2/dot). Default: embedder's default.")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="F1/precision/recall thresholds for counting clusters.")
    parser.add_argument("--cache_dir",
                        help="Directory to cache embeddings.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=26042024)

    # Output
    parser.add_argument("--fig_format", default="png", choices=["png", "pdf"])

    args = parser.parse_args()
    _resolve_dirs(args, "clustering")

    # Parse lists
    species_list = [s.strip() for s in args.species.split(",")]
    samples = [int(s) for s in args.samples.split(",")]
    algos = [a.strip() for a in args.cluster_algos.split(",")]
    kmeans_ks = [int(k) for k in args.kmeans_k_range.split(",")]
    coverage_levels = [int(c) for c in args.coverage_levels.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    fig_dir = os.path.join(args.output_dir, "figures")
    met_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    models = _collect_models(args)
    all_results = []  # list of flat dicts for CSV

    for model_stem, model_path, model_type in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_stem}  ({model_type})")
        print(f"{'='*60}")

        embedder = _load_embedder(model_path, model_type, args.device)
        metric = args.metric or embedder.default_metric

        # Cache threshold per species
        threshold_cache = {}

        for species in species_list:
            # ── Calibrate threshold from reference clustering data ──────────
            if species not in threshold_cache:
                print(f"\n[{species}] Loading reference data for calibration...")
                ref_seqs, ref_labels = _load_tsv(
                    args.data_dir, species, args.ref_sample, "clustering",
                    args.max_seq_len, 0, 0,
                )
                ref_cache = None
                if args.cache_dir:
                    ref_cache = os.path.join(
                        args.cache_dir, model_stem, species,
                        f"clustering_{args.ref_sample}.npy"
                    )
                threshold, sim_scale, _ = _calibrate_threshold(
                    embedder, ref_seqs, ref_labels, metric, cache_path=ref_cache
                )
                threshold_cache[species] = (threshold, sim_scale)
                print(f"  Threshold: {threshold:.4f}  Scale: {sim_scale:.6f}")

            threshold, sim_scale = threshold_cache[species]

            for sample in samples:
                print(f"\n[{species}, sample {sample}] Loading binning data...")
                bin_seqs, bin_labels = _load_tsv(
                    args.data_dir, species, sample, "binning",
                    args.max_seq_len, args.min_seq_len, args.min_abundance,
                )
                n_clusters_true = len(set(bin_labels.tolist()))
                print(f"  {len(bin_seqs)} sequences, {n_clusters_true} true clusters")

                bin_cache = None
                if args.cache_dir:
                    bin_cache = os.path.join(
                        args.cache_dir, model_stem, species, f"binning_{sample}.npy"
                    )
                bin_result = _embed(embedder, bin_seqs, cache_path=bin_cache)
                uncertainty = _uncertainty_scores(bin_result)

                # ── Per-algorithm evaluation ───────────────────────────────
                algo_results = {}
                algo_coverage_data = {}

                for algo in algos:
                    print(f"  Clustering: {algo}")
                    algo_coverage_data[algo] = {}

                    if algo == "kmeans":
                        # Sweep k values and pick the best by F1>0.5
                        kmeans_f1_counts = {}
                        for k in kmeans_ks:
                            labels = _run_kmeans(bin_result, k=k, seed=args.seed)
                            res = _evaluate(bin_labels, labels, thresholds,
                                            args.rejection_mode)
                            kmeans_f1_counts[k] = res["f1_counts"]
                        # Select k with best F1>0.5 count (threshold index 4 = 0.5)
                        best_k = max(kmeans_ks,
                                     key=lambda k: kmeans_f1_counts[k][4])
                        labels_100 = _run_kmeans(bin_result, k=best_k, seed=args.seed)
                        print(f"    Best k={best_k}")

                        # Save k-sweep figure
                        fig_path = os.path.join(
                            fig_dir, f"kmeans_sweep_{model_stem}_{species}_s{sample}"
                        )
                        _plot_kmeans_sweep(
                            kmeans_ks, kmeans_f1_counts, threshold_idx=4,
                            title=f"KMeans k sweep — {model_stem} | {species} sample {sample}",
                            out_path=fig_path, fig_format=args.fig_format,
                        )
                    else:
                        labels_100 = _run_registered_clusterer(
                            algo, bin_result, threshold, sim_scale,
                            args.min_bin_size, args.seed,
                        )

                    # Coverage levels
                    for cov in coverage_levels:
                        if cov == 100:
                            cov_labels = labels_100
                        else:
                            cov_labels = _apply_coverage(labels_100, uncertainty, cov)

                        res = _evaluate(bin_labels, cov_labels, thresholds,
                                        args.rejection_mode)
                        algo_coverage_data[algo][cov] = res

                        flat = {
                            "model": model_stem,
                            "species": species,
                            "sample": sample,
                            "algo": algo,
                            "coverage_pct": cov,
                            "n_assigned": int((cov_labels != -1).sum()),
                            "n_total": len(bin_labels),
                            "accuracy": res.get("accuracy", float("nan")),
                        }
                        for i, t in enumerate(thresholds):
                            flat[f"f1_count_{t}"] = res["f1_counts"][i]
                            flat[f"prec_count_{t}"] = res["precision_counts"][i]
                            flat[f"rec_count_{t}"] = res["recall_counts"][i]
                        all_results.append(flat)

                    # 100% coverage algo summary for F1 bar chart
                    algo_results[algo] = algo_coverage_data[algo].get(100, {})

                # ── Figures ─────────────────────────────────────────────────
                fig_base = f"{model_stem}_{species}_s{sample}"
                _plot_f1_counts(
                    algo_results, thresholds,
                    title=f"F1 counts — {model_stem} | {species} sample {sample}",
                    out_path=os.path.join(fig_dir, f"f1_counts_{fig_base}"),
                    fig_format=args.fig_format,
                )
                _plot_coverage_curve(
                    algo_coverage_data,
                    title=f"Coverage curve — {model_stem} | {species} sample {sample}",
                    out_path=os.path.join(fig_dir, f"coverage_{fig_base}"),
                    fig_format=args.fig_format,
                )

    # ── Save metrics ──────────────────────────────────────────────────────────
    json_path = os.path.join(met_dir, "clustering_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = os.path.join(met_dir, "clustering_results.csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"CSV saved to {csv_path}")

    import datetime
    eval_config = {
        **vars(args),
        "species_list": species_list,
        "samples": samples,
        "algos": algos,
        "coverage_levels": coverage_levels,
        "thresholds": thresholds,
        "timestamp": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
    }
    eval_cfg_path = os.path.join(met_dir, "eval_config.json")
    with open(eval_cfg_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    print(f"Eval config saved to {eval_cfg_path}")


if __name__ == "__main__":
    main()
