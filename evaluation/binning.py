import csv
import sys
import os
import argparse
import numpy as np
from datetime import datetime
from utils import filter_sequences
from utils.progress import pbar

from embedders import load_embedder, get_embedding
from clustering import get_clusterer
from evaluation.eval_utils import (
    compute_class_center_medium_similarity,
    count_high_quality_clusters,
)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_MAX_SEQ_LEN = 20000
DEFAULT_MIN_SEQ_LEN = 2500
DEFAULT_MIN_ABUNDANCE = 10

# Percentiles returned by compute_class_center_medium_similarity: [10, 20, ..., 90]
# Index 6 corresponds to the 70th percentile, used as the clustering threshold.
_THRESHOLD_PERCENTILE_IDX = 6


def load_tsv_data(data_dir: str, species: str, sample: int, task: str,
                  max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
                  min_seq_len: int = 0,
                  min_abundance: int = 0) -> tuple[list[str], np.ndarray]:
    """Load sequences and labels from a TSV file.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    species : str
        Species name (e.g., "reference", "marine", "plant").
    sample : int
        Sample index.
    task : str
        "clustering" or "binning".
    max_seq_len : int
        Truncate sequences to this length.
    min_seq_len : int
        Filter sequences shorter than this.
    min_abundance : int
        Filter classes with fewer than this many samples.

    Returns
    -------
    tuple[list[str], np.ndarray]
        (sequences, numeric_labels) tuple.
    """
    file_path = os.path.join(data_dir, species, f"{task}_{sample}.tsv")
    print(f"Loading {file_path}...")
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(pbar(reader, desc=f"Reading {task}_{sample}.tsv",
                         unit="row"))[1:]  # skip header

    if not data:
        raise ValueError(f"No data rows found in {file_path} (file may be empty or header-only)")

    sequences = [d[0][:max_seq_len] for d in data]
    labels = [d[1] for d in data]

    sequences, labels = filter_sequences(sequences, labels,
                                         min_seq_len=min_seq_len,
                                         min_abundance=min_abundance)

    # Convert string labels to numeric
    label2id = {l: i for i, l in enumerate(sorted(set(labels)))}
    numeric_labels = np.array([label2id[l] for l in labels])

    return sequences, numeric_labels


def evaluate_binning(embedder, clusterer, data_dir: str, species_list: list[str],
                     samples: list[int], metric: str, output_path: str,
                     max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
                     min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
                     min_abundance: int = DEFAULT_MIN_ABUNDANCE,
                     cache_dir: str = None, suffix: str = "",
                     discard_mode: str = "discard"):
    """Run the full binning evaluation pipeline.

    For each species and sample:
    1. Load clustering reference data and compute similarity threshold
    2. Load binning data with filtering
    3. Embed sequences
    4. Cluster with the given algorithm
    5. Evaluate precision, recall, F1, and accuracy at thresholds 0.1-0.9

    Parameters
    ----------
    discard_mode : str
        ``"discard"``: exclude unassigned points from metrics (default).
        ``"garbage"``: treat unassigned points as a garbage cluster.
    """
    # Cache thresholds per species to avoid recomputation across samples
    threshold_cache: dict[str, float] = {}

    eval_tasks = [(sp, s) for sp in species_list for s in samples]
    for species, sample in pbar(eval_tasks, desc="Evaluating", unit="task"):
        print(f"\nSpecies: {species}, Sample: {sample}, Metric: {metric}")

        if species in threshold_cache:
            threshold, sim_scale = threshold_cache[species]
            print(f"Threshold: {threshold}, Scale: {sim_scale:.6f} (cached)")
        else:
            # Step 1: Compute threshold from clustering reference data
            print("Step 1/5: Loading reference data...")
            ref_seqs, ref_labels = load_tsv_data(
                data_dir, species, sample=0, task="clustering",
                max_seq_len=max_seq_len,
            )
            num_clusters_ref = len(set(ref_labels.tolist()))
            print(f"Reference: {len(ref_seqs)} sequences, {num_clusters_ref} clusters")

            ref_cache = None
            if cache_dir:
                ref_cache = os.path.join(
                    cache_dir, species, f"clustering_0{suffix}", "embedding.npy"
                )
            print("Step 2/5: Embedding reference data...")
            ref_result = get_embedding(embedder, ref_seqs, cache_path=ref_cache)

            percentile_values, sim_scale = compute_class_center_medium_similarity(
                ref_result.point_estimate, ref_labels, metric=metric,
                variances=ref_result.variance,
                kappas=ref_result.kappa,
            )
            # Use 70th percentile of intra-class similarities as clustering threshold
            threshold = percentile_values[_THRESHOLD_PERCENTILE_IDX]
            threshold_cache[species] = (threshold, sim_scale)
            print(f"Threshold: {threshold}, Scale: {sim_scale:.6f}")

        # Step 2: Load binning data
        print("Step 3/5: Loading binning data...")
        bin_seqs, bin_labels = load_tsv_data(
            data_dir, species, sample, task="binning",
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            min_abundance=min_abundance,
        )
        num_clusters = len(set(bin_labels.tolist()))
        print(f"Binning: {len(bin_seqs)} sequences, {num_clusters} clusters")

        # Step 3: Embed
        print("Step 4/5: Embedding binning data...")
        bin_cache = None
        if cache_dir:
            bin_cache = os.path.join(
                cache_dir, species, f"binning_{sample}{suffix}", "embedding.npy"
            )
        bin_result = get_embedding(embedder, bin_seqs, cache_path=bin_cache)

        # Step 4: Cluster (use the auto-calibrated scale from reference data)
        print("Step 5/5: Clustering...")
        clusterer.scale = sim_scale
        cluster_labels = clusterer.fit_predict(
            bin_result, min_similarity=threshold
        )

        # Step 5: Evaluate
        n_assigned = int((cluster_labels != -1).sum())
        print(f"Assigned: {n_assigned} / {len(bin_labels)} (mode={discard_mode})")

        results = count_high_quality_clusters(
            bin_labels, cluster_labels, mode=discard_mode
        )

        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision results: {results['precision_counts']}")
        print(f"Recall results: {results['recall_counts']}")
        print(f"F1 results: {results['f1_counts']}")

        if output_path:
            with open(output_path, "a+") as f:
                f.write("\n")
                f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                f.write(f" species: {species}, sample: {sample}\n")
                f.write(f"mode: {discard_mode}\n")
                f.write(f"accuracy: {results['accuracy']:.4f}\n")
                f.write(f"precision_results: {results['precision_counts']}\n")
                f.write(f"recall_results: {results['recall_counts']}\n")
                f.write(f"f1_results: {results['f1_counts']}\n")
                f.write(f"threshold: {threshold}\n\n")


def main():
    """CLI entry point for binning evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate binning performance")
    parser.add_argument("--species", type=str, default="reference,marine,plant")
    parser.add_argument("--samples", type=str, default="5,6")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="Embedder name (nonlinear, kmerprofile, uncertaingen, pcl, dnabert2, etc.)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model (for learned embedders)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=4, help="K-mer size (for kmerprofile)")
    parser.add_argument("--metric", type=str, default=None,
                        help="Similarity metric (dot, l1, l2). Default: embedder's default.")
    parser.add_argument("--cluster_algo", type=str, default="kmedoid")
    parser.add_argument("--scalable", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--min_seq_len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    parser.add_argument("--min_abundance", type=int, default=DEFAULT_MIN_ABUNDANCE)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--similarity_cache_dir", type=str, default=None,
                        help="Directory to cache pairwise similarity matrices")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--discard_mode", type=str, default="discard",
                        choices=["discard", "garbage"],
                        help="How to handle unassigned points: "
                             "'discard' excludes them, 'garbage' treats them "
                             "as a misclassified garbage cluster.")
    args = parser.parse_args()

    # Load embedder
    load_kwargs = {}
    if args.model == "kmerprofile":
        load_kwargs["k"] = args.k
    embedder = load_embedder(args.model, path=args.model_path, **load_kwargs)

    metric = args.metric or embedder.default_metric

    # Load clusterer
    clusterer = get_clusterer(
        args.cluster_algo, metric=metric, scalable=args.scalable,
        cache_dir=args.similarity_cache_dir,
    )

    evaluate_binning(
        embedder=embedder,
        clusterer=clusterer,
        data_dir=args.data_dir,
        species_list=args.species.split(","),
        samples=list(map(int, args.samples.split(","))),
        metric=metric,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        min_abundance=args.min_abundance,
        cache_dir=args.cache_dir,
        suffix=args.suffix,
        discard_mode=args.discard_mode,
    )


if __name__ == "__main__":
    main()
