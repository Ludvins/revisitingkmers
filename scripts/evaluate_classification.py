"""Evaluate embedding quality with KNN, Logistic Regression, and SVM classifiers.

Loads a pretrained embedder (or sweeps a directory of models), embeds labeled
sequences, fits classifiers, and saves metrics + figures.

Translates: notebooks/genome_experiments/ classification evaluation.

Outputs land inside the model's experiment folder by default (experiment-centric layout).
Pass --output_dir / --cache_dir explicitly to override.

Examples
--------
Single model — outputs auto-land in runs/nonlinear/results/classification/:
    python scripts/evaluate_classification.py \\
        --model_path runs/nonlinear/model.model \\
        --test_data datasets/reference/clustering_0.tsv

Sweep a model directory, KNN only:
    python scripts/evaluate_classification.py \\
        --model_dir runs/ \\
        --train_data datasets/reference/clustering_0.tsv \\
        --test_data datasets/reference/clustering_1.tsv \\
        --classifiers knn --knn_k 1,3,5

Override output location explicitly:
    python scripts/evaluate_classification.py \\
        --model_path runs/nonlinear/model.model \\
        --test_data datasets/reference/clustering_0.tsv \\
        --output_dir results/classification/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from pathlib import Path


# ── Model loading helpers ────────────────────────────────────────────────────

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


def _infer_model_type(stem: str) -> str:
    """Infer the registered embedder name from a model filename stem."""
    stem_lower = stem.lower()
    for keyword, name in _NAME_KEYWORDS.items():
        if keyword in stem_lower:
            return name
    return None


def _load_model(model_path: str, model_type: str, device: str):
    from embedders import load_embedder
    return load_embedder(model_type, path=model_path, device=device)


def _collect_models(args):
    """Return list of (stem, path, type) tuples."""
    if args.model_path:
        stem = Path(args.model_path).parent.name
        mtype = args.model_type or _infer_model_type(stem)
        if mtype is None:
            raise ValueError(
                f"Cannot infer model type from '{stem}'. "
                "Pass --model_type explicitly."
            )
        return [(stem, args.model_path, mtype)]

    models = []
    for p in sorted(Path(args.model_dir).rglob("*.model")):
        stem = p.parent.name
        mtype = args.model_type or _infer_model_type(stem)
        if mtype is None:
            print(f"[SKIP] Cannot infer model type for {p}. Use --model_type to override.")
            continue
        models.append((stem, str(p), mtype))
    if not models:
        raise ValueError(f"No .model files found under {args.model_dir}")
    return models


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_labeled_tsv(path: str, max_seq_len: int, min_seq_len: int,
                      min_abundance: int) -> tuple:
    """Load (sequences, string_labels) from a TSV file."""
    import csv
    from utils import filter_sequences
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if rows and "\t" not in rows[0][0]:  # has header?
        rows = rows[1:]
    sequences = [r[0][:max_seq_len] for r in rows if len(r) >= 2]
    labels = [r[1] for r in rows if len(r) >= 2]
    sequences, labels = filter_sequences(sequences, labels,
                                         min_seq_len=min_seq_len,
                                         min_abundance=min_abundance)
    return sequences, labels


def _embed(model, sequences, cache_path=None):
    from embedders import get_embedding
    return get_embedding(model, sequences, cache_path=cache_path)


def _uncertainty_scores(result, seed):
    """Lower = more certain. Returns (scores, source_label)."""
    if result.kappa is not None:
        return 1.0 / (result.kappa + 1e-12), "kappa"
    if result.variance is not None:
        return result.variance.mean(axis=1), "variance"
    rng = np.random.default_rng(seed)
    return rng.random(len(result.mean)), "random"


# ── Classifiers ──────────────────────────────────────────────────────────────

def _fit_classifiers(X_train, y_train, classifiers, knn_k, logistic_c,
                     svm_c, svm_kernel, seed):
    """Fit all requested classifiers once. Returns list of (config_dict, clf)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    fitted = []
    if "knn" in classifiers:
        for k in knn_k:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            fitted.append(({"classifier": "knn", "k": k}, clf))
    if "logistic" in classifiers:
        for c in logistic_c:
            clf = LogisticRegression(C=c, max_iter=1000, random_state=seed)
            clf.fit(X_train, y_train)
            fitted.append(({"classifier": "logistic", "C": c}, clf))
    if "svm" in classifiers:
        for c in svm_c:
            clf = SVC(C=c, kernel=svm_kernel, random_state=seed)
            clf.fit(X_train, y_train)
            fitted.append(({"classifier": "svm", "kernel": svm_kernel, "C": c}, clf))
    return fitted


def _eval_classifiers(fitted_clfs, X_test, y_test):
    """Evaluate pre-fitted classifiers on (X_test, y_test)."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    results = []
    for config, clf in fitted_clfs:
        y_pred = clf.predict(X_test)
        result = dict(config)
        result.update({
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        })
        results.append(result)
    return results


# ── Figures ───────────────────────────────────────────────────────────────────


def _save_csv(all_results: dict, csv_path: str):
    import csv
    rows = []
    for model_name, results in all_results.items():
        for r in results:
            row = {"model": model_name}
            row.update({k: v for k, v in r.items() if k != "confusion_matrix"})
            rows.append(row)
    if not rows:
        return
    # Collect all keys across all rows (classifiers have different hyperparam fields)
    seen = {}
    for row in rows:
        for k in row:
            if k not in seen:
                seen[k] = None
    fieldnames = list(seen.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embeddings with KNN / Logistic / SVM classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_grp = parser.add_mutually_exclusive_group(required=True)
    model_grp.add_argument("--model_path", help="Single .model file.")
    model_grp.add_argument("--model_dir", help="Directory to sweep all .model files.")
    parser.add_argument("--model_type",
                        help="Embedder name (nonlinear/uncertaingen/pcl/kmerprofile/…). "
                             "Inferred from filename if omitted.")

    # Data
    parser.add_argument("--test_data", required=True,
                        help="Labeled TSV (sequence\\tlabel) used for evaluation.")
    parser.add_argument("--train_data",
                        help="Labeled TSV for fitting classifiers. "
                             "If omitted, 80/20 split of --test_data is used.")
    parser.add_argument("--val_data",
                        help="Optional labeled TSV for hyperparameter selection "
                             "(currently reported but not used for selection).")
    parser.add_argument("--output_dir", default=None,
                        help="Root output folder. Figures go to output_dir/figures/, "
                             "metrics to output_dir/metrics/. "
                             "Defaults to <model_dir>/results/classification/ when omitted.")

    # Classifiers
    parser.add_argument("--classifiers", default="knn,logistic,svm",
                        help="Comma-separated classifiers to run: knn,logistic,svm.")
    parser.add_argument("--knn_k", default="1,3,5,10",
                        help="Comma-separated k values for KNN.")
    parser.add_argument("--logistic_c", default="0.01,0.1,1.0,10.0",
                        help="Comma-separated C values for Logistic Regression.")
    parser.add_argument("--svm_kernel", default="rbf", choices=["rbf", "linear"],
                        help="SVM kernel.")
    parser.add_argument("--svm_c", default="0.1,1.0,10.0",
                        help="Comma-separated C values for SVM.")

    # Embedding
    parser.add_argument("--k", type=int, default=4, help="K-mer size.")
    parser.add_argument("--max_seq_len", type=int, default=20000)
    parser.add_argument("--min_seq_len", type=int, default=0)
    parser.add_argument("--min_abundance", type=int, default=0)
    parser.add_argument("--cache_dir",
                        help="Directory to cache embeddings (optional).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coverage_levels", default="100",
                        help="Comma-separated test coverage %% to evaluate "
                             "(e.g. '100,90,80,70'). Uses kappa/variance for "
                             "ordering if available, else random.")

    # Output
    parser.add_argument("--seed", type=int, default=26042024)

    args = parser.parse_args()
    _resolve_dirs(args, "classification")

    import datetime

    # Parse lists
    classifiers = [c.strip() for c in args.classifiers.split(",")]
    knn_k = [int(x) for x in args.knn_k.split(",")]
    logistic_c = [float(x) for x in args.logistic_c.split(",")]
    svm_c = [float(x) for x in args.svm_c.split(",")]
    coverage_levels = [int(c) for c in args.coverage_levels.split(",")]

    # Output directories
    met_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)

    # Load data
    print("Loading test data...")
    test_seqs, test_labels_str = _load_labeled_tsv(
        args.test_data, args.max_seq_len, args.min_seq_len, args.min_abundance
    )
    label2id = {l: i for i, l in enumerate(sorted(set(test_labels_str)))}
    test_labels = np.array([label2id[l] for l in test_labels_str])

    if args.train_data:
        print("Loading train data...")
        train_seqs, train_labels_str = _load_labeled_tsv(
            args.train_data, args.max_seq_len, args.min_seq_len, args.min_abundance
        )
        all_labels_str = sorted(set(train_labels_str) | set(test_labels_str))
        label2id_all = {l: i for i, l in enumerate(all_labels_str)}
        train_labels = np.array([label2id_all[l] for l in train_labels_str])
        test_labels = np.array([label2id_all.get(l, -1) for l in test_labels_str])
    else:
        # 50/50 split from test_data
        rng = np.random.default_rng(args.seed)
        n = len(test_seqs)
        idx = rng.permutation(n)
        split = int(0.5 * n)
        train_idx, eval_idx = idx[:split], idx[split:]
        train_seqs = [test_seqs[i] for i in train_idx]
        train_labels = test_labels[train_idx]
        test_seqs_eval = [test_seqs[i] for i in eval_idx]
        test_labels = test_labels[eval_idx]
        test_seqs = test_seqs_eval

    print(f"Train: {len(train_seqs)} seqs | Test: {len(test_seqs)} seqs")

    models = _collect_models(args)
    all_results: dict = {}  # {coverage_pct: {model_stem: [results]}}

    for model_stem, model_path, model_type in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_stem}  ({model_type})")
        print(f"{'='*60}")

        embedder = _load_model(model_path, model_type, args.device)

        # Build optional cache paths
        def _cache(split_name):
            if not args.cache_dir:
                return None
            # For model_dir sweeps, add model_stem subdir to separate multiple models.
            # For single model_path runs, cache_dir is already inside the model folder.
            subdir = model_stem if args.model_dir else ""
            return os.path.join(args.cache_dir, subdir, f"{split_name}.npy")

        print("Embedding train sequences...")
        X_train = _embed(embedder, train_seqs, cache_path=_cache("train")).point_estimate
        print("Embedding test sequences...")
        test_result = _embed(embedder, test_seqs, cache_path=_cache("test"))
        X_test = test_result.point_estimate

        if args.cache_dir:
            embed_meta = {
                "model_path": model_path,
                "model_type": model_type,
                "train_data": args.train_data or args.test_data,
                "test_data": args.test_data,
                "split": "50/50 random" if not args.train_data else "explicit train/test",
                "n_train": len(train_seqs),
                "n_test": len(test_seqs),
                "embedding_dim": int(X_train.shape[1]),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            subdir = model_stem if args.model_dir else ""
            embed_cfg_path = os.path.join(args.cache_dir, subdir, "embed_config.json")
            os.makedirs(os.path.dirname(embed_cfg_path), exist_ok=True)
            with open(embed_cfg_path, "w") as f:
                json.dump(embed_meta, f, indent=2)

        print(f"Fitting classifiers on {len(train_seqs)} train sequences...")
        fitted_clfs = _fit_classifiers(
            X_train, train_labels, classifiers,
            knn_k, logistic_c, svm_c, args.svm_kernel, args.seed,
        )
        print(f"  Fitted {len(fitted_clfs)} classifier configurations.")

        unc_scores, unc_source = _uncertainty_scores(test_result, args.seed)
        if unc_source == "random" and any(c != 100 for c in coverage_levels):
            print(f"  [WARN] No kappa/variance available — using random ordering for coverage.")

        for cov in coverage_levels:
            if cov == 100:
                X_test_cov = X_test
                labels_cov = test_labels
                n_kept = len(test_labels)
            else:
                n_keep = max(1, int(len(test_labels) * cov / 100))
                keep_idx = np.argsort(unc_scores)[:n_keep]
                X_test_cov = X_test[keep_idx]
                labels_cov = test_labels[keep_idx]
                n_kept = n_keep

            print(f"  Coverage {cov}%: evaluating on {n_kept}/{len(test_labels)} test points "
                  f"(uncertainty={unc_source})...")
            cov_results = _eval_classifiers(fitted_clfs, X_test_cov, labels_cov)
            for r in cov_results:
                r["coverage_pct"] = cov
                r["n_test_kept"] = n_kept
                r["uncertainty_source"] = unc_source
            all_results.setdefault(cov, {})[model_stem] = cov_results

        print(f"  Done: {len(fitted_clfs)} configs × {len(coverage_levels)} coverage levels.")

    # ── Save eval config (top-level, not per coverage) ──────────────────────
    eval_config = {
        **vars(args),
        "split": "50/50 random" if not args.train_data else "explicit train/test",
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "classifiers_run": classifiers,
        "knn_k": knn_k,
        "logistic_c": logistic_c,
        "svm_c": svm_c,
        "coverage_levels": coverage_levels,
        "timestamp": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(met_dir, "eval_config.json"), "w") as f:
        json.dump(eval_config, f, indent=2)

    # ── Save per-coverage metrics ────────────────────────────────────────────
    for cov, cov_results in all_results.items():
        cov_dir = os.path.join(args.output_dir, f"coverage_{cov}", "metrics")
        os.makedirs(cov_dir, exist_ok=True)

        slim = {m: [{k: v for k, v in r.items() if k != "confusion_matrix"}
                    for r in res]
                for m, res in cov_results.items()}
        json_path = os.path.join(cov_dir, "classification_results.json")
        with open(json_path, "w") as f:
            json.dump(slim, f, indent=2)

        csv_path = os.path.join(cov_dir, "classification_results.csv")
        _save_csv(cov_results, csv_path)
        print(f"Coverage {cov}%: saved to {cov_dir}")


if __name__ == "__main__":
    main()
