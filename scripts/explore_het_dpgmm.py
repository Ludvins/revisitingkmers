#!/usr/bin/env python3
"""explore_het_dpgmm.py

Systematic exploration of heteroscedastic DPGMM clustering quality.

Experiments
-----------
A  Embedder × Species sweep: which uncertainty source (UG, LLA, PCL kappa)
   helps het DPGMM most across reference / marine / plant species?
   Run with both algorithm=map (default) and algorithm=vbem.
B  DPGMM hyperparameter sweep: pca_dim × merge_threshold grid, plus
   individual sweeps of n_init, max_components, and (for vbem) alpha_prior,
   cov_inner_steps, cov_step_size.
C  Diagonal vs full cluster covariance in het EM (pca_dim ∈ {8, 16, 32}).
   Full covariance with algorithm=vbem raises NotImplementedError (expected).
D  LLA prior-optimization diagnostics: MacKay convergence trace,
   GGN vs empirical-Fisher comparison, variance distribution stats.
E  DPGMM convergence diagnostics: verbose per-iteration EM trace,
   merge-KL histogram, confidence-threshold sensitivity.
   Captures ELBO trace when algorithm=vbem.

Results are appended (JSONL) to <output_dir>/results.jsonl.
Embeddings are cached under <output_dir>/cache/.

Usage
-----
    # MAP-EM (default):
    python scripts/explore_het_dpgmm.py \\
        --data_dir data/dnabert/eval \\
        --output_dir results/dpgmm_exploration \\
        --experiments A,B,C,D,E \\
        --device cuda

    # VB-EM:
    python scripts/explore_het_dpgmm.py \\
        --data_dir data/dnabert/eval \\
        --output_dir results/dpgmm_exploration_vbem \\
        --experiments A,B,C,E \\
        --algorithm vbem \\
        --device cuda
"""

import argparse
import datetime
import json
import os
import sys
import warnings

import numpy as np

# ── ensure project root is on sys.path ───────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── project imports ──────────────────────────────────────────────────────────
from evaluation.binning import load_tsv_data
from evaluation.eval_utils import (
    compute_class_center_medium_similarity,
    count_high_quality_clusters,
)
from embedders import get_embedding, load_embedder
from embedders.base import EmbeddingResult
from clustering.dpgmm import DPGMMClusterer

# ── constants ─────────────────────────────────────────────────────────────────
_THRESHOLDS = [0.5, 0.9]
_MIN_SEQ_LEN = 2500
_MIN_ABUNDANCE = 10
_THRESHOLD_PCT_IDX = 6       # 70th-percentile index

# Default hyperparameters for MAP-EM path (unchanged from prior version)
_DEFAULT_MAP = dict(
    max_components=800, pca_dim=16, merge_threshold=2.0,
    n_init=5, min_bin_size=5, verbose=True,
    het_covariance_type="diag",
    algorithm="map",
)

# Default hyperparameters for VB-EM path
_DEFAULT_VBEM = dict(
    max_components=800, pca_dim=16, merge_threshold=2.0,
    n_init=5, min_bin_size=5, verbose=True,
    het_covariance_type="diag",
    algorithm="vbem",
    # VB-EM specific
    beta_0=1e-3,
    alpha_prior=None,       # defaults to 1/max_components inside DPGMMClusterer
    cov_inner_steps=3,
    cov_step_size=0.5,
)

# ── resumption / deduplication ────────────────────────────────────────────────
_DONE: set = set()

# Fields that uniquely identify a result row per experiment type
# "algorithm" is added to all rows so MAP and VB-EM results don't collide.
_KEY_FIELDS = {
    "A": ("experiment", "algorithm", "embedder", "species", "sample"),
    "B": ("experiment", "algorithm", "embedder", "species", "sample",
          "dpgmm_params.pca_dim", "dpgmm_params.merge_threshold",
          "dpgmm_params.n_init", "dpgmm_params.max_components",
          "dpgmm_params.alpha_prior", "dpgmm_params.cov_inner_steps",
          "dpgmm_params.cov_step_size"),
    "C": ("experiment", "algorithm", "embedder", "species", "sample",
          "het_covariance_type", "pca_dim"),
    "D": ("experiment", "base_model", "hessian", "species", "sample"),
    "E": ("experiment", "algorithm", "embedder", "species", "sample"),
}


def _make_key(result):
    exp = result.get("experiment", "?")
    fields = _KEY_FIELDS.get(exp, ("experiment", "algorithm", "embedder", "species", "sample"))
    parts = []
    for f in fields:
        if "." in f:
            top, sub = f.split(".", 1)
            parts.append(str(result.get(top, {}).get(sub, "")))
        else:
            parts.append(str(result.get(f, "")))
    return tuple(parts)


def _load_done(output_dir):
    """Populate _DONE from existing results.jsonl (called once at startup)."""
    global _DONE
    path = os.path.join(output_dir, "results.jsonl")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                _DONE.add(_make_key(result))
            except json.JSONDecodeError:
                pass
    print(f"[resume] loaded {len(_DONE)} completed results from {path}")


def _is_done(result):
    return _make_key(result) in _DONE


def _ts():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _save(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(result) + "\n")
    _DONE.add(_make_key(result))
    print(
        f"  [saved] exp={result.get('experiment')} alg={result.get('algorithm','?')} "
        f"emb={result.get('embedder','?')} "
        f"sp={result.get('species','?')} s{result.get('sample','?')}  "
        f"F1>0.5={result.get('f1_05_count','?')}  F1>0.9={result.get('f1_09_count','?')}"
    )


def _cache_path(cache_dir, emb_name, species, tag):
    return os.path.join(cache_dir, emb_name, species, f"{tag}.npy")


# ── data loading ──────────────────────────────────────────────────────────────

def _load_bin(data_dir, species, sample):
    return load_tsv_data(
        data_dir, species, sample, "binning",
        min_seq_len=_MIN_SEQ_LEN, min_abundance=_MIN_ABUNDANCE,
    )


def _load_ref(data_dir, species):
    return load_tsv_data(data_dir, species, 0, "clustering",
                         min_seq_len=0, min_abundance=0)


# ── embedding with manual cache ───────────────────────────────────────────────

def _embed_cached(embedder_obj, sequences, cache_path_):
    """Embed sequences with transparent on-disk caching."""
    if os.path.exists(cache_path_):
        mean = np.load(cache_path_)
        vp = cache_path_.replace(".npy", "_var.npy")
        kp = cache_path_.replace(".npy", "_kappa.npy")
        var = np.load(vp) if os.path.exists(vp) else None
        kap = np.load(kp) if os.path.exists(kp) else None
        return EmbeddingResult(mean=mean, variance=var, kappa=kap)

    result = embedder_obj.embed(sequences)

    os.makedirs(os.path.dirname(cache_path_), exist_ok=True)
    np.save(cache_path_, result.mean)
    if result.variance is not None:
        np.save(cache_path_.replace(".npy", "_var.npy"), result.variance)
    if result.kappa is not None:
        np.save(cache_path_.replace(".npy", "_kappa.npy"), result.kappa)
    return result


# ── threshold calibration ─────────────────────────────────────────────────────

def _calibrate(ref_result, ref_labels, metric):
    pcts, scale = compute_class_center_medium_similarity(
        ref_result.mean, ref_labels,
        metric=metric,
        variances=ref_result.variance,
        kappas=ref_result.kappa,
        scale=None,
    )
    return float(pcts[_THRESHOLD_PCT_IDX]), float(scale)


# ── run DPGMM + evaluate ──────────────────────────────────────────────────────

def _run_eval(embedding_result, labels, dpgmm_kwargs):
    """Fit DPGMMClusterer and evaluate. Returns (metrics_dict, clusterer)."""
    clusterer = DPGMMClusterer(**dpgmm_kwargs)
    pred = clusterer.fit_predict(embedding_result)
    res = count_high_quality_clusters(labels, pred, thresholds=_THRESHOLDS)
    metrics = {
        "f1_05_count": int(res["f1_counts"][0]),
        "f1_09_count": int(res["f1_counts"][1]),
        "n_clusters":  int(clusterer.n_active_),
        "n_assigned":  int(res["n_assigned"]),
        "n_total":     int(res["n_total"]),
    }
    return metrics, clusterer


def _default_dpgmm(algorithm):
    """Return the default DPGMM kwargs dict for the given algorithm."""
    return dict(_DEFAULT_VBEM if algorithm == "vbem" else _DEFAULT_MAP)


# ── LLA helpers ───────────────────────────────────────────────────────────────

def _ensure_linear2(model):
    """Alias mean_linear2 → linear2 for UncertainGen so LLA can hook it."""
    if not hasattr(model, "linear2") and hasattr(model, "mean_linear2"):
        model.linear2 = model.mean_linear2


def _fit_lla(base_model, train_csv, device, hessian="ggn",
             max_read_num=2000, verbose=True):
    """Fit LaplaceLastLayerEmbedder on a small paired-reads dataset."""
    from embedders.laplace_embedder import LaplaceLastLayerEmbedder
    from datasets.paired_reads import PairedReadsDataset
    from train import loss_functions

    _ensure_linear2(base_model)
    lla = LaplaceLastLayerEmbedder(base_model, prior_precision=1.0)

    dataset = PairedReadsDataset(
        train_csv,
        transform_func=base_model._feature_extractor.extract,
        neg_sample_per_pos=5,
        max_read_num=max_read_num,
        verbose=verbose,
    )
    lla.fit(dataset, loss_functions["bern"], device=device,
            loss_name="bern", hessian_factorization=hessian, verbose=verbose)
    return lla


def _mackay_trace(lla, n_steps=100):
    """Run MacKay fixed-point from tau=1 and return list of tau values."""
    import torch
    W = lla.base_model.linear2.weight.detach().cpu()
    b = lla.base_model.linear2.bias.detach().cpu()
    w_norm_sq = (W ** 2).sum().item() + (b ** 2).sum().item()
    N = lla.n_data
    kron_eigs = N * torch.outer(lla.S_A, lla.S_B)
    tau = 1.0
    trace = [float(tau)]
    for _ in range(n_steps):
        gamma = (kron_eigs / (tau + kron_eigs)).sum().item()
        tau_new = gamma / max(w_norm_sq, 1e-12)
        trace.append(float(tau_new))
        if abs(tau_new - tau) / max(abs(tau), 1e-12) < 1e-6:
            break
        tau = tau_new
    return trace


def _combine_ug_lla(ug_result, lla_result):
    """Combine UG aleatoric + LLA epistemic variance."""
    return EmbeddingResult(
        mean=ug_result.mean,
        variance=ug_result.variance + lla_result.variance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A — Embedder × Species sweep
# ─────────────────────────────────────────────────────────────────────────────

_EMB_CONFIGS = [
    # (name,  loader_type,  path,  metric,  extra)
    ("nonlinear_unlabeled",  "nonlinear",    "runs/nonlinear_unlabeled/model.model",   "l2",  {}),
    ("bern_nt_unlabeled",    "nonlinear",    "runs/bern_nt_unlabeled/model.model",     "l2",  {}),
    ("ug_unlabeled",         "uncertaingen", "runs/uncertaingen_unlabeled/model.model","l2",  {}),
    ("ug_bern_nt_unlabeled", "uncertaingen", "runs/ug_bern_nt_unlabeled/model.model",  "l2",  {}),
    ("pcl_kappa",            "pcl",          "runs/pcl_unlabeled/model.model",         "dot", {"kappa_to_variance": True}),
    # LLA variants are added dynamically inside run_exp_A
]


def run_exp_A(args, cache_dir):
    print(f"\n=== Experiment A: Embedder × Species Sweep (algorithm={args.algorithm}) ===")
    species_list = args.species.split(",")
    samples = [int(s) for s in args.samples.split(",")]

    # ── non-LLA embedders ────────────────────────────────────────────────────
    for emb_name, emb_type, model_path, metric, extra in _EMB_CONFIGS:
        model = load_embedder(emb_type, path=model_path, device=args.device)
        _run_emb_sweep(model, emb_name, metric, extra,
                       species_list, samples, args, cache_dir, "A")

    # ── LLA(NL) ──────────────────────────────────────────────────────────────
    print("\n-- Embedder: lla_nl_unlabeled (fitting LLA on GGN)...")
    nl_model = load_embedder("nonlinear", path="runs/nonlinear_unlabeled/model.model",
                             device=args.device)
    lla_nl = _fit_lla(nl_model, args.train_csv, args.device,
                      hessian="ggn", max_read_num=args.lla_max_reads)
    lla_nl.optimize_prior(method="mackay", verbose=True)
    _run_emb_sweep(lla_nl, "lla_nl_unlabeled", "l2", {},
                   species_list, samples, args, cache_dir, "A")

    # ── LLA(UG) + combined variance ───────────────────────────────────────────
    print("\n-- Embedder: lla_ug_unlabeled + combined variance...")
    ug_model = load_embedder("uncertaingen", path="runs/uncertaingen_unlabeled/model.model",
                             device=args.device)
    lla_ug = _fit_lla(ug_model, args.train_csv, args.device,
                      hessian="ggn", max_read_num=args.lla_max_reads)
    lla_ug.optimize_prior(method="mackay", verbose=True)

    for species in species_list:
        ug_ref_seqs, ref_labels = _load_ref(args.data_dir, species)
        ug_ref = _embed_cached(ug_model, ug_ref_seqs,
                               _cache_path(cache_dir, "ug_unlabeled", species, "ref"))
        lla_ref = _embed_cached(lla_ug, ug_ref_seqs,
                                _cache_path(cache_dir, "lla_ug_unlabeled", species, "ref"))
        combined_ref = _combine_ug_lla(ug_ref, lla_ref)
        threshold, _ = _calibrate(combined_ref, ref_labels, "l2")
        print(f"  {species} combined threshold={threshold:.4f}")

        for sample in samples:
            stub = {"experiment": "A", "algorithm": args.algorithm,
                    "embedder": "lla_ug_unlabeled_combined",
                    "species": species, "sample": sample}
            if _is_done(stub):
                print(f"  [skip] lla_ug_combined/{species}/s{sample} already done")
                continue
            ug_bin_seqs, bin_labels = _load_bin(args.data_dir, species, sample)
            ug_bin = _embed_cached(ug_model, ug_bin_seqs,
                                   _cache_path(cache_dir, "ug_unlabeled", species, sample))
            lla_bin = _embed_cached(lla_ug, ug_bin_seqs,
                                    _cache_path(cache_dir, "lla_ug_unlabeled", species, sample))
            combined_bin = _combine_ug_lla(ug_bin, lla_bin)
            dpgmm_kw = _default_dpgmm(args.algorithm)
            metrics, _ = _run_eval(combined_bin, bin_labels, dpgmm_kw)
            _save({**stub, "dpgmm_params": dpgmm_kw,
                   **metrics, "timestamp": _ts()},
                  args.output_dir)


def _run_emb_sweep(embedder_obj, emb_name, metric, extra,
                   species_list, samples, args, cache_dir, exp_tag):
    """Embed all (species, sample) pairs and run default DPGMM."""
    print(f"\n-- Embedder: {emb_name}")
    for species in species_list:
        ref_seqs, ref_labels = _load_ref(args.data_dir, species)
        ref_result = _embed_cached(
            embedder_obj, ref_seqs,
            _cache_path(cache_dir, emb_name, species, "ref"))
        threshold, _ = _calibrate(ref_result, ref_labels, metric)
        print(f"  {species}: threshold={threshold:.4f}")

        for sample in samples:
            stub = {"experiment": exp_tag, "algorithm": args.algorithm,
                    "embedder": emb_name, "species": species, "sample": sample}
            if _is_done(stub):
                print(f"  [skip] {emb_name}/{species}/s{sample} already done")
                continue

            bin_seqs, bin_labels = _load_bin(args.data_dir, species, sample)
            bin_result = _embed_cached(
                embedder_obj, bin_seqs,
                _cache_path(cache_dir, emb_name, species, sample))

            dpgmm_kw = _default_dpgmm(args.algorithm)
            if extra.get("kappa_to_variance"):
                dpgmm_kw["kappa_to_variance"] = True

            metrics, _ = _run_eval(bin_result, bin_labels, dpgmm_kw)
            _save({**stub, "dpgmm_params": dpgmm_kw,
                   **metrics, "timestamp": _ts()},
                  args.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B — Hyperparameter sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_exp_B(args, cache_dir, best_embedder="ug_unlabeled"):
    print(f"\n=== Experiment B: DPGMM Hyperparameter Sweep "
          f"(embedder={best_embedder}, algorithm={args.algorithm}) ===")
    species = "reference"
    samples = [5, 6]

    model = load_embedder("uncertaingen",
                          path="runs/uncertaingen_labeled/model.model",
                          device=args.device)

    ref_seqs, ref_labels = _load_ref(args.data_dir, species)
    ref_result = _embed_cached(model, ref_seqs,
                               _cache_path(cache_dir, best_embedder, species, "ref"))

    bin_data = {}
    for sample in samples:
        seqs, labels = _load_bin(args.data_dir, species, sample)
        result = _embed_cached(model, seqs,
                               _cache_path(cache_dir, best_embedder, species, sample))
        bin_data[sample] = (result, labels)

    # ── shared grid: pca_dim × merge_threshold ───────────────────────────────
    configs = []
    for pca_dim in [8, 16, 32, 64]:
        for mt in [0.5, 1.0, 1.5, 2.0, 3.0]:
            configs.append({"pca_dim": pca_dim, "merge_threshold": mt,
                            "n_init": 5, "max_components": 800,
                            "sweep": "grid"})

    # ── n_init sweep ─────────────────────────────────────────────────────────
    for n_init in [1, 10]:
        configs.append({"pca_dim": 16, "merge_threshold": 2.0,
                        "n_init": n_init, "max_components": 800,
                        "sweep": "n_init"})

    # ── max_components sweep ──────────────────────────────────────────────────
    for mc in [200, 400, 661]:
        configs.append({"pca_dim": 16, "merge_threshold": 2.0,
                        "n_init": 5, "max_components": mc,
                        "sweep": "max_components"})

    # ── VB-EM specific sweeps ─────────────────────────────────────────────────
    if args.algorithm == "vbem":
        # alpha_prior: controls DP concentration (how many clusters expected)
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            configs.append({"pca_dim": 16, "merge_threshold": 2.0,
                            "n_init": 5, "max_components": 800,
                            "alpha_prior": alpha,
                            "sweep": "alpha_prior"})
        # cov_inner_steps: gradient steps per outer iteration for C_k
        for steps in [1, 3, 5]:
            configs.append({"pca_dim": 16, "merge_threshold": 2.0,
                            "n_init": 5, "max_components": 800,
                            "cov_inner_steps": steps,
                            "sweep": "cov_inner_steps"})
        # cov_step_size: gradient step size (eta) for C_k
        for eta in [0.1, 0.3, 0.5, 1.0]:
            configs.append({"pca_dim": 16, "merge_threshold": 2.0,
                            "n_init": 5, "max_components": 800,
                            "cov_step_size": eta,
                            "sweep": "cov_step_size"})

    for cfg in configs:
        sweep_tag = cfg.pop("sweep")
        for sample in samples:
            stub = {"experiment": "B", "algorithm": args.algorithm,
                    "embedder": best_embedder,
                    "species": species, "sample": sample,
                    "dpgmm_params": cfg}
            if _is_done(stub):
                print(f"  [skip] B pca={cfg.get('pca_dim')} mt={cfg.get('merge_threshold')} "
                      f"s{sample} already done")
                continue
            bin_result, bin_labels = bin_data[sample]
            dpgmm_kw = {**_default_dpgmm(args.algorithm), **cfg, "verbose": False}
            metrics, _ = _run_eval(bin_result, bin_labels, dpgmm_kw)
            _save({**stub, "sweep": sweep_tag, **metrics, "timestamp": _ts()},
                  args.output_dir)
        cfg["sweep"] = sweep_tag  # restore for inspection


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C — Diagonal vs Full cluster covariance
# ─────────────────────────────────────────────────────────────────────────────

def run_exp_C(args, cache_dir, best_embedder="ug_unlabeled"):
    print(f"\n=== Experiment C: Diag vs Full Covariance "
          f"(embedder={best_embedder}, algorithm={args.algorithm}) ===")
    species = "reference"
    samples = [5, 6]

    model = load_embedder("uncertaingen",
                          path="runs/uncertaingen_labeled/model.model",
                          device=args.device)
    bin_data = {}
    for sample in samples:
        seqs, labels = _load_bin(args.data_dir, species, sample)
        result = _embed_cached(model, seqs,
                               _cache_path(cache_dir, best_embedder, species, sample))
        bin_data[sample] = (result, labels)

    cov_types = ["diag", "full"]
    if args.algorithm == "vbem":
        # full-covariance VB-EM raises NotImplementedError by design;
        # we still run it so the error is captured in results.jsonl.
        print("  Note: full-covariance with algorithm=vbem raises NotImplementedError (expected).")

    for cov_type in cov_types:
        for pca_dim in [8, 16, 32]:
            for sample in samples:
                stub = {"experiment": "C", "algorithm": args.algorithm,
                        "embedder": best_embedder,
                        "species": species, "sample": sample,
                        "het_covariance_type": cov_type, "pca_dim": pca_dim}
                if _is_done(stub):
                    print(f"  [skip] C cov={cov_type} pca={pca_dim} s{sample} already done")
                    continue
                bin_result, bin_labels = bin_data[sample]
                dpgmm_kw = {**_default_dpgmm(args.algorithm),
                            "het_covariance_type": cov_type,
                            "pca_dim": pca_dim, "verbose": False}
                try:
                    metrics, _ = _run_eval(bin_result, bin_labels, dpgmm_kw)
                except NotImplementedError as e:
                    print(f"  NotImplementedError cov={cov_type} pca={pca_dim} s{sample}: {e}")
                    metrics = {"f1_05_count": -1, "f1_09_count": -1,
                               "n_clusters": -1, "n_assigned": -1, "n_total": -1,
                               "error": str(e)}
                except Exception as e:
                    print(f"  ERROR cov={cov_type} pca={pca_dim} s{sample}: {e}")
                    metrics = {"f1_05_count": -1, "f1_09_count": -1,
                               "n_clusters": -1, "n_assigned": -1, "n_total": -1,
                               "error": str(e)}
                _save({**stub, "dpgmm_params": dpgmm_kw,
                       **metrics, "timestamp": _ts()},
                      args.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment D — LLA diagnostics (algorithm-independent)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp_D(args, cache_dir):
    print("\n=== Experiment D: LLA Prior Optimization Diagnostics ===")
    species = "reference"
    sample = 5

    bin_seqs, bin_labels = _load_bin(args.data_dir, species, sample)

    for base_name, base_type, model_path in [
        ("nl_labeled",    "nonlinear",    "runs/nonlinear_labeled/model.model"),
        ("ug_unlabeled",    "uncertaingen", "runs/uncertaingen_labeled/model.model"),
    ]:
        base_model = load_embedder(base_type, path=model_path, device=args.device)
        _ensure_linear2(base_model)

        for hessian in ["ggn", "ef"]:
            stub = {"experiment": "D", "base_model": base_name, "hessian": hessian,
                    "species": species, "sample": sample}
            if _is_done(stub):
                print(f"  [skip] D base={base_name} hessian={hessian} already done")
                continue
            print(f"\n  Fitting LLA({base_name}, hessian={hessian})...")
            lla = _fit_lla(base_model, args.train_csv, args.device,
                           hessian=hessian, max_read_num=args.lla_max_reads)

            tau_trace = _mackay_trace(lla, n_steps=100)
            lla.prior_precision = tau_trace[-1]

            lla_result = _embed_cached(lla, bin_seqs,
                                       _cache_path(cache_dir,
                                                   f"lla_{base_name}_{hessian}",
                                                   species, sample))
            var = lla_result.variance
            var_stats = {
                "mean": float(var.mean()),
                "std": float(var.std()),
                "p5": float(np.percentile(var, 5)),
                "p25": float(np.percentile(var, 25)),
                "p50": float(np.percentile(var, 50)),
                "p75": float(np.percentile(var, 75)),
                "p95": float(np.percentile(var, 95)),
            }

            combined_stats = None
            if base_type == "uncertaingen":
                ug_result = _embed_cached(
                    base_model, bin_seqs,
                    _cache_path(cache_dir, "ug_unlabeled", species, sample))
                if ug_result.variance is not None:
                    combined = ug_result.variance + var
                    combined_stats = {
                        "mean": float(combined.mean()),
                        "p50": float(np.percentile(combined, 50)),
                        "p95": float(np.percentile(combined, 95)),
                    }

            # Use MAP-EM for LLA diagnostics (algorithm-neutral baseline)
            metrics, clusterer = _run_eval(lla_result, bin_labels, dict(_DEFAULT_MAP))

            from embedders.base import EmbeddingResult as _ER
            pred = clusterer.fit_predict(_ER(mean=lla_result.mean, variance=var))
            per_sample_var = var.mean(axis=1)
            unassigned = (pred == -1).astype(float)
            corr = float(np.corrcoef(per_sample_var, unassigned)[0, 1])

            _save({
                **stub,
                "prior_precision_final": float(lla.prior_precision),
                "tau_trace": tau_trace,
                "tau_trace_length": len(tau_trace),
                "variance_stats": var_stats,
                "combined_variance_stats": combined_stats,
                "var_unassigned_corr": corr,
                **metrics,
                "timestamp": _ts(),
            }, args.output_dir)
            print(f"    tau_trace: {tau_trace[:5]} ... {tau_trace[-1]:.4f} "
                  f"({len(tau_trace)} steps)")
            print(f"    var_mean={var_stats['mean']:.6f}, corr(var, unassigned)={corr:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment E — DPGMM convergence diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_exp_E(args, cache_dir, best_embedder="ug_unlabeled"):
    print(f"\n=== Experiment E: DPGMM Convergence Diagnostics "
          f"(embedder={best_embedder}, algorithm={args.algorithm}) ===")
    samples = [5]

    model = load_embedder("uncertaingen",
                          path="runs/uncertaingen_labeled/model.model",
                          device=args.device)

    for species in ["reference", "marine"]:
        for sample in samples:
            stub = {"experiment": "E", "algorithm": args.algorithm,
                    "embedder": best_embedder,
                    "species": species, "sample": sample}
            if _is_done(stub):
                print(f"  [skip] E {best_embedder}/{species}/s{sample} already done")
                continue
            seqs, labels = _load_bin(args.data_dir, species, sample)
            bin_result = _embed_cached(model, seqs,
                                       _cache_path(cache_dir, best_embedder, species, sample))

            dpgmm_kw = {
                **_default_dpgmm(args.algorithm),
                "n_init": 3,        # fewer restarts for speed
                "verbose": True,
                "collect_diagnostics": True,
            }
            print(f"\n  Running verbose DPGMM on {species} sample {sample}...")
            metrics, clusterer = _run_eval(bin_result, labels, dpgmm_kw)

            diag = clusterer.diagnostics_ or {}
            merge_diag = clusterer.merge_diagnostics_ or []

            best_run = diag.get("best_run", [])
            # For VB-EM, map_obj stores the ELBO; for MAP-EM it stores the log-likelihood.
            obj_trace    = [d["map_obj"] for d in best_run]
            k_trace      = [d["K_active"] for d in best_run]
            pruned_trace = [d["n_pruned"] for d in best_run]

            merge_kl = [m["kl"] for m in merge_diag]

            clusterer2 = DPGMMClusterer(
                **{**_default_dpgmm(args.algorithm), "n_init": 1, "verbose": False})
            pred2 = clusterer2.fit_predict(bin_result)
            frac_unassigned = float((pred2 == -1).sum() / len(pred2))

            _save({
                **stub,
                "n_iter_best": int(clusterer.n_iter_),
                "converged": bool(clusterer.converged_),
                "n_restarts": dpgmm_kw["n_init"],
                # obj_trace is log-likelihood for MAP-EM, ELBO for VB-EM
                "obj_trace": obj_trace[:50],
                "k_trace": k_trace[:50],
                "pruned_trace": pruned_trace[:50],
                "n_merges": len(merge_diag),
                "merge_kl_min": float(min(merge_kl)) if merge_kl else None,
                "merge_kl_max": float(max(merge_kl)) if merge_kl else None,
                "merge_kl_median": float(np.median(merge_kl)) if merge_kl else None,
                "merge_kl_p90": float(np.percentile(merge_kl, 90)) if merge_kl else None,
                "frac_unassigned": frac_unassigned,
                "dpgmm_params": dpgmm_kw,
                **metrics,
                "timestamp": _ts(),
            }, args.output_dir)
            print(f"    iters={clusterer.n_iter_}, converged={clusterer.converged_}, "
                  f"merges={len(merge_diag)}, frac_unassigned={frac_unassigned:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Systematic exploration of heteroscedastic DPGMM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",     default="data/dnabert/eval",
                   help="Root directory for evaluation TSV files.")
    p.add_argument("--output_dir",   default="results/dpgmm_exploration",
                   help="Where to write results.jsonl and cache/.")
    p.add_argument("--experiments",  default="A,B,C,D,E",
                   help="Comma-separated list of experiments to run (A,B,C,D,E).")
    p.add_argument("--algorithm",    default="map", choices=["map", "vbem"],
                   help="Clustering algorithm: 'map' (MAP-EM) or 'vbem' (truncated DP-VBEM).")
    p.add_argument("--species",      default="reference,marine,plant",
                   help="Comma-separated species for Exp A.")
    p.add_argument("--samples",      default="5",
                   help="Comma-separated sample indices for Exp A.")
    p.add_argument("--device",       default="cuda",
                   help="PyTorch device string ('cuda' or 'cpu').")
    p.add_argument("--seed",         default=26042024, type=int)
    p.add_argument("--train_csv",    default="data/dnabert/train/val_48k.csv",
                   help="CSV of paired reads used to fit LLA Hessian.")
    p.add_argument("--lla_max_reads", default=2000, type=int,
                   help="Max paired reads to load for LLA fitting (speed/memory trade-off).")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    np.random.seed(args.seed)
    _load_done(args.output_dir)

    exps = [e.strip().upper() for e in args.experiments.split(",")]
    print(f"Experiments to run: {exps}")
    print(f"Algorithm:   {args.algorithm}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Cache dir:   {cache_dir}")
    print(f"Device:      {args.device}")

    if "A" in exps:
        run_exp_A(args, cache_dir)
    if "B" in exps:
        run_exp_B(args, cache_dir)
    if "C" in exps:
        run_exp_C(args, cache_dir)
    if "D" in exps:
        run_exp_D(args, cache_dir)
    if "E" in exps:
        run_exp_E(args, cache_dir)

    print(f"\nAll done. Results in {args.output_dir}/results.jsonl")


if __name__ == "__main__":
    main()
