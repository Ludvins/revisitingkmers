#!/usr/bin/env python3
"""compare_dpgmm_sklearn.py

Compare three clustering methods on real metagenomics data:
  1. sklearn BayesianGaussianMixture  — standard DP-GMM, ignores per-sample noise
  2. DPGMMClusterer algorithm='map'   — het MAP-EM, uses UG per-sample variances
  3. DPGMMClusterer algorithm='vbem'  — het truncated DP-VBEM, uses UG variances

Embeddings: UncertainGen unlabeled model (runs/uncertaingen_unlabeled/).
Datasets:   reference binning_5, marine binning_5.

Run:
    E:/gnome/Scripts/python.exe scripts/compare_dpgmm_sklearn.py
"""

import os
import sys
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import csv
from sklearn.model_selection import train_test_split
from evaluation.eval_utils import count_high_quality_clusters
from embedders import load_embedder
from embedders.base import EmbeddingResult
from clustering.dpgmm import DPGMMClusterer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── config ──────────────────────────────────────────────────────────────────
_UG_MODEL     = "runs/uncertaingen_unlabeled/model.model"
_DEVICE       = "cuda"        # change to "cuda" if available
_PCA_DIM      = 16
_MAX_K        = 500          # truncation level (matches notebook)
_ALPHA_PRIOR  = 1000.0       # DP concentration (high = many clusters expected)
_N_INIT       = 1
_MAX_ITER     = 500
_MERGE_THR    = 1.5
_SEED         = 26042024
_THRESHOLDS   = [0.5, 0.9]
_TSV          = "data/dnabert/eval/reference/binning_5.tsv"
_CACHE_DIR    = "results/compare_cache"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _load_bin():
    """Load reference/binning_5.tsv, stratified 50/50 split, return test half."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(_TSV) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        rows = list(reader)
    all_seqs       = [r[0] for r in rows]
    all_labels_str = [r[1] for r in rows]
    _, test_seqs, _, test_labels_str = train_test_split(
        all_seqs, all_labels_str, test_size=0.5, random_state=_SEED,
        stratify=all_labels_str)
    unique_labels = sorted(set(all_labels_str))
    lab2id = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([lab2id[l] for l in test_labels_str])
    return test_seqs, labels


def _embed_or_load(embedder_obj, sequences):
    """Embed with transparent caching."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    mp = os.path.join(_CACHE_DIR, "reference_b5_test_mean.npy")
    vp = os.path.join(_CACHE_DIR, "reference_b5_test_var.npy")
    if os.path.exists(mp):
        mean = np.load(mp)
        var  = np.load(vp) if os.path.exists(vp) else None
        print(f"    [cache] loaded {mean.shape[0]} embeddings from {mp}")
        return EmbeddingResult(mean=mean, variance=var)
    print(f"    Embedding {len(sequences)} sequences...")
    t0 = time.time()
    result = embedder_obj.embed(sequences)
    print(f"    Done in {time.time()-t0:.1f}s  "
          f"mean={result.mean.shape}  var={'yes' if result.variance is not None else 'no'}")
    np.save(mp, result.mean)
    if result.variance is not None:
        np.save(vp, result.variance)
    return result


def _pca_reduce(mean, variance, pca_dim):
    """Apply PCA to mean; project variance through PCA rotation (diag approx)."""
    scaler = StandardScaler()
    mean_s = scaler.fit_transform(mean)
    pca = PCA(n_components=pca_dim, random_state=_SEED)
    mean_p = pca.fit_transform(mean_s)
    var_p = None
    if variance is not None:
        # Project diagonal variance: var_proj[i,d] = sum_j V[d,j]^2 * var[i,j]
        # where V = pca.components_ (pca_dim x D).  Same approximation used inside
        # DPGMMClusterer when pca_dim is set, but done manually here so sklearn
        # sees the same reduced space.
        V = pca.components_   # (pca_dim, D)
        var_p = variance @ (V ** 2).T   # (N, pca_dim)
    return mean_p, var_p


def _eval(true_labels, pred_labels):
    res = count_high_quality_clusters(true_labels, pred_labels, thresholds=_THRESHOLDS)
    n_bins = int(np.unique(pred_labels[pred_labels != -1]).size)
    frac_assigned = float((pred_labels != -1).sum() / len(pred_labels))
    return {
        "F1>0.5": int(res["f1_counts"][0]),
        "F1>0.9": int(res["f1_counts"][1]),
        "n_bins": n_bins,
        "assigned%": round(100.0 * frac_assigned, 1),
    }


def _header():
    cols = ["Method", "F1>0.5", "F1>0.9", "n_bins", "assigned%", "time(s)"]
    widths = [22, 7, 7, 7, 10, 8]
    row = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(row)
    print(sep)
    print(row)
    print(sep)
    return widths


def _row(widths, method, metrics, elapsed):
    vals = [method,
            str(metrics["F1>0.5"]), str(metrics["F1>0.9"]),
            str(metrics["n_bins"]), f"{metrics['assigned%']}%",
            f"{elapsed:.1f}"]
    print(" | ".join(v.ljust(w) for v, w in zip(vals, widths)))


# ─── sklearn BGM baseline ────────────────────────────────────────────────────

def _run_sklearn_bgm(mean_pca, alpha_prior):
    """Run sklearn BayesianGaussianMixture and return (pred_labels, elapsed)."""
    bgm = BayesianGaussianMixture(
        n_components=_MAX_K,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_distribution",
        weight_concentration_prior=alpha_prior,
        max_iter=_MAX_ITER,
        n_init=_N_INIT,
        random_state=_SEED,
        verbose=0,
    )
    t0 = time.time()
    bgm.fit(mean_pca)
    pred_raw = bgm.predict(mean_pca)
    elapsed = time.time() - t0

    # Reject small clusters (< min_bin_size), matching notebook post-processing
    from collections import Counter
    counts = Counter(pred_raw.tolist())
    small = {c for c, n in counts.items() if n < 5}
    pred = np.where(np.isin(pred_raw, list(small)), -1, pred_raw)
    # remap to contiguous
    active_set = sorted(set(pred[pred != -1].tolist()))
    mapping = {k: i for i, k in enumerate(active_set)}
    pred_final = np.array([mapping.get(p, -1) for p in pred])
    return pred_final, elapsed


# ─── het DPGMM (map or vbem) ─────────────────────────────────────────────────

def _run_het_dpgmm(er_pca, algorithm, alpha_prior):
    """Run our DPGMMClusterer with given algorithm."""
    kw = dict(
        max_components=_MAX_K,
        pca_dim=None,               # already reduced
        merge_threshold=_MERGE_THR,
        n_init=_N_INIT,
        min_bin_size=5,
        verbose=True,
        het_covariance_type="diag",
        algorithm=algorithm,
        alpha_prior=alpha_prior,
        random_state=_SEED,
    )
    if algorithm == "vbem":
        kw.update(beta_0=1e-3, cov_inner_steps=3, cov_step_size=0.5)

    clusterer = DPGMMClusterer(**kw)
    t0 = time.time()
    pred = clusterer.fit_predict(er_pca)
    elapsed = time.time() - t0
    print(f"      converged={getattr(clusterer, 'converged_', '?')}  "
          f"n_iter={getattr(clusterer, 'n_iter_', '?')}  "
          f"K_active={getattr(clusterer, 'n_active_', '?')}")
    return pred, elapsed


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading UncertainGen unlabeled model...")
    ug = load_embedder("uncertaingen", path=_UG_MODEL, device=_DEVICE)

    widths = _header()

    print(f"\n{'='*60}")
    print(f"  Dataset: reference/binning_5 (stratified test 50%, seed={_SEED})")

    sequences, true_labels = _load_bin()
    N = len(sequences)
    K_true = len(np.unique(true_labels))
    print(f"  N={N}  K_true={K_true}")

    print("  Embedding...")
    er = _embed_or_load(ug, sequences)
    print(f"  Var range: [{er.variance.min():.4f}, {er.variance.max():.4f}]"
          if er.variance is not None else "  No variance available")

    print(f"  PCA -> {_PCA_DIM}d ...")
    mean_p, var_p = _pca_reduce(er.mean, er.variance, _PCA_DIM)
    er_pca = EmbeddingResult(mean=mean_p, variance=var_p)

    # ── sklearn BGM ──────────────────────────────────────────────────────
    print(f"\n  [1/3] sklearn BayesianGaussianMixture (alpha={_ALPHA_PRIOR})...")
    pred_sk, t_sk = _run_sklearn_bgm(mean_p, _ALPHA_PRIOR)
    m_sk = _eval(true_labels, pred_sk)
    _row(widths, "sklearn BGM", m_sk, t_sk)

    # ── het DPGMM MAP ────────────────────────────────────────────────────
    print(f"\n  [2/3] Het DPGMM (algorithm=map, alpha={_ALPHA_PRIOR})...")
    pred_map, t_map = _run_het_dpgmm(er_pca, "map", _ALPHA_PRIOR)
    m_map = _eval(true_labels, pred_map)
    _row(widths, "het DPGMM map", m_map, t_map)

    # ── het DPGMM VB-EM ──────────────────────────────────────────────────
    print(f"\n  [3/3] Het DPGMM (algorithm=vbem, alpha={_ALPHA_PRIOR})...")
    pred_vb, t_vb = _run_het_dpgmm(er_pca, "vbem", _ALPHA_PRIOR)
    m_vb = _eval(true_labels, pred_vb)
    _row(widths, "het DPGMM vbem", m_vb, t_vb)

    print("-" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
