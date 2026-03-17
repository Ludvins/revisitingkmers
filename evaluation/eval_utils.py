import logging
import numpy as np
import sklearn.metrics
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)
from metrics.similarity import (
    compute_similarity,
    compute_probabilistic_similarity,
    compute_vmf_similarity,
    _ppk_log_self_similarity,
)


def align_labels_via_hungarian_algorithm(true_labels, predicted_labels) -> dict:
    """Align predicted cluster labels with true labels using the Hungarian algorithm.

    Returns
    -------
    dict
        Dictionary mapping predicted labels to aligned true labels.
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.ndim != 1:
        raise ValueError(
            f"true_labels must be 1D, got shape {true_labels.shape}"
        )
    if predicted_labels.ndim != 1:
        raise ValueError(
            f"predicted_labels must be 1D, got shape {predicted_labels.shape}"
        )
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"true_labels and predicted_labels must have the same length, "
            f"got {len(true_labels)} and {len(predicted_labels)}"
        )

    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    # Build confusion matrix: rows=true, cols=predicted
    np.add.at(confusion_matrix, (true_labels, predicted_labels), 1)

    # Hungarian algorithm: find optimal 1-to-1 label alignment
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    return {
        predicted_label: true_label
        for true_label, predicted_label in zip(row_ind, col_ind)
    }


def compute_class_center_medium_similarity(embeddings: np.ndarray,
                                            labels: np.ndarray,
                                            metric: str = "dot",
                                            variances: np.ndarray = None,
                                            kappas: np.ndarray = None,
                                            scale: float = None,
                                            k_form: str = "adaptive",
                                            alpha: float = 1.0) -> tuple[list[float], float]:
    """Compute percentile values of per-sample similarity to class centers.

    Used to determine the clustering similarity threshold from reference data.
    Supports three modes: deterministic (metric-based), Gaussian (variance),
    and vMF (kappa concentration).

    When ``scale`` is None, auto-calibrates so that median intra-class
    similarity is ~0.5.  The calibrated scale is returned alongside the
    percentile values so it can be reused for clustering.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) array of mean embeddings.
    labels : np.ndarray
        (N,) integer class labels.
    metric : str
        Similarity metric (used only in deterministic mode).
    variances : np.ndarray, optional
        (N, D) diagonal variances for Gaussian probabilistic similarity.
    kappas : np.ndarray, optional
        (N,) vMF concentration parameters for vMF similarity.
    scale : float, optional
        Scale coefficient. If None, auto-calibrates (recommended).
    k_form : str
        Kernel form for probabilistic similarity.
    alpha : float
        Regularizer for adaptive/identity forms.

    Returns
    -------
    tuple[list[float], float]
        (percentile_values, calibrated_scale) — percentile values at
        [10, 20, ..., 90] and the scale used (auto-calibrated or as given).
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(
            f"embeddings must be a numpy array, got {type(embeddings).__name__}"
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D, got shape {embeddings.shape}"
        )
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be 1D, got shape {labels.shape}"
        )
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"embeddings and labels must have the same number of samples, "
            f"got {embeddings.shape[0]} and {labels.shape[0]}"
        )
    if variances is None and kappas is None:
        valid_metrics = {"dot", "l2", "euclidean", "l1"}
        if metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {valid_metrics}, got '{metric}'"
            )

    use_vmf = kappas is not None
    use_gaussian = variances is not None and not use_vmf

    # Sort by label so we can split into contiguous class groups
    idx = np.argsort(labels)
    embeddings = embeddings[idx]
    labels = labels[idx]
    if use_gaussian:
        variances = variances[idx]
    if use_vmf:
        kappas = kappas[idx]

    unique_labels, counts = np.unique(labels, return_counts=True)

    # Split indices for np.split: cumulative count of each class except last
    cumsum = np.cumsum(counts[:-1])
    class_groups = np.split(embeddings, cumsum)
    class_means = np.array([g.mean(axis=0) for g in class_groups])

    if use_gaussian:
        var_groups = np.split(variances, cumsum)
        class_center_vars = np.array([g.mean(axis=0) for g in var_groups])

    if use_vmf:
        kappa_groups = np.split(kappas, cumsum)
        class_center_kappas = np.array([g.mean() for g in kappa_groups])

    # Step 1: Compute raw distances (before exp) to class centers
    all_raw_dist = np.empty(len(embeddings))
    offset = 0
    for i, group in enumerate(class_groups):
        if use_vmf:
            cos_sim = group @ class_means[i]  # (n_i,)
            d_cos = 1.0 - cos_sim
            if k_form == "adaptive":
                inv_ki = 1.0 / np.clip(kappa_groups[i], 1e-6, None)
                inv_kj = 1.0 / max(class_center_kappas[i], 1e-6)
                raw = d_cos / (inv_ki + inv_kj + alpha)
            elif k_form == "ppk":
                # PPK with Cauchy-Schwarz normalization so values are in [0, 1]
                from metrics.similarity import _log_vmf_norm_const
                D = group.shape[1]
                ki = kappa_groups[i].astype(np.float64)
                kj = float(class_center_kappas[i])
                kappa_prod_sq = ki**2 + kj**2 + 2 * ki * kj * cos_sim
                kappa_prod = np.sqrt(np.clip(kappa_prod_sq, 1e-12, None))
                log_c_i = _log_vmf_norm_const(ki, D)
                log_c_j = _log_vmf_norm_const(np.array([kj]), D)[0]
                log_c_prod = _log_vmf_norm_const(kappa_prod, D)
                log_sim = log_c_i + log_c_j - log_c_prod
                # Normalize by self-similarity: log_K_norm = log_K - 0.5*(log_K_ii + log_K_jj)
                log_self_i = _ppk_log_self_similarity(ki, D)
                log_self_j = _ppk_log_self_similarity(np.array([kj]), D)[0]
                log_sim_norm = log_sim - 0.5 * (log_self_i + log_self_j)
                # raw = -log_sim_norm >= 0 (distance form for calibration)
                raw = -log_sim_norm
            else:  # cosine
                raw = d_cos
        elif use_gaussian:
            delta = group - class_means[i]
            cvar = var_groups[i] + class_center_vars[i]
            if k_form == "identity":
                scale_d = cvar + alpha
                raw = np.sum(0.5 * np.log(scale_d / alpha) + 0.5 * delta * delta / scale_d,
                             axis=1)
            elif k_form == "expected_distance":
                raw = np.sum(delta * delta, axis=1) + np.sum(cvar, axis=1)
            else:
                raw = np.sum(delta * delta / (cvar + alpha), axis=1)
        elif metric in ("l2", "euclidean"):
            raw = np.sum((group - class_means[i]) ** 2, axis=1)
        elif metric == "l1":
            raw = np.sum(np.abs(group - class_means[i]), axis=1)
        else:  # dot — no exponential kernel, skip calibration
            raw = None
        if raw is not None:
            all_raw_dist[offset:offset + len(group)] = raw
        offset += len(group)

    # Step 2: Auto-calibrate scale if needed
    if metric == "dot" and not use_gaussian and not use_vmf:
        # Dot product doesn't use exp kernel; scale is meaningless
        calibrated_scale = 1.0
    elif use_vmf and k_form == "cosine_direct":
        # Raw cosine similarity — no exponential kernel, scale unused
        calibrated_scale = 1.0
    elif scale is not None:
        calibrated_scale = scale
    else:
        median_dist = np.median(all_raw_dist)
        if median_dist > 1e-12:
            # Set scale so that median intra-class similarity = 0.5
            calibrated_scale = np.log(2) / median_dist
        else:
            calibrated_scale = 1.0
        print(f"Auto-calibrated scale: {calibrated_scale:.6f} "
              f"(median raw distance: {median_dist:.4f})")

    # Step 3: Compute similarities with calibrated scale
    all_similarities = np.empty(len(embeddings))
    offset = 0
    for i, group in enumerate(class_groups):
        if use_vmf:
            sims = compute_vmf_similarity(
                group, kappa_groups[i], class_means[i], class_center_kappas[i],
                scale=calibrated_scale, k_form=k_form, alpha=alpha,
            )
        elif use_gaussian:
            sims = compute_probabilistic_similarity(
                group, var_groups[i], class_means[i], class_center_vars[i],
                scale=calibrated_scale, k_form=k_form, alpha=alpha,
            )
        elif metric == "dot":
            sims = compute_similarity(group, class_means[i], metric)
        else:
            sims = compute_similarity(
                group, class_means[i], metric, scale=calibrated_scale,
            )
        all_similarities[offset:offset + len(group)] = sims
        offset += len(group)

    all_similarities.sort()
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    percentile_values = [
        all_similarities[int(p / 100 * len(embeddings))] for p in percentiles
    ]
    logger.debug("Percentile values: %s", percentile_values)

    return percentile_values, calibrated_scale


def count_high_quality_clusters(true_labels, predicted_labels,
                                thresholds=None, mode="discard",
                                precomputed_alignment=None) -> dict:
    """Evaluate clustering quality with precision, recall, F1, and accuracy.

    For each threshold tau, counts how many classes achieve metric > tau.
    Supports two modes for handling unassigned (``-1``) points.

    Parameters
    ----------
    true_labels : array-like
        Ground-truth integer class labels ``(N,)``.
    predicted_labels : array-like
        Predicted cluster labels ``(N,)``. ``-1`` means unassigned.
    thresholds : list[float], optional
        Thresholds to evaluate. Default: ``[0.1, 0.2, ..., 0.9]``.
    precomputed_alignment : dict, optional
        Pre-computed mapping from predicted labels to true labels (as
        returned by ``align_labels_via_hungarian_algorithm``).  When
        provided, the function skips its internal Hungarian alignment
        and uses this mapping instead.  Useful when multiple evaluation
        calls must share a single, fixed alignment computed on the full
        (pre-rejection) clustering.
    mode : str
        ``"discard"``: exclude unassigned points from metrics (default).
        ``"garbage"``: treat unassigned points as a single garbage cluster
        that is never aligned to any true class — every garbage point is
        always misclassified.

    Returns
    -------
    dict
        ``per_class_f1``, ``per_class_precision``, ``per_class_recall``:
            sorted ndarrays of per-class scores.
        ``accuracy``: float, overall accuracy.
        ``thresholds``: list of threshold values used.
        ``f1_counts``, ``precision_counts``, ``recall_counts``:
            list of int, # classes exceeding each threshold.
        ``counts``: alias for ``f1_counts`` (backward compat).
        ``alignment``: dict mapping predicted -> true labels.
        ``n_assigned``: int, number of assigned (non -1) samples.
        ``n_total``: int, total number of samples.
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    mask = predicted_labels != -1
    n_assigned = int(mask.sum())
    n_total = len(true_labels)

    empty_result = {
        "per_class_f1": np.array([]),
        "per_class_precision": np.array([]),
        "per_class_recall": np.array([]),
        "accuracy": 0.0,
        "thresholds": thresholds,
        "f1_counts": [0] * len(thresholds),
        "precision_counts": [0] * len(thresholds),
        "recall_counts": [0] * len(thresholds),
        "counts": [0] * len(thresholds),
        "alignment": {},
        "n_assigned": 0,
        "n_total": n_total,
    }

    if n_assigned == 0:
        return empty_result

    # Use precomputed alignment if provided, otherwise compute via Hungarian
    if precomputed_alignment is not None:
        alignment = precomputed_alignment
    else:
        alignment = align_labels_via_hungarian_algorithm(
            true_labels[mask], predicted_labels[mask]
        )

    if mode == "garbage":
        # Garbage label: a label that matches no true class
        garbage_label = int(true_labels.max()) + 1
        aligned = np.array([
            alignment.get(p, garbage_label) if p != -1 else garbage_label
            for p in predicted_labels
        ])
        eval_true = true_labels
        eval_pred = aligned
    else:
        # Discard mode: only evaluate assigned points
        aligned = np.array([alignment.get(p, -1) for p in predicted_labels[mask]])
        eval_true = true_labels[mask]
        eval_pred = aligned

    per_class_precision = sklearn.metrics.precision_score(
        eval_true, eval_pred, average=None, zero_division=0
    )
    per_class_recall = sklearn.metrics.recall_score(
        eval_true, eval_pred, average=None, zero_division=0
    )
    per_class_f1 = sklearn.metrics.f1_score(
        eval_true, eval_pred, average=None, zero_division=0
    )
    accuracy = float(np.mean(eval_pred == eval_true))

    per_class_precision.sort()
    per_class_recall.sort()
    per_class_f1.sort()

    f1_counts = [int((per_class_f1 > t).sum()) for t in thresholds]
    precision_counts = [int((per_class_precision > t).sum()) for t in thresholds]
    recall_counts = [int((per_class_recall > t).sum()) for t in thresholds]

    return {
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "accuracy": accuracy,
        "thresholds": thresholds,
        "f1_counts": f1_counts,
        "counts": f1_counts,           # alias for backward compatibility
        "precision_counts": precision_counts,
        "recall_counts": recall_counts,
        "alignment": alignment,
        "n_assigned": n_assigned,
        "n_total": n_total,
    }
