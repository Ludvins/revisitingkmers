"""Image clustering evaluation benchmark.

Embeds test set images, clusters with K-Means, and evaluates against
ground-truth labels using accuracy, NMI, ARI, and F1.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    f1_score, accuracy_score,
)
from evaluation.eval_utils import align_labels_via_hungarian_algorithm
from utils.progress import pbar


def evaluate_image_clustering(embedder, dataset_name: str = "mnist",
                               data_root: str = "./data",
                               max_samples: int = 0,
                               n_clusters: int = 10,
                               batch_size: int = 1000,
                               seed: int = 0,
                               verbose: bool = True) -> dict:
    """Evaluate embedder on image clustering task.

    Parameters
    ----------
    embedder : object
        Trained embedder with embed() method.
    dataset_name : str
        "mnist" or "cifar10".
    data_root : str
        Root directory for torchvision data.
    max_samples : int
        Max test samples (0 = all).
    n_clusters : int
        Number of clusters for K-Means.
    batch_size : int
        Batch size for embedding.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Dict with accuracy, nmi, ari, macro_f1, per_class_f1.
    """
    import random
    import torchvision
    import torchvision.transforms as T

    random.seed(seed)
    np.random.seed(seed)

    if dataset_name == "mnist":
        ds = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True,
            transform=T.ToTensor(),
        )
    elif dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=T.ToTensor(),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n = len(ds)
    if max_samples > 0 and max_samples < n:
        indices = sorted(random.sample(range(n), max_samples))
    else:
        indices = list(range(n))

    # Load and flatten
    features_list = []
    true_labels = []
    for i in pbar(indices, desc=f"Loading {dataset_name} test set",
                  unit="img", disable=not verbose):
        img, label = ds[i]
        features_list.append(img.reshape(-1).numpy())
        true_labels.append(label)

    features = np.array(features_list)
    true_labels = np.array(true_labels)

    # Embed in batches
    all_embeddings = []
    for start in pbar(range(0, len(features), batch_size),
                      desc="Embedding", unit="batch", disable=not verbose):
        batch = features[start:start + batch_size]
        result = embedder.embed(batch)
        all_embeddings.append(result.mean)

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Cluster
    if verbose:
        print(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)

    # Align labels
    label_map = align_labels_via_hungarian_algorithm(true_labels, pred_labels)
    aligned_preds = np.array([label_map.get(p, p) for p in pred_labels])

    # Metrics
    acc = accuracy_score(true_labels, aligned_preds)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, aligned_preds, average="macro")
    per_class_f1 = f1_score(true_labels, aligned_preds, average=None)

    results = {
        "accuracy": acc,
        "nmi": nmi,
        "ari": ari,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
    }

    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print(f"NMI:      {nmi:.4f}")
        print(f"ARI:      {ari:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Per-class F1: {[f'{f:.3f}' for f in per_class_f1]}")

    return results
