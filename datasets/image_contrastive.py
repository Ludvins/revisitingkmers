"""Contrastive dataset from labeled image datasets (MNIST, CIFAR-10, FashionMNIST).

Three pairing modes (controlled by augmentation_mode and strict_negatives):
- Label-based (default): positive pairs from same class, negatives from full pool.
- Augmentation-based: positive pairs via data augmentation, negatives randomly
  sampled from the full pool regardless of class.  Two augmentation types:
    - "affine": rotation + translation (easy for CNNs).
    - "simclr": RandomResizedCrop + GaussianBlur + RandomErasing + ColorJitter
      (harder for CNNs — forces semantic invariance learning).
- Oracle (label-based + strict_negatives): positives from same class, negatives
  guaranteed from different classes.

Returns the same (left, right, labels) contract as PairedReadsDataset
for use with train_contrastive().
"""
import random
import numpy as np
import torch
from typing import Tuple
from datasets.base import BaseContrastiveDataset
from utils.progress import pbar


class ImageContrastiveDataset(BaseContrastiveDataset):
    """Contrastive dataset built from a labeled image dataset.

    Downloads MNIST or CIFAR-10 via torchvision, flattens images to 1-D
    feature vectors, and constructs contrastive pairs:
        - Positive pair: anchor + random same-class image
        - Negative samples: random images from the full pool

    Parameters
    ----------
    dataset_name : str
        "mnist" or "cifar10".
    split : str
        "train" or "test".
    neg_sample_per_pos : int
        Number of negative samples per positive pair.
    max_samples : int
        Max samples to use (0 = all).
    data_root : str
        Root directory for torchvision data download.
    verbose : bool
        Print progress.
    seed : int
        Random seed.
    """

    def __init__(self, dataset_name: str = "mnist", split: str = "train",
                 neg_sample_per_pos: int = 200, max_samples: int = 0,
                 data_root: str = "./data", verbose: bool = True,
                 seed: int = 0, class_subset: list = None,
                 augmentation_mode: bool = False,
                 strict_negatives: bool = False,
                 rotation_range: float = 15.0,
                 translation_range: tuple = (0.1, 0.1),
                 augmentation_type: str = "affine"):
        _SUPPORTED_DATASETS = ("mnist", "cifar10", "cifar100", "fashionmnist")
        if dataset_name not in _SUPPORTED_DATASETS:
            raise ValueError(f"dataset_name must be one of {_SUPPORTED_DATASETS}, got '{dataset_name}'")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        if neg_sample_per_pos <= 0:
            raise ValueError(f"neg_sample_per_pos must be positive, got {neg_sample_per_pos}")

        import torchvision
        import torchvision.transforms as T

        self._neg_sample_per_pos = neg_sample_per_pos
        self._seed = seed
        self._dataset_name = dataset_name
        self._augmentation_mode = augmentation_mode
        self._strict_negatives = strict_negatives

        random.seed(seed)
        torch.manual_seed(seed)

        is_train = (split == "train")

        if dataset_name == "mnist":
            ds = torchvision.datasets.MNIST(
                root=data_root, train=is_train, download=True,
                transform=T.ToTensor(),
            )
        elif dataset_name == "fashionmnist":
            ds = torchvision.datasets.FashionMNIST(
                root=data_root, train=is_train, download=True,
                transform=T.ToTensor(),
            )
        elif dataset_name == "cifar10":
            ds = torchvision.datasets.CIFAR10(
                root=data_root, train=is_train, download=True,
                transform=T.ToTensor(),
            )
        elif dataset_name == "cifar100":
            ds = torchvision.datasets.CIFAR100(
                root=data_root, train=is_train, download=True,
                transform=T.ToTensor(),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}.")

        # Filter to a subset of classes (for unknown-k experiments)
        if class_subset is not None:
            class_set = set(class_subset)
            full_indices = [i for i in range(len(ds)) if ds.targets[i] in class_set]
        else:
            full_indices = list(range(len(ds)))

        # Subsample if requested
        n = len(full_indices)
        if max_samples > 0 and max_samples < n:
            sampled = sorted(random.sample(full_indices, max_samples))
        else:
            sampled = full_indices

        # Load and flatten images
        features_list = []
        labels_list = []
        for i in pbar(sampled, desc=f"Loading {dataset_name} ({split})",
                      unit="img", disable=not verbose):
            img, label = ds[i]
            features_list.append(img.reshape(-1))  # flatten (C,H,W) -> (C*H*W,)
            labels_list.append(label)

        self._features = torch.stack(features_list)  # (N, feat_dim)
        self._labels = torch.tensor(labels_list, dtype=torch.long)

        # Build per-class index for positive pair sampling
        self._class_indices = {}
        for idx, label in enumerate(labels_list):
            self._class_indices.setdefault(label, []).append(idx)

        self._ones = torch.ones(len(self._features))

        # Precompute per-class "other" indices for strict negative sampling
        if strict_negatives:
            all_idx_set = set(range(len(self._features)))
            self._other_class_indices = {}
            for cls, indices in self._class_indices.items():
                other = list(all_idx_set - set(indices))
                self._other_class_indices[cls] = torch.tensor(other, dtype=torch.long)

        # Store image shape for augmentation (infer from first sample)
        sample_img, _ = ds[sampled[0]]
        self._image_shape = tuple(sample_img.shape)  # (C, H, W)

        self._augmentation_type = augmentation_type
        if augmentation_mode:
            if augmentation_type == "simclr":
                H = self._image_shape[1]
                self._augment = T.Compose([
                    T.RandomResizedCrop(H, scale=(0.4, 1.0)),
                    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
                    T.ColorJitter(brightness=0.4, contrast=0.4),
                    T.RandomErasing(p=0.5, scale=(0.05, 0.2)),
                ])
            else:
                self._augment = T.RandomAffine(
                    degrees=rotation_range, translate=translation_range,
                )

        if verbose:
            pos_str = f"augmentation ({augmentation_type})" if augmentation_mode else "oracle (label)"
            neg_str = "strict (different-class)" if strict_negatives else "random (non-strict)"
            mode_str = f"pos={pos_str}, neg={neg_str}"
            print(f"Dataset loaded: {len(self._features)} samples, "
                  f"feature_dim={self._features.shape[1]}, "
                  f"classes={len(self._class_indices)}, "
                  f"pairing={mode_str}")

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the flattened image features."""
        return self._features.shape[1]

    @property
    def labels(self) -> torch.Tensor:
        """Integer class labels for all samples."""
        return self._labels

    @property
    def num_classes(self) -> int:
        """Number of distinct classes in the dataset."""
        return len(self._class_indices)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a contrastive sample: one positive pair plus negative samples.

        In label-based mode (default): positive = random same-class image.
        In augmentation mode: positive = augmented version of anchor.
        Negatives are always random samples from the full pool.

        Parameters
        ----------
        idx : int
            Index of the anchor image.

        Returns
        -------
        left_features : torch.Tensor
            ``(1 + neg_sample_per_pos, feature_dim)`` — anchor + negative lefts.
        right_features : torch.Tensor
            ``(1 + neg_sample_per_pos, feature_dim)`` — positive + negative rights.
        labels : torch.Tensor
            ``(1 + neg_sample_per_pos,)`` — 1 for positive pair, 0 for negatives.
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")

        anchor = self._features[idx]

        if self._augmentation_mode:
            # Positive: augmented version of the anchor
            C, H, W = self._image_shape
            anchor_img = anchor.reshape(C, H, W)
            positive = self._augment(anchor_img).reshape(-1)
        else:
            # Positive: random same-class image (distinct from anchor when possible)
            anchor_label = self._labels[idx].item()
            same_class = self._class_indices[anchor_label]
            if len(same_class) > 1:
                candidates = [i for i in same_class if i != idx]
                pos_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                pos_idx = same_class[0]
            positive = self._features[pos_idx]

        # Negative samples
        if self._strict_negatives:
            # Guaranteed different-class negatives
            anchor_label_val = self._labels[idx].item()
            other_pool = self._other_class_indices[anchor_label_val]
            rand_idx = torch.randint(len(other_pool), (2 * self._neg_sample_per_pos,))
            neg_indices = other_pool[rand_idx]
        else:
            # Random from entire pool (may include same-class)
            neg_indices = torch.multinomial(
                self._ones, replacement=True,
                num_samples=2 * self._neg_sample_per_pos,
            )

        left_features = torch.cat([
            anchor.unsqueeze(0),
            self._features[neg_indices[:self._neg_sample_per_pos]]
        ])
        right_features = torch.cat([
            positive.unsqueeze(0),
            self._features[neg_indices[self._neg_sample_per_pos:]]
        ])
        labels = torch.tensor(
            [1] + [0] * self._neg_sample_per_pos, dtype=torch.float
        )

        return left_features, right_features, labels
