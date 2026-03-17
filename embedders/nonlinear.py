import os
import random
import numpy as np
import torch
from features.base import BaseFeatureExtractor
from features.kmer import KmerFeatureExtractor
from embedders.base import BaseEmbedder, EmbeddingResult


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


class NonLinearEmbedder(BaseEmbedder, torch.nn.Module):
    """Feed-forward neural network embedding with pluggable feature extraction.

    Architecture: [feature_extractor] -> Linear(input_dim, 512) -> BatchNorm
                  -> Sigmoid -> Dropout(0.2) -> Linear(512, dim)

    The feature extractor can be:
    - KmerFeatureExtractor (DNA k-mer profiles) — pass k=...
    - Any BaseFeatureExtractor (numpy-based) — pass feature_extractor=...
    - Any nn.Module-based extractor (trained end-to-end) — pass feature_extractor=...
    - None (raw numeric input) — pass input_dim=...
    """

    def __init__(self, k: int = None, dim: int = 256, device: str = "cpu",
                 verbose: bool = False, seed: int = 0,
                 input_dim: int = None,
                 feature_extractor: BaseFeatureExtractor = None):
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")

        sources = sum(x is not None for x in (k, input_dim, feature_extractor))
        if sources != 1:
            raise ValueError(
                f"Exactly one of k, input_dim, or feature_extractor must be provided "
                f"(got {sources}: k={k!r}, input_dim={input_dim!r}, "
                f"feature_extractor={feature_extractor!r})"
            )

        if k is not None and (not isinstance(k, int) or k <= 0):
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if input_dim is not None and (not isinstance(input_dim, int) or input_dim <= 0):
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}")

        self._device = torch.device(device) if isinstance(device, str) else device
        self._verbose = verbose
        self._k = k
        self._dim = dim
        self._seed = seed
        self._input_dim = input_dim

        # Input dimension is resolved from exactly one of three sources:
        if feature_extractor is not None:
            self._feature_extractor = feature_extractor
            if isinstance(feature_extractor, torch.nn.Module):
                self.feature_net = feature_extractor
            input_dim = feature_extractor.feature_dim
            self._input_dim = input_dim
        elif k is not None:
            self._feature_extractor = KmerFeatureExtractor(k=k, normalized=True)
            input_dim = self._feature_extractor.feature_dim
            self._input_dim = input_dim
        elif input_dim is not None:
            self._feature_extractor = None
        else:
            raise ValueError("Provide 'k', 'input_dim', or 'feature_extractor'")

        set_seed(seed)

        self.linear1 = torch.nn.Linear(input_dim, 512, dtype=torch.float, device=self._device)
        self.batch1 = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self._device)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(512, self._dim, dtype=torch.float, device=self._device)

    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder network.

        Parameters
        ----------
        features : torch.Tensor
            Input features of shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded embeddings of shape ``(N, dim)``.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        output = self.linear1(features)
        output = self.batch1(output)
        output = self.activation1(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        return output

    def forward(self, left_features: torch.Tensor,
                right_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode left and right features for contrastive training.

        Parameters
        ----------
        left_features : torch.Tensor
            Left input features of shape ``(N, input_dim)``.
        right_features : torch.Tensor
            Right input features of shape ``(N, input_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Encoded (left_embeddings, right_embeddings), each of shape ``(N, dim)``.
        """
        if hasattr(self, 'feature_net'):
            left_features = self.feature_net(left_features)
            right_features = self.feature_net(right_features)
        if left_features.ndim != 2:
            raise ValueError(f"left_features must be 2D, got shape {left_features.shape}")
        if right_features.ndim != 2:
            raise ValueError(f"right_features must be 2D, got shape {right_features.shape}")
        if left_features.shape[-1] != right_features.shape[-1]:
            raise ValueError(
                f"Feature dimension mismatch: left {left_features.shape[-1]} "
                f"vs right {right_features.shape[-1]}"
            )
        return self.encoder(left_features), self.encoder(right_features)

    def embed(self, inputs) -> EmbeddingResult:
        """Embed inputs through the feature extractor + encoder pipeline.

        Parameters
        ----------
        inputs : object
            Raw inputs for the feature extractor. Type depends on
            the extractor: list[str] for DNA, np.ndarray/Tensor for
            images, etc. If no feature extractor, must be numeric.
        """
        if len(inputs) == 0:
            raise ValueError("inputs must not be empty")
        self.eval()
        with torch.inference_mode():
            if self._feature_extractor is not None:
                if isinstance(self._feature_extractor, torch.nn.Module):
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.as_tensor(
                            np.asarray(inputs), dtype=torch.float
                        )
                    features_t = self.feature_net(inputs.to(self._device))
                else:
                    profiles = self._feature_extractor.extract_batch(inputs)
                    features_t = torch.from_numpy(profiles).to(
                        torch.float
                    ).to(self._device)
            else:
                features_t = torch.as_tensor(
                    np.asarray(inputs), dtype=torch.float
                ).to(self._device)
            embs = self.encoder(features_t).cpu().numpy()
        return EmbeddingResult(mean=embs)

    def get_k(self) -> int:
        """Return the k-mer size, or None if not using k-mers."""
        return self._k

    def get_dim(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    def get_device(self):
        """Return the device this model is on."""
        return self._device

    def save(self, path: str) -> None:
        """Save model parameters and constructor kwargs to disk."""
        if self._verbose:
            print(f"Saving model to: {path}")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if isinstance(self, torch.nn.DataParallel):
            module = self.module
        else:
            module = self

        kwargs = {
            "k": module._k,
            "dim": module._dim,
            "device": str(module._device),
            "input_dim": module._input_dim,
        }
        torch.save([kwargs, module.state_dict()], path)

    @classmethod
    def load(cls, path: str, device: str = "cpu",
             feature_extractor=None) -> "NonLinearEmbedder":
        """Load a saved NonLinearEmbedder from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        device : str, optional
            Device to load onto. Default is ``"cpu"``.
        feature_extractor : BaseFeatureExtractor or nn.Module, optional
            Feature extractor with the same architecture used during
            training. Required when the model was trained with an
            nn.Module feature extractor (e.g. CNNFeatureExtractor),
            since the architecture is not serialized. Weights are
            restored from the state dict.

        Returns
        -------
        NonLinearEmbedder
            The loaded model with restored weights.
        """
        kwargs, state_dict = torch.load(path, map_location=torch.device(device))
        kwargs["device"] = device
        # Drop legacy keys from old checkpoints
        kwargs.pop("activation", None)
        kwargs.pop("use_batchnorm", None)
        if feature_extractor is not None:
            kwargs.pop("k", None)
            kwargs.pop("input_dim", None)
            kwargs["feature_extractor"] = feature_extractor
        elif kwargs.get("k") is not None:
            kwargs.pop("input_dim", None)
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    @property
    def default_metric(self) -> str:
        return "l2"


def _bce_from_log_prob(log_p: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy computed directly from log-probability.

    Given log(p) where p ∈ (0, 1] is the predicted probability of similarity,
    computes the BCE loss:  L = -[ y·log(p) + (1-y)·log(1-p) ]

    Why not use PyTorch's built-in BCE functions?
    - F.binary_cross_entropy expects p, not log(p). Going exp→clamp→log
      loses precision for very small p (e.g. p = exp(-1000) rounds to 0).
    - F.binary_cross_entropy_with_logits expects logits (where p = sigmoid(x)),
      not log-probabilities (where p = exp(log_p)). Different parameterization.

    So we compute BCE directly from log(p):
    - The positive term y·log(p) is just y·log_p (we already have log_p).
    - The negative term (1-y)·log(1-p) requires log(1-p), computed as
      log(1 - exp(log_p)) via the numerically stable log1p(-exp(·)) identity:
        log_p very negative  → exp(log_p) ≈ 0 → log1p(0) = 0       (trivial)
        log_p near 0         → exp(log_p) ≈ 1 → log1p(-1) = -inf   (boundary)

    The naive formula  -(labels * log_p + (1-labels) * log_1mp)  suffers from
    0 * (-inf) = NaN in IEEE 754 arithmetic. This happens when:
      - log_p = -inf and labels = 0:  labels * log_p = 0 * (-inf) = NaN
      - log_p ≈ 0 and labels = 1:    (1-labels) * log_1mp = 0 * (-inf) = NaN
    We avoid this by using torch.where to select the relevant term per sample,
    so the unused branch is never multiplied by zero.

    Parameters
    ----------
    log_p : torch.Tensor
        (N,) log-probabilities, always ≤ 0 by construction.
    labels : torch.Tensor
        (N,) binary labels — 1 for positive, 0 for negative.

    Returns
    -------
    torch.Tensor
        Scalar mean BCE loss.
    """
    # log(1 - p) = log(1 - exp(log_p)), computed via log1p for stability
    log_1mp = torch.log1p(-torch.exp(log_p))

    # Select the active term per sample to avoid 0 * (-inf) = NaN:
    #   label=1 (positive pair): loss = -log(p)    = -log_p
    #   label=0 (negative pair): loss = -log(1-p)  = -log_1mp
    bce = torch.where(labels == 1, -log_p, -log_1mp)
    return torch.mean(bce)


def contrastive_loss(left_embeddings: torch.Tensor, right_embeddings: torch.Tensor,
                     labels: torch.Tensor, name: str = "bern",
                     neg_threshold: float = None,
                     scale: float = 1.0) -> torch.Tensor:
    """Contrastive loss functions for paired embeddings.

    Parameters
    ----------
    left_embeddings : torch.Tensor
        (N, D) left sequence embeddings.
    right_embeddings : torch.Tensor
        (N, D) right sequence embeddings.
    labels : torch.Tensor
        (N,) binary labels — 1 for positive pair, 0 for negative.
    name : str
        Loss function name — "bern" (Bernoulli), "poisson", or "hinge".
    neg_threshold : float, optional
        When set (bern only), negative pairs with ``log_p <= neg_threshold``
        get zero gradient via ``torch.nn.functional.threshold``, and a
        ``+1e-6`` floor inside ``log()`` prevents infinity. Typical value:
        ``-1.0``. When ``None`` (default), uses standard log-space BCE.
    scale : float
        Coefficient applied to the squared distance: ``log_p = -scale * d²``.
        Default 1.0 matches the standard formulation. Use 0.25 to reproduce
        the original paper's Phase 1 mahalanobis loss (``-d²/4``), which
        corresponds to kernel K=2I in the Lemma 3.1 framework.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if left_embeddings.ndim != 2:
        raise ValueError(f"left_embeddings must be 2D, got shape {left_embeddings.shape}")
    if right_embeddings.ndim != 2:
        raise ValueError(f"right_embeddings must be 2D, got shape {right_embeddings.shape}")
    if left_embeddings.shape != right_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: left {left_embeddings.shape} vs right {right_embeddings.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if labels.shape[0] != left_embeddings.shape[0]:
        raise ValueError(
            f"Batch size mismatch: labels has {labels.shape[0]} elements "
            f"but embeddings have {left_embeddings.shape[0]} rows"
        )
    valid_names = ("bern", "poisson", "hinge")
    if name not in valid_names:
        raise ValueError(f"name must be one of {valid_names}, got {name!r}")

    sq_dist = torch.norm(left_embeddings - right_embeddings, p=2, dim=1) ** 2

    if name == "bern":
        log_p = -scale * sq_dist
        if neg_threshold is not None:
            thresholded = torch.nn.functional.threshold(log_p, neg_threshold, 0.0)
            log_1mp = torch.log(1.0 - torch.exp(thresholded) + 1e-6)
            bce = torch.where(labels == 1, -log_p, -log_1mp)
            return torch.mean(bce)
        return _bce_from_log_prob(log_p, labels)

    if name == "poisson":
        log_lambda = -scale * sq_dist
        return torch.mean(-(labels * log_lambda) + torch.exp(log_lambda))

    # name == "hinge" (guaranteed by validation above)
    d = torch.sqrt(sq_dist + 1e-8)
    return torch.mean(
        labels * scale * (d ** 2) + (1 - labels) * torch.nn.functional.relu(1 - d) ** 2
    )
