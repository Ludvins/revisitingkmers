"""UncertainGen: Uncertainty-Aware Representations for Metagenomic Binning.

Implements the probabilistic embedding framework from:
    Celikkanat et al., "UncertainGen: Uncertainty-Aware Representations of DNA
    Sequences for Metagenomic Binning" (arXiv:2509.26116)

Each DNA fragment is mapped to a Gaussian distribution N(mu, diag(sigma^2))
in latent space. The variance captures sequence-level uncertainty from
inter-species DNA sharing and ambiguous k-mer profiles.

Training uses two phases:
    Phase 1: Train mean network with deterministic contrastive loss (bern).
    Phase 2: Freeze mean network, train variance network with the marginal
             probabilistic loss from Lemma 3.1.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from features.base import BaseFeatureExtractor
from features.kmer import KmerFeatureExtractor
from embedders.base import BaseEmbedder, EmbeddingResult
from utils.progress import pbar

_VALID_K_FORMS = {"adaptive", "identity", "expected_distance", "mc"}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


class UncertainGenEmbedder(BaseEmbedder, nn.Module):
    """Probabilistic embedding model with pluggable feature extraction.

    Architecture:
        [feature_extractor] ->
        Mean network:     Linear(input_dim, 512) -> BatchNorm -> Sigmoid -> Dropout -> Linear(512, dim)
        Variance network: Linear(input_dim, 512) -> BatchNorm -> Sigmoid -> Dropout -> Linear(512, dim) -> Softplus

    The feature extractor can be:
    - KmerFeatureExtractor (DNA k-mer profiles) — pass k=...
    - Any BaseFeatureExtractor (numpy-based) — pass feature_extractor=...
    - Any nn.Module-based extractor (trained end-to-end) — pass feature_extractor=...
    - None (raw numeric input) — pass input_dim=...

    The mean network produces point embeddings (mu). The variance network
    produces per-dimension uncertainty (sigma^2). Together they define a
    diagonal Gaussian embedding N(mu, diag(sigma^2)) for each input.

    When ``scalar_variance=True``, the variance network outputs a single
    scalar per sample (isotropic variance), expanded to all dimensions:
    N(mu, sigma^2 * I). This is analogous to PCL's single kappa per sample.
    """

    def __init__(self, k: int = None, dim: int = 256, alpha: float = 1.0,
                 k_form: str = None, device: str = "cpu",
                 verbose: bool = False, seed: int = 0,
                 input_dim: int = None,
                 feature_extractor: BaseFeatureExtractor = None,
                 scalar_variance: bool = False):
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")

        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha!r}")

        if k_form is None:
            raise ValueError(
                "k_form is required -- specify 'adaptive' for K=alpha*(Si+Sj) "
                "or 'identity' for K=I"
            )
        if k_form not in _VALID_K_FORMS:
            raise ValueError(
                f"k_form must be one of {_VALID_K_FORMS}, got {k_form!r}"
            )

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
        self._alpha = alpha
        self._k_form = k_form
        self._seed = seed
        self._input_dim = input_dim
        self._scalar_variance = scalar_variance

        # Determine input dimension from one of three sources
        if feature_extractor is not None:
            self._feature_extractor = feature_extractor
            if isinstance(feature_extractor, nn.Module):
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

        # Mean network (same architecture as NonLinearEmbedder)
        self.mean_linear1 = nn.Linear(input_dim, 512, dtype=torch.float, device=self._device)
        self.mean_bn1 = nn.BatchNorm1d(512, dtype=torch.float, device=self._device)
        self.mean_act1 = nn.Sigmoid()
        self.mean_dropout1 = nn.Dropout(0.2)
        self.mean_linear2 = nn.Linear(512, self._dim, dtype=torch.float, device=self._device)

        # Variance network (same hidden layers + Softplus output)
        var_out_dim = 1 if scalar_variance else self._dim
        self.var_linear1 = nn.Linear(input_dim, 512, dtype=torch.float, device=self._device)
        self.var_bn1 = nn.BatchNorm1d(512, dtype=torch.float, device=self._device)
        self.var_act1 = nn.Sigmoid()
        self.var_dropout1 = nn.Dropout(0.2)
        self.var_linear2 = nn.Linear(512, var_out_dim, dtype=torch.float, device=self._device)
        self.var_softplus = nn.Softplus()
        # Initialize var_linear2 bias to -5 so Softplus starts near zero (~0.007).
        # Without this, default bias → Softplus(~0) ≈ 0.69 per dim, which makes the
        # normalization term in the probabilistic loss crush all probabilities to ~0,
        # causing the Phase 2 loss to plateau instead of training.
        nn.init.constant_(self.var_linear2.bias, -5.0)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the mean network. Returns mu."""
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        x = self.mean_linear1(x)
        x = self.mean_bn1(x)
        x = self.mean_act1(x)
        x = self.mean_dropout1(x)
        x = self.mean_linear2(x)
        return x

    def encode_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the variance network. Returns sigma^2 > 0.

        When ``scalar_variance=True``, the network outputs a single scalar
        per sample which is expanded to (N, D) as an isotropic variance.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        x = self.var_linear1(x)
        x = self.var_bn1(x)
        x = self.var_act1(x)
        x = self.var_dropout1(x)
        x = self.var_linear2(x)
        x = self.var_softplus(x)
        if self._scalar_variance:
            x = x.expand(-1, self._dim)
        return x

    def copy_mean_from(self, source) -> None:
        """Copy pre-trained mean network weights from a NonLinearEmbedder.

        This allows skipping Phase 1 training by reusing an already-trained
        deterministic model's weights for the mean network.  The feature
        extractor (e.g. CNN) weights are also copied when both models use
        nn.Module-based extractors.

        Parameters
        ----------
        source : NonLinearEmbedder
            Pre-trained model whose encoder weights will be copied into
            this model's mean network.  Must have matching layer dimensions.
        """
        from embedders.nonlinear import NonLinearEmbedder
        if not isinstance(source, NonLinearEmbedder):
            raise TypeError(
                f"source must be a NonLinearEmbedder, got {type(source).__name__}"
            )

        # Copy feature extractor (e.g. CNN) weights
        if hasattr(self, 'feature_net') and hasattr(source, 'feature_net'):
            self.feature_net.load_state_dict(source.feature_net.state_dict())

        # Copy mean network layers
        self.mean_linear1.load_state_dict(source.linear1.state_dict())
        self.mean_bn1.load_state_dict(source.batch1.state_dict())
        self.mean_linear2.load_state_dict(source.linear2.state_dict())

    def forward(self, left_features: torch.Tensor,
                right_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning means only.

        Compatible with train_contrastive() for Phase 1 training.
        """
        if left_features.ndim != 2:
            raise ValueError(f"left_features must be 2D, got shape {left_features.shape}")
        if right_features.ndim != 2:
            raise ValueError(f"right_features must be 2D, got shape {right_features.shape}")
        if left_features.shape[-1] != right_features.shape[-1]:
            raise ValueError(
                f"Feature dimension mismatch: left {left_features.shape[-1]} "
                f"vs right {right_features.shape[-1]}"
            )
        if hasattr(self, 'feature_net'):
            left_features = self.feature_net(left_features)
            right_features = self.feature_net(right_features)
        return self.encode_mean(left_features), self.encode_mean(right_features)

    def forward_full(self, left_features: torch.Tensor,
                     right_features: torch.Tensor) -> tuple[
                         torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning means and variances.

        Used for Phase 2 training with the marginal probabilistic loss.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (left_mean, right_mean, left_var, right_var)
        """
        if left_features.ndim != 2:
            raise ValueError(f"left_features must be 2D, got shape {left_features.shape}")
        if right_features.ndim != 2:
            raise ValueError(f"right_features must be 2D, got shape {right_features.shape}")
        if left_features.shape[-1] != right_features.shape[-1]:
            raise ValueError(
                f"Feature dimension mismatch: left {left_features.shape[-1]} "
                f"vs right {right_features.shape[-1]}"
            )
        if hasattr(self, 'feature_net'):
            left_features = self.feature_net(left_features)
            right_features = self.feature_net(right_features)
        return (self.encode_mean(left_features), self.encode_mean(right_features),
                self.encode_variance(left_features), self.encode_variance(right_features))

    def embed(self, inputs) -> EmbeddingResult:
        """Embed inputs as Gaussian distributions.

        Parameters
        ----------
        inputs : object
            Raw inputs for the feature extractor. Type depends on
            the extractor: list[str] for DNA, np.ndarray/Tensor for
            images, etc. If no feature extractor, must be numeric.

        Returns
        -------
        EmbeddingResult
            EmbeddingResult with both mean and variance populated.
        """
        if len(inputs) == 0:
            raise ValueError("inputs must not be empty")
        self.eval()
        with torch.inference_mode():
            if self._feature_extractor is not None:
                if isinstance(self._feature_extractor, nn.Module):
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
            mean = self.encode_mean(features_t).cpu().numpy()
            variance = self.encode_variance(features_t).cpu().numpy()
        return EmbeddingResult(mean=mean, variance=variance)

    def selfsim_uncertainty(self, inputs) -> np.ndarray:
        """Compute 1 - P(1|x,x) using the Lemma 3.1 closed form.

        For each input, evaluates the expected Bernoulli self-similarity
        under the learned Gaussian embedding. When the embedding is
        certain (variance -> 0), returns 0; when highly uncertain, approaches 1.

        Uses K = alpha*I with the model's alpha:
            P(1|x,x) = prod_d (2*sigma^2_d / alpha + 1)^{-1/2}
            uncertainty = 1 - P(1|x,x)

        Returns
        -------
        np.ndarray
            (N,) array of per-sample uncertainty scores in [0, 1].
        """
        result = self.embed(inputs)
        # v_d = 2 * sigma^2_d (self-comparison: both sides same distribution)
        # log P(1|x,x) = sum_d -0.5 * log(2*sigma^2_d / alpha + 1)
        log_p = -0.5 * np.sum(
            np.log(2.0 * result.variance / self._alpha + 1.0), axis=1
        )
        return 1.0 - np.exp(log_p)

    def mean_parameters(self):
        """Iterator over mean network + feature extractor parameters (for Phase 1 optimizer).

        Includes feature extractor parameters (when it's an nn.Module) since
        it is shared and should be trained jointly with the mean network in Phase 1.
        """
        for name, param in self.named_parameters():
            if name.startswith("mean_") or name.startswith(("feature_net.", "_feature_extractor.")):
                yield param

    def variance_parameters(self):
        """Iterator over variance network parameters (for Phase 2 optimizer)."""
        for name, param in self.named_parameters():
            if name.startswith("var_"):
                yield param

    def freeze_mean_network(self):
        """Freeze mean network and feature_net parameters for Phase 2 training.

        Also sets BatchNorm and Dropout layers to eval mode so running
        statistics are not updated and dropout is disabled during Phase 2.
        """
        for param in self.mean_parameters():
            param.requires_grad = False
        self.mean_bn1.eval()
        self.mean_dropout1.eval()
        if hasattr(self, 'feature_net'):
            self.feature_net.eval()

    def unfreeze_mean_network(self):
        """Unfreeze mean network and feature_net parameters."""
        for param in self.mean_parameters():
            param.requires_grad = True
        self.mean_bn1.train()
        self.mean_dropout1.train()
        if hasattr(self, 'feature_net'):
            self.feature_net.train()

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

        module = self.module if isinstance(self, nn.DataParallel) else self
        kwargs = {
            "k": module._k,
            "dim": module._dim,
            "alpha": module._alpha,
            "k_form": module._k_form,
            "device": str(module._device),
            "input_dim": module._input_dim,
            "scalar_variance": module._scalar_variance,
        }
        torch.save([kwargs, module.state_dict()], path)

    @classmethod
    def load(cls, path: str, device: str = "cpu",
             feature_extractor=None) -> "UncertainGenEmbedder":
        """Load a saved UncertainGenEmbedder from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        device : str, optional
            Device to load onto. Default is ``"cpu"``.
        feature_extractor : BaseFeatureExtractor or nn.Module, optional
            Feature extractor with the same architecture used during
            training. Required when the model was trained with an
            nn.Module feature extractor, since the architecture is
            not serialized. Weights are restored from the state dict.

        Returns
        -------
        UncertainGenEmbedder
            The loaded model with restored weights.
        """
        kwargs, state_dict = torch.load(path, map_location=torch.device(device))
        kwargs["device"] = device
        # Backward compat: old checkpoints without k_form default to adaptive
        if "k_form" not in kwargs:
            kwargs["k_form"] = "adaptive"
        # Backward compat: old checkpoints without scalar_variance default to False
        if "scalar_variance" not in kwargs:
            kwargs["scalar_variance"] = False
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


def uncertaingen_loss(left_mean: torch.Tensor, right_mean: torch.Tensor,
                      left_var: torch.Tensor, right_var: torch.Tensor,
                      labels: torch.Tensor, alpha: float = 0.1,
                      k_form: str = "adaptive",
                      neg_threshold: float = None) -> torch.Tensor:
    """Marginal probabilistic contrastive loss from Lemma 3.1.

    Computes the closed-form expectation (Lemma 3.1):

        p_ij = |K⁻¹(Si+Sj) + I|^{-1/2}
               · exp(-1/2 · (μi-μj)^T (Si+Sj+K)⁻¹ (μi-μj))

    where z_i ~ N(μ_i, diag(σ²_i)), z_j ~ N(μ_j, diag(σ²_j)), and
    K is a positive definite matrix controlling the similarity kernel.

    Two K forms are supported:

    **k_form="adaptive"** — K = α(Si+Sj):
        Simplifies to scalar normalization + inverse-variance Mahalanobis:
        log(p) = D/2·log(α/(α+1)) - 1/(2(1+α))·Σ_d Δ²_d/(σ²_i,d+σ²_j,d)

    **k_form="identity"** — K = αI:
        Per-dimension normalization with α controlling the metric tightness:
        log(p) = Σ_d [-1/2·log(v_d/α + 1) - 1/2·Δ²_d/(v_d + α)]
        where v_d = σ²_i,d + σ²_j,d.  α=1 recovers the original K=I form;
        smaller α = tighter metric, stronger normalization penalty.

    When variance = 0 and δ = 0, the identity form gives p = 1 for any α.
    The expected_distance form reduces to exp(-1/2·||δ||²) at var = 0.
    The adaptive form is undefined at var = 0 (division by zero);
    in practice, an eps floor prevents this.

    Parameters
    ----------
    left_mean : torch.Tensor
        (N, D) mean embeddings for left sequences.
    right_mean : torch.Tensor
        (N, D) mean embeddings for right sequences.
    left_var : torch.Tensor
        (N, D) variance (sigma^2) for left sequences.
    right_var : torch.Tensor
        (N, D) variance (sigma^2) for right sequences.
    labels : torch.Tensor
        (N,) binary labels — 1 for positive, 0 for negative.
    alpha : float
        For adaptive form: scaling factor in K = α(Σi+Σj).
        For identity form: diagonal of K = αI (metric tightness).
        For expected_distance: scale in exp(-α·d²).
    k_form : str
        Which K matrix form to use: ``"adaptive"``, ``"identity"``,
        or ``"expected_distance"``.
    neg_threshold : float, optional
        When set, uses the paper's original BCE formulation: negative pairs
        with ``log_p <= neg_threshold`` get zero gradient via
        ``torch.nn.functional.threshold``, and a ``+1e-6`` floor inside
        ``log()`` prevents infinity.  Typical value: ``-1.0``.
        When ``None`` (default), uses log-space BCE with a clamp.

    Returns
    -------
    torch.Tensor
        Scalar BCE loss.
    """
    if left_mean.ndim != 2:
        raise ValueError(f"left_mean must be 2D, got shape {left_mean.shape}")
    if right_mean.ndim != 2:
        raise ValueError(f"right_mean must be 2D, got shape {right_mean.shape}")
    if left_var.ndim != 2:
        raise ValueError(f"left_var must be 2D, got shape {left_var.shape}")
    if right_var.ndim != 2:
        raise ValueError(f"right_var must be 2D, got shape {right_var.shape}")
    if left_mean.shape != right_mean.shape:
        raise ValueError(
            f"Shape mismatch: left_mean {left_mean.shape} vs right_mean {right_mean.shape}"
        )
    if left_var.shape != right_var.shape:
        raise ValueError(
            f"Shape mismatch: left_var {left_var.shape} vs right_var {right_var.shape}"
        )
    if left_mean.shape != left_var.shape:
        raise ValueError(
            f"Shape mismatch: mean {left_mean.shape} vs var {left_var.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if labels.shape[0] != left_mean.shape[0]:
        raise ValueError(
            f"Batch size mismatch: labels has {labels.shape[0]} elements "
            f"but embeddings have {left_mean.shape[0]} rows"
        )
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if k_form not in _VALID_K_FORMS:
        raise ValueError(
            f"k_form must be one of {_VALID_K_FORMS}, got {k_form!r}"
        )

    eps = 1e-4  # large enough to survive float16 (AMP); 1e-8 rounds to 0
    delta_sq = (left_mean - right_mean) ** 2  # (N, D)
    sum_var = left_var + right_var  # (N, D)

    if k_form == "adaptive":
        # K = α(Si+Sj)
        # (Si+Sj+K)⁻¹ = ((1+α)(Si+Sj))⁻¹
        #
        # The normalization constant D/2·log(α/(α+1)) from the determinant
        # |K⁻¹(Si+Sj)+I| is independent of parameters and variance, so it
        # carries no gradient signal and is omitted.  This means log_p can
        # reach 0 when delta_sq ≈ 0 (negative pairs whose frozen means
        # happen to coincide), causing log(1−exp(0)) = −∞ in the BCE.
        # We clamp log_p < 0 to cap the loss for those unfixable pairs.
        log_p = -0.5 / (1.0 + alpha) * torch.sum(
            delta_sq / (sum_var + eps), dim=1
        )  # (N,)
        log_p = log_p.clamp(max=-1e-6)  # prevent p=1 singularity
    elif k_form == "identity":
        # K = αI — α controls metric tightness (noise floor per dimension).
        # α=1 recovers the original K=I form; smaller α = tighter metric,
        # stronger normalization penalty, more meaningful variance gradients.
        # |K⁻¹(Si+Sj) + I| = prod(v_d/α + 1)  →  per-dim log(v_d/α + 1)
        # (Si+Sj+K)⁻¹ = diag(1/(v_d + α))
        scale = sum_var + alpha  # (N, D)
        log_p = torch.sum(
            -0.5 * torch.log(scale / alpha) - 0.5 * delta_sq / scale, dim=1
        )  # (N,)
    elif k_form == "expected_distance":
        # E[||z_i - z_j||^2] = ||delta||^2 + trace(Sigma_i) + trace(Sigma_j)
        # Variance is an additive cost, not a divisor.
        # alpha acts as the scale s in sim = exp(-s * E[d^2]).
        expected_dist = torch.sum(delta_sq, dim=1) + torch.sum(sum_var, dim=1)
        log_p = -alpha * expected_dist
        log_p = log_p.clamp(max=-1e-6)  # prevent p=1 singularity

    if neg_threshold is not None:
        # Paper's approach: threshold zeros gradient for well-classified
        # negatives (log_p <= neg_threshold), +1e-6 prevents log(0).
        thresholded = torch.nn.functional.threshold(log_p, neg_threshold, 0.0)
        log_1mp = torch.log(1.0 - torch.exp(thresholded) + 1e-6)
        bce = torch.where(labels == 1, -log_p, -log_1mp)
        loss = torch.mean(bce)

        # Effective loss for reporting: exclude constant from clamped negatives
        with torch.no_grad():
            clamped = (labels == 0) & (log_p <= neg_threshold)
            n_active = (~clamped).sum().clamp(min=1)
            effective = bce.detach().clone()
            effective[clamped] = 0.0
            effective_loss = effective.sum() / n_active

        return loss, effective_loss.item()

    from embedders.nonlinear import _bce_from_log_prob
    return _bce_from_log_prob(log_p, labels)


def mc_expected_loss(left_mean: torch.Tensor, right_mean: torch.Tensor,
                     left_var: torch.Tensor, right_var: torch.Tensor,
                     labels: torch.Tensor, scale: float = 1.0,
                     n_samples: int = 8,
                     neg_threshold: float = None) -> torch.Tensor:
    """E[L(p)] via the reparameterization trick.

    Instead of computing the loss on the closed-form expected similarity,
    this draws MC samples from the Gaussian embeddings and averages the
    deterministic BCE loss over those samples:

        z_i = mu_i + sigma_i * eps_i,   eps ~ N(0, I)
        L_k = BCE(exp(-scale * ||z_i - z_j||^2), y)
        loss = (1/K) * sum_k L_k

    This gives unbiased gradients for both mean and variance networks.

    Parameters
    ----------
    left_mean, right_mean : torch.Tensor
        (N, D) mean embeddings.
    left_var, right_var : torch.Tensor
        (N, D) variance (sigma^2) for each embedding dimension.
    labels : torch.Tensor
        (N,) binary labels -- 1 for positive, 0 for negative.
    scale : float
        Scale in exp(-scale * d^2). Default 1.0 matches deterministic
        training with LOSS_SCALE=1.
    n_samples : int
        Number of MC samples per pair. Default 8.
    neg_threshold : float, optional
        Same as in ``uncertaingen_loss``.

    Returns
    -------
    torch.Tensor
        Scalar loss (or tuple with effective loss if neg_threshold is set).
    """
    from embedders.nonlinear import _bce_from_log_prob

    left_std = torch.sqrt(left_var + 1e-8)
    right_std = torch.sqrt(right_var + 1e-8)

    total_loss = torch.tensor(0.0, device=left_mean.device, dtype=left_mean.dtype)

    for _ in range(n_samples):
        eps_l = torch.randn_like(left_mean)
        eps_r = torch.randn_like(right_mean)
        z_l = left_mean + left_std * eps_l
        z_r = right_mean + right_std * eps_r

        sq_dist = torch.sum((z_l - z_r) ** 2, dim=1)
        log_p = -scale * sq_dist

        if neg_threshold is not None:
            thresholded = torch.nn.functional.threshold(log_p, neg_threshold, 0.0)
            log_1mp = torch.log(1.0 - torch.exp(thresholded) + 1e-6)
            bce = torch.where(labels == 1, -log_p, -log_1mp)
            total_loss = total_loss + torch.mean(bce)
        else:
            total_loss = total_loss + _bce_from_log_prob(log_p, labels)

    return total_loss / n_samples


def train_variance_phase(model: "UncertainGenEmbedder", dataset, lr: float,
                         epochs: int, device: str = "cpu", batch_size: int = 0,
                         num_workers: int = 1, alpha: float = 0.1,
                         verbose: bool = True, neg_threshold: float = None,
                         n_samples: int = 8, scale: float = 1.0):
    """Phase 2 training: freeze mean network, train variance network.

    Uses forward_full() which returns (left_mean, right_mean, left_var, right_var)
    and the marginal probabilistic loss.

    Parameters
    ----------
    model : UncertainGenEmbedder
        UncertainGenEmbedder with mean network already trained.
    dataset : object
        Contrastive dataset.
    lr : float
        Learning rate for variance network.
    epochs : int
        Number of Phase 2 epochs.
    device : str
        Device string.
    batch_size : int
        Batch size (0 = full dataset).
    num_workers : int
        DataLoader workers.
    alpha : float
        Covariance scaling factor (used by all k_forms except "mc").
    verbose : bool
        Print progress.
    neg_threshold : float, optional
        Negative pair loss clamping threshold.
    n_samples : int
        Number of MC samples per pair (only used when k_form="mc").
    scale : float
        Scale in exp(-scale * d^2) (only used when k_form="mc").

    Returns
    -------
    list[float]
        Loss history (average loss per epoch).
    """
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"lr must be positive, got {lr!r}")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    if not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}")

    from torch.utils.data import DataLoader

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.freeze_mean_network()

    effective_batch_size = batch_size if batch_size > 0 else len(dataset)
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        dataset, batch_size=effective_batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=use_cuda and num_workers > 0,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    optimizer = torch.optim.Adam(model.variance_parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    loss_history = []

    epoch_bar = pbar(range(epochs), desc="Phase 2 (variance)", unit="epoch",
                     disable=not verbose)
    for epoch in epoch_bar:
        model.train(True)
        # Re-apply mean network eval after model.train() resets all modules
        model.mean_bn1.eval()
        model.mean_dropout1.eval()
        if hasattr(model, 'feature_net'):
            model.feature_net.eval()
        epoch_loss = 0.0

        batch_bar = pbar(loader, desc=f"  Epoch {epoch + 1}/{epochs}",
                         unit="batch", leave=False, disable=not verbose)
        for data in batch_bar:
            left_features, right_features, labels = data
            optimizer.zero_grad()

            left_features = left_features.reshape(-1, left_features.shape[-1]).to(device, non_blocking=True)
            right_features = right_features.reshape(-1, right_features.shape[-1]).to(device, non_blocking=True)
            labels = labels.reshape(-1).to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                left_mean, right_mean, left_var, right_var = model.forward_full(
                    left_features, right_features
                )
                if model._k_form == "mc":
                    loss_result = mc_expected_loss(
                        left_mean, right_mean, left_var, right_var, labels,
                        scale=scale, n_samples=n_samples,
                        neg_threshold=neg_threshold,
                    )
                else:
                    loss_result = uncertaingen_loss(
                        left_mean, right_mean, left_var, right_var, labels,
                        alpha=alpha, k_form=model._k_form,
                        neg_threshold=neg_threshold,
                    )

            if neg_threshold is not None:
                batch_loss, effective = loss_result
            else:
                batch_loss = loss_result
                effective = batch_loss.item()

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += effective
            batch_bar.set_postfix_str(f"loss={effective:.4f}")

            del batch_loss, left_features, right_features, labels
            del left_mean, right_mean, left_var, right_var
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        epoch_bar.set_postfix_str(f"loss={avg_loss:.6f}")

    model.unfreeze_mean_network()
    return loss_history
