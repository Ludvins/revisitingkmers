"""Probabilistic Contrastive Learning (PCL) for genome embeddings.

Implements the MCInfoNCE framework from:
    Kirchhof et al., "Probabilistic Contrastive Learning Recovers the
    Correct Aleatoric Uncertainty of Ambiguous Inputs" (ICML 2023)

Each DNA fragment is mapped to a von Mises-Fisher (vMF) distribution on the
unit hypersphere, parameterized by a mean direction mu and concentration kappa.
The MCInfoNCE loss samples from these vMF distributions during training and
provably recovers the correct aleatoric uncertainty.
"""
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.base import BaseFeatureExtractor
from features.kmer import KmerFeatureExtractor
from embedders.base import BaseEmbedder, EmbeddingResult
from utils.progress import pbar


EPS = 1e-7

_ACT_MAP = {
    "sigmoid": nn.Sigmoid,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "relu": nn.ReLU,
}


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int],
               activation: str = "sigmoid", use_batchnorm: bool = True,
               dropout: float = 0.2, device: str = "cpu") -> nn.Sequential:
    """Build an MLP with configurable depth, width, activation, BN, and dropout."""
    if activation not in _ACT_MAP:
        raise ValueError(f"activation must be one of {list(_ACT_MAP.keys())}, got {activation!r}")
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim, dtype=torch.float, device=device))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h_dim, dtype=torch.float, device=device))
        act = _ACT_MAP[activation]
        layers.append(act() if isinstance(act, type) else act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim, dtype=torch.float, device=device))
    return nn.Sequential(*layers)


def _migrate_state_dict(old_sd: dict) -> dict:
    """Migrate old-format state dict (mu_linear1, mu_bn1, ...) to new Sequential format."""
    KEY_MAP = {
        "mu_linear1.weight": "mu_net.0.weight",
        "mu_linear1.bias": "mu_net.0.bias",
        "mu_bn1.weight": "mu_net.1.weight",
        "mu_bn1.bias": "mu_net.1.bias",
        "mu_bn1.running_mean": "mu_net.1.running_mean",
        "mu_bn1.running_var": "mu_net.1.running_var",
        "mu_bn1.num_batches_tracked": "mu_net.1.num_batches_tracked",
        "mu_linear2.weight": "mu_net.4.weight",
        "mu_linear2.bias": "mu_net.4.bias",
        "kappa_linear1.weight": "kappa_net.0.weight",
        "kappa_linear1.bias": "kappa_net.0.bias",
        "kappa_bn1.weight": "kappa_net.1.weight",
        "kappa_bn1.bias": "kappa_net.1.bias",
        "kappa_bn1.running_mean": "kappa_net.1.running_mean",
        "kappa_bn1.running_var": "kappa_net.1.running_var",
        "kappa_bn1.num_batches_tracked": "kappa_net.1.num_batches_tracked",
        "kappa_linear2.weight": "kappa_net.4.weight",
        "kappa_linear2.bias": "kappa_net.4.bias",
    }
    new_sd = {}
    for old_key, value in old_sd.items():
        new_sd[KEY_MAP.get(old_key, old_key)] = value
    return new_sd


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# vMF sampling (Wood 1994 rejection sampling)
# ---------------------------------------------------------------------------

class VonMisesFisher:
    """Von Mises-Fisher distribution on the unit hypersphere.

    Implements sampling via the rejection method of Wood (1994).

    Parameters
    ----------
    mu : torch.Tensor
        (..., D) mean direction vectors (must be L2-normalized).
    kappa : torch.Tensor
        (...,) or (..., 1) concentration parameters > 0.
    """

    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor):
        self.mu = mu
        if kappa.dim() == mu.dim():
            kappa = kappa.squeeze(-1)
        self.kappa = kappa
        self.dim = mu.shape[-1]

    def rsample(self, n_samples: int = 1) -> torch.Tensor:
        """Draw samples from the vMF distribution.

        Returns
        -------
        torch.Tensor
            (n_samples, ..., D) samples on the unit sphere.
        """
        shape = self.mu.shape  # (..., D)
        batch_shape = shape[:-1]
        D = self.dim

        # Vectorize over n_samples: expand mu/kappa to (n_samples, ..., D/1)
        # and sample everything in one pass instead of a Python loop.
        expanded_shape = (n_samples, *batch_shape)
        mu_exp = self.mu.unsqueeze(0).expand(n_samples, *shape)
        kappa_exp = self.kappa.unsqueeze(0).expand(expanded_shape)

        # Step 1: Sample w (radial component) for all n_samples at once
        w = self._sample_w(expanded_shape, D, kappa_exp)  # (n_samples, ...)

        # Step 2: Sample v (uniform direction on S^{D-2})
        v = torch.randn(*expanded_shape, D - 1, device=self.mu.device)
        v = F.normalize(v, p=2, dim=-1)

        # Step 3: Combine: z = w * e_1 + sqrt(1 - w^2) * v
        w_sq = torch.clamp(1.0 - w ** 2, min=EPS)
        z = torch.cat([w.unsqueeze(-1), torch.sqrt(w_sq).unsqueeze(-1) * v], dim=-1)

        # Step 4: Rotate from e_1 to mu via Householder reflection
        z = self._householder_rotation(z, mu=mu_exp)

        return z  # (n_samples, ..., D)

    def _sample_w(self, batch_shape, D, kappa=None):
        """Sample the radial component w via rejection sampling."""
        m = D - 1  # dimension of the tangent space
        if kappa is None:
            kappa = self.kappa
        device = self.mu.device

        # Wood (1994) parameters
        b = (-2.0 * kappa + torch.sqrt(4.0 * kappa ** 2 + m ** 2)) / m
        a = (m + 2.0 * kappa + torch.sqrt(4.0 * kappa ** 2 + m ** 2)) / 4.0
        d = 4.0 * a * b / (1.0 + b) - m * math.log(m)

        w = torch.zeros(*batch_shape, device=device)
        done = torch.zeros(*batch_shape, dtype=torch.bool, device=device)

        max_iter = 1000
        for _ in range(max_iter):
            if done.all():
                break

            # Step a: Sample from Beta distribution
            eps = torch.clamp(torch.rand(*batch_shape, device=device), EPS, 1.0 - EPS)
            cos_theta = (1.0 - (1.0 + b) * eps) / (1.0 - (1.0 - b) * eps)

            # Step b: Acceptance test
            t = 2.0 * a * b / (1.0 - (1.0 - b) * eps)
            u = torch.rand(*batch_shape, device=device)
            accept = (m * torch.log(t) - t + d) >= torch.log(u)
            accept = accept & ~done

            w = torch.where(accept, cos_theta, w)
            done = done | accept

        # Fallback for any remaining unaccepted samples: use mean direction
        w = torch.where(done, w, torch.ones_like(w))
        return w

    def _householder_rotation(self, z, mu=None):
        """Rotate z from canonical direction e_1=(1,0,...,0) to mu."""
        if mu is None:
            mu = self.mu
        e1 = torch.zeros_like(mu)
        e1[..., 0] = 1.0

        # Householder vector
        u = e1 - mu
        u_norm = torch.clamp(torch.norm(u, p=2, dim=-1, keepdim=True), min=EPS)
        u = u / u_norm

        # Householder reflection: z - 2 * (z . u) * u
        dot = torch.sum(z * u, dim=-1, keepdim=True)
        return z - 2.0 * dot * u


# ---------------------------------------------------------------------------
# MCInfoNCE Loss
# ---------------------------------------------------------------------------

class MCInfoNCE(nn.Module):
    """Monte Carlo InfoNCE loss with vMF sampling.

    Implements the loss from Kirchhof et al. (2023). Contains a global kappa
    parameter that acts as the inverse temperature for the contrastive
    similarity. Can be learnable or fixed (paper uses fixed kappa=20).

    Parameters
    ----------
    kappa_init : float
        Initial value for the temperature kappa.
    n_samples : int
        Number of Monte Carlo samples per pair.
    device : str or torch.device
        Device for the parameter/buffer.
    learn_loss_kappa : bool
        If True, kappa is a learnable nn.Parameter.
        If False, kappa is a fixed buffer (as in the paper).
    """

    def __init__(self, kappa_init: float = 16.0, n_samples: int = 8,
                 device: str = "cpu", learn_loss_kappa: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.learn_loss_kappa = learn_loss_kappa
        if learn_loss_kappa:
            self.kappa = nn.Parameter(
                torch.ones(1, device=torch.device(device)) * kappa_init
            )
        else:
            self.register_buffer(
                "kappa",
                torch.ones(1, device=torch.device(device)) * kappa_init,
            )

    def forward(self, mu_ref, kappa_ref, mu_pos, kappa_pos,
                mu_neg, kappa_neg):
        """Compute MCInfoNCE loss.

        Parameters
        ----------
        mu_ref : torch.Tensor
            (B, D) reference mean directions.
        kappa_ref : torch.Tensor
            (B,) reference concentrations.
        mu_pos : torch.Tensor
            (B, 1, D) positive mean directions.
        kappa_pos : torch.Tensor
            (B, 1) positive concentrations.
        mu_neg : torch.Tensor
            (B, n_neg, D) negative mean directions.
        kappa_neg : torch.Tensor
            (B, n_neg) negative concentrations.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Expand ref to (B, 1, D) for broadcasting
        mu_ref = mu_ref.unsqueeze(1)
        kappa_ref = kappa_ref.unsqueeze(1)

        # Sample from vMF: (n_samples, B, *, D)
        samples_ref = VonMisesFisher(mu_ref, kappa_ref).rsample(self.n_samples)
        samples_pos = VonMisesFisher(mu_pos, kappa_pos).rsample(self.n_samples)

        if mu_neg is not None:
            samples_neg = VonMisesFisher(mu_neg, kappa_neg).rsample(self.n_samples)
        else:
            # Roll positive samples as negatives (within-batch negatives)
            samples_neg = torch.roll(samples_pos, 1, dims=1)

        # Dot product similarities scaled by learnable kappa
        # sim_pos: (n_samples, B, 1)
        sim_pos = torch.sum(samples_ref * samples_pos, dim=-1) * self.kappa
        # sim_neg: (n_samples, B, n_neg)
        sim_neg = torch.sum(samples_ref * samples_neg, dim=-1) * self.kappa

        # InfoNCE: log p(pos) = sim_pos - logsumexp(sim_pos, sim_neg)
        n_neg = sim_neg.shape[-1]
        # logsumexp over negatives: (n_samples, B)
        neg_lse = torch.logsumexp(sim_neg, dim=-1) if n_neg > 0 else torch.zeros_like(sim_pos.squeeze(-1))
        # Combine pos and neg for denominator
        log_denom = torch.logsumexp(
            torch.stack([sim_pos.squeeze(-1), neg_lse], dim=0), dim=0
        )
        log_prob = sim_pos.squeeze(-1) - log_denom  # (n_samples, B)

        # logmeanexp over MC samples: logsumexp - log(n_samples)
        log_prob = torch.logsumexp(log_prob, dim=0) - math.log(self.n_samples)

        return -torch.mean(log_prob)


# ---------------------------------------------------------------------------
# PCLEmbedder
# ---------------------------------------------------------------------------

class PCLEmbedder(BaseEmbedder, nn.Module):
    """Probabilistic Contrastive Learning embedder.

    Produces vMF embeddings: each input is mapped to a direction mu on the
    unit hypersphere and a concentration parameter kappa. Higher kappa means
    more confident (less uncertain).

    The architecture is configurable via ``hidden_dims``, ``activation``,
    ``use_batchnorm``, and ``dropout``. Defaults reproduce the original
    2-layer architecture for backwards compatibility.

    Parameters
    ----------
    k : int, optional
        K-mer size for KmerFeatureExtractor.
    dim : int
        Embedding dimension.
    device : str
        Device for model parameters.
    kappa_mode : str
        ``"implicit"`` (kappa = norm of raw embedding) or ``"explicit"``
        (separate network head with calibration).
    kappa_min : float
        Minimum kappa value (explicit mode calibration target).
    kappa_max : float
        Maximum kappa value (explicit mode calibration target).
    kappa_init : float
        Initial kappa value (explicit mode, for bias initialization).
    hidden_dims : list[int], optional
        Hidden layer widths for the mean network. Defaults to ``[512]``.
    activation : str
        Activation function: ``"sigmoid"``, ``"leaky_relu"``, or ``"relu"``.
    use_batchnorm : bool
        Whether to use BatchNorm after each hidden layer.
    dropout : float
        Dropout probability (0.0 to disable).
    kappa_hidden_dims : list[int], optional
        Hidden layer widths for the kappa network (explicit mode).
        Defaults to same as ``hidden_dims``.
    """

    def __init__(self, k: int = None, dim: int = 256, device: str = "cpu",
                 verbose: bool = False, seed: int = 0,
                 input_dim: int = None,
                 feature_extractor: BaseFeatureExtractor = None,
                 kappa_mode: str = "implicit",
                 kappa_min: float = 16.0, kappa_max: float = 128.0,
                 kappa_init: float = 32.0,
                 hidden_dims: list[int] = None,
                 activation: str = "sigmoid",
                 use_batchnorm: bool = True,
                 dropout: float = 0.2,
                 kappa_hidden_dims: list[int] = None):
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if kappa_mode not in ("implicit", "explicit"):
            raise ValueError(f"kappa_mode must be 'implicit' or 'explicit', got {kappa_mode!r}")

        sources = sum(x is not None for x in (k, input_dim, feature_extractor))
        if sources != 1:
            raise ValueError(
                f"Exactly one of k, input_dim, or feature_extractor must be provided "
                f"(got {sources})"
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
        self._kappa_mode = kappa_mode
        self._kappa_min = kappa_min
        self._kappa_max = kappa_max
        self._kappa_init = kappa_init
        self._hidden_dims = hidden_dims if hidden_dims is not None else [512]
        self._activation = activation
        self._use_batchnorm = use_batchnorm
        self._dropout = dropout
        self._kappa_hidden_dims = kappa_hidden_dims

        # Determine input dimension
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

        # Mean network (mu_net)
        self.mu_net = _build_mlp(
            input_dim, dim, hidden_dims=self._hidden_dims,
            activation=self._activation, use_batchnorm=self._use_batchnorm,
            dropout=self._dropout, device=str(self._device),
        )

        # Kappa network (explicit mode only)
        if kappa_mode == "explicit":
            kh = self._kappa_hidden_dims if self._kappa_hidden_dims is not None else self._hidden_dims
            self.kappa_net = _build_mlp(
                input_dim, 1, hidden_dims=kh,
                activation=self._activation, use_batchnorm=self._use_batchnorm,
                dropout=self._dropout, device=str(self._device),
            )
            self.register_buffer(
                "kappa_upscale", torch.tensor(1.0, device=self._device)
            )
            self.register_buffer(
                "kappa_add", torch.tensor(0.0, device=self._device)
            )
            self._init_kappa_bias()

    def _init_kappa_bias(self):
        """Initialize explicit kappa network bias so initial output ~ kappa_init."""
        target = math.log(max(self._kappa_init, 1.0))
        last_linear = self.kappa_net[-1]
        nn.init.constant_(last_linear.bias, target)

    def _rescale_kappa(self, sample_features: torch.Tensor = None):
        """Calibrate kappa_upscale and kappa_add to map output to [kappa_min, kappa_max].

        Should be called once after initialization with a sample batch
        to calibrate the output range, following Kirchhof et al.
        """
        if self._kappa_mode != "explicit":
            return

        with torch.no_grad():
            if sample_features is not None:
                raw = self._kappa_net_raw(sample_features)  # (N, 1)
                raw = F.softplus(raw).squeeze(-1)  # (N,)
                raw_min = raw.min().item()
                raw_max = raw.max().item()
            else:
                # Assume output range from softplus(bias) ± some range
                raw_min = 0.1
                raw_max = 5.0

            # We want: exp(upscale * raw_min + add) = kappa_min
            #          exp(upscale * raw_max + add) = kappa_max
            log_kmin = math.log(max(self._kappa_min, EPS))
            log_kmax = math.log(max(self._kappa_max, EPS))

            if abs(raw_max - raw_min) > EPS:
                upscale = (log_kmax - log_kmin) / (raw_max - raw_min)
                add = log_kmin - upscale * raw_min
            else:
                upscale = 1.0
                add = math.log(max(self._kappa_init, EPS))

            self.kappa_upscale.fill_(upscale)
            self.kappa_add.fill_(add)

    def _kappa_net_raw(self, features: torch.Tensor) -> torch.Tensor:
        """Raw kappa network output before transformation."""
        return self.kappa_net(features)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, kappa).

        Parameters
        ----------
        features : torch.Tensor
            (N, input_dim) input features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (mu, kappa) where mu is (N, D) L2-normalized and kappa is (N,).
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")

        raw = self.mu_net(features)  # (N, dim) unnormalized

        if self._kappa_mode == "implicit":
            kappa = torch.norm(raw, p=2, dim=1)  # (N,)
            kappa = torch.clamp(kappa, min=EPS)
            mu = raw / kappa.unsqueeze(1)
        else:
            mu = F.normalize(raw, p=2, dim=1)
            kappa_raw = self._kappa_net_raw(features)  # (N, 1)
            kappa = torch.exp(
                self.kappa_upscale * F.softplus(kappa_raw) + self.kappa_add
            ).squeeze(-1)  # (N,)

        return mu, kappa

    def forward(self, left_features: torch.Tensor,
                right_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning means only (for deterministic training/warmup).

        Compatible with train_contrastive() from train.py.
        """
        if hasattr(self, 'feature_net'):
            left_features = self.feature_net(left_features)
            right_features = self.feature_net(right_features)
        left_mu, _ = self.encode(left_features)
        right_mu, _ = self.encode(right_features)
        return left_mu, right_mu

    def forward_full(self, left_features: torch.Tensor,
                     right_features: torch.Tensor):
        """Forward pass returning (mu, kappa) for both sides (for MCInfoNCE).

        Returns
        -------
        tuple
            (left_mu, right_mu, left_kappa, right_kappa)
        """
        if hasattr(self, 'feature_net'):
            left_features = self.feature_net(left_features)
            right_features = self.feature_net(right_features)
        left_mu, left_kappa = self.encode(left_features)
        right_mu, right_kappa = self.encode(right_features)
        return left_mu, right_mu, left_kappa, right_kappa

    def embed(self, inputs) -> EmbeddingResult:
        """Embed inputs as vMF distributions.

        Returns EmbeddingResult with distribution="vmf", mean=mu (L2-normalized),
        and kappa (concentration parameter).
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

            mu, kappa = self.encode(features_t)
            mean_np = mu.cpu().numpy()
            kappa_np = kappa.cpu().numpy()

        return EmbeddingResult(mean=mean_np, kappa=kappa_np, distribution="vmf")

    def copy_mean_from(self, source, strict: bool = True) -> None:
        """Copy pre-trained mean network weights from a NonLinearEmbedder.

        This allows initializing from a trained deterministic model.
        Requires matching architecture (hidden_dims=[512], sigmoid, BN, dropout=0.2)
        unless strict=False.
        """
        from embedders.nonlinear import NonLinearEmbedder
        if not isinstance(source, NonLinearEmbedder):
            raise TypeError(
                f"source must be a NonLinearEmbedder, got {type(source).__name__}"
            )

        if hasattr(self, 'feature_net') and hasattr(source, 'feature_net'):
            self.feature_net.load_state_dict(source.feature_net.state_dict())

        # Build source state dict with Sequential-style keys
        # NonLinearEmbedder has: linear1, batch1, (Sigmoid), (Dropout), linear2
        # which maps to Sequential indices [0, 1, 2, 3, 4] when
        # hidden_dims=[512], use_batchnorm=True, dropout=0.2
        source_sd = {}
        source_sd['0.weight'] = source.linear1.weight.data
        source_sd['0.bias'] = source.linear1.bias.data
        source_sd['1.weight'] = source.batch1.weight.data
        source_sd['1.bias'] = source.batch1.bias.data
        source_sd['1.running_mean'] = source.batch1.running_mean
        source_sd['1.running_var'] = source.batch1.running_var
        source_sd['1.num_batches_tracked'] = source.batch1.num_batches_tracked
        source_sd['4.weight'] = source.linear2.weight.data
        source_sd['4.bias'] = source.linear2.bias.data

        try:
            self.mu_net.load_state_dict(source_sd, strict=strict)
        except RuntimeError as e:
            if strict:
                raise RuntimeError(
                    f"Architecture mismatch between PCLEmbedder "
                    f"(hidden_dims={self._hidden_dims}) and NonLinearEmbedder "
                    f"(hidden_dims=[512]). Use strict=False for partial copy, "
                    f"or use matching architecture. Original error: {e}"
                ) from e
            else:
                import warnings
                own_sd = self.mu_net.state_dict()
                for key in source_sd:
                    if key in own_sd and own_sd[key].shape == source_sd[key].shape:
                        own_sd[key] = source_sd[key]
                    else:
                        warnings.warn(
                            f"Skipping key '{key}' during copy_mean_from "
                            f"(shape mismatch or missing)"
                        )
                self.mu_net.load_state_dict(own_sd)

    def freeze_mu_net(self):
        """Freeze mean network parameters."""
        for param in self.mu_net.parameters():
            param.requires_grad = False

    def unfreeze_mu_net(self):
        """Unfreeze mean network parameters."""
        for param in self.mu_net.parameters():
            param.requires_grad = True

    def freeze_kappa_net(self):
        """Freeze kappa network parameters (explicit mode only)."""
        if self._kappa_mode != "explicit":
            return
        for param in self.kappa_net.parameters():
            param.requires_grad = False

    def unfreeze_kappa_net(self):
        """Unfreeze kappa network parameters (explicit mode only)."""
        if self._kappa_mode != "explicit":
            return
        for param in self.kappa_net.parameters():
            param.requires_grad = True

    def save(self, path: str) -> None:
        """Save model parameters and constructor kwargs to disk."""
        if self._verbose:
            print(f"Saving model to: {path}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        module = self.module if isinstance(self, torch.nn.DataParallel) else self
        kwargs = {
            "k": module._k,
            "dim": module._dim,
            "device": str(module._device),
            "input_dim": module._input_dim,
            "kappa_mode": module._kappa_mode,
            "kappa_min": module._kappa_min,
            "kappa_max": module._kappa_max,
            "kappa_init": module._kappa_init,
            "hidden_dims": module._hidden_dims,
            "activation": module._activation,
            "use_batchnorm": module._use_batchnorm,
            "dropout": module._dropout,
            "kappa_hidden_dims": module._kappa_hidden_dims,
        }
        torch.save([kwargs, module.state_dict()], path)

    @classmethod
    def load(cls, path: str, device: str = "cpu",
             feature_extractor=None) -> "PCLEmbedder":
        """Load a saved PCLEmbedder from disk."""
        kwargs, state_dict = torch.load(
            path, map_location=torch.device(device), weights_only=True
        )
        kwargs["device"] = device

        # Backwards compatibility: old checkpoints lack architecture params
        if "hidden_dims" not in kwargs:
            kwargs["hidden_dims"] = [512]
            kwargs["activation"] = "sigmoid"
            kwargs["use_batchnorm"] = True
            kwargs["dropout"] = 0.2
        if "kappa_hidden_dims" not in kwargs:
            kwargs["kappa_hidden_dims"] = None

        if feature_extractor is not None:
            kwargs.pop("k", None)
            kwargs.pop("input_dim", None)
            kwargs["feature_extractor"] = feature_extractor
        elif kwargs.get("k") is not None:
            kwargs.pop("input_dim", None)

        model = cls(**kwargs)

        # Handle state dict key migration from old format
        if "mu_linear1.weight" in state_dict and "mu_net.0.weight" not in state_dict:
            state_dict = _migrate_state_dict(state_dict)

        model.load_state_dict(state_dict)
        model.to(device)
        return model

    @classmethod
    def from_paper_config(cls, k: int = None, dim: int = 256,
                          device: str = "cpu", width_mult: int = 2,
                          **kwargs) -> "PCLEmbedder":
        """Create a PCLEmbedder with the Kirchhof et al. (2023) architecture.

        Paper: 6 hidden layers, LeakyReLU, no BN, no Dropout.
        Kappa network has 1 fewer hidden layer than mu network.

        Parameters
        ----------
        k : int, optional
            K-mer size.
        dim : int
            Embedding dimension D.
        device : str
            Device.
        width_mult : int
            Width multiplier for hidden layers. Paper uses 50 (50*D).
            For small datasets, use 2 (=512 for D=256).
        **kwargs
            Additional kwargs forwarded to __init__ (e.g., kappa_mode, seed).
        """
        D = dim
        W = width_mult
        hidden_dims = [10 * D, W * D, W * D, W * D, W * D, W * D, 10 * D, 10 * D]
        kappa_hidden_dims = hidden_dims[:-1]

        return cls(
            k=k, dim=dim, device=device,
            hidden_dims=hidden_dims,
            kappa_hidden_dims=kappa_hidden_dims,
            activation="leaky_relu",
            use_batchnorm=False,
            dropout=0.0,
            **kwargs,
        )

    @property
    def default_metric(self) -> str:
        return "l2"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_pcl(model: PCLEmbedder, dataset, lr: float, device: str = "cpu",
              batch_size: int = 0, num_workers: int = 0,
              n_phases: int = 0, n_batches_per_half_phase: int = 10000,
              n_mc_samples: int = 8, loss_kappa_init: float = 16.0,
              learn_loss_kappa: bool = True,
              lr_decrease_after_phase: float = 0.5,
              save_path: str = None, verbose: bool = True) -> list[float]:
    """Train a PCLEmbedder with MCInfoNCE loss.

    Parameters
    ----------
    model : PCLEmbedder
        Model to train.
    dataset : BaseContrastiveDataset
        Dataset yielding (left_features, right_features, labels).
    lr : float
        Learning rate.
    device : str
        Training device.
    batch_size : int
        Batch size (0 = auto).
    num_workers : int
        DataLoader workers.
    n_phases : int
        Number of alternating phases. 0 = joint training.
    n_batches_per_half_phase : int
        Batches per half-phase (mu or kappa training).
    n_mc_samples : int
        MC samples for MCInfoNCE.
    loss_kappa_init : float
        Initial value for the temperature in MCInfoNCE.
    learn_loss_kappa : bool
        If True (default), loss kappa is learnable. If False, fixed.
    lr_decrease_after_phase : float
        LR decay factor after each half-phase.
    save_path : str, optional
        Path to save the model after training.
    verbose : bool
        Print progress.

    Returns
    -------
    list[float]
        Loss values per batch.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.train()

    neg_per_pos = dataset._neg_sample_per_pos

    effective_batch_size = batch_size if batch_size > 0 else len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=effective_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    loss_fn = MCInfoNCE(
        kappa_init=loss_kappa_init, n_samples=n_mc_samples, device=str(device),
        learn_loss_kappa=learn_loss_kappa,
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    all_losses = []

    if n_phases == 0:
        # Joint training
        total_batches = n_batches_per_half_phase * 2  # same total as 1 phase
        params = list(model.parameters()) + list(loss_fn.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_batches_per_half_phase,
            gamma=lr_decrease_after_phase,
        )

        batch_count = 0
        data_iter = iter(loader)
        for batch_count in pbar(range(total_batches), desc="PCL joint training",
                                unit="batch", disable=not verbose,
                                mininterval=5):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            loss = _train_step(model, loss_fn, batch, optimizer, scaler,
                               device, use_amp, neg_per_pos)
            scheduler.step()
            all_losses.append(loss)

            if verbose and (batch_count + 1) % 500 == 0:
                print(f"  batch {batch_count + 1}/{total_batches}, "
                      f"loss={loss:.4f}, loss_kappa={loss_fn.kappa.item():.2f}")
    else:
        # Alternating training
        for phase in range(n_phases):
            if verbose:
                print(f"\n--- Phase {phase + 1}/{n_phases} ---")

            # Half-phase 1: Train mu, freeze kappa
            model.unfreeze_mu_net()
            model.freeze_kappa_net()
            params = [p for p in model.parameters() if p.requires_grad]
            params += list(loss_fn.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)

            data_iter = iter(loader)
            for b in pbar(range(n_batches_per_half_phase),
                          desc=f"Phase {phase+1} mu", unit="batch",
                          disable=not verbose, mininterval=5):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                loss = _train_step(model, loss_fn, batch, optimizer, scaler,
                                   device, use_amp, neg_per_pos)
                all_losses.append(loss)

            lr *= lr_decrease_after_phase

            # Half-phase 2: Train kappa, freeze mu
            model.freeze_mu_net()
            model.unfreeze_kappa_net()
            params = [p for p in model.parameters() if p.requires_grad]
            params += list(loss_fn.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)

            data_iter = iter(loader)
            for b in pbar(range(n_batches_per_half_phase),
                          desc=f"Phase {phase+1} kappa", unit="batch",
                          disable=not verbose, mininterval=5):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                loss = _train_step(model, loss_fn, batch, optimizer, scaler,
                                   device, use_amp, neg_per_pos)
                all_losses.append(loss)

            lr *= lr_decrease_after_phase

            if verbose:
                print(f"  Phase {phase+1} done. loss_kappa={loss_fn.kappa.item():.2f}")

        # Unfreeze everything at the end
        model.unfreeze_mu_net()
        model.unfreeze_kappa_net()

    if save_path:
        model.save(save_path)

    return all_losses


def _train_step(model, loss_fn, batch, optimizer, scaler, device,
                use_amp, neg_per_pos):
    """Single training step: forward + MCInfoNCE loss + backward."""
    left_features, right_features, labels = batch
    left_features = left_features.to(device, non_blocking=True)
    right_features = right_features.to(device, non_blocking=True)

    # Reshape from (B, 1+neg, feat) to (B*(1+neg), feat)
    B = left_features.shape[0]
    group_size = 1 + neg_per_pos
    left_features = left_features.reshape(-1, left_features.shape[-1])
    right_features = right_features.reshape(-1, right_features.shape[-1])

    optimizer.zero_grad()

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        left_mu, right_mu, left_kappa, right_kappa = model.forward_full(
            left_features, right_features
        )

        # Reshape back to grouped form
        left_mu = left_mu.view(B, group_size, -1)
        right_mu = right_mu.view(B, group_size, -1)
        left_kappa = left_kappa.view(B, group_size)
        right_kappa = right_kappa.view(B, group_size)

        # Extract ref (anchor), pos, neg
        mu_ref = left_mu[:, 0, :]          # (B, D)
        kappa_ref = left_kappa[:, 0]       # (B,)
        mu_pos = right_mu[:, 0:1, :]      # (B, 1, D)
        kappa_pos = right_kappa[:, 0:1]   # (B, 1)
        mu_neg = right_mu[:, 1:, :]       # (B, neg, D)
        kappa_neg = right_kappa[:, 1:]    # (B, neg)

        loss = loss_fn(mu_ref, kappa_ref, mu_pos, kappa_pos, mu_neg, kappa_neg)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()
