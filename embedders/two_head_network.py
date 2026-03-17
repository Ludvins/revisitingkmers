"""TwoHeadNetwork embedder with shared backbone and split mean/variance heads.

Architecture:
    input → [backbone: any nn.Module] → shared features (N, backbone_dim)
                                           ├─→ Linear(backbone_dim, dim)              → mean (μ)
                                           └─→ Linear(backbone_dim, dim) → Softplus   → variance (σ²)

Unlike UncertainGen (which has completely separate mean and variance networks),
this model shares all computation up to the last layer, then splits into two
linear heads.

Three training strategies are supported (``training_strategy`` parameter):

``"joint"`` (default):
    All parameters train together with a single probabilistic loss.

``"warmup"``:
    Freeze the variance head for the first ``warmup_epochs`` epochs so the
    backbone and mean head stabilize first, then unfreeze with a lower LR
    (scaled by ``variance_lr_scale``).  Inspired by:
    - Detlefsen et al., "Reliable Training and Estimation of Variance
      Networks", NeurIPS 2019 — recommends disabling variance for the first
      N iterations as "standard practice".
    - Kirchhof et al., "Probabilistic Contrastive Learning", ICML 2023 —
      explicit two-phase ``--n_phases`` parameter.
    - Shi & Jain, "Probabilistic Face Embeddings", ICCV 2019 — fully
      pre-trains backbone, then trains uncertainty module with frozen backbone.

``"stop_gradient"``:
    Backbone and mean head receive gradients only from a deterministic
    contrastive loss; the variance head receives gradients only from the
    probabilistic loss with detached means and detached shared features.
    This provably preserves mean quality.  Based on:
    - Stirn & Knowles, "Faithful Heteroscedastic Regression", AISTATS 2023 —
      two stop-gradient ops prevent the variance objective from corrupting the
      mean fit, giving the same guarantee as a homoscedastic baseline.
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from embedders.base import BaseEmbedder, EmbeddingResult
from embedders.nonlinear import contrastive_loss
from embedders.uncertaingen import uncertaingen_loss, _VALID_K_FORMS

_VALID_STRATEGIES = {"joint", "warmup", "stop_gradient"}
from utils.progress import pbar


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


class TwoHeadNetworkEmbedder(BaseEmbedder, nn.Module):
    """Probabilistic embedder with shared backbone and split heads.

    The backbone is any nn.Module that maps (N, input_dim) → (N, backbone_dim).
    Two linear heads project the shared representation to mean and variance.

    Parameters
    ----------
    backbone : nn.Module
        Module mapping inputs to shared features.
    backbone_dim : int
        Output dimension of the backbone.
    dim : int, optional
        Embedding dimension (output of each head). Default is 256.
    alpha : float, optional
        Covariance scaling factor for the probabilistic loss. Default is 1.0.
    k_form : str
        Which K matrix form for Lemma 3.1: ``"adaptive"`` or ``"identity"``.
    training_strategy : str, optional
        ``"joint"`` (default), ``"warmup"``, or ``"stop_gradient"``.
        See module docstring for details.
    warmup_epochs : int, optional
        Epochs to freeze variance head (``"warmup"`` strategy only). Default 0.
    variance_lr_scale : float, optional
        LR multiplier for the variance head relative to base LR. Default 1.0.
        Applied in ``"warmup"`` and ``"joint"`` strategies.
    device : str, optional
        Device string. Default is ``"cpu"``.
    verbose : bool, optional
        Print progress. Default is False.
    seed : int, optional
        Random seed. Default is 0.
    """

    def __init__(self, backbone: nn.Module, backbone_dim: int, dim: int = 256,
                 alpha: float = 1.0, k_form: str = None,
                 training_strategy: str = "joint",
                 warmup_epochs: int = 0, variance_lr_scale: float = 1.0,
                 device: str = "cpu",
                 verbose: bool = False, seed: int = 0):
        if not isinstance(backbone, nn.Module):
            raise TypeError(f"backbone must be an nn.Module, got {type(backbone).__name__}")
        if not isinstance(backbone_dim, int) or backbone_dim <= 0:
            raise ValueError(f"backbone_dim must be a positive integer, got {backbone_dim}")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if k_form is None:
            raise ValueError(
                "k_form is required -- specify 'adaptive' for K=alpha*(Si+Sj) "
                "or 'identity' for K=I"
            )
        if k_form not in _VALID_K_FORMS:
            raise ValueError(f"k_form must be one of {_VALID_K_FORMS}, got {k_form!r}")
        if training_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"training_strategy must be one of {_VALID_STRATEGIES}, "
                f"got {training_strategy!r}"
            )
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if variance_lr_scale <= 0:
            raise ValueError(f"variance_lr_scale must be > 0, got {variance_lr_scale}")

        super().__init__()

        self._device = torch.device(device) if isinstance(device, str) else device
        self._verbose = verbose
        self._backbone_dim = backbone_dim
        self._dim = dim
        self._alpha = alpha
        self._k_form = k_form
        self._training_strategy = training_strategy
        self._warmup_epochs = warmup_epochs
        self._variance_lr_scale = variance_lr_scale
        self._seed = seed

        set_seed(seed)

        self.backbone = backbone
        self.mean_head = nn.Linear(backbone_dim, dim, dtype=torch.float, device=self._device)

        var_linear = nn.Linear(backbone_dim, dim, dtype=torch.float, device=self._device)
        # Initialize bias to -5 so Softplus outputs small values (~0.007) at start.
        # Without this, large initial variance pushes all probabilities to 0 and
        # the loss gets stuck in a plateau.
        nn.init.constant_(var_linear.bias, -5.0)
        self.var_head = nn.Sequential(var_linear, nn.Softplus())

        # Move everything (including the caller-provided backbone) to device
        self.to(self._device)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass input through backbone, then mean and variance heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, input_dim)``.

        Returns
        -------
        mean : torch.Tensor
            Mean embeddings of shape ``(N, dim)``.
        variance : torch.Tensor
            Variance embeddings of shape ``(N, dim)``.
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")

        shared = self.backbone(x)
        return self.mean_head(shared), self.var_head(shared)

    def forward(self, left_features: torch.Tensor,
                right_features: torch.Tensor) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass returning means and variances.

        Parameters
        ----------
        left_features : torch.Tensor
            Left input tensor of shape ``(N, input_dim)``.
        right_features : torch.Tensor
            Right input tensor of shape ``(N, input_dim)``.

        Returns
        -------
        left_mean : torch.Tensor
            Mean embeddings for left inputs, shape ``(N, dim)``.
        right_mean : torch.Tensor
            Mean embeddings for right inputs, shape ``(N, dim)``.
        left_var : torch.Tensor
            Variance for left inputs, shape ``(N, dim)``.
        right_var : torch.Tensor
            Variance for right inputs, shape ``(N, dim)``.
        """
        if left_features.ndim != 2:
            raise ValueError(f"left_features must be 2D, got shape {left_features.shape}")
        if right_features.ndim != 2:
            raise ValueError(f"right_features must be 2D, got shape {right_features.shape}")
        if left_features.shape[-1] != right_features.shape[-1]:
            raise ValueError(
                f"left_features and right_features must have matching last dimension, "
                f"got {left_features.shape[-1]} and {right_features.shape[-1]}"
            )

        left_mean, left_var = self.encode(left_features)
        right_mean, right_var = self.encode(right_features)
        return left_mean, right_mean, left_var, right_var

    def forward_stop_gradient(self, left_features: torch.Tensor,
                              right_features: torch.Tensor) -> tuple[
                                  torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        """Forward pass with stop-gradient decoupling (Stirn & Knowles, 2023).

        Backbone and mean head receive normal gradients. The variance head
        receives shared features with ``detach()``, so its gradients cannot
        flow back into the backbone.

        Returns the same tuple as ``forward()``.
        """
        left_shared = self.backbone(left_features)
        right_shared = self.backbone(right_features)

        left_mean = self.mean_head(left_shared)
        right_mean = self.mean_head(right_shared)

        # Detach shared features: variance gradients do not reach backbone
        left_var = self.var_head(left_shared.detach())
        right_var = self.var_head(right_shared.detach())

        return left_mean, right_mean, left_var, right_var

    def embed(self, inputs) -> EmbeddingResult:
        """Embed inputs as Gaussian distributions.

        Parameters
        ----------
        inputs : np.ndarray or torch.Tensor
            Raw inputs to embed.

        Returns
        -------
        EmbeddingResult
            Result with both ``mean`` and ``variance`` populated.
        """
        if isinstance(inputs, torch.Tensor):
            if inputs.numel() == 0:
                raise ValueError("inputs must not be empty")
        elif isinstance(inputs, np.ndarray):
            if inputs.size == 0:
                raise ValueError("inputs must not be empty")
        else:
            if len(inputs) == 0:
                raise ValueError("inputs must not be empty")

        self.eval()
        with torch.inference_mode():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.as_tensor(np.asarray(inputs), dtype=torch.float)
            inputs = inputs.to(self._device)
            mean, variance = self.encode(inputs)
        return EmbeddingResult(mean=mean.cpu().numpy(), variance=variance.cpu().numpy())

    def get_k(self) -> int:
        """Return None (TwoHeadNetworkEmbedder does not use k-mers)."""
        return None

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
            "backbone_dim": module._backbone_dim,
            "dim": module._dim,
            "alpha": module._alpha,
            "k_form": module._k_form,
            "training_strategy": module._training_strategy,
            "warmup_epochs": module._warmup_epochs,
            "variance_lr_scale": module._variance_lr_scale,
            "device": str(module._device),
        }
        torch.save([kwargs, module.state_dict()], path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", backbone: nn.Module = None,
             **kwargs) -> "TwoHeadNetworkEmbedder":
        """Load a saved TwoHeadNetworkEmbedder.

        The backbone architecture is not serialized -- the caller must provide
        the same backbone used during training. The backbone weights ARE
        restored from the state dict.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        device : str, optional
            Device to load onto. Default is ``"cpu"``.
        backbone : nn.Module, optional
            Module with the same architecture used during training. If None,
            a dummy linear layer is used (only valid if the original backbone
            was also linear).
        **kwargs
            Additional keyword arguments forwarded to the constructor.

        Returns
        -------
        TwoHeadNetworkEmbedder
            The loaded model with restored weights.
        """
        saved_kwargs, state_dict = torch.load(path, map_location=torch.device(device))
        saved_kwargs["device"] = device
        # Backward compat: old checkpoints may not have these fields
        if "k_form" not in saved_kwargs:
            saved_kwargs["k_form"] = "adaptive"
        if "training_strategy" not in saved_kwargs:
            saved_kwargs["training_strategy"] = "joint"
        if "warmup_epochs" not in saved_kwargs:
            saved_kwargs["warmup_epochs"] = 0
        if "variance_lr_scale" not in saved_kwargs:
            saved_kwargs["variance_lr_scale"] = 1.0
        if backbone is None:
            # Reconstruct a dummy linear backbone from backbone_dim
            # (only works if the original backbone was also linear)
            backbone = nn.Linear(saved_kwargs["backbone_dim"], saved_kwargs["backbone_dim"])
        model = cls(backbone=backbone, **saved_kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    @property
    def default_metric(self) -> str:
        return "l2"


def train_two_head_network(model: TwoHeadNetworkEmbedder, dataset, lr: float, epochs: int,
              device: str = "cpu", batch_size: int = 0, num_workers: int = 1,
              alpha: float = 1.0, save_path: str = None,
              verbose: bool = True, neg_threshold: float = None):
    """Training loop for TwoHeadNetworkEmbedder.

    The training strategy is read from ``model._training_strategy``:

    ``"joint"``:
        All parameters train together with the probabilistic loss.
        If ``model._variance_lr_scale != 1.0``, the variance head gets a
        separate (scaled) learning rate.

    ``"warmup"`` (Detlefsen et al., NeurIPS 2019; Kirchhof et al., ICML 2023):
        Freeze variance head for the first ``model._warmup_epochs`` epochs.
        Backbone + mean head train with the probabilistic loss (variance fixed
        at init values).  After warmup, unfreeze variance head at
        ``lr * model._variance_lr_scale``.

    ``"stop_gradient"`` (Stirn & Knowles, AISTATS 2023):
        Backbone + mean head receive gradients only from a deterministic
        contrastive loss.  Variance head receives gradients only from the
        probabilistic loss with detached means and detached backbone features.
        The two losses are summed.

    Parameters
    ----------
    model : TwoHeadNetworkEmbedder
        Model to train (strategy params read from its attributes).
    dataset : BaseContrastiveDataset
        Contrastive dataset yielding ``(left, right, labels)`` tuples.
    lr : float
        Learning rate for backbone + mean head.
    epochs : int
        Number of training epochs.
    device : str, optional
        Device string. Default is ``"cpu"``.
    batch_size : int, optional
        Batch size. ``0`` uses the full dataset. Default is 0.
    num_workers : int, optional
        DataLoader workers. Default is 1.
    alpha : float, optional
        Covariance scaling factor for the loss. Default is 1.0.
    save_path : str, optional
        Path to save the final model. Default is None.
    verbose : bool, optional
        Print training progress. Default is True.

    Returns
    -------
    dict[str, list[float]]
        Per-epoch average losses.  Always contains ``"total"``.
        For ``stop_gradient``, also contains ``"mean"`` and ``"var"``.
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")

    from torch.utils.data import DataLoader

    strategy = model._training_strategy
    warmup_epochs = model._warmup_epochs
    var_lr_scale = model._variance_lr_scale

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)

    effective_batch_size = batch_size if batch_size > 0 else len(dataset)
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        dataset, batch_size=effective_batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=use_cuda and num_workers > 0,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # --- Build optimizer with separate parameter groups ---
    backbone_mean_params = list(model.backbone.parameters()) + \
                           list(model.mean_head.parameters())
    var_params = list(model.var_head.parameters())

    optimizer = torch.optim.Adam([
        {"params": backbone_mean_params, "lr": lr},
        {"params": var_params, "lr": lr * var_lr_scale},
    ])

    # --- Warmup: freeze variance head initially ---
    if strategy == "warmup" and warmup_epochs > 0:
        for p in var_params:
            p.requires_grad = False
        if verbose:
            print(f"  Variance head frozen for first {warmup_epochs} epochs "
                  f"(var_lr_scale={var_lr_scale})")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    track_components = (strategy == "stop_gradient")
    loss_history = {"total": []}
    if track_components:
        loss_history["mean"] = []
        loss_history["var"] = []

    strategy_label = {"joint": "joint", "warmup": "warmup",
                      "stop_gradient": "stop-grad"}[strategy]
    epoch_bar = pbar(range(epochs), desc=f"Training ({strategy_label})",
                     unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        model.train(True)
        epoch_loss = 0.0
        epoch_mean_loss = 0.0
        epoch_var_loss = 0.0

        # --- Warmup: unfreeze variance head after warmup_epochs ---
        if strategy == "warmup" and epoch == warmup_epochs:
            for p in var_params:
                p.requires_grad = True
            if verbose:
                print(f"\n  Variance head unfrozen at epoch {epoch + 1} "
                      f"(lr={lr * var_lr_scale:.1e})")

        batch_bar = pbar(loader, desc=f"  Epoch {epoch + 1}/{epochs}",
                         unit="batch", leave=False, disable=not verbose)
        for data in batch_bar:
            left_features, right_features, labels = data
            optimizer.zero_grad()

            left_features = left_features.reshape(-1, left_features.shape[-1]).to(device, non_blocking=True)
            right_features = right_features.reshape(-1, right_features.shape[-1]).to(device, non_blocking=True)
            labels = labels.reshape(-1).to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if strategy == "stop_gradient":
                    # Stirn & Knowles (AISTATS 2023): separate gradient paths.
                    left_mean, right_mean, left_var, right_var = \
                        model.forward_stop_gradient(left_features, right_features)

                    mean_loss = contrastive_loss(left_mean, right_mean, labels)
                    var_result = uncertaingen_loss(
                        left_mean.detach(), right_mean.detach(),
                        left_var, right_var, labels,
                        alpha=alpha, k_form=model._k_form,
                        neg_threshold=neg_threshold,
                    )
                    if neg_threshold is not None:
                        var_loss, effective_var = var_result
                    else:
                        var_loss = var_result
                        effective_var = var_loss.item()
                    batch_loss = mean_loss + var_loss
                else:
                    left_mean, right_mean, left_var, right_var = model(
                        left_features, right_features
                    )
                    loss_result = uncertaingen_loss(
                        left_mean, right_mean, left_var, right_var, labels,
                        alpha=alpha, k_form=model._k_form,
                        neg_threshold=neg_threshold,
                    )
                    if neg_threshold is not None:
                        batch_loss, effective_var = loss_result
                    else:
                        batch_loss = loss_result
                        effective_var = batch_loss.item()

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            effective_total = (mean_loss.item() + effective_var
                               if track_components else effective_var)
            epoch_loss += effective_total
            if track_components:
                epoch_mean_loss += mean_loss.item()
                epoch_var_loss += effective_var
            if track_components:
                batch_bar.set_postfix_str(
                    f"mean={mean_loss.item():.4f} var={effective_var:.4f}")
            else:
                batch_bar.set_postfix_str(f"loss={effective_total:.4f}")

            del batch_loss, left_features, right_features, labels
            del left_mean, right_mean, left_var, right_var
            torch.cuda.empty_cache()

        n_batches = len(loader)
        loss_history["total"].append(epoch_loss / n_batches)
        if track_components:
            loss_history["mean"].append(epoch_mean_loss / n_batches)
            loss_history["var"].append(epoch_var_loss / n_batches)
            epoch_bar.set_postfix_str(
                f"mean={epoch_mean_loss / n_batches:.4f} "
                f"var={epoch_var_loss / n_batches:.4f}")
        else:
            epoch_bar.set_postfix_str(f"loss={epoch_loss / n_batches:.6f}")

    if save_path:
        model.save(save_path)
        if verbose:
            print(f"Model saved to: {save_path}")

    return loss_history
