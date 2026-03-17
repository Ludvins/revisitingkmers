import re
import torch
from torch.utils.data import DataLoader
from datasets.base import BaseContrastiveDataset
from utils.progress import pbar

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def train_contrastive(model: torch.nn.Module, dataset: BaseContrastiveDataset,
                      loss_fn, lr: float, epochs: int, device: str = "cpu",
                      batch_size: int = 0, num_workers: int = 1,
                      loss_name: str = "bern",
                      save_path: str = None, loss_log_path: str = None,
                      checkpoint_interval: int = 0, verbose: bool = True,
                      parameters=None):
    """Generic contrastive training loop.

    Works with any model that accepts (left_features, right_features) and returns
    (left_embeddings, right_embeddings), any contrastive dataset, and any loss
    function with signature loss_fn(left_emb, right_emb, labels, name).

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model with forward(left, right) -> (left_emb, right_emb).
    dataset : BaseContrastiveDataset
        A BaseContrastiveDataset yielding (left, right, labels).
    loss_fn : callable
        Contrastive loss function.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    device : str
        Device string ("cpu" or "cuda").
    batch_size : int
        Batch size (0 = full dataset).
    num_workers : int
        DataLoader workers.
    loss_name : str
        Name passed to loss_fn (e.g., "bern", "poisson", "hinge").
    save_path : str, optional
        Path to save the final model.
    loss_log_path : str, optional
        Path for TensorBoard logs.
    checkpoint_interval : int
        Save checkpoint every N epochs (0 = disabled).
    verbose : bool
        Print training progress.
    parameters : iterable, optional
        Optional iterable of parameters to optimize. Defaults to model.parameters().

    Returns
    -------
    list[float]
        Per-epoch average training loss.
    """
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if not callable(loss_fn):
        raise TypeError(f"loss_fn must be callable, got {type(loss_fn).__name__}")
    if checkpoint_interval < 0:
        raise ValueError(f"checkpoint_interval must be non-negative, got {checkpoint_interval}")

    device = torch.device(device) if isinstance(device, str) else device

    # Wrap in DataParallel for multi-GPU; must unwrap via .module when saving
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs!")
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

    # Allow caller to restrict which parameters are optimized (e.g., Phase 1 mean-only)
    params = parameters if parameters is not None else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    writer = SummaryWriter(loss_log_path) if loss_log_path and SummaryWriter else None

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    loss_history = []

    epoch_bar = pbar(range(epochs), desc="Training", unit="epoch",
                     disable=not verbose)
    for epoch in epoch_bar:
        model.train(True)
        epoch_loss = 0.0

        batch_bar = pbar(loader, desc=f"  Epoch {epoch + 1}/{epochs}",
                         unit="batch", leave=False, disable=not verbose)
        for data in batch_bar:
            left_features, right_features, labels = data

            optimizer.zero_grad()

            # Flatten batch dimensions while preserving feature shape:
            #   2D features (batch, 1+neg, feat) -> (-1, feat)
            #   3D features (batch, 1+neg, W, feat) -> (-1, W, feat)
            feat_dims = left_features.shape[2:]
            left_features = left_features.reshape(-1, *feat_dims).to(device, non_blocking=True)
            right_features = right_features.reshape(-1, *feat_dims).to(device, non_blocking=True)
            labels = labels.reshape(-1).to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                left_emb, right_emb = model(left_features, right_features)
                batch_loss = loss_fn(left_emb, right_emb, labels, name=loss_name)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += batch_loss.item()
            batch_bar.set_postfix_str(f"loss={batch_loss.item():.4f}")

            del batch_loss, left_features, right_features, labels, left_emb, right_emb
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        epoch_bar.set_postfix_str(f"loss={avg_loss:.6f}")

        if writer:
            writer.add_scalar("Loss/train", avg_loss, epoch + 1)
            writer.flush()

        if save_path and checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = re.sub(r"epoch.*_LR", f"epoch={epoch + 1}_LR", save_path)
            _save_model(model, checkpoint_path, verbose)

    if writer:
        writer.close()

    if save_path:
        _save_model(model, save_path, verbose)

    return loss_history


def _save_model(model: torch.nn.Module, path: str, verbose: bool = True):
    """Save model, handling DataParallel wrapper."""
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    actual_model.save(path)
    if verbose:
        print(f"Model saved to: {path}")
