"""Image feature extractors for MNIST, CIFAR-10, and custom image data."""

import numpy as np
import torch
import torch.nn as nn
from features.base import BaseFeatureExtractor


class ImageFeatureExtractor(BaseFeatureExtractor):
    """Flattens images to pixel vectors with optional normalization.

    Numpy-based: features are pre-extracted at dataset init time (no gradients).
    Suitable for feeding into the MLP encoder directly.
    """

    def __init__(self, image_shape: tuple[int, ...], normalize: bool = False):
        """
        Parameters
        ----------
        image_shape : tuple[int, ...]
            (C, H, W) shape of input images.
            MNIST: (1, 28, 28) -> dim=784
            CIFAR-10: (3, 32, 32) -> dim=3072
        normalize : bool
            If True, divide pixel values by 255 when they are
            in [0, 255] range. torchvision.ToTensor() already
            normalizes to [0, 1], so set to False when using it.
        """
        if not isinstance(image_shape, tuple):
            raise TypeError(f"image_shape must be a tuple, got {type(image_shape).__name__}")
        if len(image_shape) == 0:
            raise ValueError("image_shape must be a non-empty tuple of positive integers")
        for i, d in enumerate(image_shape):
            if not isinstance(d, int) or d <= 0:
                raise ValueError(f"All image_shape dimensions must be positive integers, got {d!r} at index {i}")

        self._image_shape = image_shape
        self._normalize = normalize
        self._dim = 1
        for d in image_shape:
            self._dim *= d

    @property
    def feature_dim(self) -> int:
        return self._dim

    def extract(self, image) -> np.ndarray:
        """Flatten a single image to a pixel vector.

        Parameters
        ----------
        image : torch.Tensor or array-like
            Input image in (C, H, W) format.

        Returns
        -------
        np.ndarray
            Flattened float32 pixel vector of length ``feature_dim``.
        """
        if not isinstance(image, (torch.Tensor, np.ndarray, list, tuple)):
            raise TypeError(f"image must be a Tensor, ndarray, list, or tuple, got {type(image).__name__}")

        if isinstance(image, torch.Tensor):
            flat = image.cpu().numpy().flatten().astype(np.float32)
        else:
            flat = np.asarray(image, dtype=np.float32).flatten()
        if self._normalize and flat.max() > 1.0:
            flat = flat / 255.0
        return flat

    def extract_batch(self, images, **kwargs) -> np.ndarray:
        """Flatten a batch of images to pixel vectors.

        Parameters
        ----------
        images : torch.Tensor or list
            Batch of images. Tensor shape ``(N, C, H, W)`` or list of images.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, feature_dim)``.
        """
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            flat = images.reshape(images.shape[0], -1).cpu().numpy().astype(np.float32)
            if self._normalize and flat.max() > 1.0:
                flat = flat / 255.0
            return flat
        return np.array([self.extract(img) for img in images])


class CNNFeatureExtractor(BaseFeatureExtractor, nn.Module):
    """Learnable CNN feature extractor for images.

    When passed to an embedder, it is registered as an nn.Module submodule
    and trained end-to-end with the MLP encoder.

    Architecture: Conv→BN→ReLU→Pool → Conv→BN→ReLU→Pool → Flatten → Linear

    Accepts flattened pixel vectors (N, C*H*W) and internally reshapes
    to (N, C, H, W) before conv layers. This makes it compatible with
    datasets that store flattened images.
    """

    def __init__(self, in_channels: int, image_size: int, output_dim: int = 256):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (1 for MNIST, 3 for CIFAR-10).
        image_size : int
            Spatial dimension (28 for MNIST, 32 for CIFAR-10).
        output_dim : int
            Output feature dimension after FC layer.
        """
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels must be a positive integer, got {in_channels!r}")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim!r}")

        nn.Module.__init__(self)
        self._in_channels = in_channels
        self._image_size = image_size
        self._output_dim = output_dim
        self._flat_dim = in_channels * image_size * image_size

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Compute flattened size after two pool layers
        pooled_size = image_size // 4  # two 2x2 pools
        conv_flat_dim = 64 * pooled_size * pooled_size

        self.fc = nn.Linear(conv_flat_dim, output_dim)

    @property
    def feature_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (N, C*H*W) -> (N, output_dim)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor (batch_size, features), got shape {tuple(x.shape)}")

        # Unflatten from (N, C*H*W) to (N, C, H, W)
        x = x.view(-1, self._in_channels, self._image_size, self._image_size)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def extract(self, image) -> np.ndarray:
        """Extract CNN features from a single image without gradients.

        Parameters
        ----------
        image : torch.Tensor or array-like
            Single image (any shape, will be flattened internally).

        Returns
        -------
        np.ndarray
            Feature vector of length ``output_dim``.
        """
        if isinstance(image, torch.Tensor):
            flat = image.flatten().unsqueeze(0).float()
        else:
            flat = torch.tensor(image, dtype=torch.float32).flatten().unsqueeze(0)
        with torch.inference_mode():
            return self.forward(flat).squeeze(0).cpu().numpy()

    def extract_batch(self, images, **kwargs) -> np.ndarray:
        """Extract CNN features from a batch of images without gradients.

        Parameters
        ----------
        images : torch.Tensor or list
            Batch of images.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, output_dim)``.
        """
        if isinstance(images, torch.Tensor):
            batch = images.reshape(images.shape[0], -1).float()
        else:
            batch = torch.tensor(np.array([
                np.asarray(img, dtype=np.float32).flatten() for img in images
            ]))
        with torch.inference_mode():
            return self.forward(batch).cpu().numpy()


class DeepCNNFeatureExtractor(BaseFeatureExtractor, nn.Module):
    """Configurable-depth CNN feature extractor for images.

    Like CNNFeatureExtractor but supports arbitrary number of
    Conv->BN->ReLU->Pool blocks via the ``channels`` parameter.

    Architecture: [Conv->BN->ReLU->Pool] x N -> Flatten -> Linear

    Accepts flattened pixel vectors (N, C*H*W) and internally reshapes
    to (N, C, H, W) before conv layers.
    """

    def __init__(self, in_channels: int, image_size: int, output_dim: int = 256,
                 channels: list[int] = None):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (1 for MNIST, 3 for CIFAR).
        image_size : int
            Spatial dimension (28 for MNIST, 32 for CIFAR).
        output_dim : int
            Output feature dimension after FC layer.
        channels : list[int], optional
            Number of output channels for each conv block.
            Default: ``[32, 64]`` (same as CNNFeatureExtractor).
        """
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError(f"in_channels must be a positive integer, got {in_channels!r}")
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim!r}")

        if channels is None:
            channels = [32, 64]
        if not channels:
            raise ValueError("channels must be a non-empty list of positive integers")
        for i, ch in enumerate(channels):
            if not isinstance(ch, int) or ch <= 0:
                raise ValueError(f"All channel values must be positive integers, got {ch!r} at index {i}")

        n_layers = len(channels)
        pooled_size = image_size // (2 ** n_layers)
        if pooled_size < 1:
            raise ValueError(
                f"Too many conv layers ({n_layers}) for image_size={image_size}: "
                f"spatial dimension would be {image_size / (2 ** n_layers):.1f}"
            )

        nn.Module.__init__(self)
        self._in_channels = in_channels
        self._image_size = image_size
        self._output_dim = output_dim
        self._flat_dim = in_channels * image_size * image_size
        self._channels = list(channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.convs.append(nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        conv_flat_dim = channels[-1] * pooled_size * pooled_size
        self.fc = nn.Linear(conv_flat_dim, output_dim)

    @property
    def feature_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (N, C*H*W) -> (N, output_dim)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor (batch_size, features), got shape {tuple(x.shape)}")

        x = x.view(-1, self._in_channels, self._image_size, self._image_size)

        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(self.relu(bn(conv(x))))

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def extract(self, image) -> np.ndarray:
        """Extract features from a single image without gradients."""
        if isinstance(image, torch.Tensor):
            flat = image.flatten().unsqueeze(0).float()
        else:
            flat = torch.tensor(image, dtype=torch.float32).flatten().unsqueeze(0)
        with torch.inference_mode():
            return self.forward(flat).squeeze(0).cpu().numpy()

    def extract_batch(self, images, **kwargs) -> np.ndarray:
        """Extract features from a batch of images without gradients."""
        if isinstance(images, torch.Tensor):
            batch = images.reshape(images.shape[0], -1).float()
        else:
            batch = torch.tensor(np.array([
                np.asarray(img, dtype=np.float32).flatten() for img in images
            ]))
        with torch.inference_mode():
            return self.forward(batch).cpu().numpy()


class PretrainedFeatureExtractor(BaseFeatureExtractor):
    """Wraps any pretrained torch model as a frozen feature extractor.

    Features are extracted via forward pass with no_grad. The model's
    parameters are NOT trained — use this for fixed pretrained backbones.

    For end-to-end training with a pretrained backbone, use
    CNNFeatureExtractor or pass a custom nn.Module directly.
    """

    def __init__(self, model: nn.Module, output_dim: int,
                 input_is_flat: bool = True, device: str = "cpu"):
        """
        Parameters
        ----------
        model : nn.Module
            Any torch model that returns (N, output_dim) tensors.
        output_dim : int
            Dimension of the model's output.
        input_is_flat : bool
            If True, inputs are flat numpy arrays. If False,
            inputs are passed to the model as-is.
        device : str
            Device for inference.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module instance, got {type(model).__name__}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim!r}")

        self._model = model
        self._output_dim = output_dim
        self._input_is_flat = input_is_flat
        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return self._output_dim

    def extract(self, item) -> np.ndarray:
        """Extract features from a single item using the frozen model.

        Parameters
        ----------
        item : torch.Tensor or array-like
            Single input item.

        Returns
        -------
        np.ndarray
            Feature vector of length ``output_dim``.
        """
        if isinstance(item, torch.Tensor):
            x = item.unsqueeze(0).float().to(self._device)
        else:
            x = torch.tensor(item, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            return self._model(x).squeeze(0).cpu().numpy()

    def extract_batch(self, items, **kwargs) -> np.ndarray:
        """Extract features from a batch of items using the frozen model.

        Parameters
        ----------
        items : torch.Tensor or list
            Batch of input items.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, output_dim)``.
        """
        if isinstance(items, torch.Tensor):
            x = items.float().to(self._device)
        else:
            x = torch.tensor(np.array(items), dtype=torch.float32).to(self._device)
        with torch.inference_mode():
            return self._model(x).cpu().numpy()
