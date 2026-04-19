"""Image preprocessing utilities aligned with MedFusionNet_v4.ipynb."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from .config import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:  # pragma: no cover - script execution fallback
    from config import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_transforms(
    train: bool = False,
    img_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """Notebook-compatible preprocessing and optional train augmentations."""
    if train:
        resize_size = max(img_size, int(round(img_size * 256 / 224)))
        return transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(image: str | Path | Image.Image) -> Image.Image:
    """Load an image source into RGB PIL format."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def preprocess_image(
    image: str | Path | Image.Image,
    img_size: int = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Convert an image source into a normalized ``(1, 3, H, W)`` tensor."""
    pil_image = load_image(image)
    tensor = build_transforms(train=False, img_size=img_size)(pil_image)
    return tensor.unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display purposes."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    image = tensor.detach().cpu() * std.cpu() + mean.cpu()
    image = image.permute(1, 2, 0).clamp(0, 1).numpy()
    return image


class ImagePreprocessor:
    """Small callable wrapper used by the inference engine."""

    def __init__(self, img_size: int = DEFAULT_IMAGE_SIZE) -> None:
        self.img_size = img_size
        self.transform = build_transforms(train=False, img_size=img_size)

    def __call__(self, image: str | Path | Image.Image) -> torch.Tensor:
        pil_image = load_image(image)
        return self.transform(pil_image).unsqueeze(0)
