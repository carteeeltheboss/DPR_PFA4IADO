"""Notebook-v4 MedFusionNet architecture, split back into a module."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - dependency issue
    raise RuntimeError(
        "timm is required for MedFusionNet. Install dependencies from requirements.txt."
    ) from exc


class MedFusionNet(nn.Module):
    """
    Swin-Tiny + DenseNet-121 fusion model used by the v4 notebook checkpoints.

    The checkpoint in ``DPR_MedFusionNet/runs/.../Model_19Apr26.pth`` stores
    weights for:
    - ``swin_tiny_patch4_window7_224``
    - ``densenet121`` with pooled features
    - a 3-layer MLP fusion head
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.densenet = timm.create_model(
            "densenet121",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        swin_dim = self.swin.num_features
        dense_dim = self.densenet.num_features
        fused_dim = swin_dim + dense_dim

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return branch features plus the concatenated fusion vector."""
        swin_features = self.swin(x)
        dense_features = self.densenet(x)

        if swin_features.ndim > 2:
            swin_features = swin_features.flatten(1)
        if dense_features.ndim > 2:
            dense_features = dense_features.flatten(1)

        fused_features = torch.cat([swin_features, dense_features], dim=1)
        return swin_features, dense_features, fused_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, fused_features = self.forward_features(x)
        return self.fusion(fused_features)

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
    ) -> list[dict[str, object]]:
        """Compatibility helper kept from the notebook."""
        backbone_params = list(self.swin.parameters()) + list(self.densenet.parameters())
        head_params = list(self.fusion.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]
