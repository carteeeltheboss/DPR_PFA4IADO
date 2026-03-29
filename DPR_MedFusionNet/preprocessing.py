"""
preprocessing.py
================
Pipeline de prétraitement pour les radiographies thoraciques (CXR).

Étapes :
1. Redimensionnement : 384 × 384 pixels
2. Amélioration du contraste : CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. Normalisation Z-score (μ=0.485, σ=0.229 — stats ImageNet mono-canal)

Usage :
    from preprocessing import build_transforms, CXRPreprocessor
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional

# ─── Constantes ────────────────────────────────────────────────────────────────
IMG_SIZE      = 384            # Résolution cible (384 × 384)
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # Moyennes ImageNet (3 canaux)
IMAGENET_STD  = [0.229, 0.224, 0.225]   # Écarts-types ImageNet (3 canaux)

# Paramètres CLAHE
CLAHE_CLIP_LIMIT    = 2.0      # Limite de coupure (plus élevé → plus de contraste)
CLAHE_TILE_GRID     = (8, 8)   # Grille de tuiles pour l'égalisation locale


# ─── CLAHE sur image PIL ────────────────────────────────────────────────────────
class CLAHETransform:
    """
    Applique CLAHE sur une image PIL en niveaux de gris puis la convertit en RGB.
    
    CLAHE améliore le contraste local des radiographies, rendant visibles
    les opacités subtiles liées à la pneumonie.
    """

    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img: Image PIL (L ou RGB)

        Returns:
            Image PIL RGB après application de CLAHE
        """
        # Convertir en niveaux de gris uint8
        img_gray = np.array(img.convert("L"))          # (H, W), uint8

        # Créer l'objet CLAHE (non-picklable, donc instancié ici)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

        # Appliquer CLAHE
        img_clahe = clahe.apply(img_gray)              # (H, W), uint8

        # Convertir en RGB (3 canaux identiques) pour la compatibilité avec ImageNet
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)  # (H, W, 3)
        return Image.fromarray(img_rgb)


# ─── Normalisation Z-score personnalisée ───────────────────────────────────────
class ZScoreNormalize:
    """
    Normalisation Z-score : x_norm = (x - μ) / σ
    
    Si compute_stats=True, calcule μ et σ sur l'image elle-même (per-image).
    Sinon, utilise les statistiques ImageNet pré-calculées.
    """

    def __init__(
        self,
        mean: list = IMAGENET_MEAN,
        std: list  = IMAGENET_STD,
        per_image:  bool = False,
    ):
        self.mean      = torch.tensor(mean).view(3, 1, 1)
        self.std       = torch.tensor(std).view(3, 1, 1)
        self.per_image = per_image

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Tensor (C, H, W) flottant dans [0, 1]

        Returns:
            Tensor normalisé
        """
        if self.per_image:
            # Normalisation par image (robuste aux variations d'exposition)
            mean = tensor.mean(dim=[1, 2], keepdim=True)
            std  = tensor.std(dim=[1, 2], keepdim=True).clamp(min=1e-6)
            return (tensor - mean) / std
        else:
            # Normalisation avec statistiques ImageNet
            return (tensor - self.mean) / self.std


# ─── Constructeur de transforms ────────────────────────────────────────────────
def build_transforms(
    mode: str = "train",
    img_size: int = IMG_SIZE,
    use_clahe: bool = True,
    per_image_norm: bool = False,
) -> transforms.Compose:
    """
    Construit le pipeline de transformations selon le mode d'utilisation.

    Args:
        mode          : 'train' | 'val' | 'test'
        img_size      : Taille cible (défaut: 384)
        use_clahe     : Appliquer CLAHE avant ToTensor
        per_image_norm: Normalisation Z-score par image (vs statistiques globales)

    Returns:
        transforms.Compose prêt à l'emploi
    """
    assert mode in ("train", "val", "test"), \
        f"mode doit être 'train', 'val' ou 'test', reçu: '{mode}'"

    # ── Étape 1 : CLAHE (optionnel, appliqué en espace PIL) ──
    initial = [CLAHETransform()] if use_clahe else []

    # ── Étape 2 : Augmentations spécifiques au mode d'entraînement ──
    if mode == "train":
        augmentations = [
            transforms.Resize((img_size + 20, img_size + 20)),   # Légèrement plus grand
            transforms.RandomCrop((img_size, img_size)),          # Crop aléatoire
            transforms.RandomHorizontalFlip(p=0.5),               # Flip horizontal
            transforms.RandomRotation(degrees=10),                 # Rotation ±10°
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Jitter radiologique
            transforms.RandomAffine(                               # Déformation légère
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
        ]
    else:
        # Val / Test : resize déterministe sans augmentation
        augmentations = [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((img_size, img_size)),
        ]

    # ── Étape 3 : Conversion en Tensor + Normalisation Z-score ──
    post = [
        transforms.ToTensor(),                        # [0,1], shape (3, H, W)
        ZScoreNormalize(per_image=per_image_norm),    # Normalisation Z-score
    ]

    return transforms.Compose(initial + augmentations + post)


# ─── Préprocesseur standalone (sans DataLoader) ────────────────────────────────
class CXRPreprocessor:
    """
    Préprocesseur standalone pour traiter une image CXR unique.
    Utile pour l'inférence ou la visualisation.

    Exemple :
        preprocessor = CXRPreprocessor()
        tensor = preprocessor("path/to/image.png")   # torch.Tensor (1, 3, 384, 384)
    """

    def __init__(self, img_size: int = IMG_SIZE, use_clahe: bool = True):
        self.transform = build_transforms(
            mode="test",
            img_size=img_size,
            use_clahe=use_clahe,
        )

    def __call__(self, image_path: str) -> torch.Tensor:
        """
        Args:
            image_path: Chemin vers l'image (PNG, JPEG, DICOM non supporté ici)

        Returns:
            Tensor (1, 3, H, W) prêt pour le modèle
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img)          # (3, H, W)
        return tensor.unsqueeze(0)            # (1, 3, H, W) — batch dimension


# ─── Statistiques de dataset (pour normalisation globale) ──────────────────────
def compute_dataset_stats(
    image_paths: list,
    img_size: int = IMG_SIZE,
    sample_size: Optional[int] = None,
) -> Tuple[list, list]:
    """
    Calcule la moyenne et l'écart-type globaux d'un dataset pour la normalisation.

    Args:
        image_paths : Liste de chemins vers les images
        img_size    : Taille de redimensionnement
        sample_size : Nombre d'images à échantillonner (None = tout)

    Returns:
        (mean, std) — listes de 3 valeurs (canaux R, G, B)
    """
    if sample_size is not None:
        image_paths = np.random.choice(image_paths, size=sample_size, replace=False)

    transform = transforms.Compose([
        CLAHETransform(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),              # (3, H, W) dans [0, 1]
    ])

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = len(image_paths)

    for path in image_paths:
        try:
            img    = Image.open(path).convert("RGB")
            tensor = transform(img)               # (3, H, W)
            mean  += tensor.mean(dim=[1, 2])
            std   += tensor.std(dim=[1, 2])
        except Exception as e:
            print(f"[WARN] Impossible de lire {path}: {e}")
            n -= 1

    mean /= n
    std  /= n

    print(f"[INFO] Statistiques dataset ({n} images):")
    print(f"       Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"       Std : [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean.tolist(), std.tolist()


# ─── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("=" * 60)
    print("  MedFusionNet — Test du pipeline de prétraitement")
    print("=" * 60)

    # Test avec une image synthétique
    dummy_img = Image.fromarray(
        (np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
    )

    for mode in ("train", "val", "test"):
        tfm    = build_transforms(mode=mode)
        tensor = tfm(dummy_img)
        print(f"  [{mode:5s}] Output shape: {tensor.shape} | "
              f"min={tensor.min():.3f} | max={tensor.max():.3f}")

    print("\n  ✅ Pipeline de prétraitement opérationnel !")
