"""
dataset.py
==========
Dataset PyTorch compatible avec la structure Kaggle Chest X-Ray Images (Pneumonia).

Structure attendue du dataset Kaggle :
    data/
    ├── train/
    │   ├── NORMAL/   *.jpeg
    │   └── PNEUMONIA/ *.jpeg
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/

Usage :
    from dataset import PneumoniaDataset, build_dataloaders
"""

import os
import glob
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np

from preprocessing import build_transforms

# ─── Labels ────────────────────────────────────────────────────────────────────
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ─── Dataset ───────────────────────────────────────────────────────────────────
class PneumoniaDataset(Dataset):
    """
    Dataset PyTorch pour la classification binaire Normal / Pneumonie.

    Supporte optionnellement des annotations de boîtes englobantes (bbox)
    pour la supervision de localisation (L_loc).

    Args:
        root_dir   : Répertoire contenant les dossiers NORMAL/ et PNEUMONIA/
        mode       : 'train' | 'val' | 'test'
        img_size   : Taille de redimensionnement (défaut: 384)
        use_clahe  : Appliquer CLAHE (défaut: True)
        bbox_file  : Fichier CSV optionnel avec annotations bbox
                     (colonnes: image_path, x1, y1, x2, y2)
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        img_size: int = 384,
        use_clahe: bool = True,
        bbox_file: Optional[str] = None,
    ):
        self.root_dir   = Path(root_dir)
        self.mode       = mode
        self.transform  = build_transforms(mode=mode, img_size=img_size, use_clahe=use_clahe)
        self.img_size   = img_size

        # ── Collecte des fichiers images ──────────────────────────────────────
        self.samples: List[Tuple[Path, int]] = []
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Dossier introuvable : {class_dir}")
                continue
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for fp in class_dir.glob(ext):
                    self.samples.append((fp, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {self.root_dir}. "
                               f"Vérifiez la structure du dataset.")

        print(f"[INFO] Dataset '{mode}' → {len(self.samples)} images | "
              f"NORMAL: {self._count(0)} | PNEUMONIA: {self._count(1)}")

        # ── Boîtes englobantes optionnelles ───────────────────────────────────
        self.bboxes: Dict[str, List[float]] = {}
        if bbox_file and Path(bbox_file).exists():
            import csv
            with open(bbox_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.bboxes[row["image_path"]] = [
                        float(row["x1"]), float(row["y1"]),
                        float(row["x2"]), float(row["y2"]),
                    ]
            print(f"[INFO] {len(self.bboxes)} annotations bbox chargées.")

    # ─── Helpers ───────────────────────────────────────────────────────────────
    def _count(self, label: int) -> int:
        return sum(1 for _, l in self.samples if l == label)

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids inverses des fréquences de classe pour WeightedRandomSampler.

        Returns:
            Tensor de poids par échantillon (len = dataset size)
        """
        n_total = len(self.samples)
        n_normal    = self._count(0)
        n_pneumonia = self._count(1)

        # Poids inversement proportionnels à la fréquence de classe
        w_normal    = n_total / (2 * n_normal)    if n_normal    > 0 else 0
        w_pneumonia = n_total / (2 * n_pneumonia) if n_pneumonia > 0 else 0

        weights = [
            w_normal if label == 0 else w_pneumonia
            for _, label in self.samples
        ]
        return torch.tensor(weights, dtype=torch.float)

    # ─── Interface PyTorch ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Chargement et transformation de l'image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Impossible de lire {img_path}: {e}")

        tensor = self.transform(img)   # (3, H, W)

        item = {
            "image": tensor,                                      # (3, 384, 384)
            "label": torch.tensor(label, dtype=torch.long),      # scalaire
            "path":  str(img_path),                               # pour débogage
        }

        # Supervision de localisation (bbox normalisée dans [0, 1])
        key = str(img_path)
        if key in self.bboxes:
            bbox = self.bboxes[key]
            # Normaliser par la taille de l'image d'origine
            # Ici on suppose que les coordonnées sont déjà en pixels 384×384
            item["bbox"] = torch.tensor(
                [b / self.img_size for b in bbox], dtype=torch.float
            )                                                     # (4,) — x1,y1,x2,y2
        else:
            item["bbox"] = torch.zeros(4, dtype=torch.float)     # Pas de bbox

        return item


# ─── Constructeur de DataLoaders ───────────────────────────────────────────────
def build_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 384,
    num_workers: int = 4,
    use_clahe: bool = True,
    use_weighted_sampler: bool = True,
    bbox_file: Optional[str] = None,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Construit les DataLoaders pour train / val / test.

    Args:
        data_dir             : Répertoire racine contenant train/, val/, test/
        batch_size           : Taille de batch
        img_size             : Taille des images
        num_workers          : Processus de chargement parallèle
        use_clahe            : Activer CLAHE
        use_weighted_sampler : Sampling pondéré pour corriger le déséquilibre de classes
        bbox_file            : Fichier CSV avec annotations bbox (optionnel)
        pin_memory           : Copie en mémoire verrouillée (plus rapide sur GPU)

    Returns:
        Dictionnaire {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    data_dir = Path(data_dir)
    loaders  = {}

    for mode in ("train", "val", "test"):
        mode_dir = data_dir / mode
        if not mode_dir.exists():
            print(f"[WARN] Dossier '{mode}' introuvable, ignoré.")
            continue

        dataset = PneumoniaDataset(
            root_dir=str(mode_dir),
            mode=mode,
            img_size=img_size,
            use_clahe=use_clahe,
            bbox_file=bbox_file,
        )

        # Weighted Random Sampler uniquement pour l'entraînement
        sampler = None
        shuffle = (mode == "train")

        if mode == "train" and use_weighted_sampler:
            class_weights = dataset.get_class_weights()
            sampler = WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(class_weights),
                replacement=True,
            )
            shuffle = False   # Incompatible avec sampler

        loaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(mode == "train"),   # Évite les micro-batchs en fin d'époque
        )

    return loaders


# ─── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    # Crée un dataset factice pour tester le pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ("train", "val", "test"):
            for cls in ("NORMAL", "PNEUMONIA"):
                d = Path(tmpdir) / split / cls
                d.mkdir(parents=True)
                for i in range(4):
                    # Images synthétiques 64×64 RGB
                    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), np.uint8))
                    img.save(d / f"img_{i}.png")

        loaders = build_dataloaders(tmpdir, batch_size=4, img_size=64, num_workers=0)
        for split, loader in loaders.items():
            batch = next(iter(loader))
            print(f"  [{split}] images: {batch['image'].shape} | "
                  f"labels: {batch['label']}")

    print("\n  ✅ Dataset et DataLoaders opérationnels !")
