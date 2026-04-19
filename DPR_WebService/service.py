"""Shared services for the MedFusionNet web application."""

from __future__ import annotations

import base64
import io
import random
from pathlib import Path
from threading import Lock
from typing import Any

import matplotlib.cm as cm
import numpy as np
from PIL import Image

from DPR_MedFusionNet.checkpoint_utils import resolve_checkpoint
from DPR_MedFusionNet.config import CLASS_NAMES, CLASS_TO_IDX, SUPPORTED_IMAGE_EXTENSIONS
from DPR_MedFusionNet.inference import MedFusionInference
from DPR_MedFusionNet.preprocessing import load_image


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
MEDFUSION_ROOT = REPO_ROOT / "DPR_MedFusionNet"
TEST_ROOT = MEDFUSION_ROOT / "data" / "test"
RUNS_ROOT = MEDFUSION_ROOT / "runs"
RUNTIME_ROOT = WEB_ROOT / "runtime"
UPLOADS_ROOT = RUNTIME_ROOT / "uploads"

for directory in (RUNTIME_ROOT, UPLOADS_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


def repo_relative(path: str | Path) -> str:
    """Render a path relative to the repository root when possible."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sorted_image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def list_checkpoints() -> list[dict[str, str]]:
    """Discover available checkpoints under the MedFusionNet runs tree."""
    checkpoints = sorted(RUNS_ROOT.glob("**/checkpoints/*.pth"))
    return [
        {
            "value": str(path.resolve()),
            "label": repo_relative(path),
        }
        for path in checkpoints
    ]


def list_all_samples() -> list[dict[str, str]]:
    """Enumerate all sample images from the test split for dropdown browsing."""
    samples: list[dict[str, str]] = []
    for path in _sorted_image_paths(TEST_ROOT):
        relative = path.relative_to(TEST_ROOT)
        label = relative.parts[0]
        samples.append(
            {
                "relative_path": relative.as_posix(),
                "label": label,
                "display_name": f"{label} / {relative.name}",
            }
        )
    return samples


def list_featured_samples(limit_per_class: int = 6) -> list[dict[str, str]]:
    """Build a small gallery of representative test images."""
    return list_featured_samples_for_seed(limit_per_class=limit_per_class, seed=None)


def list_featured_samples_for_seed(
    *,
    limit_per_class: int = 6,
    seed: str | None = None,
) -> list[dict[str, str]]:
    """Build a gallery of representative test images, optionally shuffled by seed."""
    rng = random.Random(seed) if seed is not None else None
    featured: list[dict[str, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = TEST_ROOT / class_name
        class_paths = _sorted_image_paths(class_dir)
        if rng is not None:
            rng.shuffle(class_paths)
        for path in class_paths[:limit_per_class]:
            relative = path.relative_to(TEST_ROOT)
            featured.append(
                {
                    "relative_path": relative.as_posix(),
                    "label": class_name,
                    "display_name": relative.name,
                }
            )
    return featured


def resolve_sample_path(relative_path: str) -> Path:
    """Resolve a user-selected sample path while keeping it inside the test tree."""
    if not relative_path:
        raise ValueError("No sample image was selected.")

    candidate = (TEST_ROOT / relative_path).resolve()
    if TEST_ROOT.resolve() not in {candidate, *candidate.parents}:
        raise ValueError("Invalid sample image path.")
    if not candidate.exists():
        raise FileNotFoundError(f"Sample image not found: {relative_path}")
    if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported sample image extension.")
    return candidate


def pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image into a data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


def array_to_data_url(array: np.ndarray) -> str:
    """Convert a 2D or 3D numpy image array into a PNG data URL."""
    if array.ndim == 2:
        array = cm.get_cmap("jet")(array)[..., :3]
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return pil_to_data_url(Image.fromarray(array))


def probabilities_for_display(probabilities: dict[str, float]) -> list[dict[str, Any]]:
    """Normalize probability rows for template rendering."""
    return [
        {
            "label": class_name,
            "score": float(probabilities.get(class_name, 0.0)),
        }
        for class_name in CLASS_NAMES
    ]


def result_to_view_model(
    result: dict[str, Any],
    *,
    source_label: str,
    source_value: str,
) -> dict[str, Any]:
    """Convert inference output into template-friendly structures."""
    return {
        "source_label": source_label,
        "source_value": source_value,
        "prediction": result["pred_label"],
        "pred_probability": result["pred_probability"],
        "pneumonia_probability": result["pneumonia_probability"],
        "checkpoint_path": repo_relative(result["checkpoint_path"]),
        "checkpoint_epoch": result["checkpoint_epoch"],
        "checkpoint_auc": result["checkpoint_auc"],
        "probabilities": probabilities_for_display(result["probabilities"]),
        "original_image": array_to_data_url(result["image"]),
        "heatmap_image": array_to_data_url(result["heatmap"]) if result["heatmap"] is not None else None,
        "overlay_image": array_to_data_url(result["overlay"]) if result["overlay"] is not None else None,
    }


class ModelStore:
    """Lazy, thread-safe model loader for the web application."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._engine: MedFusionInference | None = None
        self._checkpoint_path: Path | None = None

    def load(self, checkpoint_path: str | Path | None = None) -> MedFusionInference:
        """Load the default model or switch to a specific checkpoint."""
        resolved = resolve_checkpoint(checkpoint_path, runs_root=RUNS_ROOT)
        with self._lock:
            if self._engine is not None and self._checkpoint_path == resolved:
                return self._engine
            if self._engine is not None:
                self._engine.close()
            self._engine = MedFusionInference(checkpoint_path=resolved)
            self._checkpoint_path = resolved
            return self._engine

    def current_engine(self) -> MedFusionInference:
        """Return the active engine, loading the default checkpoint if needed."""
        return self.load()

    def current_model_summary(self) -> dict[str, Any]:
        """Expose model information for the UI header."""
        engine = self.current_engine()
        return {
            "checkpoint_path": repo_relative(engine.checkpoint_path),
            "device": str(engine.device),
            "checkpoint_epoch": engine.checkpoint_meta.get("epoch"),
            "checkpoint_auc": engine.checkpoint_meta.get("auc"),
        }


model_store = ModelStore()
