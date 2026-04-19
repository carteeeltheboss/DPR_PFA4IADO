"""Local inference utilities for the notebook-v4 MedFusionNet checkpoint."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_mpl_dir = Path(tempfile.gettempdir()) / "medfusionnet-mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

try:
    from .checkpoint_utils import (
        extract_state_dict,
        infer_num_classes,
        resolve_checkpoint,
        torch_load_compat,
    )
    from .config import CLASS_NAMES, DEFAULT_IMAGE_SIZE
    from .model import MedFusionNet
    from .preprocessing import ImagePreprocessor, denormalize, load_image
except ImportError:  # pragma: no cover - script execution fallback
    from checkpoint_utils import (
        extract_state_dict,
        infer_num_classes,
        resolve_checkpoint,
        torch_load_compat,
    )
    from config import CLASS_NAMES, DEFAULT_IMAGE_SIZE
    from model import MedFusionNet
    from preprocessing import ImagePreprocessor, denormalize, load_image


def auto_device(device: str | torch.device | None = None) -> torch.device:
    """Pick a sensible default device for local inference."""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def overlay_cam(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap over a display image."""
    heatmap_rgb = cm.jet(heatmap)[..., :3]
    blended = alpha * heatmap_rgb + (1.0 - alpha) * image_np
    return np.clip(blended, 0.0, 1.0)


class GradCAM:
    """Grad-CAM implementation aligned with the notebook target layer."""

    def __init__(self, model: MedFusionNet) -> None:
        self.model = model
        self.target_layer = self.model.densenet.features.denseblock4
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """Generate a normalized Grad-CAM heatmap for a single image tensor."""
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("GradCAM expects a single image tensor with shape (1, C, H, W).")

        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, class_idx, logits.detach()

    def close(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()


class MedFusionInference:
    """High-level inference wrapper for local scripts and the future web UI."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        img_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.device = auto_device(device)
        self.img_size = img_size
        self.checkpoint_path = resolve_checkpoint(checkpoint_path)

        checkpoint_obj = torch_load_compat(self.checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint_obj)
        num_classes = infer_num_classes(state_dict)

        self.model = MedFusionNet(num_classes=num_classes, pretrained=False).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.class_names = CLASS_NAMES[:num_classes]
        self.preprocessor = ImagePreprocessor(img_size=img_size)
        self.gradcam = GradCAM(self.model)
        self.checkpoint_meta = checkpoint_obj if isinstance(checkpoint_obj, dict) else {}

    def _prepare_input(
        self,
        image: str | Path | Image.Image | torch.Tensor,
    ) -> tuple[torch.Tensor, Image.Image | None, str | None]:
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = load_image(image)
            tensor = self.preprocessor(pil_image).to(self.device)
            return tensor, pil_image, image_path

        if isinstance(image, Image.Image):
            pil_image = load_image(image)
            tensor = self.preprocessor(pil_image).to(self.device)
            return tensor, pil_image, None

        if not torch.is_tensor(image):
            raise TypeError("image must be a path, PIL.Image, or torch.Tensor")

        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("Tensor input must have shape (3, H, W) or (1, 3, H, W).")
        return tensor.to(self.device), None, None

    def _display_image(
        self,
        tensor: torch.Tensor,
        pil_image: Image.Image | None,
    ) -> np.ndarray:
        if pil_image is not None:
            resized = pil_image.resize((self.img_size, self.img_size))
            return np.asarray(resized, dtype=np.float32) / 255.0
        return denormalize(tensor)

    def predict(
        self,
        image: str | Path | Image.Image | torch.Tensor,
        compute_cam: bool = True,
    ) -> dict[str, Any]:
        tensor, pil_image, image_path = self._prepare_input(image)

        if compute_cam:
            heatmap, pred_idx, logits = self.gradcam.generate(tensor)
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor)
            pred_idx = int(logits.argmax(dim=1).item())
            heatmap = None

        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        display_image = self._display_image(tensor, pil_image)

        result: dict[str, Any] = {
            "image_path": image_path,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint_meta.get("epoch"),
            "checkpoint_auc": self.checkpoint_meta.get("auc"),
            "pred_index": pred_idx,
            "pred_label": self.class_names[pred_idx],
            "pred_probability": float(probs[pred_idx]),
            "pneumonia_probability": float(probs[1]) if len(probs) > 1 else float(probs[0]),
            "probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probs.tolist())
            },
            "image": display_image,
            "heatmap": heatmap,
            "overlay": overlay_cam(display_image, heatmap) if heatmap is not None else None,
        }
        return result

    def predict_batch(
        self,
        image_paths: list[str | Path],
        output_csv: str | Path | None = None,
        compute_cam: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for image_path in image_paths:
            result = self.predict(image_path, compute_cam=compute_cam)
            results.append(result)

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_path",
                        "pred_label",
                        "pred_index",
                        "pred_probability",
                        "pneumonia_probability",
                        "checkpoint_path",
                        "checkpoint_epoch",
                        "checkpoint_auc",
                    ],
                )
                writer.writeheader()
                for result in results:
                    writer.writerow(
                        {
                            "image_path": result["image_path"],
                            "pred_label": result["pred_label"],
                            "pred_index": result["pred_index"],
                            "pred_probability": result["pred_probability"],
                            "pneumonia_probability": result["pneumonia_probability"],
                            "checkpoint_path": result["checkpoint_path"],
                            "checkpoint_epoch": result["checkpoint_epoch"],
                            "checkpoint_auc": result["checkpoint_auc"],
                        }
                    )
        return results

    def visualize(
        self,
        result: dict[str, Any],
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(
            f"MedFusionNet | Pred: {result['pred_label']} "
            f"({result['pred_probability'] * 100:.1f}%)",
            fontsize=13,
        )

        axes[0].imshow(result["image"])
        axes[0].set_title("Original")
        axes[0].axis("off")

        if result["heatmap"] is not None:
            axes[1].imshow(result["heatmap"], cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        if result["overlay"] is not None:
            axes[2].imshow(result["overlay"])
        axes[2].set_title(
            f"PNEUMONIA prob: {result['pneumonia_probability']:.3f}"
        )
        axes[2].axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def close(self) -> None:
        self.gradcam.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
