"""CLI for local MedFusionNet inference."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import SUPPORTED_IMAGE_EXTENSIONS
    from .inference import MedFusionInference
except ImportError:  # pragma: no cover - script execution fallback
    from config import SUPPORTED_IMAGE_EXTENSIONS
    from inference import MedFusionInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedFusionNet inference on one image or a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto",
        help=(
            "Checkpoint path or 'auto' to prefer Model_<today>.pth under "
            "DPR_MedFusionNet/runs/**/checkpoints."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps")
    parser.add_argument("--img_size", type=int, default=224, help="Inference image size")
    parser.add_argument("--no_cam", action="store_true", help="Skip Grad-CAM generation")
    parser.add_argument("--save_vis", type=str, default=None, help="Save visualization for single-image mode")
    parser.add_argument("--output_csv", type=str, default=None, help="Save CSV results for directory mode")

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", type=str, help="Path to a single image")
    target.add_argument("--input_dir", type=str, help="Path to a directory of images")
    return parser.parse_args()


def list_images(input_dir: Path) -> list[str]:
    images = [
        str(path)
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")
    return images


def print_single_result(result: dict) -> None:
    print("=" * 72)
    print(f"Checkpoint           : {result['checkpoint_path']}")
    print(f"Image                : {result['image_path']}")
    print(f"Prediction           : {result['pred_label']}")
    print(f"Predicted class prob : {result['pred_probability']:.4f}")
    print(f"Pneumonia prob       : {result['pneumonia_probability']:.4f}")
    probs = ", ".join(f"{name}={prob:.4f}" for name, prob in result["probabilities"].items())
    print(f"Class probabilities  : {probs}")
    if result["checkpoint_epoch"] is not None:
        print(f"Checkpoint epoch     : {result['checkpoint_epoch']}")
    if result["checkpoint_auc"] is not None:
        print(f"Checkpoint AUC       : {result['checkpoint_auc']:.4f}")


def main() -> None:
    args = parse_args()
    engine = MedFusionInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size,
    )

    if args.image:
        result = engine.predict(args.image, compute_cam=not args.no_cam)
        print_single_result(result)
        if args.save_vis:
            engine.visualize(result, save_path=args.save_vis, show=False)
    else:
        images = list_images(Path(args.input_dir))
        results = engine.predict_batch(
            images,
            output_csv=args.output_csv,
            compute_cam=not args.no_cam,
        )
        print("=" * 72)
        print(f"Checkpoint           : {engine.checkpoint_path}")
        print(f"Images processed     : {len(results)}")
        if args.output_csv:
            print(f"CSV saved            : {args.output_csv}")


if __name__ == "__main__":
    main()
