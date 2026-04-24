"""REST API blueprint for MedFusionNet local inference and benchmarking."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Blueprint, Response, jsonify, request

from DPR_MedFusionNet.config import SUPPORTED_IMAGE_EXTENSIONS
from DPR_WebService.benchmark import run_benchmark
from DPR_WebService.service import model_store


api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

_NO_IMAGE_MESSAGE = "No image provided. Send as multipart field 'image' or raw image body."
_RAW_IMAGE_EXTENSIONS = {
    "image/jpg": ".jpg",
    "image/jpeg": ".jpeg",
    "image/pjpeg": ".jpeg",
    "image/png": ".png",
    "image/x-png": ".png",
}


def _error_response(message: str, status_code: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status_code


def _content_type_extension(content_type: str | None) -> str | None:
    if not content_type:
        return None
    mime_type = content_type.split(";", 1)[0].strip().lower()
    return _RAW_IMAGE_EXTENSIONS.get(mime_type)


def _open_image_bytes(payload: bytes) -> Image.Image:
    if not payload:
        raise ValueError(_NO_IMAGE_MESSAGE)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image file. Please provide a PNG or JPEG chest X-ray.") from exc


def _extract_request_image() -> Image.Image:
    upload = request.files.get("image")
    if upload is not None and upload.filename:
        extension = Path(upload.filename).suffix.lower()
        if extension not in SUPPORTED_IMAGE_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(f"Unsupported image extension. Allowed extensions: {allowed}.")
        return _open_image_bytes(upload.read())

    content_type = request.content_type or ""
    if "image" in content_type.lower():
        extension = _content_type_extension(content_type)
        if extension not in SUPPORTED_IMAGE_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(f"Unsupported image content type. Allowed extensions: {allowed}.")
        return _open_image_bytes(request.get_data(cache=False))

    raise ValueError(_NO_IMAGE_MESSAGE)


def _ndarray_to_b64png(arr: np.ndarray) -> str:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _prediction_payload(result: dict[str, Any], inference_time_s: float) -> dict[str, Any]:
    return {
        "prediction": result["pred_label"],
        "pneumonia_probability": float(result["pneumonia_probability"]),
        "probabilities": {
            label: float(score)
            for label, score in result["probabilities"].items()
        },
        "checkpoint_epoch": result["checkpoint_epoch"],
        "checkpoint_auc": result["checkpoint_auc"],
        "inference_time_s": round(inference_time_s, 3),
    }


def _benchmark_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_path": result["model_path"],
        "dataset_size": result["dataset_size"],
        "duration_seconds": result["duration_seconds"],
        "metrics": result["metrics"],
        "per_class": result["per_class"],
    }


def _parse_max_images(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return None
    try:
        parsed = int(stripped)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


@api_bp.route("/health", methods=["GET"])
def health() -> Response | tuple[Response, int]:
    try:
        summary = model_store.current_model_summary()
    except Exception as exc:
        return _error_response(f"Model not available: {exc}", 503)

    return jsonify(
        {
            "status": "ok",
            "model": "MedFusionNet",
            "device": summary["device"],
            "checkpoint_path": summary["checkpoint_path"],
            "checkpoint_epoch": summary["checkpoint_epoch"],
            "checkpoint_auc": summary["checkpoint_auc"],
        }
    )


@api_bp.route("/predict", methods=["POST"])
def predict() -> Response | tuple[Response, int]:
    try:
        image = _extract_request_image()
    except ValueError as exc:
        return _error_response(str(exc), 400)

    try:
        started = time.perf_counter()
        engine = model_store.current_engine()
        result = engine.predict(image, compute_cam=False)
        inference_time_s = time.perf_counter() - started
    except Exception as exc:
        return _error_response(f"Inference failed: {exc}", 500)

    return jsonify(_prediction_payload(result, inference_time_s))


@api_bp.route("/predict/gradcam", methods=["POST"])
def predict_gradcam() -> Response | tuple[Response, int]:
    try:
        image = _extract_request_image()
    except ValueError as exc:
        return _error_response(str(exc), 400)

    try:
        started = time.perf_counter()
        engine = model_store.current_engine()
        result = engine.predict(image, compute_cam=True)
        inference_time_s = time.perf_counter() - started
    except Exception as exc:
        return _error_response(f"Inference failed: {exc}", 500)

    payload = _prediction_payload(result, inference_time_s)
    overlay = result.get("overlay")
    heatmap = result.get("heatmap")
    heatmap_rgb = cm.jet(heatmap)[..., :3] if isinstance(heatmap, np.ndarray) and heatmap.ndim == 2 else heatmap

    payload["gradcam_overlay_b64"] = _ndarray_to_b64png(overlay) if isinstance(overlay, np.ndarray) else None
    payload["gradcam_heatmap_b64"] = (
        _ndarray_to_b64png(np.asarray(heatmap_rgb)) if isinstance(heatmap_rgb, np.ndarray) else None
    )
    return jsonify(payload)


@api_bp.route("/benchmark", methods=["GET"])
def benchmark() -> Response | tuple[Response, int]:
    max_images = _parse_max_images(request.args.get("max_images"))

    try:
        result = run_benchmark(max_images=max_images)
    except Exception as exc:
        return _error_response(f"Benchmark failed: {exc}", 500)

    return jsonify(_benchmark_payload(result))
