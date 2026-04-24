"""Standalone CLI client for the MedFusionNet REST API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import requests
except ImportError:  # pragma: no cover - optional runtime dependency
    requests = None


JsonDict = dict[str, Any]


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="http://127.0.0.1:8000", help="API host URL.")
    common.add_argument("--json", action="store_true", help="Print raw JSON responses.")

    parser = argparse.ArgumentParser(description="CLI client for the MedFusionNet Flask API.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", parents=[common], help="Check API health.")

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Run inference on one image.")
    predict_parser.add_argument("image_path", help="Path to the input image.")
    predict_parser.add_argument("--gradcam", action="store_true", help="Request Grad-CAM outputs.")
    predict_parser.add_argument("--save", help="Save Grad-CAM overlay PNG to this path.")

    benchmark_parser = subparsers.add_parser("benchmark", parents=[common], help="Run the benchmark endpoint.")
    benchmark_parser.add_argument("--max_images", type=int, help="Optional benchmark subset size.")

    return parser


def _append_query_params(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    filtered = {
        key: value
        for key, value in params.items()
        if value is not None
    }
    if not filtered:
        return url
    query = urllib.parse.urlencode(filtered)
    separator = "&" if urllib.parse.urlparse(url).query else "?"
    return f"{url}{separator}{query}"


def _multipart_payload(field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----MedFusionNetBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    lines = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode("utf-8"),
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


def _request_json_with_requests(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None,
    file_path: Path | None,
    timeout: float,
) -> JsonDict | None:
    request_kwargs: dict[str, Any] = {
        "method": method,
        "url": url,
        "params": params,
        "timeout": timeout,
    }

    file_handle = None
    try:
        if file_path is not None:
            mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            file_handle = file_path.open("rb")
            request_kwargs["files"] = {"image": (file_path.name, file_handle, mime_type)}

        response = requests.request(**request_kwargs)
    except Exception as exc:  # pragma: no cover - depends on runtime transport
        _print_error(f"Request failed (HTTP status: n/a): {exc}")
        return None
    finally:
        if file_handle is not None:
            file_handle.close()

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if not response.ok:
        detail = payload.get("error") if isinstance(payload, dict) else response.text.strip() or response.reason
        _print_error(f"Request failed (HTTP {response.status_code}): {detail}")
        return None

    if not isinstance(payload, dict):
        _print_error(f"Request failed (HTTP {response.status_code}): invalid JSON response")
        return None
    return payload


def _request_json_with_urllib(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None,
    file_path: Path | None,
    timeout: float,
) -> JsonDict | None:
    final_url = _append_query_params(url, params)
    headers = {"Accept": "application/json"}
    body: bytes | None = None

    if file_path is not None:
        body, content_type = _multipart_payload("image", file_path)
        headers["Content-Type"] = content_type

    request_obj = urllib.request.Request(final_url, data=body, method=method, headers=headers)

    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            status = response.status
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_error)
            detail = payload.get("error", raw_error) if isinstance(payload, dict) else raw_error
        except json.JSONDecodeError:
            detail = raw_error or exc.reason
        _print_error(f"Request failed (HTTP {exc.code}): {detail}")
        return None
    except urllib.error.URLError as exc:
        _print_error(f"Request failed (HTTP status: n/a): {exc.reason}")
        return None
    except Exception as exc:  # pragma: no cover - defensive fallback
        _print_error(f"Request failed (HTTP status: n/a): {exc}")
        return None

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        _print_error(f"Request failed (HTTP {status}): invalid JSON response")
        return None

    if not isinstance(payload, dict):
        _print_error(f"Request failed (HTTP {status}): invalid JSON response")
        return None
    return payload


def _request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    file_path: Path | None = None,
    timeout: float = 30.0,
) -> JsonDict | None:
    if requests is not None:
        return _request_json_with_requests(method, url, params=params, file_path=file_path, timeout=timeout)
    return _request_json_with_urllib(method, url, params=params, file_path=file_path, timeout=timeout)


def _format_float(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _print_kv(label: str, value: str) -> None:
    print(f"  {label:<15}: {value}")


def _save_base64_png(payload: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(payload)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.save(destination, format="PNG")


def _heatmap_path(overlay_path: Path) -> Path:
    if overlay_path.suffix:
        return overlay_path.with_name(f"{overlay_path.stem}_heatmap{overlay_path.suffix}")
    return overlay_path.with_name(f"{overlay_path.name}_heatmap.png")


def _run_health(args: argparse.Namespace) -> int:
    payload = _request_json("GET", f"{args.host.rstrip('/')}/api/v1/health", timeout=15.0)
    if payload is None:
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"MedFusionNet API — {args.host}")
    _print_kv("Status", str(payload.get("status", "unknown")))
    _print_kv("Device", str(payload.get("device", "unknown")))
    _print_kv("Checkpoint", str(payload.get("checkpoint_path", "unknown")))
    _print_kv("Epoch", str(payload.get("checkpoint_epoch", "n/a")))
    _print_kv("AUC", _format_float(payload.get("checkpoint_auc"), 4))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    image_path = Path(args.image_path)
    if not image_path.exists():
        _print_error(f"Image not found: {image_path}")
        return 1

    endpoint = "/api/v1/predict/gradcam" if args.gradcam else "/api/v1/predict"
    payload = _request_json(
        "POST",
        f"{args.host.rstrip('/')}{endpoint}",
        file_path=image_path,
        timeout=300.0,
    )
    if payload is None:
        return 1

    overlay_path: Path | None = None
    heatmap_path: Path | None = None
    if args.save:
        if not args.gradcam:
            _print_error("--save requires --gradcam.")
            return 1
        overlay_payload = payload.get("gradcam_overlay_b64")
        heatmap_payload = payload.get("gradcam_heatmap_b64")
        if not isinstance(overlay_payload, str) or not isinstance(heatmap_payload, str):
            _print_error("The API response did not include Grad-CAM images.")
            return 1
        overlay_path = Path(args.save)
        heatmap_path = _heatmap_path(overlay_path)
        _save_base64_png(overlay_payload, overlay_path)
        _save_base64_png(heatmap_payload, heatmap_path)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    probabilities = payload.get("probabilities", {})
    normal_prob = probabilities.get("NORMAL")
    pneumonia_prob = probabilities.get("PNEUMONIA")

    _print_kv("Image", image_path.name)
    _print_kv("Prediction", str(payload.get("prediction", "unknown")))
    _print_kv("Pneumonia prob", _format_float(payload.get("pneumonia_probability"), 4))
    _print_kv("NORMAL prob", _format_float(normal_prob, 4))
    _print_kv("PNEUMONIA prob", _format_float(pneumonia_prob, 4))
    if overlay_path is not None and heatmap_path is not None:
        _print_kv("GradCAM overlay", f"saved to {overlay_path}")
        _print_kv("GradCAM heatmap", f"saved to {heatmap_path}")
    _print_kv("Inference time", f"{float(payload.get('inference_time_s', 0.0)):.3f} s")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    params = {"max_images": args.max_images} if args.max_images is not None else None
    payload = _request_json(
        "GET",
        f"{args.host.rstrip('/')}/api/v1/benchmark",
        params=params,
        timeout=1800.0,
    )
    if payload is None:
        return 1
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    metrics = payload.get("metrics", {})
    print(
        f"Benchmark — {payload.get('dataset_size', 'unknown')} images "
        f"in {float(payload.get('duration_seconds', 0.0)):.1f} s"
    )
    _print_kv("Accuracy", f"{float(metrics.get('accuracy', 0.0)) * 100:.2f}%")
    _print_kv("AUC-ROC", _format_float(metrics.get("auc_roc"), 4))
    _print_kv("Sensitivity", f"{float(metrics.get('sensitivity', 0.0)) * 100:.2f}%")
    _print_kv("Specificity", f"{float(metrics.get('specificity', 0.0)) * 100:.2f}%")
    _print_kv("F1 (Pneumonia)", _format_float(metrics.get("f1_pneumonia"), 4))
    _print_kv("False negatives", f"{int(metrics.get('false_negative_count', 0))}   (missed pneumonia)")
    _print_kv("False positives", f"{int(metrics.get('false_positive_count', 0))}   (false alarm)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        return _run_health(args)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "benchmark":
        return _run_benchmark(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
