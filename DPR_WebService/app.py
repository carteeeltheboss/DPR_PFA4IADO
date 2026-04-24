"""Flask web service for MedFusionNet inference and benchmarking."""

from __future__ import annotations

import os
import secrets
import sys
import threading
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPR_MedFusionNet.config import SUPPORTED_IMAGE_EXTENSIONS
from DPR_WebService.api import api_bp
from DPR_WebService.benchmark import load_benchmark_from_disk, run_benchmark, save_benchmark_to_disk
from DPR_WebService.service import (
    TEST_ROOT,
    list_all_samples,
    list_checkpoints,
    list_featured_samples_for_seed,
    load_image,
    model_store,
    repo_relative,
    resolve_sample_path,
    result_to_view_model,
)

_IMAGE_MAGIC = {
    b"\x89PNG",
    b"\xff\xd8\xff",
}


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("DPR_WEBSERVICE_SECRET", "medfusionnet-local-dev")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.register_blueprint(api_bp)

_benchmark_cache: dict[str, dict[str, Any] | None] = {}
_benchmark_status: dict[str, str] = {}

_persisted = load_benchmark_from_disk()
if _persisted:
    _benchmark_cache["__persisted__"] = _persisted


def _is_valid_image_upload(file_storage) -> tuple[bool, str]:
    """
    Validate an uploaded FileStorage object.
    Returns (is_valid, error_message).
    """
    if not file_storage or not file_storage.filename:
        return False, "No file was selected."

    ext = Path(file_storage.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, (
            f"Unsupported file type '{ext}'. "
            f"Accepted formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    header = file_storage.stream.read(16)
    file_storage.stream.seek(0)

    if not any(header.startswith(magic) for magic in _IMAGE_MAGIC):
        return False, (
            "The uploaded file does not appear to be a valid PNG or JPEG image."
        )

    return True, ""


def current_sample_seed() -> str:
    seed = session.get("sample_seed")
    if not seed:
        seed = secrets.token_hex(8)
        session["sample_seed"] = seed
    return seed


def common_context() -> dict[str, Any]:
    """Template context shared by the main pages."""
    model_summary = model_store.current_model_summary()
    checkpoint_key = model_summary["checkpoint_path"]
    return {
        "model_summary": model_summary,
        "available_checkpoints": list_checkpoints(),
        "all_samples": list_all_samples(),
        "featured_samples": list_featured_samples_for_seed(seed=current_sample_seed()),
        "benchmark_status": _benchmark_status.get(checkpoint_key, "idle"),
    }


@app.route("/", methods=["GET"])
def index():
    context = common_context()
    context["prediction"] = None
    return render_template("index.html", **context)


@app.route("/load-model", methods=["POST"])
def load_model():
    checkpoint_path = request.form.get("checkpoint_path") or None
    try:
        model_store.load(checkpoint_path)
        flash("Model loaded successfully.", "success")
    except Exception as exc:  # pragma: no cover - exercised through runtime tests
        flash(f"Unable to load the selected checkpoint: {exc}", "error")
    return redirect(url_for("index"))


@app.route("/shuffle-samples", methods=["POST"])
def shuffle_samples():
    session["sample_seed"] = secrets.token_hex(8)
    flash("Sample gallery shuffled.", "success")
    return redirect(url_for("index"))


@app.route("/predict", methods=["POST"])
def predict():
    context = common_context()
    prediction = None
    sample_path = request.form.get("sample_path", "").strip()
    upload = request.files.get("image_file")

    try:
        engine = model_store.current_engine()
        if upload and upload.filename:
            valid, error_msg = _is_valid_image_upload(upload)
            if not valid:
                flash(error_msg, "error")
                context["prediction"] = None
                return render_template("index.html", **context)
            image = load_image(upload.stream)
            result = engine.predict(image, compute_cam=True)
            prediction = result_to_view_model(
                result,
                source_label="Uploaded image",
                source_value=upload.filename,
            )
        elif sample_path:
            resolved_sample = resolve_sample_path(sample_path)
            result = engine.predict(resolved_sample, compute_cam=True)
            prediction = result_to_view_model(
                result,
                source_label="Sample image",
                source_value=repo_relative(resolved_sample),
            )
        else:
            raise ValueError("Choose a sample image or upload one before running inference.")
    except Exception as exc:  # pragma: no cover - exercised through runtime tests
        flash(str(exc), "error")

    context["prediction"] = prediction
    return render_template("index.html", **context)


@app.route("/benchmark", methods=["GET"])
def benchmark():
    context = common_context()
    checkpoint_key = context["model_summary"]["checkpoint_path"]
    context["benchmark"] = _benchmark_cache.get(checkpoint_key) or _benchmark_cache.get("__persisted__")
    context["benchmark_status"] = _benchmark_status.get(checkpoint_key, "idle")
    return render_template("benchmark.html", **context)


@app.route("/benchmark/run", methods=["POST"])
def run_benchmark_view():
    context = common_context()
    checkpoint_key = context["model_summary"]["checkpoint_path"]
    max_images_arg = request.args.get("max_images", "").strip()
    max_images = int(max_images_arg) if max_images_arg.isdigit() else None

    if _benchmark_status.get(checkpoint_key) == "running":
        flash(
            "Benchmark is already running. Refresh in a few seconds.",
            "success",
        )
        return redirect(url_for("benchmark"))

    def _run() -> None:
        _benchmark_status[checkpoint_key] = "running"
        try:
            result = run_benchmark(max_images=max_images)
            _benchmark_cache[checkpoint_key] = result
            _benchmark_cache["__persisted__"] = {key: value for key, value in result.items() if key != "charts"}
            save_benchmark_to_disk(result)
            _benchmark_status[checkpoint_key] = "done"
        except Exception as exc:
            _benchmark_cache[checkpoint_key] = None
            _benchmark_status[checkpoint_key] = f"error: {exc}"

    threading.Thread(target=_run, daemon=True).start()
    flash(
        "Benchmark started in background. This takes ~20 seconds — refresh the page to see results."
        if max_images is None
        else f"Smoke-test benchmark started ({max_images} images). Refresh in a few seconds.",
        "success",
    )
    return redirect(url_for("benchmark"))


@app.route("/sample-image/<path:relative_path>", methods=["GET"])
def sample_image(relative_path: str):
    image_path = resolve_sample_path(relative_path)
    return send_file(image_path)


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok", "model": model_store.current_model_summary()}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
