"""Flask web service for MedFusionNet inference and benchmarking."""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DPR_WebService.benchmark import run_benchmark
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


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("DPR_WEBSERVICE_SECRET", "medfusionnet-local-dev")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_benchmark_cache: dict[str, dict[str, Any]] = {}


def current_sample_seed() -> str:
    seed = session.get("sample_seed")
    if not seed:
        seed = secrets.token_hex(8)
        session["sample_seed"] = seed
    return seed


def common_context() -> dict[str, Any]:
    """Template context shared by the main pages."""
    return {
        "model_summary": model_store.current_model_summary(),
        "available_checkpoints": list_checkpoints(),
        "all_samples": list_all_samples(),
        "featured_samples": list_featured_samples_for_seed(seed=current_sample_seed()),
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
    context["benchmark"] = _benchmark_cache.get(checkpoint_key)
    return render_template("benchmark.html", **context)


@app.route("/benchmark/run", methods=["POST"])
def run_benchmark_view():
    context = common_context()
    checkpoint_key = context["model_summary"]["checkpoint_path"]
    max_images_arg = request.args.get("max_images", "").strip()
    max_images = int(max_images_arg) if max_images_arg.isdigit() else None

    try:
        benchmark_result = run_benchmark(max_images=max_images)
        _benchmark_cache[checkpoint_key] = benchmark_result
        context["benchmark"] = benchmark_result
        flash(
            "Benchmark completed successfully."
            if max_images is None
            else f"Benchmark smoke test completed on {benchmark_result['dataset_size']} images.",
            "success",
        )
    except Exception as exc:  # pragma: no cover - exercised through runtime tests
        context["benchmark"] = _benchmark_cache.get(checkpoint_key)
        flash(f"Benchmark failed: {exc}", "error")

    return render_template("benchmark.html", **context)


@app.route("/sample-image/<path:relative_path>", methods=["GET"])
def sample_image(relative_path: str):
    image_path = resolve_sample_path(relative_path)
    return send_file(image_path)


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok", "model": model_store.current_model_summary()}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
