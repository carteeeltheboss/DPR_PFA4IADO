"""Benchmark helpers for evaluating the MedFusionNet checkpoint on the test set."""

from __future__ import annotations

import json
import io
import time
import base64
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from DPR_MedFusionNet.config import CLASS_NAMES, CLASS_TO_IDX, SUPPORTED_IMAGE_EXTENSIONS
from DPR_MedFusionNet.preprocessing import load_image

from .service import TEST_ROOT, model_store

_BENCHMARK_CACHE_FILE = Path(__file__).resolve().parent / "runtime" / "benchmark_cache.json"


def save_benchmark_to_disk(result: dict[str, Any]) -> None:
    """Save benchmark result without charts to disk for persistence."""
    _BENCHMARK_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    saveable = {key: value for key, value in result.items() if key != "charts"}
    _BENCHMARK_CACHE_FILE.write_text(json.dumps(saveable, indent=2))


def load_benchmark_from_disk() -> dict[str, Any] | None:
    """Load the last benchmark result from disk if it exists."""
    if not _BENCHMARK_CACHE_FILE.exists():
        return None
    try:
        loaded = json.loads(_BENCHMARK_CACHE_FILE.read_text())
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _iter_test_samples(test_root: Path = TEST_ROOT) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name in CLASS_NAMES:
        class_dir = test_root / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                samples.append((path, CLASS_TO_IDX[class_name]))
    return samples


def _balanced_subset(samples: list[tuple[Path, int]], max_images: int) -> list[tuple[Path, int]]:
    by_class = {
        label: [sample for sample in samples if sample[1] == label]
        for label in CLASS_TO_IDX.values()
    }
    per_class = max(max_images // len(CLASS_NAMES), 1)
    subset: list[tuple[Path, int]] = []

    for label in sorted(by_class):
        subset.extend(by_class[label][:per_class])

    remaining_slots = max_images - len(subset)
    if remaining_slots > 0:
        leftovers: list[tuple[Path, int]] = []
        for label in sorted(by_class):
            leftovers.extend(by_class[label][per_class:])
        subset.extend(leftovers[:remaining_slots])

    return subset[:max_images]


def _batched(items: list[tuple[Path, int]], batch_size: int) -> list[list[tuple[Path, int]]]:
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def _figure_data_url(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _confusion_matrix_chart(matrix: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "#13243A"
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _figure_data_url(fig)


def _roc_chart(labels: np.ndarray, positive_scores: np.ndarray) -> str:
    fpr, tpr, _ = roc_curve(labels, positive_scores)
    auc = roc_auc_score(labels, positive_scores)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(fpr, tpr, color="#0E7490", linewidth=2.5, label=f"MedFusionNet (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    return _figure_data_url(fig)


def _precision_recall_chart(labels: np.ndarray, positive_scores: np.ndarray) -> str:
    precision, recall, _ = precision_recall_curve(labels, positive_scores)
    avg_precision = average_precision_score(labels, positive_scores)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(recall, precision, color="#DC2626", linewidth=2.5, label=f"AP = {avg_precision:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    return _figure_data_url(fig)


def _score_distribution_chart(labels: np.ndarray, positive_scores: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.hist(
        positive_scores[labels == 0],
        bins=20,
        alpha=0.70,
        color="#2563EB",
        label="True NORMAL",
    )
    ax.hist(
        positive_scores[labels == 1],
        bins=20,
        alpha=0.70,
        color="#EA580C",
        label="True PNEUMONIA",
    )
    ax.axvline(0.5, color="#111827", linestyle="--", linewidth=1.5, label="Decision threshold")
    ax.set_xlabel("PNEUMONIA probability")
    ax.set_ylabel("Image count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(alpha=0.2)
    return _figure_data_url(fig)


def run_benchmark(
    *,
    batch_size: int = 16,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Run a full or partial benchmark on the current checkpoint."""
    engine = model_store.current_engine()
    samples = _iter_test_samples()
    if max_images is not None:
        samples = _balanced_subset(samples, max_images)
    if not samples:
        raise RuntimeError(f"No test images found under {TEST_ROOT}")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in _batched(samples, batch_size):
            tensors = []
            batch_paths: list[Path] = []
            batch_labels: list[int] = []
            for path, label in batch:
                pil_image = load_image(path)
                tensors.append(engine.preprocessor.transform(pil_image))
                batch_paths.append(path)
                batch_labels.append(label)

            inputs = torch.stack(tensors).to(engine.device)
            logits = engine.model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)

            for path, label, prediction, probs in zip(batch_paths, batch_labels, predictions, probabilities):
                rows.append(
                    {
                        "image_path": str(path),
                        "relative_path": str(path.relative_to(TEST_ROOT)),
                        "true_label": label,
                        "true_class": CLASS_NAMES[label],
                        "pred_label": int(prediction),
                        "pred_class": CLASS_NAMES[int(prediction)],
                        "normal_probability": float(probs[0]),
                        "pneumonia_probability": float(probs[1]),
                    }
                )

    labels = np.array([row["true_label"] for row in rows], dtype=np.int64)
    predictions = np.array([row["pred_label"] for row in rows], dtype=np.int64)
    positive_scores = np.array([row["pneumonia_probability"] for row in rows], dtype=np.float64)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    auc_roc = roc_auc_score(labels, positive_scores)
    avg_precision = average_precision_score(labels, positive_scores)
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    report = classification_report(labels, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    misclassified = [
        row
        for row in rows
        if row["true_label"] != row["pred_label"]
    ]
    misclassified.sort(
        key=lambda row: max(row["normal_probability"], row["pneumonia_probability"]),
        reverse=True,
    )

    finished = time.perf_counter()
    return {
        "model_path": str(engine.checkpoint_path),
        "dataset_size": len(rows),
        "duration_seconds": finished - started,
        "metrics": {
            "accuracy": float(accuracy),
            "auc_roc": float(auc_roc),
            "avg_precision": float(avg_precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision_pneumonia": float(precision[1]),
            "recall_pneumonia": float(recall[1]),
            "f1_pneumonia": float(f1[1]),
            "false_positive_count": int(fp),
            "false_negative_count": int(fn),
            "true_positive_count": int(tp),
            "true_negative_count": int(tn),
        },
        "per_class": [
            {
                "label": class_name,
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1": float(report[class_name]["f1-score"]),
                "support": int(report[class_name]["support"]),
            }
            for class_name in CLASS_NAMES
        ],
        "misclassified": misclassified[:15],
        "charts": {
            "confusion_matrix": _confusion_matrix_chart(matrix),
            "roc_curve": _roc_chart(labels, positive_scores),
            "precision_recall": _precision_recall_chart(labels, positive_scores),
            "score_distribution": _score_distribution_chart(labels, positive_scores),
        },
    }
