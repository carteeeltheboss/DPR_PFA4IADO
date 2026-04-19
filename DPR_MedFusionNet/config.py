"""Shared configuration for the local MedFusionNet inference stack."""

from __future__ import annotations

from pathlib import Path


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_RUNS_ROOT = PACKAGE_ROOT / "runs"

MONTH_ABBR = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
