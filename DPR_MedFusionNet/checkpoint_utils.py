"""Checkpoint discovery and loading helpers for local MedFusionNet inference."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import torch

try:
    from .config import DEFAULT_RUNS_ROOT, MONTH_ABBR, PROJECT_ROOT
except ImportError:  # pragma: no cover - script execution fallback
    from config import DEFAULT_RUNS_ROOT, MONTH_ABBR, PROJECT_ROOT


def torch_load_compat(path: str | Path, map_location=None):
    """Load a checkpoint across torch versions that may or may not expose ``weights_only``."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def format_model_date(day: date | None = None) -> str:
    """Return notebook-style checkpoint suffixes such as ``19Apr26``."""
    day = day or date.today()
    return f"{day.day:02d}{MONTH_ABBR[day.month]}{day.year % 100:02d}"


def _checkpoint_candidates(runs_root: Path) -> list[Path]:
    return sorted(runs_root.glob("**/checkpoints/*.pth"))


def _prefer_by_name(paths: Iterable[Path], filename: str) -> Path | None:
    matches = [path for path in paths if path.name == filename]
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    today: date | None = None,
) -> Path:
    """
    Resolve the checkpoint to use.

    Priority:
    1. explicit path
    2. ``Model_<today>.pth``
    3. latest ``Model_*.pth``
    4. latest ``best.pth``
    5. latest ``last.pth``
    """
    if checkpoint and str(checkpoint).lower() not in {"auto", "default"}:
        candidate = Path(checkpoint).expanduser()
        if not candidate.is_absolute():
            direct = candidate.resolve()
            repo_relative = (PROJECT_ROOT / candidate).resolve()
            if direct.exists():
                return direct
            if repo_relative.exists():
                return repo_relative
        elif candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    runs_root = Path(runs_root).expanduser().resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")

    candidates = _checkpoint_candidates(runs_root)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under: {runs_root}")

    today_name = f"Model_{format_model_date(today)}.pth"
    chosen = _prefer_by_name(candidates, today_name)
    if chosen is not None:
        return chosen

    model_candidates = [path for path in candidates if path.name.startswith("Model_")]
    if model_candidates:
        return max(model_candidates, key=lambda path: (path.stat().st_mtime, str(path)))

    for fallback in ("best.pth", "last.pth"):
        chosen = _prefer_by_name(candidates, fallback)
        if chosen is not None:
            return chosen

    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def extract_state_dict(checkpoint_obj) -> dict[str, torch.Tensor]:
    """Normalize checkpoint formats to a bare state dict."""
    if isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj.get("model") or checkpoint_obj.get("model_state") or checkpoint_obj
    else:
        state_dict = checkpoint_obj

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dict.")

    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer the classifier output dimension from the saved fusion head."""
    for key in ("fusion.8.weight", "fusion.8.bias"):
        if key in state_dict:
            tensor = state_dict[key]
            return int(tensor.shape[0])
    raise KeyError("Unable to infer num_classes from checkpoint state dict.")
