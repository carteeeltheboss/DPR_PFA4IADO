"""main.py — MedFusionNet Manim v2 (visual, matrix-level)"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenes.scene_A_preprocessing import SceneAPreprocessing as _SceneAPreprocessing
from scenes.scene_A_preprocessing import SceneACLAHE as _SceneACLAHE
from scenes.scene_B_training_tensor import SceneBTrainingTensor as _SceneBTrainingTensor


class SceneAPreprocessing(_SceneAPreprocessing):
    pass


class SceneACLAHE(_SceneACLAHE):
    pass


class SceneBTrainingTensor(_SceneBTrainingTensor):
    pass


__all__ = ["SceneAPreprocessing", "SceneACLAHE", "SceneBTrainingTensor"]
