"""
main.py  —  MedFusionNet Manim Video Series
============================================
Import ALL scenes so Manim can find them by class name.

Usage (render one scene):
    manim -pqh main.py Scene01Dataset
    manim -pqh main.py Scene02Preprocessing
    manim -pqh main.py Scene03Architecture
    manim -pqh main.py Scene04Fusion
    manim -pqh main.py Scene05Training
    manim -pqh main.py Scene06GradCAM
    manim -pqh main.py Scene07Evaluation

Usage (render ALL scenes in one command):
    for scene in Scene01Dataset Scene02Preprocessing Scene03Architecture \
        Scene04Fusion Scene05Training Scene06GradCAM Scene07Evaluation; do
        manim -pqh main.py $scene
    done

Flags:
    -p   preview after rendering
    -q   quality: l=low(480p)  m=medium(720p)  h=high(1080p)  k=4K
    --save_last_frame   export only the final frame as PNG (useful for thumbnails)
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Manim CLI only discovers scene classes defined in this module.
from scenes.scene_01_dataset import Scene01Dataset as _Scene01Dataset
from scenes.scene_02_preprocessing import Scene02Preprocessing as _Scene02Preprocessing
from scenes.scene_03_architecture import Scene03Architecture as _Scene03Architecture
from scenes.scene_04_fusion import Scene04Fusion as _Scene04Fusion
from scenes.scene_05_training import Scene05Training as _Scene05Training
from scenes.scene_06_gradcam_uncertainty import Scene06GradCAM as _Scene06GradCAM
from scenes.scene_07_evaluation import Scene07Evaluation as _Scene07Evaluation


class Scene01Dataset(_Scene01Dataset):
    pass


class Scene02Preprocessing(_Scene02Preprocessing):
    pass


class Scene03Architecture(_Scene03Architecture):
    pass


class Scene04Fusion(_Scene04Fusion):
    pass


class Scene05Training(_Scene05Training):
    pass


class Scene06GradCAM(_Scene06GradCAM):
    pass


class Scene07Evaluation(_Scene07Evaluation):
    pass

__all__ = [
    "Scene01Dataset",
    "Scene02Preprocessing",
    "Scene03Architecture",
    "Scene04Fusion",
    "Scene05Training",
    "Scene06GradCAM",
    "Scene07Evaluation",
]
