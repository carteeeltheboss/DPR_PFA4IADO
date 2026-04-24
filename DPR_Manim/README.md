# MedFusionNet — Manim Animation Series

Fully self-contained Manim video project explaining the complete training
pipeline of **MedFusionNet** (Swin Transformer + DenseNet-121 hybrid for
pneumonia detection).

---

## Folder structure

```
manim_medfusion/
├── main.py                          ← single entrypoint, import all scenes here
├── README.md
├── utils/
│   └── styles.py                    ← shared palette, fonts, reusable mobjects
├── scenes/
│   ├── scene_01_dataset.py          ← Dataset structure, label encoding
│   ├── scene_02_preprocessing.py    ← Preprocessing transforms + augmentation
│   ├── scene_03_architecture.py     ← Two-branch architecture diagram
│   ├── scene_04_fusion.py           ← Feature concat + MLP classification head
│   ├── scene_05_training.py         ← Training loop, loss, LR curve
│   ├── scene_06_gradcam_uncertainty.py ← Grad-CAM + MC-Dropout
│   └── scene_07_evaluation.py       ← Metrics, confusion matrix, full recap
└── assets/
    ├── sample_xray_normal.jpeg      ← ← REPLACE: put a real NORMAL X-ray here
    ├── sample_xray_pneumonia.jpeg   ← ← REPLACE: put a real PNEUMONIA X-ray here
    ├── gradcam_overlay.png          ← ← REPLACE: use gradcam_results.png from DPR_tex/
    └── benchmark_training.png       ← ← REPLACE: use benchmark_training.png from DPR_tex/
```

---

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install Manim Community Edition
pip install manim

# 3. On Linux you also need:
sudo apt-get install -y libcairo2-dev libpango1.0-dev ffmpeg texlive-full

# 4. On macOS:
brew install cairo pango ffmpeg
pip install manim
```

---

## Render a single scene

```bash
# Low quality (fast preview, 480p)
manim -pql main.py Scene01Dataset

# Medium quality (720p)
manim -pqm main.py Scene02Preprocessing

# High quality (1080p, for presentation)
manim -pqh main.py Scene03Architecture

# 4K (for export / demo)
manim -pqk main.py Scene04Fusion
```

Available scene names:
| Class name              | Content                              | Est. duration |
|-------------------------|--------------------------------------|--------------|
| `Scene01Dataset`        | Dataset structure, label encoding    | ~60 s        |
| `Scene02Preprocessing`  | Transforms, augmentation             | ~70 s        |
| `Scene03Architecture`   | Two-branch architecture diagram      | ~90 s        |
| `Scene04Fusion`         | Concat + MLP head + threshold        | ~70 s        |
| `Scene05Training`       | Training loop, loss, LR curve        | ~90 s        |
| `Scene06GradCAM`        | Grad-CAM + MC-Dropout                | ~80 s        |
| `Scene07Evaluation`     | Confusion matrix + full recap        | ~80 s        |

---

## Render ALL scenes in one go

### Linux / macOS
```bash
for scene in Scene01Dataset Scene02Preprocessing Scene03Architecture \
    Scene04Fusion Scene05Training Scene06GradCAM Scene07Evaluation; do
    manim -qh main.py $scene
done
```

### Windows (PowerShell)
```powershell
$scenes = @("Scene01Dataset","Scene02Preprocessing","Scene03Architecture",
            "Scene04Fusion","Scene05Training","Scene06GradCAM","Scene07Evaluation")
foreach ($s in $scenes) { manim -qh main.py $s }
```

Rendered `.mp4` files appear in `media/videos/main/1080p60/`.

---

## Export thumbnail (last frame only)

```bash
manim --save_last_frame main.py Scene07Evaluation
```

Output: `media/images/main/Scene07Evaluation_ManimCE_v0.xx.x.png`

---

## Adding real assets (recommended upgrades)

### 1. Real chest X-ray images
```python
# In any scene, replace the placeholder VGroup with:
xray = ImageMobject("assets/sample_xray_pneumonia.jpeg")
xray.set_height(2.6)
xray.move_to(LEFT * 4.5)
self.play(FadeIn(xray))
```
Source images: `DPR_MedFusionNet/data/test/PNEUMONIA/person*.jpeg`

### 2. Real Grad-CAM output
```python
# Replace make_heatmap_overlay() with:
overlay = ImageMobject("assets/gradcam_overlay.png")
overlay.set_height(2.6)
```
Use the file already in your repo: `DPR_tex/gradcam_results.png`

### 3. Real training curves
Export your `metrics.csv` from a training run and load it:
```python
import csv
epochs, train_loss, val_auc = [], [], []
with open("assets/metrics.csv") as f:
    for row in csv.DictReader(f):
        epochs.append(float(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_auc.append(float(row["val_auc"]))

# Then use ax.plot_line_graph(epochs, train_loss) in Scene05Training
```

### 4. Real benchmark plots
```python
# In Scene07Evaluation:
bench = ImageMobject("assets/benchmark_training.png")
bench.set_height(3.5)
self.play(FadeIn(bench))
```
Use: `DPR_tex/benchmark_training.png` and `DPR_tex/benchmark_testset.png`

### 5. Add voiceover (Manim-Voiceover plugin)
```bash
pip install manim-voiceover
```
```python
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class Scene01Dataset(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        with self.voiceover("The dataset contains 5863 chest X-rays..."):
            self.play(FadeIn(title))
```

### 6. Export short clips for presentation slides
```bash
# Trim a clip (ffmpeg):
ffmpeg -i media/videos/main/1080p60/Scene03Architecture.mp4 \
       -ss 00:00:10 -t 00:00:20 -c copy architecture_clip.mp4
```

---

## Storyboard summary

| Scene | Title                  | Key visuals                                               | Duration |
|-------|------------------------|-----------------------------------------------------------|----------|
| 01    | Dataset Preparation    | Folder tree, class counts, imbalance bar, label encoding  | 60 s     |
| 02    | Preprocessing          | Transform pipeline, normalization formula, augmentations  | 70 s     |
| 03    | Architecture           | DenseNet chain + Swin stages + concat + MLP               | 90 s     |
| 04    | Feature Fusion         | Feature vector strips, concat, MLP layers, prob bars      | 70 s     |
| 05    | Training Loop          | Hyperparams table, loop cycle, CE+LS formula, loss curve  | 90 s     |
| 06    | Grad-CAM + Uncertainty | 3-panel overlay, CAM formula, MC-Dropout prob bars        | 80 s     |
| 07    | Evaluation + Recap     | Confusion matrix, metric cards, full pipeline grid        | 80 s     |

**Total runtime: ~8.5 minutes**

---

## Design decisions

| Choice | Reason |
|--------|--------|
| Dark background `#0D1117` | Matches GitHub dark theme; professional for jury/demo |
| Monospace font everywhere | Code-style legibility; consistent with technical content |
| Teal = DenseNet, Purple = Swin | Persistent colour coding across all scenes |
| Real test results in Scene07 | AUC=0.98, 93% accuracy, 13 FN/33 FP from actual run |
| Placeholder X-rays as VGroups | Works without assets; easy to swap for real images |
| One Scene class per phase | Run/test/iterate any phase independently |
