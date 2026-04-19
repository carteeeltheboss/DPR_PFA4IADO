# MedFusionNet Local Runtime

This repository is now split around the current notebook model in
`MedFusionNet_v4.ipynb`.

## What lives where

- `MedFusionNet_v4.ipynb`
  Source notebook used for the current Colab training workflow.
- `DPR_MedFusionNet/`
  Local inference/runtime package rebuilt from the notebook so the model can be
  imported and used from Python scripts or a future web interface.
- `DPR_MedFusionNet/runs/run_v1/checkpoints/`
  Saved checkpoints. Local inference auto-selects `Model_<today>.pth` first,
  which currently resolves to `Model_19Apr26.pth`.

## Local package contents

`DPR_MedFusionNet/` now contains the inference-only split of the notebook:

- `config.py` for shared constants and paths
- `model.py` for the Swin-Tiny + DenseNet-121 fusion architecture
- `preprocessing.py` for notebook-compatible image transforms
- `checkpoint_utils.py` for checkpoint discovery and loading
- `inference.py` for prediction and Grad-CAM
- `predict.py` for the CLI entry point

Local training scripts that no longer matched the notebook model were removed on
purpose. Training remains in Colab through `MedFusionNet_v4.ipynb`.

## Setup

```bash
cd DPR_MedFusionNet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Single-image inference

From inside `DPR_MedFusionNet/`:

```bash
python predict.py \
  --image data/test/NORMAL/IM-0001-0001.jpeg \
  --save_vis runs/run_v1/outputs/example_gradcam.png
```

This automatically uses
`DPR_MedFusionNet/runs/run_v1/checkpoints/Model_19Apr26.pth` unless you pass
`--checkpoint`.

## Python usage

From the repository root:

```python
from DPR_MedFusionNet.inference import MedFusionInference

engine = MedFusionInference()
result = engine.predict("DPR_MedFusionNet/data/test/PNEUMONIA/person147_bacteria_706.jpeg")
print(result["pred_label"], result["pneumonia_probability"])
```

## Verified locally

The current local split was smoke-tested against the saved checkpoint:

- `PNEUMONIA` sample predicted as `PNEUMONIA`
- `NORMAL` sample predicted as `NORMAL`
- Grad-CAM visualization export works
