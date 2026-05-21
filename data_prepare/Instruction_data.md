# Data Preparation

This document describes the environment setup and data preparation pipeline for real and synthetic data.

## Environment Setup

```bash
conda create -n ext_cond python=3.10
conda activate ext_cond
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e submodule/segment-anything-1
pip install -e submodule/segment-anything-2
pip install git+https://github.com/asomoza/image_gen_aux.git
pip install -r requirements.txt
```

## Real Data

### 3D Tracking

```bash
cd Spatracking
bash run.sh
```

### Segmentation

```bash
cd AutoSeg_SAM2
bash run.sh
```

## Synthetic Data

- `blender/script.py`: render scenes with Blender Python.
- `blender/render_with_random_bg.py`: batch render scenes with randomized backgrounds.

## Acknowledgements

- `AutoSeg_SAM2`: https://github.com/zrporz/AutoSeg-SAM2
- `SpaTracking`: https://github.com/henry123-boy/SpaTracker
- `blender`: https://github.com/igl-hkust/diffusionasshader