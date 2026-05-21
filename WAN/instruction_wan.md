# FlexTraj WAN Implementation

This directory contains the WAN-based implementation of FlexTraj.

## Environment Setup

```bash
cd WAN/
conda create -n flextraj_wan python=3.10
conda activate flextraj_wan
pip install -e .
pip install -r requirements.txt
```

## WAN Official Checkpoints

Download the required WAN checkpoints from Hugging Face:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir models/Wan-AI/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Wan-AI/Wan2.1-T2V-1.3B
```

## Inference

Run the inference script with:

```bash
bash run_infer.sh
```

## Credits

This implementation is built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio.git).