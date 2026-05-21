# FlexTraj CogVideoX Implementation

This directory contains the CogVideoX-based implementation of FlexTraj.

## Environment Setup

```bash
cd CogVideoX
conda create -n flextraj_cogvideox python=3.10
conda activate flextraj_cogvideox
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## CogVideoX Official Checkpoint
Download the required CogVideoX checkpoint from Hugging Face:

```bash
huggingface-cli download THUDM/CogVideoX-5b-I2V
```

## Inference

Run the inference script with:

```bash
bash run_infer.sh
```

## Credits

This implementation is built upon [CogVideo](https://github.com/zai-org/CogVideo.git).