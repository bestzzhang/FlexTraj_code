export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

export MODEL_PATH="THUDM/CogVideoX-5b-I2V"
export CKPT_LORA="../checkpoints/flextraj_cogvideox.safetensors"

CUDA_VISIBLE_DEVICES=0 python infer_control.py \
  --validation_file ../FlexBench/flexbench.json \
  --out_folder outputs \
  --test_root '../' \
  --model_path $MODEL_PATH \
  --checkpoint_path $CKPT_LORA
