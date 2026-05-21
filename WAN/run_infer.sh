export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

CUDA_VISIBLE_DEVICES=1 python infer_control.py \
        --trained_ckpt "../checkpoints/flextraj_wan.safetensors" \
        --test_root "../" \
        --validation_file "../FlexBench/flexbench.json" \
        --output_folder outputs/ \
        --kv_cache