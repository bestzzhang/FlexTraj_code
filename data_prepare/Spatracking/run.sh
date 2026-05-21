
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONWARNINGS="ignore"
export ROOT="../test_data"

  
CUDA_VISIBLE_DEVICES=2 python accelerate_tracking.py \
  --root $ROOT/video \
  --outdir $ROOT/tracking_reverse \
  --do_inv


CUDA_VISIBLE_DEVICES=2 python accelerate_tracking.py \
  --root $ROOT/video \
  --outdir $ROOT/tracking
