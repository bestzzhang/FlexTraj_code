

export root="../test_data"
CUDA_VISIBLE_DEVICES=2 python auto-mask-extract.py \
         --data_folder ${root}/video \
         --out_folder ${root}/segs  \
         --part 1/1
