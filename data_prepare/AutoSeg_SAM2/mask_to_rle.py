import numpy as np
from PIL import Image
from first_mask_extract import masks_to_rles
from pathlib import Path
import pickle
import os

def save_masks_pkl_all(mask_list_all,save_path):
    rles_all = {}
    for i, mask in enumerate(mask_list_all):
        rles = masks_to_rles(mask)
        rles_all[i] = rles

    path = Path(save_path)
    with path.open("wb") as f:
        pickle.dump(rles_all, f, protocol=5)

def mask2rle(mask_paths, save_dir="out"):
    masks = np.stack([np.array(Image.open(mp)) for mp in mask_paths]) > 0
    mask_list_all = [masks[:,None,:,:]]
    save_masks_pkl_all(mask_list_all,save_dir)


rle_root = f"../test_dataset/cinemaster/rles"
os.makedirs(rle_root, exist_ok=True)

for name in ["drive", "jump_down", "around"]:
    mask_root = f"../test_dataset/cinemaster/first_frame_masks/{name}"
    
    mask_paths = [f"{mask_root}/1.png", f"{mask_root}/2.png"]
    mask2rle(mask_paths, f"{rle_root}/{name}.pkl")