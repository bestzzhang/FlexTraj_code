import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import argparse
from loguru import logger
import yaml
from types import SimpleNamespace
import shutil
from pathlib import Path
import pickle
from pycocotools import mask as coco_mask

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.is_available():
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator




def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou

            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    # import ipdb; ipdb.set_trace()
    return selected_idx

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask(mask,frame_idx,save_dir):
    image_array = (mask * 255).astype(np.uint8)
    # 创建图像对象
    image = Image.fromarray(image_array[0])

    # 保存图像
    image.save(os.path.join(save_dir,f'{frame_idx:03}.png'))

def save_masks(mask_list,frame_idx,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    if len(mask_list[0].shape) == 3:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))
    else:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir,f"mask_{frame_idx:03}.png"))

def save_masks_npy(mask_list,frame_idx,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    np.save(os.path.join(save_dir,f"mask_{frame_idx:03}.npy"),mask_list)
    
def search_new_obj(masks_from_prev, mask_list,mask_ratio_thresh=0,ratio=0.5, area_threash = 5000):
    new_mask_list = []

    # 计算mask_none，表示不包含在任何一个之前的mask中的区域
    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask['segmentation']
        if (mask_none & seg).sum()/seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)
    
    for mask in new_mask_list:
        mask_none &= ~mask['segmentation']

    logger.info(len(new_mask_list))
    logger.info("now ratio:",mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) )
    logger.info("expected ratios:",mask_ratio_thresh)
    logger.info(len(new_mask_list))

    return new_mask_list

def cal_no_mask_area_ratio(out_mask_list):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return(mask_none.sum() / (h * w))


class Prompts:
    def __init__(self,bs:int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list = []
        self.key_frame_list = []
        self.key_frame_obj_begin_list = []

    def add(self,obj_id,frame_id,mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id,mask))
        if frame_id not in self.key_frame_list and new_obj:
            # import ipdb; ipdb.set_trace()
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            logger.info("key_frame_obj_begin_list:",self.key_frame_obj_begin_list)
    
    def get_obj_num(self):
        return len(self.obj_list)
    
    def __len__(self):
        if self.obj_list % self.batch_size == 0:
            return len(self.obj_list) // self.batch_size
        else:
            return len(self.obj_list) // self.batch_size +1
    
    def __iter__(self):
        # self.batch_index = 0
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list)-1:
                end_idx = min(self.start_idx+self.batch_size, len(self.obj_list))
            else:
                if self.start_idx+self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex+1]:
                    end_idx = self.start_idx+self.batch_size
                else:
                    end_idx =  self.key_frame_obj_begin_list[self.iter_frameindex+1]
                    self.iter_frameindex+=1
                # end_idx = min(self.start_idx+self.batch_size, self.key_frame_obj_begin_list[self.iter_frameindex+1])
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        # if self.batch_index * self.batch_size < len(self.obj_list):
        #     start_idx = self.batch_index * self.batch_size
        #     end_idx = min(start_idx + self.batch_size, len(self.obj_list))
        #     batch_keys = self.obj_list[start_idx:end_idx]
        #     batch_prompts = {key: self.prompts[key] for key in batch_keys}
        #     self.batch_index += 1
        #     return batch_prompts
        else:
            raise StopIteration
        
def get_video_segments(prompts_loader,predictor,inference_state,step,start_frame_idx,final_output=False):
    video_segments = {}
    for batch_prompts in tqdm(prompts_loader,desc="processing prompts\n"):
        predictor.reset_state(inference_state)
        for id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                # import ipdb; ipdb.set_trace()
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt[0],
                    obj_id=id,
                    mask=prompt[1]
                )
        # start_frame_idx = 0 if final_output else None
        if not final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = { }
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
        
        else:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = { }
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,reverse=True):
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id]= (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments

def load_sam():
    ##### load Sam2 and Sam1 Model #####
    sam2_checkpoint = "../../checkpoints/sam2/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    sam_ckpt_path="../../checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=args.pred_iou_thresh, 
        box_nms_thresh=args.box_nms_thresh, 
        stability_score_thresh=args.stability_score_thresh, 
        # crop_n_layers=1,
        min_mask_region_area=100,
    )
    return predictor, mask_generator



def load_frames(video_path_or_dir):
    if video_path_or_dir.endswith(".mp4"):
        # Extract video name without extension
        video_name = os.path.splitext(os.path.basename(video_path_or_dir))[0]
        tmp_dir = os.path.join("tmp", video_name)
        os.makedirs(tmp_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path_or_dir)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmp_dir, f"{frame_idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()
        video_dir = tmp_dir
        is_tmp_dir = True
    else:
        video_dir = video_path_or_dir
        is_tmp_dir = False

    # Read frames from directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return frame_names, video_dir, is_tmp_dir

def cleanup_tmp_dir(tmp_dir):
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

def generate_and_filter_masks(image, postnms=True, level='large'):
    import time
    start_time = time.time()
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    print(f"Mask extract elapsed time: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    if postnms:
        masks_default, masks_s, masks_m, masks_l = \
            masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    if level == 'default':
        masks = [mask for mask in masks_default]
    elif level == 'small':
        masks = [mask for mask in masks_s]
    elif level == 'middle':
        masks = [mask for mask in masks_m]
    elif level == 'large':
        masks = [mask for mask in masks_l]
    else:
        raise NotImplementedError
    if len(masks) == 0:
        masks = [mask for mask in masks_default]
    print(f"Mask update elapsed time: {time.time() - start_time:.2f} seconds")
    return masks


def plot_masks2(img, video_segments, save_dir, out_frame_idx):
    os.makedirs(save_dir,exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"frame {out_frame_idx}")
    ax.imshow(img)

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, ax, obj_id=out_obj_id,random_color=False)
    
    plt.savefig(os.path.join(save_dir, f"frame_{out_frame_idx}.png"))
    plt.close(fig)

def plot_masks1(masks, save_dir, out_frame_idx):
    os.makedirs(save_dir,exist_ok=True)
    masks = np.array(masks)

    # Generate a random color for each mask
    num_masks = masks.shape[0]
    colors = [tuple(np.random.rand(3)) for _ in range(num_masks)]

    # Create an empty RGB image
    height, width = masks.shape[1:]
    color_mask_image = np.zeros((height, width, 3), dtype=np.float32)

    # Apply each mask with its color
    for i, mask in enumerate(masks):
        for c in range(3):
            color_mask_image[:, :, c] += (mask > 0) * colors[i][c]

    # Normalize the image to make sure pixel values are in [0, 1]
    color_mask_image = np.clip(color_mask_image, 0, 1)

    # Plot the image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(color_mask_image)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"frame_{out_frame_idx}.png"))
    plt.close(fig)
    

def visualize_seg_masks(npy_dir_or_list, image_dir, video_path):
    import imageio
    if isinstance(npy_dir_or_list, str):
        npy_name_list = sorted(os.listdir(npy_dir_or_list))
        npy_list = [np.load(os.path.join(npy_dir_or_list, name)) for name in npy_name_list]
    else:
        npy_list = npy_dir_or_list

    image_name_list = sorted(os.listdir(image_dir))
    image_list = [Image.open(os.path.join(image_dir, name)) for name in image_name_list]
    
    num_masks = max(len(masks) for masks in npy_list)
    colors = [tuple((np.random.rand(3) * 255).astype(np.uint8)) for _ in range(num_masks)]

    video_frames = []
    for frame_id, (masks, image) in tqdm(enumerate(zip(npy_list, image_list)), total=len(npy_list)):
        image_np = np.array(image)
        mask_combined = np.zeros_like(image_np, dtype=np.uint8)
        # Overlay each mask with its corresponding random color
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            # Ensure the mask is binary (0 or 1)
            mask_binary = (mask[0] > 0).astype(np.uint8)
            for j in range(3):  # for each channel in RGB
                mask_combined[:, :, j] += mask_binary * color[j]
        mask_combined = np.clip(mask_combined, 0, 255)
        # Blend the original image with the colored mask
        blended_image = cv2.addWeighted(image_np, 0.3, mask_combined, 0.7, 0)
        blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        video_frames.append(blended_image)
    # Create video from generated images
    imageio.mimwrite(video_path, video_frames, fps=15)
    print(f"Video saved at {video_path}")

def masks_to_rles(masks):
    if masks.dtype != np.bool_:
        raise TypeError("masks 必须是 bool 类型")
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError("masks 形状应为 (n, 1, H, W)")

    n, _, H, W = masks.shape
    rles = []
    for i in range(n):
        # RLE 要求列主序（Fortran）内存布局
        mask_i = np.asfortranarray(masks[i, 0].astype(np.uint8))
        rle = coco_mask.encode(mask_i)      # 返回 dict，counts 为 bytes
        rles.append(rle)

    return rles

def save_masks_pkl(mask,frame_idx,save_dir, protocol=5):
    os.makedirs(save_dir,exist_ok=True)
    
    rles = masks_to_rles(mask)
    path = Path(os.path.join(save_dir,f"mask_{frame_idx:03}.pkl"))
    with path.open("wb") as f:
        pickle.dump(rles, f, protocol=protocol)


def save_masks_pkl_all(mask_list_all,save_dir):
    os.makedirs(save_dir,exist_ok=True)

    rles_all = {}
    for i, mask in enumerate(mask_list_all):
        rles = masks_to_rles(mask)
        rles_all[i] = rles

    path = Path(os.path.join(save_dir,f"all.pkl"))
    with path.open("wb") as f:
        pickle.dump(rles_all, f, protocol=5)

def get_splits(video_paths, parts):
    import random

    random.seed(42)
    random.shuffle(video_paths)
    a, b = map(int, parts.split("/"))

    length = len(video_paths)
    split_size = length // b
    arrs = [video_paths[i * split_size:(i + 1) * split_size] for i in range(b)]
    current_inputs = arrs[a-1]
    print(f"Processing {a}/{b} of all times; Total Size {len(current_inputs)}/{length}")
    return current_inputs
# visualize_seg_masks(out_mask_list, video_dir, save_vid_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder",type=str,required=True)
    parser.add_argument("--out_folder",type=str,default="output")
    parser.add_argument("--part", type=str)
    args = parser.parse_args()

    # load extract config
    with open("extract_configs.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    args = SimpleNamespace(**vars(args), **config_dict)

    # load models
    predictor, mask_generator = load_sam()

    data_folder = args.data_folder
    video_paths = get_splits(sorted(os.listdir(data_folder)), args.part)
    for video_path in video_paths:
        try:
            base_dir = os.path.join(args.out_folder, video_path.split('.')[0])
            if os.path.isfile(os.path.join(base_dir, args.level, "seg.mp4")):
                print("Skipping existing output directory:", video_path.split('.')[0])
                continue
            else:
                print("Start processing:", video_path.split('.')[0])

            video_path = os.path.join(data_folder, video_path)
            
            # init logger
            logger.add(os.path.join(base_dir,f'{args.level}.log'), rotation="500 MB")
            logger.info(args)

            # load video frames
            frame_names, video_dir, is_tmp_dir = load_frames(video_path)
            inference_state = predictor.init_state(video_path=video_dir)
            prompts_loader = Prompts(bs=args.batch_size)  # hold all the clicks we add for visualization

            now_frame = 0 # current frame index
            sum_id = 0 # record the number of objects
            masks_from_prev = []  # tracked mask from previous keyframe
            is_key_frame = True # whether use sam
            
            # iterate tracking
            for now_frame in range(0, len(frame_names), args.detect_stride):
                logger.info(f"frame: {now_frame}")
                logger.info(f"is_key_frame: {is_key_frame}")

                # Run detection if it is a key frame
                if is_key_frame:
                    sum_id = prompts_loader.get_obj_num()
                    image_path = os.path.join(video_dir,frame_names[now_frame])
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    masks = generate_and_filter_masks(image, postnms=args.postnms, level=args.level)
                    print(len(masks))
                    if args.verbose == 1:
                        save_dir = os.path.join(base_dir, args.level, f"mask_each_frame_sam1")
                        plot_masks1([mask['segmentation'] for mask in masks], save_dir, now_frame)
                    
                # Add tracked masks to prompts_loader
                for id,mask in enumerate(masks_from_prev):
                    if mask.sum() == 0:
                        continue
                    prompts_loader.add(id,now_frame,mask[0])

                # Add new masks detected to prompts_loader
                if now_frame == 0:
                    ann_obj_id_list = range(len(masks))

                    for ann_obj_id in ann_obj_id_list:
                        seg = masks[ann_obj_id]['segmentation']
                        prompts_loader.add(ann_obj_id,0,seg)
                    logger.info(f"obj num: {prompts_loader.get_obj_num()}")
                elif is_key_frame:
                    new_mask_list = search_new_obj(masks_from_prev, masks, mask_ratio_thresh)
                    logger.info(f"number of new obj: {len(new_mask_list)}")
                    
                    for i in range(len(new_mask_list)):
                        new_mask = new_mask_list[i]['segmentation']
                        prompts_loader.add(sum_id+i,now_frame,new_mask)
                    
                # propagate the masks in the video
                video_segments = get_video_segments(prompts_loader,predictor,inference_state, args.detect_stride, now_frame)

                # Stop tracking if video ends
                out_frame_idx = now_frame+args.detect_stride
                if out_frame_idx >= len(frame_names):
                    break

                # Plot the masks
                if args.verbose == 1:
                    save_dir = os.path.join(base_dir, args.level, f"mask_each_frame_sam2")
                    img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                    plot_masks2(img, video_segments, save_dir, out_frame_idx)
                
                # Set Flag(is_key_frame) based on no_mask_ratio
                out_mask_list = [out_mask for _, out_mask in video_segments[out_frame_idx].items()]
                no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
                if now_frame == 0:
                    mask_ratio_thresh = no_mask_ratio

                if no_mask_ratio > min(0.15, mask_ratio_thresh + 0.01):
                    masks_from_prev = out_mask_list
                    mask_ratio_thresh = no_mask_ratio
                    logger.info(f"no_mask_ratio: {no_mask_ratio}, mask_ratio_thresh: {mask_ratio_thresh}")
                    is_key_frame = True
                else:
                    masks_from_prev = out_mask_list
                    is_key_frame = False
                    logger.info(f"no_mask_ratio: {no_mask_ratio}, mask_ratio_thresh: {mask_ratio_thresh}")


            ###### Final output ######
            video_segments = get_video_segments(prompts_loader,predictor,inference_state,args.detect_stride,0,final_output=True)
            mask_list_all = []
            for out_frame_idx in tqdm(range(0, len(frame_names), 1)):
                out_mask_list = []
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    out_mask_list.append(out_mask)

                # no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
                # logger.info(no_mask_ratio)

                out_mask_list = np.array(out_mask_list)

                mask_list_all.append(out_mask_list)
            
            save_dir = os.path.join(base_dir,args.level,"rle")
            save_masks_pkl_all(mask_list_all,save_dir)

            save_vid_path = os.path.join(base_dir,args.level,"seg.mp4")
            visualize_seg_masks(mask_list_all, video_dir, save_vid_path)

            predictor.reset_state(inference_state)

            if is_tmp_dir:
                cleanup_tmp_dir(video_dir)
        except Exception as e:
            print(e)
