#!/usr/bin/env python3
import os
import argparse
import cv2
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from itertools import chain

from utils import load_config
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions on train and test sets via unified_loader")
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--gt_file', required=True)
    parser.add_argument('--output_dir', default='output/visualize_pred_testset')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--sampling_step', type=int, default=1)
    return parser.parse_args()

def get_device(device_arg):
    if device_arg and 'cuda' in device_arg and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        return 'cpu'
    return device_arg or ('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_gt(path):
    cols = ['frame','id','x','y','w','h','not_ignored','cls','vis','conf']
    df = pd.read_csv(path, header=None, names=cols)
    df['frame'] = df['frame'].astype(int)
    df['id'] = df['id'].astype(int)
    # Get the center point of each object
    df['x'] = df['x'] + df['w'] / 2.0
    df['y'] = df['y'] + df['h'] / 2.0
    return df

def get_object_traj(df_gt, obj_id):
    return df_gt[df_gt['id'] == obj_id].sort_values('frame')

def draw_circle_safe(img, pt, radius, color, thickness):
    # pt: tuple, list, or  np.ndarray
    if (
        isinstance(pt, (tuple, list, np.ndarray))
        and len(pt) == 2
        and not (np.isnan(pt[0]) or np.isnan(pt[1]))
    ):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(img, (x, y), radius, color, thickness)


def main():
    args = parse_args()
    device_str = get_device(args.device)
    device = torch.device(device_str)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(argparse.Namespace(
        config_file=args.config_file,
        mode='test',
        gpu=str(device.index) if device.type=='cuda' else ''
    ))
    model = Build_Model(cfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    metrics = Build_Metrics(cfg)
    scale = float(getattr(cfg.DATA, "SCALE", 50.0))

    df_gt = load_gt(args.gt_file)

    # Loaders for train and test
    # start = time.time()
    loader_train = unified_loader(cfg, split='train', rand=False, batch_size=args.batch_size)
    loader_test  = unified_loader(cfg, split='test',  rand=False, batch_size=args.batch_size)
    # end = time.time()
    # loading_data_time = end - start
    # print(f"Loading data time:, {loading_data_time:.6f}s")

    # Merge two loaders to one general loader
    loader = chain(loader_train, loader_test)

    color_pred = (0,0,255)    # Red
    color_gt   = (255,0,0)    # Blue
    color_obs  = (0,255,0)    # Green

    visualize_dict = dict()
    for batch_count, batch in enumerate(tqdm(loader, desc="Processing prediction"), start=1):
        # Move tensors to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
            elif isinstance(batch[k], list) and len(batch[k]) > 0 and isinstance(batch[k][0], torch.Tensor):
                batch[k] = [x.to(device) for x in batch[k]]

        # Extract observation array [B, obs_len, 6]
        obs = batch['obs'].cpu().numpy()
        batch_size = obs.shape[0]
        obs_len = obs.shape[1]

        if obs_len < 8:
            # Skip the whole batch if the observation sequence is too short
            continue

        # Find valid indices (objects without NaN in first 8 obs points)
        valid_indices = [i for i in range(batch_size) if not np.isnan(obs[i, :8, :2]).any()]

        if len(valid_indices) == 0:
            # Skip this batch if all objects are invalid
            continue

        # Filter the batch to only keep valid objects (for prediction)
        filtered_batch = {}
        for k, v in batch.items():
            # If tensor and batch dimension equals batch_size, select only valid indices
            if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                filtered_batch[k] = v[valid_indices]
            # If list and length equals batch_size, select only valid indices
            elif isinstance(v, list) and len(v) == batch_size:
                filtered_batch[k] = [v[i] for i in valid_indices]
            else:
                # Otherwise, keep the original value (e.g., scalars, configs)
                filtered_batch[k] = v

        # Batch prediction for all valid objects
        with torch.no_grad():
            result_dict = model.predict(deepcopy(filtered_batch), return_prob=False)
            dict_list_denomalize = metrics.denormalize([result_dict])
            result_dict_denomalize = dict_list_denomalize[0]
            # pred_seq: [B_valid, pred_len, 2]

        # Collect visualization info for each valid object
        for idx_in_batch, i in enumerate(valid_indices):
            # Extract prediction for this object
            pred_seq = result_dict_denomalize[("pred", 0)][idx_in_batch]
            pred_seq = pred_seq.cpu().numpy() * scale

            # Meta info for visualization
            index = filtered_batch['index'][idx_in_batch]  # Example: ['cuip', np.int64(frame_idx), 'PEDESTRIAN/9']
            frame_idx = int(index[1])
            object_str = index[2]
            object_id = int(object_str.split('/')[-1])

            sampling_step = args.sampling_step
            frame_gt = (frame_idx - 1) * sampling_step + 1
            obj_traj = get_object_traj(df_gt, object_id)
            # frames_all = obj_traj['frame'].tolist()
            obs_frames = [frame_gt - (obs_len - j - 1) * sampling_step for j in range(obs_len)]
            future_frames = [frame_gt + (j + 1) * sampling_step for j in range(pred_seq.shape[0])]

            object_info = {
                'object_id': object_id,
                'obs_xy': obs[i, :8, :2] * scale,  # [obs_len, 2]
                'pred_seq': pred_seq,              # [pred_len, 2]
                'obj_traj': obj_traj,
                'obs_frames': obs_frames,
                'future_frames': future_frames,
            }
            if frame_gt not in visualize_dict:
                visualize_dict[frame_gt] = []
            visualize_dict[frame_gt].append(object_info)

        if args.max_batches is not None and batch_count >= args.max_batches:
            print(f"Reached max_batches={args.max_batches}, stopping early.")
            break

    # Block visualization
    # Get all unique frames from gt
    all_frames = sorted(df_gt['frame'].unique())

    # Store last predicted sequence for each object
    object_pred_history = {}

    for frame in tqdm(all_frames, desc="Drawing all frames"):
        img_path = os.path.join(args.images_dir, f"{frame:06d}.png")
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Find all objects appearing in this frame (from gt)
        frame_objs = df_gt[df_gt['frame'] == frame]['id'].unique()
        for object_id in frame_objs:
            obj_traj = get_object_traj(df_gt, object_id)
            # Check if this frame has a new prediction for this object
            updated_pred = None
            if frame in visualize_dict:
                for obj in visualize_dict[frame]:
                    if obj['object_id'] == object_id:
                        updated_pred = obj
                        break

            if updated_pred is not None:
                # New prediction available, update history
                object_pred_history[object_id] = {
                    'pred_seq': updated_pred['pred_seq'],
                    'future_frames': updated_pred['future_frames'],
                    'obj_traj': updated_pred['obj_traj'],
                    'last_pred_frame': frame,
                }
            elif object_id in object_pred_history:
                # No new prediction, so remove only the first point (shift by 1 each frame)
                pred_seq = object_pred_history[object_id]['pred_seq']
                if len(pred_seq) > 1:
                    object_pred_history[object_id]['pred_seq'] = pred_seq[1:]
                else:
                    # Only one (or zero) point left, remove history to stop drawing
                    del object_pred_history[object_id]

            # Draw prediction line if exists and object is still in scene
            if object_id in object_pred_history:
                if not obj_traj[obj_traj['frame'] == frame].empty:
                    pred_seq = object_pred_history[object_id]['pred_seq']
                    pred_seq_valid = np.array([pt for pt in pred_seq if not np.any(np.isnan(pt))])
                    for pt in pred_seq_valid:
                        draw_circle_safe(img, (int(pt[0]), int(pt[1])), 5, color_pred, -1)
                    min_pred_points_to_draw = 2
                    if len(pred_seq_valid) >= min_pred_points_to_draw:
                        cv2.polylines(img, [pred_seq_valid.astype(np.int32)], False, color_pred, 2)
                else:
                    del object_pred_history[object_id]

            # Draw obs from gt for this frame
            obs_len = 8 # Or use updated_pred['obs_frames'] if available
            sampling_step = args.sampling_step
            obs_frames = [frame - (obs_len - i - 1)*sampling_step for i in range(obs_len)]
            obs_xy_gt = []
            for f in obs_frames:
                row = obj_traj[obj_traj['frame'] == f]
                if not row.empty:
                    x, y = row.iloc[0]['x'], row.iloc[0]['y']
                    if np.isnan(x) or np.isnan(y):
                        continue
                    obs_xy_gt.append((int(x), int(y)))
            for pt in obs_xy_gt:
                # cv2.circle(img, pt, 5, color_obs, -1)
                draw_circle_safe(img, pt, 5, color_obs, -1)
            if len(obs_xy_gt) > 1:
                polyline_obs_xy_gt = np.array(obs_xy_gt, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [polyline_obs_xy_gt], False, color_obs, 2)

            # Draw future gt pts
            # Always compute future_frames from current frame
            future_length = 12  # Change this to your model's prediction length if needed
            sampling_step = args.sampling_step
            future_frames = [frame + (i+1)*sampling_step for i in range(future_length)]

            future_gt_pts = []
            for f in future_frames:
                row = obj_traj[obj_traj['frame'] == f]
                if not row.empty:
                    x, y = row.iloc[0]['x'], row.iloc[0]['y']
                    if np.isnan(x) or np.isnan(y):
                        continue
                    future_gt_pts.append((int(x), int(y)))
                    # cv2.circle(img, (int(x), int(y)), 5, color_gt, -1)
                    draw_circle_safe(img, (int(x), int(y)), 5, color_gt, -1)
            if len(future_gt_pts) > 1:
                polyline_future_gt_pts = np.array(future_gt_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [polyline_future_gt_pts], False, color_gt, 2)

        out_fn = os.path.join(args.output_dir, f'frame_{frame:06d}.png')
        cv2.imwrite(out_fn, img)

    print(f"Visualization done. Output in {args.output_dir}")

if __name__ == "__main__":
    main()
