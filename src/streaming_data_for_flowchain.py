import os
import argparse
import time
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from ultralytics import YOLO

from data.TP.environment import Environment, Scene, Node, derivative_of

from yacs.config import CfgNode
from torch.utils.data import DataLoader, Subset

from utils import load_config
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics

np.random.seed(123)

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.1

# standardization = {
#     'PEDESTRIAN': {
#         'position': {
#             'x': {'mean': 0, 'std': 1},
#             'y': {'mean': 0, 'std': 1}
#         },
#         'velocity': {
#             'x': {'mean': 0, 'std': 2},
#             'y': {'mean': 0, 'std': 2}
#         },
#         'acceleration': {
#             'x': {'mean': 0, 'std': 1},
#             'y': {'mean': 0, 'std': 1}
#         }
#     }
# }

#VEHICLE, position device by 50 and already interpolated to make sure there are only consecutive frames
standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 17.82, 'std': 5.52}, 
            'y': {'mean': 8.93, 'std': 1.44}
        }, 
        'velocity': {
        'x': {'mean': 0.25, 'std': 2.14},
        'y': {'mean': 0.07, 'std': 0.72}
        }, 
        'acceleration': {
        'x': {'mean': 0.04, 'std': 6.93}, 
        'y': {'mean': 0.12, 'std': 3.10}
        }
    }
}

# Configuration
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
RTSP_URL = os.environ.get("RTSP_URL", "")
MODEL_PATH  = "./src/models/yolo11l_retrain.pt"
TARGET_FPS  = 10         # process at 10 FPS
WINDOW_SZ   = 8          # sliding window size

# Class mapping if you need class IDs later
CLASS_MAP = {"car":1,"truck":2,"bus":3,"bicycle":4,"person":5,"cyclist":6}


def make_df_from_results(results, frame_id):
    """Helper: turn a YOLO track() results into a per-frame DataFrame."""
    recs = []
    for box in results.boxes:

        if box.id is not None:
            tid = int(box.id.item())
        else:
            continue

        x1, y1, x2, y2 = box.xyxy.cpu().numpy().reshape(-1)
        cx, cy = float((x1 + x2)/2), float((y1 + y2)/2)

        recs.append({
            "frame":   frame_id,
            "trackId": tid,
            "x":       cx,
            "y":       cy,
            "sceneId": "Georgie_NW",
            "metaId":  tid,
        })

        print(recs)
    return pd.DataFrame(recs, columns=["frame","trackId","x","y","sceneId","metaId"])

def interpolate_window(df_window):
    out = []
    for tid, grp in df_window.groupby("trackId"):
        grp = grp.set_index("frame").sort_index()

        full_idx = range(grp.index.min(), grp.index.max() + 1)
        # reindex to full frame range
        grp_full = grp.reindex(full_idx)

        # sceneId & metaId are constant per track → fill them
        grp_full["sceneId"] = grp_full["sceneId"].ffill().bfill()
        grp_full["metaId"]  = grp_full["metaId"].ffill().bfill()

        # interpolate x,y linearly over the new index
        grp_full[["x", "y"]] = grp_full[["x", "y"]].interpolate()

        # record the trackId and frame index
        grp_full["trackId"] = tid
        grp_full = grp_full.reset_index().rename(columns={"index": "frame"})

        out.append(grp_full)
    # combine all tracks, sort, and ensure no duplicate rows
    return (
        pd.concat(out, ignore_index=True)
          .drop_duplicates(["frame","trackId"])
          .sort_values(["frame","trackId"])
          .reset_index(drop=True)
    )

# Process Stanford Drone and CUIP (pixel-based). Data obtained from Y-Net github repo
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
def create_flowchain_enviroment(df):
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 15.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")
    for scene, data in group:

        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        #data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Mean Position
        #data['x'] = data['x'] - data['x'].mean()
        #data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        if len(data) > 0:

            scene = Scene(timesteps=max_timesteps+1, dt=dt, name=scene, aug_func=None)
            n=0
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]


                if len(node_df) > 1:
                    assert np.all(np.diff(node_df['frame']) == 1)
                    if not np.all(np.diff(node_df['frame']) == 1):
                        import pdb;pdb.set_trace()
                        

                    node_values = node_df[['x', 'y']].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame'].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {('position', 'x'): x,
                                    ('position', 'y'): y,
                                    ('velocity', 'x'): vx,
                                    ('velocity', 'y'): vy,
                                    ('acceleration', 'x'): ax,
                                    ('acceleration', 'y'): ay}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
            # print(scene)
            scenes.append(scene)
    env.scenes = scenes

    return env    

def custom_unified_loader(cfg: CfgNode, env: Environment, rand=True, split="train", batch_size=None, data_fraction=1.0) -> DataLoader:
    # train, val, test
    if cfg.DATA.TASK == "TP":
        from data.TP.trajectron_dataset import EnvironmentDataset, hypers
        
        if 'longer' in cfg.DATA.DATASET_NAME and split != "train":
            i = int(cfg.DATA.DATASET_NAME[-1])
            cfg.defrost()
            cfg.DATA.OBSERVE_LENGTH -= i
            cfg.DATA.DATASET_NAME = cfg.DATA.DATASET_NAME[:-8]
            cfg.freeze()
            
        if (cfg.DATA.DATASET_NAME == 'stanford' or cfg.DATA.DATASET_NAME == 'cuip' or cfg.DATA.DATASET_NAME == 'cuip_0.1' or cfg.DATA.DATASET_NAME == 'cuip_0.1_viz') and split != 'train':
            i = cfg.DATA.PREDICT_LENGTH - 12
            cfg.defrost()
            cfg.DATA.OBSERVE_LENGTH -= i
            cfg.freeze()

        dataset = EnvironmentDataset(env,
                                    state=hypers[cfg.DATA.TP.STATE],
                                    pred_state=hypers[cfg.DATA.TP.PRED_STATE],
                                    node_freq_mult=hypers['scene_freq_mult_train'],
                                    scene_freq_mult=hypers['node_freq_mult_train'],
                                    hyperparams=hypers,
                                    min_history_timesteps=1 if cfg.DATA.TP.ACCEPT_NAN and split == 'train' else cfg.DATA.OBSERVE_LENGTH - 1,
                                    min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
                                    #augment=hypers['augment'] and split == 'train'
                                    )
        # print(f"[DEBUG] Number of node_type_datasets: {len(dataset.node_type_datasets)}")
        # for nt_dataset in dataset.node_type_datasets:
        #     print(f"[DEBUG] Node type: {nt_dataset.node_type}, Number of samples: {len(nt_dataset)}")


        # assume we have only 'PEDESTRIAN' node
        for node_type_dataset in dataset:
            if node_type_dataset.node_type == 'PEDESTRIAN':
                dataset = node_type_dataset
                break                                         
        
    # Choose the appropriate collate function
    if cfg.DATA.TASK == "TP":
        from data.TP.preprocessing import dict_collate as seq_collate

    # Subset the dataset if a fraction less than 1.0 is requested
    if data_fraction < 1.0:
        total_length = len(dataset)
        subset_length = max(1, int(total_length * data_fraction))
        indices = list(range(subset_length))
        dataset = Subset(dataset, indices)
    
    if batch_size is None:
        batch_size = cfg.DATA.BATCH_SIZE
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=seq_collate,
        drop_last=True if split == 'train' else False,
        pin_memory=True)
    
    return loader

def parse_args():
    parser = argparse.ArgumentParser(description="Streaming data for FlowChain via custom_unified_loader")
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint', required=True)
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

def main():
    args = parse_args()
    device_str = get_device(args.device)
    device = torch.device(device_str)

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

    # Streaming Initialization
    if not RTSP_URL:
        raise RuntimeError("Set the RTSP_URL environment variable before running live streaming.")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("Cannot open RTSP stream (check RTSP_URL and network access).")

    # Calculate how many grabs to skip to get ~10 FPS
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip    = max(1, int(round(src_fps / TARGET_FPS)))

    model_yolo = YOLO(MODEL_PATH)
    frame_count = 0

    # This deque holds the last WINDOW_SZ per-frame DataFrames
    window = deque(maxlen=WINDOW_SZ)

    # Main loop
    while True:
        # a) Skip decode for skip−1 frames
        for _ in range(skip-1):
            if not cap.grab():
                break

        # b) Grab the one we process
        if not cap.grab():
            print("Stream ended.")
            break

        ret, frame = cap.retrieve()
        if not ret:
            print("Failed retrieve.")
            break

        frame_count += 1

        # c) Run YOLO tracker
        results = model_yolo.track(frame, device=0, persist=True)[0]

        # if no boxes were detected, skip to the next frame
        if results.boxes.shape[0] == 0:
            continue

        df_this = make_df_from_results(results, frame_count)
        
        if df_this.empty:
            continue

        window.append(df_this)

        if len(window) == WINDOW_SZ:
            # d) Concatenate current window
            df_window = pd.concat(window, ignore_index=True)
            
            # e) Interpolate single-frame gaps
            df_interp = interpolate_window(df_window)

            # ===> ADD DUMMY POINTS <===
            obs_needed = 8
            total_needed = 20
            for trackId in df_interp['trackId'].unique():
                track_df = df_interp[df_interp['trackId'] == trackId]
                if len(track_df) == obs_needed:
                    # Take the last observation row for this track
                    last_row = track_df.iloc[-1].copy()
                    # Add dummy points to reach total_needed
                    new_rows = []
                    for i in range(total_needed - obs_needed):
                        new_row = last_row.copy()
                        new_row['frame'] = last_row['frame'] + i + 1
                        new_rows.append(new_row)
                    # Append dummy rows to df_interp
                    df_interp = pd.concat([df_interp, pd.DataFrame(new_rows)], ignore_index=True)
            # Sort again by trackId and frame to keep the data consistent
            df_interp = df_interp.sort_values(['trackId', 'frame']).reset_index(drop=True)

            # >>> HERE is where you have your **live** sliding-{WINDOW_SZ}-frame DataFrame:
            #     `df_interp` has columns ["frame","trackId","x","y","sceneId","metaId"]
            #
            # For example, just to see it:
            print(f"\n=== Sliding Window ending at frame {frame_count} ===")
            print(df_interp.groupby('trackId').size())
            print(df_interp)

            start = time.time()
            env = create_flowchain_enviroment(deepcopy(df_interp))
            # for scene in env.scenes:
            #     print(f"[DEBUG] Scene: {scene}, Number of nodes: {len(scene.nodes)}")
            end = time.time()
            create_env_time = end - start
            print(f"Total creating Flowchain enviroment time: {create_env_time:.6f}s")

            start = time.time()
            loader  = custom_unified_loader(cfg, env, split='test',  rand=False, batch_size=args.batch_size)
            print('[DEBUG] Loader length:', len(loader))
            # print('[DEBUG] loader: ', loader)
            end = time.time()
            data_loading_time = end - start 
            print(f"Total FlowChain data loading time: {data_loading_time:.6f}s")

            
            # start_total_prediction = time.time()
            visualize_dict = dict()
            # for batch_count, batch in enumerate(tqdm(loader, desc="Processing prediction"), start=1):
            for batch_count, batch in enumerate(loader, start=1):
                start = time.time()
                # Move tensors to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                    elif isinstance(batch[k], list) and len(batch[k]) > 0 and isinstance(batch[k][0], torch.Tensor):
                        batch[k] = [x.to(device) for x in batch[k]]

                # Extract observation array [B, obs_len, 6]
                # obs = batch['obs'].cpu().numpy()
                # batch_size = obs.shape[0]
                # obs_len = obs.shape[1]
        
                # On device, don't convert unless visualization
                obs = batch['obs']
                batch_size = obs.shape[0]
                obs_len = obs.shape[1]

                # print('obs shape:', obs.shape)
                # print('obs (first 2):', obs[:2]) 

                if obs_len < 8:
                    # Skip the whole batch if the observation sequence is too short
                    continue

                # Find valid indices (objects without NaN in first 8 obs points)
                # Create a boolean mask for valid indices
                mask = ~torch.isnan(obs[:, :8, :2]).any(dim=(1,2))
                valid_indices = torch.where(mask)[0]

                # print(f'Valid indices: {valid_indices}')

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
                end = time.time()
                move_to_tensors_and_check_valid_indices_time = end - start
                print(f"Each batch move to tensors and check valid indices time: {move_to_tensors_and_check_valid_indices_time:.6f}s")

                # Batch prediction for all valid objects
                start_total_prediction = time.time()
                with torch.no_grad():
                    start = time.time()
                    result_dict = model.predict(deepcopy(filtered_batch), return_prob=False)
                    end = time.time()
                    model_prediction_time = end - start
                    print(f"Each batch model prediction time: {model_prediction_time:.6f}s")

                    start = time.time()
                    dict_list_denomalize = metrics.denormalize([result_dict])
                    result_dict_denomalize = dict_list_denomalize[0]
                    end = time.time()
                    model_denormalization_time = end - start
                    print(f"Each batch model denormalization time: {model_denormalization_time:.6f}s")
                    # pred_seq: [B_valid, pred_len, 2]
                end_total_prediction = time.time()
                trajectory_prediction_time = end_total_prediction - start_total_prediction
                print(f"Each batch total trajectory prediction time: {trajectory_prediction_time:.6f}s")

                start_creating_data_visualization_time = time.time()
                # Collect visualization info for each valid object
                for idx_in_batch, i in enumerate(valid_indices):
                    # Extract prediction for this object
                    # pred_seq = result_dict_denomalize[("pred", 0)][idx_in_batch]
                    # pred_seq = pred_seq.cpu().numpy() * scale
                    pred_seq = result_dict_denomalize[("pred", 0)][idx_in_batch] * scale

                    # Meta info for visualization
                    index = filtered_batch['index'][idx_in_batch]  # Example: ['cuip', np.int64(frame_idx), 'PEDESTRIAN/9']
                    frame_idx = int(index[1])
                    object_str = index[2]
                    object_id = int(object_str.split('/')[-1])

                    sampling_step = args.sampling_step
                    frame_gt = (frame_idx - 1) * sampling_step + 1
                    # obs_frames = [frame_gt - (obs_len - j - 1) * sampling_step for j in range(obs_len)]
                    # future_frames = [frame_gt + (j + 1) * sampling_step for j in range(pred_seq.shape[0])]

                    object_info = {
                        'object_id': object_id,
                        'obs_xy': obs[i, :8, :2] * scale,  # [obs_len, 2]
                        'pred_seq': pred_seq,              # [pred_len, 2]
                        # 'obs_frames': obs_frames,
                        # 'future_frames': future_frames,
                    }
                    if frame_gt not in visualize_dict:
                        visualize_dict[frame_gt] = []
                    visualize_dict[frame_gt].append(object_info)
                
                end_creating_data_visualization_time = time.time()
                creating_data_visualization_time = end_creating_data_visualization_time - start_creating_data_visualization_time
                print(f"Time of Creating Data for Visualization: {creating_data_visualization_time:.6f}s")

                if args.max_batches is not None and batch_count >= args.max_batches:
                    print(f"Reached max_batches={args.max_batches}, stopping early.")
                    break

            # fps = WINDOW_SZ / trajectory_prediction_time
            # print(f"FPS of trajectory prediction: {fps:.2f}")
            print('[DEBUG] visualize_dict: ', visualize_dict)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
