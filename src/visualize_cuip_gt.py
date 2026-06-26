import cv2
import pandas as pd
import os
from tqdm import tqdm
import random

# Paths
gt_path = 'dataset/cuip_data_flowchain/gt/gt/gt.txt'
labels_path = 'dataset/cuip_data_flowchain/gt/gt/labels.txt'
img_dir = 'dataset/cuip_data_flowchain/images'
output_dir = 'output/visualized_frames'
output_video = 'output/gt_visualized.mp4'
os.makedirs(output_dir, exist_ok=True)

# Read class labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Read gt.txt (frame, id, x, y, w, h, not_ignored, class_id, visibility, conf)
cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'not_ignored', 'cls', 'vis', 'conf']
gt = pd.read_csv(gt_path, header=None, names=cols)

num_frames = gt['frame'].max()

colors = {}
def get_color(idx):
    if idx not in colors:
        random.seed(idx)
        colors[idx] = tuple(random.randint(64,255) for _ in range(3))
    return colors[idx]

first_img = cv2.imread(os.path.join(img_dir, f'{1:06d}.png'))
H, W = first_img.shape[:2]

# Store trajectory points and last seen frame for each object
trajectories = {}       # {obj_id: [(cx, cy), ...]}
last_seen_frame = {}    # {obj_id: frame_id}

history_len = 30

for frame_id in tqdm(range(1, num_frames + 1)):
    img_path = os.path.join(img_dir, f'{frame_id:06d}.png')
    img = cv2.imread(img_path)
    if img is None:
        continue

    frame_gt = gt[gt['frame'] == frame_id]

    for _, row in frame_gt.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        obj_id, cls_id = int(row['id']), int(row['cls'])
        color = get_color(obj_id)
        label = f"{labels[cls_id-1]}-{obj_id}"

        cx, cy = x + w // 2, y + h // 2

        if obj_id not in trajectories:
            trajectories[obj_id] = []
        trajectories[obj_id].append((cx, cy))
        # Only keep last N points
        if len(trajectories[obj_id]) > history_len:
            trajectories[obj_id] = trajectories[obj_id][-history_len:]
        # Update last seen frame
        last_seen_frame[obj_id] = frame_id

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw only trajectories for objects seen within last `history_len` frames
    for obj_id, pts in list(trajectories.items()):
        if frame_id - last_seen_frame.get(obj_id, 0) <= history_len:
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(img, pts[i-1], pts[i], get_color(obj_id), 2)
            if len(pts) > 0:
                # cv2.circle(img, pts[-1], 4, get_color(obj_id), -1) # Bbox color
                cv2.circle(img, pts[-1], 4, (255, 255, 255), -1)   # White color
        else:
            # Optionally: remove old object completely to save memory
            # del trajectories[obj_id]
            pass

    out_path = os.path.join(output_dir, f'{frame_id:06d}.png')
    cv2.imwrite(out_path, img)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

for frame_id in tqdm(range(1, num_frames + 1)):
    img_path = os.path.join(output_dir, f'{frame_id:06d}.png')
    img = cv2.imread(img_path)
    if img is not None:
        video_writer.write(img)

video_writer.release()
print(f'Done! Video saved at: {output_video}')
