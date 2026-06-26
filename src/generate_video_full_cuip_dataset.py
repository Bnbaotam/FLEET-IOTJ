import cv2
import os
from tqdm import tqdm


# Paths
img_dir = 'output/visualize_pred_fullset_6000_frames_cuip_0.1_radius_15_batch_size_20'
output_dir = 'output/visualize_pred_fullset_6000_frames_cuip_0.1_radius_15_batch_size_20_video'
output_video = 'output/visualize_pred_fullset_6000_frames_cuip_0.1_radius_15_batch_size_20_video/pred_fullset_6000_frames_cuip_0.1_radius_15_batch_size_20_visualized.mp4'
os.makedirs(output_dir, exist_ok=True)

num_frames = len(os.listdir(img_dir))

first_img = cv2.imread(os.path.join(img_dir, f'frame_{5:06d}.png'))
H, W = first_img.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

for frame_id in tqdm(range(1, num_frames + 1)):
    img_path = os.path.join(img_dir, f'frame_{frame_id:06d}.png')
    img = cv2.imread(img_path)
    if img is not None:
        video_writer.write(img)

video_writer.release()
print(f'Done! Video saved at: {output_video}')
