import sys
sys.path.append('./src')
import os
import numpy as np
import pandas as pd
import dill
import pickle

from data.TP.environment import Environment, Scene, Node, derivative_of

np.random.seed(123)

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

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

        node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0

raw_path = './src/data/TP/raw_data/'
data_folder_name = './src/data/TP/processed_data/'

maybe_makedirs(data_folder_name)


standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {
                'mean': 17.85,
                'std': 3.05
            },
            'y': {
                'mean': 7.025,
                'std': 0.965
            }
        },
        'velocity': {
            'x': {
                'mean': 0.0,
                'std': 1.525
            },
            'y': {
                'mean': 0.32,
                'std': 0.68
            }
        },
        'acceleration': {
            'x': {
                'mean': 0.0,
                'std': 10.305
            },
            'y': {
                'mean': 0.265,
                'std': 4.175
            }
        }
    }
}


file_list = [
    # # --- CityFlow (C0xx) ---
    # "C001_test.pkl",
    # "C001_train.pkl",
    # "C002_test.pkl",
    # "C002_train.pkl",
    # "C003_test.pkl",
    # "C003_train.pkl",
    # "C004_test.pkl",
    # "C004_train.pkl",
    # "C005_test.pkl",
    # "C005_train.pkl",
    # "C028_test.pkl",
    # "C028_train.pkl",
    # "C035_test.pkl",
    # "C035_train.pkl",
    # "C041_test.pkl",
    # "C041_train.pkl",
    # "C042_test.pkl",
    # "C042_train.pkl",
    # "C043_test.pkl",
    # "C043_train.pkl",
    # "C044_test.pkl",
    # "C044_train.pkl",
    # "C045_test.pkl",
    # "C045_train.pkl",
    # "C046_test.pkl",
    # "C046_train.pkl",
    # # --- Named intersections (Chattanooga) ---
    # "Broad_2_test.pkl",
    # "Broad_2_train.pkl",
    # "Georgia_1_test.pkl",
    # "Georgia_1_train.pkl",
    # "Georgia_2_test.pkl",
    # "Georgia_2_train.pkl",
    # "Hwy27_1_test.pkl",
    # "Hwy27_1_train.pkl",
    # "Hwy27_2_test.pkl",
    # "Hwy27_2_train.pkl",
    # "Lindsay_1_test.pkl",
    # "Lindsay_1_train.pkl",
    # "Lindsay_2_test.pkl",
    # "Lindsay_2_train.pkl",
    # "Market_1_test.pkl",
    # "Market_1_train.pkl",
    # "Pine_1_test.pkl",
    # "Pine_1_train.pkl",
    # "Pine_2_test.pkl",
    # "Pine_2_train.pkl",
    # --- TUMTraf (s110) ---
    "s110_camera_basler_south1_8mm_test.pkl",
    "s110_camera_basler_south1_8mm_train.pkl",
    "s110_camera_basler_south2_8mm_test.pkl",
    "s110_camera_basler_south2_8mm_train.pkl",
]

# ---- config ----
dt               = 0.1   # 10 FPS, no downsampling
frame_diff       = 1     # consecutive frames differ by 1
desired_frame_diff = 1
desired_max_time = 100
pred_indices     = [2, 3]
state_dim        = 6


data_columns = pd.MultiIndex.from_product(
    [['position', 'velocity', 'acceleration'], ['x', 'y']]
)

# ---- process each file ----
for filename in file_list:
    data_path = os.path.join(raw_path, filename)
    data_out_path = os.path.join(data_folder_name, filename)  # same name

    # determine train/test for augmentation
    is_train = "train" in filename.lower()
    data_class = "train" if is_train else "test"

    print(f"\nProcessing [{data_class}]: {filename}")

    if not os.path.exists(data_path):
        print(f"  WARNING: File not found, skipping -> {data_path}")
        continue

    df = pickle.load(open(data_path, "rb"))

    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 15.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")
    for scene_id, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')
        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # scale as PECNet
        data['x'] = data['x'] / 50
        data['y'] = data['y'] / 50

        max_timesteps = data['frame'].max()

        if len(data) > 0:
            scene = Scene(
                timesteps=max_timesteps + 1,
                dt=dt,
                name=scene_id,
                aug_func=augment if is_train else None
            )

            for node_id in pd.unique(data['node_id']):
                node_df = data[data['node_id'] == node_id]

                if len(node_df) > 1:
                    if not np.all(np.diff(node_df['frame']) == 1):
                        print(f"  WARNING: Non-consecutive frames for node {node_id} in scene {scene_id}, skipping.")
                        continue

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

                    data_dict = {
                        ('position', 'x'): x, ('position', 'y'): y,
                        ('velocity', 'x'): vx, ('velocity', 'y'): vy,
                        ('acceleration', 'x'): ax, ('acceleration', 'y'): ay
                    }

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(
                        node_type=env.NodeType.PEDESTRIAN,
                        node_id=node_id,
                        data=node_data
                    )
                    node.first_timestep = new_first_idx
                    scene.nodes.append(node)

            if is_train:
                scene.augmented = list()
                angles = np.arange(0, 360, 15)
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(f"  {scene}")
            scenes.append(scene)

    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print(f"  Saved -> {data_out_path}")
    else:
        print(f"  WARNING: No scenes found, output not saved for {filename}")

print("\nAll files processed.")

