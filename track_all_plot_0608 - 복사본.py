import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import re

def find_json_folders(base_path):
    json_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(file.endswith('.json') for file in files):
            json_folders.append(root)
    return json_folders

def read_json_file(file_path):
    print(f"Reading file from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def track_person(folder_path):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None, None

    pos1 = []
    pre_tracking_data = []
    previous_data = None

    for i, file_name in enumerate(json_files):
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])
        pre_tracking_data.append(people)

        if previous_data is None:
            if len(people) > 0:
                pos1.append(people[0]['pose_keypoints_2d'])
                previous_data = people[0]['pose_keypoints_2d']
            else:
                pos1.append(np.zeros(75))
        else:
            best_match = None
            best_distance = float('inf')
            for person in people:
                p1 = np.array(person['pose_keypoints_2d'])
                distance = np.sum(np.sqrt((previous_data - p1) ** 2))
                if distance < best_distance:
                    best_match = p1
                    best_distance = distance
            if best_match is not None:
                pos1.append(best_match)
                previous_data = best_match
            else:
                pos1.append(np.zeros(75))

    return np.array(pos1), json_files, pre_tracking_data

def animate_pre_post_tracking(all_pre_tracking_data, all_post_tracking_data, frame_step=10, interval=100):
    num_folders = len(all_pre_tracking_data)
    fig, axs = plt.subplots(num_folders, 2, figsize=(15, 5 * num_folders))

    def update(frame):
        for i in range(num_folders):
            pre_ax, post_ax = axs[i]
            pre_ax.clear()
            post_ax.clear()
            pre_ax.set_title(f'Pre-Tracking Folder {i + 1}')
            pre_ax.set_xlim([0, 4000])
            pre_ax.set_ylim([-3000, 0])
            post_ax.set_title(f'Post-Tracking Folder {i + 1}')
            post_ax.set_xlim([0, 4000])
            post_ax.set_ylim([-3000, 0])

            # Pre-tracking data
            if frame < len(all_pre_tracking_data[i]):
                people = all_pre_tracking_data[i][frame]
                hulls = []
                for person in people:
                    x_data = person['pose_keypoints_2d'][0::3]
                    y_data = person['pose_keypoints_2d'][1::3]
                    valid_points = [(x_data[j], y_data[j]) for j in range(len(x_data)) if x_data[j] != 0 and y_data[j] != 0]
                    if len(valid_points) >= 3:
                        hull = ConvexHull(valid_points)
                        hulls.append(hull)
                        pre_ax.plot(x_data, -np.array(y_data), 'o')
                for hull in hulls:
                    for simplex in hull.simplices:
                        pre_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')

            # Post-tracking data
            if frame < all_post_tracking_data[i].shape[0]:
                x_data = all_post_tracking_data[i][frame, 0::3]
                y_data = all_post_tracking_data[i][frame, 1::3]
                if len(x_data) == len(y_data):
                    valid_points = [(x_data[j], y_data[j]) for j in range(len(x_data)) if x_data[j] != 0 and y_data[j] != 0]
                    if len(valid_points) >= 3:
                        hull = ConvexHull(valid_points)
                        post_ax.plot(x_data, -np.array(y_data), 'bo')
                        for simplex in hull.simplices:
                            post_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')

    max_frames = max(len(pre) for pre in all_pre_tracking_data)
    frames = range(0, max_frames, frame_step)
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()

def save_data(data, json_files, folder_path):
    base_folder = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    tracked_folder_name = folder_name + '_tracked'
    save_folder = os.path.join(base_folder, tracked_folder_name)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, file_name in enumerate(json_files):
        frame_data = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": data[i].tolist(), 
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(frame_data, f)
    print(f"Saved processed data to {save_folder}")

if __name__ == "__main__":
    base_path = r'C:\Users\5W555A\Desktop\FFF\pose2sim-w-Marker-Augmenter-Sync\Pose2Sim\S00_Demo_BatchSession\S00_P00_SingleParticipant\S00_P00_T01_BalancingTrial\pose\kicking10_4'
    json_folders = find_json_folders(base_path)

    all_pre_tracking_data = []
    all_post_tracking_data = []

    for folder in json_folders:
        pos1, json_files, pre_tracking_data = track_person(folder)
        if pos1 is None:
            continue
        all_pre_tracking_data.append(pre_tracking_data)
        all_post_tracking_data.append(pos1)
        save_data(pos1, json_files, folder)
    
    animate_pre_post_tracking(all_pre_tracking_data, all_post_tracking_data, frame_step=50, interval=30)
