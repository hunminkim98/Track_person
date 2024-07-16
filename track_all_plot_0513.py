import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import re

def load_json_files(base_path):
    folders = [os.path.join(base_path, d) for d in sorted(os.listdir(base_path)) 
               if os.path.isdir(os.path.join(base_path, d)) and re.search(r'json\d+$', d)]
    return folders

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

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []
    pre_tracking_data = []
    skip_frame = False

    def on_key(event):
        nonlocal skip_frame
        if event.key == 'n':
            skip_frame = True
            plt.close()

    for i, file_name in enumerate(json_files):
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])
        pre_tracking_data.append(people)

        if not detected and not right_person:
            if not people:
                print("No people detected in the frame.")
                pos1.append(np.zeros(75))
            elif len(people) >= 2:
                fig, ax = plt.subplots(figsize=(12, 8))
                av1 = []
                hulls = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    valid_points = [(p1[3 * j], p1[3 * j + 1]) for j in range(len(p1) // 3) if p1[3 * j] != 0 and p1[3 * j + 1] != 0]
                    if len(valid_points) >= 3:
                        try:
                            hull = ConvexHull(valid_points)
                            hulls.append(hull)
                            av1.append(hull.volume)
                            x, y = zip(*valid_points)
                            ax.plot(x, -np.array(y), 'o')
                        except Exception as e:
                            print(f"Error in ConvexHull calculation: {e}")
                    else:
                        print("Not enough valid points for ConvexHull")
                for hull in hulls:
                    for simplex in hull.simplices:
                        ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                plt.legend([str(i + 1) for i in range(len(people))])
                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()
                
                if skip_frame:
                    pos1.append(np.zeros(75))
                    detected = True
                    right_person = False
                    skip_frame = False
                else:
                    num_to_track = int(input("Select num: "))
                    pos1.append(people[num_to_track - 1]['pose_keypoints_2d'])
                    data_to_track = pos1[-1]
                    detected = True
                    right_person = True
            elif len(people) == 1:
                print("Single person detected, automatically tracking this person.")
                pos1.append(people[0]['pose_keypoints_2d'])
                data_to_track = pos1[-1]
                detected = True
                right_person = True
        elif detected and right_person:
            if people:
                mae = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    x0, y0 = np.array(data_to_track[::3]), np.array(data_to_track[1::3])
                    x1, y1 = p1[::3], p1[1::3]
                    valid = np.where((x0 != 0) & (y0 != 0) & (x1 != 0) & (y1 != 0))[0]
                    if valid.size == 0:
                        x_mae, y_mae = float('inf'), float('inf')
                    else:
                        x_mae = np.mean(np.abs(x0[valid] - x1[valid]))
                        y_mae = np.mean(np.abs(y0[valid] - y1[valid]))
                    mae.append(np.mean([x_mae, y_mae]))
                min_avg, I1 = min((val, idx) for (idx, val) in enumerate(mae))
                print(f"min_avg: {min_avg}, I1: {I1}")
                if min_avg > 100:
                    pos1.append(np.zeros(75))
                    detected = False
                    right_person = False
                else:
                    pos1.append(people[I1]['pose_keypoints_2d'])
                    data_to_track = pos1[-1]
            else:
                pos1.append(np.zeros(75))
                detected = False
                right_person = False
        elif detected and not right_person:
            if not people:
                pos1.append(np.zeros(75))
                detected = False
                right_person = False
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                hulls = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    valid_points = [(p1[3 * j], p1[3 * j + 1]) for j in range(len(p1) // 3) if p1[3 * j] != 0 and p1[3 * j + 1] != 0]
                    if len(valid_points) >= 3:
                        try:
                            hull = ConvexHull(valid_points)
                            hulls.append(hull)
                            x, y = zip(*valid_points)
                            ax.plot(x, -np.array(y), 'o')
                        except Exception as e:
                            print(f"Error in ConvexHull calculation: {e}")
                    else:
                        print("Not enough valid points for ConvexHull")
                for hull in hulls:
                    for simplex in hull.simplices:
                        ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                plt.legend([str(i + 1) for i in range(len(people))])
                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()

                if skip_frame:
                    pos1.append(np.zeros(75))
                    detected = True
                    right_person = False
                    skip_frame = False
                else:
                    num_to_track = int(input("Select num: "))
                    pos1.append(people[num_to_track - 1]['pose_keypoints_2d'])
                    data_to_track = pos1[-1]
                    detected = True
                    right_person = True

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
    cam_num = os.path.basename(folder_path).split('_')[-1]
    save_folder = os.path.join(folder_path, 'processed')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, file_name in enumerate(json_files):
        frame_data = {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": data[i].tolist(), 
                      "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": [],
                      "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []}]}
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(frame_data, f)

if __name__ == "__main__":
    base_path = r'C:\Users\5W555A\Desktop\240423_liun\pose2sim\Pose2Sim\S00_Demo_BatchSession\S00_P00_SingleParticipant'
    folders = load_json_files(base_path)

    all_pre_tracking_data = []
    all_post_tracking_data = []

    for folder in folders:
        pos1, json_files, pre_tracking_data = track_person(folder)
        if pos1 is None:
            continue
        all_pre_tracking_data.append(pre_tracking_data)
        all_post_tracking_data.append(pos1)
        save_data(pos1, json_files, folder)
    
    animate_pre_post_tracking(all_pre_tracking_data, all_post_tracking_data, frame_step=50, interval=30)
