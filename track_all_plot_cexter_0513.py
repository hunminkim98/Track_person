import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import re
from matplotlib.animation import FuncAnimation

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

def calculate_centroid(points):
    return np.mean(points, axis=0)

def assign_person_id(people, current_id_map):
    person_id_map = {}
    current_ids = [v[0] for v in current_id_map.values()]
    new_id = max(current_ids, default=0) + 1
    for idx, person in enumerate(people):
        p1 = np.array(person['pose_keypoints_2d'])
        valid_points = [(p1[3 * j], p1[3 * j + 1]) for j in range(len(p1) // 3) if p1[3 * j] != 0 and p1[3 * j + 1] != 0]
        if len(valid_points) >= 3:
            hull = ConvexHull(valid_points)
            if idx in current_id_map:
                person_id_map[idx] = current_id_map[idx]
            else:
                person_id_map[idx] = (new_id, hull)
                new_id += 1
    return person_id_map

def calculate_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

def calculate_mae(person1, person2):
    p1 = np.array(person1['pose_keypoints_2d'])
    p2 = np.array(person2['pose_keypoints_2d'])
    x1, y1 = p1[::3], p1[1::3]
    x2, y2 = p2[::3], p2[1::3]
    valid = np.where((x1 != 0) & (y1 != 0) & (x2 != 0) & (y2 != 0))[0]
    if valid.size == 0:
        return float('inf')
    x_mae = np.mean(np.abs(x1[valid] - x2[valid]))
    y_mae = np.mean(np.abs(y1[valid] - y2[valid]))
    return np.mean([x_mae, y_mae])

def find_closest_person(last_known_centroid, people):
    min_distance = float('inf')
    closest_person_index = -1
    for j, person in enumerate(people):
        p1 = np.array(person['pose_keypoints_2d'])
        valid_points = [(p1[3 * k], p1[3 * k + 1]) for k in range(len(p1) // 3) if p1[3 * k] != 0 and p1[3 * k + 1] != 0]
        if len(valid_points) >= 3:
            centroid = calculate_centroid(valid_points)
            distance = calculate_distance(centroid, last_known_centroid)
            if distance < min_distance:
                min_distance = distance
                closest_person_index = j
    return closest_person_index, min_distance

def on_key(event):
    global user_input_received, user_input_value
    if event.key.isdigit():
        user_input_value = int(event.key) - 1
    elif event.key.lower() == 'n':
        user_input_value = 'n'
    user_input_received = True
    plt.close()

def check_for_reentry(people, last_known_centroid, data_to_track, person_id_map):
    closest_person_index, min_distance = find_closest_person(last_known_centroid, people)
    if min_distance < 100 and closest_person_index != -1:
        closest_person = people[closest_person_index]
        mae = calculate_mae(closest_person, {'pose_keypoints_2d': data_to_track})
        if mae < 20:
            tracking_person_id = person_id_map[closest_person_index][0]
            return tracking_person_id, closest_person['pose_keypoints_2d'], True
    return None, None, False

def track_person(folder_path):
    global user_input_received, user_input_value
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None, None

    pos1 = []
    pre_tracking_data = []
    prev_hull_centroid = None
    detected = False
    tracking_person_id = None
    last_known_centroid = None
    person_id_map = {}
    user_input_received = False
    user_input_value = None

    for i, file_name in enumerate(json_files):
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])
        pre_tracking_data.append(people)

        if not people:
            print("No people detected in the frame.")
            pos1.append(np.zeros(75))
            continue

        person_id_map = assign_person_id(people, person_id_map)

        if not detected:
            if len(people) >= 2:
                fig, ax = plt.subplots(figsize=(12, 8))
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                legend_handles = []
                for k, (pid, hull) in enumerate(person_id_map.items()):
                    color = colors[k % len(colors)]
                    x, y = zip(*hull[1].points)
                    handle, = ax.plot(x, -np.array(y), 'o', color=color, label=str(k + 1))
                    legend_handles.append(handle)
                    for simplex in hull[1].simplices:
                        ax.plot(hull[1].points[simplex, 0], -hull[1].points[simplex, 1], 'k-')
                    centroid = calculate_centroid(hull[1].points)
                    ax.text(centroid[0], -centroid[1], str(k + 1), color=color, fontsize=12, fontweight='bold')
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                ax.legend(handles=legend_handles, loc='upper right')
                fig.canvas.mpl_connect('key_press_event', on_key)
                plt.show()

                while not user_input_received:
                    plt.pause(0.1)

                if user_input_value != 'n' and user_input_value in person_id_map:
                    tracking_person_id, _ = person_id_map[user_input_value]
                    data_to_track = people[user_input_value]['pose_keypoints_2d']
                    pos1.append(data_to_track)
                    detected = True
                    prev_hull_centroid = calculate_centroid([(data_to_track[3 * j], data_to_track[3 * j + 1]) 
                                                             for j in range(len(data_to_track) // 3) if data_to_track[3 * j] != 0 and data_to_track[3 * j + 1] != 0])
                else:
                    pos1.append(np.zeros(75))
            elif len(people) == 1:
                print("Single person detected, automatically tracking this person.")
                tracking_person_id, _ = person_id_map[0]
                data_to_track = people[0]['pose_keypoints_2d']
                pos1.append(data_to_track)
                detected = True
                prev_hull_centroid = calculate_centroid([(data_to_track[3 * j], data_to_track[3 * j + 1]) 
                                                         for j in range(len(data_to_track) // 3) if data_to_track[3 * j] != 0 and data_to_track[3 * j + 1] != 0])
        else:
            person_idx = next((idx for idx, (pid, hull) in person_id_map.items() if pid == tracking_person_id), None)
            if person_idx is not None:
                hull = person_id_map[person_idx][1]
                centroids = [calculate_centroid([(person['pose_keypoints_2d'][3 * j], person['pose_keypoints_2d'][3 * j + 1]) 
                                                 for j in range(len(person['pose_keypoints_2d']) // 3) if person['pose_keypoints_2d'][3 * j] != 0 and person['pose_keypoints_2d'][3 * j + 1] != 0]) 
                             for person in people]
                current_centroid = calculate_centroid(hull.points)
                centroid_diff = np.linalg.norm(centroids[person_idx] - prev_hull_centroid)
                if centroid_diff > 500:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    valid_points = [(data_to_track[3 * j], data_to_track[3 * j + 1]) 
                                    for j in range(len(data_to_track) // 3) if data_to_track[3 * j] != 0 and data_to_track[3 * j + 1] != 0]
                    if len(valid_points) >= 3:
                        hull = ConvexHull(valid_points)
                        x, y = zip(*valid_points)
                        ax.plot(x, -np.array(y), 'bo', label='Previous Frame')
                        for simplex in hull.simplices:
                            ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                    valid_points = [(people[person_idx]['pose_keypoints_2d'][3 * j], people[person_idx]['pose_keypoints_2d'][3 * j + 1]) 
                                    for j in range(len(people[person_idx]['pose_keypoints_2d']) // 3) if people[person_idx]['pose_keypoints_2d'][3 * j] != 0 and people[person_idx]['pose_keypoints_2d'][3 * j + 1] != 0]
                    if len(valid_points) >= 3:
                        hull = ConvexHull(valid_points)
                        x, y = zip(*valid_points)
                        ax.plot(x, -np.array(y), 'ro', label='Current Frame')
                        for simplex in hull.simplices:
                            ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                    ax.set_xlim([0, 4000])
                    ax.set_ylim([-3000, 0])
                    ax.legend()
                    fig.canvas.mpl_connect('key_press_event', on_key)
                    plt.show()

                    while not user_input_received:
                        plt.pause(0.1)

                    if user_input_value == 'n':
                        pos1.append(np.zeros(75))
                        detected = False
                        tracking_person_id = None
                        last_known_centroid = prev_hull_centroid  # Update last known centroid
                        continue
                pos1.append(people[person_idx]['pose_keypoints_2d'])
                data_to_track = pos1[-1]
                prev_hull_centroid = centroids[person_idx]
            else:
                print("Tracked person is out of frame.")
                last_known_centroid = prev_hull_centroid
                pos1.append(np.zeros(75))
                detected = False
                tracking_person_id = None

        # Check for re-entry
        if not detected and last_known_centroid is not None:
            tracking_person_id, data_to_track, found = check_for_reentry(people, last_known_centroid, data_to_track, person_id_map)
            if found:
                pos1.append(data_to_track)
                detected = True
                prev_hull_centroid = calculate_centroid([(data_to_track[3 * j], data_to_track[3 * j + 1]) 
                                                         for j in range(len(data_to_track) // 3) if data_to_track[3 * j] != 0 and data_to_track[3 * j + 1] != 0])
                last_known_centroid = None
            else:
                pos1.append(np.zeros(75))
                last_known_centroid = None

    return np.array(pos1), json_files, pre_tracking_data

def save_data(data, json_files, folder_path):
    save_folder = os.path.join(folder_path, 'processed')
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

def animate_pre_post_tracking(all_pre_tracking_data, all_post_tracking_data, frame_step=30, interval=30):
    num_folders = len(all_pre_tracking_data)
    fig, axs = plt.subplots(num_folders, 2, figsize=(15, 5 * num_folders))

    if num_folders == 1:
        axs = [axs]

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
                for idx, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    x_data, y_data = p1[0::3], p1[1::3]
                    valid_points = [(x_data[j], y_data[j]) for j in range(len(x_data)) if x_data[j] != 0 and y_data[j] != 0]
                    if len(valid_points) >= 3:
                        hull = ConvexHull(valid_points)
                        pre_ax.plot(x_data, -y_data, 'o')
                        for simplex in hull.simplices:
                            pre_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                        centroid = calculate_centroid(valid_points)
                        pre_ax.text(centroid[0], -centroid[1], str(idx + 1), color='red', fontdict={'weight': 'bold', 'size': 12})

            # Post-tracking data
            if frame < all_post_tracking_data[i].shape[0]:
                x_data = all_post_tracking_data[i][frame, 0::3]
                y_data = all_post_tracking_data[i][frame, 1::3]
                valid_points = [(x_data[j], y_data[j]) for j in range(len(x_data)) if x_data[j] != 0 and y_data[j] != 0]
                if len(valid_points) >= 3:
                    hull = ConvexHull(valid_points)
                    post_ax.plot(x_data, -y_data, 'bo')
                    for simplex in hull.simplices:
                        post_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')
                    centroid = calculate_centroid(valid_points)
                    post_ax.text(centroid[0], -centroid[1], str(i + 1), color='blue', fontdict={'weight': 'bold', 'size': 12})

    max_frames = max(len(pre) for pre in all_pre_tracking_data)
    frames = range(0, max_frames, frame_step)
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_path = r'C:\Users\5W555A\Desktop\240423_liun\pose2sim\Pose2Sim\S00_Demo_BatchSession\S00_P00_SingleParticipant\S00_P00_T00_assis-debout\pose'
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

    animate_pre_post_tracking(all_pre_tracking_data, all_post_tracking_data, frame_step=10, interval=10)
1