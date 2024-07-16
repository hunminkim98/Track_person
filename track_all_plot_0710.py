import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import re
from tqdm import tqdm

def find_json_folders(base_path):
    json_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(file.endswith('.json') for file in files):
            json_folders.append(root)
    return json_folders

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def track_person(folder_path):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None, None, None

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []
    pre_tracking_data = []
    selected_person_idx = None
    total_min_avg = 0
    count_min_avg = 0
    keypoint_count = 0

    def on_click(event):
        nonlocal selected_person_idx
        for i, person in enumerate(people):
            p1 = np.array(person['pose_keypoints_2d'])
            x = p1[::3]
            y = -p1[1::3]
            if np.any((np.abs(x - event.xdata) < 20) & (np.abs(y - event.ydata) < 20)):
                selected_person_idx = i
                plt.close()
                break

    for i, file_name in tqdm(enumerate(json_files), total=len(json_files), desc="Processing files", ncols=100):
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])
        pre_tracking_data.append(people)

        if i == 0:
            if people:
                keypoint_count = len(people[0]['pose_keypoints_2d'])
                print(f"Detected {keypoint_count} keypoints in the first frame\n")
            else:
                print("No people detected in the first frame")
                keypoint_count = 75  # Default to 25 keypoints * 3 (x, y, confidence)

        current_pos = np.zeros(keypoint_count)

        if not detected and not right_person:
            if not people:
                pos1.append(current_pos)
            elif len(people) >= 2:
                fig, ax = plt.subplots(figsize=(12, 8))
                hulls = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    valid_points = [(p1[3*j], p1[3*j+1]) for j in range(len(p1)//3) if p1[3*j] != 0 and p1[3*j+1] != 0]
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
                plt.legend([str(i+1) for i in range(len(people))])
                fig.canvas.mpl_connect('button_press_event', on_click)
                plt.show()
                
                if selected_person_idx is not None:
                    current_pos = np.array(people[selected_person_idx]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    detected = True
                    right_person = True
                else:
                    detected = True
                    right_person = False
            elif len(people) == 1:
                print("Single person detected, automatically tracking this person.")
                current_pos = np.array(people[0]['pose_keypoints_2d'])
                data_to_track = current_pos
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
                if min_avg > 100:
                    detected = False
                    right_person = False
                else:
                    current_pos = np.array(people[I1]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    total_min_avg += min_avg
                    count_min_avg += 1
            else:
                detected = False
                right_person = False
        elif detected and not right_person:
            if not people:
                detected = False
                right_person = False
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                hulls = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    valid_points = [(p1[3*j], p1[3*j+1]) for j in range(len(p1)//3) if p1[3*j] != 0 and p1[3*j+1] != 0]
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
                plt.legend([str(i+1) for i in range(len(people))])
                fig.canvas.mpl_connect('button_press_event', on_click)
                plt.show()

                if selected_person_idx is not None:
                    current_pos = np.array(people[selected_person_idx]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    detected = True
                    right_person = True
                else:
                    detected = True
                    right_person = False

        # 현재 포즈 데이터의 길이가 keypoint_count와 다른 경우 패딩
        if len(current_pos) < keypoint_count:
            current_pos = np.pad(current_pos, (0, keypoint_count - len(current_pos)), 'constant')
        elif len(current_pos) > keypoint_count:
            keypoint_count = len(current_pos)
            # 이전 프레임들의 데이터도 새로운 keypoint_count에 맞춰 패딩
            pos1 = [np.pad(p, (0, keypoint_count - len(p)), 'constant') for p in pos1]

        pos1.append(current_pos)

    avg_min_avg = total_min_avg / count_min_avg if count_min_avg > 0 else 0
    print(f"\nAverage min_avg: {avg_min_avg:.2f}")

    return np.array(pos1), json_files, pre_tracking_data, avg_min_avg

def animate_pre_post_tracking(pre_tracking_data, post_tracking_data, folder_name, frame_step=10, interval=100):
    fig, (pre_ax, post_ax) = plt.subplots(1, 2, figsize=(15, 5))
    
    def update(frame):
        pre_ax.clear()
        post_ax.clear()
        pre_ax.set_title(f'Pre-Tracking: {folder_name}')
        pre_ax.set_xlim([0, 4000])
        pre_ax.set_ylim([-3000, 0])
        post_ax.set_title(f'Post-Tracking: {folder_name}')
        post_ax.set_xlim([0, 4000])
        post_ax.set_ylim([-3000, 0])

        # Pre-tracking data
        if frame < len(pre_tracking_data):
            people = pre_tracking_data[frame]
            hulls = []
            for person in people:
                x_data = person['pose_keypoints_2d'][0::3]
                y_data = person['pose_keypoints_2d'][1::3]
                valid_points = [(x, y) for x, y in zip(x_data, y_data) if x != 0 and y != 0]
                if len(valid_points) >= 3:
                    hull = ConvexHull(valid_points)
                    hulls.append(hull)
                    pre_ax.plot([p[0] for p in valid_points], [-p[1] for p in valid_points], 'o')
            for hull in hulls:
                for simplex in hull.simplices:
                    pre_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')

        # Post-tracking data
        if frame < post_tracking_data.shape[0]:
            x_data = post_tracking_data[frame, 0::3]
            y_data = post_tracking_data[frame, 1::3]
            valid_points = [(x, y) for x, y in zip(x_data, y_data) if x != 0 and y != 0]
            if len(valid_points) >= 3:
                hull = ConvexHull(valid_points)
                post_ax.plot([p[0] for p in valid_points], [-p[1] for p in valid_points], 'bo')
                for simplex in hull.simplices:
                    post_ax.plot(hull.points[simplex, 0], -hull.points[simplex, 1], 'k-')

    max_frames = max(len(pre_tracking_data), post_tracking_data.shape[0])
    frames = list(range(0, max_frames, frame_step))
    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # 애니메이션의 총 지속 시간 계산
    total_duration = len(frames) * interval / 1000  # 초 단위
    
    # 애니메이션이 끝난 후 창 닫기
    plt.pause(total_duration + 1)  # 애니메이션 시간 + 1초 대기
    plt.close(fig)

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
    return save_folder

if __name__ == "__main__":
    base_path = r'C:\Users\5W555A\Desktop\FFF\pose2sim-w-Marker-Augmenter-Sync\Pose2Sim\S00_Demo_BatchSession\S00_P00_SingleParticipant'
    json_folders = find_json_folders(base_path)

    print(f"Found {len(json_folders)} folders with JSON files")

    all_avg_min_avg = []
    save_folders = []

    for folder in json_folders:
        print(f"\nProcessing folder: {folder}")
        pos1, json_files, pre_tracking_data, avg_min_avg = track_person(folder)
        if pos1 is None:
            print(f"Skipping folder {folder} due to tracking failure")
            continue
        
        all_avg_min_avg.append(avg_min_avg)
        save_folder = save_data(pos1, json_files, folder)
        save_folders.append(save_folder)

        print(f"\nStarting animation for folder: {folder}")
        print("The animation window will close automatically when finished.")
        animate_pre_post_tracking(pre_tracking_data, pos1, os.path.basename(folder), frame_step=50, interval=30)
        print("Animation completed for this folder.")
    
    print("\nAll processing completed.")
    print(f"Average min_avg across all folders: {np.mean(all_avg_min_avg):.2f}")
    print(f"Processed data saved to: {save_folders}")
    print(f"Processed data saved to: {save_folders}")