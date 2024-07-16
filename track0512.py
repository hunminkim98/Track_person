import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import re

def load_json_files(base_path):
    # Load subfolders
    folders = [os.path.join(base_path, d) for d in sorted(os.listdir(base_path)) 
               if os.path.isdir(os.path.join(base_path, d)) and re.search(r'json\d+$', d)]
    return folders

def read_json_file(file_path):
    print(f"Reading file from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def natural_sort_key(s):
    # 자연스러운 정렬을 위한 키 생성 함수
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def track_person(folder_path):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []

    for i, file_name in enumerate(json_files):
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])

        if not detected and not right_person:
            if not people:
                print("No people detected in the frame.")
                pos1.append(np.zeros(75))
            elif len(people) >= 2:
                fig, ax = plt.subplots(figsize=(12, 8))
                av1 = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    valid_points = [(p1[3 * j], p1[3 * j + 1]) for j in range(len(p1) // 3) if p1[3 * j] != 0 and p1[3 * j + 1] != 0]
                    if len(valid_points) >= 3:
                        try:
                            hull = ConvexHull(valid_points)
                            av1.append(hull.volume)
                            x, y = zip(*valid_points)
                            ax.plot(x, -np.array(y), 'o')
                        except Exception as e:
                            print(f"Error in ConvexHull calculation: {e}")
                    else:
                        print("Not enough valid points for ConvexHull")
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                plt.legend([str(i + 1) for i in range(len(people))])
                plt.show()
                num_to_track = int(input("Select num or 100 to skip: "))
                plt.close()

                if num_to_track == 100:
                    pos1.append(np.zeros(75))
                    detected = True
                    right_person = False
                else:
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
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    ax.plot(p1[::3], -p1[1::3], 'o')
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                plt.legend([str(i + 1) for i in range(len(people))])
                plt.show()
                num_to_track = int(input("Select num or 100 to skip: "))
                plt.close()

                if num_to_track == 100:
                    pos1.append(np.zeros(75))
                    detected = True
                    right_person = False
                else:
                    pos1.append(people[num_to_track - 1]['pose_keypoints_2d'])
                    data_to_track = pos1[-1]
                    detected = True
                    right_person = True

    return np.array(pos1), json_files

def plot_data(data):
    for i in range(0, data.shape[0], 20):
        plt.clf()
        x_data = data[i, 0::3]
        y_data = data[i, 1::3]
        if len(x_data) == len(y_data):
            plt.plot(x_data, -y_data, 'bo')
            plt.xlim([0, 4000])
            plt.ylim([-3000, 0])
            plt.pause(0.0001)
        else:
            print(f"Skipping frame {i} due to length mismatch: x_data length = {len(x_data)}, y_data length = {len(y_data)}")
    plt.close()

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
    base_path = r'C:\Users\5W555A\Desktop\tracking\Person_tracking\demo'
    folders = load_json_files(base_path)
    for folder in folders:
        pos1, json_files = track_person(folder)
        if pos1 is None:
            continue
        plot_data(pos1)
        save_data(pos1, json_files, folder)
