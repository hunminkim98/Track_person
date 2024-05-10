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
    # JSON 파일을 읽기
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def natural_sort_key(s):
    # 자연스러운 정렬을 위한 키 생성 함수
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def track_person(folder_path):
    # JSON 파일을 정렬된 순서로 로드
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key)
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []
    counter = 0

    for i, file_name in enumerate(json_files):
        # print(f"Processing file {i + 1}/{len(json_files)}: {file_name}")
        data = read_json_file(os.path.join(folder_path, file_name))
        people = data.get('people', [])
        # print(f"Number of people detected: {len(people)}")

        # 처음으로 사람을 탐지하는 경우
        if not detected and not right_person:
            if not people:
                print("No people detected in the frame.")
                pos1.append(np.zeros(75))
            elif len(people) >= 2:
                # 여러 명이 탐지된 경우
                fig, ax = plt.subplots(figsize=(12, 8))  # 창 크기를 조정합니다.
                av1 = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    hull = ConvexHull(p1[::3], p1[1::3])
                    av1.append(hull.volume)
                    ax.plot(p1[::3], -p1[1::3], 'o')
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                plt.legend([str(i+1) for i in range(len(people))])
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
                # 한 명이 탐지된 경우
                print("Single person detected, automatically tracking this person.")
                pos1.append(people[0]['pose_keypoints_2d'])
                data_to_track = pos1[-1]
                detected = True
                right_person = True
        elif detected and right_person:
            # 선택된 사람을 추적하는 경우
            if people:
                mae = []
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    x0, y0 = np.array(data_to_track[::3]), np.array(data_to_track[1::3])
                    x1, y1 = p1[::3], p1[1::3]
                    # print(f"x0: {x0}")
                    # print(f"y0: {y0}")
                    # print(f"x1: {x1}")
                    # print(f"y1: {y1}")
                    valid = np.where((x0 != 0) & (y0 != 0) & (x1 != 0) & (y1 != 0))[0]
                    # print(f"valid: {valid}")
                    if valid.size == 0:
                        x_mae, y_mae = float('inf'), float('inf')
                    else:
                        x_mae = np.mean(np.abs(x0[valid] - x1[valid]))
                        y_mae = np.mean(np.abs(y0[valid] - y1[valid]))
                    # print(f"x_mae: {x_mae}")
                    # print(f"y_mae: {y_mae}")
                    mae.append(np.mean([x_mae, y_mae]))
                # print(f"MAE: {mae}")
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
            # 사람을 다시 선택해야 하는 경우
            if not people:
                pos1.append(np.zeros(75))
                detected = False
                right_person = False
            else:
                fig, ax = plt.subplots(figsize=(12, 8))  # 창 크기를 조정합니다.
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    ax.plot(p1[::3], -p1[1::3], 'o')
                ax.set_xlim([0, 4000])
                ax.set_ylim([-3000, 0])
                plt.legend([str(i+1) for i in range(len(people))])
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

        counter += 1

    return np.array(pos1), json_files

def plot_data(data):
    for i in range(0, data.shape[0], 20):  # Change here
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
    base_path = r'C:\Users\5W555A\Desktop\240423_liun\cam\pose2sim\Pose2Sim\S01_Demo_SingleTrial\pose'
    folders = load_json_files(base_path)
    for folder in folders:
        pos1, json_files = track_person(folder)
        if pos1 is None:
            continue
        plot_data(pos1)
        save_data(pos1, json_files, folder)
