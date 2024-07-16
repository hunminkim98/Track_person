import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_first_frame(first_frame_data, folder_name):
    people = first_frame_data.get('people', [])
    if not people:
        print(f"No people found in the first frame of folder: {folder_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"First Frame with Person IDs - {folder_name}")

    for person in people:
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        confidence = keypoints[:, 2]
        
        valid_points = confidence > 0
        ax.scatter(x[valid_points], y[valid_points], s=20)
        
        center = np.mean(keypoints[valid_points, :2], axis=0)
        ax.text(center[0], center[1], f"ID: {person['person_id'][0]}", fontsize=12, color='red')

    ax.invert_yaxis()
    plt.show()

def get_user_input(people, folder_name):
    valid_ids = [person['person_id'][0] for person in people]
    while True:
        try:
            print(f"\nSelecting person for folder: {folder_name}")
            selected_id = int(input(f"Enter the person ID you want to track {valid_ids}: "))
            if selected_id in valid_ids:
                return selected_id
            else:
                print("Invalid ID. Please try again.")
        except ValueError:
            print("Please enter a valid integer ID.")

def find_json_folders(root_folder):
    json_folders = []
    for root, dirs, files in os.walk(root_folder):
        if any(file.endswith('.json') for file in files):
            json_folders.append(root)
    return json_folders

def filter_person_data(folder_path, selected_person_id):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    filtered_data = []

    print(f"Filtering data for selected person in {folder_path}...")
    for file_name in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for person in data.get('people', []):
            if person['person_id'][0] == selected_person_id:
                filtered_data.append({
                    "frame": file_name,
                    "keypoints": person['pose_keypoints_2d'],
                    "original_path": file_path
                })
                break

    return filtered_data

def save_filtered_data(filtered_data, folder_path, selected_person_id):
    output_folder = os.path.join(folder_path, f"person_{selected_person_id}_data")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Saving filtered data for folder: {folder_path}")
    for frame_data in tqdm(filtered_data, desc="Saving files"):
        output_file = os.path.join(output_folder, frame_data['frame'])
        with open(output_file, 'w') as f:
            json.dump({
                "version": 1.3,
                "people": [{
                    "person_id": [selected_person_id],
                    "pose_keypoints_2d": frame_data['keypoints'],
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                }]
            }, f, indent=4)

    print(f"Saved filtered data for person ID {selected_person_id} in {output_folder}")

def process_folder(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in folder: {folder_path}")
        return

    # 첫 프레임 데이터 로드
    with open(os.path.join(folder_path, json_files[0]), 'r') as f:
        first_frame_data = json.load(f)

    # 첫 프레임 plot
    plot_first_frame(first_frame_data, os.path.basename(folder_path))

    # 사용자에게 tracking할 ID 입력 받기
    selected_person_id = get_user_input(first_frame_data['people'], os.path.basename(folder_path))

    # 선택된 person_id 정보만 필터링 및 저장
    filtered_data = filter_person_data(folder_path, selected_person_id)
    save_filtered_data(filtered_data, folder_path, selected_person_id)

if __name__ == "__main__":
    root_folder = r'C:\Users\5W555A\Desktop\FFF\pose2sim-w-Marker-Augmenter-Sync\Pose2Sim\S00_Demo_BatchSession\S00_P00_SingleParticipant\S00_P00_T01_BalancingTrial\pose\kicking10_4'
    
    print("Finding folders with JSON files...")
    json_folders = find_json_folders(root_folder)
    
    if not json_folders:
        print("No folders with JSON files found in the specified directory and its subdirectories.")
        exit()

    print(f"Found {len(json_folders)} folders with JSON files.")

    for folder in json_folders:
        process_folder(folder)

    print("All folders processed successfully.")