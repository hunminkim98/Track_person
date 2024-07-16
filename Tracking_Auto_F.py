import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
import re
from tqdm import tqdm

# Find json folder from base_path (.json file should include)
def find_json_folders(base_path):
    json_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(file.endswith('.json') for file in files): # find .json file in the base_path
            json_folders.append(root) # add folder if there's json file.
    return json_folders

# Read json file
def read_json_file(file_path):
    with open(file_path, 'r') as f:  # open json file in read mode
        data = json.load(f)
    return data

# Sort files using number
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Calculate area of Convex Hull
def calculate_hull_area(points):
    if len(points) < 3: # if points less than 3, ignore
        return 0
    try:
        hull = ConvexHull(points)
        return hull.area
    except Exception: 
        return 0
    
# Calculate distance from center of image.
def calculate_distance_from_center(points, image_size):
    center_x, center_y = image_size[0] / 2, image_size[1] / 2 # half of the image size
    person_center = np.mean(points, axis=0) # mean of valid person points
    return np.sqrt((person_center[0] - center_x)**2 + (person_center[1] - center_y)**2) # Euclidean distance

# Find proper person for tracking in the large amount of data
def select_person_automatically(people, image_size):
    max_score = float('-inf') # initial score is negative inf for find max score 
    selected_person_idx = None
    
    for i, person in enumerate(people):
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3) # N,3 2d
        valid_points = keypoints[np.all(keypoints[:, :2] != 0, axis=1)][:, :2] # select valid point if x,y are not 0 (exclude confidence but it might be useful consider confidence threshold)
        
        if len(valid_points) < 3: # exclude if valid body points less than 3
            continue
        
        hull_area = calculate_hull_area(valid_points) # Convex area
        center_distance = calculate_distance_from_center(valid_points, image_size) # Distance from center of image
        
        # Normalize scores
        normalized_area = hull_area / (image_size[0] * image_size[1]) # normalize Convex area
        normalized_distance = 1 - (center_distance / (np.sqrt(image_size[0]**2 + image_size[1]**2) / 2)) # normalize center of image
        
        # Calculate combined score (you can adjust weights if needed)
        score = normalized_area * 0.7 + normalized_distance * 0.3 # I think Convex area has more higher priority due to interesting person's position is closer to camera in most cases.
        
        if score > max_score:
            max_score = score
            selected_person_idx = i
    
    return selected_person_idx

# Track interesting person
def track_person(folder_path):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')], key=natural_sort_key) # sort json file using number
    if not json_files:
        print(f"No files found in the specified directory: {folder_path}")
        return None, None, None, None

    detected = False
    right_person = False
    data_to_track = None
    pos1 = []
    pre_tracking_data = []
    total_min_avg = 0
    count_min_avg = 0
    keypoint_count = 0
    image_size = (3840, 2160)  # image size should adjust for precise person tracking

    for i, file_name in tqdm(enumerate(json_files), total=len(json_files), desc="Processing files", ncols=100): # processing bar
        data = read_json_file(os.path.join(folder_path, file_name)) 
        people = data.get('people', []) # find 'people'
        pre_tracking_data.append(people)

        if i == 0:
            if people: # if detected person in the first frame
                keypoint_count = len(people[0]['pose_keypoints_2d']) # grab number of keypoints
                print(f"Detected {keypoint_count} keypoints in the first frame\n")
            else:
                print("No people detected in the first frame")
                keypoint_count = 75  # Default to 25 keypoints * 3 (x, y, confidence)

        current_pos = np.zeros(keypoint_count)

        if not detected and not right_person:
            if not people:
                pos1.append(current_pos)
            else:
                selected_person_idx = select_person_automatically(people, image_size) # find interesting person automatically
                if selected_person_idx is not None:
                    current_pos = np.array(people[selected_person_idx]['pose_keypoints_2d'])
                    data_to_track = current_pos
                    detected = True
                    right_person = True
                    print(f"Automatically selected person {selected_person_idx + 1}")
                else:
                    print("No suitable person found for tracking")
        elif detected and right_person:
            if people:
                mae = [] # Mean Absulte Error
                for k, person in enumerate(people):
                    p1 = np.array(person['pose_keypoints_2d'])
                    x0, y0 = np.array(data_to_track[::3]), np.array(data_to_track[1::3])
                    x1, y1 = p1[::3], p1[1::3]
                    valid = np.where((x0 != 0) & (y0 != 0) & (x1 != 0) & (y1 != 0))[0]
                    if valid.size == 0:
                        x_mae, y_mae = float('inf'), float('inf') # mae will set 0 if x,y not valid
                    else:
                        x_mae = np.mean(np.abs(x0[valid] - x1[valid]))
                        y_mae = np.mean(np.abs(y0[valid] - y1[valid]))
                    mae.append(np.mean([x_mae, y_mae])) # calculate mean of MAE
                min_avg, I1 = min((val, idx) for (idx, val) in enumerate(mae)) # Select person who has the most minimun mean of MAE
                if min_avg > 100: # if min of MAE is too large, ignore(the threshold hard coded now)
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

        ## This padding algorithm might be unstable. So it should improved
        # if length of current pose shorter than previous one, pad current pose
        if len(current_pos) < keypoint_count:
            current_pos = np.pad(current_pos, (0, keypoint_count - len(current_pos)), 'constant')

        # if length of current pose longer than previous one, pad previous poses
        elif len(current_pos) > keypoint_count:
            keypoint_count = len(current_pos)
            pos1 = [np.pad(p, (0, keypoint_count - len(p)), 'constant') for p in pos1]
        pos1.append(current_pos)

    avg_min_avg = total_min_avg / count_min_avg if count_min_avg > 0 else 0 # avagrage value of mean of MAE
    print(f"\nAverage min_avg: {avg_min_avg:.2f}")

    return np.array(pos1), json_files, pre_tracking_data, avg_min_avg

# Animate result for checking visually by users
def animate_pre_post_tracking(pre_tracking_data, post_tracking_data, folder_name, frame_step=10, interval=100):
    fig, (pre_ax, post_ax) = plt.subplots(1, 2, figsize=(15, 5))
    
    def update(frame):
        # x,y value should be determined from image size in the future
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
    
    # total time of animation
    total_duration = len(frames) * interval / 1000  # sec
    
    # close animation automatically when it finished
    plt.pause(total_duration + 1)
    plt.close(fig)
\
# Save the results
def save_data(data, json_files, folder_path):
    base_folder = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    tracked_folder_name = folder_name + '_tracked' # original folder name_tracking
    save_folder = os.path.join(base_folder, tracked_folder_name) # create new folder
    
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

# call functions
if __name__ == "__main__":
    base_path = r'D:\석사\석사\Validation\김리언\KLUpose' # put your parent folder
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