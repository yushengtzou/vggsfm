import pandas as pd

def parse_images_txt(file_path):
    """
    解析 images.txt 檔案，提取相機姿態資訊。
    """
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳過前四行的註解和統計資訊
    i = 4
    while i < len(lines):
        # 這是包含姿態資訊的行
        pose_line = lines[i].strip().split()
        image_id = int(pose_line[0])
        qw, qx, qy, qz = map(float, pose_line[1:5])
        tx, ty, tz = map(float, pose_line[5:8])
        camera_id = int(pose_line[8])
        name = pose_line[9]

        poses.append({
            'name': name,
            'image_id': image_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz
        })

        # 跳過下一行的 2D points 資訊
        i += 2

    return pd.DataFrame(poses)


import json
from typing import Optional

def parse_frame_annotations(file_path: str) -> Optional[pd.DataFrame]:
    """
    Parses a nested JSON file of frame annotations and flattens it into a DataFrame.

    This function specifically uses pandas.json_normalize to handle the nested
    structure, creating separate columns for nested data like 'image.path' and 'viewpoint.R'.

    Args:
        file_path (str): The path to the 'frame_annotations.json' file.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the flattened data,
                                or None if an error occurs.
    """
    try:
        # Open and load the file's content first
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use json_normalize to flatten the data into a clean table
        ground_truth_df = pd.json_normalize(data)
        return ground_truth_df

    except FileNotFoundError:
        print(f"❌ Error: The file was not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: The file at '{file_path}' is not a valid JSON.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # --- 1. Define your target and load data ---
    PREDICTED_PATH = '/media/daniel/storage1/2.research/1.3d-reconstruct/results/CO3D/110_13051_23361/text/images.txt'
    GT_PATH = '/media/daniel/storage1/2.research/1.3d-reconstruct/dataset/CO3D/apple/frame_annotations.json'
    TARGET_SEQUENCE = "110_13051_23361" # The sequence name from your file path

    predicted_df = parse_images_txt(PREDICTED_PATH)
    ground_truth_df = parse_frame_annotations(GT_PATH)

    if predicted_df is not None and ground_truth_df is not None:
        # --- 2. Filter Ground Truth to the specific sequence ---
        print(f"\n--- Filtering for sequence: {TARGET_SEQUENCE} ---")
        filtered_gt_df = ground_truth_df[ground_truth_df['sequence_name'] == TARGET_SEQUENCE].copy()
        print(f"Found {len(filtered_gt_df)} matching frames in ground truth.")

        # --- 3. Prepare the key for merging ---
        filtered_gt_df['filename'] = filtered_gt_df['image.path'].str.split('/').str[-1]

        # --- 4. Perform a CLEAN merge ---
        merged_df = pd.merge(
            predicted_df,
            filtered_gt_df,
            left_on='name',
            right_on='filename'
        )

        # --- 5. Sort the final result by frame number ---
        print("\n--- Sorting by frame number to ensure correct sequence ---")
        merged_df.sort_values(by='frame_number', inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        print("\n✅ Final, Cleaned, and Sorted Data!")
        print(merged_df[['name', 'frame_number', 'qw', 'tx', 'ty', 'tz', 'viewpoint.R', 'viewpoint.T']].head())
        # print(merged_df[['name', 'frame_number', 'qw', 'tx', 'viewpoint.R', 'viewpoint.T']].head())

        merged_df.to_csv('merged_data.csv', index=False)