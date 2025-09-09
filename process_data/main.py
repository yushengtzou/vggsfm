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

# --- Example of how to use it ---
# gt_df = parse_frame_annotations('path/to/your/frame_annotations.json')
# if gt_df is not None:
#     print("✅ Successfully loaded and parsed the ground truth data:")
#     print(gt_df[['image.path', 'viewpoint.R', 'viewpoint.T']].head())

if __name__ == "__main__":
    predicted_df = parse_images_txt('/media/daniel/storage1/2.research/1.3d-reconstruct/results/CO3D/110_13051_23361/text/images.txt')
    print("預測姿態 (Predicted Poses):")
    print(predicted_df.head())

    ground_truth_df = parse_frame_annotations('/media/daniel/storage1/2.research/1.3d-reconstruct/dataset/CO3D/apple/frame_annotations.json')
    print("真實姿態 (Ground Truth Poses):")
    print(ground_truth_df.head())
    print(ground_truth_df[['image.path', 'viewpoint.R', 'viewpoint.T']].head())