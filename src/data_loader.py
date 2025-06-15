import os
import json
from utils import time_str_to_seconds

def load_video_metadata(videos_base_path, labels_path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    video_metadata_list = []
    for video_id, info in labels_data.items():
        gt_start_str = info.get('start')
        gt_end_str = info.get('end')
        if not gt_start_str or not gt_end_str:
            continue
        start_sec = time_str_to_seconds(gt_start_str)
        end_sec = time_str_to_seconds(gt_end_str)
        if end_sec <= start_sec:
            print(f'Skipping video {video_id}, incorrect timestamp')
            continue
        video_metadata_list.append({
            "id": video_id,
            "path": os.path.join(videos_base_path, video_id, f"{video_id}.mp4"),
            "gt_start_sec": start_sec,
            "gt_end_sec": end_sec,
        })
    return video_metadata_list