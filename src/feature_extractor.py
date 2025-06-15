import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

def extract_frames_from_video(video_path, fps_extraction=1, max_duration_sec=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video file {video_path} could not be opened. Skipping.")
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(int(video_fps / fps_extraction), 1)
    frames, timestamps = [], []

    current_frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if max_duration_sec and timestamp_sec > max_duration_sec:
            break
        if current_frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            timestamps.append(timestamp_sec)
        current_frame_idx += 1

    cap.release()
    return frames, timestamps

def get_clip_embeddings_for_frames(frames, clip_processor, clip_model, device="cpu"):
    if not frames:
        return None
    with torch.no_grad():
        inputs = clip_processor(images=frames, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(device)
        return clip_model(pixel_values=pixel_values).pooler_output.cpu()