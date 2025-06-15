from transformers import CLIPProcessor, CLIPVisionModel
# generate_dataset.py
import torch
import os
import random
from tqdm import tqdm
from torchvision.transforms import ColorJitter
from feature_extractor import extract_frames_from_video, get_clip_embeddings_for_frames
from data_loader import load_video_metadata
from config import PathsConfig, FeatureExtractionConfig
from utils import time_str_to_seconds

def augment_frame(frame):
    img = frame
    if random.random() < 0.3:
        img = ColorJitter(brightness=0.4)(img)
    if random.random() < 0.3:
        img = ColorJitter(contrast=0.4)(img)
    return img

def generate_dataset(video_dir, labels_path, output_path, is_train=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_processor = CLIPProcessor.from_pretrained(FeatureExtractionConfig.CLIP_MODEL_NAME, use_fast=False)
    clip_model = CLIPVisionModel.from_pretrained(FeatureExtractionConfig.CLIP_MODEL_NAME).to(device).eval()

    metadata_list = load_video_metadata(video_dir, labels_path)
    dataset = []

    for metadata in tqdm(metadata_list, desc="Processing videos"):
        video_path = metadata["path"]
        video_id = metadata["id"]
        intro_start = metadata["gt_start_sec"]
        intro_end = metadata["gt_end_sec"]

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}. Skipping.")
            continue

        frames, timestamps = extract_frames_from_video(
            video_path, FeatureExtractionConfig.FPS_EXTRACTION, FeatureExtractionConfig.MAX_DURATION_SEC
        )
        if not frames:
            continue

        embeddings = get_clip_embeddings_for_frames(frames, clip_processor, clip_model, device)
        if embeddings is None:
            continue

        window_size = FeatureExtractionConfig.WINDOW_SIZE_SEC
        stride = FeatureExtractionConfig.WINDOW_STRIDE_SEC if is_train else window_size

        for start_idx in range(0, len(embeddings) - window_size + 1, stride):
            window_embeddings = embeddings[start_idx:start_idx + window_size]
            labels = torch.tensor([intro_start <= t <= intro_end for t in timestamps[start_idx:start_idx + window_size]], dtype=torch.float32)

            if labels.sum() > 0 or (is_train and random.random() < 0.4):
                dataset.append({
                    "video_id": video_id,
                    "window_start_frame_idx": start_idx,
                    "embeddings": window_embeddings,
                    "labels": labels
                })

        if is_train:
            intro_frames = [frame for frame, t in zip(frames, timestamps) if intro_start <= t <= intro_end]
            intro_timestamps = [t for t in timestamps if intro_start <= t <= intro_end]

            for _ in range(2):
                aug_frames = [augment_frame(frame) for frame in intro_frames]
                aug_embeddings = get_clip_embeddings_for_frames(aug_frames, clip_processor, clip_model, device)

                if aug_embeddings is None:
                    continue

                for start_idx in range(0, len(aug_embeddings) - window_size + 1, stride):
                    window_embeddings = aug_embeddings[start_idx:start_idx + window_size]
                    labels = torch.ones(window_size)

                    dataset.append({
                        "video_id": video_id,
                        "window_start_frame_idx": start_idx,
                        "embeddings": window_embeddings,
                        "labels": labels
                    })

    if dataset:
        os.makedirs(PathsConfig.PROCESSED_DATASET_DIR, exist_ok=True)
        torch.save(dataset, output_path)
        print(f"Dataset saved to {output_path}")
    else:
        print("Dataset is empty! Check video paths and labels.json.")

if __name__ == "__main__":
    generate_dataset(
        video_dir=PathsConfig.TRAIN_VIDEOS_PATH,
        labels_path=PathsConfig.TRAIN_LABELS_JSON_PATH,
        output_path=os.path.join(PathsConfig.PROCESSED_DATASET_DIR, PathsConfig.DATASET_FILENAME),
        is_train=True
    )
    generate_dataset(
        video_dir=PathsConfig.TEST_VIDEOS_PATH,
        labels_path=PathsConfig.TEST_LABELS_JSON_PATH,
        output_path=os.path.join(PathsConfig.PROCESSED_DATASET_DIR, "test_dataset2.pt"),
        is_train=False
    )