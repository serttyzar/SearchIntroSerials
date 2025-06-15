import os
from transformers import CLIPProcessor, CLIPVisionModel
import torch
import numpy as np
from scipy.ndimage import label as scipy_label
from feature_extractor import extract_frames_from_video, get_clip_embeddings_for_frames
from model import IntroDetectionTransformer

def predict_intro(video_path, model_path, device="cpu", max_duration_sec=180):
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return None

    try:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        model = IntroDetectionTransformer(d_model=768, n_heads=12, n_layers=6, class_weights = torch.tensor([1.0, 2.0])).to(device)
        # для v3 требуется так же class_weights = torch.tensor([1.0, 2.0])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        boost_self_transition = 1.5  # Увеличиваем вероятность остаться в состоянии "заставка"
        penalize_exit_transition = -1.5 # Уменьшаем вероятность выйти из состояния "заставка"

        model.crf.transitions.data[1, 1] += boost_self_transition
        model.crf.transitions.data[1, 0] += penalize_exit_transition
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

    frames, timestamps = extract_frames_from_video(video_path, 1, max_duration_sec)
    if not frames:
        print("No frames extracted.")
        return None

    embeddings = get_clip_embeddings_for_frames(frames, clip_processor, clip_model, device)
    if embeddings is None:
        print("Failed to extract embeddings.")
        return None

    all_predictions = []
    window_size = 60
    for i in range(0, len(embeddings) - window_size + 1, window_size):
        window = embeddings[i:i+window_size].unsqueeze(0).to(device)
        pred = model(window)
        all_predictions.extend(pred[0])

    final_labels = np.array(all_predictions)
    labeled_segments, num_groups = scipy_label(final_labels)

    if num_groups == 0:
        print("Intro not detected.")
        return None

    max_len = 0
    best_group = -1
    for group in range(1, num_groups + 1):
        length = np.sum(labeled_segments == group)
        if length > max_len:
            max_len, best_group = length, group

    if max_len < 3:
        print("Detected segment too short.")
        return None

    indices = np.where(labeled_segments == best_group)[0]
    return timestamps[indices[0]], timestamps[indices[-1]]

# Тестировка модели
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "models\modelv3.pt"
    video_path = "data_test_short\-220020068_456249358\-220020068_456249358.mp4"

    result = predict_intro(video_path, model_path, device)
    if result:
        start_sec, end_sec = result
        print(f"Заставка обнаружена: {start_sec:.2f}с - {end_sec:.2f}с")
    else:
        print("Заставка не обнаружена")

if __name__ == "__main__":
    main()