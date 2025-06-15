import os

class PathsConfig:
    TRAIN_VIDEOS_PATH = "data_train_short"
    TEST_VIDEOS_PATH = "data_test_short"
    TRAIN_LABELS_JSON_PATH = os.path.join(TRAIN_VIDEOS_PATH, "labels.json")
    TEST_LABELS_JSON_PATH = os.path.join(TEST_VIDEOS_PATH, "labels.json")
    PROCESSED_DATASET_DIR = "processed_data"
    DATASET_FILENAME = "train_dataset2.pt"
    MODELS_PATH = "models"

class FeatureExtractionConfig:
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    FPS_EXTRACTION = 1
    WINDOW_SIZE_SEC = 60
    WINDOW_STRIDE_SEC = 30
    MAX_DURATION_SEC = 180

class TrainingConfig:
    RANDOM_SEED = 42