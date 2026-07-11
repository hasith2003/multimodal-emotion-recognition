import os
import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Path Configuration
WORKSPACE_DIR = r"D:\Emotion"
MELD_RAW_DIR = os.path.join(WORKSPACE_DIR, "MELD.Raw", "MELD.Raw")

DATASET_SPLITS = {
    "train": {
        "csv": os.path.join(MELD_RAW_DIR, "train_sent_emo.csv"),
        "videos": os.path.join(MELD_RAW_DIR, "train", "train_splits")
    },
    "dev": {
        "csv": os.path.join(MELD_RAW_DIR, "dev_sent_emo.csv"),
        "videos": os.path.join(MELD_RAW_DIR, "dev", "dev_splits_complete")
    },
    "test": {
        "csv": os.path.join(MELD_RAW_DIR, "test_sent_emo.csv"),
        "videos": os.path.join(MELD_RAW_DIR, "test", "output_repeated_splits_test")
    }
}

# Output Paths for Features
FEATURES_VIDEO_DIR = os.path.join(WORKSPACE_DIR, "meld_features")
FEATURES_AUDIO_DIR = os.path.join(WORKSPACE_DIR, "meld_features_audio")

# Model Paths & Keys
FABNET_WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "release_bmvc_fabnet", "release", "aflw_4views.pth")
SAVED_MODEL_PATH = os.path.join(WORKSPACE_DIR, "trimodal_emotion_model.pth")

# Label Mapping
LABEL_MAP = {
    'neutral': 0,
    'joy': 1,
    'sadness': 2,
    'anger': 3,
    'surprise': 4,
    'fear': 5,
    'disgust': 6
}

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 5
MAX_TEXT_LEN = 128
