import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
from retinaface import RetinaFace

try:
    img = cv2.imread(r"D:\Emotion\MELD.Raw\MELD.Raw\train\train_splits\dia0_utt0.mp4")
    # Actually, we can just pass a dummy numpy array
    import numpy as np
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    faces = RetinaFace.detect_faces(dummy_img)
    print("SUCCESS: RetinaFace detected faces without crashing")
except Exception as e:
    print(f"FAILED: {e}")
