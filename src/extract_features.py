import os
import sys
import glob
import subprocess
import argparse
import numpy as np
import cv2
import torch
import torchaudio
import imageio_ffmpeg
from tqdm import tqdm
from facenet_pytorch import MTCNN
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Add workspace to system path so we can import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    DEVICE, DATASET_SPLITS, FEATURES_VIDEO_DIR, FEATURES_AUDIO_DIR,
    FABNET_WEIGHTS_PATH
)

# Import FAb-Net model architecture
fabnet_code_path = os.path.join(os.getcwd(), 'FAb-Net', 'FAb-Net', 'code')
if fabnet_code_path not in sys.path:
    sys.path.append(fabnet_code_path)

from models_multiview import FrontaliseModelMasks_wider


# --- 1. Video Feature Extraction Helper Functions ---

def load_fabnet(weights_path):
    inner_nc = 256
    num_additional_ids = 32
    model = FrontaliseModelMasks_wider(3, inner_nc=inner_nc, num_additional_ids=num_additional_ids)
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict_model'])
    model.eval()
    return model


def extract_face_from_frame(frame, mtcnn_model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn_model.detect(rgb_frame)
    
    if boxes is None or len(boxes) == 0:
        return None
    
    largest_box = None
    max_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = box
            
    x1, y1, x2, y2 = [int(b) for b in largest_box]
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    cropped_face = rgb_frame[y1:y2, x1:x2]
    try:
        resized_face = cv2.resize(cropped_face, (256, 256))
        return resized_face
    except Exception:
        return None


def process_video_to_tensor(video_path, fabnet_model, mtcnn_model, device):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip 2 out of every 3 frames to save processing time
        if frame_count % 3 != 0:
            continue
            
        face_img = extract_face_from_frame(frame, mtcnn_model)
        
        if face_img is not None:
            face_tensor = torch.from_numpy(face_img).float().permute(2, 0, 1) / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = fabnet_model.encoder(face_tensor)
                embedding = embedding.view(1, -1)
                embeddings.append(embedding.cpu())
                
    cap.release()
    
    if len(embeddings) == 0:
        return torch.zeros(1, 256)
        
    return torch.cat(embeddings, dim=0)


# --- 2. Audio Feature Extraction Helper Functions ---

def extract_audio_features(video_path, processor, model, device):
    temp_wav = "temp_audio_chunk.wav"
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, '-i', video_path, 
            '-y',
            '-ar', '16000',
            '-ac', '1',
            '-loglevel', 'error',
            temp_wav
        ], check=True)
        
        waveform, sr = torchaudio.load(temp_wav)
        audio_array = waveform.squeeze().numpy()
        
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state
            
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return hidden_states.cpu()
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return torch.zeros(1, 1, 768)


# --- 3. Main Extraction Pipelines ---

def run_video_extraction():
    print(f"\n🎬 Starting Video Feature Extraction on {DEVICE}...")
    mtcnn = MTCNN(keep_all=True, device=DEVICE)
    fabnet = load_fabnet(FABNET_WEIGHTS_PATH).to(DEVICE)
    
    for split_name, paths in DATASET_SPLITS.items():
        print(f"\n=== Processing VIDEO for {split_name.upper()} Set ===")
        output_dir = os.path.join(FEATURES_VIDEO_DIR, split_name)
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = glob.glob(os.path.join(paths["videos"], "*.mp4"))
        print(f"Found {len(video_files)} videos. Extracting...")
        
        for video_path in tqdm(video_files, desc=f"Video Embeddings ({split_name})", unit="video"):
            base_name = os.path.basename(video_path).replace('.mp4', '')
            save_path = os.path.join(output_dir, f"{base_name}.pt")
            
            if os.path.exists(save_path):
                continue
                
            try:
                feature_tensor = process_video_to_tensor(video_path, fabnet, mtcnn, DEVICE)
                torch.save(feature_tensor, save_path)
            except Exception as e:
                print(f"Video {base_name} extraction failed: {e}")
                torch.save(torch.zeros(1, 256), save_path)


def run_audio_extraction():
    print(f"\n🔊 Starting Audio Feature Extraction on {DEVICE}...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
    wav2vec.eval()
    
    for split_name, paths in DATASET_SPLITS.items():
        print(f"\n=== Processing AUDIO for {split_name.upper()} Set ===")
        output_dir = os.path.join(FEATURES_AUDIO_DIR, split_name)
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = glob.glob(os.path.join(paths["videos"], "*.mp4"))
        print(f"Found {len(video_files)} videos. Extracting...")
        
        for video_path in tqdm(video_files, desc=f"Audio Embeddings ({split_name})", unit="audio"):
            base_name = os.path.basename(video_path).replace('.mp4', '')
            save_path = os.path.join(output_dir, f"{base_name}.pt")
            
            if os.path.exists(save_path):
                continue
                
            feature_tensor = extract_audio_features(video_path, processor, wav2vec, DEVICE)
            torch.save(feature_tensor, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Audio and Video embeddings for MELD dataset")
    parser.add_argument("--modality", choices=["audio", "video", "all"], default="all", help="Which feature modality to extract")
    args = parser.parse_args()
    
    if args.modality in ["video", "all"]:
        run_video_extraction()
    if args.modality in ["audio", "all"]:
        run_audio_extraction()
    print("\nFeature extraction process complete!")
