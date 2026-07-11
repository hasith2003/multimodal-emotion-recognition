import os
import sys
import torch
from torch.utils.data import DataLoader

# Add workspace to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEVICE, DATASET_SPLITS
from src.dataset import MeldDataset, pad_collate_fn
from src.models import Trimodal_SSE_FT

def main():
    print("--- Trimodal Pipeline Dry Run Verification ---")
    
    # 1. Dataset verification
    print("Loading test split dataset...")
    test_dataset = MeldDataset(
        csv_path=DATASET_SPLITS["test"]["csv"],
        video_dir="./meld_features/test",
        audio_dir="./meld_features_audio/test"
    )
    
    print(f"Total entries in raw CSV: {len(test_dataset)}")
    
    # 2. Loader verification
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)
    
    # Find the first valid batch
    batch = None
    for b in test_loader:
        if b is not None:
            batch = b
            break
            
    if batch is None:
        print("Error: Could not find any valid pre-extracted samples. Make sure features are pre-extracted.")
        sys.exit(1)
        
    audios, text_ids, text_masks, videos, labels = batch
    print("\nBatch loaded successfully:")
    print(f"  Audio tensor shape: {audios.shape}")
    print(f"  Text IDs shape:     {text_ids.shape}")
    print(f"  Text Mask shape:   {text_masks.shape}")
    print(f"  Video tensor shape: {videos.shape}")
    print(f"  Labels shape:       {labels.shape}")

    # 3. Model verification
    print("\nInitializing model...")
    model = Trimodal_SSE_FT(num_classes=7).to(DEVICE)
    model.eval()
    
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(
            audios.to(DEVICE), 
            text_ids.to(DEVICE), 
            text_masks.to(DEVICE), 
            videos.to(DEVICE)
        )
        
    print(f"Logits output shape: {logits.shape}")
    if logits.shape == (audios.size(0), 7):
        print("\n[PASS] Integration Check Passed! Model and Dataset components are fully functional.")
    else:
        print("\n[FAIL] Integration Check Failed: Unexpected shape output.")
        sys.exit(1)

if __name__ == "__main__":
    main()
