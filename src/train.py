import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

# Add workspace to system path so we can import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    DEVICE, DATASET_SPLITS, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    SAVED_MODEL_PATH, LABEL_MAP
)
from src.dataset import MeldDataset, pad_collate_fn
from src.models import Trimodal_SSE_FT, FocalLoss


def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}")
    for batch in progress_bar:
        if batch is None:
            continue
            
        audios, text_ids, text_masks, videos, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits = model(audios, text_ids, text_masks, videos)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct / total:.4f}"
        })
        
    avg_loss = total_loss / len(loader)
    acc = correct / total
    print(f"Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Train Acc: {acc * 100:.2f}%")
    return avg_loss, acc


def evaluate(model, loader, device, split_name):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    target_names = list(LABEL_MAP.keys())
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            if batch is None:
                continue
                
            audios, text_ids, text_masks, videos, labels = [b.to(device) for b in batch]
            logits = model(audios, text_ids, text_masks, videos)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = (correct / total) * 100
    print(f"\nFinal {split_name.upper()} Accuracy: {accuracy:.2f}%")
    print(f"\nDetailed {split_name.upper()} Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Trimodal SSE-FT Emotion Recognition Training and Evaluation")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation using saved weights")
    parser.add_argument("--model_path", type=str, default=SAVED_MODEL_PATH, help="Path to load/save model weights")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce", help="Loss function to use (ce = CrossEntropy, focal = Focal Loss)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    args = parser.parse_args()

    print(f"Initializing Trimodal pipeline on {DEVICE}...")

    # Initialize model
    model = Trimodal_SSE_FT(num_classes=len(LABEL_MAP)).to(DEVICE)

    if args.eval_only:
        if not os.path.exists(args.model_path):
            print(f"Error: Model checkpoint not found at {args.model_path}")
            sys.exit(1)
            
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        
        # Load validation and test loaders
        dev_dataset = MeldDataset(
            csv_path=DATASET_SPLITS["dev"]["csv"],
            video_dir=DATASET_SPLITS["dev"]["videos"].replace(DATASET_SPLITS["dev"]["videos"], "./meld_features/dev"),
            audio_dir=DATASET_SPLITS["dev"]["videos"].replace(DATASET_SPLITS["dev"]["videos"], "./meld_features_audio/dev")
        )
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=pad_collate_fn)
        
        test_dataset = MeldDataset(
            csv_path=DATASET_SPLITS["test"]["csv"],
            video_dir=DATASET_SPLITS["test"]["videos"].replace(DATASET_SPLITS["test"]["videos"], "./meld_features/test"),
            audio_dir=DATASET_SPLITS["test"]["videos"].replace(DATASET_SPLITS["test"]["videos"], "./meld_features_audio/test")
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=pad_collate_fn)
        
        evaluate(model, dev_loader, DEVICE, "Validation (Dev)")
        evaluate(model, test_loader, DEVICE, "Test")
        
    else:
        # Load datasets
        print("Loading Training Dataset...")
        train_dataset = MeldDataset(
            csv_path=DATASET_SPLITS["train"]["csv"],
            video_dir=DATASET_SPLITS["train"]["videos"].replace(DATASET_SPLITS["train"]["videos"], "./meld_features/train"),
            audio_dir=DATASET_SPLITS["train"]["videos"].replace(DATASET_SPLITS["train"]["videos"], "./meld_features_audio/train")
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
        
        print("Loading Validation Dataset...")
        dev_dataset = MeldDataset(
            csv_path=DATASET_SPLITS["dev"]["csv"],
            video_dir=DATASET_SPLITS["dev"]["videos"].replace(DATASET_SPLITS["dev"]["videos"], "./meld_features/dev"),
            audio_dir=DATASET_SPLITS["dev"]["videos"].replace(DATASET_SPLITS["dev"]["videos"], "./meld_features_audio/dev")
        )
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=pad_collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Loss selection
        if args.loss == "focal":
            print("Calculating Class Weights for Focal Loss...")
            class_counts = train_dataset.df['Emotion'].value_counts()
            ordered_emotions = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
            counts_array = [class_counts.get(emo, 1) for emo in ordered_emotions]
            total_samples = sum(counts_array)
            weights = [total_samples / c for c in counts_array]
            class_weights = torch.FloatTensor(weights).to(DEVICE)
            class_weights = class_weights / class_weights.sum()
            print(f"Computed Class Weights: {class_weights.cpu().numpy()}")
            criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
            print(f"Using Focal Loss (gamma={args.gamma})")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropy Loss")
        
        best_val_acc = 0.0
        
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch, args.epochs)
            val_acc = evaluate(model, dev_loader, DEVICE, f"Validation Epoch {epoch}")
            
            # Save checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.model_path)
                print(f"Saved new best model to {args.model_path} (Val Acc: {val_acc:.2f}%)")


if __name__ == "__main__":
    main()
