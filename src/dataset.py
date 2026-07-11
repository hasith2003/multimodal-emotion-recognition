import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import LABEL_MAP, MAX_TEXT_LEN


class MeldDataset(Dataset):
    def __init__(self, csv_path, video_dir, audio_dir, max_len=MAX_TEXT_LEN):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.max_len = max_len
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = LABEL_MAP

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID']
        filename = f"dia{dia_id}_utt{utt_id}.pt"
        
        video_path = os.path.join(self.video_dir, filename)
        audio_path = os.path.join(self.audio_dir, filename)
        
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return None

        # Load pre-extracted embeddings
        try:
            video_tensor = torch.load(video_path, map_location='cpu', weights_only=True)
            audio_tensor = torch.load(audio_path, map_location='cpu', weights_only=True)
        except Exception:
            return None
        
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)

        # Process textual transcript
        text = str(row['Utterance'])
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        text_ids = tokens['input_ids'].squeeze(0)
        text_mask = tokens['attention_mask'].squeeze(0)
        
        emotion = row['Emotion']
        if emotion not in self.label_map:
            return None
            
        label = self.label_map[emotion]
        
        return {
            'audio': audio_tensor,
            'text_ids': text_ids,
            'text_mask': text_mask,
            'video': video_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }


def pad_collate_fn(batch):
    # Filter out missing records
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    audios = [b['audio'] for b in batch]
    text_ids = torch.stack([b['text_ids'] for b in batch])
    text_masks = torch.stack([b['text_mask'] for b in batch])
    videos = [b['video'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    
    # Pad variable-length videos/audio sequences
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_videos = pad_sequence(videos, batch_first=True)
    
    return padded_audios, text_ids, text_masks, padded_videos, labels
