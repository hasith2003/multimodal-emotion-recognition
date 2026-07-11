import torch
import torch.nn as nn
from transformers import RobertaModel


class IMABlock(nn.Module):
    """
    Inter-Modality Attention Block
    Uses query projected to target sequence dimension to compute multihead cross-attention.
    """
    def __init__(self, d_query, d_target, num_heads=8):
        super(IMABlock, self).__init__()
        self.query_proj = nn.Linear(d_query, d_target)
        self.mha = nn.MultiheadAttention(embed_dim=d_target, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_target)

    def forward(self, query_cls, target_seq):
        # query_cls: (batch_size, 1, d_query)
        # target_seq: (batch_size, seq_len, d_target)
        q_projected = self.query_proj(query_cls) # (batch_size, 1, d_target)
        attn_output, _ = self.mha(query=q_projected, key=target_seq, value=target_seq)
        output = self.layer_norm(q_projected + attn_output)
        return output


class Trimodal_SSE_FT(nn.Module):
    """
    Trimodal Transformer-based Self-Supervised Feature Fusion network.
    Fuses speech features, text embeddings (RoBERTa), and video facial features (FAb-Net).
    """
    def __init__(self, num_classes=7, d_speech=768, d_text=1024, d_video=256, fusion_dim=512):
        super(Trimodal_SSE_FT, self).__init__()
        
        # Load and freeze Roberta
        self.roberta = RobertaModel.from_pretrained("roberta-large")
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Learnable CLS tokens for summarizing sequence modalities
        self.speech_cls_token = nn.Parameter(torch.randn(1, 1, d_speech))
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, d_video))
        
        # Intra-modal Transformer Encoders
        s_encoder_layer = nn.TransformerEncoderLayer(d_model=d_speech, nhead=8, batch_first=True)
        self.speech_self_attn = nn.TransformerEncoder(s_encoder_layer, num_layers=1)
        
        v_encoder_layer = nn.TransformerEncoderLayer(d_model=d_video, nhead=8, batch_first=True)
        self.video_self_attn = nn.TransformerEncoder(v_encoder_layer, num_layers=1)

        # Cross-modality attention pairs
        self.ima_s2t = IMABlock(d_query=d_speech, d_target=d_text)
        self.ima_s2v = IMABlock(d_query=d_speech, d_target=d_video)
        self.ima_t2s = IMABlock(d_query=d_text, d_target=d_speech)
        self.ima_t2v = IMABlock(d_query=d_text, d_target=d_video)
        self.ima_v2s = IMABlock(d_query=d_video, d_target=d_speech)
        self.ima_v2t = IMABlock(d_query=d_video, d_target=d_text)

        # Projection heads for fusion
        self.proj_s2t = nn.Linear(d_text, fusion_dim)
        self.proj_s2v = nn.Linear(d_video, fusion_dim)
        self.proj_t2s = nn.Linear(d_speech, fusion_dim)
        self.proj_t2v = nn.Linear(d_video, fusion_dim)
        self.proj_v2s = nn.Linear(d_speech, fusion_dim)
        self.proj_v2t = nn.Linear(d_text, fusion_dim)

        # Trimodal Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pre_extracted_audio, input_text_ids, input_text_mask, pre_extracted_video):
        batch_size = pre_extracted_audio.size(0)

        # 1. Intra-Modal Summarization
        # Audio
        cls_s = self.speech_cls_token.expand(batch_size, -1, -1)
        speech_seq = torch.cat((cls_s, pre_extracted_audio), dim=1)
        speech_seq = self.speech_self_attn(speech_seq)
        speech_cls = speech_seq[:, 0:1, :] 
        
        # Video
        cls_v = self.video_cls_token.expand(batch_size, -1, -1)
        video_seq = torch.cat((cls_v, pre_extracted_video), dim=1)
        video_seq = self.video_self_attn(video_seq)
        video_cls = video_seq[:, 0:1, :] 

        # Text (Compute RoBERTa representations on the fly)
        text_out = self.roberta(input_ids=input_text_ids, attention_mask=input_text_mask).last_hidden_state
        text_cls = text_out[:, 0:1, :] 

        # 2. Inter-Modality Attention (IMA)
        out_s2t = self.ima_s2t(speech_cls, text_out) 
        out_s2v = self.ima_s2v(speech_cls, video_seq) 
        out_t2s = self.ima_t2s(text_cls, speech_seq) 
        out_t2v = self.ima_t2v(text_cls, video_seq) 
        out_v2s = self.ima_v2s(video_cls, speech_seq) 
        out_v2t = self.ima_v2t(video_cls, text_out) 

        # 3. Dimension Projection & Hadamard Products
        speech_final = torch.mul(self.proj_s2t(out_s2t.squeeze(1)), self.proj_s2v(out_s2v.squeeze(1)))
        text_final   = torch.mul(self.proj_t2s(out_t2s.squeeze(1)), self.proj_t2v(out_t2v.squeeze(1)))
        video_final  = torch.mul(self.proj_v2s(out_v2s.squeeze(1)), self.proj_v2t(out_v2t.squeeze(1)))

        # 4. Concatenation & Classification
        combined_features = torch.cat((speech_final, text_final, video_final), dim=1)
        logits = self.classifier(combined_features) 
        return logits
