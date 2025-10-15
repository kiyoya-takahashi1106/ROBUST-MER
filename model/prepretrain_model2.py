import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, RobertaModel, VideoMAEModel
from peft import LoraConfig, get_peft_model


class PrepretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int):
        super(PrepretrainModel, self).__init__()
        self.input_modality = input_modality
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        if input_modality == "audio":
            self.encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
            
            # 最初は全てのパラメータをフリーズ
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            L = self.encoder.config.num_hidden_layers
            
            # 最後の2層はフルで学習可
            for n, p in self.encoder.named_parameters():
                if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                    p.requires_grad = True
            
            # 全LayerNormを追加で学習可に
            for n, p in self.encoder.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        elif input_modality == "text":
            self.encoder_model = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)

        elif input_modality == "video":
            self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            
            # エンコーダーを8層に削減
            self.encoder.encoder.layer = self.encoder.encoder.layer[:8]
            self.encoder.config.num_hidden_layers = 8
            
            # LoRA設定
            finetuning_target_modules = ["query", "value"]
            finetuning_r = 4
            lora_cfg = LoraConfig(
                r=finetuning_r,
                lora_alpha=finetuning_r * 2,
                lora_dropout=0.1,
                bias="none",
                target_modules=finetuning_target_modules
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            
            # LoRA以外のパラメータをフリーズ
            for n, p in self.encoder.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, attn_mask):
        if self.input_modality == "audio":
            hidden = self.encoder(x, attention_mask=attn_mask).last_hidden_state
            
            # WavLMは約320倍ダウンサンプリングするため、マスクも調整
            seq_len = hidden.size(1)
            attn_mask = F.adaptive_avg_pool1d(
                attn_mask.float().unsqueeze(1), 
                seq_len
            ).squeeze(1) > 0.5
            attn_mask = attn_mask.float()
            
            m = attn_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            f = (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)
            
        elif (self.input_modality == "text"):
            outputs = self.encoder_model(input_ids=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state
            f = hidden[:, 0, :]
        
        elif self.input_modality == "video":
            hidden = self.encoder(x).last_hidden_state
            f = hidden[:, 0, :]  # CLSトークン
        
        f = self.layer_norm(f)
        f = self.gelu(f)
        f = self.dropout(f)
        y = self.decoder(f)
        
        return y