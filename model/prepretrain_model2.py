import torch
import torch.nn as nn
from transformers import WavLMModel, VideoMAEModel
from peft import LoraConfig, get_peft_model


class PrepretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int):
        super(PrepretrainModel, self).__init__()

        self.input_modality = input_modality
        self.hidden_dim = hidden_dim

        self.num_classes = num_classes

        if input_modality == "audio":
            self.encoder_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
            for p in self.encoder_model.parameters():
                p.requires_grad = False
            L = self.encoder_model.config.num_hidden_layers
            # 最後の2層はフルで学習可
            for n, p in self.encoder_model.named_parameters():
                if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            # 全LayerNormを追加で学習可に
            for n, p in self.encoder_model.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        elif input_modality == "video":
            self.encoder_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            self.encoder_model.encoder.layer = self.encoder_model.encoder.layer[:8]
            finetuning_target_modules=["query","value"]
            finetuning_r = 4
            self.encoder_model.config.num_hidden_layers = 8 
            lora_cfg = LoraConfig(
                r=finetuning_r,              
                lora_alpha=finetuning_r * 2,  
                lora_dropout=0.1,
                bias="none",
                target_modules=finetuning_target_modules
            )
            self.encoder_model = get_peft_model(self.encoder_model, lora_cfg)
            for n, p in self.encoder_model.named_parameters():
                if ("lora_" not in n):
                    p.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, x, attn_mask):
        if (self.input_modality == "audio"): 
            outputs = self.encoder_model(input_values=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state
            # m = attn_mask.unsqueeze(-1).type_as(hidden)   # [B,T,1]
            # f = (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)
            f = hidden.mean(dim=1)
        elif (self.input_modality == "video"):
            outputs = self.encoder_model(pixel_values=x, return_dict=True)
            f = outputs.last_hidden_state[:, 0, :]
        f = self.layer_norm(f)
        f = self.gelu(f)
        f = self.dropout(f)

        y = self.decoder(f)
        
        return y