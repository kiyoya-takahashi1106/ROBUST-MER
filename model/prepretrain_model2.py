import torch
import torch.nn as nn
from transformers import WavLMModel, VideoMAEModel
from peft import LoraConfig, get_peft_model


class PrepretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int):
        super(PrepretrainModel, self).__init__()

        self.input_modality = input_modality
        self.hidden_dim = hidden_dim

        self.activation = nn.ReLU()

        self.num_classes = num_classes

        if input_modality == "audio":
            self.encoder_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        elif input_modality == "video":
            self.encoder_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            lora_cfg = LoraConfig(
                r=12,              
                lora_alpha=24,   # r*2 base
                lora_dropout=0.1,
                bias="none",
                target_modules=["query","key","value","output.dense","attn.proj","qkv","proj"]
            )
            trainable = [(n, p.numel()) for n, p in self.encoder_model.named_parameters() if p.requires_grad]
            print(f"Trainable params: {sum(p.numel() for _, p in trainable):,}")
            self.encoder_model = get_peft_model(self.encoder_model, lora_cfg)
            for n, p in self.encoder_model.named_parameters():
                if ("lora_" not in n):
                    p.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, x, attn_mask):
        if (self.input_modality == "audio"): 
            outputs = self.encoder_model(input_values=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state
            f = hidden.mean(dim=1)
        elif (self.input_modality == "video"):
            outputs = self.encoder_model(pixel_values=x, return_dict=True)
            f = outputs.last_hidden_state[:, 0, :]
        f = self.layer_norm(f)
        f = nn.GELU()(f)

        y = self.decoder(f)
        
        return y