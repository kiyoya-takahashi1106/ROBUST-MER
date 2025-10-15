import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, RobertaModel, VideoMAEModel


class PrepretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int):
        super(PrepretrainModel, self).__init__()
        self.input_modality = input_modality
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        if (input_modality == "audio"):
            self.encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
            for p in self.encoder.parameters():
                p.requires_grad = False
            L = self.encoder.config.num_hidden_layers
            for n, p in self.encoder.named_parameters():
                if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                    p.requires_grad = True
            for n, p in self.encoder.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        elif (input_modality == "text"):
            self.encoder = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
            for p in self.encoder.parameters():
                p.requires_grad = False
            L = self.encoder.config.num_hidden_layers
            for n, p in self.encoder.named_parameters():
                if any(f"encoder.layer.{i}." in n for i in [L - 2, L - 1]):
                    p.requires_grad = True
            for n, p in self.encoder.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        elif (input_modality == "video"):
            self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            for p in self.encoder.parameters():
                p.requires_grad = False
            L = self.encoder.config.num_hidden_layers
            for n, p in self.encoder.named_parameters():
                if any(f"encoder.layer.{i}." in n for i in [L - 2, L - 1]):
                    p.requires_grad = True
            for n, p in self.encoder.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, x, attn_mask):
        if (self.input_modality == "audio"):
            outputs = self.encoder(input_values=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state
            f = hidden.mean(dim=1)
            
        elif (self.input_modality == "text"):
            x = x.squeeze(1)
            attn_mask = attn_mask.squeeze(1)
            outputs = self.encoder(input_ids=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state    
            f = hidden[:, 0, :]
        
        elif (self.input_modality == "video"):
            outputs = self.encoder(x)
            hidden = outputs.last_hidden_state        
            f = hidden.mean(dim=1)
         
        f = self.layer_norm(f)
        f = self.gelu(f)
        f = self.dropout(f)

        y = self.decoder(f)
        
        return y