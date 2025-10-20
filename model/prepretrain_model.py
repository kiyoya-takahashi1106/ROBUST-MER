import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, RobertaModel, VideoMAEModel


class PrepretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int, dropout_rate: float, dataset: str, cremad_weight_file: str):
        super(PrepretrainModel, self).__init__()
        self.input_modality = input_modality
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cremad_weight_file = f"./saved_models/prepretrain/{input_modality}/{cremad_weight_file}"

        # audio
        if (input_modality == "audio"):
            self.encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
            self.load_pretrained_layer_weights(self.cremad_weight_file)

            for p in self.encoder.parameters():
                p.requires_grad = False
            L = self.encoder.config.num_hidden_layers
            for n, p in self.encoder.named_parameters():
                if (dataset == "CREMA-D"):
                    if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                        p.requires_grad = True
                elif (dataset == "MOSI"):
                    if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                        p.requires_grad = True
            for n, p in self.encoder.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        # video
        elif (input_modality == "video"):
            self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
            self.load_pretrained_layer_weights(self.cremad_weight_file)
            
            for p in self.encoder.parameters():
                p.requires_grad = False
            L = self.encoder.config.num_hidden_layers
            for n, p in self.encoder.named_parameters():
                if (dataset == "CREMA-D"):
                    if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                        p.requires_grad = True
                elif (dataset == "MOSI"):
                    if any(f"encoder.layers.{i}." in n for i in [L - 2, L - 1]):
                        p.requires_grad = True
            for n, p in self.encoder.named_parameters():
                if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                    p.requires_grad = True

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

        self.check_pretrained_loaded(self.encoder, self.cremad_weight_file, prefix="encoder.")
        self.check_pretrained_loaded(self.layer_norm, self.cremad_weight_file, prefix="layer_norm.")


    def load_pretrained_layer_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        encoder_weights = {}
        layer_norm_weights = {}
        
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder.", "", 1)
                encoder_weights[new_key] = value
            if key.startswith("layer_norm."):
                new_key = key.replace("layer_norm.", "")
                layer_norm_weights[new_key] = value

        self.encoder.load_state_dict(encoder_weights, strict=False)
        self.layer_norm.load_state_dict(layer_norm_weights)

    
    def check_pretrained_loaded(self, model, path, prefix):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        sd_keys = [k for k in sd.keys() if k.startswith(prefix)]

        model_keys = [k for k in model.state_dict().keys()]
        matched = [k for k in sd_keys if k.replace(prefix, "", 1) in model_keys]
        print(f"Found {len(matched)} / {len(sd_keys)} matching keys for {prefix}")
        if len(matched) > 0:
            print("✅ Loaded successfully (keys matched)")
        else:
            print("❌ No matching keys — checkpoint not loaded")
    

    def forward(self, x, attn_mask):
        if (self.input_modality == "audio"):
            outputs = self.encoder(input_values=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state
            f = hidden.mean(dim=1)
            
        # dataset: CREMA-Dの時は使わない
        elif (self.input_modality == "text"):
            x = x.squeeze(1)
            attn_mask = attn_mask.squeeze(1)
            outputs = self.encoder(input_ids=x, attention_mask=attn_mask, return_dict=True)
            hidden = outputs.last_hidden_state    
            f = hidden[:, 0, :]
        
        elif (self.input_modality == "video"):
            outputs = self.encoder(x)
            hidden = outputs.last_hidden_state        
            f = hidden[:, 0, :]
        
        f = self.layer_norm(f)
        f = self.gelu(f)
        f = self.dropout(f)

        y = self.decoder(f)
        
        return y