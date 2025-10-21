import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel  


class PrepretrainModel(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout_rate: float, pretrained_model_file: str):
        super(PrepretrainModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.pretrained_model_path = "./saved_models/prepretrain/" + "video" + "/" + pretrained_model_file

        # video
        self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.load_pretrained_layer_weights(self.pretrained_model_path)

        for p in self.encoder.parameters():
            p.requires_grad = False
        L = self.encoder.config.num_hidden_layers
        for n, p in self.encoder.named_parameters():
            if any(f"encoder.layer.{i}." in n for i in [L -2, L - 1]):
                p.requires_grad = True
        for n, p in self.encoder.named_parameters():
            if any(k in n.lower() for k in ["layer_norm", "layernorm", "final_layer_norm"]):
                p.requires_grad = True

        self.check_pretrained_loaded(self.encoder, self.pretrained_model_path, prefix="encoder.")
        self.check_pretrained_loaded(self.layer_norm, self.pretrained_model_path, prefix="layer_norm.")

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.encoder.parameters())
        print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")



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
        outputs = self.encoder(x)
        hidden = outputs.last_hidden_state        
        f = hidden[:, 0, :]
    
        f = self.layer_norm(f)
        f = self.gelu(f)
        f = self.dropout(f)

        y = self.decoder(f)
        return y