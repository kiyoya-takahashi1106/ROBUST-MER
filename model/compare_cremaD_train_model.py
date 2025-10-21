# CREMA-D 用の 共通線形変換なしmodel (encoderは学習済み)

import torch
import torch.nn as nn
from transformers import WavLMModel, VideoMAEModel


class No_Linear_Model(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout_rate: float, dataset_name: str, audio_pretrained_model_file: str, video_pretrained_model_file: str):
        super(No_Linear_Model, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name

        audio_path = "./saved_models/prepretrain/audio/" + audio_pretrained_model_file
        video_path = "./saved_models/prepretrain/video/" + video_pretrained_model_file

        # Encoder + LN
        # 音声 (事前学習済み)
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.audio_encoder_layer_norm = nn.LayerNorm(self.hidden_dim)
        if (audio_pretrained_model_file != "test.pth"):
            self.load_pretrained_encoder_layer_weights(self.audio_encoder, self.audio_encoder_layer_norm, audio_path)
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder_layer_norm.parameters():
                param.requires_grad = False
            self.check_pretrained_loaded(self.audio_encoder, audio_path, prefix="encoder.")
            self.check_pretrained_loaded(self.audio_encoder_layer_norm, audio_path, prefix="layer_norm.")

        # 映像 (事前学習済み)
        self.video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.video_encoder_layer_norm = nn.LayerNorm((self.hidden_dim))
        if (video_pretrained_model_file != "test.pth"):
            self.load_pretrained_encoder_layer_weights(self.video_encoder, self.video_encoder_layer_norm, video_path)
            for param in self.video_encoder.parameters():
                param.requires_grad = False
            for param in self.video_encoder_layer_norm.parameters():
                param.requires_grad = False
            self.check_pretrained_loaded(self.video_encoder, video_path, prefix="encoder.")
            self.check_pretrained_loaded(self.video_encoder_layer_norm, video_path, prefix="layer_norm.")

        self.dropout = nn.Dropout(self.dropout_rate)


        # fusion
        if (self.dataset_name == "CREMA-D"):
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )
        elif (self.dataset_name == "MOSI"):
            transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=12)
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
            self.fusion = nn.Sequential(
                nn.Linear(self.hidden_dim*3, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )

        # decoder
        self.decoder = nn.Linear(self.hidden_dim, num_classes)



    def load_pretrained_encoder_layer_weights(self, encoder, layer_norm, path):
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
            elif key.startswith("layer_norm."):
                new_key = key.replace("layer_norm.", "")
                layer_norm_weights[new_key] = value

        encoder.load_state_dict(encoder_weights, strict=False)
        layer_norm.load_state_dict(layer_norm_weights, strict=False)



    def check_pretrained_loaded(self, model, path, prefix="encoder."):
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



    def one_forward(self, modality, x, encoder, attn_mask, encoder_layer_norm):
        with torch.no_grad():
            hidden = encoder(x, attention_mask=attn_mask)
            if (modality == "audio"): 
                f = hidden.last_hidden_state[:, 1:, :].mean(1)
            elif (modality == "video"):
                f = hidden.last_hidden_state[:, 0, :]
            f = encoder_layer_norm(f)
        f = self.dropout(f)
        return f



    def forward(self, audio_x, video_x, audio_attn_mask, video_attn_mask):
        audio_divided_f = self.one_forward("audio", audio_x, self.audio_encoder, audio_attn_mask, self.audio_encoder_layer_norm)
        video_divided_f = self.one_forward("video", video_x, self.video_encoder, video_attn_mask, self.video_encoder_layer_norm)

        fusion_f = torch.cat((audio_divided_f, video_divided_f), dim=1)
        fusion_f = self.fusion(fusion_f)

        y = self.decoder(fusion_f)

        return y 