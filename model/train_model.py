import torch
import torch.nn as nn
from transformers import RobertaModel, WavLMModel, VideoMAEModel
from peft import LoraConfig, get_peft_model


class Model(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout_rate: float, dataset_name: str, audio_pretrained_model_file: str, text_pretrained_model_file: str, video_pretrained_model_file: str):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name

        audio_path = "./saved_models/pretrain/audio/" + audio_pretrained_model_file
        text_path = "./saved_models/prepretrain/text/" + text_pretrained_model_file
        video_path = "./saved_models/prepretrain/video/" + video_pretrained_model_file

        # Encoder + LN
        # 音声 (事前学習済み)
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.audio_encoder_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.load_pretrained_encoder_layer_weights(self.audio_encoder, self.audio_encoder_layer_norm, audio_path)
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder_layer_norm.parameters():
            param.requires_grad = False

        self.check_pretrained_loaded(self.audio_encoder, audio_path, prefix="encoder.")
        self.check_pretrained_loaded(self.audio_encoder_layer_norm, audio_path, prefix="layer_norm.")

        # テキスト
        self.text_encoder = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.text_encoder_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.load_pretrained_encoder_layer_weights(self.text_encoder, self.text_encoder_layer_norm, text_path)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder_layer_norm.parameters():
            param.requires_grad = False

        self.check_pretrained_loaded(self.text_encoder, text_path, prefix="encoder.")
        self.check_pretrained_loaded(self.text_encoder_layer_norm, text_path, prefix="layer_norm.")

        # 映像 (事前学習済み)
        self.video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.video_encoder_layer_norm = nn.LayerNorm((self.hidden_dim))
        self.load_pretrained_encoder_layer_weights(self.video_encoder, self.video_encoder_layer_norm, video_path)
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        for param in self.video_encoder_layer_norm.parameters():
            param.requires_grad = False

        self.check_pretrained_loaded(self.video_encoder, video_path, prefix="encoder.")
        self.check_pretrained_loaded(self.video_encoder_layer_norm, video_path, prefix="layer_norm.")


        # 共通分離
        # 音声
        self.audio_shared = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.audio_shared_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.load_pretrained_division_layer_weights(self.audio_shared, self.audio_shared_layer_norm, audio_path)
        for param in self.audio_shared.parameters():
            param.requires_grad = False
        for param in self.audio_shared_layer_norm.parameters():
            param.requires_grad = False
        self.audio_shared_dropout = nn.Dropout(self.dropout_rate)

        self.check_pretrained_loaded(self.audio_shared, audio_path, prefix="shared.0.")
        self.check_pretrained_loaded(self.audio_shared_layer_norm, audio_path, prefix="fusion.0.")

        # テキスト
        self.text_shared_dropout = nn.Dropout(self.dropout_rate)

        # 映像
        self.video_shared = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.video_shared_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.load_pretrained_division_layer_weights(self.video_shared, self.video_shared_layer_norm, video_path)
        for param in self.video_shared.parameters():
            param.requires_grad = False
        for param in self.video_shared_layer_norm.parameters():
            param.requires_grad = False
        self.video_shared_dropout = nn.Dropout(self.dropout_rate)

        # self.check_pretrained_loaded(self.audio_shared, audio_path, prefix="shared.0.")
        # self.check_pretrained_loaded(self.audio_shared_layer_norm, audio_path, prefix="fusion.0.")


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


    def load_pretrained_division_layer_weights(self, linear, layer_norm, path):
        """
        pretrain_model から共通分離層の重みを読み込む
        - linear: pretrain_model の self.shared (nn.Sequential内のnn.Linear) から読み込む
        - layer_norm: pretrain_model の self.fusion (nn.Sequential内のnn.LayerNorm) から読み込む
        """
        checkpoint = torch.load(path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        linear_weights = {}
        layer_norm_weights = {}

        for key, value in state_dict.items():
            if key.startswith("shared.0."):
                new_key = key.replace("shared.0.", "")
                linear_weights[new_key] = value
            elif key.startswith("fusion.0."):
                new_key = key.replace("fusion.0.", "")
                layer_norm_weights[new_key] = value

        if linear_weights:
            linear.load_state_dict(linear_weights, strict=False)
        if layer_norm_weights:
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


    def one_forward(self, modality, x, encoder, attn_mask, encoder_layer_norm, linear, shared_layer_norm, dropout):
        if (modality == "audio"): 
            with torch.no_grad():
                encoder_output = encoder(x, attention_mask=attn_mask)
                f = encoder_output.last_hidden_state[:, 1:, :].mean(1)
                f = encoder_layer_norm(f)
                f = linear(f)
                f = shared_layer_norm(f)
        elif (modality == "text"):
            encoder_output = encoder(x, attention_mask=attn_mask)
            f = encoder_output.last_hidden_state[:, 0, :]   # CLSトークン
            f = encoder_layer_norm(f)
        elif (modality == "video"):
            with torch.no_grad():
                encoder_output = encoder(x, attention_mask=attn_mask)
                f = encoder_output.last_hidden_state[:, 1:, :].mean(1)
                f = encoder_layer_norm(f)
                # f = linear(f)
                # f = shared_layer_norm(f)

        f = dropout(f)

        return f


    def forward(self, audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask):
        """
        マルチモーダル特徴量抽出のフォワードパス
        Args:
            text_x: テキスト入力 (batch_size, seq_len)
            video_x: 映像入力 (seq_len, batch_size, video_dim)  
            audio_x: 音声入力 (seq_len, batch_size, audio_dim)
            lengths: 各シーケンスの長さ (batch_size)
        Returns:
            text_f, video_f, audio_f: 各モダリティの特徴量 (batch_size, hidden_dim)
        """

        audio_divided_f = self.one_forward("audio", audio_x, self.audio_encoder, audio_attn_mask, self.audio_encoder_layer_norm, self.audio_shared, self.audio_shared_layer_norm, self.audio_shared_dropout)
        text_divided_f = self.one_forward("text", text_x, self.text_encoder, text_attn_mask, self.text_encoder_layer_norm, None, None, self.text_shared_dropout)
        video_divided_f = self.one_forward("video", video_x, self.video_encoder, video_attn_mask, self.video_encoder_layer_norm, self.video_shared, self.video_shared_layer_norm, self.video_shared_dropout)

        if (self.dataset_name == "CREMA-D"):
            fusion_f = torch.cat((audio_divided_f, video_divided_f), dim=1)
            fusion_f = self.fusion(fusion_f)
        elif (self.dataset_name == "MOSI"):
            fusion_f = torch.stack((audio_divided_f, text_divided_f, video_divided_f), dim=0)
            fusion_f = self.transformer_encoder(fusion_f)
            fusion_f = torch.cat((fusion_f[0], fusion_f[1], fusion_f[2]), dim=1)
            fusion_f = self.fusion(fusion_f)

        y = self.decoder(fusion_f)

        return y 