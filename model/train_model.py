import torch
import torch.nn as nn
from transformers import RobertaModel, WavLMModel, VideoMAEModel


class Model(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout_rate: float, audio_pretrained_model_file: str, video_pretrained_model_file: str):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout_rate = dropout_rate
        self.activation = nn.ReLU()
        
        # Encoder
        # 音声
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
        # テキスト
        self.text_model = RobertaModel.from_pretrained("roberta-base")
        # 映像
        self.video_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        self.layer_norm = nn.LayerNorm((self.hidden_dim))

        # Extraction Linear
        # 音声
        audio_path = "./saved_models/pretrain/audio" + audio_pretrained_model_file
        self.audio_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.load_pretrained_layer_weights(self.audio_linear, audio_path)
        # 映像
        video_path = "./saved_models/pretrain/video" + video_pretrained_model_file
        self.video_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.load_pretrained_layer_weights(self.video_linear, video_path)

        self.sigmoid = nn.Sigmoid()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim*3, self.hidden_dim),
            nn.Dropout(dropout_rate),
            self.activation,
            nn.Linear(self.hidden_dim, num_classes)
        )


    def load_pretrained_layer_weights(self, layer, path):
        pretrained_model = torch.load(path, map_location="cpu")
        if ("state_dict" in pretrained_model):
            pretrained_model = pretrained_model["state_dict"]
        
        shared_linear_dict = {
            k.replace("module.", ""): v for k, v in pretrained_model.items()
            if k.startswith("shared.0.")
        }
        weight = shared_linear_dict["shared.0.weight"]
        bias = shared_linear_dict["shared.0.bias"]
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)


    def one_forward(self, modality, x, encoder, attn_mask, linear):
        with torch.no_grad():
            encoder_output = encoder(x, attention_mask=attn_mask)
            if (modality == "audio"): 
                    f = encoder_output.last_hidden_state[:, 1:, :].mean(1)
            elif (modality == "text"):
                f = encoder_output.last_hidden_state[:, 0, :]  # CLSトークン
            elif (modality == "video"):
                f = encoder_output.last_hidden_state[:, 0, :]
        f = self.layer_norm(f)

        with torch.no_grad():
            divided_f = linear(f)
        divided_f = self.sigmoid(divided_f)

        return divided_f


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

        audio_divided_f = self.one_forward("audio", audio_x, self.audio_encoder, audio_attn_mask, self.audio_linear)
        text_divided_f = self.one_forward("text", text_x, self.text_model, text_attn_mask, nn.Identity())
        video_divided_f = self.one_forward("video", video_x, self.video_encoder, video_attn_mask, self.video_linear)

        fusion_f = torch.stack((audio_divided_f, text_divided_f, video_divided_f), dim=0)
        fusion_f = self.transformer_encoder(fusion_f)

        fusion_f = torch.cat((fusion_f[0], fusion_f[1], fusion_f[2]), dim=1)
        y = self.fusion(fusion_f)
        
        return y