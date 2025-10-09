import torch
import torch.nn as nn
from transformers import WavLMModel, VideoMAEModel


class PretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int, dropout_rate: float):
        super(PretrainModel, self).__init__()

        self.input_modality = input_modality
        self.hidden_dim = hidden_dim

        self.dropout_rate = dropout_rate
        self.activation = nn.ReLU()

        if input_modality == "audio":
            self.encoder_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        elif input_modality == "video":
            self.encoder_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        for param in self.encoder_model.parameters():
                param.requires_grad = False

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # shared division encoder
        self.shared = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        # private division encoders
        self.private1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.private2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.private3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.private4 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        # reconstruction decoders
        self.recon1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.recon2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.recon3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.recon4 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # fusion
        self.fusion = nn.Linear(self.hidden_dim, num_classes)
        # self.fusion = nn.Sequential(
        #     # 4group => 2つに圧縮 => class数
        #     nn.Linear(self.hidden_dim*4, self.hidden_dim*2),
        #     nn.Dropout(self.dropout_rate),
        #     self.activation,
        #     nn.Linear(self.hidden_dim*2, num_classes)
        # )

        # discriminator が入力がどのgruoupか混乱するようにmodelを訓練したい.
        # 入力: grpoup1, group2, group3, group4 のいずれかのshared特徴量
        self.discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 4)
        )

        # {入力：4つの共通の平均、　4種類の固有}、{出力：(5クラス分類の結果)}
        self.sp_discriminator = nn.Linear(self.hidden_dim, 5)


    def one_forward(self, x, attn_mask, private, recon):
        with torch.no_grad():
            output_encoder_model = self.encoder_model(x, attention_mask=attn_mask)

            if (self.input_modality == "audio"): 
                f = output_encoder_model.last_hidden_state[:, 1:, :].mean(1)
            elif (self.input_modality == "video"):
                f = output_encoder_model.last_hidden_state[:, 0, :]
        f = self.layer_norm(f)

        s = self.shared(f)
        p = private(f)
        r = recon(s + p)

        return f, s, p, r



    def forward(self, x1, x2, x3, x4, attn_mask1, attn_mask2, attn_mask3, attn_mask4):
        """
        Args:
            x1, x2, x3, x4: 各モダリティの入力
        Returns:
            y: 予測結果 (batch_size, num_classes)
            f1, f2, f3, f4: 各モダリティの特徴量 (batch_size, hidden_dim)
            s1, s2, s3, s4: 各モダリティの共通特徴量 (batch_size, hidden_dim)
            p1, p2, p3, p4: 各モダリティの固有特徴量 (batch_size, hidden_dim)
            r1, r2, r3, r4: 各モダリティの再構成特徴量 (batch_size, hidden_dim)
        """
        f1, s1, p1, r1 = self.one_forward(x1, attn_mask1, self.private1, self.recon1)
        f2, s2, p2, r2 = self.one_forward(x2, attn_mask2, self.private2, self.recon2)
        f3, s3, p3, r3 = self.one_forward(x3, attn_mask3, self.private3, self.recon3)
        f4, s4, p4, r4 = self.one_forward(x4, attn_mask4, self.private4, self.recon4)

        y = self.fusion((s1 + s2 + s3 + s4) / 4)

        return y, [f1, f2, f3, f4], [s1, s2, s3, s4], [p1, p2, p3, p4], [r1, r2, r3, r4]
