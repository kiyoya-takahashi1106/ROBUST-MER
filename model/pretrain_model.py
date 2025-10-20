import torch
import torch.nn as nn
from transformers import WavLMModel, VideoMAEModel


class PretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int, dropout_rate: float, 
                 pretrained_model_file: str, prepretrained_dataset: str, prepretrained_classnum: int):
        super(PretrainModel, self).__init__()

        self.input_modality = input_modality
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        if (prepretrained_dataset == "MOSI"  and  prepretrained_classnum == 2):
            self.num_classes = 2
        else:
            self.num_classes = num_classes

        if (input_modality == "audio"):
            self.encoder = WavLMModel.from_pretrained("microsoft/wavlm-base")
            premodel_path = "./saved_models/prepretrain/audio/" + pretrained_model_file

        elif (input_modality == "video"):
            self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            premodel_path = "./saved_models/prepretrain/video/" + pretrained_model_file

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        if (pretrained_model_file != "test.pth"):
            self.load_pretrained_layer_weights(premodel_path)
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.layer_norm.parameters():
                param.requires_grad = False

            self.check_pretrained_loaded(self.encoder, premodel_path, prefix="encoder.")
            self.check_pretrained_loaded(self.layer_norm, premodel_path, prefix="layer_norm.")

        # shared division encoder
        self.shared = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # private division encoders
        self.private1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.private2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.private3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.private4 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # reconstruction decoders
        self.recon1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.recon2 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.recon3 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.recon4 = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        # fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

        # 入力: grpoup1, group2, group3, group4 のいずれかのprivate特徴量
        self.private_discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 4)
        )

        # # {入力：4つの共通の平均、　4種類の固有}、{出力：(5クラス分類の結果)}
        # self.sp_discriminator = nn.Linear(self.hidden_dim, 5)


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


    def one_forward(self, x, attn_mask, private, recon):
        with torch.no_grad():
            output_encoder_model = self.encoder(x, attention_mask=attn_mask)

            if (self.input_modality == "audio"): 
                f = output_encoder_model.last_hidden_state[:, 1:, :].mean(1)
            elif (self.input_modality == "video"):
                f = output_encoder_model.last_hidden_state[:, 1:, :].mean(1)
            f = self.layer_norm(f)

        s = self.shared(f)
        p = private(f)
        sp = torch.cat([s, p], dim=1)
        r = recon(sp)

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

        p1_logits = self.private_discriminator(p1)
        p2_logits = self.private_discriminator(p2)
        p3_logits = self.private_discriminator(p3)
        p4_logits = self.private_discriminator(p4)

        y = self.fusion((s1 + s2 + s3 + s4) / 4)

        return y, [f1, f2, f3, f4], [s1, s2, s3, s4], [p1, p2, p3, p4], [r1, r2, r3, r4], [p1_logits, p2_logits, p3_logits, p4_logits]
