import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PretrainModel(nn.Module):
    def __init__(self, input_modality, input_dim_video, input_dim_audio, hidden_dim, num_classes, bert_model_name, dropout_rate):
        super(PretrainModel, self).__init__()

        self.input_modality = input_modality

        self.dropout_rate = dropout_rate
        self.activation = nn.ReLU()

        # 次元数
        if (input_modality == "video"):
            self.input_dim = input_dim_video
        elif (input_modality == "audio"):
            self.input_dim = input_dim_audio
        self.hidden_dim = hidden_dim

        # 特徴量抽出model
        self.rnn1 = nn.LSTM(self.input_dim, self.input_dim, bidirectional=True)
        self.rnn2 = nn.LSTM(2*self.input_dim, self.input_dim, bidirectional=True)

        self.layer_norm = nn.LayerNorm((self.input_dim*2))

        self.project = nn.Sequential(
            nn.Linear(self.input_dim*4, self.hidden_dim),
            self.activation,
            nn.LayerNorm(self.hidden_dim)
        )

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


    def feature_extraction(self, sequence, lengths, batch_size, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        packed_h1, (final_h1, _) = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)   # パッキング解除
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        _, (final_h2, _) = rnn2(packed_normed_h1)

        f = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        return f


    def forward(self, x1, x2, x3, x4, lengths):
        """
        Args:
            x1, x2, x3, x4: 各モダリティの入力 (batch_size, seq_len, feature_dim)
        Returns:
            y: 予測結果 (batch_size, num_classes)
            f1, f2, f3, f4: 各モダリティの特徴量 (batch_size, hidden_dim)
            s1, s2, s3, s4: 各モダリティの共通特徴量 (batch_size, hidden_dim)
            p1, p2, p3, p4: 各モダリティの固有特徴量 (batch_size, hidden_dim)
            r1, r2, r3, r4: 各モダリティの再構成特徴量 (batch_size, hidden_dim)
        """
        batch_size = x1.size(0)

        # 特徴量抽出
        f1 = self.feature_extraction(x1, lengths, batch_size, self.rnn1, self.rnn2, self.layer_norm)
        f2 = self.feature_extraction(x2, lengths, batch_size, self.rnn1, self.rnn2, self.layer_norm)
        f3 = self.feature_extraction(x3, lengths, batch_size, self.rnn1, self.rnn2, self.layer_norm)
        f4 = self.feature_extraction(x4, lengths, batch_size, self.rnn1, self.rnn2, self.layer_norm)

        # 共通特徴量と固有特徴量に分割
        s1 = self.shared(f1)
        s2 = self.shared(f2)
        s3 = self.shared(f3)
        s4 = self.shared(f4)

        p1 = self.private1(f1)
        p2 = self.private2(f2)
        p3 = self.private3(f3)
        p4 = self.private4(f4)

        # 再構成
        r1 = self.recon1(s1 + p1)
        r2 = self.recon2(s2 + p2)
        r3 = self.recon3(s3 + p3)
        r4 = self.recon4(s4 + p4)

        y = self.fusion((s1 + s2 + s3 + s4) / 4)

        return y, [f1, f2, f3, f4], [s1, s2, s3, s4], [p1, p2, p3, p4], [r1, r2, r3, r4]
