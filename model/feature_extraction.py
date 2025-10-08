import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FeatureExtractionModel(nn.Module):
    def __init__(self, input_dim_audio, input_dim_video, hidden_dim, bert_model_name='bert-base-uncased'):
        super(FeatureExtractionModel, self).__init__()

        # 活性化関数
        self.activation = nn.ReLU()

        # 次元数
        self.input_dim_video = input_dim_video
        self.input_dim_audio = input_dim_audio
        self.hidden_dim = hidden_dim
        
        # Encoder
        # テキスト
        self.bertmodel = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bertmodel.config.hidden_size   # 768 (bert-base)のような値を動的に取得
        # 映像
        self.vrnn1 = nn.LSTM(self.input_dim_video, self.input_dim_video, bidirectional=True)
        self.vrnn2 = nn.LSTM(2*self.input_dim_video, self.input_dim_video, bidirectional=True)
        # 音声
        self.arnn1 = nn.LSTM(self.input_dim_audio, self.input_dim_audio, bidirectional=True)
        self.arnn2 = nn.LSTM(2*self.input_dim_audio, self.input_dim_audio, bidirectional=True)

        # layer norm
        self.vlayer_norm = nn.LayerNorm((self.input_dim_video*2,))
        self.alayer_norm = nn.LayerNorm((self.input_dim_audio*2,))

        # 各モダリティのベクトルを 線形変換 で hidden_sizeにそろえる
        # テキスト
        self.project_t = nn.Sequential(
            nn.Linear(self.bert_hidden_size, self.hidden_dim),
            self.activation,
            nn.LayerNorm(self.hidden_dim)
        )
        # 映像
        self.project_v = nn.Sequential(
            nn.Linear(self.input_dim_video*4, self.hidden_dim),
            self.activation,
            nn.LayerNorm(self.hidden_dim)
        )
        # 音声
        self.project_a = nn.Sequential(
            nn.Linear(self.input_dim_audio*4, self.hidden_dim),
            self.activation,
            nn.LayerNorm(self.hidden_dim)
        )
    

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        """
        RNNを使用してシーケンスから特徴量を抽出する
        Args:
            sequence: 入力シーケンス
            lengths: 各シーケンスの長さ
            rnn1, rnn2: LSTM層
            layer_norm: 正規化層
        """
        # 長さでソートされていることを前提とする
        packed_sequence = pack_padded_sequence(sequence, lengths, enforce_sorted=False)

        packed_h1, (final_h1, _) = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)   # パッキング解除
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, enforce_sorted=False)

        _, (final_h2, _) = rnn2(packed_normed_h1)

        return final_h1, final_h2
    

    def forward(self, text_x, video_x, audio_x, lengths):
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
        batch_size = lengths.size(0)

        bert_outputs = self.bertmodel(text_x)
        text_f = bert_outputs.last_hidden_state[:, 0, :]  # CLSトークンを使用

        # 映像の特徴抽出
        final_h1v, final_h2v = self.extract_features(video_x, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        video_f = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # 音声の特徴抽出
        final_h1a, final_h2a = self.extract_features(audio_x, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        audio_f = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # 各モダリティの特徴量を統一された次元に変換
        text_f = self.project_t(text_f)
        video_f = self.project_v(video_f)
        audio_f = self.project_a(audio_f)

        return text_f, video_f, audio_f