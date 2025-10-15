import torch    
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoTokenizer, VideoMAEImageProcessor
from decord import VideoReader
import torchaudio
torchaudio.set_audio_backend("soundfile")

import os
from pathlib import Path
import pandas as pd
import numpy as np


class MOSIDataset(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        self.audio_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.filename_list = load_filename_list(dataset, split)
        self.text, self.text_mask = load_text_data(dataset, split, self.tokenizer, self.filename_list)

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]
        audio, video, audio_mask, video_mask, label = pre_process(
            self.dataset, 
            self.split, 
            filename,
            self.audio_processor,
            self.video_processor
        )
        return audio, self.text[index], video, audio_mask, self.text_mask[index], video_mask, label



def load_filename_list(dataset, split):
    folder_path = Path(f"data/{dataset}/segment/{split}/video")
    filename_list = []
    for file_path in sorted(folder_path.glob("*.flv")):
        filename = file_path.stem
        filename_list.append(filename)
    
    return filename_list



def load_text_data(dataset, split, tokenizer, filename_list, max_length=64):
    text_path = Path(f"data/{dataset}/segment/{split}/text")

    # すべてのテキストを読み込む（1つのリストにまとめる）
    all_texts = []
    for filename in filename_list:
        file_path = text_path / f"{filename}.txt"
        with open(file_path, "r", encoding="utf-8") as f:
            all_texts.append(f.read().strip())

    # まとめてトークナイズ（attention_mask自動生成）
    encoded = tokenizer(
        all_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    # data構造を元の関数に合わせて返す
    # text_data は各要素が tokenizer 出力(dict型)
    text_data = []
    text_mask = []

    for i in range(len(all_texts)):
        item = {
            'input_ids': encoded['input_ids'][i].unsqueeze(0),  # 形を(1, seq_len)に合わせる
            'attention_mask': encoded['attention_mask'][i].unsqueeze(0)
        }
        text_data.append(item['input_ids'])
        text_mask.append(item['attention_mask'])

    return text_data, text_mask



# 指定されたファイルの各データに対して前処理を行いテンソルで返す
def pre_process(dataset, split, filename, audio_processor, video_processor):
    segment_dir = Path(f"data/{dataset}/segment/{split}")
    audio_path = segment_dir / "audio" / f"{filename}.wav"
    video_path = segment_dir / "video" / f"{filename}.flv"
    label_csv_path = Path(f"data/{dataset}/raw/label.csv")

    # ===== 1. Audio 処理 =====
    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform, audio_mask = fix_length_and_mask(waveform.squeeze(0))
    processor_output = audio_processor(waveform, sampling_rate=16000, return_tensors="pt")
    audio = processor_output['input_values'].squeeze(0)
    
    # ===== 2. Video 処理 =====
    vr = VideoReader(str(video_path))
    T = 16
    indices = np.linspace(0, len(vr) - 1, T).astype(int)
    frames = [vr[i].asnumpy() for i in indices]
    processor_output = video_processor(frames, return_tensors="pt")
    video = processor_output['pixel_values'].squeeze(0)
    video_mask = torch.ones((T,), dtype=torch.long)   # ★ None を避ける

    # ===== 3. Label 取得 =====
    df = pd.read_csv(label_csv_path)
    parts = filename.rsplit('-', 1)
    video_id = parts[0]
    clip_id = int(parts[1])
    
    row = df[(df['video_id'] == video_id) & (df['clip_id'] == clip_id)]
    label = float(row.iloc[0]['label'])

    label = torch.tensor(label, dtype=torch.float32)

    return audio, video, audio_mask, video_mask, label



# 音声dataを5秒×16000Hzにする
TARGET_SEC = 5
TARGET_LEN = 16000 * TARGET_SEC  
def fix_length_and_mask(wav_1d: torch.Tensor, target_len: int = TARGET_LEN):
    T = wav_1d.size(0)
    if T >= target_len:
        # ✅ 均等にダウンサンプリング
        indices = torch.linspace(0, T - 1, target_len).long()
        wav_fixed = wav_1d[indices]
        attn_mask = torch.ones(target_len, dtype=torch.long)
    else:
        pad = target_len - T
        wav_fixed = F.pad(wav_1d, (0, pad))
        attn_mask = torch.cat([torch.ones(T, dtype=torch.long),
                              torch.zeros(pad, dtype=torch.long)], dim=0)
    return wav_fixed, attn_mask