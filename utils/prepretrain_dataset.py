import torch    
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, RobertaTokenizer, VideoMAEImageProcessor
from decord import VideoReader
import torchaudio
torchaudio.set_audio_backend("soundfile")

import os
from pathlib import Path
import pandas as pd
import numpy as np


class MOSIDataset(Dataset):
    def __init__(self, dataset, split, input_modality, class_num):
        self.dataset = dataset
        self.split = split
        self.input_modality = input_modality
        self.class_num = class_num   # 2  or  7
        self.filename_list = load_filename_list(dataset, split)

        if (self.input_modality == "audio"):
            self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        elif (self.input_modality == "text"):
            self.processor = RobertaTokenizer.from_pretrained("roberta-base")
            self.text, self.text_mask, self.label = load_text_data(dataset, split, self.processor, self.filename_list)
        elif (self.input_modality == "video"):
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        if (self.input_modality == "text"):
            x = self.text[index]
            x_mask = self.text_mask[index]
            label = self.label[index]
        elif (self.input_modality == "audio") or (self.input_modality == "video"):
            filename = self.filename_list[index]
            x, x_mask, label = pre_process(
                self.dataset, 
                self.split, 
                filename,
                self.processor,
                self.input_modality
            )

        if (self.class_num == 2):
            label = torch.tensor(1) if label.item() >= 0 else torch.tensor(0)
        elif (self.class_num == 7):
            # -3 ~ +3 実装中
            pass

        return x, x_mask, label



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
    
    # ===== Label 取得（各ファイルごとにリストを作成） =====
    df = pd.read_csv(Path(f"data/{dataset}/raw/label.csv"))
    labels = []
    for filename in filename_list:
        parts = filename.rsplit('-', 1)
        if len(parts) != 2:
            raise ValueError(f"Unexpected filename format: {filename}")
        video_id = parts[0]
        try:
            clip_id = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid clip id in filename: {filename}")

        row = df[(df['video_id'] == video_id) & (df['clip_id'] == clip_id)]
        if row.empty:
            raise ValueError(f"Label not found for video_id={video_id}, clip_id={clip_id}")
        lbl = float(row.iloc[0]['label'])
        labels.append(torch.tensor(lbl, dtype=torch.float32))

    return text_data, text_mask, labels



# 指定されたファイルの各データに対して前処理を行いテンソルで返す
def pre_process(dataset, split, filename, processor, input_modality):
    segment_dir = Path(f"data/{dataset}/segment/{split}")
    label_csv_path = Path(f"data/{dataset}/raw/label.csv")

    if input_modality == "audio":
        # ===== Audio 処理 =====
        audio_path = segment_dir / "audio" / f"{filename}.wav"
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform, mask = fix_length_and_mask(waveform.squeeze(0))
        processor_output = processor(waveform, sampling_rate=16000, return_tensors="pt")
        x = processor_output['input_values'].squeeze(0)
    
    elif input_modality == "video":
        # ===== Video 処理 =====
        video_path = segment_dir / "video" / f"{filename}.flv"
        vr = VideoReader(str(video_path))
        T = 16
        indices = np.linspace(0, len(vr) - 1, T).astype(int)
        frames = [vr[i].asnumpy() for i in indices]
        processor_output = processor(frames, return_tensors="pt")
        x = processor_output['pixel_values'].squeeze(0)
        mask = torch.ones((T,), dtype=torch.long)
    
    else:
        raise ValueError(f"Unknown input_modality: {input_modality}")

    # ===== Label 取得 =====
    df = pd.read_csv(label_csv_path)
    parts = filename.rsplit('-', 1)
    video_id = parts[0]
    clip_id = int(parts[1])
    
    row = df[(df['video_id'] == video_id) & (df['clip_id'] == clip_id)]
    label = float(row.iloc[0]['label'])
    label = torch.tensor(label, dtype=torch.float32)

    return x, mask, label



# 音声dataを5秒×16000Hzにする
TARGET_SEC = 5
TARGET_LEN = 16000 * TARGET_SEC  
def fix_length_and_mask(wav_1d: torch.Tensor, target_len: int = TARGET_LEN):
    T = wav_1d.size(0)
    if (T >= target_len):
        wav_fixed = wav_1d[:target_len]
        attn_mask = torch.ones(target_len, dtype=torch.long)
    else:
        pad = target_len - T
        wav_fixed = F.pad(wav_1d, (0, pad))
        attn_mask = torch.cat([torch.ones(T, dtype=torch.long),
                              torch.zeros(pad, dtype=torch.long)], dim=0)
    return wav_fixed, attn_mask