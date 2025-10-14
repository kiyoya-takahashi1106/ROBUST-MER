import torch    
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, VideoMAEImageProcessor
from decord import VideoReader
import torchaudio
torchaudio.set_audio_backend("soundfile")

import os
from pathlib import Path
import pandas as pd
import numpy as np



class MOSIDataset(Dataset):
    def __init__(self, dataset, split, input_modality):
        self.dataset = dataset
        self.split = split
        self.input_modality = input_modality    
        if (self.input_modality == "audio"):
            self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        elif (self.input_modality == "video"):
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        else:
            raise ValueError(f"Unknown input_modality: {input_modality}")
        
        self.filename_list = load_filename_list(dataset, split)

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]
        x, x_mask, label = pre_process(
            self.dataset, 
            self.split, 
            filename,
            self.processor,
            self.input_modality
        )
        return x, x_mask, label



def load_filename_list(dataset, split):
    folder_path = Path(f"data/{dataset}/segment/{split}/video")
    filename_list = []
    for file_path in sorted(folder_path.glob("*.flv")):
        filename = file_path.stem
        filename_list.append(filename)
    
    return filename_list



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