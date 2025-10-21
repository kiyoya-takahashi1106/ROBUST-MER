import torch    
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import AutoFeatureExtractor, VideoMAEImageProcessor
from decord import VideoReader
import torchaudio
torchaudio.set_audio_backend("soundfile")

import pandas as pd
import numpy as np
import random
import os


class CREMADDataset(Dataset):
    def __init__(self, split):
        self.split = split

        self.data = get_data(self.split)

        self.audio_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        filename, label, group_label = sample
        audio_x, audio_mask = pre_process(filename, "audio", self.audio_processor)
        video_x, video_mask = pre_process(filename, "video", self.video_processor)
        return audio_x, video_x, audio_mask, video_mask, label, group_label



def get_data(split):
    data_folder_path = f"./data/CREMA-D/{split}/AudioWAV/"   # videoでも拡張子 以外のところは同じファイル名なのでこれでOK
    emotion_label_dct = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

    # group_label: 0=若い男性, 1=年配男性, 2=若い女性, 3=年配女性
    group_label_dct = {}   # key: actor_id, value: group_label
    for _, row in pd.read_csv('./data/CREMA-D/raw/VideoDemographics.csv').iterrows():
        age_threshold = 40
        actor_id = int(row["ActorID"])
        age = row["Age"]
        gender = row["Sex"]
        if (age < age_threshold):
            if (gender == 'Male'):
                group_label_dct[actor_id] = 0
            else:
                group_label_dct[actor_id] = 2
        else:
            if (gender == 'Male'):
                group_label_dct[actor_id] = 1
            else:
                group_label_dct[actor_id] = 3

    data = []
    for file in os.listdir(data_folder_path):
        file = file.replace(".wav", "")
        label = emotion_label_dct[file.split("_")[2]]
        actor_id = int(file.split("_")[0])
        data.append((file, label, group_label_dct[actor_id]))

    return data



# 指定されたファイルの各データに対して前処理を行いテンソルで返す
def pre_process(filename, input_modality, processor):
    parent_dir = "AudioWAV" if input_modality == "audio" else "VideoFlash"
    file_path = f"data/CREMA-D/raw/{parent_dir}/{filename}"
    
    # モダリティ別処理
    if (input_modality == "audio"):
        file_path += ".wav"
        waveform, sr = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        waveform, attn_mask = fix_length_and_mask(waveform.squeeze(0))

        processor_output = processor(waveform, sampling_rate=16000, return_tensors="pt")
        result = processor_output['input_values'].squeeze(0)

    elif (input_modality == "video"):
        file_path += ".flv"
        vr = VideoReader(file_path)
        T = 16
        indices = np.linspace(0, len(vr) - 1, T).astype(int)
        frames = [vr[i].asnumpy() for i in indices]
        processor_output = processor(frames, return_tensors="pt")
        result = processor_output['pixel_values'].squeeze(0)
        attn_mask = torch.ones((T,), dtype=torch.long)   # ★ None を避ける

    return result, attn_mask



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
        wav_fixed = F.pad(wav_1d, (0, pad))   # 末尾ゼロ埋め
        attn_mask = torch.cat([torch.ones(T, dtype=torch.long),
                              torch.zeros(pad, dtype=torch.long)], dim=0)
    return wav_fixed, attn_mask