import torch    
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import AutoFeatureExtractor, VideoMAEImageProcessor
from decord import VideoReader
import torchaudio
torchaudio.set_audio_backend("soundfile")

import random
import pandas as pd
import numpy as np


class CREMADDataProvider:
    def __init__(self):
        self.train_dataset, self.val_dataset = select_data()

    def get_dataset(self):
        return self.train_dataset, self.val_dataset



class CREMADDataset(Dataset):
    def __init__(self, data, input_modality):
        self.data = data
        self.input_modality = input_modality

        if (self.input_modality == "audio") :
            self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        elif (self.input_modality == "video"):
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        x, label = sample
        x, attn_mask = pre_process(x, self.input_modality, self.processor)
        return x, attn_mask, label



def select_data():
    # CSVファイルを読み込み
    datas_file = pd.read_csv('./data/CREMA-D/raw/SentenceFilenames.csv')
    datas_file_content = datas_file.copy()

    emotion_label_dct = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
    XX_data_lst = []
    train_data_lst = []
    val_data_lst = []

    for _, row in datas_file_content.iterrows():
        filename = row["Filename"]
        part = filename.split('_')
        emotion = part[2]
        label = emotion_label_dct[emotion]
        strong = part[3]

        if (strong != "XX"):
            continue

        XX_data_lst.append([filename, label])

    random.shuffle(XX_data_lst)

    #  train/val分割
    train_num = int(len(XX_data_lst) * 0.8)
    for i in range(len(XX_data_lst)):
        if (i < train_num):
            train_data_lst.append(XX_data_lst[i])
        else:
            val_data_lst.append(XX_data_lst[i])

    return train_data_lst, val_data_lst



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