# pretrain2.py {train, val} と同じ分け方

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
    def __init__(self, seed):
        self.seed = seed
        self.train_sentence_emotion_group_dct, self.val_sentence_emotion_group_dct = cremed_classification(self.seed)
        self.train_dataset = transform2prepre(self.train_sentence_emotion_group_dct)
        self.val_dataset = transform2prepre(self.val_sentence_emotion_group_dct)

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



# すべてのデータを (テキスト12 × 感情6) × grpoup4 に分類するし、それをtrainとvalに分割
def cremed_classification(seed):
    # CSVファイルを読み込み
    actors_file = pd.read_csv('./data/CREMA-D/raw/VideoDemographics.csv')
    actors_file_content = actors_file.copy()
    datas_file = pd.read_csv('./data/CREMA-D/raw/SentenceFilenames.csv')
    datas_file_content = datas_file.copy()

    # 各actorがgroupのどこに属するかを確認
    actor_dct = {}
    age_threshold = 40
    for _, row in actors_file_content.iterrows():
        actor_id = row["ActorID"]
        age = row["Age"]
        gender = row["Sex"]
            
        if age <= age_threshold:
            if gender == 'Male':  
                actor_dct[actor_id] = 1
            else:  
                actor_dct[actor_id] = 3
        else: 
            if gender == 'Male':
                actor_dct[actor_id] = 2
            else:
                actor_dct[actor_id] = 4

    sentence_emotion_group_dct = {}   # テキスト_感情_group -> datafile名のリスト
    for _, row in datas_file_content.iterrows():
        filename = row["Filename"]
        part = filename.split('_')
        actor_id = int(part[0])
        sentence_emotion = part[1] + "_" + part[2]

        if sentence_emotion not in sentence_emotion_group_dct:
            sentence_emotion_group_dct[sentence_emotion] = {1: [], 2: [], 3: [], 4: []}
        
        actor_group = actor_dct[actor_id]
        sentence_emotion_group_dct[sentence_emotion][actor_group].append(filename)

    # train/val分割
    train_sentence_emotion_group_dct = {}
    val_sentence_emotion_group_dct = {}
    rng = random.Random(seed)   # valの組み合わせは毎回同じ
    
    for sentence_emotion, group_dct in sentence_emotion_group_dct.items():
        min_len = min(len(group_dct[1]), len(group_dct[2]), len(group_dct[3]), len(group_dct[4]))
        train_sentence_emotion_group_dct[sentence_emotion] = {1: [], 2: [], 3: [], 4: []}
        val_sentence_emotion_group_dct[sentence_emotion] = {1: [], 2: [], 3: [], 4: []}

        for group_num in [1, 2, 3, 4]:
            filename_lst = group_dct[group_num].copy()
            
            val_num = int(min_len * 0.2)
            val_filename_lst = filename_lst[:val_num]
            rng.shuffle(val_filename_lst)
            val_sentence_emotion_group_dct[sentence_emotion][group_num] = val_filename_lst

            train_filename_lst = filename_lst[val_num:]
            random.shuffle(train_filename_lst)
            if (len(train_filename_lst) > min_len-val_num):
                train_filename_lst = train_filename_lst[:min_len-val_num]
            train_sentence_emotion_group_dct[sentence_emotion][group_num] = train_filename_lst

    return train_sentence_emotion_group_dct, val_sentence_emotion_group_dct



def transform2prepre(sentence_emotion_group_dct):
    data = []
    emotion_label_dct = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

    for sentence_emotion, group_dct in sentence_emotion_group_dct.items():
        emotion = sentence_emotion.split('_')[1]
        label = emotion_label_dct[emotion]
        for group_num in [1, 2, 3, 4]:
            for i in range(len(group_dct[group_num])):
                sample = [group_dct[group_num][i]]
                sample = sample + [label]
                data.append(sample)

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