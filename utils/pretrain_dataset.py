# 各group間のdataが一番多いgroupに合わせてたのを一番小さい方に合わせる (XXに限定しない)

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
import os


"""
# 1, 2, 3, 4 のlistすべて同じ要素数
sentence_emotion_group_dct = {
    sentence1_emotion1: {1: [filename ...], 2: [], 3: [], 4: []},
    sentence1_emotion2: {1: [], 2: [], 3: [], 4: []},
    sentence1_emotion3: {1: [], 2: [], 3: [], 4: []},
    ...
    sentence12_emotion6: {1: [], 2: [], 3: [], 4: []}
}

train_dataset = [
    [group1の任意のdatatensor, group2の任意のdatatensor, group3の任意のdatatensor, group4の任意のdatatensor, 1~6の感情ラベル],
    [ , , , , ],
    [ , , , , ],
    ...
    [ , , , , ]
]
"""


class CREMADDataset(Dataset):
    def __init__(self, split, input_modality, epoch, prepretrained_dataset, prepretrained_classnum):
        self.split = split
        self.input_modality = input_modality
        self.epoch = epoch

        self.prepretrained_dataset = prepretrained_dataset
        self.prepretrained_classnum = prepretrained_classnum

        self.sentence_emotion_group_dct = cremed_classification(split, input_modality, epoch)
        self.data = make_data_combination(self.sentence_emotion_group_dct, self.prepretrained_dataset, self.prepretrained_classnum)

        if (self.input_modality == "audio") :
            self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        elif (self.input_modality == "video"):
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        group1_file, group2_file, group3_file, group4_file, label = sample
        group1, attn_mask1 = pre_process(group1_file, self.input_modality, self.processor)
        group2, attn_mask2 = pre_process(group2_file, self.input_modality, self.processor)
        group3, attn_mask3 = pre_process(group3_file, self.input_modality, self.processor)
        group4, attn_mask4 = pre_process(group4_file, self.input_modality, self.processor)
        return group1, group2, group3, group4, attn_mask1, attn_mask2, attn_mask3, attn_mask4, label



# すべてactorを4分割して、
def cremed_classification(split, input_modality, epoch):
    group_dct = {"group1": [], "group2": [], "group3": [], "group4": []}
    actors_file = pd.read_csv('./data/CREMA-D/raw/VideoDemographics.csv')
    age_threshold = 40
    for _, row in actors_file.iterrows():
        actor_id = row["ActorID"]
        age = row["Age"]
        gender = row["Sex"]
        if (age < age_threshold):
            if (gender == 'Male'):
                group_dct["group1"].append(actor_id)
            else:
                group_dct["group3"].append(actor_id)
        else:
            if (gender == 'Male'):
                group_dct["group2"].append(actor_id)
            else:
                group_dct["group4"].append(actor_id)

    
    sentence_emotion_group_dct = {}
    data_folder = f"./data/CREMA-D/{split}/{"AudioWAV" if input_modality == "audio" else "VideoFlash"}/"
    for filename in os.listdir(data_folder):
        if (input_modality == "audio"):
            filename = filename.replace(".wav", "")
        elif (input_modality == "video"):
            filename = filename.replace(".flv", "")
        part = filename.split('_')
        actor_id = int(part[0])
        for group_num, actor_lst in group_dct.items():
            if actor_id in actor_lst:
                actor_group = int(group_num[-1])    # "group1" -> 1
                break
        sentence_emotion = part[1] + "_" + part[2]

        if (sentence_emotion not in sentence_emotion_group_dct):
            sentence_emotion_group_dct[sentence_emotion] = {1: [], 2: [], 3: [], 4: []}
        sentence_emotion_group_dct[sentence_emotion][actor_group].append(filename)

    
    # 各リストの要素のずれをなくす (最小値に合わせる)
    for sentence_emotion, group_dct in sentence_emotion_group_dct.items():
        min_len = min(len(group_dct[1]), len(group_dct[2]), len(group_dct[3]), len(group_dct[4]))
        for group_num in [1, 2, 3, 4]:
            # 学習のときは組み合わせにランダム性を持たせる
            if (split == "train"):
                rng = random.Random(epoch + group_num)
            else:
                rng = random.Random(group_num)
            rng.shuffle(group_dct[group_num])
            group_dct[group_num] = group_dct[group_num][:min_len]

    return sentence_emotion_group_dct



#  実際のデータを作成 (要素はfilename)
def make_data_combination(sentence_emotion_group_dct, prepretrained_dataset, prepretrained_classnum):
    data = []
    if (prepretrained_dataset == "MOSI" and prepretrained_classnum == 2):
        emotion_label_dct = {"ANG": 0, "HAP": 1}
    else:
        emotion_label_dct = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

    for sentence_emotion, group_dct in sentence_emotion_group_dct.items():
        emotion = sentence_emotion.split('_')[1]
        if (prepretrained_dataset == "MOSI" and prepretrained_classnum == 2):
            if (emotion not in ["ANG", "HAP"]):
                continue
        label = emotion_label_dct[emotion]

        for i in range(len(group_dct[1])):
            sample = [group_dct[1][i], group_dct[2][i], group_dct[3][i], group_dct[4][i]]
            sample = sample + [label]
            data.append(sample)

    return data



# 指定されたファイルの各データに対して前処理を行いテンソルで返す
def pre_process(filename, input_modality, processor):
    parent_dir = "AudioWAV" if input_modality == "audio" else "VideoFlash"
    file_path = f"./data/CREMA-D/raw/{parent_dir}/{filename}"
    
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