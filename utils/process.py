import torchaudio
from decord import VideoReader

import random
import numpy as np
import pandas as pd


"""
sentence_emotion_group_dct = {
    sentence1_emotion1: {1: [filename ...], 2: [], 3: [], 4: []},
    sentence1_emotion2: {1: [], 2: [], 3: [], 4: []},
    sentence1_emotion3: {1: [], 2: [], 3: [], 4: []},
    ...
    sentence12_emotion6: {1: [], 2: [], 3: [], 4: []}
}

epoch_data = [
    [group1の任意のdata, group2の任意のdata, group3の任意のdata, group4の任意のdata, 1~6の感情ラベル],
    [ , , , , ],
    [ , , , , ],
    ...
    [ , , , , ]
]
"""


# すべてのXXデータを (テキスト12 × 感情6) × grpoup4 に分類する
def cremed_classification():
    # CSVファイルを読み込み
    actors_file = pd.read_csv('../data/CREMA-D/raw/VideoDemographics.csv')
    actors_file_content = actors_file.copy()
    datas_file = pd.read_csv('../data/CREMA-D/raw/SentenceFilenames.csv')
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
        strong = part[3]

        if not strong == "XX":
            continue

        if sentence_emotion not in sentence_emotion_group_dct:
            sentence_emotion_group_dct[sentence_emotion] = {1: [], 2: [], 3: [], 4: []}
        
        actor_group = actor_dct[actor_id]
        sentence_emotion_group_dct[sentence_emotion][actor_group].append(filename)

    return sentence_emotion_group_dct



# 各グループのデータを最大長に合わせて拡張し、train/valデータを作成 (要素はfilename)
def make_data_combination(sentence_emotion_group_dct):
    epoch_train_data = []
    epoch_val_data = []
    emotion_label_dct = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}

    for sentence_emotion, group_dct in sentence_emotion_group_dct.items():
        max_len = max(len(group_dct[1]), len(group_dct[2]), len(group_dct[3]), len(group_dct[4]))
        emotion = sentence_emotion.split('_')[1]
        label = emotion_label_dct[emotion]

        # 各グループを最大長に合わせて拡張
        for group_num in [1, 2, 3, 4]:
            filename_lst = group_dct[group_num]
            random.shuffle(filename_lst)
            length = len(filename_lst)
            
            if (length < max_len):
                needed = max_len - length
                if (needed == 0):
                    continue

                extended = []
                while len(extended) < needed:
                    # 元のグループから循環的に追加
                    extended.extend(filename_lst[:min(len(filename_lst), needed - len(extended))])
                # 拡張したデータをシャッフルして追加
                group_dct[group_num] = filename_lst + extended
                random.shuffle(group_dct[group_num])

        for i in range(max_len):
            train_num = int(max_len * 0.8)
            if (i < train_num):
                epoch_train_data.append([group_dct[1][i], group_dct[2][i], group_dct[3][i], group_dct[4][i], label])
            else:
                epoch_val_data.append([group_dct[1][i], group_dct[2][i], group_dct[3][i], group_dct[4][i], label])

    return epoch_train_data, epoch_val_data



# 指定されたファイルの各データに対して前処理を行いテンソルで返す
def pre_process(filename, input_modality, processor):
    # ファイルパス構築
    parent_dir = "AudioWAV" if input_modality == "audio" else "VideoFlash"
    file_path = f"../data/CREMA-D/raw/{parent_dir}/{filename}"
    
    # モダリティ別処理
    if (input_modality == "audio"):
        waveform, sr = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        result = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    
    elif (input_modality == "video"):
        vr = VideoReader(file_path)
        indices = np.linspace(0, len(vr) - 1, 16).astype(int)
        frames = [vr[i].asnumpy() for i in indices]
        result = processor(frames, return_tensors="pt")
    
    return result