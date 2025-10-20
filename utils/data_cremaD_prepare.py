# 1. 感情レベルだけ違う (actor, sentence, emotion同じ) のデータ群を 一番感情レベルの高いものに統合する.
# 2. actorごとに train/val/test 分割を行う (各actorの年齢・性別のバランスを考慮).
# 3. 2.の結果を用いて、dataを train/val/testに分割する.

import pandas as pd
import random
import os


def eliminate_data():
    actors_file = pd.read_csv('../data/CREMA-D/raw/SentenceFilenames.csv')
    strong_dct = {'LO': 1, 'MD': 2, 'HI': 3, 'XX': 4}

    # グループごとに最大感情レベルのデータを保存
    group_max = {}
    for _, row in actors_file.iterrows():
        filename = row["Filename"]
        part = filename.split('_')
        actor = int(part[0])
        sentence = part[1]
        emotion = part[2]
        emotion_strong = part[3]
        emotion_strong_id = strong_dct[emotion_strong]
        key = (actor, sentence, emotion)

        # 最大感情レベルのデータだけ残す
        if key not in group_max or emotion_strong_id > group_max[key][0]:
            group_max[key] = (emotion_strong_id, row["Stimulus_Number"])

    use_data_id_lst = [v[1] for v in group_max.values()]
    return use_data_id_lst



def split_actor():
    actors_file = pd.read_csv('../data/CREMA-D/raw/VideoDemographics.csv')
    
    group_dct = {"group1": [], "group2": [], "group3": [], "group4": []}
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

    # グループ内でシャッフル
    for group_name, actor_lst in group_dct.items():
        random.shuffle(actor_lst)
    
    train_actor_lst = []
    val_actor_lst = []
    test_actor_lst = []
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    for group_name, actor_lst in group_dct.items():
        total_num = len(actor_lst)
        train_num = int(total_num * train_ratio)
        val_num = int(total_num * val_ratio)
        for i, actor_id in enumerate(actor_lst):
            if (i < train_num):
                train_actor_lst.append(actor_id)
            elif (i < train_num + val_num):
                val_actor_lst.append(actor_id)
            else:
                test_actor_lst.append(actor_id)

    return train_actor_lst, val_actor_lst, test_actor_lst



def split_data(use_data_id_lst, actor_lst):
    datas_file = pd.read_csv('../data/CREMA-D/raw/SentenceFilenames.csv')
    data_name_lst = []   # actor_sentenceCode_emotionCode_strongCode

    for index, row in datas_file.iterrows():
        data_id = row["Stimulus_Number"]
        filename = row["Filename"]
        if (data_id in use_data_id_lst):
            part = filename.split('_')
            actor_id = int(part[0])
            if (actor_id in actor_lst):
                data_name_lst.append(filename)

    return data_name_lst



# main
if (__name__ == "__main__"):
    use_data_id_lst = eliminate_data()
    "----------------------------------------------------------------"
    train_actor_lst, val_actor_lst, test_actor_lst = split_actor()
    train_data_id_lst = split_data(use_data_id_lst, train_actor_lst)
    val_data_id_lst = split_data(use_data_id_lst, val_actor_lst)
    test_data_id_lst = split_data(use_data_id_lst, test_actor_lst)
    "----------------------------------------------------------------"
    for split_name in ["train", "val", "test"]:
        os.makedirs(f'../data/CREMA-D/{split_name}/', exist_ok=True)
        if (split_name == "train"):
            data_id_lst = train_data_id_lst
        elif (split_name == "val"):
            data_id_lst = val_data_id_lst
        else:
            data_id_lst = test_data_id_lst

        for data_name in data_id_lst:
            for modality in ["AudioWAV", "VideoFlash"]: 
                os.makedirs(f'../data/CREMA-D/{split_name}/{modality}/', exist_ok=True)
                src_path = f'../data/CREMA-D/raw/{modality}/{data_name}.{"wav" if modality=="AudioWAV" else "flv"}'
                dst_path = f'../data/CREMA-D/{split_name}/{modality}/{data_name}.{"wav" if modality=="AudioWAV" else "flv"}'
                os.system(f'cp {src_path} {dst_path}')