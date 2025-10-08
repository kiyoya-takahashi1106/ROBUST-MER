from transformers import Wav2Vec2Processor, VideoMAEImageProcessor
import pandas as pd

from utils.process import cremed_classification, make_data_combination, pre_process


class CREMADDataLoader:
    def __init__(self, input_modality, epoch):
        self.input_modality = input_modality
        self.epoch = epoch
        
        self.sentence_emotion_group_dct = cremed_classification()

        if (self.input_modality == "audio") :
            self.processor = Wav2Vec2Processor.from_pretrained("microsoft/wavlm-base")
        elif (self.input_modality == "video"):
            self.data = self.load_video_data()
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

        self.all_epoch_train_data = []
        self.all_epoch_val_data = []
        for _ in range(self.epoch):
            epoch_train_data, epoch_val_data = make_data_combination(self.sentence_emotion_group_dct)
            self.all_epoch_train_data.append(epoch_train_data)
            self.all_epoch_val_data.append(epoch_val_data)  # 今回は検証データも同じにしているが、実際には分けるべき
        
    