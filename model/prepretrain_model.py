import torch
import torch.nn as nn
from transformers import WavLMModel, VideoMAEModel


class PrepretrainModel(nn.Module):
    def __init__(self, input_modality: str, hidden_dim: int, num_classes: int):
        super(PrepretrainModel, self).__init__()

        self.input_modality = input_modality
        self.hidden_dim = hidden_dim

        self.activation = nn.ReLU()

        self.num_classes = num_classes

        if input_modality == "audio":
            self.encoder_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        elif input_modality == "video":
            self.encoder_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.decoder = nn.Linear(self.hidden_dim, self.num_classes)


    def forward(self, x, attn_mask):
        f = self.encoder_model(input_values=x, attention_mask=attn_mask)

        if (self.input_modality == "audio"): 
            outputs = self.encoder_model(input_values=x,
                                         attention_mask=attn_mask,
                                         return_dict=True)
            hidden = outputs.last_hidden_state
            f = hidden.mean(dim=1)
        elif (self.input_modality == "video"):
            f = self.encoder_model.last_hidden_state[:, 0, :]
        f = self.layer_norm(f)

        y = self.decoder(f)

        return y