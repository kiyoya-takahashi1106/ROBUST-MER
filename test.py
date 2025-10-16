import torch

path = "./saved_models/prepretrain/video/MOSI_classNum2_epoch10_20251016_134922_0.6594_seed42_dropout0.3.pth"
checkpoint = torch.load(path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint

print("\n--- state_dict keys ---")
for k in state_dict.keys():
    if k.startswith("encoder."):
        print(k)

print("\n--- encoder keys ---")
from transformers import VideoMAEModel
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
encoder = model
for k in encoder.state_dict().keys():
    print(k)