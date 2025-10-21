import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.train_model import Model
import argparse
from utils.utility import set_seed
from utils.train_dataset_CREMAD import CREMADDataset
from tqdm import tqdm
import numpy as np


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset_name", default="CREMA-D", type=str)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--trained_model_file", required=True, type=str)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--audio_pretrained_model_file", default="test.pth", type=str)
    parser.add_argument("--text_pretrained_model_file", default="test.pth", type=str)
    parser.add_argument("--video_pretrained_model_file", default="test.pth", type=str)
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
        dropout_rate=0.0,
        dataset_name=args.dataset_name,
        audio_pretrained_model_file=args.audio_pretrained_model_file,
        text_pretrained_model_file=args.text_pretrained_model_file,
        video_pretrained_model_file=args.video_pretrained_model_file
    )
    model = model.to(device)

    trained_model_path = f"./saved_models/train/{args.trained_model_file}"
    ckpt = torch.load(trained_model_path, map_location=device)

    model.load_state_dict(ckpt, strict=True)
    model.eval()

    if (args.dataset_name == "CREMA-D"):
        test_dataset = CREMADDataset(split="test")
    elif (args.dataset_name == "MOSI"):
        test_dataset = MOSIDataset(dataset=args.dataset_name, split="test", class_num=args.class_num)
    else:
        raise ValueError("Unknown dataset_name")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Test dataset size:", len(test_dataset))

    correct = 0
    total = 0
    group_num_dct = {0:0, 1:0, 2:0, 3:0}
    group_correct_dct = {0:0, 1:0, 2:0, 3:0}
    y_max_values = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            if args.dataset_name == "CREMA-D":
                audio_x, video_x, audio_attn_mask, video_attn_mask, label, group_label = batch
                text_x, text_attn_mask = None, None
                group_label = group_label.to(device)
            elif args.dataset_name == "MOSI":
                audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask, label = batch
                text_x = text_x.to(device)
                text_attn_mask = text_attn_mask.to(device)
            audio_x = audio_x.to(device)
            video_x = video_x.to(device)
            audio_attn_mask = audio_attn_mask.to(device)
            video_attn_mask = video_attn_mask.to(device)
            label = label.to(device)

            logits = model(audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask)
            y = F.softmax(logits, dim=1)
            predictions = y.argmax(dim=1)
            correct += (predictions == label).sum().item()
            total += label.size(0)
            max_values = y.max(dim=1)[0]
            y_max_values.extend(max_values.cpu().tolist())

            # グループごとの問題数、正解数をカウント
            if (args.dataset_name == "CREMA-D"):
                for i in range(label.size(0)):
                    group = group_label[i].item()
                    group_num_dct[group] += 1
                    if (predictions[i] == label[i]):
                        group_correct_dct[group] += 1

    acc = correct / total if total > 0 else 0
    avg_y_max = np.mean(y_max_values) if y_max_values else 0

    if (args.dataset_name == "CREMA-D"):
        for group in group_num_dct.keys():
            print(f"Group{group}:  {group_correct_dct[group]} / {group_num_dct[group]}")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Average of y max values: {avg_y_max:.4f}")

if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    test(_args)