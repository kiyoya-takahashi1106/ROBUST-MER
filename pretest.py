import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.pretrain_model import PretrainModel
import argparse
from utils.utility import set_seed
from utils.pretrain_dataset import CREMADDataset
from tqdm import tqdm
import numpy as np


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset_name", default="CREMA-D", type=str)
    parser.add_argument("--prepretrained_dataset", default="CREMA-D", type=str)
    parser.add_argument("--prepretrained_classnum", default=6, type=int)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--input_modality", default="audio", type=str, help="audio or video")
    parser.add_argument("--trained_model_file", default="", required=True, type=str)
    parser.add_argument("--hidden_dim", default=768, type=int)
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) モデル生成（学習時と同じ引数で）
    model = PretrainModel(
        input_modality=args.input_modality, 
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
        dropout_rate=0.0,
        prepretrained_dataset=args.prepretrained_dataset,
        prepretrained_classnum=args.prepretrained_classnum,
        pretrained_model_file="test.pth"
    )
    model = model.to(device)

    # 2) チェックポイント読込
    trained_model_path = f"./saved_models/pretrain/{args.input_modality}/{args.trained_model_file}"
    ckpt = torch.load(trained_model_path, map_location=device)

    # 3) state_dict取り出し
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else (
        ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    )

    # 4) ロード
    model.load_state_dict(state, strict=True)

    # 5) 推論モード
    model.eval()

    test_dataset = CREMADDataset(
        split="test", 
        input_modality=args.input_modality, 
        epoch=0, 
        prepretrained_dataset=args.prepretrained_dataset, 
        prepretrained_classnum=args.prepretrained_classnum
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Test dataset size:", len(test_dataset))

    correct = 0
    total = 0
    y_max_values = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            group1, group2, group3, group4, attn_mask1, attn_mask2, attn_mask3, attn_mask4, label = batch
            group1 = group1.to(device)
            group2 = group2.to(device)
            group3 = group3.to(device)
            group4 = group4.to(device)
            attn_mask1 = attn_mask1.to(device)
            attn_mask2 = attn_mask2.to(device)
            attn_mask3 = attn_mask3.to(device)
            attn_mask4 = attn_mask4.to(device)
            label = label.to(device)

            y, *_ = model(group1, group2, group3, group4, attn_mask1, attn_mask2, attn_mask3, attn_mask4)
            y = F.softmax(y, dim=1)
            predictions = y.argmax(dim=1)
            correct += (predictions == label).sum().item()
            total += label.size(0)
            max_values = y.max(dim=1)[0]
            y_max_values.extend(max_values.cpu().tolist())

    acc = correct / total if total > 0 else 0
    avg_y_max = np.mean(y_max_values) if y_max_values else 0
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Average of y max values: {avg_y_max:.4f}")

if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    test(_args)