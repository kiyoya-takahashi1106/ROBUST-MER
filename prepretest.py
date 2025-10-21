import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.prepretrain_model import PrepretrainModel
import argparse
from utils.utility import set_seed
from utils.prepretrain_dataset_CREMAD import CREMADDataset
from tqdm import tqdm
import numpy as np


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset_name", default="CREMA-D", type=str)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--input_modality", default="audio", type=str, help="audio or video")
    parser.add_argument("--trained_model_file", default="", required=True, type=str)
    parser.add_argument("--hidden_dim", default=768, type=int)
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # モデル生成
    model = PrepretrainModel(
        input_modality=args.input_modality, 
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
        dropout_rate=0.0,
        dataset=args.dataset_name
    )
    model = model.to(device)

    # チェックポイント読込
    trained_model_path = f"./saved_models/prepretrain/{args.input_modality}/{args.trained_model_file}"
    state = torch.load(trained_model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.eval()

    test_dataset = CREMADDataset(
        split="test", 
        input_modality=args.input_modality
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Test dataset size:", len(test_dataset))

    correct = 0
    total = 0
    y_max_values = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            x, attn_mask, label = batch
            x = x.to(device)
            attn_mask = attn_mask.to(device)
            label = label.to(device)

            y = model(x, attn_mask)
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