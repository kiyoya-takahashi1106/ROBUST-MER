import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model.prepretrain_model import PrepretrainModel

import os
import numpy as np
import argparse
from tqdm import tqdm

from utils.utility import set_seed
from utils.prepretrain_dataset import CREMADDataProvider, CREMADDataset

print(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--dataset_name", default="CREMA-D", type=str)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--input_modality", default="audio", type=str, help="audio or video")
    parser.add_argument("--hidden_dim", default=768, type=int)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrepretrainModel(
        input_modality=args.input_modality, 
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
    )
    
    # TensorBoard Writer設定
    log_dir = os.path.join("runs", "prepretrain", args.input_modality, f"{args.dataset_name}_seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    data_provider = CREMADDataProvider()
    train_data, val_data = data_provider.get_dataset()
    train_dataset = CREMADDataset(train_data, input_modality=args.input_modality)
    val_dataset = CREMADDataset(val_data, input_modality=args.input_modality)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    # モデル全体をGPUに移動
    model = model.to(device)

    acc_lst = []
    loss_lst = []

    for epoch in tqdm(range(args.epochs)):
        model.train()
        avg_loss = []

        for batch in tqdm(train_dataloader):
            # バッチから画像、テキスト、ラベルを取得
            x, attn_mask, label = batch
            x = x.to(device)
            attn_mask = attn_mask.to(device)
            label = label.to(device)

            # モデルの順伝搬
            y = model(x, attn_mask)

            loss =F.cross_entropy(y, label)

            avg_loss.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        epoch_loss = np.mean(avg_loss)

        loss_lst.append(epoch_loss)

        # TensorBoard: エポックレベルでの記録
        writer.add_scalars('Loss/Train/Epoch/Task_Losses', {
            'Task': epoch_loss,
        }, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch}, loss: {epoch_loss})")


        # Test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, batch in enumerate(tqdm(val_dataloader)):
                x, attn_mask, label = batch
                x = x.to(device)
                attn_mask = attn_mask.to(device)
                label = label.to(device)

                # モデルの順伝搬
                y = model(x, attn_mask)

                predictions = y.argmax(dim=1)
                correct += (predictions == label).sum().item()
                total   += label.size(0)

            print("correct, total:", correct, total)

        acc = correct / total
        acc_lst.append(acc)

        writer.add_scalar('Accuracy/Test', acc, epoch)
        print(f"Epoch {epoch} Acc: {acc}")

        if (acc >= max(acc_lst)):
            os.makedirs("saved_models/prepretrain/" + args.input_modality, exist_ok=True)
            torch.save(model.state_dict(),
                       f"saved_models/prepretrain/{args.input_modality}/{args.dataset_name}_epoch{epoch}_{acc:.4f}_seed{args.seed}.pth")
            print(f"We’ve saved the new model.")
        print("----------------------------------------------------------------------------")

    print("best acc: ", max(acc_lst))

    # 最終的な結果をTensorBoardに記録
    writer.add_hparams({
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset_name': args.dataset_name,
        'input_modality': args.input_modality,
    }, {
        'best_accuracy': max(acc_lst),
    })

    writer.close()
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)