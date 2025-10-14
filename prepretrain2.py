# MOSIでfinetuning

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
from utils.prepretrain_dataset2 import MOSIDataset

print(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--dataset_name", default="MOSI", type=str)
    parser.add_argument("--class_num", default=1, type=int)
    parser.add_argument("--input_modality", default="audio", type=str, help="audio or video")
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--patience", default=5, type=int, help="Early stopping patience")
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

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    train_dataset = MOSIDataset(split="train", dataset=args.dataset_name, input_modality=args.input_modality)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = MOSIDataset(split="valid", dataset=args.dataset_name, input_modality=args.input_modality)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))

    # モデル全体をGPUに移動
    model = model.to(device)

    mae_lst = []
    loss_lst = []
    
    # Early Stopping用の変数
    best_mae = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        avg_loss = []

        for batch in tqdm(train_dataloader):
            x, attn_mask, label = batch
            x = x.to(device)
            attn_mask = attn_mask.to(device)
            label = label.to(device)
            
            y = model(x, attn_mask)
            y = y.squeeze(-1)

            loss = F.mse_loss(y, label)
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
        print(f"Epoch {epoch}, MSE_loss: {epoch_loss}")


        # Validation
        model.eval()
        with torch.no_grad():
            total_mae = 0.0
            
            for batch in tqdm(valid_dataloader):
                x, attn_mask, label = batch
                x = x.to(device)
                attn_mask = attn_mask.to(device)
                label = label.to(device)

                y = model(x, attn_mask)
                y = y.squeeze(-1)

                # MAE を計算
                mae = torch.abs(y - label).sum().item()
                total_mae += mae

        avg_mae = total_mae / len(valid_dataset)
        mae_lst.append(avg_mae)

        writer.add_scalar('MAE/Test', avg_mae, epoch)
        print(f"Epoch {epoch} MAE: {avg_mae:.4f}")

        # Best model保存
        if (avg_mae < best_mae):
            best_mae = avg_mae
            patience_counter = 0
            
            os.makedirs("saved_models/prepretrain/" + args.input_modality, exist_ok=True)
            torch.save(model.state_dict(),
                       f"saved_models/prepretrain/{args.input_modality}/{args.dataset_name}_epoch{epoch}_{avg_mae:.4f}_seed{args.seed}.pth")
            print(f"We've saved the new model (MAE: {avg_mae:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
            
            # Early Stopping判定
            if (patience_counter >= args.patience):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        print("----------------------------------------------------------------------------")

    print(f"Best MAE: {min(mae_lst):.4f}")

    # 最終的な結果をTensorBoardに記録
    writer.add_hparams({
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset_name': args.dataset_name,
        'input_modality': args.input_modality,
    }, {
        'best_mae': min(mae_lst),
    })

    writer.close()
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)