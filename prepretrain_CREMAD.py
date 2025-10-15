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
from datetime import datetime
date = datetime.now().strftime("%Y%m%d_%H%M%S")

from utils.utility import set_seed
from utils.prepretrain_dataset_CREMAD import CREMADDataProvider, CREMADDataset

print(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--dataset_name", default="CREMA-D", type=str)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--input_modality", default="audio", type=str, help="audio or text or video")
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

    data_provider = CREMADDataProvider()
    train_data, val_data = data_provider.get_dataset()
    train_dataset = CREMADDataset(train_data, input_modality=args.input_modality)
    val_dataset = CREMADDataset(val_data, input_modality=args.input_modality)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(val_dataset))
    
    # モデル全体をGPUに移動
    model = model.to(device)

    acc_lst = []
    loss_lst = []
    
    # Early Stopping用の変数
    best_acc = 0.0
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

            loss = F.cross_entropy(y, label)
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


        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            y_max_values = []   # y の最大値を保存するリスト

            for batch in tqdm(val_dataloader):
                x, attn_mask, label = batch
                x = x.to(device)
                attn_mask = attn_mask.to(device)
                label = label.to(device)

                # モデルの順伝搬
                y = model(x, attn_mask)

                y = F.softmax(y, dim=1)
                predictions = y.argmax(dim=1)
                correct += (predictions == label).sum().item()
                total   += label.size(0)

                # y の各サンプルの最大値を取得
                max_values = y.max(dim=1)[0]   # [batch_size] の Tensor
                y_max_values.extend(max_values.cpu().tolist())

            print("correct, total:", correct, total)

            # y の最大値の平均を計算
            avg_y_max = np.mean(y_max_values)
            print(f"Average of y max values: {avg_y_max:.4f}")

        acc = correct / total
        acc_lst.append(acc)

        writer.add_scalar('Accuracy/Test', acc, epoch)
        print(f"Epoch {epoch} Acc: {acc}")

        # Best model保存
        if (acc > best_acc):
            best_acc = acc
            patience_counter = 0
            
            os.makedirs("saved_models/prepretrain/" + args.input_modality, exist_ok=True)
            torch.save(model.state_dict(),
                       f"saved_models/prepretrain/{args.input_modality}/{args.dataset_name}_epoch{epoch}_{date}_{acc:.4f}_seed{args.seed}.pth")
            print(f"We've saved the new model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
            
            # Early Stopping判定
            if (patience_counter >= args.patience):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
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