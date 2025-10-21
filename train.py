import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model.train_model import Model

import os
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
date = datetime.now().strftime("%Y%m%d_%H%M%S")

from utils.utility import set_seed
from utils.train_dataset_CREMAD import CREMADDataset

print(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--dataset_name", default="MOSI", type=str)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--audio_pretrained_model_file", type=str)
    parser.add_argument("--text_pretrained_model_file", type=str)
    parser.add_argument("--video_pretrained_model_file", type=str)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
        dropout_rate=args.dropout_rate,
        dataset_name=args.dataset_name,
        audio_pretrained_model_file=args.audio_pretrained_model_file,
        text_pretrained_model_file=args.text_pretrained_model_file,
        video_pretrained_model_file=args.video_pretrained_model_file
    )
    
    # TensorBoard Writer設定
    log_dir = os.path.join("runs", "train", f"{args.dataset_name}_seed{args.seed}_dropout{args.dropout_rate}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    if (args.dataset_name == "CREMA-D"):
        train_dataset = CREMADDataset("train")
        val_dataset = CREMADDataset("val")
    elif (args.dataset_name == "MOSI"):
        train_dataset = MOSIDataset(dataset=args.dataset_name, split="train", class_num=args.class_num)
        val_dataset = MOSIDataset(dataset=args.dataset_name, split="valid", class_num=args.class_num)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(val_dataset))

    # モデル全体をGPUに移動 
    model = model.to(device)

    best_acc = 0.0
    task_loss_lst = []
    acc_lst = []

    for epoch in tqdm(range(args.epochs)):
        model.train()
        avg_task_loss = []

        for batch in tqdm(train_dataloader):
            if (args.dataset_name == "CREMA-D"):
                audio_x, video_x, audio_attn_mask, video_attn_mask, label, _ = batch
                text_x, text_attn_mask = None, None
                _ = _.to(device)
            elif (args.dataset_name == "MOSI"):
                audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask, label = batch
                text_x = text_x.to(device)
                text_attn_mask = text_attn_mask.to(device)
            audio_x = audio_x.to(device)
            video_x = video_x.to(device)
            audio_attn_mask = audio_attn_mask.to(device)
            video_attn_mask = video_attn_mask.to(device)
            label = label.to(device)

            with autocast():
                logits = model(audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask)
                task_loss = F.cross_entropy(logits, label)

            avg_task_loss.append(task_loss.item())

            optimizer.zero_grad()
            scaler.scale(task_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        epoch_task_loss = np.mean(avg_task_loss)
        task_loss_lst.append(epoch_task_loss)

        # TensorBoard: エポックレベルでの記録
        writer.add_scalars('Loss/Train/Epoch/task_Losses', {'Task': epoch_task_loss}, epoch)
        # writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch}, CrossEntropy_loss: {epoch_task_loss}")


        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            group_num_dct = {0:0, 1:0, 2:0, 3:0}
            group_correct_dct = {0:0, 1:0, 2:0, 3:0}
            for _, batch in enumerate(tqdm(val_dataloader)):
                if (args.dataset_name == "CREMA-D"):
                    audio_x, video_x, audio_attn_mask, video_attn_mask, label, group_label = batch
                    text_x, text_attn_mask = None, None
                    group_label = group_label.to(device)
                elif (args.dataset_name == "MOSI"):
                    audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask, label = batch
                    text_x = text_x.to(device)
                    text_attn_mask = text_attn_mask.to(device)
                audio_x = audio_x.to(device)
                video_x = video_x.to(device)
                audio_attn_mask = audio_attn_mask.to(device)
                video_attn_mask = video_attn_mask.to(device)
                label = label.to(device)

                logits = model(audio_x, text_x, video_x, audio_attn_mask, text_attn_mask, video_attn_mask)
                
                # 予測クラスを取得
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == label).sum().item()
                total += label.size(0)

                # グループごとの問題数、正解数をカウント
                if (args.dataset_name == "CREMA-D"):
                    for i in range(label.size(0)):
                        group = group_label[i].item()
                        group_num_dct[group] += 1
                        if (predictions[i] == label[i]):
                            group_correct_dct[group] += 1

        accuracy = correct / total
        acc_lst.append(accuracy)

        writer.add_scalar('Accuracy/Val', accuracy, epoch)
        print(f"Epoch {epoch} Accuracy: {accuracy:.4f}")

        if (args.dataset_name == "CREMA-D"):
            for group in group_num_dct.keys():
                print(f"Group{group}:  {group_correct_dct[group]} / {group_num_dct[group]}")

        if (accuracy >= best_acc):
            best_acc = accuracy
            os.makedirs("saved_models/train/", exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/train/{args.dataset_name}_classNum{args.class_num}_{date}_epoch{epoch}_{accuracy:.4f}_seed{args.seed}_dropout{args.dropout_rate}.pth")
            print(f"We've saved the new model (Accuracy: {accuracy:.4f})")
        print("----------------------------------------------------------------------------")

    print(f"Best Accuracy: {best_acc:.4f}")

    # 最終的な結果をTensorBoardに記録
    writer.add_hparams({
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'dataset_name': args.dataset_name,
        'audio_pretrained_model_file': args.audio_pretrained_model_file,
        'text_pretrained_model_file': args.text_pretrained_model_file,
        'video_pretrained_model_file': args.video_pretrained_model_file,
    }, {
        'best_accuracy': best_acc,
    })

    writer.close()
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)