import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from aaa.train_CREMAD_model import Model
from datetime import datetime
date = datetime.now().strftime("%Y%m%d_%H%M%S")

import os
import numpy as np
import argparse
from tqdm import tqdm

from utils.utility import set_seed
from aaa.train_dataset import CREMADDataProvider, CREMADDataset



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
    parser.add_argument("--video_pretrained_model_file", type=str)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
        dropout_rate=args.dropout_rate,
        audio_pretrained_model_file=args.audio_pretrained_model_file,
        video_pretrained_model_file=args.video_pretrained_model_file
    )
    
    # TensorBoard Writer設定
    log_dir = os.path.join("runs", "train", f"{args.dataset_name}", f"seed{args.seed}")
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
            # バッチから画像、テキスト、ラベルを取得
            audio_x, video_x, audio_attn_mask, video_attn_mask, label = batch
            audio_x = audio_x.to(device)
            video_x = video_x.to(device)
            audio_attn_mask = audio_attn_mask.to(device)
            video_attn_mask = video_attn_mask.to(device)
            label = label.to(device)
            
            logits = model(audio_x, video_x, audio_attn_mask, video_attn_mask)

            # クロスエントロピー損失
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
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch}, CrossEntropy_loss: {epoch_task_loss}")


        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, batch in enumerate(tqdm(val_dataloader)):
                audio_x, video_x, audio_attn_mask, video_attn_mask, label = batch
                audio_x = audio_x.to(device)
                video_x = video_x.to(device)
                audio_attn_mask = audio_attn_mask.to(device)
                video_attn_mask = video_attn_mask.to(device)
                label = label.to(device).long()

                logits = model(audio_x, video_x, audio_attn_mask, video_attn_mask)
                
                # 予測クラスを取得
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == label).sum().item()
                total += label.size(0)

        accuracy = correct / total
        acc_lst.append(accuracy)

        writer.add_scalar('Accuracy/Val', accuracy, epoch)
        print(f"Epoch {epoch} Accuracy: {accuracy:.4f}")

        if (accuracy >= best_acc):
            best_acc = accuracy
            os.makedirs("saved_models/train/" + args.dataset_name, exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/train/{args.dataset_name}/epoch{epoch}_{accuracy:.4f}_seed{args.seed}.pth")
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