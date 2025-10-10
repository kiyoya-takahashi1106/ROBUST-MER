import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model.pretrain_model import PretrainModel

import os
import numpy as np
import argparse
from tqdm import tqdm

from utils.utility import set_seed
from utils.pretrain_dataset import CREMADDataProvider, CREMADDataset
from utils.function import COSLOSS, DIFFLOSS, MSELOSS

print(torch.__version__)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--dataset_name", default="CREMA-D", type=str)
    parser.add_argument("--class_num", default=6, type=int)
    parser.add_argument("--input_modality", default="audio", type=str, help="audio or video")
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--weight_sim", default=1.0, type=float)
    parser.add_argument("--weight_diff", default=1.0, type=float)
    parser.add_argument("--weight_recon", default=1.0, type=float)
    parser.add_argument("--weight_task", default=1.0, type=float)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainModel(
        input_modality=args.input_modality, 
        hidden_dim=args.hidden_dim, 
        num_classes=args.class_num, 
        dropout_rate=args.dropout_rate
    )
    
    # TensorBoard Writer設定
    log_dir = os.path.join("runs", "pretrain", args.input_modality, f"{args.dataset_name}_seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    # モデル全体をGPUに移動
    model = model.to(device)

    acc_lst = []
    sim_loss_lst = []
    diff_loss_lst = []
    recon_loss_lst = []
    task_loss_lst = []
    loss_lst = []

    for epoch in tqdm(range(args.epochs)):
        model.train()
        avg_sim_loss = []
        avg_diff_loss = []
        avg_recon_loss = []
        avg_task_loss = []
        avg_loss = []

        # dataloaderの準備
        data_provider = CREMADDataProvider()
        train_data, val_data = data_provider.get_dataset()
        train_dataset = CREMADDataset(train_data, input_modality=args.input_modality)
        val_dataset = CREMADDataset(val_data, input_modality=args.input_modality)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        for batch in tqdm(train_dataloader):
            # バッチから画像、テキスト、ラベルを取得
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

            # モデルの順伝搬
            y, f_lst, s_lst, p_lst, r_lst = model(group1, group2, group3, group4, attn_mask1, attn_mask2, attn_mask3, attn_mask4)

            # 損失計算
            sim_loss = args.weight_sim * COSLOSS(s_lst)
            diff_loss = args.weight_diff * DIFFLOSS(s_lst, p_lst)
            recon_loss = args.weight_recon * MSELOSS(f_lst, r_lst)
            task_loss = args.weight_task * F.cross_entropy(y, label)

            loss = sim_loss + diff_loss + recon_loss + task_loss

            avg_sim_loss.append(sim_loss.item())
            avg_diff_loss.append(diff_loss.item())
            avg_recon_loss.append(recon_loss.item())
            avg_task_loss.append(task_loss.item())
            avg_loss.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        epoch_sim_loss = np.mean(avg_sim_loss)
        epoch_diff_loss = np.mean(avg_diff_loss)
        epoch_recon_loss = np.mean(avg_recon_loss)
        epoch_task_loss = np.mean(avg_task_loss)
        epoch_loss = np.mean(avg_loss)

        sim_loss_lst.append(epoch_sim_loss)
        diff_loss_lst.append(epoch_diff_loss)
        recon_loss_lst.append(epoch_recon_loss)
        task_loss_lst.append(epoch_task_loss)
        epoch_avg_loss = epoch_loss
        loss_lst.append(epoch_avg_loss)

        # TensorBoard: エポックレベルでの記録
        writer.add_scalars('Loss/Train/Epoch/individual_Losses', {
            'Similarity': epoch_sim_loss,
            'Difference': epoch_diff_loss,
            'Reconstruction': epoch_recon_loss,
            'Task': epoch_task_loss,
        }, epoch)
        writer.add_scalar('Loss/Train/Epoch/Total', epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch}, loss: {epoch_avg_loss}, sim_loss: {epoch_sim_loss}, diff_loss: {epoch_diff_loss}, recon_loss: {epoch_recon_loss}, task_loss: {epoch_task_loss}")


        # Test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, batch in enumerate(tqdm(val_dataloader)):
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

                y, f_lst, s_lst, p_lst, r_lst = model(group1, group2, group3, group4, attn_mask1, attn_mask2, attn_mask3, attn_mask4)

                predictions = y.argmax(dim=1)
                correct += (predictions == label).sum().item()
                total   += label.size(0)

            print("correct, total:", correct, total)

        acc = correct / total
        acc_lst.append(acc)

        writer.add_scalar('Accuracy/Test', acc, epoch)
        print(f"Epoch {epoch} Acc: {acc}")

        if (acc >= max(acc_lst)):
            os.makedirs("saved_models/pretrain/" + args.input_modality, exist_ok=True)
            torch.save(model.state_dict(),
                       f"saved_models/pretrain/{args.input_modality}/{args.dataset_name}_epoch{epoch}_{acc:.4f}_seed{args.seed}.pth")
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