import os, sys
import numpy
import hydra
import torch
from torch import optim
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
from termcolor import cprint

from src.utils import set_seed
from src.model import CLIP_Module, ContrastiveLoss
from src.datasets import ImageRetrievalDataModule
from src.metrics import compute_top_k_accuracy
from tools import *

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir= hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    #------------------------------------
    #           DataLoader
    #------------------------------------
    dataset_args = {
        "artifact_dir": args.datasets.artifact_dir,
        "dataset_name": args.datasets.dataset_name,
        "val_split": args.datasets.val_split,
        "tokenizer_alias": args.datasets.tokenizer_alias,
        "target_size": args.datasets.target_size,
        "max_length": args.datasets.max_length,
        "train_batch_size": args.datasets.train_batch_size,
        "val_batch_size": args.datasets.val_batch_size,
        "num_workers": args.datasets.num_workers,
    }

    data_module = ImageRetrievalDataModule(**dataset_args)

    # データモジュールのセットアップ
    data_module.setup()

    # トレーニングデータローダー
    train_dataloader = data_module.train_dataloader()
    print("Train DataLoader Initialized:")
    for batch in train_dataloader:
        print("Batch keys:", batch.keys())
        print("Image Tensor Shape:", batch["image"].shape)
        print("Caption:", batch["caption"])
        break  # 確認用に1バッチのみ出力

    # 検証データローダー
    val_dataloader = data_module.val_dataloader()
    print("Validation DataLoader Initialized:")
    for batch in val_dataloader:
        print("Batch keys:", batch.keys())
        print("Image Tensor Shape:", batch["image"].shape)
        print("Caption:", batch["caption"])
        break  # 確認用に1バッチのみ出力

    #------------------------------------
    #            Model
    #------------------------------------
    model = CLIP_Module(**args.model)
    model.to(args.device)

    #------------------------------------
    #          Optimizer
    #------------------------------------
    optimizer = optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": 1e-5},
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": model.image_projection.parameters(), "lr": 3e-4},
        {"params": model.text_projection.parameters(), "lr": 3e-4},
    ], weight_decay=1e-2)

    #------------------------------------
    #         Start Training
    #------------------------------------
    criterion = ContrastiveLoss(temperature=0.07)

    max_val_top_k_acc = 0.0

    writer = WandBMetricsWriter(project_name = f"{args.writer.project_name}",
                                    model_name = args.writer.name)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, val_loss, train_top_k_acc, val_top_k_acc = [], [], [], []

        model.train()

        for batch in tqdm(train_dataloader, desc="Train"):
            # バッチデータを取得
            images = batch["image"].to(args.device) 
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            # 勾配をリセット
            optimizer.zero_grad()

            # 順伝播
            image_embeddings, text_embeddings = model({
                "image": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })

            # 損失計算
            loss = criterion(image_embeddings, text_embeddings)
            train_loss.append(loss.item())

            # 類似度行列計算
            similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)

            # Top-k Accuracy計算
            labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
            top_k_acc = compute_top_k_accuracy(similarity_matrix, labels, k=5)
            train_top_k_acc.append(top_k_acc)

            optimizer.zero_grad()

            # 逆伝播
            loss.backward()

            # パラメータ更新
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                images = batch["image"].to(args.device)
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)

                # 順伝播
                image_embeddings, text_embeddings = model({
                    "image": images,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                })

                # 損失計算
                loss = criterion(image_embeddings, text_embeddings)
                val_loss.append(loss.item())

                similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)
                labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
                top_k_acc = compute_top_k_accuracy(similarity_matrix, labels, k=5)
                val_top_k_acc.append(val_top_k_acc)
                print("Now")

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train top_k_acc: {np.mean(train_top_k_acc)} | val loss: {np.mean(val_loss):.3f} | val top_k_acc: {np.mean(val_top_k_acc)} ")

        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))

        if np.mean(val_top_k_acc) > max_val_top_k_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_top_k_acc = np.mean(val_top_k_acc)
        
        writer(
            epoch = epoch,
            train_loss = np.mean(train_loss),
            train_top_k_acc = np.mean(train_top_k_acc),
            val_loss = np.mean(val_loss),
            val_top_k_acc = np.mean(val_top_k_acc)
        )




if __name__ == "__main__":
    run()
