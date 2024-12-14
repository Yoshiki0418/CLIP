import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from .ViT import ViT_B32
from .text_encoder import TextEncoder

class ProjectionHead(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            projection_dim: int,
            drop_out: float,
        ) -> None:
        super().__init__()

        self.projection = nn.Linear(emb_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        # 残差接続（スケーリング付き）
        x = (x + projected) * (1 / torch.sqrt(torch.tensor(2.0, device=x.device)))
        x = self.norm(x)

        return x
    
class CLIP_Module(nn.Module):
    def __init__(
            self,
            img_size: int,                    # 画像のサイズ（例: 224）
            patch_size: int,                  # パッチサイズ（例: 32）
            embed_dim: int,                   # 埋め込み次元（例: 512）
            num_layers: int,                  # トランスフォーマーの層数（例: 12）
            num_heads: int,                   # アテンションヘッドの数（例: 8）
            text_model_name: str,             # テキストエンコーダのモデル名
            image_embedding_dims: int,        # 画像エンコーダ出力の埋め込み次元
            text_embedding_dims: int,         # テキストエンコーダ出力の埋め込み次元
            projection_dims: int,             # 射影ヘッドの出力次元（例: 512）
            dropout: float,                   # ドロップアウト率（例: 0.1）
            dim_head: int,                    # Attentionの各ヘッドの次元数

    ) -> None:
        super().__init__()

        self.image_encoder = ViT_B32(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_head=dim_head,
        )

        self.text_encoder = TextEncoder(model_name=text_model_name)

        self.image_projection = ProjectionHead(
            emb_dim=image_embedding_dims,
            projection_dim=projection_dims,
            drop_out=dropout,
        )
        self.text_projection = ProjectionHead(
            emb_dim=text_embedding_dims,
            projection_dim=projection_dims,
            drop_out=dropout,
        )
        
    def encode_text(self, input_ids, attention_mask):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_embeddings = self.text_projection(text_features)
        return text_embeddings
    
    def encode_image(self, image):
        image_features = self.image_encoder(image)
        image_embeddings = self.image_projection(image_features)
        return image_embeddings
    
    def forward(self, inputs):
        image_embeddings = self.encode_image(inputs["image"])
        text_embeddings = self.encode_text(inputs["input_ids"], inputs["attention_mask"])
        return image_embeddings, text_embeddings
    
# 損失関数（例: コサイン類似度に基づくコントラスト損失）
class ContrastiveLoss(nn.Module):
    def __init__(self, initial_temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, img_features, text_features):
        # L2ノルム正規化
        img_features = F.normalize(img_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # コサイン類似度計算
        logits_per_image = torch.matmul(img_features, text_features.T) / self.temperature
        logits_per_text = logits_per_image.T

        # 正例のラベルを定義
        labels = torch.arange(img_features.size(0)).to(img_features.device)
        loss_img = nn.functional.cross_entropy(logits_per_image, labels)
        loss_text = nn.functional.cross_entropy(logits_per_text, labels)
        return (loss_img + loss_text) / 2