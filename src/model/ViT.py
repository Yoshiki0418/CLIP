import torch
from torch import nn

from .common import (
    Patch_Embedding, 
    Positional_Embedding,
    Extra_learnable_Embedding,
    Transformer_Encoder_Block,
)

class ViT_B32(nn.Module):
    def __init__(self,
        img_size: int, 
        patch_size: int, 
        embed_dim: int, 
        num_layers: int,
        num_heads: int,
        dim_head: int,
    ) -> None:
        super().__init__()

        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = Patch_Embedding(img_size, patch_size, embed_dim)
        self.cls_tokens = Extra_learnable_Embedding(embed_dim)
        self.pos_emb = Positional_Embedding(seq_len=num_patches+1, d_model=embed_dim)

        self.encoder = nn.ModuleList([
            Transformer_Encoder_Block(embed_dim, num_heads, dim_head) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.cls_tokens(x)

        x = self.pos_emb(x)

        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)
        cls_token = x[:, 0]
        return cls_token

        
        
