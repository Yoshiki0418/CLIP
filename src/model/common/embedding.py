import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat

#----------------------------------------
#  Linear Projection of Flattened Patches
#----------------------------------------
class Patch_Embedding(nn.Module):
    def __init__(
            self,
            img_size: int,
            patch_size: int,
            emb_dim: int,
            channels: int = 3,
        ) -> None:
        super().__init__()

        image_height = img_size
        image_width = img_size
        patch_height = patch_size
        patch_width = patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_embedding = nn.Sequential(
            # 平坦化(Flatten)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(img)
        return  x

#--------------------------------
#     Positional Embedding
#--------------------------------
class Positional_Embedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        emb_dim: int,
    ) -> None:
        """
        Args:
            seq_len (int): Length of the input sequence (number of tokens).
            d_model (int): Dimension of each embedding vector for the tokens.
        """
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add position embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Input tensor with position embeddings added, shape [batch_size, seq_len, d_model].
        """
        return x + self.position_embedding
    
#-----------------------------------------
#   Extra learnable [class] embedding
#-----------------------------------------
class Extra_learnable_Embedding(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        return x