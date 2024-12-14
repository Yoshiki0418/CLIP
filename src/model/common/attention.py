import torch
from torch import nn, einsum
from einops import rearrange

#--------------------------------
#   Multi-Head Self-Attention
#--------------------------------
class MultiHead_Self_Attention(nn.Module):
    """
    dim : int
        入力データの次元数．埋め込み次元数と一致する．
    heads : int
        ヘッドの数．
    dim_head : int
        各ヘッドのデータの次元数．
     dropout : float
        Dropoutの確率(default=0.)．
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # 線形変換
        q, k, v = [rearrange(t, "b (h d) l -> b h d l", h=self.heads) for t in qkv]
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b (h d) l", l=l)
        out = self.to_out(out)
        return out