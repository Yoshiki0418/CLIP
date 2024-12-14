import torch
from torch import nn

#------------------------------
#     Feed Forward(MLP)
#------------------------------
class FFN(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim*expansion_factor)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(dim*expansion_factor, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x