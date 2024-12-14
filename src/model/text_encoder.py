from transformers import FlaxAutoModel, AutoConfig
import torch
from torch import nn

class TextEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        config = AutoConfig.from_pretrained(model_name)
        self.model = FlaxAutoModel.from_config(config)
        self.target_token_idx = 0

    def __call__(self, input_ids: int, attention_mask: int) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # 出力を取得する
        last_hidden_state = output.last_hidden_state 
        return last_hidden_state[:, self.target_token_idx, :]
