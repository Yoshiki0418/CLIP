import random
import numpy as np
import torch

def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)