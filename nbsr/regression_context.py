from dataclasses import dataclass

import torch

@dataclass
class RegressionContext:
    X: torch.Tensor
    W: torch.Tensor | None
    size_factors: torch.Tensor
    mu: torch.Tensor
    pi: torch.Tensor | None
