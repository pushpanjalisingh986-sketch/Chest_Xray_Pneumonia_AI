import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def enable_mc_dropout(model: nn.Module) -> None:
    """Forces Dropout layers to remain active during evaluation for Monte Carlo sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def predict_with_uncertainty(model: nn.Module, input_tensor: torch.Tensor, num_samples: int = 15) -> Tuple[float, float]:
    """
    Runs N stochastic forward passes using MC Dropout to compute mean probability (μ) 
    and epistemic uncertainty standard deviation (σ).
    """
    model.eval()
    enable_mc_dropout(model)

    probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)

    probs = np.array(probs)
    mean_prob = float(np.mean(probs))
    std_dev = float(np.std(probs))

    return mean_prob, std_dev