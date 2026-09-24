import torch


def complex_sparsity_loss(feature_acts: torch.Tensor, coeff: float, d_sae: int, eps: float = 1e-8) -> torch.Tensor:
    real, imag = feature_acts[..., :d_sae], feature_acts[..., d_sae:]
    magnitude = torch.sqrt(real.square() + imag.square() + eps)
    return coeff * magnitude.sum(dim=-1).mean()
