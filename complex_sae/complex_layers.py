import torch
from torch import nn


class ComplexLinear(nn.Module):
    """Real-valued implementation of a complex linear transform."""

    def __init__(self, in_features: int, out_features: int, *, imag_init_scale: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_real = nn.Parameter(torch.empty(in_features, out_features))
        self.W_imag = nn.Parameter(torch.empty(in_features, out_features))
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))

        nn.init.kaiming_uniform_(self.W_real, a=5**0.5)
        nn.init.uniform_(self.W_imag, -imag_init_scale, imag_init_scale)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_real = x_real @ self.W_real - x_imag @ self.W_imag + self.bias_real
        y_imag = x_real @ self.W_imag + x_imag @ self.W_real + self.bias_imag
        return y_real, y_imag


def split_complex(feature_acts: torch.Tensor, d_sae: int) -> tuple[torch.Tensor, torch.Tensor]:
    real = feature_acts[..., :d_sae]
    imag = feature_acts[..., d_sae:]
    return real, imag


def magnitude_and_phase(real: torch.Tensor, imag: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.sqrt(real.square() + imag.square() + eps)
    phase = torch.atan2(imag, real)
    return magnitude, phase


def real_imag_from_magnitude_phase(magnitude: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return magnitude * torch.cos(phase), magnitude * torch.sin(phase)
