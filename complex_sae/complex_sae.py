from __future__ import annotations

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import TrainStepInput, TrainingSAE, TrainCoefficientConfig

from .complex_layers import ComplexLinear, magnitude_and_phase, real_imag_from_magnitude_phase, split_complex
from .config import ComplexSAEConfig


class ComplexSAE(TrainingSAE[ComplexSAEConfig]):
    """A real-valued, SAELens-compatible complex SAE implementation."""

    b_enc_real: nn.Parameter
    b_enc_imag: nn.Parameter
    W_enc_real: nn.Parameter
    W_enc_imag: nn.Parameter
    W_dec_real: nn.Parameter
    W_dec_imag: nn.Parameter

    def __init__(self, cfg: ComplexSAEConfig, use_error_term: bool = False):
        self.phase_bias_param = None
        self.magnitude_bias_param = None
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()

        self.W_dec_real = nn.Parameter(
            torch.empty(self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device)
        )
        self.W_dec_imag = nn.Parameter(
            torch.empty(self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device)
        )
        self.W_enc_real = nn.Parameter(
            torch.empty(self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.W_enc_imag = nn.Parameter(
            torch.empty(self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        nn.init.kaiming_uniform_(self.W_dec_real, a=5**0.5)
        nn.init.uniform_(self.W_dec_imag, -0.05, 0.05)
        self.W_enc_real.data.copy_(self.W_dec_real.data.T.clone())
        self.W_enc_imag.data.copy_(self.W_dec_imag.data.T.clone())

        self.W_dec = nn.Parameter(
            torch.cat([self.W_dec_real.data.clone(), self.W_dec_imag.data.clone()], dim=0)
        )
        self.W_enc = nn.Parameter(
            torch.cat([self.W_enc_real.data.clone(), self.W_enc_imag.data.clone()], dim=-1)
        )

        self.b_enc_real = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_enc_imag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        if self.cfg.use_phase_bias:
            self.phase_bias_param = nn.Parameter(
                torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )
        else:
            self.phase_bias_param = None

        if self.cfg.use_magnitude_pathway:
            self.magnitude_bias_param = nn.Parameter(
                torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )
        else:
            self.magnitude_bias_param = None

        if self.cfg.decoder_init_norm is not None:
            with torch.no_grad():
                self.W_dec_real.data /= self.W_dec_real.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                self.W_dec_real.data *= self.cfg.decoder_init_norm
                self.W_enc_real.data = self.W_dec_real.data.T.clone().detach().contiguous()

    @override
    def get_coefficients(self):
        return {
            "sparsity": TrainCoefficientConfig(
                value=self.cfg.sparsity_coefficient,
                warm_up_steps=0,
            )
        }

    def _process_complex_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        compatibility_real = sae_in @ self.W_enc[:, : self.cfg.d_sae]
        compatibility_imag = sae_in @ self.W_enc[:, self.cfg.d_sae :]
        z_pre_real = sae_in @ self.W_enc_real + self.b_enc_real
        z_pre_imag = sae_in @ self.W_enc_imag + self.b_enc_imag
        z_pre_real = 0.5 * z_pre_real + 0.5 * compatibility_real
        z_pre_imag = 0.5 * z_pre_imag + 0.5 * compatibility_imag
        return z_pre_real, z_pre_imag

    def get_complex_latents(self, feature_acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return split_complex(feature_acts, self.cfg.d_sae)

    def get_magnitude(self, feature_acts: torch.Tensor) -> torch.Tensor:
        real, imag = self.get_complex_latents(feature_acts)
        magnitude, _ = magnitude_and_phase(real, imag, eps=self.cfg.eps)
        return magnitude

    def get_phase(self, feature_acts: torch.Tensor) -> torch.Tensor:
        real, imag = self.get_complex_latents(feature_acts)
        _, phase = magnitude_and_phase(real, imag, eps=self.cfg.eps)
        return phase

    def _complex_activation(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude, phase = magnitude_and_phase(z_real, z_imag, eps=self.cfg.eps)

        if self.cfg.use_phase_bias and self.phase_bias_param is not None:
            phase = phase + self.phase_bias_param

        if self.cfg.use_magnitude_pathway and self.magnitude_bias_param is not None:
            magnitude = magnitude + self.magnitude_bias_param

        if self.cfg.use_phase_gating:
            gate = torch.cos(phase)
            magnitude = magnitude * (1.0 + 0.1 * gate)

        if self.cfg.use_complex_activation:
            if self.cfg.use_batch_norm and self.training:
                if not hasattr(self, "magnitude_bn"):
                    self.magnitude_bn = nn.BatchNorm1d(
                        self.cfg.d_sae,
                        affine=True,
                        device=self.device,
                        dtype=self.dtype,
                    )
                magnitude = self.magnitude_bn(magnitude.reshape(-1, self.cfg.d_sae)).reshape_as(magnitude)
            magnitude = torch.relu(magnitude)

        out_real, out_imag = real_imag_from_magnitude_phase(magnitude, phase)
        return out_real, out_imag

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pre_real, z_pre_imag = self._process_complex_input(x)
        hidden_pre = torch.cat([z_pre_real, z_pre_imag], dim=-1)
        post_real, post_imag = self._complex_activation(z_pre_real, z_pre_imag)
        feature_acts = torch.cat([post_real, post_imag], dim=-1)
        return feature_acts, hidden_pre

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        real, imag = self.get_complex_latents(feature_acts)
        compatibility = real @ self.W_dec[: self.cfg.d_sae] - imag @ self.W_dec[self.cfg.d_sae :] + self.b_dec
        sae_out_pre = real @ self.W_dec_real - imag @ self.W_dec_imag + self.b_dec
        sae_out_pre = 0.5 * sae_out_pre + 0.5 * compatibility
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        coefficient = step_input.coefficients.get("sparsity", self.cfg.sparsity_coefficient)
        magnitude = self.get_magnitude(feature_acts)
        sparsity = coefficient * magnitude.sum(dim=-1).mean()
        return {"sparsity_loss": sparsity}

    def complex_linear_forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x_real @ self.W_dec_real.T, x_real @ self.W_dec_imag.T


__all__ = ["ComplexSAE", "ComplexSAEConfig"]
