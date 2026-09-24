"""Utilities for training a ComplexSAE on Hugging Face model activations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from sae_lens.saes.sae import TrainStepInput

from .complex_sae import ComplexSAE


def move_to_device(value: Any, device: torch.device | str) -> Any:
    """Move tensors in a nested batch to a device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def first_tensor(value: Any) -> torch.Tensor | None:
    """Find the first tensor in a layer output, including tuple-like outputs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class ActivationCapture:
    """Forward-hook based activation capture for an arbitrary module."""

    def __init__(self, model: nn.Module, layer_name: str):
        self.layer_name = layer_name
        self.activation: torch.Tensor | None = None
        self._handle = model.get_submodule(layer_name).register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        activation = first_tensor(output)
        if activation is None:
            raise TypeError(f"Layer {self.layer_name!r} did not return a tensor")
        self.activation = activation.detach()

    def clear(self) -> None:
        self.activation = None

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "ActivationCapture":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@dataclass
class EvaluationMetrics:
    reconstruction_mse: float
    explained_variance: float
    mean_l0: float
    mean_magnitude: float
    tokens: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "reconstruction_mse": self.reconstruction_mse,
            "explained_variance": self.explained_variance,
            "mean_l0": self.mean_l0,
            "mean_magnitude": self.mean_magnitude,
            "tokens": self.tokens,
        }


class HuggingFaceSAETrainer:
    """Train and evaluate a ComplexSAE on one Hugging Face model layer."""

    def __init__(
        self,
        model: nn.Module,
        sae: ComplexSAE,
        layer_name: str,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.model = model
        self.sae = sae
        self.layer_name = layer_name
        self.device = next(sae.parameters()).device
        self.optimizer = optimizer or torch.optim.AdamW(sae.parameters(), lr=3e-4)
        self.capture = ActivationCapture(model, layer_name)

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> "HuggingFaceSAETrainer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _run_model(self, batch: Any) -> torch.Tensor:
        self.capture.clear()
        batch = move_to_device(batch, self.device)
        with torch.no_grad():
            if isinstance(batch, Mapping):
                self.model(**batch)
            else:
                self.model(batch)

        if self.capture.activation is None:
            raise RuntimeError(f"No activation captured from layer {self.layer_name!r}")
        activation = self.capture.activation
        if activation.ndim < 2:
            raise ValueError("The selected layer must return [batch, ..., hidden_size] activations")
        return activation.to(device=self.device, dtype=self.sae.dtype)

    def activation_from_batch(self, batch: Any) -> torch.Tensor:
        """Run one model batch and return the selected layer activation."""
        return self._run_model(batch)

    def train(self, dataloader: Any, steps: int, log_every: int = 100) -> list[float]:
        """Train for an exact number of batches and return recorded losses."""
        if steps < 1:
            raise ValueError("steps must be positive")

        self.sae.train()
        losses: list[float] = []
        batches: Iterator[Any] = iter(dataloader)

        for step in range(steps):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(dataloader)
                batch = next(batches)

            activations = self._run_model(batch)
            step_input = TrainStepInput(
                sae_in=activations,
                coefficients={"sparsity": self.sae.cfg.sparsity_coefficient},
                dead_neuron_mask=None,
                n_training_steps=step,
                is_logging_step=(step % log_every == 0),
            )
            step_output = self.sae.training_forward_pass(step_input)

            self.optimizer.zero_grad(set_to_none=True)
            step_output.loss.backward()
            self.optimizer.step()

            loss = float(step_output.loss.detach().cpu())
            losses.append(loss)
            if log_every > 0 and step % log_every == 0:
                print(f"step={step} loss={loss:.6f}")

        return losses

    @torch.no_grad()
    def evaluate(self, dataloader: Any, batches: int = 10) -> EvaluationMetrics:
        """Measure reconstruction quality and complex feature activity."""
        if batches < 1:
            raise ValueError("batches must be positive")

        self.sae.eval()
        total_squared_error = 0.0
        total_variance = 0.0
        total_l0 = 0.0
        total_magnitude = 0.0
        total_tokens = 0

        for batch_index, batch in enumerate(dataloader):
            if batch_index >= batches:
                break
            activations = self._run_model(batch)
            feature_acts = self.sae.encode(activations)
            reconstruction = self.sae.decode(feature_acts)
            error = (reconstruction - activations).square()
            magnitude = self.sae.get_magnitude(feature_acts)
            token_count = activations.reshape(-1, activations.shape[-1]).shape[0]

            total_squared_error += float(error.sum().cpu())
            total_variance += float((activations - activations.mean(dim=-1, keepdim=True)).square().sum().cpu())
            total_l0 += float((magnitude > 0).sum().cpu())
            total_magnitude += float(magnitude.sum().cpu())
            total_tokens += token_count

        if total_tokens == 0:
            raise ValueError("The evaluation dataloader produced no batches")

        feature_count = total_tokens * self.sae.cfg.d_sae
        return EvaluationMetrics(
            reconstruction_mse=total_squared_error / (total_tokens * self.sae.cfg.d_in),
            explained_variance=1.0 - total_squared_error / max(total_variance, 1e-12),
            mean_l0=total_l0 / feature_count,
            mean_magnitude=total_magnitude / feature_count,
            tokens=total_tokens,
        )