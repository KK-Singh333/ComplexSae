import torch
from torch import nn

from sae_lens.saes.sae import TrainStepInput

from complex_sae import ComplexSAE, ComplexSAEConfig, HuggingFaceSAETrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.block(inputs)


def test_complex_sae_shapes_and_round_trip():
    d_in = 12
    d_sae = 8
    cfg = ComplexSAEConfig(d_in=d_in, d_sae=d_sae, device="cpu", dtype="float32")
    sae = ComplexSAE(cfg)

    x = torch.randn(4, 16, d_in)
    feature_acts = sae.encode(x)
    assert feature_acts.shape == (4, 16, 2 * d_sae)

    feature_acts_2, hidden_pre = sae.encode_with_hidden_pre(x)
    assert feature_acts_2.shape == feature_acts.shape
    assert hidden_pre.shape == (4, 16, 2 * d_sae)

    x_hat = sae.decode(feature_acts)
    assert x_hat.shape == x.shape

    output = sae(x)
    assert output.shape == x.shape

    loss = ((output - x) ** 2).mean()
    loss.backward()

    for name, param in sae.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"

    magnitude = sae.get_magnitude(feature_acts)
    assert magnitude.shape == (4, 16, d_sae)
    assert (magnitude >= 0).all()

    phase = sae.get_phase(feature_acts)
    assert phase.shape == (4, 16, d_sae)

    real, imag = sae.get_complex_latents(feature_acts)
    assert real.shape == (4, 16, d_sae)
    assert imag.shape == (4, 16, d_sae)
    z = real + 1j * imag
    recon = z.real * 0.0 + 0.0
    assert recon.shape == real.shape


def test_training_step_and_registration():
    cfg = ComplexSAEConfig(d_in=9, d_sae=6, device="cpu", dtype="float32")
    sae = ComplexSAE(cfg)

    step_input = TrainStepInput(
        sae_in=torch.randn(3, 9),
        coefficients={"sparsity": 0.1},
        dead_neuron_mask=None,
        n_training_steps=1,
        is_logging_step=True,
    )
    step_output = sae.training_forward_pass(step_input)
    assert step_output.sae_out.shape == (3, 9)
    assert step_output.feature_acts.shape == (3, 2 * 6)
    assert step_output.loss.ndim == 0

    assert sae.get_sae_class_for_architecture("complex_sae") is ComplexSAE


def test_save_and_load_round_trip(tmp_path):
    cfg = ComplexSAEConfig(d_in=10, d_sae=7, device="cpu", dtype="float32")
    sae = ComplexSAE(cfg)
    x = torch.randn(2, 3, 10)

    before = sae(x)
    sae.save_model(tmp_path / "complex_sae_test")
    loaded = ComplexSAE.load_from_disk(tmp_path / "complex_sae_test")
    after = loaded(x)

    assert torch.allclose(before, after, atol=1e-5, rtol=1e-4)


def test_huggingface_trainer_hooks_selected_layer():
    model = TinyModel()
    cfg = ComplexSAEConfig(d_in=4, d_sae=3, device="cpu", dtype="float32")
    sae = ComplexSAE(cfg)
    batches = [torch.randn(2, 5, 4) for _ in range(2)]

    with HuggingFaceSAETrainer(model, sae, "block") as trainer:
        losses = trainer.train(batches, steps=2, log_every=0)
        metrics = trainer.evaluate(batches, batches=2)

    assert len(losses) == 2
    assert metrics.tokens == 20
    assert metrics.reconstruction_mse >= 0
    assert all(parameter.grad is None for parameter in model.parameters())
