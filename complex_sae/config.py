from dataclasses import dataclass

from sae_lens.saes.sae import TrainingSAEConfig


@dataclass
class ComplexSAEConfig(TrainingSAEConfig):
    sparsity_coefficient: float = 1.0
    use_phase_bias: bool = True
    use_magnitude_pathway: bool = True
    use_phase_gating: bool = False
    use_batch_norm: bool = False
    use_complex_activation: bool = True
    eps: float = 1e-8
    activation_threshold: float = 0.0

    @classmethod
    def architecture(cls) -> str:
        return "complex_sae"
