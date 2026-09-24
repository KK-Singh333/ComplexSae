import sae_lens as SAE
from sae_lens.saes.sae import TrainingSAEConfig, TrainingSAE
from typing_extensions import override
from ComplexAutoEncoder.codebase.model.ComplexAutoEncoder import ComplexAutoEncoder
opt={

    'model':{
        'hidden_dim': 64,
        'latent_dim': 128,
    }
}

autoencoder = ComplexAutoEncoder(opt)
class ComplexSAEConfig(TrainingSAEConfig):
    @override
    @classmethod
    def architecture(cls) -> str:
        return "complex_sae"

class ComplexSAE(TrainingSAE):
    @override
    def 

