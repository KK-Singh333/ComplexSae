# ComplexSAE

This repository provides a complex-feature sparse autoencoder compatible with
SAE-Lens. It also includes a Hugging Face training workflow that captures one
selected model layer, trains the SAE on those activations, and evaluates the
reconstruction.

## Install

Install the project dependencies used by the existing tests, then install the
optional Hugging Face dependencies:

```bash
pip install torch sae-lens
pip install -r requirements-huggingface.txt
```

## Train on a Hugging Face model

The CLI accepts any model supported by `transformers` and any valid PyTorch
module path inside that model. Common layer paths are:

```text
GPT-2:   transformer.h.5.mlp
Llama:   model.layers.5.mlp
Mistral: model.layers.5.mlp
Qwen:    model.layers.5.mlp
```

Example using GPT-2 and Wikitext:

```bash
python train_hf_complex_sae.py \
  --model gpt2 \
  --layer transformer.h.5.mlp \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --d-sae 2048 \
  --steps 1000 \
  --batch-size 4 \
  --max-seq-length 128 \
  --output checkpoints/gpt2-layer5-mlp
```

The script automatically infers `d_in` from the selected layer. The model is
frozen; only the `ComplexSAE` parameters are optimized. The output directory
contains the SAE-Lens checkpoint and `metrics.json` with reconstruction MSE,
explained variance, average active-feature fraction, and average feature
magnitude.

`d_sae` is the number of logical complex features. The stored activation has
size `2 * d_sae` because real and imaginary components are concatenated.

## Python API

The reusable trainer can be used with an already-loaded Hugging Face model and
any PyTorch dataloader whose batches can be passed to the model:

```python
import torch
from transformers import AutoModel

from complex_sae import ComplexSAE, ComplexSAEConfig, HuggingFaceSAETrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("gpt2").to(device)
sae = ComplexSAE(
	ComplexSAEConfig(
		d_in=768,  # inferred from the selected layer in the CLI
		d_sae=2048,
		device=device,
		dtype="float32",
		sparsity_coefficient=0.1,
	)
).to(device)

with HuggingFaceSAETrainer(model, sae, "transformer.h.5.mlp") as trainer:
	trainer.train(train_dataloader, steps=1000)
	metrics = trainer.evaluate(eval_dataloader, batches=25)

sae.save_model("checkpoints/my-complex-sae")
print(metrics.as_dict())
```

The hook observes the selected layer and does not change the model output.
After training, use `sae.encode(activation)` for complex features,
`sae.get_magnitude(features)` for feature strengths,
`sae.get_phase(features)` for phases, and `sae.decode(features)` for the
reconstructed activation.