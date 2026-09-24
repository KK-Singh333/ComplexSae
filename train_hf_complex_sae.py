"""Train ComplexSAE on a selected layer of a Hugging Face model.

Example:
    python train_hf_complex_sae.py \
        --model gpt2 \
        --layer transformer.h.5.mlp \
        --dataset Salesforce/wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --d-sae 2048 \
        --steps 1000 \
        --output checkpoints/gpt2-layer5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from complex_sae import ComplexSAE, ComplexSAEConfig, HuggingFaceSAETrainer
from complex_sae.huggingface import ActivationCapture, move_to_device


DATASET_ALIASES = {
    "wikitext": "Salesforce/wikitext",
}


def canonical_dataset_id(dataset_id: str) -> str:
    """Return a Hub-compatible dataset ID for common legacy shorthands."""
    return DATASET_ALIASES.get(dataset_id, dataset_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model identifier or local path")
    parser.add_argument("--layer", required=True, help="PyTorch module path, for example model.layers.5.mlp")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset identifier")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-sae", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sparsity-coefficient", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sae-dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def build_dataloader(dataset, tokenizer, text_column: str, max_seq_length: int, batch_size: int):
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset has no {text_column!r} column: {dataset.column_names}")

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False)


def infer_layer_width(model, layer_name: str, dataloader, device: str) -> int:
    batch = move_to_device(next(iter(dataloader)), device)
    with ActivationCapture(model, layer_name) as capture:
        model.eval()
        with torch.no_grad():
            model(**batch)
        if capture.activation is None:
            raise RuntimeError(f"No activation captured from layer {layer_name!r}")
        if capture.activation.ndim < 2:
            raise ValueError("The selected layer must return [batch, ..., hidden_size] activations")
        return capture.activation.shape[-1]


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "This command requires the optional Hugging Face dependencies. "
            "Install them with: pip install -r requirements-huggingface.txt"
        ) from error

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model).to(device)

    dataset_id = canonical_dataset_id(args.dataset)
    train_dataset = load_dataset(dataset_id, args.dataset_config, split=args.train_split)
    eval_split = args.eval_split or args.train_split
    eval_dataset = load_dataset(dataset_id, args.dataset_config, split=eval_split)
    train_loader = build_dataloader(
        train_dataset, tokenizer, args.text_column, args.max_seq_length, args.batch_size
    )
    eval_loader = build_dataloader(
        eval_dataset, tokenizer, args.text_column, args.max_seq_length, args.batch_size
    )

    d_in = infer_layer_width(model, args.layer, train_loader, str(device))
    sae_dtype = getattr(torch, args.sae_dtype)
    cfg = ComplexSAEConfig(
        d_in=d_in,
        d_sae=args.d_sae,
        device=str(device),
        dtype=args.sae_dtype,
        sparsity_coefficient=args.sparsity_coefficient,
    )
    sae = ComplexSAE(cfg).to(device=device, dtype=sae_dtype)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.learning_rate)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    with HuggingFaceSAETrainer(model, sae, args.layer, optimizer) as trainer:
        trainer.train(train_loader, steps=args.steps)
        metrics = trainer.evaluate(eval_loader, batches=args.eval_batches)

    sae.save_model(output_path)
    (output_path / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=2) + "\n")
    print(json.dumps(metrics.as_dict(), indent=2))


if __name__ == "__main__":
    main()