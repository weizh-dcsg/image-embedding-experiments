#!/usr/bin/env python3
"""Build and optionally import a SigLIP text embedding model with Eland."""

from __future__ import annotations

import sys
import argparse
import json
import os
from pathlib import Path

import torch
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-512"
SAMPLE_TEXT = "running shoes"
EXPECTED_DIMENSIONS = 768
DEFAULT_MAX_LENGTH = 64
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


def _elasticsearch_client(es_url: str) -> Elasticsearch:
    api_key = os.environ.get("ELASTIC_API_KEY")
    username = os.environ.get("ELASTIC_USERNAME", "elastic")
    password = os.environ.get("ELASTIC_PASSWORD")

    if api_key:
        return Elasticsearch(
            es_url,
            api_key=api_key,
            request_timeout=600,
            max_retries=0,
            retry_on_timeout=False,
        )
    if password:
        return Elasticsearch(
            es_url,
            basic_auth=(username, password),
            request_timeout=600,
            max_retries=0,
            retry_on_timeout=False,
        )
    raise RuntimeError(
        "Set ELASTIC_API_KEY or ELASTIC_PASSWORD (with optional ELASTIC_USERNAME) "
        "before importing a model."
    )


class SiglipTextEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = getattr(model, "text_projection", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vector = output.pooler_output
        if self.text_projection is not None:
            vector = self.text_projection(vector)
        return torch.nn.functional.normalize(vector.float(), p=2, dim=-1)


class SiglipTraceableModel:
    def __init__(self, tokenizer: object, model: torch.nn.Module, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.max_length = max_length

    def compatible_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            SAMPLE_TEXT,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        token_type_ids = torch.zeros_like(inputs["input_ids"])
        position_ids = torch.arange(
            self.max_length,
            dtype=inputs["input_ids"].dtype,
        ).unsqueeze(0)
        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            token_type_ids,
            position_ids,
        )

    def trace(self) -> torch.jit.ScriptModule:
        return torch.jit.trace(self.model, self.compatible_inputs())

    def sample_output(self) -> torch.Tensor:
        return self.model(*self.compatible_inputs())


def _eland_bundle(
    model: torch.nn.Module,
    tokenizer: object,
    output_dir: Path,
    max_length: int,
) -> tuple[Path, Path, object]:
    from eland.ml.pytorch.nlp_ml_model import (
        NlpBertTokenizationConfig,
        NlpTrainedModelConfig,
        TextEmbeddingInferenceOptions,
        TrainedModelInput,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    traceable = SiglipTraceableModel(tokenizer, model, max_length)
    traced_path = output_dir / "traced_pytorch_model.pt"
    torch.jit.save(torch.jit.freeze(traceable.trace()), str(traced_path))

    special_tokens = {
        tokenizer.pad_token: "[SEP]",
        tokenizer.unk_token: "[UNK]",
        "<pad>": "[PAD]",
        "<0x00>": "[CLS]",
    }
    vocab = [
        special_tokens.get(token, token)
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    ]
    vocab_config: dict[str, object] = {"vocabulary": vocab}
    sp_model = getattr(tokenizer, "sp_model", None)
    if sp_model is not None:
        vocab_config["scores"] = [sp_model.get_score(index) for index in range(len(vocab))]
    vocab_path = output_dir / "vocabulary.json"
    vocab_path.write_text(json.dumps(vocab_config), encoding="utf-8")

    tokenization = NlpBertTokenizationConfig(
        do_lower_case=False,
        max_sequence_length=max_length,
    )
    config = NlpTrainedModelConfig(
        description=f"{MODEL_ID} SigLIP text encoder",
        inference_config=TextEmbeddingInferenceOptions(
            tokenization=tokenization,
            embedding_size=EXPECTED_DIMENSIONS,
        ),
        input=TrainedModelInput(field_names=["text_field"]),
        model_type="pytorch",
        prefix_strings=None,
    )
    return traced_path, vocab_path, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("siglip_eland_bundle"),
    )
    parser.add_argument("--es-url")
    parser.add_argument("--es-model-id", default="siglip-base-patch16-512-text")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    inputs = processor(
        text=[SAMPLE_TEXT],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    with torch.no_grad():
        reference = model.get_text_features(**inputs)
        if not isinstance(reference, torch.Tensor):
            reference = reference.pooler_output
        reference = torch.nn.functional.normalize(reference.float(), p=2, dim=-1)
        wrapper = SiglipTextEncoder(model).eval()
        exported = wrapper(inputs["input_ids"], inputs["attention_mask"])

    max_error = (reference - exported).abs().max().item()
    print(f"text feature shape: {tuple(exported.shape)}")
    print(f"L2 norm: {exported.norm(dim=-1).item():.7f}")
    print(f"wrapper max absolute error: {max_error:.9g}")
    if exported.shape != (1, EXPECTED_DIMENSIONS):
        raise RuntimeError(
            f"Unexpected SigLIP text feature shape: {tuple(exported.shape)}; "
            f"expected (1, {EXPECTED_DIMENSIONS})"
        )
    if max_error > 1e-5:
        raise RuntimeError(f"Wrapper does not match Transformers output: max error {max_error}")

    traced_path, vocab_path, config = _eland_bundle(
        wrapper,
        processor.tokenizer,
        args.output_dir,
        args.max_length,
    )
    print(f"Saved Eland bundle: {args.output_dir}")

    if args.es_url:
        from eland.ml.pytorch import PyTorchModel

        deployed_model = PyTorchModel(_elasticsearch_client(args.es_url), args.es_model_id)
        deployed_model.import_model(
            model_path=str(traced_path),
            config_path=None,
            config=config,
            vocab_path=str(vocab_path),
            chunk_size=args.chunk_size,
        )
        print(f"Imported model {args.es_model_id} into {args.es_url}")
        if args.start:
            deployed_model.start()
            print(f"Started deployment for {args.es_model_id}")


if __name__ == "__main__":
    main()
