#!/usr/bin/env python3
"""Build and optionally import the Jina CLIP v2 text encoder with Eland."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoProcessor

JINA_CLIP_MODEL_ID = "jinaai/jina-clip-v2"
SAMPLE_TEXT = "running shoes"
EXPECTED_DIMENSIONS = 1024
DEFAULT_MAX_LENGTH = 77
DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


def _elasticsearch_client(es_url: str) -> Elasticsearch:
    api_key = os.environ.get("ELASTIC_API_KEY")

    if api_key:
        return Elasticsearch(
            es_url,
            api_key=api_key,
            request_timeout=600,
            max_retries=0,
            retry_on_timeout=False,
        )
    raise RuntimeError(
        "Set ELASTIC_API_KEY or ELASTIC_PASSWORD (with optional ELASTIC_USERNAME) "
        "before importing a model."
    )


class JinaClipTextEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection
        self.pad_token_id = int(model.text_model.config.pad_token_id)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        # Reimplement HFTextEncoder.forward without its fixed-length token bookkeeping so
        # the traced graph stays valid for any sequence length Elasticsearch sends.
        encoder = self.text_model
        mask = (input_ids != self.pad_token_id).long()
        output = encoder.transformer(input_ids=input_ids, attention_mask=mask)
        vector = self.text_projection(encoder.proj(encoder.pooler(output, mask)))
        return torch.nn.functional.normalize(vector.float(), p=2, dim=-1)


def _merge_jina_lora_weights(model: torch.nn.Module) -> None:
    """Fold the default LoRA adapter into the base weights.

    Jina's adapter path loops over the tasks present in ``adapter_mask``, which traces
    into a graph with the calibration sequence length baked in as a constant. Merging the
    adapter lets the model run with ``adapter_mask=None`` and keeps the graph shape-generic.

    The merged weights are written into the parametrization's ``original`` tensor rather
    than removing the parametrization, because Jina's attention layers branch on the
    presence of ``parametrizations`` to decide whether ``Wqkv`` returns a residual.
    """
    from torch.nn.utils import parametrize

    text_encoder = model.text_model
    task_id = text_encoder.default_loraid
    if task_id is None:
        raise RuntimeError("Jina CLIP text encoder has no default LoRA task to merge")

    merged_layers = 0
    for module in text_encoder.transformer.modules():
        if not parametrize.is_parametrized(module, "weight"):
            continue
        parametrization = module.parametrizations.weight[0]
        if not hasattr(parametrization, "lora_forward"):
            continue
        with torch.no_grad():
            merged = parametrization.lora_forward(module.weight, current_task=task_id).clone()
            module.parametrizations.weight.original.copy_(merged)
        merged_layers += 1

    text_encoder._default_loraid = None
    print(f"merged LoRA task {task_id} into {merged_layers} layers")


def _torch_rotary_qkv(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_k: torch.Tensor | None = None,
    sin_k: torch.Tensor | None = None,
    interleaved: bool = False,
    **kwargs: object,
) -> torch.Tensor:
    del kwargs
    if interleaved:
        raise RuntimeError("Jina CLIP export expects non-interleaved rotary embeddings")

    rotary_dim = cos.shape[-1] * 2

    def rotate(x: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Non-interleaved layout: the rotated half-pairs are (i, i + rotary_dim / 2), so
        # cos/sin must be duplicated by concatenation rather than interleaving.
        c = torch.cat((c[: x.shape[1]], c[: x.shape[1]]), dim=-1).unsqueeze(1)
        s = torch.cat((s[: x.shape[1]], s[: x.shape[1]]), dim=-1).unsqueeze(1)
        first, second = x[..., :rotary_dim].chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return torch.cat((x[..., :rotary_dim] * c + rotated * s, x[..., rotary_dim:]), dim=-1)

    cos_k = cos if cos_k is None else cos_k
    sin_k = sin if sin_k is None else sin_k
    return torch.stack((rotate(qkv[:, :, 0], cos, sin), rotate(qkv[:, :, 1], cos_k, sin_k), qkv[:, :, 2]), dim=2)


def _patch_jina_rotary_for_export() -> None:
    import importlib

    rotary = importlib.import_module(
        "transformers_modules.jinaai.xlm_hyphen_roberta_hyphen_flash_hyphen_implementation.bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3.rotary"
    )
    rotary.apply_rotary_emb_qkv_ = _torch_rotary_qkv


def _verify_jina_trace(
    wrapper: torch.nn.Module,
    traced_path: Path,
    tokenizer: object,
    max_length: int,
) -> None:
    """Check the reloaded trace against eager output at several sequence lengths.

    Elasticsearch pads each batch to its own longest sequence, so the graph must not
    depend on the calibration length.
    """
    reloaded = torch.jit.load(str(traced_path)).eval()
    probes = [SAMPLE_TEXT, "shoes", "waterproof insulated hiking boots for cold weather"]
    for probe in probes:
        batch = tokenizer(probe, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            expected = wrapper(batch["input_ids"], batch["attention_mask"])
            actual = reloaded(batch["input_ids"], batch["attention_mask"])
        error = (expected - actual).abs().max().item()
        seq_len = batch["input_ids"].shape[1]
        print(f"trace check seq_len={seq_len}: max absolute error {error:.3g}")
        if error > 1e-5:
            raise RuntimeError(
                f"Traced model diverges at sequence length {seq_len}: max error {error}"
            )


def _jina_clip_bundle(
    model: torch.nn.Module,
    tokenizer: object,
    output_dir: Path,
    max_length: int,
) -> tuple[Path, Path, object]:
    from eland.ml.pytorch.nlp_ml_model import (
        NlpTrainedModelConfig,
        NlpXLMRobertaTokenizationConfig,
        TextEmbeddingInferenceOptions,
        TrainedModelInput,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    traceable = JinaClipTextEncoder(model).eval()
    inputs = tokenizer(
        SAMPLE_TEXT,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    traced_path = output_dir / "traced_pytorch_model.pt"
    traced = torch.jit.freeze(torch.jit.trace(traceable, (inputs["input_ids"], inputs["attention_mask"])))
    torch.jit.save(traced, str(traced_path))
    _verify_jina_trace(traceable, traced_path, tokenizer, max_length)

    tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
    model_vocab = tokenizer_json["model"]["vocab"]
    vocab = [entry[0] for entry in model_vocab]
    scores = [float(entry[1]) for entry in model_vocab]
    vocab_path = output_dir / "vocabulary.json"
    vocab_path.write_text(json.dumps({"vocabulary": vocab, "scores": scores}), encoding="utf-8")

    tokenization = NlpXLMRobertaTokenizationConfig(max_sequence_length=max_length)
    config = NlpTrainedModelConfig(
        description=f"{JINA_CLIP_MODEL_ID} text encoder",
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
        default=Path(__file__).with_name("jina_clip_v2_eland_bundle"),
    )
    parser.add_argument("--hf-model-id", default=JINA_CLIP_MODEL_ID)
    parser.add_argument("--es-url")
    parser.add_argument("--es-model-id", default="jina-clip-v2-text")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.hf_model_id}")
    model = AutoModel.from_pretrained(args.hf_model_id, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(args.hf_model_id, trust_remote_code=True)
    inputs = processor(
        text=[SAMPLE_TEXT],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        # Capture the reference with the stock rotary kernel and active LoRA adapter, so
        # the comparison below validates both export transformations end to end.
        reference = model.get_text_features(**inputs)
        reference = torch.nn.functional.normalize(reference.float(), p=2, dim=-1)
        _patch_jina_rotary_for_export()
        _merge_jina_lora_weights(model)
        wrapper = JinaClipTextEncoder(model).eval()
        exported = wrapper(inputs["input_ids"], inputs["attention_mask"])

    max_error = (reference - exported).abs().max().item()
    print(f"text feature shape: {tuple(exported.shape)}")
    print(f"L2 norm: {exported.norm(dim=-1).item():.7f}")
    print(f"wrapper max absolute error: {max_error:.9g}")
    if exported.shape != (1, EXPECTED_DIMENSIONS):
        raise RuntimeError(
            f"Unexpected text feature shape: {tuple(exported.shape)}; "
            f"expected (1, {EXPECTED_DIMENSIONS})"
        )
    if max_error > 1e-5:
        raise RuntimeError(f"Wrapper does not match Transformers output: max error {max_error}")

    traced_path, vocab_path, config = _jina_clip_bundle(
        model, processor.tokenizer, args.output_dir, args.max_length
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
