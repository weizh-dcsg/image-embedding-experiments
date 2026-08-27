#!/usr/bin/env python3
"""Export and validate the SigLIP text tower locally; do not upload to Elasticsearch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-512"
SAMPLE_TEXT = "running shoes"
EXPECTED_DIMENSIONS = 768


class SiglipTextEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        vector = self.text_projection(output.pooler_output)
        return torch.nn.functional.normalize(vector.float(), p=2, dim=-1)


def main() -> None:
    print(f"Loading {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    inputs = processor(
        text=[SAMPLE_TEXT],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

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

    output_path = Path(__file__).with_name("siglip_text_encoder.pt")
    traced = torch.jit.trace(wrapper, (inputs["input_ids"], inputs["attention_mask"]))
    traced.save(str(output_path))
    print(f"Saved TorchScript wrapper: {output_path}")
    print("Local export is valid; Eland upload remains a separate, admin-approved step.")


if __name__ == "__main__":
    main()
