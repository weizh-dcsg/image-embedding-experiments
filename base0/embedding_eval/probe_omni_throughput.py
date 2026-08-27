#!/usr/bin/env python3
"""Throughput probe for the Jina v5 omni image tower before committing to a full catalogue run.

Loads the model with modality="vision" (skips the audio tower), encodes a handful of real
catalogue images and queries, and reports embedding dim + items/s so the full-run cost can be
estimated. Not part of the pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jinaai/jina-embeddings-v5-omni-nano")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {args.repo} (modality=vision) on {device}")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.repo, trust_remote_code=True, default_task="retrieval", modality="vision"
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(args.repo, trust_remote_code=True)
    print(f"load: {time.time() - t0:.1f}s")

    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)
    ok = manifest.loc[manifest["ok"] == True, "ecode"].astype(str).tolist()[: args.n]  # noqa: E712
    paths = [config.IMAGE_DIR / f"{e}.jpg" for e in ok]

    # text side, batched
    t0 = time.time()
    inputs = proc(
        text=["Query: nike soccer cleats", "Query: yeti tumbler"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        tv = model.embed(**inputs.to(device))
    print(f"text  dim={tv.shape[-1]}  {2 / (time.time() - t0):.1f} items/s (batch of 2)")

    # image side, one forward per image
    t0 = time.time()
    dims = None
    for p in paths:
        im = Image.open(p).convert("RGB")
        inputs = proc(images=im, text="Document: <image>", return_tensors="pt")
        with torch.no_grad():
            v = model.embed(**inputs.to(device))
        dims = v.shape[-1]
    dt = time.time() - t0
    rate = len(paths) / dt
    print(f"image dim={dims}  {rate:.2f} items/s  ({dt / len(paths) * 1000:.0f} ms/image)")
    print(f"projected for 80,608 products: {80608 / rate / 3600:.1f} GPU-hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
