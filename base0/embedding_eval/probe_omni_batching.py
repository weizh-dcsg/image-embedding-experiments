#!/usr/bin/env python3
"""Does the omni processor accept a batch of images in one forward pass?

The model card's sentence-transformers path forwards media one sample at a time. If the raw
processor batches, throughput improves several-fold with no change to the model or its inputs.
Also verifies batched vectors match per-sample vectors, so batching cannot silently alter results.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REPO = "jinaai/jina-embeddings-v5-omni-nano"


def main() -> int:
    from transformers import AutoModel, AutoProcessor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        REPO, trust_remote_code=True, default_task="retrieval", modality="vision"
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)

    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)
    ok = manifest.loc[manifest["ok"] == True, "ecode"].astype(str).tolist()[:16]  # noqa: E712
    imgs = [Image.open(config.IMAGE_DIR / f"{e}.jpg").convert("RGB") for e in ok]

    # per-sample reference
    t0 = time.time()
    single = []
    for im in imgs[:8]:
        inp = proc(images=im, text="Document: <image>", return_tensors="pt")
        with torch.no_grad():
            single.append(model.embed(**inp.to(device)).float().cpu().numpy()[0])
    per_rate = 8 / (time.time() - t0)
    single = np.vstack(single)
    print(f"per-sample: {per_rate:.2f} items/s")

    for bs in (4, 8, 16):
        batch = imgs[:bs]
        try:
            t0 = time.time()
            inp = proc(
                images=batch,
                text=["Document: <image>"] * bs,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = model.embed(**inp.to(device)).float().cpu().numpy()
            rate = bs / (time.time() - t0)
            ok_shape = out.shape == (bs, single.shape[1])
            drift = float(np.abs(out[:8] - single[: min(8, bs)]).max()) if ok_shape and bs >= 8 else float("nan")
            print(f"batch={bs:>2}: {rate:.2f} items/s  shape={out.shape}  speedup={rate / per_rate:.2f}x  max|Δ| vs per-sample={drift:.2e}")
        except Exception as exc:
            print(f"batch={bs:>2}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
