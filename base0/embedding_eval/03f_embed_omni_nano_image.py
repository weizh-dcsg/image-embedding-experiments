#!/usr/bin/env python3
"""Step 3f: encode the Jina v5 omni-nano IMAGE tower over the FULL W6 test set.

W7/W8 only ran the omni image towers on a seeded 600-query/28,201-product subset (image encoding
is 3-5x slower than SigLIP per image). This encodes all 80,608 products so omni-nano can be used
as a second *target modality* in the W6 fusion sweep (09_fusion_experiment.py), fused with every
text representation at the same scale as SigLIP.

Query-side vectors are reused verbatim from embeddings/jina_omni_nano_text.npz (03e_embed_omni_text.py):
same model, same task="retrieval", same "Query: " prefix, so the text tower output is identical
regardless of whether the model was loaded with modality="text" or modality="vision" -- only the
(expensive) image tower needs a fresh forward pass here.

Resumable: image progress is checkpointed, so a re-run continues where it stopped.

Output: embeddings/jina_omni_nano_image.npz (queries, ecodes, query_emb, image_emb), aligned with
siglip.npz so 09_fusion_experiment.py can treat it as a second target modality.
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

REPO = "jinaai/jina-embeddings-v5-omni-nano"


def load_progress(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    d = np.load(path, allow_pickle=True)
    return {str(k): v for k, v in zip(d["ecodes"], d["image_emb"])} if len(d["ecodes"]) else {}


def save_progress(path: Path, queries: list[str], query_emb: np.ndarray, e_done: dict[str, np.ndarray]) -> None:
    """Write to a temp file then atomically replace, so a concurrent reader (e.g.
    09_fusion_experiment.py) never observes a half-written checkpoint mid-save."""
    dim_e = len(next(iter(e_done.values()))) if e_done else 0
    tmp_path = path.with_name(path.name + ".tmp")
    # pass an open file object rather than a path string -- np.savez_compressed appends ".npz"
    # to string/path filenames that lack it, which would otherwise mangle the ".tmp" suffix.
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            queries=np.array(queries, dtype=object),
            query_emb=query_emb,
            ecodes=np.array(list(e_done), dtype=object),
            image_emb=np.array(list(e_done.values()), dtype=np.float32).reshape(-1, dim_e),
        )
    tmp_path.replace(path)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    args = ap.parse_args()

    config.ensure_dirs()
    siglip = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    queries = [str(q) for q in siglip["queries"]]
    ecodes = [str(e) for e in siglip["ecodes"]]

    text_path = config.EMB_DIR / "jina_omni_nano_text.npz"
    if not text_path.exists():
        print(f"missing {text_path} -- run 03e_embed_omni_text.py first")
        return 1
    text_d = np.load(text_path, allow_pickle=True)
    text_q_idx = {str(q): i for i, q in enumerate(text_d["queries"])}
    missing_q = [q for q in queries if q not in text_q_idx]
    if missing_q:
        print(f"{text_path} is missing {len(missing_q)} queries; rerun 03e_embed_omni_text.py")
        return 1
    # NpzFile.__getitem__ decompresses from the zip archive on every call -- pull the array into
    # memory once rather than re-decompressing it inside the per-query list comprehension below.
    text_query_emb = text_d["query_emb"]
    query_emb = np.stack([text_query_emb[text_q_idx[q]] for q in queries])

    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)
    ok_ecodes = set(manifest.loc[manifest["ok"] == True, "ecode"].astype(str))  # noqa: E712
    ecodes = [e for e in ecodes if e in ok_ecodes]

    out_path = config.EMB_DIR / "jina_omni_nano_image.npz"
    e_done = load_progress(out_path)
    todo_e = [e for e in ecodes if e not in e_done]
    print(f"products {len(ecodes)} ({len(todo_e)} to do)")
    if not todo_e:
        save_progress(out_path, queries, query_emb, e_done)
        print(f"complete -> {out_path}")
        return 0

    from transformers import AutoModel, AutoProcessor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {REPO} on {device} (modality=vision)")
    model = AutoModel.from_pretrained(
        REPO, trust_remote_code=True, default_task="retrieval", modality="vision"
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)

    t0 = time.time()
    for i, ecode in enumerate(todo_e, start=1):
        try:
            im = Image.open(config.IMAGE_DIR / f"{ecode}.jpg").convert("RGB")
            inp = proc(images=im, text="Document: <image>", return_tensors="pt")
            with torch.no_grad():
                v = model.embed(**inp.to(device)).float().cpu().numpy()[0]
            e_done[ecode] = l2(v)
        except Exception as exc:
            print(f"\n  skip {ecode}: {type(exc).__name__}: {exc}")
        if i % 50 == 0 or i == len(todo_e):
            rate = i / (time.time() - t0)
            eta = (len(todo_e) - i) / rate / 60
            print(f"  image {i}/{len(todo_e)}  {rate:.2f}/s  eta {eta:.0f}m", end="\r")
        if i % args.checkpoint_every == 0:
            save_progress(out_path, queries, query_emb, e_done)
    print()

    save_progress(out_path, queries, query_emb, e_done)
    print(f"saved -> {out_path}  ({len(e_done)} products)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
