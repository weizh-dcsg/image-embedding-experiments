#!/usr/bin/env python3
"""Step 19 (W9): encode the FULL test set with the Jina CLIP v2 text and image towers.

W7/W8 only ever needed the 600-query subset, so jina_clip_v2_w7.npz covers 28k products. W9
evaluates against the full 3-month judgement list (6.5k queries / 80k products), which needs the
remaining ~52k product photos encoded.

Resumable and seeded: already-encoded vectors are carried over from jina_clip_v2_w7.npz and from
any partial run of this script, so an interrupted run continues instead of restarting.

Usage:
  python 19_embed_jina_clip_v2_full.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MODEL_ID = "jinaai/jina-clip-v2"
DIM = 1024
SEED_NPZ = config.EMB_DIR / "jina_clip_v2_w7.npz"


def load_progress(*paths: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    queries: dict[str, np.ndarray] = {}
    images: dict[str, np.ndarray] = {}
    for path in paths:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        for name, vec in zip(data["queries"], data["query_emb"]):
            queries[str(name)] = vec
        for name, vec in zip(data["ecodes"], data["image_emb"]):
            images[str(name)] = vec
        print(f"  seeded from {path.name}: {len(data['queries'])} queries, {len(data['ecodes'])} images")
    return queries, images


def save_progress(path: Path, queries: dict[str, np.ndarray], images: dict[str, np.ndarray]) -> None:
    # Must end in .npz: savez_compressed silently appends the extension otherwise.
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        queries=np.array(list(queries), dtype=object),
        query_emb=np.array(list(queries.values()), dtype=np.float32).reshape(-1, DIM),
        ecodes=np.array(list(images), dtype=object),
        image_emb=np.array(list(images.values()), dtype=np.float32).reshape(-1, DIM),
    )
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=config.EMB_DIR / "jina_clip_v2_full.npz")
    args = parser.parse_args()

    config.ensure_dirs()

    import torch
    from transformers import AutoModel

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv", usecols=["search_term", "ecode"])
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)

    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)
    ok_ecodes = set(manifest.loc[manifest["ok"] == True, "ecode"].astype(str))  # noqa: E712
    test_set = test_set[test_set["ecode"].isin(ok_ecodes)]

    queries = sorted(test_set["search_term"].unique())
    ecodes = sorted(test_set["ecode"].unique())

    q_done, e_done = load_progress(SEED_NPZ, args.output)
    # A seed file may contain products that dropped out of the current manifest; keep only the
    # ecodes this run is actually meant to cover so the saved axis matches the evaluation set.
    q_done = {k: v for k, v in q_done.items() if k in set(queries)}
    e_done = {k: v for k, v in e_done.items() if k in set(ecodes)}

    todo_q = [q for q in queries if q not in q_done]
    todo_e = [e for e in ecodes if e not in e_done]
    print(f"{MODEL_ID} on {device}")
    print(f"queries {len(queries)} ({len(todo_q)} to do)   products {len(ecodes)} ({len(todo_e)} to do)")
    if not todo_q and not todo_e:
        print(f"complete -> {args.output}")
        return 0

    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
    torch_device = torch.device(device)

    if todo_q:
        print(f"encoding {len(todo_q)} queries")
        vecs = model.encode_text(
            todo_q,
            batch_size=args.batch_size,
            show_progress_bar=True,
            device=torch_device,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        q_done.update(zip(todo_q, vecs))
        save_progress(args.output, q_done, e_done)

    t0 = time.time()
    for start in range(0, len(todo_e), args.checkpoint_every):
        chunk = todo_e[start : start + args.checkpoint_every]
        paths = [str(config.IMAGE_DIR / f"{ecode}.jpg") for ecode in chunk]
        vecs = model.encode_image(
            paths,
            batch_size=args.batch_size,
            show_progress_bar=False,
            device=torch_device,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        e_done.update(zip(chunk, vecs))
        save_progress(args.output, q_done, e_done)
        done = start + len(chunk)
        rate = done / (time.time() - t0)
        eta = (len(todo_e) - done) / rate / 60
        print(f"  image {done}/{len(todo_e)}  {rate:.2f}/s  eta {eta:.0f}m", flush=True)

    save_progress(args.output, q_done, e_done)
    print(f"saved -> {args.output}  ({len(q_done)} queries, {len(e_done)} products)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
