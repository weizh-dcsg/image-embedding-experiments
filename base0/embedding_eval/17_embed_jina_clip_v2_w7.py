#!/usr/bin/env python3
"""Encode the W7 subset with Jina CLIP v2 text and image towers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MODEL_ID = "jinaai/jina-clip-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--product-limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=config.EMB_DIR / "jina_clip_v2_w7.npz")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    test_set = pd.read_csv(config.DATA_DIR / "w7_subset.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)
    queries = sorted(test_set["search_term"].unique())
    if args.query_limit is not None:
        queries = queries[: args.query_limit]
        test_set = test_set[test_set["search_term"].isin(queries)]
    if args.product_limit is not None:
        products = sorted(test_set["ecode"].unique())[: args.product_limit]
        test_set = test_set[test_set["ecode"].isin(products)]
    ecodes = sorted(test_set["ecode"].unique())
    image_paths = [config.IMAGE_DIR / f"{ecode}.jpg" for ecode in ecodes]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"missing {len(missing)} W7 images; first: {missing[0]}")

    print(f"loading {MODEL_ID} on {device}")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()

    print(f"encoding {len(queries)} queries")
    query_emb = model.encode_text(
        queries,
        batch_size=args.batch_size,
        show_progress_bar=True,
        device=torch.device(device),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"encoding {len(ecodes)} images")
    image_emb = model.encode_image(
        [str(path) for path in image_paths],
        batch_size=args.batch_size,
        show_progress_bar=True,
        device=torch.device(device),
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if query_emb.shape != (len(queries), 1024):
        raise RuntimeError(f"unexpected query shape: {query_emb.shape}")
    if image_emb.shape != (len(ecodes), 1024):
        raise RuntimeError(f"unexpected image shape: {image_emb.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, queries=np.asarray(queries, dtype=object), ecodes=np.asarray(ecodes, dtype=object), query_emb=query_emb, image_emb=image_emb)
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
