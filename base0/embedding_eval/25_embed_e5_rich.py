#!/usr/bin/env python3
"""Step 25 (W9): does enriching the E5 document text with the Big-4 attributes help?

Production embeds `passage: {name}` and nothing else -- verified, cosine 1.0 against the stored
vector. The Big-4 attributes (brand, product type, activity, gender/age) are indexed for BM25 but
never reach the vector. This tests whether folding them into the embedded text helps, which would
be an indexing change rather than a new model.

Document variants, all multilingual-e5-small with the model-card conventions
(`passage: ` prefix, mean pooling, L2 normalise) so they are directly comparable to `e5_local`:

  title        product title only                  -- what production does
  attr         Big-4 attribute string only         -- already measured in W1
  title_attr   title + Big-4 attributes            -- the proposal

Writes embeddings/e5_small_multi_rich.npz with a `title_attr_emb` array; queries and the ecode
axis are reused verbatim from e5_small_multi.npz so nothing else shifts.

Usage:
  python 25_embed_e5_rich.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"
OUT = config.EMB_DIR / "e5_small_multi_rich.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    base = np.load(config.EMB_DIR / "e5_small_multi.npz", allow_pickle=True)
    ecodes = [str(e) for e in base["ecodes"]]

    products = pd.read_csv(config.DATA_DIR / "products.csv")
    products["ecode"] = products["ecode"].astype(str)
    titles = products.set_index("ecode")["product_title"].astype(str).to_dict()

    attrs = pd.read_csv(config.ATTRIBUTES_CSV)
    attrs["ecode"] = attrs["ecode"].astype(str)
    attr_text = attrs.set_index("ecode")["attr_text"].fillna("").astype(str).to_dict()

    texts = []
    for e in ecodes:
        title = titles.get(e, "")
        extra = attr_text.get(e, "")
        texts.append(f"{title} {extra}".strip() if extra else title)
    n_with_attrs = sum(1 for e in ecodes if attr_text.get(e))
    print(f"{len(texts):,} documents, {n_with_attrs:,} have Big-4 attributes")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    out = np.zeros((len(texts), 384), dtype=np.float32)
    for start in range(0, len(texts), args.batch_size):
        batch = [f"passage: {t}" for t in texts[start : start + args.batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=192, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        out[start : start + len(batch)] = pooled.cpu().numpy()
        if start % (args.batch_size * 100) == 0:
            print(f"  {start:,}/{len(texts):,}", end="\r", flush=True)
    print()

    np.savez_compressed(
        OUT,
        queries=base["queries"],
        query_emb=base["query_emb"],
        ecodes=np.asarray(ecodes, dtype=object),
        title_attr_emb=out,
    )
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
