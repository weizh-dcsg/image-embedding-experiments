#!/usr/bin/env python3
"""Step 3b: encode the same queries/products with Jina v5 text small (677M) as a capacity control.

Mirrors 03_embed.py's jina encoding but against config.JINA_SMALL_MODEL, writing a separate
embeddings/jina_small.npz so it doesn't clobber the nano-model embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from importlib import import_module  # noqa: E402

embed = import_module("03_embed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")

    queries = sorted(test_set["search_term"].unique())
    products = test_set[["ecode", "product_title"]].drop_duplicates(subset="ecode").sort_values("ecode")
    ecodes = products["ecode"].tolist()
    titles = products["product_title"].astype(str).tolist()

    attr_texts: list[str] | None = None
    if config.ATTRIBUTES_CSV.exists():
        attr_map = (
            pd.read_csv(config.ATTRIBUTES_CSV)
            .assign(attr_text=lambda d: d["attr_text"].fillna("").astype(str))
            .set_index("ecode")["attr_text"]
            .to_dict()
        )
        attr_texts = [attr_map.get(e) or t for e, t in zip(ecodes, titles)]

    out_path = config.EMB_DIR / "jina_small.npz"
    required = ("attr_emb",) if attr_texts is not None else ()
    if not args.force and embed.is_cached(out_path, queries, ecodes, required):
        print(f"cached  -> {out_path}")
        return 0

    device = embed.pick_device(args.device)
    cache = None if args.force else embed.load_cache(out_path, required)
    new_q, new_e = embed.missing_items(cache, queries, ecodes)
    print(f"queries: {len(queries)}   products: {len(ecodes)}")
    print(f"model:   {config.JINA_SMALL_MODEL}")
    print(f"encoding {len(new_q)} new queries, {len(new_e)} new products")

    e_pos = {e: i for i, e in enumerate(ecodes)}
    rows = [e_pos[e] for e in new_e]

    # embed.encode_jina reads config.JINA_MODEL internally; point it at the small model for
    # this run only, then restore so any later import of config in this process is unaffected.
    original_model = config.JINA_MODEL
    config.JINA_MODEL = config.JINA_SMALL_MODEL
    try:
        jina = embed.encode_jina(
            device,
            new_q,
            [titles[r] for r in rows],
            args.batch_size,
            attrs=[attr_texts[r] for r in rows] if attr_texts is not None else None,
        )
    finally:
        config.JINA_MODEL = original_model

    payload = {
        "queries": np.array(queries, dtype=object),
        "ecodes": np.array(ecodes, dtype=object),
        "query_emb": embed.assemble(cache, "query_emb", queries, new_q, jina["query"], "q_idx"),
        "title_emb": embed.assemble(cache, "title_emb", ecodes, new_e, jina["title"], "e_idx"),
    }
    if attr_texts is not None:
        payload["attr_emb"] = embed.assemble(cache, "attr_emb", ecodes, new_e, jina["attr"], "e_idx")
    np.savez_compressed(out_path, **payload)
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
