#!/usr/bin/env python3
"""Step 3e: encode the full W6 test set with the Jina v5 omni-nano TEXT tower.

W7/W8 only used omni-nano's *image* tower (query text vs product photo) on a 600-query subset.
This adds omni-nano as a text-representation fusion partner for W6 (query text vs product title /
Big-4 attribute string), on the full test set, so it's directly comparable to text-jina,
text-jina-small, etc. Loaded with modality="text" (skips the vision/audio towers) since only the
text encoder is needed here.

Retrieval prefixes apply to every modality per the model card: "Query: " / "Document: ".

Output: embeddings/jina_omni_nano_text.npz (queries, ecodes, query_emb, title_emb, attr_emb),
same shape as jina_small.npz so 09_fusion_experiment.py can treat it identically.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

embed = import_module("03_embed")


def encode_omni_text(
    device, queries: list[str], titles: list[str], batch_size: int, attrs: list[str] | None = None
) -> dict[str, np.ndarray]:
    from transformers import AutoModel, AutoProcessor

    repo = config.JINA_OMNI_NANO_MODEL
    print(f"loading {repo} on {device} (modality=text)")
    model = AutoModel.from_pretrained(
        repo, trust_remote_code=True, default_task="retrieval", modality="text"
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

    def encode(texts: list[str], prefix: str, label: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inp = proc(text=[f"{prefix}: {t}" for t in batch], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                vecs = model.embed(**inp.to(device)).float().cpu().numpy()
            out.append(vecs)
            print(f"  omni_text {label}: {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return np.vstack(out)

    return {
        "query": encode(queries, "Query", "query"),
        "title": encode(titles, "Document", "title"),
        **({"attr": encode(attrs, "Document", "attr")} if attrs is not None else {}),
    }


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

    out_path = config.EMB_DIR / "jina_omni_nano_text.npz"
    required = ("attr_emb",) if attr_texts is not None else ()
    if not args.force and embed.is_cached(out_path, queries, ecodes, required):
        print(f"cached  -> {out_path}")
        return 0

    device = embed.pick_device(args.device)
    cache = None if args.force else embed.load_cache(out_path, required)
    new_q, new_e = embed.missing_items(cache, queries, ecodes)
    print(f"queries: {len(queries)} ({len(new_q)} to do)   products: {len(ecodes)} ({len(new_e)} to do)")

    e_pos = {e: i for i, e in enumerate(ecodes)}
    rows = [e_pos[e] for e in new_e]

    omni = encode_omni_text(
        device,
        new_q,
        [titles[r] for r in rows],
        args.batch_size,
        attrs=[attr_texts[r] for r in rows] if attr_texts is not None else None,
    )

    payload = {
        "queries": np.array(queries, dtype=object),
        "ecodes": np.array(ecodes, dtype=object),
        "query_emb": embed.assemble(cache, "query_emb", queries, new_q, omni["query"], "q_idx"),
        "title_emb": embed.assemble(cache, "title_emb", ecodes, new_e, omni["title"], "e_idx"),
    }
    if attr_texts is not None:
        payload["attr_emb"] = embed.assemble(cache, "attr_emb", ecodes, new_e, omni["attr"], "e_idx")
    np.savez_compressed(out_path, **payload)
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
