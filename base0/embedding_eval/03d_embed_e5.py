#!/usr/bin/env python3
"""Step 3d: encode the full W6 test set with three E5 text models, as fusion partners for SigLIP.

  e5-base            intfloat/e5-base-v2                    (110M, English-only)
  e5-small-multi      intfloat/multilingual-e5-small          (118M, 94 languages)
  e5-large-instruct   intfloat/multilingual-e5-large-instruct (560M, 94 languages, instruct queries)

Mirrors 03_embed.py's caching/assembly pattern (queries, ecodes, query_emb, title_emb, attr_emb),
writing embeddings/e5_base.npz, embeddings/e5_small_multi.npz, embeddings/e5_large_instruct.npz so
09_fusion_experiment.py can treat them exactly like jina_small.npz.

E5 conventions (from each model card):
  e5-base-v2, multilingual-e5-small  -- prefix "query: " / "passage: ", mean pooling, L2 normalize.
  multilingual-e5-large-instruct     -- query gets "Instruct: {task}\\nQuery: {q}"; documents get
                                         NO prefix at all. Same mean pooling / normalize.

Usage:
  python 03d_embed_e5.py --variant base
  python 03d_embed_e5.py --variant small-multi
  python 03d_embed_e5.py --variant large-instruct
  python 03d_embed_e5.py --variant all
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

INSTRUCT_TASK = "Given a web search query, retrieve relevant product titles that answer the query"

VARIANTS = {
    "base": (config.E5_BASE_MODEL, False, 32),
    "small-multi": (config.E5_SMALL_MULTI_MODEL, False, 32),
    "large-instruct": (config.E5_LARGE_INSTRUCT_MODEL, True, 16),
}
OUT_NAMES = {"base": "e5_base", "small-multi": "e5_small_multi", "large-instruct": "e5_large_instruct"}


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_e5(
    model_name: str,
    instruct: bool,
    device: torch.device,
    queries: list[str],
    titles: list[str],
    batch_size: int,
    attrs: list[str] | None = None,
) -> dict[str, np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    print(f"loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(texts: list[str], is_query: bool, label: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if instruct:
                prefixed = (
                    [f"Instruct: {INSTRUCT_TASK}\nQuery: {t}" for t in batch] if is_query else list(batch)
                )
            else:
                prefixed = [f"{'query' if is_query else 'passage'}: {t}" for t in batch]
            enc = tok(
                prefixed, max_length=512, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
                pooled = average_pool(hidden, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
            print(f"  e5 {label}: {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return np.vstack(out)

    return {
        "query": encode(queries, True, "query"),
        "title": encode(titles, False, "title"),
        **({"attr": encode(attrs, False, "attr")} if attrs is not None else {}),
    }


def run_variant(variant: str, device_arg: str, force: bool) -> None:
    model_name, instruct, default_batch = VARIANTS[variant]
    out_path = config.EMB_DIR / f"{OUT_NAMES[variant]}.npz"
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

    required = ("attr_emb",) if attr_texts is not None else ()
    if not force and embed.is_cached(out_path, queries, ecodes, required):
        print(f"cached  -> {out_path}")
        return

    device = embed.pick_device(device_arg)
    cache = None if force else embed.load_cache(out_path, required)
    new_q, new_e = embed.missing_items(cache, queries, ecodes)
    print(f"variant: {variant} ({model_name})")
    print(f"queries: {len(queries)} ({len(new_q)} to do)   products: {len(ecodes)} ({len(new_e)} to do)")

    e_pos = {e: i for i, e in enumerate(ecodes)}
    rows = [e_pos[e] for e in new_e]

    e5 = encode_e5(
        model_name,
        instruct,
        device,
        new_q,
        [titles[r] for r in rows],
        default_batch,
        attrs=[attr_texts[r] for r in rows] if attr_texts is not None else None,
    )

    payload = {
        "queries": np.array(queries, dtype=object),
        "ecodes": np.array(ecodes, dtype=object),
        "query_emb": embed.assemble(cache, "query_emb", queries, new_q, e5["query"], "q_idx"),
        "title_emb": embed.assemble(cache, "title_emb", ecodes, new_e, e5["title"], "e_idx"),
    }
    if attr_texts is not None:
        payload["attr_emb"] = embed.assemble(cache, "attr_emb", ecodes, new_e, e5["attr"], "e_idx")
    np.savez_compressed(out_path, **payload)
    print(f"saved -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for variant in variants:
        run_variant(variant, args.device, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
