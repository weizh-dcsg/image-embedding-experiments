#!/usr/bin/env python3
"""Step 8: modality complementarity, oracle headroom, and routing.

Fusion beating both single modalities implies the two carry complementary signal. This quantifies
how much, and asks whether a cheap router can capture it without paying for both encoders at
query time.

Three routers are evaluated against the oracle upper bound:
  brand     lexical -- route to text when the query contains a catalogue brand token
  margin    unsupervised -- route to whichever system is more confident on this query
            (higher top-1 similarity after per-query z-normalisation)
  oracle    upper bound -- per query, take the better of the two systems (not achievable)

Outputs: results/routing.csv, results/complementarity.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

IMG = "siglip_image_crop"
TXT = "jina_text"


def dcg(gains: np.ndarray) -> float:
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def ndcg_at_k(rels_ranked: np.ndarray, k: int = 10) -> float:
    gains = (2.0 ** rels_ranked[:k]) - 1.0
    ideal = (2.0 ** np.sort(rels_ranked)[::-1][:k]) - 1.0
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def brand_vocabulary(products: pd.DataFrame) -> set[str]:
    vocab: set[str] = set()
    for b in products["brand_name"].dropna().astype(str):
        for tok in re.findall(r"[a-z0-9']+", b.lower()):
            if len(tok) > 2:
                vocab.add(tok)
    return vocab


def main() -> int:
    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    products = pd.read_csv(config.PRODUCTS_CSV)

    siglip = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    jina = np.load(config.EMB_DIR / "jina.npz", allow_pickle=True)
    queries = [str(q) for q in siglip["queries"]]
    ecodes = [str(e) for e in siglip["ecodes"]]
    q_idx = {q: i for i, q in enumerate(queries)}
    e_idx = {e: i for i, e in enumerate(ecodes)}

    emb = {
        IMG: {"q": siglip["query_emb"], "d": siglip["image_crop_emb"]},
        TXT: {"q": jina["query_emb"], "d": jina["title_emb"]},
    }

    brands = brand_vocabulary(products)
    rows = []

    for term, pool in test_set.groupby("search_term"):
        if term not in q_idx or len(pool) < 5:
            continue
        rels = pool["relevance"].to_numpy(dtype=float)
        if rels.max() <= 0:
            continue

        cand = np.array([e_idx[e] for e in pool["ecode"]])
        qi = q_idx[term]

        sims, scores = {}, {}
        for name, e in emb.items():
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                s = np.ascontiguousarray(e["d"][cand]) @ e["q"][qi]
            sims[name] = s
            scores[name] = ndcg_at_k(rels[np.argsort(-s, kind="stable")])

        fused = 0.5 * (zscore(sims[IMG]) + zscore(sims[TXT]))
        fusion_score = ndcg_at_k(rels[np.argsort(-fused, kind="stable")])

        # confidence = how far the best candidate stands out from the pool, in pool std units
        margin = {n: float(zscore(s).max()) for n, s in sims.items()}
        has_brand = any(t in brands for t in re.findall(r"[a-z0-9']+", term.lower()))

        rows.append(
            {
                "search_term": term,
                "n_words": len(term.split()),
                "has_brand_token": has_brand,
                "ndcg_image": scores[IMG],
                "ndcg_text": scores[TXT],
                "ndcg_fusion": fusion_score,
                "margin_image": margin[IMG],
                "margin_text": margin[TXT],
                "ndcg_oracle": max(scores[IMG], scores[TXT]),
                "ndcg_router_brand": scores[TXT] if has_brand else scores[IMG],
                "ndcg_router_margin": scores[IMG] if margin[IMG] >= margin[TXT] else scores[TXT],
                "image_better": scores[IMG] > scores[TXT],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "routing.csv", index=False)

    rng = np.random.default_rng(config.RANDOM_SEED)

    def boot(a: str, b: str) -> dict:
        d = (df[a] - df[b]).to_numpy()
        idx = rng.integers(0, len(d), size=(config.BOOTSTRAP_SAMPLES, len(d)))
        bs = d[idx].mean(axis=1)
        return {
            "delta": float(d.mean()),
            "rel_pct": float(100 * d.mean() / df[b].mean()),
            "ci_low": float(np.percentile(bs, 2.5)),
            "ci_high": float(np.percentile(bs, 97.5)),
            "p": float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
        }

    best_single = "ndcg_image" if df["ndcg_image"].mean() >= df["ndcg_text"].mean() else "ndcg_text"
    summary = {
        "n_queries": int(len(df)),
        "mean_ndcg": {
            "image": float(df["ndcg_image"].mean()),
            "text": float(df["ndcg_text"].mean()),
            "fusion": float(df["ndcg_fusion"].mean()),
            "router_brand": float(df["ndcg_router_brand"].mean()),
            "router_margin": float(df["ndcg_router_margin"].mean()),
            "oracle": float(df["ndcg_oracle"].mean()),
        },
        "queries_where_image_better": int(df["image_better"].sum()),
        "brand_token_queries": int(df["has_brand_token"].sum()),
        "brand_router_accuracy": float((df["has_brand_token"] != df["image_better"]).mean()),
        "margin_router_accuracy": float(
            ((df["margin_image"] >= df["margin_text"]) == df["image_better"]).mean()
        ),
        "oracle_vs_best_single": boot("ndcg_oracle", best_single),
        "fusion_vs_best_single": boot("ndcg_fusion", best_single),
        "router_brand_vs_best_single": boot("ndcg_router_brand", best_single),
        "router_margin_vs_best_single": boot("ndcg_router_margin", best_single),
        "fusion_share_of_oracle_headroom": float(
            (df["ndcg_fusion"].mean() - df[best_single].mean())
            / (df["ndcg_oracle"].mean() - df[best_single].mean())
        ),
    }
    (config.RESULTS_DIR / "complementarity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
