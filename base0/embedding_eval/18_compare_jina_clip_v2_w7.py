#!/usr/bin/env python3
"""Compare Jina CLIP v2 with existing W7 systems on the full W7 subset."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

evaluate = import_module("04_evaluate")
bootstrap_contrast = import_module("09_fusion_experiment").bootstrap_contrast

SYSTEMS = ["jina_clip_v2", "siglip_image", "omni_nano_image", "omni_small_image", "production", "random"]
LABELS = {"jina_clip_v2": "jina-clip-v2"}


def main() -> int:
    test_set = pd.read_csv(config.DATA_DIR / "w7_subset.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)
    queries = sorted(test_set["search_term"].unique())
    ecodes = sorted(test_set["ecode"].unique())

    jina = np.load(config.EMB_DIR / "jina_clip_v2_w7.npz", allow_pickle=True)
    jina_q = {str(k): v for k, v in zip(jina["queries"], jina["query_emb"])}
    jina_d = {str(k): v for k, v in zip(jina["ecodes"], jina["image_emb"])}
    missing = set(test_set["search_term"]) - set(jina_q) | (set(test_set["ecode"]) - set(jina_d))
    if missing:
        raise RuntimeError(f"missing {len(missing)} encoded W7 keys")

    siglip = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    sig_q = {str(k): v for k, v in zip(siglip["queries"], siglip["query_emb"])}
    sig_d = {str(k): v for k, v in zip(siglip["ecodes"], siglip["image_emb"])}
    omni = {}
    for variant in ("nano", "small"):
        data = np.load(config.EMB_DIR / f"jina_omni_{variant}.npz", allow_pickle=True)
        omni[variant] = {
            "q": {str(k): v for k, v in zip(data["queries"], data["query_emb"])},
            "d": {str(k): v for k, v in zip(data["ecodes"], data["image_emb"])},
        }

    rng = np.random.default_rng(config.RANDOM_SEED)
    rows: list[dict] = []
    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        if len(pool) < 5:
            continue
        pool_ecodes = pool["ecode"].tolist()
        jina_scores = np.vstack([jina_d[e] for e in pool_ecodes]) @ jina_q[term]
        siglip_scores = np.vstack([sig_d[e] for e in pool_ecodes]) @ sig_q[term]
        omni_scores = {
            variant: np.vstack([omni[variant]["d"][e] for e in pool_ecodes]) @ omni[variant]["q"][term]
            for variant in ("nano", "small")
        }
        rankings = {
            "jina_clip_v2": np.argsort(-jina_scores, kind="stable"),
            "siglip_image": np.argsort(-siglip_scores, kind="stable"),
            "omni_nano_image": np.argsort(-omni_scores["nano"], kind="stable"),
            "omni_small_image": np.argsort(-omni_scores["small"], kind="stable"),
            "production": np.argsort(pool["mean_position"].to_numpy(), kind="stable"),
            "random": rng.permutation(len(pool)),
        }
        for label_set, column in (("ltr", "relevance"), ("raw_ctr", "relevance_raw")):
            rels = pool[column].to_numpy(dtype=float)
            if rels.max() <= 0:
                continue
            for system, order in rankings.items():
                rows.append({
                    "label_set": label_set,
                    "search_term": term,
                    "query_tier": pool["query_tier"].iloc[0],
                    "system": system,
                    "pool_size": len(pool),
                    "n_positive": int((rels > 0).sum()),
                    "query_impressions": float(pool["total_impressions"].sum()),
                    **evaluate.score_ranking(rels, order),
                })

    per_query = pd.DataFrame(rows)
    out_dir = config.RESULTS_DIR
    per_query.to_csv(out_dir / "w7_jina_clip_v2_full_per_query.csv", index=False)
    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]

    def summarise(keys: list[str]) -> pd.DataFrame:
        macro = per_query.groupby(keys)[metric_cols].mean().reset_index()
        macro["weighting"] = "macro"
        def wmean(group: pd.DataFrame) -> pd.Series:
            weights = group["query_impressions"].to_numpy()
            return pd.Series({c: float(np.average(group[c], weights=weights)) for c in metric_cols})
        weighted = per_query.groupby(keys)[metric_cols + ["query_impressions"]].apply(wmean).reset_index()
        weighted["weighting"] = "impression"
        counts = per_query.groupby(keys)["search_term"].nunique().reset_index()
        return pd.concat([macro, weighted], ignore_index=True).merge(
            counts.rename(columns={"search_term": "n_queries"}), on=keys
        )

    summary = summarise(["label_set", "system"])
    summary.to_csv(out_dir / "w7_jina_clip_v2_full_summary.csv", index=False)
    subset = per_query[per_query["label_set"] == "ltr"]
    wide = subset.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
    volume = subset.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
    significance = []
    for baseline in SYSTEMS[1:]:
        significance.append(bootstrap_contrast(wide, volume, "jina_clip_v2", baseline, rng, 500))
    pd.DataFrame(significance).to_csv(out_dir / "w7_jina_clip_v2_full_significance.csv", index=False)
    meta = {
        "model": "jinaai/jina-clip-v2",
        "n_queries": int(per_query.search_term.nunique()),
        "n_products": len(ecodes),
        "dimensions": 1024,
        "systems": SYSTEMS,
        "bootstrap_samples": 500,
    }
    (out_dir / "w7_jina_clip_v2_full_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary[(summary.label_set == "ltr") & (summary.weighting == "macro")][["system", "n_queries", "ndcg@10", "map"]].sort_values("ndcg@10", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
