#!/usr/bin/env python3
"""Add deployed SigLIP query rankings to the existing W7 per-query evaluation."""

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

SYSTEM = "siglip_es_image"


def main() -> int:
    test_set = pd.read_csv(config.DATA_DIR / "w7_subset.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)
    es_data = np.load(config.EMB_DIR / "siglip_es_w7.npz", allow_pickle=True)
    es_queries = {str(k): v for k, v in zip(es_data["queries"], es_data["query_emb"])}
    siglip = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    ecodes = {str(k): i for i, k in enumerate(siglip["ecodes"])}
    test_set = test_set[test_set["search_term"].isin(es_queries)]
    query_names = sorted(test_set["search_term"].unique())
    product_names = sorted(test_set["ecode"].unique())
    query_matrix = np.ascontiguousarray([es_queries[name] for name in query_names])
    image_matrix = np.ascontiguousarray(siglip["image_emb"][[ecodes[name] for name in product_names]])
    score_matrix = query_matrix @ image_matrix.T
    score_index = {name: i for i, name in enumerate(query_names)}
    product_index = {name: i for i, name in enumerate(product_names)}

    rows: list[dict] = []
    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        if len(pool) < 5:
            continue
        scores = score_matrix[score_index[term], [product_index[ecode] for ecode in pool["ecode"]]]
        order = np.argsort(-scores, kind="stable")
        for label_set, column in (("ltr", "relevance"), ("raw_ctr", "relevance_raw")):
            rels = pool[column].to_numpy(dtype=float)
            if rels.max() <= 0:
                continue
            rows.append({
                "label_set": label_set,
                "search_term": term,
                "query_tier": pool["query_tier"].iloc[0],
                "system": SYSTEM,
                "pool_size": len(pool),
                "n_positive": int((rels > 0).sum()),
                "query_impressions": float(pool["total_impressions"].sum()),
                **evaluate.score_ranking(rels, order),
            })

    old = pd.read_csv(config.RESULTS_DIR / "w7_per_query.csv")
    old = old[old["system"] != SYSTEM]
    per_query = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    per_query.to_csv(config.RESULTS_DIR / "w7_per_query.csv", index=False)
    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]

    def summarise(keys: list[str]) -> pd.DataFrame:
        macro = per_query.groupby(keys)[metric_cols].mean().reset_index()
        macro["weighting"] = "macro"

        def wmean(group: pd.DataFrame) -> pd.Series:
            weights = group["query_impressions"].to_numpy()
            return pd.Series({c: float(np.average(group[c], weights=weights)) for c in metric_cols})

        weighted = per_query.groupby(keys)[metric_cols + ["query_impressions"]].apply(wmean).reset_index()
        weighted["weighting"] = "impression"
        out = pd.concat([macro, weighted], ignore_index=True)
        counts = per_query.groupby(keys)["search_term"].nunique().reset_index()
        return out.merge(counts.rename(columns={"search_term": "n_queries"}), on=keys)

    summary = summarise(["label_set", "system"])
    summary.sort_values(
        ["weighting", "label_set", "ndcg@10"], ascending=[True, True, False]
    ).to_csv(config.RESULTS_DIR / "w7_summary.csv", index=False)
    by_tier = summarise(["label_set", "query_tier", "system"])
    by_tier["query_tier"] = pd.Categorical(by_tier["query_tier"], ["head", "torso", "tail"], ordered=True)
    by_tier.sort_values(
        ["weighting", "label_set", "query_tier", "ndcg@10"], ascending=[True, True, True, False]
    ).to_csv(config.RESULTS_DIR / "w7_summary_by_tier.csv", index=False)

    subset = per_query[per_query["label_set"] == "ltr"]
    wide = subset.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
    volume = subset.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
    active = ["siglip_image", "omni_nano_image", "omni_small_image", SYSTEM]
    pairs = [(a, b) for i, a in enumerate(active) for b in active[i + 1:]]
    pairs += [(a, r) for a in active for r in ("siglip_attr", "production", "random")]
    rng = np.random.default_rng(config.RANDOM_SEED)
    significance = []
    bootstrap = import_module("09_fusion_experiment").bootstrap_contrast
    for system, baseline in pairs:
        if system not in wide or baseline not in wide:
            continue
        significance.append(bootstrap(wide, volume, system, baseline, rng, 200))
    pd.DataFrame(significance).to_csv(config.RESULTS_DIR / "w7_significance.csv", index=False)

    meta = json.loads((config.RESULTS_DIR / "w7_meta.json").read_text())
    meta["image_systems"] = active
    meta["bootstrap_samples"] = 200
    (config.RESULTS_DIR / "w7_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary[(summary.label_set == "ltr") & (summary.weighting == "macro")][["system", "n_queries", "ndcg@10", "map"]].sort_values("ndcg@10", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
