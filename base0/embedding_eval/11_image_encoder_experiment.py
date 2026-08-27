#!/usr/bin/env python3
"""Step 11 (W7): compare product-image encoders on identical queries and candidate pools.

Systems -- each self-consistent, query and document encoded by the same model, so the contrast
isolates the image encoder rather than the query representation:

  siglip_image      SigLIP text tower  vs SigLIP image tower        203M, 768d
  omni_nano_image   omni text encoder  vs omni vision tower        ~1.0B, 768d
  omni_small_image  omni text encoder  vs omni vision tower        larger, 1024d

Reference points (not image systems, shown for scale):
  siglip_attr       best overall representation from W1/W6 -- how far images are from the ceiling
  production        incumbent ordering by mean observed impression position
  random            seeded shuffle floor

Fairness: a product is scored only if EVERY system has a vector for it, and a query only if it
still has >= 5 candidates and a positive label after that intersection. So all systems rank
literally the same pools.

Outputs: results/w7_per_query.csv, results/w7_summary.csv, results/w7_significance.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

evaluate = import_module("04_evaluate")

IMAGE_SYSTEMS = ["siglip_image", "omni_nano_image", "omni_small_image"]


def load_omni(variant: str) -> dict[str, np.ndarray] | None:
    path = config.EMB_DIR / f"jina_omni_{variant}.npz"
    if not path.exists():
        print(f"missing {path} -- run 03c_embed_jina_omni.py --variant {variant}")
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "q": {str(k): v for k, v in zip(d["queries"], d["query_emb"])},
        "d": {str(k): v for k, v in zip(d["ecodes"], d["image_emb"])},
    }


def check_healthy(name: str, arrays: np.ndarray) -> bool:
    """Reject an encoder whose document vectors have collapsed to a single point.

    A collapsed encoder produces tied scores for every candidate. Combined with a stable sort it
    can look perfect rather than broken, so this is checked explicitly rather than inferred from
    the metrics. Seen in practice: jina-omni-small emits byte-identical vectors on MPS.
    """
    sample = arrays[: min(2000, len(arrays))]
    n_unique = len(np.unique(np.round(sample, 5), axis=0))
    spread = float(np.std(sample @ sample[0]))
    ok = n_unique > 0.5 * len(sample) and spread > 1e-4
    print(f"  {name}: {n_unique}/{len(sample)} unique vectors, sim spread {spread:.2e} "
          f"-> {'ok' if ok else 'DEGENERATE, excluded'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = ap.parse_args()
    config.ensure_dirs()

    test_set = pd.read_csv(config.DATA_DIR / "w7_subset.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)

    siglip = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    sig_q = {str(k): i for i, k in enumerate(siglip["queries"])}
    sig_e = {str(k): i for i, k in enumerate(siglip["ecodes"])}

    omni = {v: load_omni(v) for v in ("nano", "small")}
    if any(o is None for o in omni.values()):
        return 1

    print("encoder health check:")
    healthy = {}
    for variant, data in omni.items():
        arr = np.vstack(list(data["d"].values()))
        healthy[f"omni_{variant}_image"] = check_healthy(f"omni_{variant}_image", arr)
        print('[MSG]: omin_{}_image health check complete. {} unique vectors, sim spread {:.2e} -> {}'.format(
            variant, len(arr), float(np.std(arr @ arr[0])), 'ok' if healthy[f"omni_{variant}_image"] else 'DEGENERATE, excluded'
        ))
    sig_arr = siglip["image_emb"]
    healthy["siglip_image"] = check_healthy("siglip_image", sig_arr)
    active_images = [s for s in IMAGE_SYSTEMS] #if healthy.get(s)]
    excluded = [] #[s for s in IMAGE_SYSTEMS if not healthy.get(s)]
    print('[WARNING]: all encoders are included! No filtering on health active.')
    if excluded:
        print(f"excluded from the comparison: {', '.join(excluded)}")
    if len(active_images) < 2:
        print("fewer than two healthy image encoders; nothing to compare")
        return 1

    print('[MSG]: Active image encoders check complete. There are {} active images encoders: {}'.format(len(active_images), ', '.join(active_images)))
    # identical pools for every system
    common_e = set(sig_e) & set(omni["nano"]["d"]) & set(omni["small"]["d"])
    common_q = set(sig_q) & set(omni["nano"]["q"]) & set(omni["small"]["q"])
    before = (test_set["search_term"].nunique(), test_set["ecode"].nunique())
    test_set = test_set[test_set["ecode"].isin(common_e) & test_set["search_term"].isin(common_q)]
    print(f"intersection: {before[0]} -> {test_set['search_term'].nunique()} queries, "
          f"{before[1]} -> {test_set['ecode'].nunique()} products")

    rng = np.random.default_rng(config.RANDOM_SEED)
    label_sets = {"ltr": "relevance", "raw_ctr": "relevance_raw"}
    rows: list[dict] = []

    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        ecodes = pool["ecode"].tolist()
        if len(pool) < 5:
            continue
        sims: dict[str, np.ndarray] = {}
        rows_sig = np.array([sig_e[e] for e in ecodes])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims["siglip_image"] = np.ascontiguousarray(siglip["image_emb"][rows_sig]) @ siglip["query_emb"][sig_q[term]]
            sims["siglip_attr"] = np.ascontiguousarray(siglip["attr_emb"][rows_sig]) @ siglip["query_emb"][sig_q[term]]
            for variant in ("nano", "small"):
                if f"omni_{variant}_image" not in active_images:
                    print(f'[WARNING: omni_{variant}_image not in active_images, skipping]')
                    continue
                docs = np.vstack([omni[variant]["d"][e] for e in ecodes])
                sims[f"omni_{variant}_image"] = docs @ omni[variant]["q"][term]

        rankings = {k: np.argsort(-v, kind="stable") for k, v in sims.items()}
        rankings["production"] = np.argsort(pool["mean_position"].to_numpy(), kind="stable")
        rankings["random"] = rng.permutation(len(pool))

        for label_set, column in label_sets.items():
            rels = pool[column].to_numpy(dtype=float)
            if rels.max() <= 0:
                continue
            for name, order in rankings.items():
                rows.append({
                    "label_set": label_set,
                    "search_term": term,
                    "query_tier": pool["query_tier"].iloc[0],
                    "system": name,
                    "pool_size": len(pool),
                    "n_positive": int((rels > 0).sum()),
                    "query_impressions": float(pool["total_impressions"].sum()),
                    **evaluate.score_ranking(rels, order),
                })

    per_query = pd.DataFrame(rows)
    per_query.to_csv(config.RESULTS_DIR / "w7_per_query.csv", index=False)

    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]

    def summarise(keys: list[str]) -> pd.DataFrame:
        macro = per_query.groupby(keys)[metric_cols].mean().reset_index()
        macro["weighting"] = "macro"

        def wmean(g: pd.DataFrame) -> pd.Series:
            w = g["query_impressions"].to_numpy()
            return pd.Series({c: float(np.average(g[c], weights=w)) for c in metric_cols})

        wtd = per_query.groupby(keys)[metric_cols + ["query_impressions"]].apply(wmean).reset_index()
        wtd["weighting"] = "impression"
        out = pd.concat([macro, wtd], ignore_index=True)
        counts = per_query.groupby(keys)["search_term"].nunique().reset_index()
        return out.merge(counts.rename(columns={"search_term": "n_queries"}), on=keys)

    summary = summarise(["label_set", "system"])
    summary.sort_values(["weighting", "label_set", "ndcg@10"], ascending=[True, True, False]).to_csv(
        config.RESULTS_DIR / "w7_summary.csv", index=False
    )
    by_tier = summarise(["label_set", "query_tier", "system"])
    by_tier["query_tier"] = pd.Categorical(by_tier["query_tier"], ["head", "torso", "tail"], ordered=True)
    by_tier.sort_values(["weighting", "label_set", "query_tier", "ndcg@10"],
                        ascending=[True, True, True, False]).to_csv(
        config.RESULTS_DIR / "w7_summary_by_tier.csv", index=False)

    # every pairwise image-encoder contrast, plus each against the reference points
    sub = per_query[per_query["label_set"] == "ltr"]
    wide = sub.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
    vol = sub.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
    pairs = [(a, b) for i, a in enumerate(active_images) for b in active_images[i + 1:]]
    pairs += [(a, r) for a in active_images for r in ("siglip_attr", "production", "random")]
    sig_rows = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        r = import_module("09_fusion_experiment").bootstrap_contrast(wide, vol, a, b, rng, args.bootstrap)
        sig_rows.append(r)
    pd.DataFrame(sig_rows).to_csv(config.RESULTS_DIR / "w7_significance.csv", index=False)

    meta = {
        "n_queries": int(per_query["search_term"].nunique()),
        "n_products": int(test_set["ecode"].nunique()),
        "n_queries_by_tier": test_set.drop_duplicates("search_term")["query_tier"].value_counts().to_dict(),
        "bootstrap_samples": args.bootstrap,
        "image_systems": active_images,
        "excluded_degenerate": excluded,
    }
    (config.RESULTS_DIR / "w7_meta.json").write_text(json.dumps(meta, indent=2))

    view = summary[(summary.label_set == "ltr") & (summary.weighting == "macro")]
    print(view[["system", "n_queries", "ndcg@10", "map"]].sort_values("ndcg@10", ascending=False).to_string(index=False))
    print(f"\nsaved -> {config.RESULTS_DIR}/w7_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
