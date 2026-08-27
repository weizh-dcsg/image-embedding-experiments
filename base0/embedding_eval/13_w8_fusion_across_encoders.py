#!/usr/bin/env python3
"""Step 13 (W8): does the W6 fusion recipe survive a change of image encoder?

W6 found, using SigLIP as the only image tower, that:
  1. RRF beats z-score averaging and mean-cosine, for every text partner
  2. attr-siglip is the best text representation to fuse with
  3. fusing image into attr-siglip does not beat attr-siglip alone

Those are statements about a fusion *recipe*, but every one was measured with a single image
encoder. W8 re-runs the sweep with each of the three W7 image encoders as the fusion target, on
the same W7 subset, to separate "RRF is the right method" from "RRF is the right method for
SigLIP".

Grid: 3 image encoders x 6 text representations x 3 fusion methods = 54 combinations, plus the
9 standalone systems. All from cached embeddings.

Outputs: results/w8_per_query.csv, results/w8_summary.csv, results/w8_significance.csv
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
fusion = import_module("09_fusion_experiment")

IMAGE_SYSTEMS = ["siglip_image", "omni_nano_image", "omni_small_image"]
TEXT_SYSTEMS = {
    "siglip_text": ("siglip", "title_emb"),
    "jina_text": ("jina", "title_emb"),
    "jina_small_text": ("jina_small", "title_emb"),
    "siglip_attr": ("siglip", "attr_emb"),
    "jina_attr": ("jina", "attr_emb"),
    "jina_small_attr": ("jina_small", "attr_emb"),
}
METHODS = ["mean_cosine", "rrf", "zscore_avg"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = ap.parse_args()
    config.ensure_dirs()

    test_set = pd.read_csv(config.DATA_DIR / "w7_subset.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)

    dense = {}
    for name in ("siglip", "jina", "jina_small"):
        d = np.load(config.EMB_DIR / f"{name}.npz", allow_pickle=True)
        dense[name] = {
            "q": {str(k): i for i, k in enumerate(d["queries"])},
            "e": {str(k): i for i, k in enumerate(d["ecodes"])},
            "arr": {k: d[k] for k in d.files if k.endswith("_emb")},
        }
    omni = {}
    for variant in ("nano", "small"):
        p = config.EMB_DIR / f"jina_omni_{variant}.npz"
        if not p.exists():
            print(f"missing {p} -- run 03c_embed_jina_omni.py --variant {variant}")
            return 1
        d = np.load(p, allow_pickle=True)
        omni[variant] = {
            "q": {str(k): v for k, v in zip(d["queries"], d["query_emb"])},
            "d": {str(k): v for k, v in zip(d["ecodes"], d["image_emb"])},
        }

    common_e = set(dense["siglip"]["e"]) & set(omni["nano"]["d"]) & set(omni["small"]["d"])
    common_q = set(dense["siglip"]["q"]) & set(omni["nano"]["q"]) & set(omni["small"]["q"])
    test_set = test_set[test_set["ecode"].isin(common_e) & test_set["search_term"].isin(common_q)]
    print(f"pools: {test_set['search_term'].nunique()} queries, {test_set['ecode'].nunique()} products")

    # drop any image encoder whose vectors collapsed (see W7); fusing with a constant is meaningless
    w7_meta_path = config.RESULTS_DIR / "w7_meta.json"
    active_images = list(IMAGE_SYSTEMS)
    if w7_meta_path.exists():
        active_images = json.loads(w7_meta_path.read_text()).get("image_systems", active_images)
    dropped = [s for s in IMAGE_SYSTEMS if s not in active_images]
    if dropped:
        print(f"excluding degenerate image encoders: {', '.join(dropped)}")

    rng = np.random.default_rng(config.RANDOM_SEED)
    label_sets = {"ltr": "relevance", "raw_ctr": "relevance_raw"}
    rows: list[dict] = []

    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        ecodes = pool["ecode"].tolist()
        if len(pool) < 5:
            continue

        sims: dict[str, np.ndarray] = {}
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            s = dense["siglip"]
            r = np.array([s["e"][e] for e in ecodes])
            qv = s["arr"]["query_emb"][s["q"][term]]
            sims["siglip_image"] = np.ascontiguousarray(s["arr"]["image_emb"][r]) @ qv
            for variant in ("nano", "small"):
                if f"omni_{variant}_image" not in active_images:
                    continue
                docs = np.vstack([omni[variant]["d"][e] for e in ecodes])
                sims[f"omni_{variant}_image"] = docs @ omni[variant]["q"][term]
            for tname, (model, field) in TEXT_SYSTEMS.items():
                m = dense[model]
                rr = np.array([m["e"][e] for e in ecodes])
                sims[tname] = np.ascontiguousarray(m["arr"][field][rr]) @ m["arr"]["query_emb"][m["q"][term]]

        rankings = {k: np.argsort(-v, kind="stable") for k, v in sims.items()}
        for img in active_images:
            for tname in TEXT_SYSTEMS:
                for method in METHODS:
                    fused = fusion.fuse(sims[img], sims[tname], method)
                    rankings[f"fusion[{method}]-{img}+{tname}"] = np.argsort(-fused, kind="stable")
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
                    "query_impressions": float(pool["total_impressions"].sum()),
                    **evaluate.score_ranking(rels, order),
                })

    per_query = pd.DataFrame(rows)
    per_query.to_csv(config.RESULTS_DIR / "w8_per_query.csv", index=False)

    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]
    macro = per_query.groupby(["label_set", "system"])[metric_cols].mean().reset_index()
    macro["weighting"] = "macro"

    def wmean(g: pd.DataFrame) -> pd.Series:
        w = g["query_impressions"].to_numpy()
        return pd.Series({c: float(np.average(g[c], weights=w)) for c in metric_cols})

    wtd = per_query.groupby(["label_set", "system"])[metric_cols + ["query_impressions"]].apply(wmean).reset_index()
    wtd["weighting"] = "impression"
    summary = pd.concat([macro, wtd], ignore_index=True)
    counts = per_query.groupby(["label_set", "system"])["search_term"].nunique().reset_index()
    summary = summary.merge(counts.rename(columns={"search_term": "n_queries"}), on=["label_set", "system"])
    summary.sort_values(["weighting", "label_set", "ndcg@10"], ascending=[True, True, False]).to_csv(
        config.RESULTS_DIR / "w8_summary.csv", index=False)

    # For each image encoder: is RRF still the best method, and attr-siglip still the best partner?
    sub = per_query[per_query["label_set"] == "ltr"]
    wide = sub.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
    vol = sub.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
    sig_rows = []
    for img in active_images:
        best = f"fusion[rrf]-{img}+siglip_attr"
        for method in METHODS:
            if method == "rrf":
                continue
            sig_rows.append({**fusion.bootstrap_contrast(wide, vol, best, f"fusion[{method}]-{img}+siglip_attr", rng, args.bootstrap),
                             "note": f"method: rrf vs {method} (image={img})"})
        for tname in TEXT_SYSTEMS:
            if tname == "siglip_attr":
                continue
            sig_rows.append({**fusion.bootstrap_contrast(wide, vol, best, f"fusion[rrf]-{img}+{tname}", rng, args.bootstrap),
                             "note": f"text: siglip_attr vs {tname} (image={img})"})
        for ref in (img, "siglip_attr"):
            sig_rows.append({**fusion.bootstrap_contrast(wide, vol, best, ref, rng, args.bootstrap),
                             "note": f"fusion vs {ref} alone (image={img})"})
    pd.DataFrame(sig_rows).to_csv(config.RESULTS_DIR / "w8_significance.csv", index=False)

    meta = {"n_queries": int(per_query["search_term"].nunique()),
            "n_products": int(test_set["ecode"].nunique()),
            "image_systems": active_images,
            "excluded_degenerate": dropped,
            "n_combinations": len(active_images) * len(TEXT_SYSTEMS) * len(METHODS),
            "bootstrap_samples": args.bootstrap}
    (config.RESULTS_DIR / "w8_meta.json").write_text(json.dumps(meta, indent=2))

    view = summary[(summary.label_set == "ltr") & (summary.weighting == "macro")].sort_values("ndcg@10", ascending=False)
    print(view[["system", "ndcg@10"]].head(15).to_string(index=False))
    print(f"\nsaved -> {config.RESULTS_DIR}/w8_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
