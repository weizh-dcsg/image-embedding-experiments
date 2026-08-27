#!/usr/bin/env python3
"""Step 4: rank each query's candidate pool with every embedding system and score relevance.

Systems compared (identical queries, identical candidate pools):
  siglip_image  SigLIP text query  vs SigLIP image of the product photo
  siglip_text   SigLIP text query  vs SigLIP text of the product title
  jina_text     Jina v5 query      vs Jina v5 document of the product title
  fusion        mean of z-scored siglip_image and jina_text similarities
  production    current on-site ranking (mean observed impression position)
  random        seeded shuffle, floor reference

Outputs: results/per_query_metrics.csv, results/summary.csv, results/significance.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def shuffle_pool(pool: pd.DataFrame, term: str) -> pd.DataFrame:
    """Deterministically permute a candidate pool before ranking.

    The judgement-list SQL emits rows ordered by relevance DESC, and np.argsort(kind="stable")
    preserves input order on ties. Without this, any system that returns tied or degenerate scores
    reproduces the label order exactly and scores near-perfect NDCG -- an artefact, not a result.
    Seeded per query so the permutation is reproducible across runs and identical for every system.
    """
    seed = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")
    order = np.random.default_rng(seed ^ config.RANDOM_SEED).permutation(len(pool))
    return pool.iloc[order]


def dcg(gains: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(rels_ranked: np.ndarray, k: int) -> float:
    gains = (2.0 ** rels_ranked[:k]) - 1.0
    ideal = (2.0 ** np.sort(rels_ranked)[::-1][:k]) - 1.0
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def mrr_at_k(rels_ranked: np.ndarray, k: int) -> float:
    hits = np.flatnonzero(rels_ranked[:k] > 0)
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def recall_at_k(rels_ranked: np.ndarray, k: int) -> float:
    total = int((rels_ranked > 0).sum())
    return float((rels_ranked[:k] > 0).sum()) / total if total else 0.0


def precision_at_k(rels_ranked: np.ndarray, k: int) -> float:
    return float((rels_ranked[:k] > 0).sum()) / k


def average_precision(rels_ranked: np.ndarray) -> float:
    positives = rels_ranked > 0
    total = int(positives.sum())
    if not total:
        return 0.0
    hits = np.cumsum(positives)
    precisions = hits / np.arange(1, len(rels_ranked) + 1)
    return float(np.sum(precisions * positives) / total)


def score_ranking(rels: np.ndarray, order: np.ndarray) -> dict[str, float]:
    ranked = rels[order]
    out: dict[str, float] = {}
    for k in config.K_VALUES:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, k)
        out[f"recall@{k}"] = recall_at_k(ranked, k)
        out[f"precision@{k}"] = precision_at_k(ranked, k)
        out[f"mrr@{k}"] = mrr_at_k(ranked, k)
    out["map"] = average_precision(ranked)
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else np.zeros_like(x)


def build_systems():
    siglip = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    jina = np.load(config.EMB_DIR / "jina.npz", allow_pickle=True)

    queries = [str(q) for q in siglip["queries"]]
    ecodes = [str(e) for e in siglip["ecodes"]]
    if [str(q) for q in jina["queries"]] != queries or [str(e) for e in jina["ecodes"]] != ecodes:
        raise SystemExit("siglip.npz and jina.npz are not aligned; rerun 03_embed.py")

    q_idx = {q: i for i, q in enumerate(queries)}
    e_idx = {e: i for i, e in enumerate(ecodes)}

    systems = {
        "siglip_image": {"q": siglip["query_emb"], "d": siglip["image_emb"]},
        "siglip_text": {"q": siglip["query_emb"], "d": siglip["title_emb"]},
        "jina_text": {"q": jina["query_emb"], "d": jina["title_emb"]},
    }
    if "image_crop_emb" in siglip.files:
        systems["siglip_image_crop"] = {"q": siglip["query_emb"], "d": siglip["image_crop_emb"]}
    if "image_naive_emb" in siglip.files:
        systems["siglip_image_naive"] = {"q": siglip["query_emb"], "d": siglip["image_naive_emb"]}
    if "attr_emb" in siglip.files:
        systems["siglip_attr"] = {"q": siglip["query_emb"], "d": siglip["attr_emb"]}
    if "attr_emb" in jina.files:
        systems["jina_attr"] = {"q": jina["query_emb"], "d": jina["attr_emb"]}

    small_path = config.EMB_DIR / "jina_small.npz"
    if small_path.exists():
        small = np.load(small_path, allow_pickle=True)
        if [str(e) for e in small["ecodes"]] == ecodes:
            systems["jina_small_text"] = {"q": small["query_emb"], "d": small["title_emb"]}
            if "attr_emb" in small.files:
                systems["jina_small_attr"] = {"q": small["query_emb"], "d": small["attr_emb"]}

    # W6 add-ons: three E5 text models plus the Jina v5 omni-nano text tower.
    extra_files = {
        "e5_base": ("e5_base_text", "e5_base_attr"),
        "e5_small_multi": ("e5_small_multi_text", "e5_small_multi_attr"),
        "e5_large_instruct": ("e5_large_instruct_text", "e5_large_instruct_attr"),
        "jina_omni_nano_text": ("omni_nano_text", "omni_nano_attr"),
    }
    for stem, (text_name, attr_name) in extra_files.items():
        path = config.EMB_DIR / f"{stem}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        if [str(e) for e in data["ecodes"]] != ecodes:
            continue
        systems[text_name] = {"q": data["query_emb"], "d": data["title_emb"]}
        if "attr_emb" in data.files:
            systems[attr_name] = {"q": data["query_emb"], "d": data["attr_emb"]}
    return systems, q_idx, e_idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")

    systems, q_idx, e_idx = build_systems()
    rng = np.random.default_rng(config.RANDOM_SEED)

    rows: list[dict] = []
    label_sets = {"ltr": "relevance", "raw_ctr": "relevance_raw"}

    for search_term, pool in test_set.groupby("search_term"):
        if search_term not in q_idx:
            continue

        pool = shuffle_pool(pool, search_term)
        cand_rows = np.array([e_idx[e] for e in pool["ecode"]])
        qi = q_idx[search_term]

        sims: dict[str, np.ndarray] = {}
        for name, emb in systems.items():
            docs = np.ascontiguousarray(emb["d"][cand_rows])
            # Accelerate/BLAS on Apple silicon raises spurious FP status flags here; the
            # result is verified finite below.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims[name] = docs @ emb["q"][qi]
            if not np.isfinite(sims[name]).all():
                raise SystemExit(f"non-finite similarities for {name} / {search_term}")
        sims["fusion"] = 0.5 * (zscore(sims["siglip_image"]) + zscore(sims["jina_text"]))

        rankings = {name: np.argsort(-s, kind="stable") for name, s in sims.items()}
        rankings["production"] = np.argsort(pool["mean_position"].to_numpy(), kind="stable")
        rankings["random"] = rng.permutation(len(pool))

        for label_set, column in label_sets.items():
            rels = pool[column].to_numpy(dtype=float)
            if rels.max() <= 0 or len(pool) < 5:
                continue
            for name, order in rankings.items():
                rows.append(
                    {
                        "label_set": label_set,
                        "search_term": search_term,
                        "query_tier": pool["query_tier"].iloc[0],
                        "system": name,
                        "pool_size": len(pool),
                        "n_positive": int((rels > 0).sum()),
                        "query_impressions": float(pool["total_impressions"].sum()),
                        **score_ranking(rels, order),
                    }
                )

    per_query = pd.DataFrame(rows)
    per_query.to_csv(config.RESULTS_DIR / "per_query_metrics.csv", index=False)

    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]

    # Macro treats every query equally; impression weighting answers "how well does this rank a
    # typical search impression", which is the business-relevant quantity. They can disagree.
    macro = per_query.groupby(["label_set", "system"])[metric_cols].mean().reset_index()
    macro["weighting"] = "macro"

    def weighted_mean(g: pd.DataFrame) -> pd.Series:
        w = g["query_impressions"].to_numpy()
        return pd.Series({c: float(np.average(g[c], weights=w)) for c in metric_cols})

    wtd = (
        per_query.groupby(["label_set", "system"])[metric_cols + ["query_impressions"]]
        .apply(weighted_mean)
        .reset_index()
    )
    wtd["weighting"] = "impression"

    summary = pd.concat([macro, wtd], ignore_index=True)
    counts = (
        per_query.groupby(["label_set", "system"])["search_term"]
        .nunique()
        .reset_index()
        .rename(columns={"search_term": "n_queries"})
    )
    summary = summary.merge(counts, on=["label_set", "system"])
    summary = summary.sort_values(["weighting", "label_set", "ndcg@10"], ascending=[True, True, False])
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)

    # Same macro / impression-weighted split as the aggregate summary above, computed within each
    # query_tier so a system that only wins on the traffic-weighted average because head queries
    # dominate it (or only wins on tail queries hidden inside the macro average) is visible.
    tier_macro = per_query.groupby(["label_set", "query_tier", "system"])[metric_cols].mean().reset_index()
    tier_macro["weighting"] = "macro"

    def tier_weighted_mean(g: pd.DataFrame) -> pd.Series:
        w = g["query_impressions"].to_numpy()
        return pd.Series({c: float(np.average(g[c], weights=w)) for c in metric_cols})

    tier_wtd = (
        per_query.groupby(["label_set", "query_tier", "system"])[metric_cols + ["query_impressions"]]
        .apply(tier_weighted_mean)
        .reset_index()
    )
    tier_wtd["weighting"] = "impression"

    by_tier = pd.concat([tier_macro, tier_wtd], ignore_index=True)
    tier_counts = (
        per_query.groupby(["label_set", "query_tier", "system"])["search_term"]
        .nunique()
        .reset_index()
        .rename(columns={"search_term": "n_queries"})
    )
    by_tier = by_tier.merge(tier_counts, on=["label_set", "query_tier", "system"])
    by_tier["query_tier"] = pd.Categorical(by_tier["query_tier"], categories=["head", "torso", "tail"], ordered=True)
    by_tier = by_tier.sort_values(
        ["weighting", "label_set", "query_tier", "ndcg@10"], ascending=[True, True, True, False]
    )
    by_tier.to_csv(config.RESULTS_DIR / "summary_by_tier.csv", index=False)

    # paired bootstrap over queries against four reference points:
    #   jina_text   -- the text-embedding system under test
    #   siglip_text -- isolates modality from model (same encoder, title instead of photo)
    #   production / random -- external reference points
    sig_rows = []
    for label_set in label_sets:
        subset = per_query[per_query["label_set"] == label_set]
        wide = subset.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
        vol = (
            subset.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
        )
        for baseline in (
            "jina_text",
            "siglip_text",
            "siglip_attr",
            "jina_attr",
            "jina_small_text",
            "jina_small_attr",
            "siglip_image",
            "siglip_image_naive",
            "siglip_image_crop",
            "production",
            "random",
        ):
            if baseline not in wide.columns:
                continue
            for system in wide.columns:
                if system == baseline:
                    continue
                diff = (wide[system] - wide[baseline]).to_numpy()
                idx = rng.integers(0, len(diff), size=(args.bootstrap, len(diff)))
                boot = diff[idx].mean(axis=1)
                boot_w = np.array(
                    [np.average(diff[i], weights=vol[i]) for i in idx]
                )
                w_delta = float(np.average(diff, weights=vol))
                sig_rows.append(
                    {
                        "label_set": label_set,
                        "system": system,
                        "baseline": baseline,
                        "metric": "ndcg@10",
                        "delta": float(diff.mean()),
                        "rel_delta_pct": float(100 * diff.mean() / max(wide[baseline].mean(), 1e-9)),
                        "ci_low": float(np.percentile(boot, 2.5)),
                        "ci_high": float(np.percentile(boot, 97.5)),
                        "p_value": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
                        "win_rate": float((diff > 0).mean()),
                        "wtd_delta": w_delta,
                        "wtd_ci_low": float(np.percentile(boot_w, 2.5)),
                        "wtd_ci_high": float(np.percentile(boot_w, 97.5)),
                        "wtd_p_value": float(2 * min((boot_w <= 0).mean(), (boot_w >= 0).mean())),
                    }
                )
    significance = pd.DataFrame(sig_rows)
    significance.to_csv(config.RESULTS_DIR / "significance.csv", index=False)

    meta = {
        "n_queries": int(per_query["search_term"].nunique()),
        "n_queries_by_tier": {
            tier: int(n)
            for tier, n in test_set.drop_duplicates("search_term")["query_tier"].value_counts().items()
        },
        "n_pairs": int(len(test_set)),
        "n_products": int(test_set["ecode"].nunique()),
        "mean_pool_size": float(per_query.groupby("search_term")["pool_size"].first().mean()),
        "siglip_model": config.SIGLIP_MODEL,
        "jina_model": config.JINA_MODEL,
        "bootstrap_samples": args.bootstrap,
        "label_sets": list(label_sets),
    }
    (config.RESULTS_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("by tier (macro, ndcg@10):")
    tier_view = by_tier[(by_tier["label_set"] == "ltr") & (by_tier["weighting"] == "macro")]
    tier_view = tier_view[["query_tier", "system", "n_queries", "ndcg@10"]]
    print(tier_view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    key = significance[significance["baseline"].isin(("jina_text", "siglip_text"))]
    cols = ["label_set", "system", "baseline", "delta", "p_value", "wtd_delta", "wtd_p_value"]
    print(key[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {config.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
