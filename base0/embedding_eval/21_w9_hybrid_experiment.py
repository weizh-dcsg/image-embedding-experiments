#!/usr/bin/env python3
"""Step 21 (W9): does a Jina CLIP v2 image tower add anything on top of the live hybrid?

Baseline is the retrieval stack that is already in production: Lucene BM25 over `catalog-1`
fused with multilingual-e5-small vectors by reciprocal rank fusion. The contrast adds the Jina
CLIP v2 image tower -- query text scored against the product photograph -- as a third RRF arm.

Systems (every one ranks the same candidate pool per query):
  bm25                  production BM25 score, `catalog-1`
  e5                    cosine, production multilingual-e5-small vectors
  jina_clip_v2_image    cosine, Jina CLIP v2 text query vs Jina CLIP v2 product image
  rrf_bm25_e5           BASELINE  -- 2-way RRF
  rrf_bm25_e5_jina      CONTRAST  -- 3-way RRF
  rrf_bm25_jina         ablation: does the image tower substitute for E5?
  rrf_e5_jina           ablation: how much of the baseline is carried by the lexical arm?
  production            incumbent ordering by mean observed impression position
  random                seeded shuffle, floor reference

Pools are NOT intersected on signal coverage. A retriever that cannot score a candidate simply
omits it from its ranked list and contributes nothing for it, which is how RRF behaves in
Elasticsearch and how the live stack behaves when a product is missing from the vector index.
Intersecting instead would quietly delete the baseline's real coverage gap and flatter it.

Ties are broken by the seeded per-query shuffle from 04_evaluate, so candidates that no retriever
scores land in a reproducible pseudo-random order rather than in judgement-list order.

Outputs: results/w9_per_query.csv, w9_summary.csv, w9_tier_summary.csv, w9_significance.csv
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

RRF_K = 60
BASELINE = "rrf_bm25_e5"
CONTRAST = "rrf_bm25_e5_jina"
HEADLINE_METRIC = "ndcg@10"
# A query is treated as "baseline at full strength" when E5 can score at least this share of its
# pool. Splitting on it separates a real gain from the image tower merely patching coverage holes.
E5_FULL_COVERAGE = 0.8

COMBINATIONS = {
    "rrf_bm25_e5": ("bm25", "e5"),
    "rrf_bm25_e5_jina": ("bm25", "e5", "jina_clip_v2_image"),
    "rrf_bm25_jina": ("bm25", "jina_clip_v2_image"),
    "rrf_e5_jina": ("e5", "jina_clip_v2_image"),
    # Coverage control: identical recipe, but the E5 arm is the locally computed
    # multilingual-e5-small, which covers every product. Isolates the effect of the production
    # vector index's partial coverage from the effect of adding an image tower.
    "rrf_bm25_e5local": ("bm25", "e5_local"),
    "rrf_bm25_e5local_jina": ("bm25", "e5_local", "jina_clip_v2_image"),
    "rrf_e5local_jina": ("e5_local", "jina_clip_v2_image"),
    # Encoder check: W1 scores siglip_image well above jina_clip_v2_image on this same test set,
    # so the recipe is repeated with SigLIP to make sure W9 is not recommending the weaker tower.
    "rrf_bm25_e5_siglip": ("bm25", "e5", "siglip_image"),
    "rrf_bm25_siglip": ("bm25", "siglip_image"),
    "rrf_e5local_siglip": ("e5_local", "siglip_image"),
}


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def rrf(pool_size: int, ranked_lists: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion over per-retriever ranked lists of pool positions.

    Each list holds only the pool positions that retriever could score, best first. A candidate
    absent from a list contributes nothing for that arm.
    """
    scores = np.zeros(pool_size, dtype=float)
    for ranked in ranked_lists:
        scores[ranked] += 1.0 / (k + np.arange(1, len(ranked) + 1))
    return scores


def load_signals() -> tuple[dict, dict]:
    """Return (signals, meta). Each signal exposes what it can score and how it ranks a pool."""
    bm25_path = config.DATA_DIR / "w9_bm25_catalog1.csv"
    e5_path = config.EMB_DIR / "e5_prod.npz"
    jina_path = config.EMB_DIR / "jina_clip_v2_full.npz"
    for path in (bm25_path, e5_path, jina_path):
        if not path.exists():
            raise SystemExit(f"missing {path} -- run 19_embed_jina_clip_v2_full.py and 20_fetch_es_baseline.py")

    bm25 = pd.read_csv(bm25_path)
    bm25["search_term"] = bm25["search_term"].astype(str)
    bm25["ecode"] = bm25["ecode"].astype(str)
    bm25_lookup = {
        term: dict(zip(grp["ecode"], grp["bm25_score"]))
        for term, grp in bm25.groupby("search_term")
    }

    e5 = np.load(e5_path, allow_pickle=True)
    e5_q = {str(q): v for q, v in zip(e5["queries"], l2_normalise(e5["query_emb"]))}
    e5_d = {str(e): v for e, v in zip(e5["ecodes"], l2_normalise(e5["doc_emb"]))}

    jina = np.load(jina_path, allow_pickle=True)
    jina_q = {str(q): v for q, v in zip(jina["queries"], l2_normalise(jina["query_emb"]))}
    jina_d = {str(e): v for e, v in zip(jina["ecodes"], l2_normalise(jina["image_emb"]))}

    signals = {
        "bm25": {"kind": "score", "table": bm25_lookup},
        "e5": {"kind": "vector", "q": e5_q, "d": e5_d},
        "jina_clip_v2_image": {"kind": "vector", "q": jina_q, "d": jina_d},
    }
    meta = {
        "bm25_rows": len(bm25),
        "bm25_queries": len(bm25_lookup),
        "e5_queries": len(e5_q),
        "e5_documents": len(e5_d),
        "jina_queries": len(jina_q),
        "jina_documents": len(jina_d),
    }

    local_path = config.EMB_DIR / "e5_small_multi.npz"
    if local_path.exists():
        local = np.load(local_path, allow_pickle=True)
        signals["e5_local"] = {
            "kind": "vector",
            "q": {str(q): v for q, v in zip(local["queries"], l2_normalise(local["query_emb"]))},
            "d": {str(e): v for e, v in zip(local["ecodes"], l2_normalise(local["title_emb"]))},
        }
        meta["e5_local_documents"] = len(signals["e5_local"]["d"])

    siglip_path = config.EMB_DIR / "siglip.npz"
    if siglip_path.exists():
        sig = np.load(siglip_path, allow_pickle=True)
        signals["siglip_image"] = {
            "kind": "vector",
            "q": {str(q): v for q, v in zip(sig["queries"], l2_normalise(sig["query_emb"]))},
            "d": {str(e): v for e, v in zip(sig["ecodes"], l2_normalise(sig["image_emb"]))},
        }
        meta["siglip_documents"] = len(signals["siglip_image"]["d"])
    return signals, meta


def ranked_list(signal: dict, term: str, ecodes: list[str]) -> np.ndarray:
    """Pool positions this signal can score, ordered best first."""
    if signal["kind"] == "score":
        table = signal["table"].get(term)
        if not table:
            return np.empty(0, dtype=int)
        positions = np.array([i for i, e in enumerate(ecodes) if e in table], dtype=int)
        if not len(positions):
            return positions
        scores = np.array([table[ecodes[i]] for i in positions], dtype=float)
        return positions[np.argsort(-scores, kind="stable")]

    query_vector = signal["q"].get(term)
    if query_vector is None:
        return np.empty(0, dtype=int)
    positions = np.array([i for i, e in enumerate(ecodes) if e in signal["d"]], dtype=int)
    if not len(positions):
        return positions
    docs = np.stack([signal["d"][ecodes[i]] for i in positions])
    scores = docs @ query_vector
    return positions[np.argsort(-scores, kind="stable")]


def bootstrap_contrast(
    wide: pd.DataFrame, vol: np.ndarray, system: str, baseline: str,
    rng: np.random.Generator, n_boot: int,
) -> dict:
    diff = (wide[system] - wide[baseline]).to_numpy()
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    boot_w = (diff[idx] * vol).sum(axis=1) / vol.sum()
    return {
        "system": system,
        "baseline": baseline,
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
        "win_rate": float((diff > 0).mean()),
        "loss_rate": float((diff < 0).mean()),
        "wtd_delta": float(np.average(diff, weights=vol)),
        "wtd_ci_low": float(np.percentile(boot_w, 2.5)),
        "wtd_ci_high": float(np.percentile(boot_w, 97.5)),
        "wtd_p_value": float(2 * min((boot_w <= 0).mean(), (boot_w >= 0).mean())),
        "n_queries": int(len(diff)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)

    signals, meta = load_signals()
    print(json.dumps(meta, indent=2))
    combinations = {
        name: arms for name, arms in COMBINATIONS.items() if all(a in signals for a in arms)
    }
    for name in set(COMBINATIONS) - set(combinations):
        print(f"  skipping {name}: required signal not available")

    rng = np.random.default_rng(config.RANDOM_SEED)
    rows: list[dict] = []
    coverage_rows: list[dict] = []
    label_sets = {"ltr": "relevance", "raw_ctr": "relevance_raw"}

    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        ecodes = pool["ecode"].tolist()
        n = len(ecodes)

        lists = {name: ranked_list(signal, term, ecodes) for name, signal in signals.items()}
        coverage_rows.append(
            {
                "search_term": term,
                "pool_size": n,
                **{f"{name}_scored": len(l) for name, l in lists.items()},
            }
        )

        scores: dict[str, np.ndarray] = {}
        for name, ranked in lists.items():
            # A single arm is itself an RRF list of one, so unscored candidates sink instead of
            # inheriting whatever order the pool happened to arrive in.
            scores[name] = rrf(n, [ranked])
        for combo, arms in combinations.items():
            scores[combo] = rrf(n, [lists[a] for a in arms])

        rankings = {name: np.argsort(-s, kind="stable") for name, s in scores.items()}
        rankings["production"] = np.argsort(pool["mean_position"].to_numpy(), kind="stable")
        rankings["random"] = rng.permutation(n)

        for label_set, column in label_sets.items():
            rels = pool[column].to_numpy(dtype=float)
            if rels.max() <= 0 or n < 5:
                continue
            for name, order in rankings.items():
                rows.append(
                    {
                        "label_set": label_set,
                        "search_term": term,
                        "query_tier": pool["query_tier"].iloc[0],
                        "system": name,
                        "pool_size": n,
                        "n_positive": int((rels > 0).sum()),
                        "query_impressions": float(pool["total_impressions"].sum()),
                        **evaluate.score_ranking(rels, order),
                    }
                )

    per_query = pd.DataFrame(rows)
    per_query.to_csv(config.RESULTS_DIR / "w9_per_query.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage["e5_share"] = coverage["e5_scored"] / coverage["pool_size"]
    coverage.to_csv(config.RESULTS_DIR / "w9_pool_coverage.csv", index=False)
    well_covered = set(coverage.loc[coverage["e5_share"] >= E5_FULL_COVERAGE, "search_term"])
    # Queries BM25 cannot match at all: plurals, misspellings, brand nicknames. The clearest test
    # of whether a non-lexical arm rescues the cases the lexical arm structurally cannot.
    bm25_blind = set(coverage.loc[coverage["bm25_scored"] == 0, "search_term"])

    total_candidates = int(coverage["pool_size"].sum())
    meta.update(
        {
            "rrf_k": RRF_K,
            "headline_metric": HEADLINE_METRIC,
            "baseline": BASELINE,
            "contrast": CONTRAST,
            "k_values": list(config.K_VALUES),
            "bootstrap_samples": args.bootstrap,
            "random_seed": config.RANDOM_SEED,
            "n_queries": int(coverage["search_term"].nunique()),
            "n_candidates": total_candidates,
            "candidate_coverage_pct": {
                name: round(100 * int(coverage[f"{name}_scored"].sum()) / total_candidates, 2)
                for name in signals
            },
            "e5_full_coverage_threshold": E5_FULL_COVERAGE,
            "n_queries_e5_covered": len(well_covered),
            "n_queries_e5_sparse": int(coverage["search_term"].nunique()) - len(well_covered),
            "n_queries_bm25_blind": len(bm25_blind),
        }
    )
    (config.RESULTS_DIR / "w9_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]

    def summarise(keys: list[str]) -> pd.DataFrame:
        macro = per_query.groupby(keys)[metric_cols].mean().reset_index()
        macro["weighting"] = "macro"

        def weighted(g: pd.DataFrame) -> pd.Series:
            w = g["query_impressions"].to_numpy()
            return pd.Series({c: float(np.average(g[c], weights=w)) for c in metric_cols})

        wtd = per_query.groupby(keys)[metric_cols + ["query_impressions"]].apply(weighted).reset_index()
        wtd["weighting"] = "impression"
        out = pd.concat([macro, wtd], ignore_index=True)
        counts = (
            per_query.groupby(keys)["search_term"].nunique().reset_index()
            .rename(columns={"search_term": "n_queries"})
        )
        return out.merge(counts, on=keys)

    summary = summarise(["label_set", "system"]).sort_values(
        ["weighting", "label_set", HEADLINE_METRIC], ascending=[True, True, False]
    )
    summary.to_csv(config.RESULTS_DIR / "w9_summary.csv", index=False)
    tier_summary = summarise(["label_set", "query_tier", "system"])
    tier_summary.to_csv(config.RESULTS_DIR / "w9_tier_summary.csv", index=False)

    # Significance: the headline is contrast vs baseline; the rest place that number in context.
    contrasts = [
        (CONTRAST, BASELINE, "headline: 3-way RRF vs production hybrid baseline"),
        (CONTRAST, "bm25", "contrast vs BM25 alone"),
        (CONTRAST, "e5", "contrast vs E5 alone"),
        (CONTRAST, "jina_clip_v2_image", "contrast vs image tower alone"),
        (BASELINE, "bm25", "baseline vs BM25 alone"),
        (BASELINE, "e5", "baseline vs E5 alone"),
        (BASELINE, "production", "baseline vs incumbent ordering"),
        (CONTRAST, "production", "contrast vs incumbent ordering"),
        ("rrf_bm25_jina", BASELINE, "ablation: image tower substituted for E5"),
        ("rrf_e5_jina", BASELINE, "ablation: image tower substituted for BM25"),
        ("rrf_bm25_e5local_jina", "rrf_bm25_e5local", "coverage control: same contrast, fully covered E5 arm"),
        ("rrf_bm25_e5local", BASELINE, "coverage control: what full E5 coverage alone is worth"),
        ("rrf_e5local_jina", BASELINE, "no lexical arm: fully covered E5 + image vs the hybrid"),
        ("rrf_e5local_jina", "rrf_e5_jina", "no lexical arm: effect of E5 coverage alone"),
        ("siglip_image", "jina_clip_v2_image", "encoder: SigLIP vs Jina CLIP v2 image tower, alone"),
        ("rrf_bm25_e5_siglip", CONTRAST, "encoder: same 3-way recipe with SigLIP instead"),
        ("rrf_e5local_siglip", "rrf_e5local_jina", "encoder: best recipe with SigLIP instead"),
        ("rrf_bm25_e5_siglip", BASELINE, "SigLIP 3-way vs production hybrid baseline"),
    ]

    sig_rows = []
    subset = per_query[per_query["label_set"] == "ltr"]

    scopes: list[tuple[str, pd.DataFrame]] = [("all", subset)]
    scopes += [(tier, subset[subset["query_tier"] == tier]) for tier in sorted(subset["query_tier"].unique())]
    scopes += [
        ("e5_covered", subset[subset["search_term"].isin(well_covered)]),
        ("e5_sparse", subset[~subset["search_term"].isin(well_covered)]),
        ("bm25_blind", subset[subset["search_term"].isin(bm25_blind)]),
    ]

    for scope, scoped in scopes:
        if scoped.empty:
            continue
        wide = scoped.pivot(index="search_term", columns="system", values=HEADLINE_METRIC).dropna()
        if wide.empty:
            continue
        vol = scoped.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
        boot_rng = np.random.default_rng(config.RANDOM_SEED)
        for system, baseline, note in contrasts:
            if system not in wide.columns or baseline not in wide.columns:
                continue
            result = bootstrap_contrast(wide, vol, system, baseline, boot_rng, args.bootstrap)
            result["query_tier"] = scope
            result["metric"] = HEADLINE_METRIC
            result["note"] = note
            sig_rows.append(result)

    significance = pd.DataFrame(sig_rows)
    significance.to_csv(config.RESULTS_DIR / "w9_significance.csv", index=False)

    headline = significance[
        (significance["query_tier"] == "all")
        & (significance["system"] == CONTRAST)
        & (significance["baseline"] == BASELINE)
    ]
    print("\n== headline")
    print(headline.to_string(index=False))
    print("\n== macro summary (ltr)")
    view = summary[(summary["label_set"] == "ltr") & (summary["weighting"] == "macro")]
    print(view[["system", "ndcg@10", "ndcg@48", "recall@48", "mrr@10", "map", "n_queries"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
