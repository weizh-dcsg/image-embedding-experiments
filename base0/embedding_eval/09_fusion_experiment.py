#!/usr/bin/env python3
"""Step 9: fusion-method x text-representation sweep, across TWO target image modalities.

Answers: when fusing an image-tower similarity with a text-based similarity, which text
representation gives the largest fusion gain, does the fusion *method* matter as much as the
text representation choice, and does any of that change with the *image encoder* being fused?

Target modalities (2):
  siglip_image      SigLIP text tower query   vs SigLIP image tower over the product photo
  omni_nano_image   Jina v5 omni-nano query   vs Jina v5 omni-nano image tower over the photo

Text representations (14): the query encoder is always paired with the matching document tower --
  siglip_text / siglip_attr             SigLIP text tower, title / Big-4 attribute string
  jina_text / jina_attr                 Jina v5 nano, title / attribute string
  jina_small_text / jina_small_attr     Jina v5 small, title / attribute string
  e5_base_text / e5_base_attr           E5 base, title / attribute string
  e5_small_multi_text / _attr           Multilingual E5 small, title / attribute string
  e5_large_instruct_text / _attr        Multilingual E5 large-instruct, title / attribute string
  omni_nano_text / omni_nano_attr       Jina v5 omni-nano text tower, title / attribute string

Fusion methods (3), each combining a target-image similarity with one text similarity, per query:
  mean_cosine   0.5 * (cos_image + cos_text)                          -- raw, unnormalised average
  rrf           1/(60+rank_image) + 1/(60+rank_text)                  -- reciprocal rank fusion, k=60
  zscore_avg    0.5 * (zscore(cos_image) + zscore(cos_text))          -- current production "fusion"

2 target images x 14 text representations x 3 methods = 84 fusion combinations, all computed fresh
here (cheap: cosine similarity + ranking over cached embeddings). Every system in this report --
standalone systems, every fusion combo, production, random -- ranks the same, common candidate
pool per query: the intersection of ecodes covered by every embedding source in use, so adding a
second image encoder (which may have a few fewer successfully-encoded photos than SigLIP) can't
silently change what any other system is being scored against.

Outputs:
  results/fusion_experiment_per_query.csv
  results/fusion_experiment_summary.csv   -- macro + impression weighting, full K_VALUES depth
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

evaluate = import_module("04_evaluate")

RRF_K = 60

TARGET_IMAGES = {
    "siglip_image": ("siglip", "image_emb"),
    "omni_nano_image": ("jina_omni_nano_image", "image_emb"),
}
TEXT_CANDIDATES = {
    "siglip_text": ("siglip", "title_emb"),
    "jina_text": ("jina", "title_emb"),
    "jina_small_text": ("jina_small", "title_emb"),
    "siglip_attr": ("siglip", "attr_emb"),
    "jina_attr": ("jina", "attr_emb"),
    "jina_small_attr": ("jina_small", "attr_emb"),
    # W6 add-ons: three E5 text models plus the Jina v5 omni-nano text tower.
    "e5_base_text": ("e5_base", "title_emb"),
    "e5_base_attr": ("e5_base", "attr_emb"),
    "e5_small_multi_text": ("e5_small_multi", "title_emb"),
    "e5_small_multi_attr": ("e5_small_multi", "attr_emb"),
    "e5_large_instruct_text": ("e5_large_instruct", "title_emb"),
    "e5_large_instruct_attr": ("e5_large_instruct", "attr_emb"),
    "omni_nano_text": ("jina_omni_nano_text", "title_emb"),
    "omni_nano_attr": ("jina_omni_nano_text", "attr_emb"),
}
METHODS = ["mean_cosine", "rrf", "zscore_avg"]


def load_embeddings() -> dict[str, dict[str, np.ndarray]]:
    """Load every embedding file referenced by TARGET_IMAGES/TEXT_CANDIDATES that actually exists.

    Lets the sweep run incrementally: a candidate whose npz hasn't been generated yet is dropped
    further down rather than raising, so add-on models can be wired in one at a time.
    """
    needed = {model for model, _ in TARGET_IMAGES.values()} | {model for model, _ in TEXT_CANDIDATES.values()}
    embs = {}
    for name in needed:
        path = config.EMB_DIR / f"{name}.npz"
        if not path.exists():
            print(f"  skipping {name}: {path} not found")
            continue
        data = np.load(path, allow_pickle=True)
        embs[name] = {k: data[k] for k in data.files}
    return embs


def ranks_desc(scores: np.ndarray) -> np.ndarray:
    """1-indexed rank per element, 1 = highest score."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def fuse(cos_image: np.ndarray, cos_text: np.ndarray, method: str) -> np.ndarray:
    if method == "mean_cosine":
        return 0.5 * (cos_image + cos_text)
    if method == "rrf":
        return 1.0 / (RRF_K + ranks_desc(cos_image)) + 1.0 / (RRF_K + ranks_desc(cos_text))
    if method == "zscore_avg":
        return 0.5 * (evaluate.zscore(cos_image) + evaluate.zscore(cos_text))
    raise ValueError(method)


def bootstrap_contrast(
    wide: pd.DataFrame, vol: np.ndarray, system: str, baseline: str, rng: np.random.Generator, n_boot: int
) -> dict:
    diff = (wide[system] - wide[baseline]).to_numpy()
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    boot_w = np.array([np.average(diff[i], weights=vol[i]) for i in idx])
    return {
        "system": system,
        "baseline": baseline,
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_value": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
        "win_rate": float((diff > 0).mean()),
        "wtd_delta": float(np.average(diff, weights=vol)),
        "wtd_ci_low": float(np.percentile(boot_w, 2.5)),
        "wtd_ci_high": float(np.percentile(boot_w, 97.5)),
        "wtd_p_value": float(2 * min((boot_w <= 0).mean(), (boot_w >= 0).mean())),
    }


def parse_fusion_system(system: str) -> tuple[str, str, str]:
    """'fusion[method]-{image}+{text}' -> (method, image, text)."""
    method, rest = system[len("fusion["):].split("]-", 1)
    image_name, text_name = rest.split("+", 1)
    return method, image_name, text_name


def compute_significance(per_query: pd.DataFrame, summary: pd.DataFrame, image_names: list[str]) -> pd.DataFrame:
    """Paired bootstrap over queries (NDCG@10), built around whichever fusion combo actually wins.

    Contrasts: best fusion vs its own image alone, vs its own text alone, vs the same (image, text)
    under the other two methods, vs the same (method, image) applied to every other text
    representation, and vs the same (method, text) applied to the other target image (does the
    image encoder choice matter, holding the rest of the recipe fixed).
    """
    label_set = "ltr"
    macro = summary[(summary["label_set"] == label_set) & (summary["weighting"] == "macro")]
    fusion_rows = macro[macro["system"].str.startswith("fusion[")]
    best_system = fusion_rows.loc[fusion_rows["ndcg@10"].idxmax(), "system"]
    best_method, best_image, best_text = parse_fusion_system(best_system)
    print(f"best fusion combo (macro ndcg@10): {best_system}")

    contrasts = [
        (best_system, best_image, "headline: best fusion vs its image alone"),
    ]
    if best_text in macro["system"].values:
        contrasts.append((best_system, best_text, "headline: best fusion vs its text representation alone"))
    for method in METHODS:
        if method != best_method:
            alt = f"fusion[{method}]-{best_image}+{best_text}"
            contrasts.append((best_system, alt, f"method: {best_method} vs {method}, image={best_image}, text={best_text}"))
    for text_name in TEXT_CANDIDATES:
        if text_name != best_text:
            alt = f"fusion[{best_method}]-{best_image}+{text_name}"
            contrasts.append((best_system, alt, f"text: {best_text} vs {text_name}, image={best_image}, method={best_method}"))
    for image_name in image_names:
        if image_name != best_image:
            alt = f"fusion[{best_method}]-{image_name}+{best_text}"
            contrasts.append((best_system, alt, f"image: {best_image} vs {image_name}, text={best_text}, method={best_method}"))

    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    subset = per_query[per_query["label_set"] == label_set]
    wide = subset.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
    vol = subset.groupby("search_term")["query_impressions"].first().reindex(wide.index).to_numpy()
    for system, baseline, note in contrasts:
        if system not in wide.columns or baseline not in wide.columns:
            print(f"  skipping contrast (missing system): {system} vs {baseline}")
            continue
        result = bootstrap_contrast(wide, vol, system, baseline, rng, config.BOOTSTRAP_SAMPLES)
        result["note"] = note
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> int:
    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)
    tier_map = test_set.drop_duplicates("search_term").set_index("search_term")["query_tier"].to_dict()

    embs = load_embeddings()

    # Drop any target image / text candidate whose backing embedding file wasn't loaded, or the
    # target images whose npz doesn't have image_emb, etc. -- lets the sweep run incrementally.
    target_images = {
        name: (model, field)
        for name, (model, field) in TARGET_IMAGES.items()
        if model in embs and field in embs[model]
    }
    if not target_images:
        raise SystemExit("no target image embeddings available -- run 03_embed.py at minimum")
    text_candidates = {
        name: (model, doc_key)
        for name, (model, doc_key) in TEXT_CANDIDATES.items()
        if model in embs and doc_key in embs[model]
    }
    skipped_images = sorted(set(TARGET_IMAGES) - set(target_images))
    skipped_text = sorted(set(TEXT_CANDIDATES) - set(text_candidates))
    if skipped_images:
        print(f"  skipping target images (embeddings not available): {skipped_images}")
    if skipped_text:
        print(f"  skipping text candidates (embeddings not available): {skipped_text}")

    # Per-system (query -> row) / (ecode -> row) lookups -- systems' npz files are not guaranteed
    # to share row order or even full ecode coverage (e.g. a handful of image encodes can fail),
    # so every lookup goes through an explicit dict rather than assuming aligned positions.
    models_in_use = {model for model, _ in target_images.values()} | {model for model, _ in text_candidates.values()}
    q_idx = {model: {str(q): i for i, q in enumerate(embs[model]["queries"])} for model in models_in_use}
    e_idx = {model: {str(e): i for i, e in enumerate(embs[model]["ecodes"])} for model in models_in_use}

    common_ecodes = set.intersection(*(set(e_idx[model]) for model in models_in_use))
    common_queries = set.intersection(*(set(q_idx[model]) for model in models_in_use))
    print(
        f"common coverage across all {len(models_in_use)} embedding sources in use: "
        f"{len(common_queries)} queries, {len(common_ecodes)} products"
    )

    combos = [(image_name, text_name, method) for image_name in target_images for text_name in text_candidates for method in METHODS]
    print(f"fusion combos to compute: {len(combos)}")

    label_sets = {"ltr": "relevance", "raw_ctr": "relevance_raw"}
    rows: list[dict] = []
    rng = np.random.default_rng(config.RANDOM_SEED)

    for search_term, pool in test_set.groupby("search_term"):
        if search_term not in common_queries:
            continue
        pool = evaluate.shuffle_pool(pool, search_term)
        pool = pool[pool["ecode"].isin(common_ecodes)]
        if len(pool) < 5:
            continue
        ecodes = pool["ecode"].tolist()

        cos_image_cache: dict[str, np.ndarray] = {}
        for image_name, (model_name, field) in target_images.items():
            e = embs[model_name]
            rows_idx = np.array([e_idx[model_name][ec] for ec in ecodes])
            qi = q_idx[model_name][search_term]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                cos_image_cache[image_name] = np.ascontiguousarray(e[field][rows_idx]) @ e["query_emb"][qi]

        cos_text_cache: dict[str, np.ndarray] = {}
        for text_name, (model_name, doc_key) in text_candidates.items():
            e = embs[model_name]
            rows_idx = np.array([e_idx[model_name][ec] for ec in ecodes])
            qi = q_idx[model_name][search_term]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                cos_text_cache[text_name] = np.ascontiguousarray(e[doc_key][rows_idx]) @ e["query_emb"][qi]

        rankings = {name: np.argsort(-s, kind="stable") for name, s in {**cos_image_cache, **cos_text_cache}.items()}
        rankings["production"] = np.argsort(pool["mean_position"].to_numpy(), kind="stable")
        rankings["random"] = rng.permutation(len(pool))
        for image_name, text_name, method in combos:
            score = fuse(cos_image_cache[image_name], cos_text_cache[text_name], method)
            rankings[f"fusion[{method}]-{image_name}+{text_name}"] = np.argsort(-score, kind="stable")

        for label_set, column in label_sets.items():
            rels = pool[column].to_numpy(dtype=float)
            if rels.max() <= 0:
                continue
            for name, order in rankings.items():
                rows.append(
                    {
                        "label_set": label_set,
                        "search_term": search_term,
                        "query_tier": tier_map.get(search_term, ""),
                        "system": name,
                        "pool_size": len(pool),
                        "n_positive": int((rels > 0).sum()),
                        "query_impressions": float(pool["total_impressions"].sum()),
                        **evaluate.score_ranking(rels, order),
                    }
                )

    per_query = pd.DataFrame(rows)
    per_query.to_csv(config.RESULTS_DIR / "fusion_experiment_per_query.csv", index=False)

    metric_cols = [c for c in per_query.columns if "@" in c or c == "map"]
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
    counts = per_query.groupby(["label_set", "system"])["search_term"].nunique().reset_index()
    counts = counts.rename(columns={"search_term": "n_queries"})
    summary = summary.merge(counts, on=["label_set", "system"])
    summary = summary.sort_values(["weighting", "label_set", "ndcg@10"], ascending=[True, True, False])
    summary.to_csv(config.RESULTS_DIR / "fusion_experiment_summary.csv", index=False)

    significance = compute_significance(per_query, summary, list(target_images))
    significance.to_csv(config.RESULTS_DIR / "fusion_experiment_significance.csv", index=False)

    print(f"queries scored: {per_query['search_term'].nunique()}")
    view = summary[(summary["label_set"] == "ltr") & (summary["weighting"] == "macro")]
    print(view[["system", "n_queries", "ndcg@10", "map"]].sort_values("ndcg@10", ascending=False).to_string(index=False))
    print()
    print(significance[["system", "baseline", "note", "delta", "p_value", "wtd_delta", "wtd_p_value"]].to_string(index=False))
    print(f"\nsaved -> {config.RESULTS_DIR / 'fusion_experiment_per_query.csv'}")
    print(f"saved -> {config.RESULTS_DIR / 'fusion_experiment_summary.csv'}")
    print(f"saved -> {config.RESULTS_DIR / 'fusion_experiment_significance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
