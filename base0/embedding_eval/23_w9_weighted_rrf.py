#!/usr/bin/env python3
"""Step 23 (W9): does weighting the RRF arms by their accuracy beat uniform RRF?

Uniform RRF gives BM25, E5 and the image tower an equal vote even though their standalone
accuracy differs substantially. This scores weighted RRF,

    score(d) = sum_i  w_i / (k + rank_i(d))

under several ways of setting w:

  uniform          w = (1, 1, 1) -- what W9 reports
  proportional     w_i proportional to arm i's standalone macro NDCG@10
  lift_over_random w_i proportional to (NDCG@10_i - NDCG@10_random). NDCG has a high floor on
                   these pools (~0.25 for a random shuffle), so raw NDCG understates how
                   different the arms actually are; this removes the floor first.
  tuned            grid search over the weight simplex

The tuned weights are searched on half the queries and reported on the other half. Tuning and
reporting on the same queries would guarantee an improvement and mean nothing.

Outputs: results/w9_weighted_rrf.csv
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

evaluate = import_module("04_evaluate")
experiment = import_module("21_w9_hybrid_experiment")

RRF_K = experiment.RRF_K
DEFAULT_ARMS = ["bm25", "e5", "siglip_image"]
RANDOM_NDCG10 = 0.253357  # seeded-shuffle floor from 04_evaluate on these pools
ARMS: list[str] = list(DEFAULT_ARMS)


def reciprocal_rank_vectors(signals: dict, test_set: pd.DataFrame) -> list[tuple]:
    """Per query: one reciprocal-rank contribution vector per arm, plus the relevance labels.

    Precomputed once so a weight sweep is a vector add rather than a re-ranking pass.
    """
    out = []
    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        ecodes = pool["ecode"].tolist()
        n = len(ecodes)
        rels = pool["relevance"].to_numpy(dtype=float)
        if rels.max() <= 0 or n < 5:
            continue
        contributions = []
        for arm in ARMS:
            vec = np.zeros(n)
            ranked = experiment.ranked_list(signals[arm], term, ecodes)
            vec[ranked] = 1.0 / (RRF_K + np.arange(1, len(ranked) + 1))
            contributions.append(vec)
        out.append((term, np.stack(contributions), rels, float(pool["total_impressions"].sum())))
    return out


def score_weights(data: list[tuple], weights: np.ndarray) -> np.ndarray:
    """NDCG@10 per query for one weight vector."""
    out = np.empty(len(data))
    for i, (_, contributions, rels, _) in enumerate(data):
        order = np.argsort(-(weights @ contributions), kind="stable")
        out[i] = evaluate.ndcg_at_k(rels[order], 10)
    return out


def simplex_grid(step: float = 0.1) -> list[np.ndarray]:
    grid = []
    ticks = int(round(1.0 / step))
    for a in range(ticks + 1):
        for b in range(ticks + 1 - a):
            c = ticks - a - b
            if a == 0 and b == 0 and c == 0:
                continue
            grid.append(np.array([a, b, c], dtype=float) * step)
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--arms", nargs=3, default=DEFAULT_ARMS, metavar="ARM")
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "w9_weighted_rrf.csv")
    args = parser.parse_args()

    global ARMS
    ARMS = list(args.arms)
    print(f"arms: {ARMS}")

    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)

    signals, _ = experiment.load_signals()
    missing = [a for a in ARMS if a not in signals]
    if missing:
        raise SystemExit(f"missing signals: {missing}")

    print("precomputing reciprocal-rank vectors...")
    data = reciprocal_rank_vectors(signals, test_set)
    print(f"  {len(data)} queries")

    summary = pd.read_csv(config.RESULTS_DIR / "w9_summary.csv")
    macro = summary[(summary.label_set == "ltr") & (summary.weighting == "macro")].set_index("system")
    standalone = np.array([macro.loc[a, "ndcg@10"] for a in ARMS])
    print("  standalone NDCG@10: " + ", ".join(f"{a}={v:.4f}" for a, v in zip(ARMS, standalone)))

    schemes = {
        "uniform": np.ones(3) / 3,
        "proportional": standalone / standalone.sum(),
        "lift_over_random": np.clip(standalone - RANDOM_NDCG10, 0, None)
        / np.clip(standalone - RANDOM_NDCG10, 0, None).sum(),
    }

    rng = np.random.default_rng(config.RANDOM_SEED)
    split = rng.permutation(len(data))
    tune_idx, hold_idx = split[: len(data) // 2], split[len(data) // 2 :]

    print(f"grid search on {len(tune_idx)} tuning queries...")
    best, best_score = None, -1.0
    for weights in simplex_grid(args.step):
        score = score_weights([data[i] for i in tune_idx], weights).mean()
        if score > best_score:
            best, best_score = weights, score
    schemes["tuned"] = best
    print(f"  best tuned weights {dict(zip(ARMS, best.round(2)))} -> {best_score:.4f} on tuning half")

    rows = []
    for name, weights in schemes.items():
        per_query = score_weights(data, weights)
        holdout = per_query[hold_idx]
        uniform_holdout = score_weights([data[i] for i in hold_idx], schemes["uniform"])
        diff = holdout - uniform_holdout
        boot = rng.integers(0, len(diff), size=(2000, len(diff)))
        deltas = diff[boot].mean(axis=1)
        rows.append(
            {
                "scheme": name,
                **{f"w_{a}": round(float(w), 3) for a, w in zip(ARMS, weights)},
                "ndcg@10_all": float(per_query.mean()),
                "ndcg@10_holdout": float(holdout.mean()),
                "delta_vs_uniform": float(diff.mean()),
                "ci_low": float(np.percentile(deltas, 2.5)),
                "ci_high": float(np.percentile(deltas, 97.5)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print()
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
