#!/usr/bin/env python3
"""Step 24 (W9): how should the three deployable arms be merged?

Fixed setting -- everything here already runs on the Elasticsearch cluster:

  bm25    Lucene BM25 over `catalog-1`
  e5      multilingual-e5-small, production vectors + deployed query encoder
  jina    Jina CLIP v2 image tower, query encoded by the deployed `jina-clip-v2-text`

The models are held constant; the only variable is the merge. Strategies compared:

  rrf                 reciprocal rank fusion, uniform weights, k swept
  rrf_weighted        RRF with per-arm weights (analytic and tuned)
  zscore              per-query z-scored raw scores, averaged (uniform and tuned)
  minmax              per-query min-max normalised scores, averaged
  staged              RRF(bm25, e5) first, then RRF that result with the image arm
  single arms         each arm alone, as reference points

Score-based strategies need a value for candidates an arm cannot score. E5 covers 56% of the
catalog, so this is not a corner case: missing candidates take that arm's per-query minimum,
which is the least-assuming floor. RRF has no such problem, and the comparison below shows how
much that matters.

Anything tuned is fitted on half the queries and reported on the held-out half.

Outputs: results/w9_merge_study.csv
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

ARMS = ["bm25", "e5", "jina_clip_v2_image"]
LABELS = {"bm25": "bm25", "e5": "e5-small", "jina_clip_v2_image": "jina-clip-v2-image"}
RANDOM_NDCG10 = 0.253357


def arm_scores(signal: dict, term: str, ecodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Raw score and availability mask for one arm over a candidate pool."""
    n = len(ecodes)
    scores = np.zeros(n)
    mask = np.zeros(n, dtype=bool)
    if signal["kind"] == "score":
        table = signal["table"].get(term)
        if table:
            for i, e in enumerate(ecodes):
                if e in table:
                    scores[i] = table[e]
                    mask[i] = True
        return scores, mask
    qv = signal["q"].get(term)
    if qv is None:
        return scores, mask
    idx = [i for i, e in enumerate(ecodes) if e in signal["d"]]
    if idx:
        docs = np.stack([signal["d"][ecodes[i]] for i in idx])
        scores[idx] = docs @ qv
        mask[idx] = True
    return scores, mask


def ranks_from(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reciprocal-rank contribution vector: 1/(k+rank) for scored items, 0 otherwise."""
    return scores, mask


def rr_vector(scores: np.ndarray, mask: np.ndarray, k: float) -> np.ndarray:
    vec = np.zeros(len(scores))
    idx = np.flatnonzero(mask)
    if not len(idx):
        return vec
    order = idx[np.argsort(-scores[idx], kind="stable")]
    vec[order] = 1.0 / (k + np.arange(1, len(order) + 1))
    return vec


def normalise(scores: np.ndarray, mask: np.ndarray, how: str) -> np.ndarray:
    """Per-query normalisation; unscored candidates take the arm's floor."""
    out = np.zeros(len(scores))
    idx = np.flatnonzero(mask)
    if not len(idx):
        return out
    vals = scores[idx]
    if how == "zscore":
        std = vals.std()
        z = (vals - vals.mean()) / std if std > 1e-9 else np.zeros_like(vals)
    else:
        lo, hi = vals.min(), vals.max()
        z = (vals - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(vals)
    out[:] = z.min()
    out[idx] = z
    return out


def build(signals: dict, test_set: pd.DataFrame) -> list[dict]:
    records = []
    for term, pool in test_set.groupby("search_term"):
        pool = evaluate.shuffle_pool(pool, term)
        ecodes = pool["ecode"].tolist()
        rels = pool["relevance"].to_numpy(dtype=float)
        if rels.max() <= 0 or len(ecodes) < 5:
            continue
        raw = {a: arm_scores(signals[a], term, ecodes) for a in ARMS}
        records.append(
            {
                "term": term,
                "rels": rels,
                "imp": float(pool["total_impressions"].sum()),
                "raw": raw,
                "rr": {k: np.stack([rr_vector(*raw[a], k) for a in ARMS]) for k in (10, 20, 60, 100)},
                "z": np.stack([normalise(*raw[a], "zscore") for a in ARMS]),
                "mm": np.stack([normalise(*raw[a], "minmax") for a in ARMS]),
            }
        )
    return records


def ndcg(record: dict, combined: np.ndarray) -> float:
    order = np.argsort(-combined, kind="stable")
    return evaluate.ndcg_at_k(record["rels"][order], 10)


def evaluate_strategy(records: list[dict], fn) -> np.ndarray:
    return np.array([ndcg(r, fn(r)) for r in records])


def simplex_grid(step: float) -> list[np.ndarray]:
    ticks = int(round(1 / step))
    return [
        np.array([a, b, ticks - a - b], dtype=float) * step
        for a in range(ticks + 1)
        for b in range(ticks + 1 - a)
        if (a, b) != (0, 0) or ticks > 0
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)

    signals, _ = experiment.load_signals()
    print("building per-query score matrices...")
    records = build(signals, test_set)
    print(f"  {len(records)} queries")

    rng = np.random.default_rng(config.RANDOM_SEED)
    split = rng.permutation(len(records))
    tune, hold = split[: len(records) // 2], split[len(records) // 2 :]
    tune_rec = [records[i] for i in tune]

    summary = pd.read_csv(config.RESULTS_DIR / "w9_summary.csv")
    macro = summary[(summary.label_set == "ltr") & (summary.weighting == "macro")].set_index("system")
    standalone = np.array([macro.loc[a, "ndcg@10"] for a in ARMS])
    lift = np.clip(standalone - RANDOM_NDCG10, 0, None)
    w_lift = lift / lift.sum()
    print("  standalone NDCG@10: " + ", ".join(f"{LABELS[a]}={v:.4f}" for a, v in zip(ARMS, standalone)))

    strategies: dict[str, tuple] = {}
    for i, arm in enumerate(ARMS):
        strategies[f"single: {LABELS[arm]}"] = ((lambda r, i=i: r["rr"][60][i]), None)
    for k in (10, 20, 60, 100):
        strategies[f"rrf uniform (k={k})"] = ((lambda r, k=k: r["rr"][k].sum(0)), None)
    strategies["rrf weighted: lift-over-random"] = ((lambda r: w_lift @ r["rr"][60]), w_lift)
    strategies["zscore average"] = ((lambda r: r["z"].mean(0)), None)
    strategies["minmax average"] = ((lambda r: r["mm"].mean(0)), None)

    def staged(r):
        text = r["rr"][60][:2].sum(0)
        vec = np.zeros(len(text))
        order = np.argsort(-text, kind="stable")
        vec[order] = 1.0 / (60 + np.arange(1, len(order) + 1))
        return vec + r["rr"][60][2]

    strategies["staged: RRF(bm25,e5) then image"] = (staged, None)

    # Tuned variants: weights fitted on the tuning half only.
    for name, key in [("rrf weighted: tuned", "rr"), ("zscore weighted: tuned", "z")]:
        best, best_score = None, -1.0
        for w in simplex_grid(args.step):
            if key == "rr":
                fn = lambda r, w=w: w @ r["rr"][60]
            else:
                fn = lambda r, w=w: w @ r["z"]
            s = evaluate_strategy(tune_rec, fn).mean()
            if s > best_score:
                best, best_score = w, s
        if key == "rr":
            strategies[name] = ((lambda r, w=best: w @ r["rr"][60]), best)
        else:
            strategies[name] = ((lambda r, w=best: w @ r["z"]), best)
        print(f"  {name}: {dict(zip([LABELS[a] for a in ARMS], best.round(2)))}")

    baseline_fn = strategies["rrf uniform (k=60)"][0]
    base_hold = evaluate_strategy([records[i] for i in hold], baseline_fn)

    rows = []
    for name, (fn, weights) in strategies.items():
        per_query = evaluate_strategy(records, fn)
        h = per_query[hold]
        diff = h - base_hold
        boot = rng.integers(0, len(diff), size=(2000, len(diff)))
        deltas = diff[boot].mean(axis=1)
        rows.append(
            {
                "strategy": name,
                "weights": "" if weights is None else "/".join(f"{w:.2f}" for w in weights),
                "ndcg@10_holdout": float(h.mean()),
                "delta_vs_rrf60": float(diff.mean()),
                "ci_low": float(np.percentile(deltas, 2.5)),
                "ci_high": float(np.percentile(deltas, 97.5)),
                "ndcg@10_all": float(per_query.mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("ndcg@10_holdout", ascending=False)
    out.to_csv(config.RESULTS_DIR / "w9_merge_study.csv", index=False)
    print()
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
