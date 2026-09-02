#!/usr/bin/env python3
"""Generate static/examples.json: torso and tail queries with their measured effect in the demo.

The offline W9 deltas measure pool re-ranking, but this demo retrieves from the whole catalog,
and the two disagree (`sabrina 3` gains +0.75 re-ranking yet retrieves swimwear). So each query
is run through the demo's own pipeline and scored against the judgement list: how many of the
top 10 are judged relevant, baseline versus image-enhanced.

Queries are the highest-traffic ones per tier, not the ones that happen to win, so the dropdowns
show regressions alongside wins.

Usage:
  python build_examples.py --per-tier 80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Colour, pattern, silhouette and material words -- the vocabulary a photograph can answer and a
# product title often cannot.
VISUAL = re.compile(
    r"\b(black|white|pink|red|blue|green|purple|grey|gray|orange|yellow|navy|beige|tan|brown"
    r"|gold|silver|teal|maroon|camo|neon|cream|olive|burgundy|pastel"
    r"|strip(?:e|ed|es)|floral|plaid|checker(?:ed)?|polka|tie ?dye|graphic|printed|patterned"
    r"|high ?tops?|low ?tops?|oversized|cropped|slim|baggy|puffer|quilted|ribbed|hooded"
    r"|sleeveless|long sleeve|short sleeve|zip|pullover|crew|v[- ]?neck|bootcut|flare|wide leg"
    r"|leather|mesh|fleece|denim|suede|knit|sherpa|satin|velvet|corduroy)\b"
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# app.py parses argv at import time, so stash our own arguments before clearing it.
ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]
import app  # noqa: E402


def rank_lists(query: str, tower: dict):
    bm25 = app.bm25_ranked(query, app.RANK_WINDOW)
    e5v = app.embed_query("e5", app.E5_MODEL, query)
    imv = app.embed_query("jina", tower["es_model"], query)
    e5l = app.vector_ranked(e5v, app.STATE["e5_ecodes"], app.STATE["e5_docs"], app.RANK_WINDOW)
    iml = app.vector_ranked(imv, tower["ecodes"], tower["docs"], app.RANK_WINDOW)

    baseline_scores = app.rrf([list(bm25), e5l])
    baseline = sorted(baseline_scores, key=lambda e: -baseline_scores[e])[:10]

    cands = list(dict.fromkeys([*bm25, *e5l, *iml]))
    zb = app.zscore_over(cands, bm25)
    ze = app.zscore_over(cands, app.vector_scores(e5v, app.STATE["e5_row"], app.STATE["e5_docs"], cands))
    zi = app.zscore_over(cands, app.vector_scores(imv, tower["row"], tower["docs"], cands))
    wb, we, wi = tower["weights"]
    fused = {c: wb * zb[c] + we * ze[c] + wi * zi[c] for c in cands}
    enhanced = sorted(cands, key=lambda e: -fused[e])[:10]
    return baseline, enhanced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tier", type=int, default=80)
    parser.add_argument("--key-file", type=Path, default=ROOT.parents[1] / "elastic-key.txt")
    args = parser.parse_args(ARGV)

    app.load_state(args.key_file, "elastic")
    tower = app.STATE["towers"]["jina"]

    ts = pd.read_csv(
        app.config.DATA_DIR / "test_set_encoded.csv",
        usecols=["search_term", "ecode", "relevance", "query_tier", "total_impressions"],
    )
    ts["ecode"] = ts.ecode.astype(str)
    ts["search_term"] = ts.search_term.astype(str)
    judged = {t: dict(zip(g.ecode, g.relevance)) for t, g in ts.groupby("search_term")}
    info = ts.groupby("search_term").agg(
        tier=("query_tier", "first"),
        imp=("total_impressions", "sum"),
        pool=("ecode", "size"),
        pos=("relevance", lambda s: int((s > 0).sum())),
    )
    info = info[(info.pool >= 20) & (info.pos >= 5)]

    out: dict[str, list] = {}
    for tier in ("torso", "tail"):
        pool = info[info.tier == tier]
        top = pool.nlargest(args.per_tier, "imp").index.tolist()
        # Long / visually descriptive queries: where an image tower should have the most to say.
        visual = pool[
            pool.index.map(lambda s: len(str(s).split()) >= 3 or bool(VISUAL.search(str(s).lower())))
        ]
        visual = visual.nlargest(args.per_tier, "imp").index.tolist()

        for key, terms in ((tier, top), (f"{tier}_visual", visual)):
            rows = []
            for i, q in enumerate(terms, 1):
                try:
                    base, enh = rank_lists(q, tower)
                except Exception as exc:
                    print(f"  skip {q!r}: {type(exc).__name__}")
                    continue
                rel = judged.get(q, {})
                rows.append(
                    {
                        "q": q,
                        "base": sum(1 for e in base if rel.get(e, 0) > 0),
                        "enh": sum(1 for e in enh if rel.get(e, 0) > 0),
                    }
                )
                print(f"  {key} {i}/{len(terms)}", end="\r", flush=True)
            rows.sort(key=lambda r: (r["enh"] - r["base"]), reverse=True)
            out[key] = rows
            print(f"\n  {key}: {len(rows)} queries, "
                  f"{sum(1 for r in rows if r['enh'] > r['base'])} improve, "
                  f"{sum(1 for r in rows if r['enh'] < r['base'])} regress")

    path = ROOT / "static" / "examples.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
