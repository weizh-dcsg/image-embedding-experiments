#!/usr/bin/env python3
"""Step 1: build the judgement list used as the embedding test set.

Scoring follows ds-ecm-search-ranking-ltr conventions (see sql/judgement_list.sql):
inverse-propensity + time-decay weighting, then relevance 0-4 from smoothed
weighted-CTR quantile bins.

Outputs:
  data/test_set.csv  -- one row per (search_term, ecode) with LTR relevance 0-4
  data/products.csv  -- unique active products (title + image URL) in the test set
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base0"))

import config  # noqa: E402
from dbx_sql import DatabricksSQL  # noqa: E402


def resolve_dates(end_date: str, lookback_days: int, days_before_today: int) -> tuple[str, str]:
    end = date.fromisoformat(end_date) if end_date else date.today() - timedelta(days=days_before_today)
    start = end - timedelta(days=lookback_days - 1)
    return start.isoformat(), end.isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default=config.END_DATE)
    parser.add_argument("--lookback-days", type=int, default=config.LOOKBACK_DAYS)
    parser.add_argument("--days-before-today", type=int, default=config.DAYS_BEFORE_TODAY)
    parser.add_argument("--max-group-size", type=int, default=config.MAX_GROUP_SIZE)
    args = parser.parse_args()

    config.ensure_dirs()
    start_date, end_date = resolve_dates(args.end_date, args.lookback_days, args.days_before_today)

    sql = (config.SQL_DIR / "judgement_list.sql").read_text().format(
        start_date=start_date,
        end_date=end_date,
        banner=config.BANNER,
        channel=config.CHANNEL,
        term_pool_head=config.TERM_POOL_HEAD,
        tail_pool_size=config.TAIL_POOL_SIZE,
        tail_min_impressions=config.TAIL_MIN_IMPRESSIONS,
        head_pctl=config.HEAD_PCTL,
        torso_pctl=config.TORSO_PCTL,
        ipw_clip_position=config.IPW_CLIP_POSITION,
        decay_flatness=config.DECAY_FLATNESS,
        decay_midpoint=config.DECAY_MIDPOINT,
        alpha=config.ALPHA,
        min_group_size=config.MIN_GROUP_SIZE,
        max_group_size=args.max_group_size,
    )

    print(f"Window:     {start_date} .. {end_date}  ({args.lookback_days}d, lag {args.days_before_today}d)")
    print(f"LTR params: banner={config.BANNER} channel={config.CHANNEL} "
          f"ipw_clip={config.IPW_CLIP_POSITION} decay=({config.DECAY_FLATNESS}, {config.DECAY_MIDPOINT}) "
          f"alpha={config.ALPHA} group_size=[{config.MIN_GROUP_SIZE}, {args.max_group_size}]")
    print(f"Term pool:  head={config.TERM_POOL_HEAD} tail={config.TAIL_POOL_SIZE} "
          f"(tail floor {config.TAIL_MIN_IMPRESSIONS} impressions); "
          f"tiers head>={config.HEAD_PCTL:.2f} torso>={config.TORSO_PCTL:.2f} by volume percentile")
    print("Running judgement-list query (this can take several minutes)...")

    pairs = pd.DataFrame(DatabricksSQL().query_dicts(sql))
    if pairs.empty:
        print("No rows returned; widen the window or relax the filters.", file=sys.stderr)
        return 1

    numeric = [
        "relevance",
        "term_impressions",
        "weighted_ctr",
        "ctr",
        "mean_position",
        "total_weighted_impressions",
        "total_weighted_clicks",
        "total_impressions",
        "total_clicks",
    ]
    for col in numeric:
        pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    pairs["relevance"] = pairs["relevance"].fillna(0).astype(int)

    # robustness label: same 0-4 bin structure on undebiased CTR, graded within query
    pct = pairs.groupby("search_term")["ctr"].rank(pct=True, method="average")
    raw = pd.Series(0, index=pairs.index, dtype=int)
    for cutoff, grade in ((0.25, 1), (0.50, 2), (0.75, 3), (0.90, 4)):
        raw[pct >= cutoff] = grade
    raw[pairs["total_clicks"] <= 0] = 0
    pairs["relevance_raw"] = raw

    pairs.to_csv(config.TEST_SET_CSV, index=False)

    products = (
        pairs[["ecode", "product_title", "image_url", "brand_name", "category_name"]]
        .drop_duplicates(subset="ecode")
        .reset_index(drop=True)
    )
    products.to_csv(config.PRODUCTS_CSV, index=False)

    n_queries = pairs["search_term"].nunique()
    pool_sizes = pairs.groupby("search_term").size()
    print(f"queries:          {n_queries}")
    print(f"query-doc pairs:  {len(pairs)}")
    print(f"unique products:  {len(products)}")
    print(f"pool size:        min {pool_sizes.min()}  median {pool_sizes.median():.0f}  max {pool_sizes.max()}")
    print("queries per tier:")
    tier_counts = pairs.groupby("query_tier")["search_term"].nunique().reindex(["head", "torso", "tail"])
    print(tier_counts.to_string())
    print("LTR relevance grades:")
    print(pairs["relevance"].value_counts().sort_index().to_string())

    # W4 standing diagnostic: a properly position-debiased label set should show |rho| < 0.05.
    # 0.05-0.15 partial, >0.15 not debiased. The undebiased raw-CTR label is the control.
    rho = pairs["mean_position"].corr(pairs["relevance"])
    rho_raw = pairs["mean_position"].corr(pairs["relevance_raw"])
    verdict = "debiased" if abs(rho) < 0.05 else ("partial" if abs(rho) < 0.15 else "NOT DEBIASED")
    print(f"corr(mean_position, relevance):     {rho:+.3f}   [{verdict}]")
    print(f"corr(mean_position, relevance_raw): {rho_raw:+.3f}   [undebiased control]")
    print("mean impression position by grade:")
    print(pairs.groupby("relevance")["mean_position"].mean().round(1).to_string())

    print(f"saved -> {config.TEST_SET_CSV}")
    print(f"saved -> {config.PRODUCTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
