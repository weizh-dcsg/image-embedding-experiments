# Search Relevance Embedding Study — Data Collection and Relevance Labels

**Scope:** how the evaluation dataset was assembled — which events, which products, which images — and how relevance grades 0–4 were derived from click behaviour.

> **TL;DR**
> 90 days of `SRLP` search traffic from `ml_events`, restricted to DSG web-active products having both a title and an image. 19,468 of 19,585 product images downloaded (99.4%). Relevance labels reproduce the `ds-ecm-search-ranking-ltr` judgement-list recipe exactly — inverse-propensity weighting, sigmoid time decay, then per-query weighted-CTR quartile bins. Final set: **300 queries, 33,279 query-product pairs, 19,468 products.**
>
> Three defects in the inherited LTR configuration were found along the way and are flagged below.

**Source files:** `01_build_test_set.py`, `sql/judgement_list.sql`, `02_download_images.py`, `config.py`

---

# Part 1 — Data Collection

## 1.0 Tables used

Three metastore tables, each with a distinct role. No other source is read.

| Table | Role | Columns consumed |
| --- | --- | --- |
| `prod_ent_silver_db.sdsc.ml_events` | Behaviour — searches, impressions, clicks | `search_event.*`, `search_result.items`, `click_event.id`, `click_event.num`, `parent_id`, `banner`, `channel`, `event_date_short` |
| `entdata.web.dim_sku_bod_web_active` | Merchandising — what is sellable, and its imagery | `ecode`, `default_ecode_image_url`, `brand_name`, `primary_category_name`, `web_chain_code` |
| `prod_ml_feature_store_db.products.ecode` | Product metadata — title and active flag | `product_title`, `dsg_web_active` |
| `prod_ml_feature_store_db.products.ecode_attribute` | The LTR "Big-4" structured product attributes | `ecode`, `attr_id`, `attr_name`, `attr_value` |

`ml_events` is referenced five times — propensity numerator and denominator, impressions, searches, and clicks. The two product tables are joined once, in the `active_products` CTE. `ecode_attribute` is fetched separately by `01b_fetch_attributes.py` and is used only by the attribute-embedding experiment.

> **Worth knowing:** the active flag `dsg_web_active` lives on the **feature-store** table while the image URL `default_ecode_image_url` lives on the **merchandising** table. Both are required and neither alone is sufficient.

## 1.1 Source

Search behaviour comes from `prod_ent_silver_db.sdsc.ml_events`, the ML event stream carrying search, impression, and click events.

*Implemented in `sql/judgement_list.sql` — CTEs `searches`, `impressions`, `clicks`, `filled_impression_positions`.*

| Filter | Value | Rationale |
| --- | --- | --- |
| `event_date_short` | 2026-05-08 to 2026-08-05 | 90 days, matching LTR's `num_days_train` |
| Reporting lag | 3 days | matching LTR's `days_before_today`; avoids incomplete recent partitions |
| `banner` | `DSG` | `ml_events` stores this **upper-cased**; clickstream tables use lower-case |
| `channel` | `WEB` | |
| `search_event.type` | `SRLP` | search results page, not category or promo surfaces |
| `search_event.page` | `0` | first page only |

> **Gotcha:** the banner casing differs between `ml_events` and the clickstream tables. Querying `ml_events` with `banner = 'dsg'` returns zero rows silently.

### On the 3-day lag

The current day is unambiguously partial — measured mid-day it carried 6.5M events against a 24–31M daily norm. But T-1 onward already looks mature in this stream: the 24.5M–31.5M spread across older days tracks weekday/weekend seasonality, not ingestion. The partial day does not distort CTR either (5.94% against a 5.66–6.04% normal range).

So a 1-day lag would likely suffice for `ml_events` alone. The 3-day value is inherited from the LTR job, which also joins Adobe clickstream — slower to settle, subject to restatement — and needs attribution windows to close. At a 90-day window the choice is immaterial: three days is 3% of the data, and those days carry near-maximal weight under the time decay described in Part 2, so erring conservative is cheap insurance.

### Event types

| Type | Meaning | Use |
| --- | --- | --- |
| `I` | Impression | products actually rendered to the shopper |
| `S`, `SPL` | Search / sponsored result list | backfills impressions the `I` event missed |
| `C` | Click | positive engagement signal |

Impressions are exploded from the nested `search_result.items` array, keeping item types `P` / `PP` / `SP` (product, promoted product, sponsored product). Clicks come from `click_event.id` and `click_event.num`.

### Impression backfill

Impression events do not always fire before a shopper interacts, so raw `I` events under-report what was actually seen. Following the LTR pipeline, we treat any product appearing in the `S`/`SPL` result list at a rank **at or above the deepest confirmed impression** as having been impressed.

The assumption is a cascade model: to reach rank 24 you must have scrolled past ranks 1–23, so the deepest confirmed impression acts as a watermark. Without this correction, deep-rank products are systematically under-counted, which inflates their apparent click-through rate.

Two limits carry forward:

- It only fills **above** the watermark. Products below the deepest confirmed impression stay uncounted, so the correction is one-sided.
- A search event with **no** `I` event at all is dropped entirely rather than backfilled — the join produces a NULL watermark and the row is filtered out.

---

## 1.2 Candidate restriction: active products

A product is a legitimate candidate only if a shopper could actually have bought it and we can represent it in both modalities. We use the active-product definition already established in `ds-ecm-search-ranking-ltr`.

*Implemented in `sql/judgement_list.sql` — CTE `active_products`.*

```sql
entdata.web.dim_sku_bod_web_active   WHERE web_chain_code = 'DSG'
  INNER JOIN prod_ml_feature_store_db.products.ecode  ON ecode
  WHERE dsg_web_active = 'Y'
    AND product_title           IS NOT NULL AND <> ''
    AND default_ecode_image_url IS NOT NULL AND <> ''
```

Catalogue coverage on the current snapshot:

| Property | Value |
| --- | --- |
| DSG web-active ecodes | 149,801 |
| With a default image URL | 149,801 (**100%**) |
| With a product title | 146,545 (97.8%) |

Complete image-URL coverage is what makes the study possible. `default_ecode_image_url` is the only usable product-imagery source we located in the metastore.

---

## 1.3 Imagery acquisition

*Implemented in `02_download_images.py` — `render_url()` handles preset stripping and percent-encoding; `download_one()` contains per-item errors.*

Image URLs arrive as Scene7 asset paths with stacked rendering presets:

```
https://dks.scene7.com/is/image/dkscdn/15RNGARSPRSTYLMFGBXNX_White_Black_Red_is?$UTPMain$?$DSG_Google_PLA$
```

Those presets encode merchandising crops of unknown geometry — different products would reach the encoder at different framings. We strip everything after the first `?` and request an explicit square render instead:

```
?wid=512&hei=512&fmt=jpeg&qlt=85
```

| Outcome | Count | Share |
| --- | --- | --- |
| Downloaded | **19,468** | 99.4% |
| HTTP 403 | 117 | 0.6% |

Failures are dropped from **all** systems equally, so they cannot bias a comparison between them.

> **Two implementation notes for anyone reusing this code**
>
> 1. Asset paths occasionally contain non-ASCII characters (e.g. `Crème`), which `urllib` cannot encode into a request line. Paths must be percent-encoded — otherwise a `UnicodeEncodeError` terminates the batch mid-run.
> 2. A single unusable URL must not propagate out of the thread pool. Catch per item, or one bad asset kills the entire download.

---

# Part 2 — Relevance Labels

## 2.1 Why we reused the production definition

An evaluation is only as meaningful as its target. Inventing a relevance definition for this study would have measured embeddings against something the deployed ranker is not trained on. We therefore reproduced the judgement-list recipe from `ds-ecm-search-ranking-ltr` exactly.

---

## 2.2 Construction

*Implemented in `sql/judgement_list.sql` — `ipw` (propensity) to `weights` (time decay) to `ctrs` to `query_quantiles` / `global_quantiles` / `smoothed_quantiles` to `scored` (grades 0–4).*

### Step 1 — Examination propensity

Deep ranks are examined less often, so a click at rank 40 is stronger evidence than a click at rank 2. Propensity per rank `k` is estimated from the observed impression-to-search ratio and inverted:

```
tau(k) = 1 / ( impressions_at_k / searches_containing_k )
```

The denominator counts search events whose result list *contained* rank `k`; the numerator counts those that actually *rendered* it. The ratio is the empirical examination probability. Measured over two days of traffic:

| Rank k | Searches containing k | Impressions at k | P(examine) | tau(k) |
| --- | --- | --- | --- | --- |
| 1 | 2,507,301 | 2,507,301 | 1.000 | 1.00 |
| 2 | 2,489,448 | 2,469,819 | 0.992 | 1.01 |
| 3 | 2,415,667 | 1,991,932 | 0.825 | 1.21 |
| 5 | 2,276,192 | 1,550,644 | 0.681 | 1.47 |
| 10 | 2,047,874 | 1,115,322 | 0.545 | 1.84 |
| 20 | 1,772,948 | 655,556 | 0.370 | 2.70 |
| 30 | 1,593,051 | 419,449 | 0.263 | 3.80 |
| 40 | 1,414,518 | 280,166 | 0.198 | 5.05 |
| 48 | 1,325,610 | 213,083 | 0.161 | **6.22** |
| 60 | 253,200 | 32,982 | 0.130 | 7.68 |
| 80 | 2,843 | 944 | 0.332 | 3.01 (noise) |

A click at rank 40 counts about **5x a click at rank 1**. Two features are worth noting:

- The drop from 0.992 at rank 2 to 0.825 at rank 3 is the **fold** — the viewport boundary on a typical device, visible directly in the data.
- Rank 80 breaks monotonicity because only 2,843 searches reach that depth. Those are atypical sessions and the estimate is noise. Ranks beyond 48 are therefore clipped to `tau(48)`, which covers just under 99% of clicks (click-rank p99 = 50).

Rank comes from `click_event.num` on the click side and `search_result.items.num` on the impression side. Both are **1-indexed** (verified: min = 1, no zeros), which is why the clip constant is a rank and not an offset. The click-rank distribution is extremely top-heavy — **23.8% of all clicks land on position 1**, median rank 4 — which is the bias this correction exists to undo.

### Step 2 — Time decay

Older behaviour is less informative about current relevance. A sigmoid decay is applied on observation age `a` in days:

```
w_decay(a) = (1 + exp(-lambda * mu)) / (1 + exp(lambda * (a - mu)))

    lambda = 0.18   (steepness)
    mu     = 30     (midpoint, in days)
```

The numerator normalises `w(0) = 1`. The shape is *flat then cliff* rather than exponential — recent behaviour is barely penalised for two weeks, then collapses:

| Age (days) | 0 | 7 | 14 | 21 | 30 | 37 | 45 | 60 | 89 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weight | 1.000 | 0.989 | 0.951 | 0.839 | **0.502** | 0.222 | 0.063 | 0.005 | 0.00002 |

> **Defect 1 — the 90-day window is effectively a 36-day window.**
> The LTR config sets `num_days_train = 90` but `decay_midpoint = 30`. Those two parameters disagree about how much history matters.

Assuming roughly uniform daily volume, weight distributes as:

| Period | Share of total weight |
| --- | --- |
| Days 0–15 | 48.1% |
| Days 15–30 | 38.4% |
| Days 30–45 | 12.2% |
| Days 45–60 | 1.2% |
| **Days 60–90** | **0.09%** |

95% of the weight sits in the first 36 days, and effective sample size is **41.4%** of a flat 90-day window. We scan 90 days of `ml_events` — the expensive part of the query — for a final third contributing under 1% of the signal.

**Recommendation:** either shorten the window to about 45 days at negligible signal cost, or raise the decay midpoint.

### Step 3 — Combined weight and normalisation

```
w = tau(k) * w_decay(a)
```

rescaled within each query so total weight equals total impressions, keeping magnitudes interpretable.

### Step 4 — Weighted CTR

The weight is applied to impressions **and** clicks:

```
wCTR(query, ecode) = SUM(click * w) / SUM(impression * w)
```

This is the production definition. Note it re-weights *which observations count* rather than rescaling the CTR level — a consequence examined separately in the evaluation-validity report.

### Step 5 — Graded binning

Cut points are the 25th / 50th / 75th percentiles of wCTR, with `q0` = min and `q4` = max. Three properties matter more than the percentile choice:

1. **Only clicked products define the cut points** (`WHERE total_weighted_clicks > 0`). Zero-click products are excluded from quantile estimation and assigned grade 0 separately — otherwise a long tail of zeros would drag every boundary toward zero.
2. **Percentiles are impression-weighted**, not per-product. `PERCENTILE(wctr, 0.25, int_impressions)` uses the third argument as a frequency weight, so high-traffic products dominate boundary placement and noisy low-traffic estimates cannot set them.
3. **Cut points are per-query.** A query converting at 8% and one converting at 1% both produce a full 0–4 spread. Grades are relative to the query's own pool, which is what NDCG needs.

Quantiles are then smoothed toward the global distribution:

```
q_smooth(i) = ( n * q_local(i) + alpha * q_global(i) ) / ( n + alpha )

    alpha = 1.0
    n     = SUM(total_weighted_impressions) for the query
```

> **Defect 2 — this smoothing is inert.**
>
> `n` runs from 248,481 to 10.5M after IPW inflation (median 487,165). The global prior therefore receives weight `alpha / (n + alpha)`, roughly 2e-6 — **0.0002%**.
>
> For `alpha = 1` to matter, `n` would need to be single digits, so the parameter appears designed for a counting scale (products, or raw clicks) rather than a sum of propensity-inflated impressions. Harmless for our high-traffic queries, but the mechanism meant to protect low-traffic queries is inactive at any plausible `alpha`.

Final grade assignment:

| Grade | Condition |
| --- | --- |
| 0 | zero weighted clicks |
| 1 | `0 < wCTR <= q1` |
| 2 | `q1 < wCTR <= q2` |
| 3 | `q2 < wCTR <= q3` |
| 4 | `q3 < wCTR <= q4` |

Because the upper tail is wide and sparse, the realised distribution is skewed rather than an even quartile split: **28% / 29% / 16% / 14% / 13%**.

> **Grade 0 is overloaded.** It means *never clicked* — not *worst clicked*. A clicked product in the bottom quartile receives grade 1, so grade 0 mixes genuinely irrelevant products with products never shown often enough to accumulate a click. Any metric that binarises the label at `grade > 0` therefore measures click incidence, not relevance.

> **Checked: the `ELSE 0` fall-through never fires.** A product whose wCTR exceeds the smoothed `q4` would drop to grade 0 — the best product in a query silently graded worst. It cannot happen: `q4_global` is the maximum over all clicked products, so `q4_global >= q4_local` always, and smoothing can only raise a query's `q4`. Verified on the built set — all 300 per-query argmax products carry grade 4.

### Step 6 — Group filters

A query is kept only if it has at least 2 distinct relevance levels (otherwise ranking is undefined) and a candidate pool of 10–240 products (`min_group_size` / `max_group_size`).

---

## 2.3 Query sampling — and a trap

*Implemented in `sql/judgement_list.sql` — `top_terms` (pool) to `level_filtered` to `group_size_filtered` to `selected_terms`. The sample happens last. Parameters in `config.py`: `TERM_POOL`, `N_QUERIES`, `MIN_GROUP_SIZE`, `MAX_GROUP_SIZE`.*

We sample 300 queries for compute reasons. The sampling must happen **after** the LTR filters, not before.

An early attempt selected the top-volume terms first, then applied filters, and lost 30 of 50 queries. The reason is structural: `max_group_size = 240` *drops* oversized groups rather than truncating them, and the highest-volume head terms are precisely the ones that exceed it. Sampling before filtering therefore selects the queries most likely to be discarded. The final design draws from a 6,000-term candidate pool and samples 300 survivors.

### How much the cap excludes

Measured over the top 6,000 terms:

| Outcome | Terms | Share |
| --- | --- | --- |
| Dropped — pool > 240 | **1,650** | **27.5%** |
| Dropped — pool < 10 | 0 | 0% |
| Kept | 4,350 | 72.5% |

Median pool before the cap is 177, p90 is 415, max 7,226. So the upper bound discards **more than a quarter of candidate queries**, all from the high-volume end, while the lower bound never binds.

> **Defect 3 — the query set is mid-head, not head.**
>
> The true head terms were removed by a constraint inherited from LTR *training economics*, not from evaluation validity. A gradient-boosted ranker caps group size because training cost scales with it; an offline embedding evaluation has no such cost — scoring the full 149,801-product catalogue takes 3.1 ms.
>
> Larger pools would also be a **harder and more discriminative** test. A sensitivity check at `max_group_size` of 1000 and 2000 is under way.

Scale at each cap:

| Cap | Queries | Unique products | Mean pool | Median | Max |
| --- | --- | --- | --- | --- | --- |
| 240 (this report) | 300 | 19,468 | 141 | 139 | 240 |
| 1000 | 300 | 37,063 | 492 | 476 | 971 |
| 2000 | 300 | 47,534 | 874 | 825 | 1,962 |
| unlimited | 300 | 71,565 | 3,048 | 1,487 | 11,232 |

---

## 2.4 Resulting dataset

*Produced by `01_build_test_set.py` into `test_set.csv` and `products.csv`.*

| Property | Value |
| --- | --- |
| Queries | 300 |
| Query-product pairs | 33,279 |
| Unique products | 19,468 |
| Pool size | min 10, median 105, mean 110.9, max 237 |

| Grade | Pairs | Share |
| --- | --- | --- |
| 4 | 4,238 | 12.7% |
| 3 | 4,602 | 13.8% |
| 2 | 5,361 | 16.1% |
| 1 | 9,750 | 29.3% |
| 0 | 9,328 | 28.0% |

A secondary label set — **raw CTR** binned into the same 0–4 structure without debiasing — is scored alongside every experiment as a robustness check.

---

# Summary of findings for the LTR team

| # | Finding | Impact | Suggested action |
| --- | --- | --- | --- |
| 1 | `num_days_train = 90` disagrees with `decay_midpoint = 30`. Days 60–90 carry 0.09% of weight; effective window is 36 days; ESS is 41.4%. | Query scans 3x more history than contributes signal. | Shorten window to ~45 days, or raise decay midpoint. |
| 2 | Smoothing parameter `alpha = 1.0` is inert — global prior gets 0.0002% weight because `n` is propensity-inflated impressions. | Low-traffic queries are unprotected despite the mechanism existing. | Rescale `alpha` to the units of `n`, or use a count-based `n`. |
| 3 | `max_group_size = 240` drops 27.5% of candidate terms, all high-volume. | Evaluation and training both operate on mid-head, not head, traffic. | Truncate oversized groups rather than dropping them, or raise the cap for evaluation. |

---

# Reproduction

| Stage | Script | Output |
| --- | --- | --- |
| 1 | `01_build_test_set.py` | `test_set.csv`, `products.csv` |
| 1b | `01b_fetch_attributes.py` | `product_attributes.csv` |
| 2 | `02_download_images.py` | `images/`, `image_manifest.csv` |

SQL lives in `sql/judgement_list.sql`. Databricks access uses the editor's OAuth session — no credentials in code.

Key parameters, all overridable by environment variable:

```
LOOKBACK_DAYS=90        DAYS_BEFORE_TODAY=3     N_QUERIES=300
TERM_POOL=6000          MIN_GROUP_SIZE=10       MAX_GROUP_SIZE=240
IPW_CLIP_POSITION=48    DECAY_FLATNESS=0.18     DECAY_MIDPOINT=30
ALPHA=1.0
```

> **Operational note:** store the data directory **outside** cloud-synced folders. Running from OneDrive CloudStorage caused a mid-run `TimeoutError` on dehydrated files and throttled image writes to roughly a third of achievable throughput. Set `EMBEDDING_EVAL_DATA` to a local path.
