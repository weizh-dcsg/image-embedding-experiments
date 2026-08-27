# W1 — Data Collection and Evaluation Setup, with a Baseline Modality Experiment

**Report 1 of 4** · DS eCom Search Ranking · 2026
Scope: build a relevance evaluation harness for product embeddings, and run the first controlled
comparison of image against text representations.

---

## Summary

We built an offline evaluation harness that scores product embeddings against the **production
learning-to-rank judgement list**, so that "relevance" in this work means the same thing it means to
the deployed ranker. The harness covers 300 queries, 33,279 query-product pairs and 19,468 active
products drawn from a 90-day window.

The first experiment compares a product's **photograph** against its **title** as the ranking
representation, using a dual-tower encoder so that the modality is the only variable that changes.
The result is **inconclusive**: images lead titles by +5.5% NDCG@10 with a confidence interval that
crosses zero (p = 0.102). We also find no measurable difference between two independently trained
text encoders (−0.8%, p = 0.843), which rules out encoder quality as an explanation for anything we
observe later.

The week's deliverable is therefore the harness plus a calibrated sense of its resolution: with 300
queries, effects below roughly 6% relative NDCG@10 are not distinguishable from noise.

---

## 1. Objective

Three questions had to be answered before any embedding comparison could be trusted:

1. What does "relevant" mean, operationally, and can we compute it from logged behaviour?
2. Which products are legitimate candidates?
3. How large an effect can this design actually detect?

Everything in this report exists to answer those.

---

## 2. Data collection

**Code:** [`01_build_test_set.py`](../01_build_test_set.py) · [`sql/judgement_list.sql`](../sql/judgement_list.sql) · [`02_download_images.py`](../02_download_images.py)

### 2.0 Tables used

Three metastore tables, each with a distinct role. No other source is read.

| Table | Role | Columns consumed |
| --- | --- | --- |
| `prod_ent_silver_db.sdsc.ml_events` | Behaviour — searches, impressions, clicks | `search_event.*`, `search_result.items`, `click_event.id`, `click_event.num`, `parent_id`, `banner`, `channel`, `event_date_short` |
| `entdata.web.dim_sku_bod_web_active` | Merchandising — what is sellable, and its imagery | `ecode`, `default_ecode_image_url`, `brand_name`, `primary_category_name`, `web_chain_code` |
| `prod_ml_feature_store_db.products.ecode` | Product metadata — title and active flag | `product_title`, `dsg_web_active` |
| `prod_ml_feature_store_db.products.ecode_attribute` | The LTR "Big-4" product attributes (§5.1b) | `ecode`, `attr_id`, `attr_name`, `attr_value` |

`ml_events` is referenced five times (propensity numerator and denominator, impressions, searches,
clicks). The two product tables are joined once, in `active_products`. Note the active flag lives on
the **feature-store** table while the image URL lives on the **merchandising** table — both are
required, and neither alone is sufficient.

### 2.1 Source

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) — CTEs `searches`, `impressions`, `clicks`, `filled_impression_positions` (backfill)

Search behaviour comes from `prod_ent_silver_db.sdsc.ml_events`, the ML event stream carrying
search, impression, and click events.

| Filter | Value | Rationale |
| --- | --- | --- |
| `event_date_short` | 2026-05-08 → 2026-08-05 | 90 days, matching LTR's `num_days_train` |
| Reporting lag | 3 days | matching LTR's `days_before_today`; avoids incomplete recent partitions |
| `banner` | `DSG` | note: `ml_events` stores this upper-cased; clickstream tables use lower-case |
| `channel` | `WEB` | |
| `search_event.type` | `SRLP` | search results page, not category or promo surfaces |
| `search_event.page` | `0` | first page only |

**On the 3-day lag.** The current day is unambiguously partial — measured mid-day it carried 6.5M
events against a 24–31M daily norm. But T−1 onward already looks mature in this stream: the
24.5M–31.5M spread across older days tracks weekday/weekend seasonality, not ingestion. Nor does the
partial day distort CTR (5.94% against a 5.66–6.04% normal range). So a 1-day lag would likely
suffice for `ml_events` alone; the 3-day value is inherited from the LTR job, which also joins Adobe
clickstream (slower to settle, subject to restatement) and needs attribution windows to close. At a
90-day window the choice is immaterial — three days is 3% of the data, and under the time decay in
§3.2 those days carry near-maximal weight, so erring conservative is cheap insurance.

Event types used:

| Type | Meaning | Use |
| --- | --- | --- |
| `I` | Impression | products actually rendered to the shopper |
| `S`, `SPL` | Search / sponsored result list | backfills impressions the `I` event missed |
| `C` | Click | positive engagement signal |

Impressions are exploded from the nested `search_result.items` array, keeping item types
`P` / `PP` / `SP` (product, promoted product, sponsored product). Clicks come from
`click_event.id` and `click_event.num`.

**Impression backfill.** Impression events do not always fire before a shopper interacts. Following
the LTR pipeline, we treat any product appearing in the `S`/`SPL` result list at a rank at or above
the deepest confirmed impression as having been impressed. Without this, deep-rank products are
systematically under-counted.

### 2.2 Candidate restriction: active products

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) — CTE `active_products`

A product is a legitimate candidate only if a shopper could actually have bought it and we can
represent it in both modalities. We use the active-product definition already established in
`ds-ecm-search-ranking-ltr`:

```
entdata.web.dim_sku_bod_web_active   WHERE web_chain_code = 'DSG'
  INNER JOIN prod_ml_feature_store_db.products.ecode  ON ecode
  WHERE dsg_web_active = 'Y'
    AND product_title      IS NOT NULL AND <> ''
    AND default_ecode_image_url IS NOT NULL AND <> ''
```

Catalogue coverage on the current snapshot:

| Property | Value |
| --- | --- |
| DSG web-active ecodes | 149,801 |
| With a default image URL | 149,801 (**100%**) |
| With a product title | 146,545 (97.8%) |

Image URL coverage being complete is what makes the whole study possible; `default_ecode_image_url`
is the only usable product-imagery source we located in the metastore.

### 2.3 Imagery acquisition

**Code:** [`02_download_images.py`](../02_download_images.py) — `render_url()` (preset stripping, percent-encoding), `download_one()` (per-item error containment)

Image URLs arrive as Scene7 asset paths with stacked rendering presets:

```
https://dks.scene7.com/is/image/dkscdn/15RNGARSPRSTYLMFGBXNX_White_Black_Red_is?$UTPMain$?$DSG_Google_PLA$
```

Those presets encode merchandising crops of unknown geometry. We strip everything after the first
`?` and request an explicit square render, so every product reaches the encoder at the same
resolution and framing:

```
?wid=512&hei=512&fmt=jpeg&qlt=85
```

| Outcome | Count | Share |
| --- | --- | --- |
| Downloaded | **19,468** | 99.4% |
| HTTP 403 | 117 | 0.6% |

Failures are dropped from **all** systems equally, so they cannot bias a comparison.

Two implementation notes worth carrying forward. Asset paths occasionally contain non-ASCII
characters (e.g. `Crème`), which `urllib` cannot encode into a request line — paths must be
percent-encoded. And a single unusable URL must not be allowed to propagate out of the thread pool,
or it terminates the entire batch.

---

## 3. Relevance labels

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) · [`config.py`](../config.py) (all tunable parameters)

### 3.1 Why we reused the production definition

An evaluation is only as meaningful as its target. Inventing a relevance definition for this study
would have measured embeddings against something the deployed ranker is not trained on. We therefore
reproduced the judgement-list recipe from `ds-ecm-search-ranking-ltr` exactly.

### 3.2 Construction

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) — `ipw` (propensity) → `weights` (time decay) → `ctrs` → `query_quantiles` / `global_quantiles` / `smoothed_quantiles` → `scored` (grades 0–4)

**Step 1 — examination propensity.** Deep ranks are examined less often, so a click at rank 40 is
stronger evidence than a click at rank 2. Propensity per rank $k$ is estimated from the observed
impression-to-search ratio and inverted:

$$\tau_k = \left( \frac{\text{impressions at } k}{\text{searches containing } k} \right)^{-1}$$

The denominator counts search events whose result list *contained* rank $k$; the numerator counts
those that actually *rendered* it. The ratio is the empirical examination probability. Measured over
two days of traffic:

| $k$ | searches containing $k$ | impressions at $k$ | $P(\text{examine})$ | $\tau_k$ |
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
| 80 | 2,843 | 944 | 0.332 | 3.01 ← noise |

So a click at rank 40 counts about **5× a click at rank 1**. Two features are worth noting. The drop
from 0.992 at rank 2 to 0.825 at rank 3 is the **fold** — the viewport boundary on a typical device,
visible directly in the data. And rank 80 breaks monotonicity because only 2,843 searches reach that
depth; those are atypical sessions and the estimate is noise. Ranks beyond 48 are therefore clipped
to $\tau_{48}$, which covers just under 99% of clicks (click-rank p99 = 50).

Rank comes from `click_event.num` on the click side and `search_result.items.num` on the impression
side. Both are **1-indexed** (verified: min = 1, no zeros), which is why the clip constant is a rank
and not an offset. The click-rank distribution is extremely top-heavy — **23.8% of all clicks land on
position 1**, median rank 4 — which is the bias this correction exists to undo.

**Step 2 — time decay.** Older behaviour is less informative about current relevance. A sigmoid decay
is applied on observation age $a$ in days:

$$w_{\text{decay}}(a) = \frac{1 + e^{-\lambda \mu}}{1 + e^{\lambda (a - \mu)}}, \qquad \lambda = 0.18,\ \mu = 30$$

$\mu = 30$ is the half-life, $\lambda$ the steepness, and the numerator normalises $w(0) = 1$. The
shape is *flat then cliff* rather than exponential — recent behaviour is barely penalised for two
weeks, then collapses:

| Age (days) | 0 | 7 | 14 | 21 | **30** | 37 | 45 | 60 | 89 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weight | 1.000 | 0.989 | 0.951 | 0.839 | **0.502** | 0.222 | 0.063 | 0.005 | 0.00002 |

**Finding — the 90-day window is effectively a 36-day window.** Assuming roughly uniform daily
volume, weight distributes as:

| Period | Share of total weight |
| --- | --- |
| Days 0–15 | 48.1% |
| Days 15–30 | 38.4% |
| Days 30–45 | 12.2% |
| Days 45–60 | 1.2% |
| **Days 60–90** | **0.09%** |

95% of the weight sits in the first 36 days, and effective sample size is **41.4%** of a flat 90-day
window. The LTR config sets `num_days_train = 90` but `decay_midpoint = 30`; those two parameters
disagree about how much history matters. We scan 90 days of `ml_events` — the expensive part of the
query — for a final third that contributes under 1% of the signal. Either shorten the window to ~45
days at negligible signal cost, or raise the decay midpoint. Flagged in W4 §5.

**Step 3 — combined weight and normalisation.** $w = \tau_k \cdot w_{\text{decay}}$, rescaled within
each query so total weight equals total impressions (keeping magnitudes interpretable).

**Step 4 — weighted CTR.** The weight is applied to impressions **and** clicks:

$$\text{wCTR}(q,e) = \frac{\sum \text{click} \cdot w}{\sum \text{impression} \cdot w}$$

This is the production definition. Note it re-weights *which observations count* rather than
rescaling the CTR level — a consequence examined in W4.

**Step 5 — graded binning.** Cut points are the 25th/50th/75th percentiles of wCTR, with $q_0$ = min
and $q_4$ = max. Three properties matter more than the percentile choice:

1. **Only clicked products define the cut points** (`WHERE total_weighted_clicks > 0`). Zero-click
   products are excluded from quantile estimation and assigned grade 0 separately — otherwise a long
   tail of zeros would drag every boundary toward zero.
2. **Percentiles are impression-weighted**, not per-product: `PERCENTILE(wctr, 0.25, int_impressions)`
   uses the third argument as a frequency weight, so high-traffic products dominate boundary
   placement and noisy low-traffic estimates cannot set them.
3. **Cut points are per-query.** A query converting at 8% and one converting at 1% both produce a
   full 0–4 spread. Grades are relative to the query's own pool, which is what NDCG needs.

Quantiles are then smoothed toward the global distribution:

$$q_i^{\text{smooth}} = \frac{n \cdot q_i^{\text{local}} + \alpha \cdot q_i^{\text{global}}}{n + \alpha}, \qquad \alpha = 1.0$$

**Finding — this smoothing is inert.** $n$ is `SUM(total_weighted_impressions)`, which after IPW
inflation runs from 248,481 to 10.5M (median 487,165). The global prior therefore receives weight
$\alpha/(n+\alpha) \approx 2\times10^{-6}$ — **0.0002%**. For $\alpha = 1$ to matter, $n$ would need to
be single digits, so the parameter appears designed for a counting scale (products, or raw clicks)
rather than a sum of propensity-inflated impressions. Harmless for our high-traffic queries, but the
mechanism meant to protect low-traffic queries is inactive at any plausible $\alpha$. Flagged in W4 §5.

| Grade | Condition |
| --- | --- |
| 0 | zero weighted clicks |
| 1 | $0 < \text{wCTR} \le q_1$ |
| 2 | $q_1 < \text{wCTR} \le q_2$ |
| 3 | $q_2 < \text{wCTR} \le q_3$ |
| 4 | $q_3 < \text{wCTR} \le q_4$ |

Because the upper tail is wide and sparse, the realised distribution is skewed rather than an even
quartile split: **28% / 29% / 16% / 14% / 13%**.

**Grade 0 is overloaded.** It means *never clicked* — not *worst clicked*. A clicked product in the
bottom quartile receives grade 1, so grade 0 mixes genuinely irrelevant products with products never
shown often enough to accumulate a click. This is what makes the binary supporting metrics in §4.2
measure click incidence rather than relevance.

**The `ELSE 0` fall-through never fires.** A product whose wCTR exceeds the smoothed $q_4$ would drop
to grade 0 — the best product in a query silently graded worst. It cannot happen: $q_4^{\text{global}}$
is the maximum over all clicked products, so $q_4^{\text{global}} \ge q_4^{\text{local}}$ always, and
smoothing can only raise a query's $q_4$. Verified on the built set — all 300 per-query argmax
products carry grade 4.

**Step 6 — group filters.** A query is kept only if it has ≥2 distinct relevance levels (otherwise
ranking is undefined) and a candidate pool of 10–240 products (`min_group_size` / `max_group_size`).

### 3.3 Query sampling — and a trap

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) — `top_terms` (pool) → `level_filtered` → `group_size_filtered` → `selected_terms` (sample happens last) · [`config.py`](../config.py) `TERM_POOL`, `N_QUERIES`, `MIN_GROUP_SIZE`, `MAX_GROUP_SIZE`

We sample 300 queries for compute reasons. The sampling must happen **after** the LTR filters, not
before.

An early attempt selected the top-volume terms first, then applied filters, and lost 30 of 50
queries. The reason is structural: `max_group_size = 240` *drops* oversized groups rather than
truncating them, and the highest-volume head terms are precisely the ones that exceed it. Sampling
before filtering therefore selects the queries most likely to be discarded. The final design draws
from a 6,000-term candidate pool and samples 300 survivors.

**How much the cap excludes.** Measured over the top 6,000 terms:

| Outcome | Terms | Share |
| --- | --- | --- |
| Dropped — pool > 240 | **1,650** | **27.5%** |
| Dropped — pool < 10 | 0 | 0% |
| Kept | 4,350 | 72.5% |

Median pool before the cap is 177, p90 is 415, max 7,226. So the upper bound discards **more than a
quarter of candidate queries**, all from the high-volume end, while the lower bound never binds.

This has a consequence for how the query set should be described. Our 300 queries are **mid-head, not
head** — the true head terms were removed by a constraint inherited from LTR training economics, not
from evaluation validity. A gradient-boosted ranker caps group size because training cost scales with
it; an offline embedding evaluation has no such cost (W4 §3: scoring the full 149,801-product
catalogue takes 3.1 ms). Larger pools would also be a **harder and more discriminative** test, which
matters given the resolution limit in §5.3.

A sensitivity check at `max_group_size` ∈ {1000, 2000} is therefore under way; results will be
reported in W5. Scale at each cap:

| Cap | Queries | Unique products | Mean pool | Median | Max |
| --- | --- | --- | --- | --- | --- |
| 240 (this report) | 300 | 19,468 | 141 | 139 | 240 |
| 1000 | 300 | 37,063 | 492 | 476 | 971 |
| 2000 | 300 | 47,534 | 874 | 825 | 1,962 |
| unlimited | 300 | 71,565 | 3,048 | 1,487 | 11,232 |

### 3.4 Resulting dataset

**Code:** [`01_build_test_set.py`](../01_build_test_set.py) → `test_set.csv`, `products.csv`

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

A secondary label set — **raw CTR** binned into the same 0–4 structure without debiasing — is scored
alongside every experiment as a robustness check.

---

## 4. Evaluation methodology

**Code:** [`04_evaluate.py`](../04_evaluate.py)

### 4.1 Task framing

**Code:** [`04_evaluate.py`](../04_evaluate.py) — `main()`, one candidate pool per `search_term` shared by all systems

Every system ranks **the same candidate pool** for **the same query**. Pools are the products
production already surfaced, so this measures **re-ranking quality**, not full-catalogue retrieval.
We make no recall claims about the catalogue as a whole.

Scoring is cosine similarity between L2-normalised query and product vectors. Products are sorted
descending; ties broken stably.

### 4.2 Metrics

**Code:** [`04_evaluate.py`](../04_evaluate.py) — `ndcg_at_k()`, `mrr_at_k()`, `recall_at_k()`, `precision_at_k()`, `average_precision()`, aggregated by `score_ranking()`

**NDCG@k** with exponential gain, the primary metric. For a ranked relevance vector $r$:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i+1)}, \qquad
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

Exponential gain matters here: with grades 0–4, it weights a grade-4 product 15× a grade-1 product,
which reflects how sharply engagement concentrates on a few products per query.

Reported at $k \in \{5, 10, 20, 48, 96, 144\}$; **NDCG@10 is primary**. The three deep cutoffs exist
to check that conclusions are not an artefact of shallow truncation: 48 matches the IPW rank clip
(§3.2), and 96 / 144 probe beyond it. They must be read with the pool sizes in mind, because the
cutoff stops binding once it exceeds the candidate list:

| Cutoff | Queries with pool ≥ k |
| --- | --- |
| 10 | 300 / 300 (100%) |
| 20 | 296 / 300 (99%) |
| **48** | 251 / 300 (84%) |
| 96 | 165 / 300 (55%) |
| 144 | 89 / 300 (30%) |

For 70% of queries NDCG@144 *is* NDCG over the whole pool, so @96 and @144 are closer to full-pool
measurements than to depth-limited ones. **@48 is the most informative of the three** — it binds for
84% of queries and matches the depth the label construction itself treats as meaningful.

Supporting metrics: **MRR@k** (rank of first relevant), **MAP** (precision averaged at each
relevant position), **Recall@k** and **Precision@k**.

**NDCG is the only metric that uses the graded scale.** The other four binarize the label at
`grade > 0`, so a grade-1 product counts exactly the same as a grade-4. Since grade 0 means *never
clicked* (§3.2), these metrics ask "did this product ever receive a click?", not "how relevant is
it?". That distinction matters because the binarization is heavily imbalanced:

| Property | Value |
| --- | --- |
| Mean pool size | 110.9 |
| Mean positives (grade > 0) | 79.8 |
| **Positive share of pool** | **76.4%** |

Two consequences, both of which mean the supporting metrics should not be used to separate systems:

**Precision@k is saturated by the base rate.** With 76.4% of the pool positive, `random` scores
0.762 at $k=10$ by construction. Every embedding system lands at 0.88–0.94 and `production` at
0.765. The column is compressed into the top quarter of its range before any system does anything.

**Recall@k is capped by arithmetic.** With ~80 positives and only 10 slots, the maximum attainable
recall@10 averages **0.174**. Observed values cluster at 0.15–0.17, so every system sits against the
ceiling and the differences reflect pool size, not ranking quality.

**A diagnostic that falls out of this.** `production` scores 0.7653 precision@10 — statistically
indistinguishable from `random` at 0.7620 — yet has by far the highest NDCG@10 (0.6176 against
0.4363 for `image`). So it is no better than chance at getting clicked products into the top 10,
but much better at ordering them *by grade*. Since grade derives from click-through rate and
`production` ranks by mean observed position, this is the position leakage of W4 §2 surfacing
through a second, independent route. The embedding systems show the opposite profile: far better
binary retrieval (0.92), lower graded score. It is a compact reason to read NDCG@10 only for
embedding-vs-embedding contrasts, and to keep `production` out of them entirely.

A graded variant of the supporting metrics — defining "positive" as grade ≥ 3 rather than > 0 —
would make Precision@k informative. It is not implemented; the binary definitions are the IR
convention and are kept for comparability.

### 4.2c Behaviour of each metric with depth

All four families are computed at every cutoff. Only NDCG stays informative across the range:

| Metric | Behaviour as k grows | Usable range |
| --- | --- | --- |
| **NDCG@k** | Rises toward 1 as k approaches pool size; system order stable throughout | All k, best ≤ 48 |
| **Recall@k** | Reaches 0.93–0.96 for *every* system including `random` by k = 144 | ≤ 20 |
| **Precision@k** | *Falls* with k — the denominator is k even when the pool is smaller | ≤ 20 |
| **MRR@k** | Flat from k = 20 onward, to four decimal places | ≤ 10 |

MRR saturates because it needs only the *first* relevant item and 76.4% of each pool is graded > 0,
so the first hit is nearly always in the top few ranks. Recall converges because at depth 144 every
system has retrieved almost all positives regardless of ordering. Precision declines because
`precision@k` divides by k unconditionally, so beyond the pool size the denominator grows while the
numerator cannot. None of these is a defect in the systems; all three are properties of a saturated
binary label on a small pool.

### 4.2b Two weightings, two questions

**Code:** [`04_evaluate.py`](../04_evaluate.py) — `query_impressions` per row, `weighted_mean()`, `weighting` column in `summary.csv`

Queries are **sampled** by impression volume but, under a plain average, **evaluated** with equal
weight. A query with 264k impressions counts the same as one with 9.6M. That asymmetry is not
neutral, so every metric is reported under both weightings:

| Weighting | Question it answers |
| --- | --- |
| **Macro** (mean over queries) | How well does this rank a *typical query*? IR convention. |
| **Impression-weighted** | How well does this rank a *typical search impression*? Closer to business impact. |

Volume across the 300 selected queries spans 264,103 to 9,573,478 (36×). The top 10 queries hold
22.4% of impressions and the top 50 hold 52.4%, so impression weighting reduces effective sample size
from 300 to **108.6**.

**They disagree, and the disagreement is systematic.** Image systems gain under traffic weighting
while text systems lose:

| System | Macro | Impression-weighted | Δ |
| --- | --- | --- | --- |
| `image+MGPL` | 0.4430 | 0.4582 | **+0.0152** |
| `image+naive` | 0.4393 | 0.4494 | +0.0101 |
| `image` | 0.4363 | 0.4492 | +0.0129 |
| `jina_text` | 0.4171 | 0.3946 | **−0.0225** |
| `text-siglip` | 0.4137 | 0.3866 | **−0.0271** |
| `fusion` | 0.4627 | 0.4651 | +0.0024 |
| `production` | 0.6176 | 0.5974 | −0.0202 |
| `random` | 0.2775 | 0.2795 | +0.0020 |

Images are relatively **better on the highest-traffic queries**; text is relatively better on the
lower-volume tail of our sample. This is consistent with W3's query-type analysis, where text wins on
proper nouns (`nike sabrina 3`, `ja 3` — lower volume) and images win on jersey and visual categories
(higher volume). Reporting only the macro average understates the modality effect where the traffic
actually is; reporting only the weighted average rests on ~109 effective queries. Both are given
throughout.

### 4.3 Significance testing

**Code:** [`04_evaluate.py`](../04_evaluate.py) — paired bootstrap (`boot`) and weighted bootstrap (`boot_w`) → `results/significance.csv`

Per-query NDCG variance greatly exceeds between-system variance — some queries are simply easier —
so unpaired tests would be badly underpowered. We use a **paired bootstrap over queries**:

1. Compute per-query metric for both systems.
2. Form per-query differences $d_i$ (pairing controls query difficulty).
3. Resample the 300 differences with replacement, 2,000 times.
4. Report mean difference, 2.5/97.5 percentile CI, two-sided p-value, and per-query win rate.

Seed fixed at 20260808. **Caveat:** bootstrap p-values carry roughly ±0.005–0.05 jitter between runs
when the set of systems changes, because the RNG stream shifts. Distinctions like "p = 0.057 versus
p = 0.033" should not be over-interpreted.

The same bootstrap is run under impression weighting, resampling queries and recomputing the weighted
mean difference on each draw. Both sets of intervals are written to `significance.csv`
(`delta` / `p_value` and `wtd_delta` / `wtd_p_value`).

No multiple-comparison correction is applied. With many contrasts reported, individual p-values near
0.05 should be treated as weak evidence.

### 4.4 Reference baselines

**Code:** [`04_evaluate.py`](../04_evaluate.py) — `build_systems()`; `production` and `random` are rebuilt per query inside the evaluation loop

| Baseline | Definition | Purpose |
| --- | --- | --- |
| `random` | Seeded shuffle | Floor |
| `production` | Rank by mean observed impression position | Incumbent ordering proxy |

`production` is **not** the LTR model — the ranker was never queried. It reflects whatever the live
stack did, as revealed by where products were shown. W4 shows why it must not be compared against
directly.

---

## 5. Experiment 1 — Does the photograph beat the title?

### 5.1 Design

**Code:** [`03_embed.py`](../03_embed.py) — `encode_siglip()` (both towers, shared space), `encode_jina()` (`retrieval.query` / `retrieval.document` prompts)

The naive comparison — a CLIP-family image tower against a separate text retriever — changes the
encoder *and* the modality at once, and cannot attribute a difference to either. Almost every
comparison of this kind in the wild is confounded this way.

We exploit a structural property of dual-tower vision-language models: both towers project into a
**shared** embedding space. So the same model can represent a product by its photograph *or* by its
title, with the query encoder held completely fixed.

| System | Query encoder | Product representation | Role |
| --- | --- | --- | --- |
| `image` | SigLIP text tower | SigLIP **image** tower over the photograph | Treatment |
| `text-siglip` | SigLIP text tower | SigLIP **text** tower over the title | **Control** |
| `text-jina` | Jina `retrieval.query` | Jina `retrieval.document` over the title | External comparator |

`text-siglip` is the contrast that actually tests the hypothesis. `text-jina` tests whether
encoder choice matters independently.

**Models.** `google/siglip-base-patch16-512` (203M params, 768-d) and
`jinaai/jina-embeddings-v5-text-nano` (239M params, 768-d). Comparable capacity, identical output
dimensionality, so neither has a structural advantage. `jinaai/jina-embeddings-v5-text-nano` is used
with its documented
`retrieval.query` / `retrieval.document` prompts.

### 5.1b Experiment 1b — structured attributes instead of the title

**Code:** [`01b_fetch_attributes.py`](../01b_fetch_attributes.py) → `product_attributes.csv` · [`03_embed.py`](../03_embed.py) (`attr_emb` in both `.npz` files)

A product title is marketing copy: brand, model name, technology trademarks, size hints, and the
product type buried at the end. The catalogue also carries a curated structured description — the
**"Big-4"** attributes the LTR pipeline already uses as features. The question is whether that
structured text is a better retrieval key than the title.

Fields, taken from `prod_ml_feature_store_db.products.ecode_attribute` using the same attribute ids
as `ds-ecm-search-ranking-ltr`:

| Field | `attr_id` | `attr_name` | Example |
| --- | --- | --- | --- |
| `brand` | `X_BRAND` | Brand | `Nike` |
| `product_type` | `5382` | Product Type | `Cleats` |
| `product_activity` | `4285` | Activity | `Soccer` |
| `gender_by_age` | `2101` | Gender by Age | `Adult` |

The four values are concatenated in that fixed order, space-separated, empties dropped:
`"Nike Cleats Soccer Adult"`. Nothing else is added — no title, no description. Coverage on the test
set is **100%** (19,585/19,585); the per-field minimum is 99.96%.

Both encoders are run over this text, giving `attr-siglip` and `attr-jina` alongside the existing
title arms. Everything else — queries, pools, labels, metrics — is unchanged, so the title/attribute
contrast is clean within each encoder.

**A third encoder is added as a capacity control.** `jinaai/jina-embeddings-v5-text-small` is the
next size tier up from the nano used elsewhere. It runs over *both* the title and the attribute text,
giving `text-jina-small` and `attr-jina-small`, so encoder capacity can be varied while holding the
representation fixed:

| Encoder | Params | Output dim |
| --- | --- | --- |
| `google/siglip-base-patch16-512` (text tower) | 203M | 768 |
| `jinaai/jina-embeddings-v5-text-nano` | 239M | 768 |
| `jinaai/jina-embeddings-v5-text-small` | **677M** | **1024** |

Note the small tier is **not a clean capacity control**: it changes parameter count *and* output
dimensionality together, so the two cannot be separated in this arm. The v5 text family publishes
only these two size tiers — there is no smaller variant below nano.

**The attribute text is far less discriminative than the title**, which makes the result in §5.3
harder to explain away:

| Property | Title | Big-4 attributes |
| --- | --- | --- |
| Mean length (words) | 7.0 | 5.5 |
| **Distinct strings** | 18,541 (**94.7%**) | 5,416 (**27.7%**) |

257 products share `Nike Cleats Soccer Adult`. The attribute representation cannot separate products
*within* a category at all.

### 5.2 Results

**Code:** [`04_evaluate.py`](../04_evaluate.py) → `results/summary.csv` (both `weighting` rows), `results/significance.csv`

All 12 systems, all four metric families, all six cutoffs. `production` is marked `*` and is **not
comparable** (W4). MAP has no cutoff and is carried on the NDCG tables.

**What `fusion` is.** It is the only composite system in the tables, defined in
[`04_evaluate.py`](../04_evaluate.py) as the equal-weight mean of two per-query z-scored similarity
vectors:

```python
sims["fusion"] = 0.5 * (zscore(sims["siglip_image"]) + zscore(sims["jina_text"]))
```

| Component | Query encoder | Product representation |
| --- | --- | --- |
| `siglip_image` | SigLIP text tower | SigLIP image tower over the **raw, uncropped** photograph |
| `jina_text` | `jinaai/jina-embeddings-v5-text-nano` | Same model over the product **title** |

Z-scoring is per query and is necessary because raw cosine scales differ between the two embedding
spaces; without it the larger-variance system dominates the sum.

> **`fusion` is built from two arms that are now known to be suboptimal.** It uses the *uncropped*
> image rather than `image+MGPL` (W2), and the *title* rather than the Big-4 attributes (§5.1b), which
> `attr-siglip` beats by +0.1295. It also spans two model families, so it pays both query encoders at
> request time — 7.2 ms plus 17.4 ms (W4 §3). Because `attr-siglip` and `siglip_image` share the same
> query encoder, an attribute-plus-image fusion would need only **one** query encode and would be
> strictly cheaper than the current one. Rebuilding it is an evaluate-only change — both embedding
> sets already exist — and is untested.

> **Name collision — W1 and W3 report different systems as `fusion`.** The W3 analysis
> ([`08_modality_analysis.py`](../08_modality_analysis.py)) builds its own fusion from
> `siglip_image_crop` + `jina_text`, giving **0.4643**; the tables here use `siglip_image` +
> `jina_text`, giving **0.4627**. The two differ only in whether the image arm is MGPL-cropped. Any
> cross-report comparison of a "fusion" number must state which one is meant.

#### Macro-averaged — every query counts once

**NDCG and MAP** — the only metrics that use the graded 0–4 scale:

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 | MAP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.6268 | 0.6176 | 0.6248 | 0.6841 | 0.7651 | 0.8045 | 0.7915 |
| `attr-siglip` | 0.5763 | 0.5466 | 0.5639 | 0.6526 | 0.7398 | 0.7806 | 0.8746 |
| `attr-jina-small` | 0.4937 | 0.4721 | 0.4955 | 0.5986 | 0.6991 | 0.7474 | 0.8696 |
| `fusion` | 0.4506 | 0.4627 | 0.5160 | 0.6303 | 0.7234 | 0.7609 | 0.9076 |
| `attr-jina` | 0.4606 | 0.4516 | 0.4932 | 0.5978 | 0.7003 | 0.7448 | 0.8735 |
| `image+MGPL` | 0.4221 | 0.4430 | 0.4950 | 0.6138 | 0.7105 | 0.7478 | 0.8957 |
| `image+naive` | 0.4233 | 0.4393 | 0.4967 | 0.6123 | 0.7100 | 0.7474 | 0.8958 |
| `image` | 0.4196 | 0.4363 | 0.4942 | 0.6130 | 0.7112 | 0.7478 | 0.8960 |
| `text-jina-small` | 0.4014 | 0.4229 | 0.4822 | 0.5922 | 0.6927 | 0.7356 | 0.8939 |
| `text-jina` | 0.3989 | 0.4171 | 0.4726 | 0.5860 | 0.6879 | 0.7321 | 0.8910 |
| `text-siglip` | 0.4151 | 0.4137 | 0.4535 | 0.5649 | 0.6751 | 0.7246 | 0.8503 |
| `random` | 0.2642 | 0.2775 | 0.3255 | 0.4552 | 0.5826 | 0.6443 | 0.7740 |

**Recall@k** — binary, converges for every system by k = 144:

| System | RECALL@5 | RECALL@10 | RECALL@20 | RECALL@48 | RECALL@96 | RECALL@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.0635 | 0.1326 | 0.2623 | 0.5550 | 0.8300 | 0.9517 |
| `attr-siglip` | 0.0773 | 0.1557 | 0.3037 | 0.6103 | 0.8521 | 0.9534 |
| `attr-jina-small` | 0.0769 | 0.1524 | 0.2991 | 0.6046 | 0.8478 | 0.9561 |
| `fusion` | 0.0832 | 0.1656 | 0.3207 | 0.6285 | 0.8627 | 0.9607 |
| `attr-jina` | 0.0776 | 0.1552 | 0.3034 | 0.6043 | 0.8530 | 0.9570 |
| `image+MGPL` | 0.0806 | 0.1612 | 0.3148 | 0.6223 | 0.8637 | 0.9593 |
| `image+naive` | 0.0813 | 0.1616 | 0.3152 | 0.6220 | 0.8624 | 0.9596 |
| `image` | 0.0811 | 0.1612 | 0.3148 | 0.6208 | 0.8640 | 0.9598 |
| `text-jina-small` | 0.0825 | 0.1645 | 0.3165 | 0.6182 | 0.8537 | 0.9573 |
| `text-jina` | 0.0823 | 0.1633 | 0.3148 | 0.6188 | 0.8491 | 0.9546 |
| `text-siglip` | 0.0785 | 0.1548 | 0.2972 | 0.5898 | 0.8409 | 0.9525 |
| `random` | 0.0671 | 0.1331 | 0.2617 | 0.5430 | 0.8121 | 0.9381 |

**Precision@k** — binary, falls with k because the denominator is k regardless of pool size:

| System | PRECISION@5 | PRECISION@10 | PRECISION@20 | PRECISION@48 | PRECISION@96 | PRECISION@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.7320 | 0.7653 | 0.7665 | 0.7323 | 0.6198 | 0.5081 |
| `attr-siglip` | 0.9060 | 0.9083 | 0.8957 | 0.8133 | 0.6418 | 0.5099 |
| `attr-jina-small` | 0.8940 | 0.8870 | 0.8728 | 0.8008 | 0.6384 | 0.5120 |
| `fusion` | 0.9400 | 0.9380 | 0.9283 | 0.8353 | 0.6520 | 0.5156 |
| `attr-jina` | 0.8893 | 0.8870 | 0.8802 | 0.8013 | 0.6426 | 0.5130 |
| `image+MGPL` | 0.9260 | 0.9227 | 0.9143 | 0.8287 | 0.6522 | 0.5144 |
| `image+naive` | 0.9267 | 0.9227 | 0.9158 | 0.8287 | 0.6510 | 0.5147 |
| `image` | 0.9293 | 0.9213 | 0.9140 | 0.8271 | 0.6526 | 0.5148 |
| `text-jina-small` | 0.9267 | 0.9230 | 0.9077 | 0.8189 | 0.6435 | 0.5128 |
| `text-jina` | 0.9207 | 0.9120 | 0.9018 | 0.8190 | 0.6395 | 0.5108 |
| `text-siglip` | 0.8940 | 0.8840 | 0.8672 | 0.7828 | 0.6305 | 0.5092 |
| `random` | 0.7667 | 0.7620 | 0.7593 | 0.7142 | 0.6018 | 0.4980 |

**MRR@k** — binary, flat from k = 20 onward:

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.7717 | 0.7784 | 0.7787 | 0.7787 | 0.7787 | 0.7787 |
| `attr-siglip` | 0.9433 | 0.9449 | 0.9449 | 0.9449 | 0.9449 | 0.9449 |
| `attr-jina-small` | 0.9277 | 0.9302 | 0.9305 | 0.9307 | 0.9307 | 0.9307 |
| `fusion` | 0.9789 | 0.9794 | 0.9794 | 0.9794 | 0.9794 | 0.9794 |
| `attr-jina` | 0.9269 | 0.9286 | 0.9293 | 0.9294 | 0.9294 | 0.9294 |
| `image+MGPL` | 0.9550 | 0.9555 | 0.9557 | 0.9557 | 0.9557 | 0.9557 |
| `image+naive` | 0.9573 | 0.9581 | 0.9581 | 0.9581 | 0.9581 | 0.9581 |
| `image` | 0.9625 | 0.9636 | 0.9639 | 0.9639 | 0.9639 | 0.9639 |
| `text-jina-small` | 0.9728 | 0.9728 | 0.9728 | 0.9732 | 0.9732 | 0.9732 |
| `text-jina` | 0.9568 | 0.9576 | 0.9578 | 0.9578 | 0.9578 | 0.9578 |
| `text-siglip` | 0.9427 | 0.9427 | 0.9427 | 0.9427 | 0.9427 | 0.9427 |
| `random` | 0.8810 | 0.8843 | 0.8843 | 0.8843 | 0.8843 | 0.8843 |

#### Impression-weighted — every query counts in proportion to its traffic

**NDCG and MAP:**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 | MAP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.6049 | 0.5974 | 0.6063 | 0.6631 | 0.7556 | 0.7976 | 0.7985 |
| `attr-siglip` | 0.5893 | 0.5630 | 0.5765 | 0.6492 | 0.7379 | 0.7841 | 0.8980 |
| `attr-jina-small` | 0.4827 | 0.4677 | 0.4890 | 0.5888 | 0.6859 | 0.7464 | 0.8924 |
| `fusion` | 0.4460 | 0.4651 | 0.5086 | 0.6173 | 0.7157 | 0.7603 | 0.9234 |
| `attr-jina` | 0.4349 | 0.4260 | 0.4815 | 0.5830 | 0.6866 | 0.7382 | 0.8954 |
| `image+MGPL` | 0.4473 | 0.4582 | 0.5039 | 0.6123 | 0.7120 | 0.7547 | 0.9127 |
| `image+naive` | 0.4366 | 0.4494 | 0.4993 | 0.6065 | 0.7075 | 0.7518 | 0.9130 |
| `image` | 0.4376 | 0.4492 | 0.4970 | 0.6069 | 0.7077 | 0.7521 | 0.9132 |
| `text-jina-small` | 0.3690 | 0.3970 | 0.4574 | 0.5704 | 0.6797 | 0.7283 | 0.9109 |
| `text-jina` | 0.3722 | 0.3946 | 0.4512 | 0.5640 | 0.6729 | 0.7238 | 0.9053 |
| `text-siglip` | 0.3971 | 0.3866 | 0.4210 | 0.5274 | 0.6528 | 0.7135 | 0.8610 |
| `random` | 0.2630 | 0.2795 | 0.3164 | 0.4438 | 0.5799 | 0.6471 | 0.7984 |

**Recall@k:**

| System | RECALL@5 | RECALL@10 | RECALL@20 | RECALL@48 | RECALL@96 | RECALL@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.0558 | 0.1152 | 0.2309 | 0.4945 | 0.7986 | 0.9395 |
| `attr-siglip` | 0.0704 | 0.1424 | 0.2809 | 0.5664 | 0.8226 | 0.9392 |
| `attr-jina-small` | 0.0692 | 0.1384 | 0.2753 | 0.5646 | 0.8202 | 0.9462 |
| `fusion` | 0.0742 | 0.1481 | 0.2907 | 0.5813 | 0.8308 | 0.9481 |
| `attr-jina` | 0.0696 | 0.1393 | 0.2781 | 0.5643 | 0.8241 | 0.9459 |
| `image+MGPL` | 0.0725 | 0.1447 | 0.2862 | 0.5741 | 0.8326 | 0.9480 |
| `image+naive` | 0.0726 | 0.1449 | 0.2867 | 0.5745 | 0.8327 | 0.9487 |
| `image` | 0.0725 | 0.1447 | 0.2860 | 0.5738 | 0.8334 | 0.9483 |
| `text-jina-small` | 0.0725 | 0.1455 | 0.2861 | 0.5735 | 0.8247 | 0.9446 |
| `text-jina` | 0.0721 | 0.1446 | 0.2848 | 0.5731 | 0.8162 | 0.9404 |
| `text-siglip` | 0.0685 | 0.1359 | 0.2670 | 0.5367 | 0.8073 | 0.9386 |
| `random` | 0.0599 | 0.1206 | 0.2384 | 0.5009 | 0.7856 | 0.9261 |

**Precision@k:**

| System | PRECISION@5 | PRECISION@10 | PRECISION@20 | PRECISION@48 | PRECISION@96 | PRECISION@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.7320 | 0.7524 | 0.7603 | 0.7397 | 0.6721 | 0.5682 |
| `attr-siglip` | 0.9306 | 0.9398 | 0.9325 | 0.8487 | 0.6978 | 0.5690 |
| `attr-jina-small` | 0.8994 | 0.9056 | 0.9029 | 0.8407 | 0.6956 | 0.5746 |
| `fusion` | 0.9568 | 0.9569 | 0.9488 | 0.8682 | 0.7059 | 0.5763 |
| `attr-jina` | 0.8946 | 0.9006 | 0.9063 | 0.8418 | 0.6991 | 0.5744 |
| `image+MGPL` | 0.9453 | 0.9412 | 0.9376 | 0.8596 | 0.7073 | 0.5761 |
| `image+naive` | 0.9408 | 0.9405 | 0.9399 | 0.8604 | 0.7074 | 0.5768 |
| `image` | 0.9432 | 0.9403 | 0.9375 | 0.8597 | 0.7083 | 0.5764 |
| `text-jina-small` | 0.9291 | 0.9266 | 0.9263 | 0.8550 | 0.6998 | 0.5730 |
| `text-jina` | 0.9132 | 0.9164 | 0.9200 | 0.8522 | 0.6913 | 0.5699 |
| `text-siglip` | 0.8875 | 0.8827 | 0.8801 | 0.8058 | 0.6834 | 0.5686 |
| `random` | 0.7737 | 0.7853 | 0.7829 | 0.7485 | 0.6586 | 0.5579 |

**MRR@k:**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production`* | 0.7876 | 0.7942 | 0.7943 | 0.7943 | 0.7943 | 0.7943 |
| `attr-siglip` | 0.9667 | 0.9676 | 0.9676 | 0.9676 | 0.9676 | 0.9676 |
| `attr-jina-small` | 0.9236 | 0.9246 | 0.9247 | 0.9248 | 0.9248 | 0.9248 |
| `fusion` | 0.9833 | 0.9835 | 0.9835 | 0.9835 | 0.9835 | 0.9835 |
| `attr-jina` | 0.9099 | 0.9109 | 0.9111 | 0.9112 | 0.9112 | 0.9112 |
| `image+MGPL` | 0.9724 | 0.9726 | 0.9727 | 0.9727 | 0.9727 | 0.9727 |
| `image+naive` | 0.9718 | 0.9722 | 0.9722 | 0.9722 | 0.9722 | 0.9722 |
| `image` | 0.9738 | 0.9751 | 0.9752 | 0.9752 | 0.9752 | 0.9752 |
| `text-jina-small` | 0.9704 | 0.9704 | 0.9704 | 0.9709 | 0.9709 | 0.9709 |
| `text-jina` | 0.9603 | 0.9616 | 0.9617 | 0.9617 | 0.9617 | 0.9617 |
| `text-siglip` | 0.9631 | 0.9631 | 0.9631 | 0.9631 | 0.9631 | 0.9631 |
| `random` | 0.9310 | 0.9335 | 0.9335 | 0.9335 | 0.9335 | 0.9335 |

System ordering is stable across all six cutoffs and both weightings. Both text systems lose ground
on high-traffic queries while `image` gains, which is what widens the modality gap from +0.0226 to
+0.0627. `attr-siglip` gains under weighting too, and is the only embedding system that does not fall
far behind `production`.

**Note on which metric separates.** `attr-siglip` leads on **NDCG at every $k$**, but not on Recall@k,
Precision@k, MRR@k or MAP, where `fusion` and the image arms are higher. That is the saturation
described in §4.2 and §4.2c: those metrics binarize the label at grade > 0 and sit near their
ceilings, so they rank systems by how many *ever-clicked* products are retrieved, not by how well the
graded order is reproduced. Only NDCG uses the grades. Bootstrap contrasts below are computed on
NDCG@10 only.

Paired-bootstrap contrasts:

| Contrast | Macro Δ | p | **Weighted Δ** | **p** |
| --- | --- | --- | --- | --- |
| **`attr-siglip` vs `text-siglip`** *(representation)* | **+0.1329** | **<0.001** | **+0.1764** | **<0.001** |
| `attr-siglip` vs `image` | +0.1103 | <0.001 | +0.1137 | <0.001 |
| `attr-siglip` vs `attr-jina` *(encoder)* | +0.0951 | <0.001 | +0.1370 | <0.001 |
| **`attr-siglip` vs `attr-jina-small`** *(encoder, 203M vs 677M)* | **+0.0745** | **<0.001** | **+0.0952** | **<0.001** |
| **`attr-siglip` vs `production`** | −0.0710 | <0.001 | **−0.0344** | **0.262** |
| `attr-jina-small` vs `text-jina-small` *(representation)* | +0.0493 | 0.007 | +0.0707 | 0.010 |
| `attr-jina` vs `text-jina` *(representation)* | +0.0345 | 0.032 | +0.0314 | 0.184 |
| `image` vs `text-siglip` *(modality)* | +0.0226 | 0.102 | **+0.0627** | **0.006** |
| **`attr-jina-small` vs `attr-jina`** *(capacity, attributes)* | +0.0206 | 0.083 | **+0.0418** | **0.015** |
| `image` vs `text-jina` | +0.0192 | 0.105 | +0.0547 | 0.003 |
| **`text-jina-small` vs `text-jina`** *(capacity, titles)* | **+0.0058** | **0.517** | **+0.0025** | **0.911** |
| `text-siglip` vs `text-jina` *(encoder)* | −0.0034 | 0.843 | −0.0080 | 0.719 |
| `random` vs `text-jina` | −0.1396 | <0.001 | −0.1150 | <0.001 |

### 5.3 Reading

#### Depth sensitivity — do the conclusions survive deeper cutoffs?

NDCG@10 is primary, but every contrast was recomputed at six cutoffs. The **system ordering is
identical at all six**; what changes is the spacing.

| k | `attr-siglip` | `text-siglip` | **Gap (macro)** | **Gap (weighted)** | `production` | `random` |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.5763 | 0.4151 | **+0.1612** | **+0.1922** | 0.6268 | 0.2642 |
| 10 | 0.5466 | 0.4137 | **+0.1329** | **+0.1764** | 0.6176 | 0.2775 |
| 20 | 0.5639 | 0.4535 | +0.1104 | +0.1555 | 0.6248 | 0.3255 |
| **48** | 0.6526 | 0.5649 | **+0.0876** | **+0.1218** | 0.6841 | 0.4552 |
| 96 | 0.7398 | 0.6751 | +0.0647 | +0.0851 | 0.7651 | 0.5826 |
| 144 | 0.7806 | 0.7246 | +0.0560 | +0.0706 | 0.8045 | 0.6443 |

The attribute advantage **shrinks monotonically with depth** but never disappears — +0.1612 at k = 5
down to +0.0560 at k = 144. This is expected: as k approaches the pool size every system converges on
the same candidate set, which is also why `random` climbs from 0.2642 to 0.6443. The compression is
an artefact of the cutoff, not evidence against the result.

The modality contrast behaves differently \u2014 it **peaks in the middle**:

| k | 5 | 10 | 20 | 48 | 96 | 144 |
| --- | --- | --- | --- | --- | --- | --- |
| `image` vs `text-siglip` (macro) | +0.0044 | +0.0226 | +0.0407 | **+0.0480** | +0.0361 | +0.0233 |
| same, impression-weighted | +0.0406 | +0.0627 | +0.0760 | **+0.0795** | +0.0549 | +0.0386 |

Images are *not* better at the very top of the ranking (+0.0044 at k = 5) but are better through the
middle of the list, peaking at k = 48. Had we reported only NDCG@5 the modality effect would have
looked like nothing at all. This is the strongest argument in the series for reporting more than one
cutoff.

**Structured attributes beat both the title and the photograph, by a wide margin.** `attr-siglip`
improves on the same-encoder title arm by **+0.1329 macro / +0.1764 weighted**, both p < 0.001, with a
64% per-query win rate. This is roughly **five times** the modality effect and the largest contrast
anywhere in the series. It also beats the best image arm by +0.11 under both weightings.

**It closes most of the gap to `production`.** Under impression weighting the difference is
**−0.0344 at p = 0.262** — not distinguishable. No other embedding system comes near, and `production`
retains the position-leakage advantage documented in W4, so its remaining lead is an overstatement.

**The win is not explained by more information.** The attribute text is *shorter* (5.5 words vs 7.0)
and vastly less unique (27.7% distinct vs 94.7%) than the title. It cannot discriminate within a
category — 257 products share `Nike Cleats Soccer Adult`. That it wins anyway indicates the LTR
judgement list rewards **category-level** matching far more than instance-level: getting the right
kind of product into the top 10 matters more than ordering within that kind. This is a property of
the label definition (§3.2), not of the encoders, and it is worth knowing independently of embeddings.

**Encoder choice matters here, though it did not for titles.** `attr-siglip` beats `attr-jina` by
+0.0951 (p < 0.001) while the two encoders were within 0.8% on titles (p = 0.843). The W1 conclusion
below must therefore be narrowed: encoder choice is irrelevant *for title-length prose*, not in
general. A plausible mechanism, untested, is that a short keyword string resembles SigLIP's
alt-text training distribution, whereas `jinaai/jina-embeddings-v5-text-nano` is tuned for longer
document retrieval. Note also
that `attr-jina` improves on `text-jina` by only +0.0345 (p = 0.032, and p = 0.184 weighted) — so most
of the attribute advantage is only realised by one of the two encoders.

**The modality contrast depends on the weighting.** Under macro averaging it is inconclusive:
+5.5%, interval crossing zero, 53% win rate. Under impression weighting it is significant: **+0.0627,
p = 0.006**. The honest summary is that images do not clearly beat titles on a *typical query*, but
do beat them on a *typical impression* — the effect is concentrated in high-traffic queries.

**Encoder choice does not matter for titles under either weighting.** Two independently developed text
encoders, different architectures, objectives and corpora, land within 0.8% of each other (p = 0.843
macro, p = 0.719 weighted), and a 2.8× capacity increase within one family adds nothing either
(p = 0.517). Any title-based difference observed later cannot be attributed to encoder quality.

**The harness has signal.** Every embedding system beats `random` by 30–50% at p < 0.001, so the
setup can detect real effects.

**Resolution estimate.** On the macro metric the CI half-width for the modality contrast is roughly
±0.026 NDCG, about ±6% relative. Effects below that are not measurable with 300 queries under equal
weighting. This sets expectations for every subsequent experiment and is the single most useful
output of the week.

---

## 6. What this hands to W2

Experiment 1 leaves an obvious question. The image system is fed a raw catalogue photograph, whole
frame, whatever the merchandising crop happened to be. Before concluding anything about the modality,
we should check whether the image *signal* is intact by the time it reaches the encoder.

W2 examines what is actually in those frames.

---

## Appendix — Reproduction

| Stage | Script | Output |
| --- | --- | --- |
| 1 | `01_build_test_set.py` | `test_set.csv`, `products.csv` |
| 2 | `02_download_images.py` | `images/`, `image_manifest.csv` |
| 3 | `03_embed.py` | `embeddings/{siglip,jina}.npz` |
| 4 | `04_evaluate.py` | `summary.csv`, `per_query_metrics.csv`, `significance.csv` |

SQL: `sql/judgement_list.sql`. Databricks access via `../dbx_sql.py`, which reuses the editor's OAuth
session — no credentials in code.

Key parameters, all overridable by environment variable: `LOOKBACK_DAYS=90`, `DAYS_BEFORE_TODAY=3`,
`N_QUERIES=300`, `TERM_POOL=6000`, `MIN_GROUP_SIZE=10`, `MAX_GROUP_SIZE=240`,
`IPW_CLIP_POSITION=48`, `DECAY_FLATNESS=0.18`, `DECAY_MIDPOINT=30`, `ALPHA=1.0`.

**Operational note.** Store `data/` outside cloud-synced folders. Running from OneDrive CloudStorage
caused a mid-run `TimeoutError` on dehydrated files and throttled image writes to roughly a third of
achievable throughput. `EMBEDDING_EVAL_DATA` overrides the location.
