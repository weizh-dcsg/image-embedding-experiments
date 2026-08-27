# Search Relevance Embedding Study — Evaluation Methodology

**Scope:** how systems are compared — task framing, metrics, weighting, significance testing, and reference baselines.

> **TL;DR**
> Every system re-ranks the same candidate pool for the same query, scored by cosine similarity. **NDCG@10 is the primary metric** and the only one that uses the graded 0–4 scale; the supporting metrics binarise the label and are saturated, so they should not be used to separate systems. Every result is reported under **two weightings** — macro (per query) and impression-weighted (per search) — because they disagree systematically. Significance is a paired bootstrap over queries.

**Source file:** `04_evaluate.py`

---

## 1. Task framing

*Implemented in `04_evaluate.py` — `main()` builds one candidate pool per `search_term`, shared by all systems.*

Every system ranks **the same candidate pool** for **the same query**. Pools are the products production already surfaced, so this measures **re-ranking quality**, not full-catalogue retrieval. We make no recall claims about the catalogue as a whole.

Scoring is cosine similarity between L2-normalised query and product vectors. Products are sorted descending, with ties broken stably.

---

## 2. Metrics

*Implemented in `04_evaluate.py` — `ndcg_at_k()`, `mrr_at_k()`, `recall_at_k()`, `precision_at_k()`, `average_precision()`, aggregated by `score_ranking()`.*

### The primary metric

**NDCG@k** with exponential gain. For a ranked relevance vector `r`:

```
DCG@k  = SUM over i=1..k of  ( 2^r(i) - 1 ) / log2(i + 1)

NDCG@k = DCG@k / IDCG@k
```

where `IDCG@k` is the same pool re-sorted descending and truncated to `k` — the best ordering achievable for that query.

Exponential gain matters here. With grades 0–4 the gains are:

| Grade | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Gain `2^r - 1` | 0 | 1 | 3 | 7 | 15 |

So a grade-4 product is worth **15x** a grade-1, which reflects how sharply engagement concentrates on a few products per query.

Reported at k = 5, 10, 20, 48, 96, 144. **NDCG@10 is primary.**

The three deep cutoffs check that conclusions are not an artefact of shallow truncation — 48 matches the IPW rank clip used in the label construction, and 96 / 144 probe beyond it. They must be read with pool sizes in mind, because a cutoff stops binding once it exceeds the candidate list:

| Cutoff | Queries with pool >= k |
| --- | --- |
| 10 | 300 / 300 (100%) |
| 20 | 296 / 300 (99%) |
| **48** | 251 / 300 (84%) |
| 96 | 165 / 300 (55%) |
| 144 | 89 / 300 (30%) |

For 70% of queries NDCG@144 *is* NDCG over the whole pool. **@48 is the most informative of the three.**

### Supporting metrics, and why they cannot separate systems

Supporting metrics are **MRR@10** (rank of first relevant), **MAP** (precision averaged at each relevant position), **Recall@k** and **Precision@k**.

**NDCG is the only metric that uses the graded scale.** The other four binarise the label at `grade > 0`, so a grade-1 product counts exactly the same as a grade-4. Since grade 0 means *never clicked*, these metrics ask "did this product ever receive a click?", not "how relevant is it?"

That distinction matters because the binarisation is heavily imbalanced:

| Property | Value |
| --- | --- |
| Mean pool size | 110.9 |
| Mean positives (grade > 0) | 79.8 |
| **Positive share of pool** | **76.4%** |

Two consequences follow.

**Precision@k is saturated by the base rate.** With 76.4% of the pool positive, `random` scores 0.762 at k = 10 by construction. Every embedding system lands at 0.88–0.94 and `production` at 0.765. The column is compressed into the top quarter of its range before any system does anything.

**Recall@k is capped by arithmetic.** With about 80 positives and only 10 slots, the maximum attainable recall@10 averages **0.174**. Observed values cluster at 0.15–0.17, so every system sits against the ceiling and the differences reflect pool size, not ranking quality.

### A diagnostic that falls out of this

`production` scores 0.7653 precision@10 — statistically indistinguishable from `random` at 0.7620 — yet has by far the highest NDCG@10, 0.6176 against 0.4363 for `image`.

So `production` is **no better than chance** at getting clicked products into the top 10, but much better at ordering them *by grade*. Since grade derives from click-through rate and `production` ranks by mean observed position, this is position leakage surfacing through a second, independent route. The embedding systems show the opposite profile: far better binary retrieval (0.92), lower graded score.

> This is a compact reason to read NDCG@10 only for embedding-vs-embedding contrasts, and to keep `production` out of them entirely.

A graded variant of the supporting metrics — defining "positive" as grade 3 or higher rather than above 0 — would make Precision@k informative. It is not implemented; the binary definitions are the IR convention and are kept for comparability.

### How each metric behaves with depth

All four families are computed at every cutoff. Only NDCG stays informative across the range:

| Metric | Behaviour as k grows | Usable range |
| --- | --- | --- |
| **NDCG@k** | Rises toward 1 as k approaches pool size; system order stable throughout | All k, best <= 48 |
| **Recall@k** | Reaches 0.93-0.96 for *every* system including `random` by k = 144 | <= 20 |
| **Precision@k** | *Falls* with k — the denominator is k even when the pool is smaller | <= 20 |
| **MRR@k** | Flat from k = 20 onward, to four decimal places | <= 10 |

MRR saturates because it needs only the *first* relevant item and 76.4% of each pool is graded above 0. Recall converges because at depth 144 every system has retrieved almost all positives regardless of order. Precision declines because it divides by k unconditionally. None of these is a defect in the systems; all three are properties of a saturated binary label on a small pool.

---

## 3. Two weightings, two questions

*Implemented in `04_evaluate.py` — `query_impressions` per row, `weighted_mean()`, and a `weighting` column in `summary.csv`.*

Queries are **sampled** by impression volume but, under a plain average, **evaluated** with equal weight. A query with 264k impressions counts the same as one with 9.6M. That asymmetry is not neutral, so every metric is reported under both weightings:

| Weighting | Question it answers |
| --- | --- |
| **Macro** (mean over queries) | How well does this rank a *typical query*? IR convention. |
| **Impression-weighted** | How well does this rank a *typical search impression*? Closer to business impact. |

Volume across the 300 selected queries spans 264,103 to 9,573,478 — a 36x range. The top 10 queries hold 22.4% of impressions and the top 50 hold 52.4%, so impression weighting reduces effective sample size from 300 to **108.6**.

### They disagree, and the disagreement is systematic

Image systems gain under traffic weighting while text systems lose:

| System | Macro | Impression-weighted | Change |
| --- | --- | --- | --- |
| `image+MGPL` | 0.4430 | 0.4582 | **+0.0152** |
| `image+naive` | 0.4393 | 0.4494 | +0.0101 |
| `image` | 0.4363 | 0.4492 | +0.0129 |
| `jina_text` | 0.4171 | 0.3946 | **-0.0225** |
| `text-siglip` | 0.4137 | 0.3866 | **-0.0271** |
| `fusion` | 0.4627 | 0.4651 | +0.0024 |
| `production` | 0.6176 | 0.5974 | -0.0202 |
| `random` | 0.2775 | 0.2795 | +0.0020 |

Images are relatively **better on the highest-traffic queries**; text is relatively better on the lower-volume tail of our sample. This is consistent with the query-type analysis reported separately, where text wins on proper nouns (`nike sabrina 3`, `ja 3` — lower volume) and images win on jersey and visual categories (higher volume).

> **Neither weighting is "the right one."** Reporting only the macro average understates the modality effect where the traffic actually is. Reporting only the weighted average rests on about 109 effective queries. Both are given throughout, and any claim that survives only one of them is flagged as such.

---

## 4. Significance testing

*Implemented in `04_evaluate.py` — paired bootstrap (`boot`) and weighted bootstrap (`boot_w`), written to `results/significance.csv`.*

Per-query NDCG variance greatly exceeds between-system variance — some queries are simply easier — so unpaired tests would be badly underpowered. We use a **paired bootstrap over queries**:

1. Compute the per-query metric for both systems.
2. Form per-query differences. Pairing controls for query difficulty.
3. Resample the 300 differences with replacement, 2,000 times.
4. Report mean difference, 2.5 / 97.5 percentile CI, two-sided p-value, and per-query win rate.

Seed is fixed at 20260808.

The same bootstrap is run under impression weighting, resampling queries and recomputing the weighted mean difference on each draw. Both sets of intervals are written to `significance.csv`:

| Column | Meaning |
| --- | --- |
| `delta`, `ci_low`, `ci_high`, `p_value` | Macro weighting |
| `wtd_delta`, `wtd_ci_low`, `wtd_ci_high`, `wtd_p_value` | Impression weighting |

> **Two caveats on reading p-values from this harness**
> Bootstrap p-values carry roughly 0.005–0.05 jitter between runs when the set of systems changes, because the RNG stream shifts. Distinctions like "p = 0.057 versus p = 0.033" should not be over-interpreted.
> No multiple-comparison correction is applied. With many contrasts reported, individual p-values near 0.05 should be treated as weak evidence.

---

## 5. Reference baselines

*Implemented in `04_evaluate.py` — `build_systems()`. Both baselines are rebuilt per query inside the evaluation loop.*

| Baseline | Definition | Purpose |
| --- | --- | --- |
| `random` | Seeded shuffle | Floor |
| `production` | Rank by mean observed impression position | Incumbent ordering proxy |

> **`production` is not the LTR model.** The ranker was never queried. It reflects whatever the live stack did, as revealed by where products were shown. It must not be compared against directly — see the evaluation-validity report for the position-leakage analysis, and the diagnostic in section 2 above.

---

# Summary — how to read results from this harness

| Rule | Reason |
| --- | --- |
| Use **NDCG@10** as the headline. | Only metric using the graded 0–4 scale. |
| Ignore Precision@k and Recall@k for system ranking. | Saturated at a 76.4% base rate and capped at 0.174 respectively. |
| Always check **both weightings**. | They disagree systematically; image gains and text loses under traffic weighting. |
| Do not compare anything to `production`. | Position leakage inflates its graded score while its binary retrieval is at chance. |
| Treat p-values near 0.05 as weak. | Bootstrap jitter plus no multiple-comparison correction. |
