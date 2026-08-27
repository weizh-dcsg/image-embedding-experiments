# W4 — Evaluation Validity and Systems Cost

**Report 4 of 4** · DS eCom Search Ranking · 2026
Scope: audit whether the judgement list measures what it claims to, and characterise what these
systems cost to serve and index.

---

## Summary

Three weeks of results rest on one judgement list. This week we audited it, and found a defect worth
reporting independently of anything about embeddings.

The production judgement list applies inverse-propensity weights to **impressions and clicks alike**.
This re-weights which observations count but does not rescale the CTR level, and consequently leaves
a **−0.29 correlation between impression position and relevance grade**. A click-only correction on
the same data leaves −0.01. Relevance grades 1 through 4 are cleanly monotone in position: grade 4
products sit at mean rank 16.0, grade 1 at 29.3.

This explains the `production` baseline's 0.6176 — it ranks by position, and position substantially
determines the labels, so it is predicting the dominant cause of its own target. Comparisons **among**
embedding systems remain valid, since none observes position. Comparisons **against** `production`
are circular and have been excluded throughout.

On cost: query-time latency is **entirely text encoding**, 7.2 ms (`google/siglip-base-patch16-512`)
or 17.4 ms (`jinaai/jina-embeddings-v5-text-nano`) at p50.
Image encoding never happens at query time. Brute-force scoring of all 149,801 products takes 3.1 ms,
so no approximate index is needed at this catalogue size. Indexing costs 0.07 GPU-hours for text,
1.98 for images, and 10.2 more if product-grounded cropping is applied.

---

## 1. The `production` anomaly

**Code:** [`04_evaluate.py`](../04_evaluate.py) — `production` ordering built per query from `mean_position`

`production` has carried a caveat since W1. It ranks each query's candidates by the mean position at
which the live site impressed them — a proxy for the incumbent ordering, not the LTR model itself
(the ranker was never queried).

It scores **0.6176** against the best embedding system's 0.4430, a 39% gap. Taken at face value this
says the deployed stack massively outperforms every embedding approach. That reading is wrong, and
the reason generalises beyond this study.

---

## 2. Auditing the judgement list

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) · [`config.py`](../config.py) · `test_set.csv` (`mean_position`, `relevance`)

### 2.1 Residual position correlation

If a label set were properly position-debiased, a ranker encoding nothing but position should score
near random.

| Label set | corr(mean position, grade) |
| --- | --- |
| Production LTR judgement list | **−0.291** |
| Raw CTR (undebiased control) | −0.270 |

The debiased labels are **barely less position-correlated than raw CTR**. Grade-by-position:

| LTR grade | Mean impression position |
| --- | --- |
| 4 | **16.0** |
| 3 | 21.9 |
| 2 | 26.3 |
| 1 | 29.3 |
| 0 | 28.3 |

Grades 1–4 are monotone. A product ranked highly was clicked more *because* it was ranked highly, and
the grade is built from those clicks.

### 2.2 Why IPW did not remove it

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) — `ipw` and `weights` CTEs; the weight multiplies impressions *and* clicks

The construction (W1 §3.2) computes:

$$\text{wCTR} = \frac{\sum \text{click} \cdot w}{\sum \text{impression} \cdot w}, \qquad w = \tau_k \cdot w_{\text{decay}}$$

with $w$ applied to **both** numerator and denominator.

This does not perform the correction the propensity model implies. Under the examination hypothesis:

$$P(\text{click}) = P(\text{examine} \mid k) \cdot P(\text{relevant})$$

so an unbiased relevance estimate divides the *click rate* by the examination probability:

$$\widehat{P(\text{relevant})} = \frac{\sum \text{click} \cdot \tau_k}{\sum \text{impression}}$$

Weighting both sides instead produces a **re-weighted average CTR** — it changes which observations
dominate, but the position-to-relevance relationship survives largely intact. Empirically: −0.29
versus −0.01 for the click-only form on identical data.

We flag this as a finding about the **labelling pipeline**, not about embeddings. If the judgement
list is intended to be position-debiased, it currently is not, and any model comparison run against
it inherits the residual bias.

### 2.3 Consequences

| Comparison | Valid? | Why |
| --- | --- | --- |
| Embedding vs embedding | **Yes** | No embedding system observes position |
| Embedding vs `production` | **No** | Only one side has label leakage |
| Embedding vs `random` | Yes | Neither observes position |

All conclusions in W1–W3 are of the first and third kinds. `production` appears in tables for context
and is excluded from every claim.

### 2.4 Proposed diagnostic

Report **corr(position, grade)** for any click-derived judgement list, as a standing requirement. It
is a one-line computation and distinguishes a debiased label set from one that merely applies a
debiasing-shaped transformation. Suggested reading: |ρ| < 0.05 debiased; 0.05–0.15 partial;
**> 0.15 not debiased** — this list is at 0.29.

---

## 3. Serving cost

**Code:** [`07_benchmark_latency.py`](../07_benchmark_latency.py) — `mode="online"`, `time_it()` → `results/latency.csv`

Measured on Apple M-series, MPS backend, PyTorch 2.13, 30 timed calls after 3 warm-ups.

| Component | Stage | Mode | p50 | p95 |
| --- | --- | --- | --- | --- |
| SigLIP text tower | query encode | online (bs=1) | **7.17 ms** | 7.54 ms |
| `jinaai/jina-embeddings-v5-text-nano` | query encode | online (bs=1) | **17.18 ms** | 19.06 ms |
| SigLIP image tower | image encode | online (bs=1) | 56.82 ms | 57.31 ms |
| DETR | object detect | online (bs=1) | 60.26 ms | 60.83 ms |
| Cosine + top-10 | 111-item pool | online | **0.004 ms** | 0.004 ms |
| Cosine + top-10 | 10,000 items | online | 0.21 ms | 0.27 ms |
| Cosine + top-10 | 149,801 items | online | **3.08 ms** | 3.26 ms |

Three observations:

1. **Query-time cost is entirely text encoding.** Product vectors are precomputed; the image tower
   never runs at query time. Adding image embeddings costs nothing at serving time.
2. **Scoring is free.** Re-ranking a 111-product pool takes 4 microseconds. Even a brute-force scan of
   the entire catalogue is 3.1 ms — **no ANN index is required at this scale**, which removes a large
   chunk of anticipated infrastructure.
3. **SigLIP's text tower is 2.4× faster than `jinaai/jina-embeddings-v5-text-nano`** (7.2 vs 17.4 ms), because it caps at 64 tokens
   with a smaller vocabulary. Since W1 showed the two are statistically indistinguishable on
   relevance, **latency and licensing are the rational bases for choosing between them** — not
   quality.

---

## 4. Indexing cost

**Code:** [`07_benchmark_latency.py`](../07_benchmark_latency.py) — `mode="offline"` → `results/latency_summary.json`

| Stage | Throughput | Full catalogue (149,801) |
| --- | --- | --- |
| Text embeddings | 591 items/s | **0.07 h** |
| Image embeddings | 21 items/s | **1.98 h** |
| MGPL, OWLv2 | 4 items/s | **10.2 h** |
| MGPL, OWL-ViT v1 | 26 items/s | 1.58 h |

Text indexing is effectively free. Image indexing is ~2 GPU-hours on laptop-class hardware, minutes
on a server GPU. Product-grounded cropping with OWLv2 costs **five times the embeddings it feeds** —
which, against W2's +0.9% unverified benefit, does not justify catalogue-wide deployment.

**Index footprint.** 768-d float32 = 3,072 bytes per product per representation → **439 MB** per
representation for the full catalogue. Fusion needs two.

**Throughput caveat.** Observed end-to-end rates fell well below model rates (6.6 img/s against a
17 img/s model rate) when the data directory sat in cloud-synced storage. The bottleneck was file
I/O, not compute. Production pipelines should write to local or object storage.

---

## 5. Consolidated recommendations

**Code:** [`sql/judgement_list.sql`](../sql/judgement_list.sql) · [`config.py`](../config.py) — the three defects below are all parameter or CTE changes

### 5.1 Three judgement-list defects

Auditing the label pipeline surfaced three independent configuration problems. None is about
embeddings; all affect any model comparison run against this judgement list.

| # | Defect | Evidence | Effect |
| --- | --- | --- | --- |
| 1 | **IPW applied to impressions *and* clicks** | residual position correlation **−0.29** vs −0.01 for click-only (§2) | Labels remain substantially position-determined; invalidates comparison against position-derived baselines |
| 2 | **Window and decay disagree** | `num_days_train = 90` but `decay_midpoint = 30`; days 60–90 carry **0.09%** of weight, 95% sits in 36 days (W1 §3.2) | A third of the expensive 90-day scan contributes ~1% of signal; effective sample size is 41.4% of a flat window |
| 3 | **Quantile smoothing is inert** | global prior weight $\alpha/(n+\alpha) \approx$ **0.0002%** because $n$ is a sum of propensity-inflated impressions (W1 §3.2) | The mechanism intended to stabilise low-traffic queries never activates, at any plausible $\alpha$ |

Suggested fixes, in priority order: divide by propensity on the click only; reconcile window length
with decay midpoint (either ~45 days or a larger midpoint); and rescale $\alpha$ to the units $n$ is
actually measured in, or switch $n$ to a product/click count.

### 5.2 Recommendations across the series

Across four weeks:

| # | Recommendation | Evidence |
| --- | --- | --- |
| 1 | **Adopt fusion over text**; fusion vs image alone is not separable | +10.9% over text (p<0.001) both weightings; vs image +0.0069, p=0.559 weighted (W3 §3.1) |
| 2 | **Do not fund catalogue-wide product-grounded cropping** on relevance grounds | +0.9%, p = 0.391 (W2); 10.2 GPU-h cost |
| 3 | **Do not deploy confidence-based modality routing** | 42.3% accuracy, below chance (W3) |
| 4 | **Choose text encoder on latency and licence, not relevance** | −0.8%, p = 0.843 (W1); 7.2 vs 17.4 ms |
| 5 | **Fix the judgement-list defects** before further model comparison | §5.1 — three independent issues |
| 6 | **Skip ANN infrastructure** at current catalogue size | 3.1 ms brute-force scan (W4) |
| 7 | **Treat modality routing as research** | oracle +16.3% unclaimed (W3) |
| 8 | **Report macro *and* impression-weighted metrics** | the two disagree systematically; images gain +0.015 and text loses −0.023 under traffic weighting (W1 §4.2b) |

### Licensing

`jinaai/jina-embeddings-v5-text-nano` is **CC BY-NC 4.0**. Commercial use requires a licence from Jina AI.
`jinaai/jina-embeddings-v5-text-small` carries the **same CC BY-NC 4.0** terms (verified from the
model card), so moving up a size tier does not resolve the restriction.
This must be resolved before any production consideration. `google/siglip-base-patch16-512` is
Apache 2.0.

---

## 6. Limitations across the series

1. **Single retailer, single vertical.** Person-contamination rates and modality splits will differ
   in categories with less on-model photography.
2. **Head queries only.** 300 high-volume terms surviving LTR group-size filters. Tail and
   attribute-heavy queries — where images should help most — are absent, so modality effects are
   likely measured at their weakest.
3. **Underpowered for small effects.** Resolution is roughly ±6% relative NDCG@10 (W1 §5.3). The
   localization result sits far below this.
4. **Click-derived labels with residual position bias** (this report).
5. **Re-ranking, not retrieval.** No claim about full-catalogue recall.
6. **One image per product.** No multi-view or alternate-angle imagery.
7. **No multiple-comparison correction** across the contrasts reported.
8. **Metric weighting changes conclusions.** Macro and impression-weighted averages disagree
   systematically (W1 §4.2b). Both are now reported, but any single-weighting result should be
   treated as partial.

---

## 7. Suggested next work

1. **Group-size sensitivity** — *in progress.* Re-run at `max_group_size` ∈ {1000, 2000}. The 240 cap
   discards 27.5% of candidate queries, all from the high-volume end, and was inherited from LTR
   training economics rather than evaluation need. Larger pools are a harder test and may recover the
   resolution that W2's negative result lacked. Reported in W5.
2. **Tail and attribute-heavy queries** — the highest-value open question, and the condition most
   favourable to images.
3. **Learned modality router** — supervised on query features, targeting the 0.5151 oracle.
4. **Learned fusion weights** — equal weighting is arbitrary and already captures 29.5% of headroom.
5. **Fix and re-run the judgement list** per §5.1, then re-validate all conclusions.
6. **Category-conditional cropping** — apply MGPL only to hard goods and jerseys, where W2 shows it
   helps.

---

## Appendix — Reproduction

```bash
python 07_benchmark_latency.py    # latency, throughput, index sizing
python 08_modality_analysis.py    # complementarity and routing
```

Outputs `results/latency.csv` and `results/latency_summary.json`. Position-correlation diagnostics
are computed directly from `test_set_encoded.csv`.
