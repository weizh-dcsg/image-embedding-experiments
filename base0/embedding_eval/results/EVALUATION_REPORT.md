# Image vs Text Embeddings for Search Relevance

Generated 2026-08-09

- Image model: `google/siglip-base-patch16-512`
- Text model: `jinaai/jina-embeddings-v5-text-nano`

## Verdict

**No statistically significant difference between the image and text systems.** Neither modality can be declared better on this test set at the current sample size.

- **SigLIP image vs Jina v5 text**: NDCG@10 0.4363 vs 0.4171 — +0.0192 (+4.6%), 95% CI [-0.0035, +0.0432], p = 0.1050, win rate 55% — not significant
- **SigLIP image vs SigLIP text** (same encoder, modality is the only change): NDCG@10 0.4363 vs 0.4137 — +0.0226 (+5.5%), 95% CI [-0.0038, +0.0491], p = 0.1020, win rate 53% — not significant

## What was compared

Every system ranks the **same candidate pool** for the **same queries**; only the product
representation changes.

| System | Query encoder | Product representation |
| --- | --- | --- |
| SigLIP image | SigLIP text tower | SigLIP image tower over the product photo |
| SigLIP text | SigLIP text tower | SigLIP text tower over the product title |
| Jina v5 text nano | Jina `retrieval.query` | Jina `retrieval.document` over the product title |
| Fusion | both | z-scored mean of SigLIP-image and Jina-text similarity |
| Production | n/a | current on-site ranking (mean observed impression position) |
| Random | n/a | seeded shuffle, floor reference |

**SigLIP text is the control that makes this experiment interpretable.** Comparing SigLIP image
to Jina text changes two things at once (encoder *and* modality). SigLIP text holds the encoder
fixed and changes only the modality, which is the contrast that actually tests the hypothesis.

## Test data

Built from ML clickstream events (`prod_ent_silver_db.sdsc.ml_events`), SRLP page 0,
banner `DSG`, channel `WEB`, over a 90-day window.

- Queries: **300** (highest-volume head terms)
- Query-product pairs: **33279**
- Unique active products: **19468**
- Mean candidate pool: **110.9** products per query

Candidates are restricted to **active products**: DSG web-active in
`entdata.web.dim_sku_bod_web_active` (`web_chain_code = 'DSG'`), joined to
`prod_ml_feature_store_db.products.ecode` with `dsg_web_active = 'Y'`, and required to have both
a product title and a default image URL. This follows the active-product pattern used in
`ds-ecm-search-ranking-ltr`.

### Relevance labels

Two label sets are scored so the conclusion can be checked against the labelling choice.

1. **Position-debiased CTR** (primary). Clicks are divided by an examination propensity estimated
   from the global click-rate-by-rank curve, mirroring the inverse-propensity weighting in
   `ds-ecm-search-ranking-ltr/sandbox/ltr_vanilla`. Grades 0-3 come from the within-query
   percentile of the debiased CTR.
2. **Raw CTR**. Identical grading on undebiased CTR, so rank position still carries signal.

Debiasing works as intended: the correlation between mean impression position and the label drops
from about -0.46 (raw) to about -0.01 (debiased). A side effect is that the `production` baseline
has almost no signal left on the primary label set — by construction, not by failure.

## Results

### Primary: ltr judgement list (ipw + time decay) labels

| system | n_queries | ndcg@5 | ndcg@10 | ndcg@20 | mrr@10 | map | recall@10 | precision@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Production on-site ranking | 300 | 0.6268 | 0.6176 | 0.6248 | 0.7784 | 0.7915 | 0.1326 | 0.7320 |
| Fusion (SigLIP image + Jina text) | 300 | 0.4506 | 0.4627 | 0.5160 | 0.9794 | 0.9076 | 0.1656 | 0.9400 |
| SigLIP image (MGPL crop) | 300 | 0.4221 | 0.4430 | 0.4950 | 0.9555 | 0.8957 | 0.1612 | 0.9260 |
| SigLIP image (largest-area crop) | 300 | 0.4233 | 0.4393 | 0.4967 | 0.9581 | 0.8958 | 0.1616 | 0.9267 |
| SigLIP image (product photo) | 300 | 0.4196 | 0.4363 | 0.4942 | 0.9636 | 0.8960 | 0.1612 | 0.9293 |
| Jina v5 text nano (product title) | 300 | 0.3989 | 0.4171 | 0.4726 | 0.9576 | 0.8910 | 0.1633 | 0.9207 |
| SigLIP text (product title) | 300 | 0.4151 | 0.4137 | 0.4535 | 0.9427 | 0.8503 | 0.1548 | 0.8940 |
| Random | 300 | 0.2642 | 0.2775 | 0.3255 | 0.8843 | 0.7740 | 0.1331 | 0.7667 |

### Robustness: raw CTR labels

| system | n_queries | ndcg@5 | ndcg@10 | ndcg@20 | mrr@10 | map | recall@10 | precision@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Production on-site ranking | 300 | 0.6101 | 0.6112 | 0.6182 | 0.7767 | 0.7512 | 0.1488 | 0.7180 |
| Fusion (SigLIP image + Jina text) | 300 | 0.4314 | 0.4540 | 0.5090 | 0.9453 | 0.8583 | 0.1747 | 0.8933 |
| SigLIP image (MGPL crop) | 300 | 0.4091 | 0.4354 | 0.4918 | 0.9263 | 0.8459 | 0.1717 | 0.8800 |
| SigLIP image (largest-area crop) | 300 | 0.4058 | 0.4307 | 0.4915 | 0.9179 | 0.8454 | 0.1716 | 0.8773 |
| SigLIP image (product photo) | 300 | 0.4077 | 0.4300 | 0.4919 | 0.9298 | 0.8464 | 0.1712 | 0.8827 |
| Jina v5 text nano (product title) | 300 | 0.3854 | 0.4070 | 0.4644 | 0.9131 | 0.8335 | 0.1695 | 0.8647 |
| SigLIP text (product title) | 300 | 0.3852 | 0.3942 | 0.4390 | 0.9088 | 0.7876 | 0.1619 | 0.8373 |
| Random | 300 | 0.2456 | 0.2636 | 0.3111 | 0.8404 | 0.6954 | 0.1330 | 0.6860 |

![NDCG@10 by system](fig_ndcg_by_system.png)

![Metric curves](fig_metric_curves.png)

## Significance

Paired bootstrap over queries (2000 resamples), metric NDCG@10,
ltr judgement list (ipw + time decay) labels.

| system | baseline | delta | rel_delta_pct | ci_low | ci_high | p_value | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fusion (SigLIP image + Jina text) | Jina v5 text nano (product title) | 0.0456 | 10.9345 | 0.0296 | 0.0619 | 0.0000 | 0.6433 |
| Production on-site ranking | Jina v5 text nano (product title) | 0.2005 | 48.0790 | 0.1615 | 0.2382 | 0.0000 | 0.7667 |
| Random | Jina v5 text nano (product title) | -0.1396 | -33.4683 | -0.1684 | -0.1126 | 0.0000 | 0.2967 |
| SigLIP image (product photo) | Jina v5 text nano (product title) | 0.0192 | 4.6000 | -0.0035 | 0.0432 | 0.1050 | 0.5467 |
| SigLIP image (MGPL crop) | Jina v5 text nano (product title) | 0.0259 | 6.2195 | 0.0015 | 0.0492 | 0.0390 | 0.5700 |
| SigLIP image (largest-area crop) | Jina v5 text nano (product title) | 0.0222 | 5.3106 | -0.0001 | 0.0473 | 0.0520 | 0.5700 |
| SigLIP text (product title) | Jina v5 text nano (product title) | -0.0034 | -0.8167 | -0.0297 | 0.0240 | 0.8430 | 0.4867 |
| Fusion (SigLIP image + Jina text) | SigLIP text (product title) | 0.0490 | 11.8480 | 0.0212 | 0.0756 | 0.0010 | 0.6067 |
| Jina v5 text nano (product title) | SigLIP text (product title) | 0.0034 | 0.8234 | -0.0226 | 0.0283 | 0.7820 | 0.5133 |
| Production on-site ranking | SigLIP text (product title) | 0.2039 | 49.2983 | 0.1688 | 0.2376 | 0.0000 | 0.7467 |
| Random | SigLIP text (product title) | -0.1362 | -32.9204 | -0.1622 | -0.1115 | 0.0000 | 0.3000 |
| SigLIP image (product photo) | SigLIP text (product title) | 0.0226 | 5.4612 | -0.0038 | 0.0491 | 0.1020 | 0.5300 |
| SigLIP image (MGPL crop) | SigLIP text (product title) | 0.0293 | 7.0941 | 0.0023 | 0.0559 | 0.0330 | 0.5700 |
| SigLIP image (largest-area crop) | SigLIP text (product title) | 0.0256 | 6.1777 | -0.0006 | 0.0518 | 0.0570 | 0.5400 |

![Head to head](fig_head_to_head.png)

The left panel is the cross-model contrast; the right panel is the controlled modality contrast.
A conclusion about modality requires the right panel to be significant, not just the left.

## Reading this honestly

- The two encoders were trained for different objectives. SigLIP is a contrastive image-text model
  trained on web alt-text; Jina v5 text nano is a distilled multilingual retrieval model. A gap
  between them is a statement about model fit to short e-commerce queries, not about pixels vs words.
- Product titles are dense with exactly the tokens head queries use (brand, sport, gender, product
  type), so text has a strong prior here. Image embeddings contribute colour, silhouette, and
  material, which head terms rarely specify. Expect the modality gap to widen on descriptive or
  visual queries and to matter less on head terms like these.
- The `fusion` row is the practical takeaway: combining the two signals is competitive with or
  better than either alone on recall and MAP, which implies the modalities carry partly
  complementary information.

## Caveats

- Labels come from click behaviour under the current ranker, so they inherit its exposure bias.
  Position debiasing reduces but does not remove this.
- Candidate pools are products the production system already surfaced, so this measures
  **re-ranking** quality, not full-catalogue retrieval.
- Only head terms are covered. Tail and descriptive queries are where image embeddings are most
  likely to pay off and are not represented here.
- One image per product (the default ecode image); no multi-view or in-context imagery.
- `jinaai/jina-embeddings-v5-text-nano` is CC BY-NC 4.0. Commercial use requires a licence from Jina AI.

## Suggested next steps

1. Re-run on tail and attribute-heavy queries (colour, pattern, style words) where the modality
   contrast should be strongest.
2. Add a same-family text baseline in the other direction (e.g. a SigLIP-class text retriever vs
   Jina) to separate encoder quality from modality once more.
3. Evaluate fusion weighting rather than an equal-weight z-score blend.

## Reproducing

```bash
./run_all.sh
```

Artifacts: `results/summary.csv`, `results/per_query_metrics.csv`, `results/significance.csv`,
`results/run_meta.json`.
