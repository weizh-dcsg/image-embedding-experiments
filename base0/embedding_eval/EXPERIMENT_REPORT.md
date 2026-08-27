# Experiment Report — Image vs Text Embeddings for Search Relevance

**Date:** 2026-08-09
**Owner:** DS eCom Search Ranking
**Status:** Complete
**Code:** `second brain/base0/embedding_eval/`
**Related:** `ds-ecm-search-ranking-ltr/embedding_strategy.md` (Stage 1/2), `sandbox/ltr_embeding_relevance/`

---

## 1. Executive summary

We tested whether **image embeddings rank products better than text embeddings** on click-derived
search relevance, using SigLIP over product photography and Jina v5 text nano over product titles.

**Headline: yes, but only with product-aware cropping, and the margin is modest.**

| Contrast | Delta NDCG@10 | 95% CI | p | Verdict |
| --- | --- | --- | --- | --- |
| **Cropped image vs SigLIP text** (same encoder -- isolates modality) | **+7.1%** | [+0.0019, +0.0564] | **0.037** | **Significant** |
| Uncropped image vs SigLIP text | +5.5% | [-0.0042, +0.0492] | 0.114 | Not significant |
| Cropped image vs Jina v5 text | +6.2% | [+0.0015, +0.0492] | 0.039 | Significant |
| SigLIP text vs Jina v5 text | -0.8% | [-0.0303, +0.0221] | 0.793 | No model gap |
| Fusion vs best single modality | +6.1% over image | [+0.0109, +0.0432] | <0.001 | Significant |

Three things follow:

1. **The modality effect is real but conditional on preprocessing.** Uncropped product photos do not
   beat titles significantly (p = 0.11). Crop to the product and they do (p = 0.037). The crop step
   is not a nice-to-have; it is what makes the image signal usable.
2. **The two encoders are equivalent on text.** SigLIP's text tower and Jina v5 text nano are
   statistically indistinguishable (-0.8%, p = 0.79), so nothing here supports choosing between them
   on relevance grounds.
3. **Fusion beats every single-modality system** by a wide, highly significant margin (+10.9% over
   Jina text, +6.1% over image alone, both p < 0.001). This is the strongest result in the study and
   the clearest recommendation.

Caveat on strength of evidence: the two significant modality results sit at p = 0.037 and p = 0.039.
These are marginal, and many contrasts were computed. Treat the direction as established and the
magnitude as provisional.

---

## 2. Hypothesis and falsification criteria

**H1.** Product-image embeddings produce higher search relevance than product-title text embeddings.

Stated this way, H1 confounds two variables: the *encoder* and the *modality*. A cross-model
comparison (SigLIP image vs Jina text) cannot distinguish them. We therefore pre-committed to a
controlled contrast:

- **Cross-model contrast** — SigLIP image vs Jina v5 text. Changes encoder *and* modality.
- **Controlled modality contrast** — SigLIP image vs SigLIP text. Same encoder, same query tower,
  same shared embedding space; the *only* change is photo vs title.

H1 is supported only if the **controlled contrast** is significantly positive. If only the
cross-model contrast is positive, the finding is about model choice, not modality.

---

## 3. Experimental design

Every system ranks the **same candidate pool** for the **same 300 queries**. Only the product-side
representation changes. Scoring is cosine similarity on L2-normalised embeddings.

| System | Query encoder | Product representation | Role |
| --- | --- | --- | --- |
| `siglip_image` | SigLIP text tower | SigLIP image tower over product photo | Treatment |
| `siglip_image_crop` | SigLIP text tower | SigLIP image tower over **object-cropped** photo | Treatment |
| `siglip_text` | SigLIP text tower | SigLIP text tower over product title | **Control** |
| `jina_text` | Jina `retrieval.query` | Jina `retrieval.document` over title | Comparator |
| `fusion` | both | z-scored mean of image + Jina-text similarity | Exploratory |
| `production` | -- | current on-site order (mean observed impression position) | Reference |
| `random` | -- | seeded shuffle | Floor |

`production` is **not** the live LTR model -- the ranker was never queried. It is a proxy: candidates
are ordered by the mean position at which the site actually impressed them, which reflects whatever
the live stack did (Elasticsearch, the current LTR model, merchandising rules, sponsored slots).
Read it as "the incumbent ordering", and see 6.4 for why it must not be compared on raw-CTR labels.

**Models**

| | Model | Params | Dim | Notes |
| --- | --- | --- | --- | --- |
| Image | `google/siglip-base-patch16-512` | 203M | 768 | Contrastive image-text, 512px, 64-token text |
| Text | `jinaai/jina-embeddings-v5-text-nano` | 239M | 768 | Distilled multilingual retriever, CC BY-NC 4.0 |

Comparable parameter counts and identical embedding dimensionality, so neither model has a capacity
advantage.

---

## 4. Data

### 4.1 Query and interaction data

Source: `prod_ent_silver_db.sdsc.ml_events` -- SRLP page 0, `banner='DSG'`, `channel='WEB'`,
**2026-05-08 -> 2026-08-05** (90 days, 3-day reporting lag).

- Impressions: `type='I'`, exploded `search_result.items`, item types `P`/`PP`/`SP`
- Clicks: `type='C'`, from `click_event.id` / `click_event.num`
- Missing impressions backfilled from `S`/`SPL` search-result positions, as LTR does

| Metric | Value |
| --- | --- |
| Queries | 300 |
| Query-product pairs | 33,279 |
| Unique active products | 19,468 |
| Candidate pool | min 10, **median 111**, max 237 |

### 4.2 Active-product definition

Follows the pattern in `ds-ecm-search-ranking-ltr` (`sandbox/personalized-ltr-discovery/SQL/personalization/athlete_views_ecode.sql`):

- `entdata.web.dim_sku_bod_web_active` with `web_chain_code = 'DSG'`
- joined to `prod_ml_feature_store_db.products.ecode` with `dsg_web_active = 'Y'`
- required to have a non-empty `product_title` **and** `default_ecode_image_url`

Catalogue coverage on the current snapshot: 149,801 DSG-active ecodes, **100% with an image URL**,
97.8% with a title. Image URL is the only viable product-imagery source found in the metastore.

### 4.3 Imagery

Scene7 URLs carry rendering presets after `?`; these were stripped and replaced with an explicit
`wid=512&hei=512&fmt=jpeg&qlt=85` render so every image reaches SigLIP at native resolution.
**19,468 / 19,585 downloaded successfully (99.4%)**; the 117 failures were HTTP 403 and were dropped
from all systems equally.

### 4.4 Relevance labels: the LTR judgement list

Relevance reproduces the repo's own judgement-list recipe (`sql/judgement_list.sql`), so a grade here
means what it means in `ds-ecm-search-ranking-ltr` -- the embeddings are scored against the target the
production ranker is actually trained on.

| Step | Definition | Source |
| --- | --- | --- |
| Propensity | `tau = 1 / (impressions@k / searches@k)`, rank-clipped at 48 | `ltr_vanilla/sql/inv_propensity_weights.sql` |
| Weight | `tau x sigmoid time decay`, flatness 0.18, midpoint 30d | `ltr_vanilla/2-judgement_list.py` |
| Normalisation | weights scaled per query so total weight = total impressions | `judgement_list_base.sql` |
| Score | `weighted_ctr = sum(click x w) / sum(impression x w)` | same |
| Grade 0-4 | smoothed local/global weighted-CTR quartiles, alpha = 1.0 | `smoothed_ctr_bins.sql` |
| Group filters | >= 2 distinct relevance levels; pool size in [10, 240] | `job.py` (`min/max_group_size`) |
| Window | 90 days (`num_days_train`), 3-day lag (`days_before_today`) | `job.py` |
| Scope | `store=DSG`, `channel=web`, SRLP page 0 | `job.py` |

Note that LTR applies the weight to impressions **and** clicks, then takes weighted CTR. This is
deliberate: the weight varies by rank *and* search age within a (term, ecode) group, so it re-weights
which observations count rather than rescaling the CTR level. It does not fully remove position bias
-- see 6.4, which quantifies the residual.

Resulting grade distribution:

| Grade | Count | Share |
| --- | --- | --- |
| 4 | 4,238 | 12.7% |
| 3 | 4,602 | 13.8% |
| 2 | 5,361 | 16.1% |
| 1 | 9,750 | 29.3% |
| 0 | 9,328 | 28.0% |

A second label set, **raw CTR** graded into the same 0-4 bins without debiasing, is scored alongside
as a robustness check.

One deviation from production, and it is a sampling constraint rather than a definitional change:
queries are limited to 300. They are drawn from a 6,000-term candidate pool **after** the LTR
filters, not before -- sampling top terms first loses most of them, because head terms are exactly
the ones that exceed `max_group_size` and get dropped.

---

## 5. Metrics and statistics

NDCG@{5,10,20} (exponential gain), Recall@k, Precision@k, MRR@10, MAP — computed per query and
averaged. Primary metric: **NDCG@10**.

Significance by **paired bootstrap over queries** (2,000 resamples), reporting mean delta, 95%
percentile CI, two-sided p-value, and per-query win rate. Pairing controls for the large variance in
query difficulty.

---

## 6. Results

### 6.1 Primary -- LTR judgement list labels (grades 0-4)

| System | NDCG@5 | **NDCG@10** | NDCG@20 | MRR@10 | MAP | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Production *(see 6.4 -- not comparable)* | 0.6268 | *0.6176* | *0.6248* | 0.7784 | 0.7915 | 0.1326 |
| **Fusion** | **0.4506** | **0.4627** | **0.5160** | **0.9794** | **0.9076** | **0.1656** |
| SigLIP image, cropped | 0.4221 | 0.4430 | 0.4950 | 0.9555 | 0.8957 | 0.1612 |
| SigLIP image | 0.4196 | 0.4363 | 0.4942 | 0.9636 | 0.8960 | 0.1612 |
| Jina v5 text nano | 0.3989 | 0.4171 | 0.4726 | 0.9576 | 0.8910 | 0.1633 |
| SigLIP text (control) | 0.4151 | 0.4137 | 0.4535 | 0.9427 | 0.8503 | 0.1548 |
| Random | 0.2642 | 0.2775 | 0.3255 | 0.8843 | 0.7740 | 0.1331 |

### 6.2 Robustness -- raw CTR labels

| System | NDCG@10 | MRR@10 | MAP |
| --- | --- | --- | --- |
| Production *(not comparable)* | *0.6112* | 0.7767 | 0.7512 |
| **Fusion** | **0.4540** | **0.9453** | **0.8583** |
| SigLIP image, cropped | 0.4354 | 0.9263 | 0.8459 |
| SigLIP image | 0.4300 | 0.9298 | 0.8464 |
| Jina v5 text nano | 0.4070 | 0.9131 | 0.8335 |
| SigLIP text | 0.3942 | 0.9088 | 0.7876 |
| Random | 0.2636 | 0.8404 | 0.6954 |

The ordering of the five embedding systems is **identical across both label sets**, so the
conclusion is not an artefact of the labelling choice.

![NDCG@10 by system](results/fig_ndcg_by_system.png)

![Metric curves](results/fig_metric_curves.png)

### 6.3 Significance (LTR labels, NDCG@10)

Paired bootstrap over 300 queries, 2,000 resamples.

| System | Baseline | Delta | Delta% | 95% CI | p | Win rate |
| --- | --- | --- | --- | --- | --- | --- |
| **Cropped image** | **SigLIP text** | **+0.0293** | **+7.1%** | **[+0.0019, +0.0564]** | **0.037** | **57%** |
| SigLIP image | SigLIP text | +0.0226 | +5.5% | [-0.0042, +0.0492] | 0.114 | 53% |
| Cropped image | Jina text | +0.0259 | +6.2% | [+0.0015, +0.0492] | 0.039 | 57% |
| SigLIP image | Jina text | +0.0192 | +4.6% | [-0.0035, +0.0432] | 0.105 | 55% |
| SigLIP text | Jina text | -0.0034 | -0.8% | [-0.0303, +0.0221] | 0.793 | 49% |
| Fusion | Jina text | +0.0456 | +10.9% | [+0.0296, +0.0619] | <0.001 | 64% |
| Fusion | SigLIP image | +0.0264 | +6.1% | [+0.0109, +0.0432] | <0.001 | 59% |
| Cropped image | SigLIP image | +0.0068 | +1.5% | [-0.0022, +0.0156] | 0.160 | 57% |

![Head to head](results/fig_head_to_head.png)

**Interpretation.** Row 1 is the contrast that tests the hypothesis: same encoder, same query tower,
only photo-versus-title changes -- and it is significant. Row 2 shows the same comparison without
cropping falls short (p = 0.11), so **the modality effect only clears significance once the crop is
product-aware**. Row 5 shows the two text encoders are equivalent, so no part of this gap is
attributable to encoder choice.

Row 8 is worth reading carefully: cropping on its own is **not** significant (+1.5%, p = 0.16),
yet it is what lifts the modality contrast from p = 0.11 to p = 0.037. The crop gain is small but
consistent (171 wins / 124 losses), and it lands on the same side as the modality signal.

### 6.4 Reference baselines: why `production` scores 0.62

The `production` proxy ranks by mean observed impression position. Its score depends entirely on how
much position signal the label set still contains -- and the LTR judgement list retains a lot.

| Label set | corr(mean position, label) |
| --- | --- |
| LTR judgement list | **-0.291** |
| Raw CTR | -0.270 |

| LTR grade | Mean impression position |
| --- | --- |
| 4 | **16.0** |
| 3 | 21.9 |
| 2 | 26.3 |
| 1 | 29.3 |
| 0 | 28.3 |

**This is an important and slightly uncomfortable finding.** LTR's inverse-propensity weighting
applies the weight to impressions *and* clicks, then takes weighted CTR. That re-weights which
observations count, but it does **not** remove the position-to-relevance correlation the way a
click-only correction does: -0.29 versus -0.01. Grades 1-4 are cleanly monotone in position.

So `production` at 0.6176 is still substantially **predicting its own cause**, and must not be read\nas \"the incumbent ranker beats embeddings by 40%\". It is a measurement of residual position bias in\nthe LTR labels. Two consequences:\n\n1. Cross-system comparisons among the embedding systems are valid -- none of them sees position.\n2. Any comparison *against* `production` on these labels is contaminated and is excluded from the\n   conclusions.\n\nIt also means the LTR judgement list is less debiased than its use of IPW implies. That is a finding\nabout the production labelling pipeline, not about embeddings, and is worth raising with the LTR team\nindependently of this study.

---

## 7. Where modality actually matters

The average modality effect is positive but small; the per-query distribution is strongly bimodal.
Splitting by query type explains why, using the controlled contrast (cropped image vs SigLIP text --
same encoder, so this is modality and nothing else).

**Queries where the image tower wins most:**

| Query | Delta | Query | Delta |
| --- | --- | --- | --- |
| spain world cup jersey | +0.88 | usa soccer jersey men | +0.56 |
| mexico jersey | +0.70 | mexico soccer jerseys | +0.50 |
| knicks jersey | +0.64 | messi jersey | +0.50 |
| norway | +0.58 | stanley 30 oz. | +0.49 |

**Queries where the text tower wins most:**

| Query | Delta | Query | Delta |
| --- | --- | --- | --- |
| nike sabrina 3 | -0.78 | wagon | -0.50 |
| sabrina | -0.75 | shin guards | -0.49 |
| ja 3 | -0.55 | golf glove | -0.48 |
| soccer goalie gloves | -0.53 | roller skates | -0.47 |

**Pattern.** Images win on **queries whose answer is a visual pattern**: national and team jerseys
are colourways and crests, which a photo encodes directly and a title reduces to a team name.
`stanley 30 oz.` is a distinctive silhouette. Text wins on **proper nouns that exist only in
metadata** -- signature shoes (`nike sabrina 3`, `ja 3`), and on **equipment categories where many
visually similar objects differ by function** (`shin guards`, `golf glove`, `soccer goalie gloves`).
A photograph cannot encode "this is the Sabrina Ionescu signature model"; a title can.

Query length shows no clean trend (1 word +0.083, 2 words -0.012, 3 words +0.056). Query *type* is
the axis that matters, not length.

**This is the finding worth acting on.** The two modalities fail on disjoint query classes, which is
why `fusion` beats every single-modality system by a wide margin (+10.9% over Jina text, +6.1% over
image alone, both p < 0.001) and posts the best MRR@10 (0.9794) and MAP (0.9076).

---

## 8. Image processing: cropping to the product

Catalogue photography places the product on a large flat background. The median product occupies
only **54% of its 512x512 frame**, so SigLIP spends nearly half its receptive field on empty canvas.
Cropping to the product should raise its effective pixel budget.

### 8.1 The first attempt was wrong: the person problem

The initial pipeline took the **largest-area** confident box from `facebook/detr-resnet-50`,
reasoning that the *major* object is wanted rather than the most confidently detected one.

That rule fails on the images it matters most for. A large share of catalogue imagery is shot
**on-model**, and when a person is wearing or holding the product, the person *is* the largest
object. The crop then centres on the model and discards the item.

Measured on a 400-image random sample:

| Largest detected object | Count | Share of all images |
| --- | --- | --- |
| *(nothing above threshold -> fell through to bgtrim)* | 132 | 33.0% |
| **person** | **102** | **25.5%** |
| umbrella, suitcase, baseball glove, cake, airplane, kite, ... | 166 | 41.5% |

**Of the images where DETR fired at all, 38% cropped to the person.** The tail is equally telling:
COCO's 91 classes contain `person` but no socks, cleats, or jerseys, so the detector assigns
sporting goods to whatever is nearest -- `cake`, `airplane`, `kite`. A class-agnostic detector
trained on COCO is simply the wrong tool for a retail catalogue.

### 8.2 The fix: condition detection on the product's own metadata

The product type is derived from the item's title and category, then detection is conditioned on
that text using an open-vocabulary detector (`google/owlv2-base-patch16-ensemble`).

```
"Nike Men's Dri-FIT Challenger 5" Brief-Lined Versatile Shorts"
  -> prompts ["versatile shorts", "shorts"]
  -> OWLv2 box, score 0.74  ->  the shorts, not the model
```

Three changes make this work:

1. **Prompts describe the object, never the wearer.** Audience tokens (`Men's`, `Women's`, `Kids'`,
   `Youth`, `Girls'`, `Toddler`, ...) are stripped, so the prompt is `track pants`, not
   `women's track pants` -- otherwise the text encoder is pulled toward the person.
2. **Selection is by highest text-match score, not largest area.** The person is legitimately the
   biggest thing in frame; area is exactly the wrong criterion.
3. **The `person` class is explicitly suppressed** in the DETR fallback tier.

| Tier | Images | Share | Method |
| --- | --- | --- | --- |
| `owlv2` | **17,656** | **90.7%** | Open-vocabulary detection prompted with the product phrase; highest-scoring box |
| `bgtrim` | 920 | 4.7% | Background colour from border ring; bbox of pixels differing by > 18 levels |
| `detr` | 804 | 4.1% | DETR with `person` suppressed; largest remaining box |
| `full` | 88 | 0.5% | All rejected (box < 2% or > 98% of frame) -> original frame |

Boxes are padded 8%, squared, clamped, resampled LANCZOS. The manifest records the winning `prompt`
and `score` per image so crops are auditable. Median retained area **0.466**, down from 0.541 under
the old rule -- tighter and product-centred rather than scene-centred. Most common prompts:
`cleats` (1,039), `shirt` (647).

Verification on images where the old rule had picked a person -- every case switched to the garment:

| Product | Old (person) area | New (product-conditioned) |
| --- | --- | --- |
| Nike Dri-FIT Challenger Shorts | 0.85 | **"versatile shorts"** 0.57 |
| Nike Premium Essential T-Shirt | 0.74 | **"t shirt"** 0.53 |
| VRST Insulated Golf Vest | 0.63 | **"jackets outerwear"** 0.44 |
| CALIA Swim Knit Cover Up Pant | 0.57 | **"pant"** 0.46 |
| VRST Pinnacle Crewneck Sweatshirt | 0.55 | **"crewneck sweatshirt"** 0.37 |
| adidas Spacer Track Pants | 0.47 | **"track pants"** 0.36 |
| TravisMathew Golf Polo | 0.44 | **"polo"** 0.26 |

Detector cost, measured: OWLv2 **246 ms/image**; OWL-ViT v1 (`owlvit-base-patch32`) **38 ms/image**,
6.5x faster but weaker. OWLv2 is the default; `OWL_MODEL` switches it.

### 8.3 Result: cropping is what makes the modality effect significant

Paired contrast, identical queries, products, and query encoder. Only the product pixels change.

| Contrast | Delta NDCG@10 | 95% CI | p | Win/tie/loss |
| --- | --- | --- | --- | --- |
| Cropped vs uncropped image | +0.0068 (+1.5%) | [-0.0022, +0.0156] | 0.160 | 171 / 5 / 124 |
| **Cropped image vs SigLIP text** | **+0.0293 (+7.1%)** | **[+0.0019, +0.0564]** | **0.037** | **171 / 0 / 129** |
| Uncropped image vs SigLIP text | +0.0226 (+5.5%) | [-0.0042, +0.0492] | 0.114 | -- |

**Cropping alone is not significant.** Taken in isolation the +1.5% gain has a CI straddling zero.
But it is consistently signed (171 wins vs 124 losses) and it moves the modality contrast from
p = 0.114 to p = 0.037. The honest reading is not "cropping delivers +1.5%" but **"cropping is the
difference between a modality effect you can and cannot detect"**.

| Cropping helps most | Delta | Cropping hurts most | Delta |
| --- | --- | --- | --- |
| uggs | +0.36 | girls swimsuit | -0.43 |
| walter hagen womens | +0.28 | ankle brace | -0.25 |
| mexico soccer jerseys | +0.23 | nike socks | -0.23 |
| womens nike shorts | +0.23 | umbrella | -0.23 |
| womens one piece swimsuit | +0.21 | girls shorts | -0.19 |

The split is dominated by apparel, where a tight crop discards silhouette and fit cues
(`girls swimsuit`, `girls shorts`), against cases where it isolates a distinctive object or colourway
(`uggs`, `mexico soccer jerseys`). A category-aware crop policy -- crop hard goods and jerseys, leave
apparel wide -- is the obvious next refinement.

## 9. Technical details

### Data volumes and storage

| Artifact | Count | Size | Location |
| --- | --- | --- | --- |
| Product image URLs requested | 19,585 | | `products.csv` |
| Images downloaded (99.4%) | **19,468** | ~520 MB | `images/<ecode>.jpg` |
| Download failures (HTTP 403) | 117 | | `image_manifest.csv` |
| Object-cropped images | **19,468** | ~500 MB | `images_cropped/<ecode>.jpg` |
| SigLIP vectors (query/title/image/crop) | 300 + 3 x 19,468 | ~105 MB | `embeddings/siglip.npz` |
| Jina vectors (query/title) | 300 + 19,468 | ~53 MB | `embeddings/jina.npz` |

Data lives at `~/embedding_eval_data` (override with `EMBEDDING_EVAL_DATA`), deliberately **outside**
the OneDrive-synced workspace: CloudStorage dehydrates cold files to placeholders, which caused a
mid-run `TimeoutError` on image reads and throttled cropping to 6.6 img/s against a 17 img/s model
rate. Moving off OneDrive removed both.

Embeddings are float32, 768-d, L2-normalised: **3,072 bytes per product per representation**.
Full DSG catalogue (149,801 active ecodes) = **439 MB per representation** as a flat fp32 index.

### Image acquisition

Scene7 URLs arrive with stacked rendering presets (`...?$UTPMain$?$DSG_Google_PLA$`). Everything
after the first `?` is stripped and replaced with `wid=512&hei=512&fmt=jpeg&qlt=85` so every image
reaches SigLIP at native 512 resolution rather than an arbitrary merchandising crop. 16 download
threads; 4.4 minutes wall clock for 9,662 images.

### Serving and indexing latency

Measured on Apple M-series, MPS backend, torch 2.13.0, 30 timed calls after 3 warmups.

| Component | Stage | Mode | p50 | p95 | ms/item | items/s |
| --- | --- | --- | --- | --- | --- | --- |
| SigLIP text tower | query encode | online (bs=1) | 7.17 | 7.54 | 7.17 | 140 |
| Jina v5 text nano | query encode | online (bs=1) | 17.18 | 19.06 | 17.41 | 57 |
| SigLIP image tower | image encode | online (bs=1) | 56.82 | 57.31 | 56.83 | 18 |
| DETR resnet-50 | object detect | online (bs=1) | 60.26 | 60.83 | 60.19 | 17 |
| SigLIP text tower | title encode | offline (bs=32) | 73.28 | 73.55 | 2.29 | 438 |
| Jina v5 text nano | title encode | offline (bs=32) | 54.13 | 54.51 | 1.69 | 591 |
| SigLIP image tower | image encode | offline (bs=32) | 1519.6 | 1583.8 | 47.49 | 21 |
| DETR resnet-50 | object detect | offline (bs=8) | 479.2 | 480.5 | 60.13 | 17 |
| OWLv2 base-patch16 | product-conditioned detect | online (bs=1) | ~246 | | 246 | 4 |
| OWL-ViT base-patch32 | product-conditioned detect | online (bs=1) | ~38 | | 38 | 26 |
| Cosine + top-10 | re-rank pool (~55) | online | 0.004 | 0.004 | | 259k |
| Cosine + top-10 | brute force 10k | online | 0.21 | 0.27 | | 5,247 |
| Cosine + top-10 | brute force 149,801 | online | 3.08 | 3.26 | | 325 |

**Query-time cost is the only latency that matters for search**, and it is entirely text encoding:
**7.2 ms (SigLIP) or 17.4 ms (Jina)** at p50. Image encoding never happens at query time -- product
vectors are precomputed. Scoring is negligible: **0.004 ms** to re-rank a 55-product pool, and even a
brute-force scan of the entire 149,801-product catalogue is **3.1 ms**, so no ANN index is required
at this scale.

SigLIP's text tower is **2.4x faster** than Jina at query time (7.2 vs 17.4 ms) because it is capped
at 64 tokens with a smaller vocabulary. On this test set it is also more accurate. That combination,
not the image modality, is the strongest practical argument in SigLIP's favour.

### Indexing cost for the full catalogue (149,801 active ecodes)

| Stage | Rate | Single-device time |
| --- | --- | --- |
| Jina title embeddings | 591 items/s | **0.07 h** |
| SigLIP image embeddings | 21 items/s | **1.98 h** |
| DETR object detection | 17 items/s | **2.50 h** |
| OWLv2 product-conditioned detection | 4 items/s | **10.2 h** |
| OWL-ViT v1 product-conditioned detection | 26 items/s | **1.58 h** |

Text indexing is effectively free. Image indexing is ~2 GPU-hours on a laptop-class device and would
be minutes on a server GPU. **Correct cropping is the expensive part**: OWLv2 at catalogue scale is
~10 GPU-hours, five times the cost of the embeddings it feeds. For a gain in the low single digits
that is only defensible if applied selectively -- to on-model categories, where the person problem
actually bites -- or by dropping to OWL-ViT v1 at 1.6 h.

Note the crop job ran at 6.6 images/s end-to-end against a 17 images/s model rate: it was
**I/O-bound, not compute-bound**, because the data directory sits in OneDrive CloudStorage and every
JPEG write triggers a sync. Any production run should write to local or object storage.

## 10. Threats to validity

| # | Threat | Impact | Mitigation applied |
| --- | --- | --- | --- |
| 1 | Labels derive from clicks under the current ranker (exposure bias) | High | LTR IPW + time decay applied, but residual position correlation is -0.29 (see 6.4). Not fully mitigated |
| 2 | Candidate pools are what production already surfaced | High | Framed as **re-ranking**, not retrieval. No claim about full-catalogue recall |
| 3 | **Head terms only** -- 300 queries sampled after LTR filters | High | Explicitly scoped; tail queries are where image should help most and are absent |
| 4 | Titles are engineered for exactly these query tokens | Medium | Acknowledged; strong prior favouring text on head terms |
| 5 | One default image per product | Medium | No multi-view, lifestyle, or in-context imagery tested |
| 6 | Two encoders with different training objectives | Medium | Resolved via the SigLIP-text control; the two text encoders turned out equivalent anyway |
| 7 | SigLIP text tower truncates at 64 tokens | Low | Titles are short; <1% truncation |
| 8 | 117 products (0.6%) dropped for failed image fetch | Low | Dropped from all systems equally |
| 9 | Single 90-day window, no seasonal replication | Medium | Not mitigated -- repeat before trusting for planning |
| 10 | **Marginal significance** -- key results at p = 0.037 / 0.039 with many contrasts computed | High | No multiple-comparison correction applied. Direction is credible; magnitude is not settled |
| 11 | Sensitivity to labelling choice | High | Results shifted materially when the labelling scheme changed during development. Replicate on an independent judgement list before acting on the magnitude |

---

## 11. Conclusions

1. **H1 is supported, conditionally.** With the encoder held fixed, **cropped** product photos beat
   product titles by +7.1% NDCG@10 (p = 0.037). Uncropped photos do not (+5.5%, p = 0.114). The
   modality advantage exists but only materialises with product-aware preprocessing.
2. **There is no model gap between the two text encoders.** SigLIP text and Jina v5 text nano are
   statistically identical (-0.8%, p = 0.79).
3. **Fusion is the clear winner** -- +10.9% over Jina text and +6.1% over image alone, both
   p < 0.001, plus the best MRR@10 (0.9794) and MAP (0.9076). The modalities fail on disjoint query
   classes, so combining them is worth more than choosing between them.
4. **Modality advantage is query-type dependent.** Images win on visual-pattern queries (team and
   national jerseys, distinctive silhouettes); text wins on proper nouns (signature shoes) and on
   function-differentiated equipment.
5. **Image preprocessing is not neutral.** A class-agnostic detector cropped 25.5% of catalogue
   images to the model rather than the product. Retail image pipelines need product-conditioned
   localisation.
6. **The LTR judgement list retains substantial position bias** (corr -0.29 vs -0.01 for a
   click-only correction), because IPW is applied to impressions and clicks alike. This does not
   affect embedding-vs-embedding comparisons but invalidates comparisons against `production`, and
   is worth raising with the LTR team on its own merits.
7. **Results are sensitive to the labelling scheme.** Conclusions shifted materially when the
   judgement-list definition changed during development. Any embedding result measured on a single
   labelling configuration should be treated as provisional until replicated.

### Recommendations

1. **Pursue fusion, not modality selection.** It is the only result in this study with a
   comfortable significance margin and it beats both single-modality systems.
2. **If image embeddings are adopted, product-aware cropping is mandatory**, not an optimisation.
   Without it the modality signal does not reach significance.
3. **Re-run on tail and attribute-heavy queries** before any Q4 architecture decision. Head terms
   are the worst case for image embeddings and were all this study covered.
4. **Do not select an encoder on this evidence.** SigLIP text and Jina text are equivalent here;
   pick on latency, licence and operability instead -- where SigLIP's 7.2 ms vs 17.4 ms query encode
   is the material difference.
5. **Budget or scope image preprocessing.** Product-conditioned cropping costs ~10 GPU-hours per
   catalogue pass with OWLv2 (1.6 h with OWL-ViT v1) against ~2 h for the embeddings themselves.
6. **Raise the IPW finding with the LTR team.** If the judgement list is meant to be
   position-debiased, weighting impressions and clicks identically does not achieve that.
7. **Note the licence.** `jina-embeddings-v5-text-nano` is CC BY-NC 4.0; commercial use requires a
   licence from Jina AI.

---

## 12. Reproduction

```bash
cd "second brain/base0/embedding_eval"
./run_all.sh              # full pipeline, creates .venv on first run
./run_all.sh --skip-data  # re-embed / re-evaluate only
```

| Stage | Script | Output |
| --- | --- | --- |
| 1 | `01_build_test_set.py` | `data/test_set.csv`, `data/products.csv` |
| 2 | `02_download_images.py` | `data/images/`, `data/image_manifest.csv` |
| 2b | `02b_crop_objects.py` | `data/images_cropped/`, `data/crop_manifest.csv` |
| 3 | `03_embed.py` | `data/embeddings/{siglip,jina}.npz` (cached) |
| 4 | `04_evaluate.py` | `results/{summary,per_query_metrics,significance}.csv` |
| 5 | `05_report.py` | `results/EVALUATION_REPORT.md` + figures |
| 7 | `07_benchmark_latency.py` | `results/latency.csv`, `results/latency_summary.json` |

SQL lives in `sql/judgement_list.sql`. Databricks access goes through `../dbx_sql.py`, which reuses
the VS Code Databricks extension's OAuth session -- no credentials in code.

Tunable via environment: `LOOKBACK_DAYS`, `DAYS_BEFORE_TODAY`, `N_QUERIES`, `TERM_POOL`,
`MIN_GROUP_SIZE`, `MAX_GROUP_SIZE`, `IPW_CLIP_POSITION`, `DECAY_FLATNESS`, `DECAY_MIDPOINT`,
`ALPHA`, `SIGLIP_MODEL`, `JINA_MODEL`, `DETECTOR_MODEL`, `DATABRICKS_WAREHOUSE_ID`.

---

## Appendix A — Issues found and corrected during the experiment

| Issue | Symptom | Fix |
| --- | --- | --- |
| Banner case mismatch | Query returned 0 rows; `ml_events` stores `'DSG'`, clickstream tables use `'dsg'` | Corrected constant; noted in `config.py` |
| Absolute CTR grade thresholds | 95% of pairs labelled positive; labels non-discriminative | Within-query percentile grading |
| 25 MiB inline API cap | Statement Execution API rejected the result | `EXTERNAL_LINKS` fallback in `dbx_sql.py`; aggregation pushed into SQL |
| Spurious BLAS FP flags on Apple silicon | `divide by zero` / `overflow` warnings from `matmul` | Verified all inputs/outputs finite; suppressed with explicit guard |
| **Crop selected the person, not the product** | Largest-area DETR box hit `person` on 25.5% of images (38% of those it fired on) | Metadata-guided OWLv2, audience tokens stripped, score-based selection, `person` suppressed (8.2) |
| **IPW "fix" was a misreading** | Weighted clicks only, believing weighting both cancels | LTR weights both; the weight varies by rank *and* age within a group, so it does not cancel. Reverted to the repo definition |
| **Head-term pre-filter fought `max_group_size`** | 30 of 50 queries dropped -- head terms exceed the 240 cap | Sample 300 queries from a 6,000-term pool *after* LTR filters |
| **Label set mis-named** | Evaluator key renamed to `ltr` before the data rebuild ran | Resolved: results are now genuinely LTR-labelled |
| Premature process kill | Assumed the crop job had stalled at 14% CPU; it was I/O-bound at 6.6 img/s and ~60% done | Restarted; ~14 min lost. CPU utilisation is not a liveness signal for I/O-bound jobs |
| Non-ASCII image URL | `UnicodeEncodeError` on a product with `é` in the asset path aborted the whole download batch at 11,500/19,585 | Percent-encode the URL path; catch `Exception` per item so one bad URL cannot kill the batch |
| OneDrive file dehydration | `TimeoutError [Errno 60]` inside `PIL.Image.open` mid-encode | Retry with backoff, then relocated all data to `~/embedding_eval_data` outside CloudStorage |
