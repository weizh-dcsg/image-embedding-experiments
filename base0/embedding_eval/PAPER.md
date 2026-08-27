# Product-Grounded Localization for Image Embeddings in E-Commerce Search: A Controlled Evaluation

**Working paper — DS eCom Search Ranking**
Date: 2026-08-09

---

## Abstract

Product images are widely assumed to carry ranking signal that product titles do not, yet controlled
evaluations of image embeddings in e-commerce retrieval frequently report null results. We
investigate one under-examined source of measurement error: how the product is localized within its
photograph before encoding.

Catalogue photography is routinely shot on-model. Across a 19,468-product catalogue we measure that a
class-agnostic detector selects a **person** as the dominant object in **25.5%** of images — 40% of
the images on which it fires — and assigns sporting goods to categorically wrong classes (`umbrella`,
`airplane`, `cake`) because its fixed taxonomy contains no retail categories. Embeddings computed
from such crops encode the model, not the merchandise.

We introduce **Metadata-Guided Product Localization (MGPL)**, a training-free procedure that derives
a product-type phrase from the item's own catalogue title and taxonomy, strips audience tokens that
describe the shopper rather than the object, and conditions an open-vocabulary detector on that
phrase. MGPL resolves 90.7% of the catalogue and reduces median retained frame area from 0.583 to
0.466, with qualitatively correct object selection.

Using a dual-tower encoder to hold the model fixed and vary only the modality, we evaluate a
three-level localization ablation against a production learning-to-rank judgement list over 300
queries and 33,279 query-product pairs. **Our central negative finding is that better localization
does not translate into statistically demonstrable ranking gains at this scale.** Ranking quality is
monotone in localization quality (0.4363 → 0.4393 → 0.4430 NDCG@10) and the modality contrast
strengthens monotonically (p = 0.102 → 0.057 → 0.033), but no pairwise localization contrast reaches
significance, and MGPL beats naive cropping by only +0.9% (p = 0.39). A visibly large defect in the
input — a quarter of images cropped to the wrong object — produces an effect too small to resolve
with 300 queries.

Two further results are robust and, we argue, more actionable. An oracle selecting the better modality
per query reaches **+16.3%** over the best single system (p < 0.001), yet fixed equal-weight fusion
captures only **29.5%** of that headroom, and two natural routers — lexical brand-token detection and
unsupervised confidence-margin selection — achieve **48%** and **42%** accuracy, the latter
significantly *below* chance. Separately, we audit the judgement list and find that applying
inverse-propensity weights to impressions and clicks alike leaves a position-label correlation of
**−0.29**, silently invalidating comparisons against production ranking.

---

## 1. Introduction

Dense retrieval in e-commerce search is dominated by text. Product titles are short, keyword-dense,
and written specifically to match query vocabulary, which makes them a strong and cheap
representation. Image embeddings are an obvious complement: colour, silhouette, pattern and material
are visible in a photograph and often absent from a title. Yet published and internal comparisons
frequently find that image embeddings do not improve ranking, and the usual explanation is that
head-query e-commerce search is a lexical matching problem where vision adds little.

We report an experiment that complicates this account. Our central observation is that the standard
evaluation pipeline for image embeddings contains a step that is treated as engineering detail but
behaves as a confound: **how the product is localized within its photograph**.

Catalogue imagery is not object-centric in the way benchmark datasets are. A substantial fraction is
lifestyle or on-model photography in which a human occupies most of the frame. A detector trained on
COCO — which contains a `person` class and no class for socks, cleats, or jerseys — will reliably
select the human. The resulting embedding describes a model wearing merchandise rather than the
merchandise. Any evaluation built on such embeddings measures a degraded version of the modality and
is biased toward the conclusion that images do not help.

### 1.1 Contributions

1. **An empirical characterisation of person contamination in retail catalogue imagery.** Across
   19,468 catalogue images, class-agnostic largest-area detection selects a person in 25.5% of images
   and 40% of images where detection fires (Section 4.1). To our knowledge this failure mode has not
   been quantified.

2. **MGPL: Metadata-Guided Product Localization.** A training-free method that conditions an
   open-vocabulary detector on a product-type phrase derived from the item's own title and taxonomy,
   with explicit removal of audience tokens (`Men's`, `Kids'`, `Youth`) that steer detection toward
   the wearer. MGPL requires no annotation and no fine-tuning, and reuses metadata already present in
   every catalogue (Section 3).

3. **A three-level localization ablation with a negative result.** No crop, naive largest-area crop,
   and MGPL crop differ monotonically in ranking quality, but no pairwise contrast is significant and
   MGPL exceeds naive cropping by only +0.9% (p = 0.39). **Correcting a defect affecting a quarter of
   inputs yields an effect too small to resolve at n = 300** (Section 5.2). We regard this as the
   paper's most important result for practitioners, because it bounds the value of a preprocessing
   investment that is otherwise intuitively compelling.

4. **A complementarity and routing analysis with a second negative result.** Per-query oracle
   selection yields +16.3% over the best single modality, fixed fusion recovers only 29.5% of it, and
   two natural routers operate at or below chance. We formalise retail modality routing as an open
   problem and provide the oracle bound as a target (Section 5.4).

5. **A judgement-list bias audit.** We show that the inverse-propensity scheme used in our production
   LTR pipeline — weighting impressions and clicks identically — leaves a −0.29 position-label
   correlation, and that this invalidates any comparison against a position-derived production
   baseline. We propose the residual correlation as a routine diagnostic (Section 5.5).

### 1.2 Scope and honesty about strength of evidence

This is a single-retailer study on 300 head queries with click-derived labels. The modality contrast
under MGPL reaches p = 0.033, but the localization contrasts that motivate MGPL do not reach
significance individually, and we do not claim they do. The routing and contamination results are
considerably more robust (p < 0.001 and a direct census respectively) and carry most of the paper's
weight.

---

## 2. Related work

**Vision-language embeddings.** CLIP (Radford et al., 2021) established contrastive image-text
pretraining with a dual-tower architecture; SigLIP (Zhai et al., 2023) replaces the softmax
contrastive objective with a pairwise sigmoid loss that decouples the loss from global batch
statistics. We exploit a structural property of these models that is rarely used in evaluation: both
towers project into a *shared* space, so the same model can represent a product by its photograph or
by its title. This makes an encoder-controlled modality contrast possible.

**Text retrieval embeddings.** We compare against `jina-embeddings-v5-text-nano` (Akram et al.,
2026), a 239M-parameter multilingual retriever distilled from a larger teacher with task-specific
contrastive objectives and explicit `retrieval.query` / `retrieval.document` prompts.

**Object detection and open-vocabulary detection.** DETR (Carion et al., 2020) provides
set-prediction detection over a fixed class inventory. OWL-ViT (Minderer et al., 2022) and OWLv2
(Minderer et al., 2023) perform detection conditioned on free-form text, enabling localization of
categories absent from any fixed taxonomy. MGPL uses this capability but its contribution is not the
detector: it is the observation that the *conditioning text is already available as catalogue
metadata*, and the specific normalisation required to make that text describe the object rather than
the shopper.

**Position bias and counterfactual evaluation.** Click logs are confounded by presentation order.
Inverse-propensity weighting (Joachims et al., 2017) corrects this by weighting observations by the
inverse probability of examination. Our audit in Section 5.5 concerns *where* the weight is applied,
which we find to be consequential and, in our pipeline, incorrect for the stated goal.

**Evaluation metrics.** We report NDCG (Järvelin & Kekäläinen, 2002) with exponential gain, plus
MRR, MAP and Recall@k.

---

## 3. Method: Metadata-Guided Product Localization

### 3.1 Problem

Let an item have photograph $I$, title $t$, and taxonomy leaf $c$. Standard practice crops $I$ to a
region $b$ before encoding. A class-agnostic detector chooses

$$b_{\text{area}} = \arg\max_{b \in \mathcal{B}(I)} \text{area}(b)$$

over confident detections $\mathcal{B}(I)$. When a person is present and wearing the product, the
person maximises area, so $b_{\text{area}}$ localizes the wearer. Selecting by confidence instead
does not fix this, because `person` is also the class a COCO-trained detector is most confident
about in such images.

### 3.2 Product phrase extraction

We derive a set of short noun phrases $P(t, c)$ describing what the item *is*. Let $W(t)$ be the
token sequence of the title. Retail titles place brand and model first and product type last
(`Nike Men's Dri-FIT Challenger 5" Brief-Lined Versatile Shorts`), so the informative tokens are the
suffix. We take the final four tokens, remove any token in an **audience vocabulary** $A$ —

$$A = \{\text{men's, women's, boys', girls', kids', youth, junior, toddler, infant, unisex}, \dots\}$$

— and emit the last bigram and last unigram, plus the taxonomy leaf reduced the same way:

$$P(t,c) = \{\text{bigram}(W(t) \setminus A),\ \text{unigram}(W(t) \setminus A),\ \text{leaf}(c) \setminus A\}$$

Removing $A$ is essential rather than cosmetic. Tokens such as `Women's` are semantically about a
person, and leaving them in the conditioning text pulls the detector's text embedding toward the
human in the frame — precisely the failure the method exists to prevent.

For the example above, $P = \{$`versatile shorts`, `shorts`$\}$.

### 3.3 Product-conditioned selection

Given an open-vocabulary detector $D$, we condition on the phrases and select by **text-match score**
rather than area:

$$b_{\text{MGPL}} = \arg\max_{(b,s) \in D(I,\ P(t,c))} s$$

Area is explicitly the wrong criterion here: the person is legitimately the largest object, and
selecting for size is what produces the failure.

### 3.4 Cascade and fallbacks

Open-vocabulary detection is expensive (Section 6.3), and no detector fires on every image. MGPL is
therefore a cascade, taking the first acceptable result:

| Tier | Rule |
| --- | --- |
| 1. `owlv2` | Product-conditioned box, highest text-match score |
| 2. `detr` | Fixed-taxonomy detector with the `person` class **suppressed**; largest remaining box |
| 3. `bgtrim` | Background colour estimated from the border ring; bounding box of pixels differing by more than a threshold |
| 4. `full` | Original frame |

A box is *acceptable* if it occupies between 2% and 98% of the frame. Accepted boxes are expanded by
8%, squared, clamped to the image, and resampled. Tier 3 requires no model and handles the flat
studio backgrounds common in catalogue photography; tier 2 exists because suppressing `person` makes
a fixed-taxonomy detector usable as a cheap intermediate.

---

## 4. Experimental design

### 4.1 Measuring the failure mode

To establish that the problem is real and not hypothetical, we ran class-agnostic largest-area
detection over the full 19,468-image catalogue and recorded the selected class.

| Selection outcome | Count | Share of catalogue |
| --- | --- | --- |
| DETR fired | 12,411 | 63.8% |
| — of which **person** | **4,967** | **25.5%** |
| Fell through to background trim | 6,341 | 32.6% |
| No usable box | 716 | 3.7% |

A person is the dominant object in **a quarter of all catalogue images** and in **40.0% of images
where the detector fires**. The class distribution among the remainder is independently diagnostic:

| Selected class | Count |
| --- | --- |
| person | 4,967 |
| umbrella | 1,194 |
| suitcase | 717 |
| baseball glove | 551 |
| airplane | 468 |
| kite | 468 |
| handbag | 456 |
| cake | 362 |

Roughly 4,400 sporting-goods images are assigned to categorically wrong classes because the fixed
taxonomy contains no shoe, jersey, or glove. A COCO-trained detector is not merely imperfect on
retail imagery; it is categorically mismatched to it.

### 4.2 Systems

All systems rank the **same candidate pool** for the **same queries**; only the product-side
representation changes. Scoring is cosine similarity over L2-normalised embeddings.

| System | Query encoder | Product representation |
| --- | --- | --- |
| `image` | SigLIP text tower | SigLIP image tower over the raw photograph |
| `image+naive` | SigLIP text tower | SigLIP image tower over the largest-area crop |
| `image+MGPL` | SigLIP text tower | SigLIP image tower over the MGPL crop |
| `text-siglip` | SigLIP text tower | **SigLIP text tower over the title** |
| `text-jina` | Jina `retrieval.query` | Jina `retrieval.document` over the title |
| `fusion` | both | Equal-weight mean of z-normalised similarities |
| `production` | — | Mean observed on-site impression position |
| `random` | — | Seeded shuffle |

The three image arms form a **localization ladder**: identical encoder, identical source photographs,
differing only in how the product is cropped. This isolates localization quality as a single
manipulated variable and separates "cropping helps" from "product-grounded cropping helps".

`text-siglip` is the control that makes the experiment interpretable. Comparing a CLIP-family
image tower against a separate text retriever varies encoder *and* modality simultaneously; almost
all published comparisons of this kind are confounded in exactly this way. Because SigLIP's towers
share an embedding space, holding the encoder fixed and swapping the photograph for the title
isolates modality as the single manipulated variable.

### 4.3 Data and labels

Queries and interactions are drawn from 90 days of search-results-page events at a large
sporting-goods retailer (DSG banner, web channel, first result page), with a 3-day reporting lag.

| Property | Value |
| --- | --- |
| Queries | 300 |
| Query-product pairs | 33,279 |
| Unique active products | 19,468 |
| Candidate pool size | min 10, median 111, max 237 |

Candidates are restricted to currently web-active products possessing both a title and a default
image. Relevance grades 0–4 are produced by the retailer's **production learning-to-rank judgement
list**: rank-clipped inverse propensities multiplied by a sigmoid time decay, normalised per query,
applied to impressions and clicks, then binned by smoothed local/global weighted-CTR quartiles. Using
the production definition rather than an ad-hoc one matters, because it means the embeddings are
scored against the target the deployed ranker is trained on. A raw-CTR label set with identical
binning is scored alongside as a robustness check.

### 4.4 Metrics

NDCG@{5,10,20} with exponential gain, MRR@10, MAP, Recall@k and Precision@k, averaged over queries.
Primary metric NDCG@10. Significance by paired bootstrap over queries (2,000 resamples), reporting
mean difference, 95% percentile interval, two-sided p-value and per-query win rate. Pairing is
essential: per-query variance greatly exceeds between-system variance.

---

## 5. Results

### 5.1 MGPL localization behaviour

| Tier | Images | Share |
| --- | --- | --- |
| `owlv2` (product-conditioned) | 17,656 | **90.7%** |
| `bgtrim` | 920 | 4.7% |
| `detr` (person suppressed) | 804 | 4.1% |
| `full` frame | 88 | 0.5% |

Product-conditioned detection resolves the overwhelming majority of the catalogue. Median retained
frame area falls from **0.583** under largest-area selection to **0.466** under MGPL: crops are
tighter and centred on merchandise rather than scene. Most frequent conditioning phrases are
`cleats` (1,039) and `shirt` (647), i.e. the method recovers genuine product types from free-text
titles without a curated taxonomy.

Qualitatively, on images where largest-area selection had chosen a person, MGPL relocates to the
garment in every case inspected:

| Product | Person box (frame share) | MGPL box, conditioning phrase |
| --- | --- | --- |
| Dri-FIT Challenger Shorts | 0.85 | 0.57 — `versatile shorts` |
| Premium Essential T-Shirt | 0.74 | 0.53 — `t shirt` |
| Insulated Golf Vest | 0.63 | 0.44 — `jackets outerwear` |
| Swim Knit Cover-Up Pant | 0.57 | 0.46 — `pant` |
| Pinnacle Crewneck Sweatshirt | 0.55 | 0.37 — `crewneck sweatshirt` |
| Spacer Track Pants | 0.47 | 0.36 — `track pants` |
| Featherweight Golf Polo | 0.44 | 0.26 — `polo` |

MGPL therefore does what it was designed to do at the level of the *input*. Section 5.2 asks whether
that translates into ranking quality.

### 5.2 The localization ladder: a negative result

Ranking quality under the production judgement list:

| System | NDCG@5 | **NDCG@10** | NDCG@20 | MRR@10 | MAP |
| --- | --- | --- | --- | --- | --- |
| `production` *(not comparable — see 5.5)* | 0.6268 | *0.6176* | *0.6248* | 0.7784 | 0.7915 |
| **`fusion`** | **0.4506** | **0.4627** | 0.5160 | **0.9794** | **0.9076** |
| `image+MGPL` | 0.4221 | 0.4430 | 0.4950 | 0.9555 | 0.8957 |
| `image+naive` | 0.4233 | 0.4393 | **0.4967** | 0.9581 | 0.8958 |
| `image` | 0.4196 | 0.4363 | 0.4942 | 0.9636 | 0.8960 |
| `text-jina` | 0.3989 | 0.4171 | 0.4726 | 0.9576 | 0.8910 |
| `text-siglip` | 0.4151 | 0.4137 | 0.4535 | 0.9427 | 0.8503 |
| `random` | 0.2642 | 0.2775 | 0.3255 | 0.8843 | 0.7740 |

The three image arms are **monotone in localization quality** — 0.4363 → 0.4393 → 0.4430 — which is
the ordering the method predicts. But the pairwise contrasts do not support a claim of effect:

| Contrast | Δ | Δ% | 95% CI | p |
| --- | --- | --- | --- | --- |
| `image+naive` vs `image` | +0.0030 | +0.7% | [−0.0042, +0.0095] | 0.373 |
| `image+MGPL` vs `image` | +0.0068 | +1.5% | [−0.0022, +0.0154] | 0.144 |
| **`image+MGPL` vs `image+naive`** | **+0.0038** | **+0.9%** | **[−0.0049, +0.0122]** | **0.391** |

**None of these is significant, and the contrast that isolates grounding from cropping — MGPL versus
naive — is the weakest of the three.** A defect affecting 25.5% of inputs, corrected in a way that is
qualitatively unambiguous (Section 5.1), yields a downstream effect of +0.9% that we cannot
distinguish from zero at n = 300.

The same ladder viewed through the encoder-controlled modality contrast is more suggestive but no more
conclusive:

| Localization level | vs `text-siglip` | Δ% | 95% CI | p |
| --- | --- | --- | --- | --- |
| `image` (no crop) | +0.0226 | +5.5% | [−0.0038, +0.0491] | 0.102 |
| `image+naive` | +0.0256 | +6.2% | [−0.0006, +0.0518] | 0.057 |
| `image+MGPL` | +0.0293 | +7.1% | [+0.0023, +0.0559] | **0.033** |

The modality contrast strengthens monotonically as localization improves, and crosses the
conventional threshold only in the MGPL condition. We caution explicitly against over-reading this.
The naive condition sits at p = 0.057; the difference between p = 0.057 and p = 0.033 is not itself
evidence of anything. An earlier draft of this work claimed that product-grounded localization is
*what makes* the modality effect detectable. The three-level ablation does not support that claim, and
we retract it. The defensible statement is:

> Ranking quality increases monotonically with localization quality, but the effect is small relative
> to per-query variance, and no individual localization step is statistically demonstrable at this
> sample size.

Two further observations survive:

**No encoder gap confounds the comparison.** `text-siglip` and `text-jina` are statistically
indistinguishable (−0.8%, p = 0.84) despite different architectures, objectives and parameter counts,
so the image-versus-text difference cannot be attributed to encoder quality.

**Fusion is the only system that clearly separates.** It exceeds `text-jina` by +10.9% and the best
single image arm by +6.1%, both at p < 0.001 — an order of magnitude more evidence than any
localization contrast.

The system ordering is identical under the raw-CTR robustness labels, so it is not an artefact of the
labelling scheme.

Full paired-bootstrap contrasts on NDCG@10:

| Contrast | Δ | Δ% | 95% CI | p | Win rate |
| --- | --- | --- | --- | --- | --- |
| `fusion` vs `text-jina` | +0.0456 | +10.9% | [+0.0296, +0.0619] | <0.001 | 64% |
| `fusion` vs `image` | +0.0264 | +6.1% | [+0.0109, +0.0432] | <0.001 | 59% |
| `image+MGPL` vs `text-siglip` | +0.0293 | +7.1% | [+0.0023, +0.0559] | 0.033 | 57% |
| `image+naive` vs `text-siglip` | +0.0256 | +6.2% | [−0.0006, +0.0518] | 0.057 | 54% |
| `image` vs `text-siglip` | +0.0226 | +5.5% | [−0.0038, +0.0491] | 0.102 | 53% |
| `image+MGPL` vs `text-jina` | +0.0259 | +6.2% | [+0.0015, +0.0492] | 0.039 | 57% |
| `text-siglip` vs `text-jina` | −0.0034 | −0.8% | [−0.0297, +0.0240] | 0.843 | 49% |
| `image+MGPL` vs `image` | +0.0068 | +1.5% | [−0.0022, +0.0154] | 0.144 | 57% |
| `image+MGPL` vs `image+naive` | +0.0038 | +0.9% | [−0.0049, +0.0122] | 0.391 | — |
| `image+naive` vs `image` | +0.0030 | +0.7% | [−0.0042, +0.0095] | 0.373 | 54% |

Only the two fusion rows carry strong evidence. Everything concerning localization sits in the
inconclusive band.

### 5.3 Where each modality wins

Using the encoder-controlled contrast, so differences are modality and nothing else:

| Image wins most | Δ | Text wins most | Δ |
| --- | --- | --- | --- |
| spain world cup jersey | +0.88 | nike sabrina 3 | −0.78 |
| mexico jersey | +0.70 | sabrina | −0.75 |
| knicks jersey | +0.64 | ja 3 | −0.55 |
| norway | +0.58 | soccer goalie gloves | −0.53 |
| usa soccer jersey men | +0.56 | wagon | −0.50 |
| stanley 30 oz. | +0.49 | shin guards | −0.49 |

Images dominate where **the answer is a visual pattern**: national and club jerseys are colourways
and crests, which a photograph encodes directly and a title compresses to a team name. Text dominates
on **proper nouns that exist only in metadata** — signature athlete models (`nike sabrina 3`, `ja 3`)
cannot be read off pixels — and on **function-differentiated equipment** where many visually similar
objects serve different purposes (`shin guards`, `soccer goalie gloves`).

Query length does not predict the effect (1 word +0.083, 2 words −0.012, 3 words +0.056). Query
*type* does.

### 5.4 Complementarity, oracle headroom, and the failure of routing

If the two modalities fail on disjoint query classes, a per-query selector should outperform both. We
evaluate an oracle and two deployable routers.

| System | NDCG@10 | Δ vs best single | 95% CI | p |
| --- | --- | --- | --- | --- |
| **Oracle** (per-query best of two) | **0.5151** | **+16.3%** | [+0.0586, +0.0865] | <0.001 |
| `fusion` (fixed equal weight) | 0.4643 | +4.8% | [+0.0048, +0.0388] | 0.015 |
| `image+MGPL` (best single) | 0.4430 | — | — | — |
| Router: brand-token lexical | 0.4329 | −2.3% | [−0.0301, +0.0103] | 0.291 |
| Router: confidence margin | 0.4200 | −5.2% | [−0.0418, −0.0044] | 0.013 |

**The complementarity is large.** Per-query oracle selection is worth +16.3% over the best single
modality, far exceeding the modality effect itself. Image wins on 171 of 300 queries and text on 129,
so neither is dominant.

**Fixed fusion captures less than a third of it** — 29.5% of the oracle headroom. Equal-weight
blending is leaving substantial value unrealised.

**Both natural routers fail.** Lexical brand-token routing achieves 48% accuracy in predicting which
modality wins — indistinguishable from chance — and loses 2.3% against simply always using images.
Unsupervised confidence-margin routing, which selects whichever system's top candidate stands out
most from its pool, achieves **42% accuracy, significantly below chance**, and loses 5.2%
(p = 0.013).

The below-chance result is the most interesting. It implies that embedding confidence is
*anti-correlated* with cross-modal correctness: the modality that looks more certain about a query is
somewhat more likely to be the wrong one to use. A plausible mechanism is that tight score
distributions arise when a modality collapses a query onto a visually or lexically homogeneous
cluster, which is exactly when it is failing to discriminate. We do not consider this established,
but it predicts that confidence-based cascades — a common production pattern — will misroute in this
setting.

We therefore state retail modality routing as an open problem, with 0.5151 as the target and 0.4643
as the current fixed-fusion baseline.

### 5.5 Judgement-list audit: IPW placement matters

Our `production` reference ranks candidates by mean observed impression position. It scores 0.6176,
far above every embedding system, which invites the reading that the deployed ranker dominates. That
reading is wrong, and the reason generalises.

| Label set | corr(mean position, label) |
| --- | --- |
| Production LTR judgement list | **−0.291** |
| Raw CTR | −0.270 |

| LTR grade | Mean impression position |
| --- | --- |
| 4 | **16.0** |
| 3 | 21.9 |
| 2 | 26.3 |
| 1 | 29.3 |
| 0 | 28.3 |

Grades 1–4 are monotone in position. The judgement list applies inverse-propensity weights to
impressions **and** clicks and then takes weighted CTR. Because the weight varies by rank and by
observation age within a group, this re-weights *which observations count* but does not rescale the
CTR level, and consequently does not remove the position-to-relevance correlation: −0.29, against
−0.01 for an otherwise identical click-only correction.

Two consequences. Comparisons **among** embedding systems remain valid, since none of them observes
position. Comparisons **against** a position-derived baseline are circular and must be excluded — a
system ranking by position is predicting the dominant cause of its own labels.

We propose the residual position-label correlation as a routine reporting requirement for
click-derived judgement lists. It is a one-line diagnostic that distinguishes a debiased label set
from one that merely applies a debiasing-shaped transformation.

---

## 6. Systems considerations

### 6.1 Query-time cost

| Component | p50 | p95 |
| --- | --- | --- |
| SigLIP text tower (query encode) | 7.17 ms | 7.54 ms |
| Jina v5 text nano (query encode) | 17.18 ms | 19.06 ms |
| Cosine + top-10 over 111-item pool | 0.004 ms | 0.004 ms |
| Cosine + top-10 over 149,801 items | 3.08 ms | 3.26 ms |

Image encoding never occurs at query time; product vectors are precomputed. Query-time cost is
therefore entirely text encoding. Brute-force scoring of the full 149,801-product catalogue takes
3.1 ms, so an approximate index is unnecessary at this scale.

### 6.2 Index footprint

768-dimensional float32 vectors: 3,072 bytes per product per representation, or 439 MB for the full
catalogue. Fusion requires two representations.

### 6.3 Indexing cost

| Stage | Throughput | Full catalogue |
| --- | --- | --- |
| Text embeddings | 591 items/s | 0.07 h |
| Image embeddings | 21 items/s | 1.98 h |
| MGPL with OWLv2 | 4 items/s | **10.2 h** |
| MGPL with OWL-ViT v1 | 26 items/s | 1.58 h |

This is the principal practical cost of the method: open-vocabulary localization is roughly five
times the cost of the embeddings it feeds. The cheaper detector reduces this to below the embedding
cost, at unmeasured accuracy loss. A category-conditional policy — applying MGPL only to on-model
categories, where the failure mode actually occurs — is the obvious optimisation and is untested.

---

## 7. Limitations

1. **Single retailer, single vertical.** All data is from one sporting-goods catalogue. Person
   contamination rates and the modality split will differ in categories with less on-model imagery.
2. **Head queries only.** The 300 queries are high-volume terms surviving the production judgement
   list's group-size filters. Tail and attribute-heavy queries — where image embeddings should help
   most — are absent, so the effect is likely measured at its weakest.
3. **Marginal significance throughout the localization analysis.** No pairwise localization contrast
   reaches significance. The single modality contrast that does (p = 0.033) sits alongside a naive
   condition at p = 0.057, and no multiple-comparison correction was applied across the ten contrasts
   reported. The oracle and routing results (p < 0.001 and p = 0.013) are considerably stronger.
4. **Underpowered for the localization question.** With 300 queries and per-query NDCG variance
   dominating, an effect of +0.9% is far below what this design can resolve. A power analysis for the
   observed effect size implies several thousand queries would be needed. The negative result should
   be read as "not demonstrable here", not "demonstrably absent".
5. **Click-derived labels.** No human relevance judgements. Labels inherit exposure bias from the
   deployed ranker; Section 5.5 shows this is only partly corrected.
6. **Re-ranking, not retrieval.** Candidate pools are products the production system already
   surfaced. We make no claim about full-catalogue recall.
7. **One image per product.** No multi-view, alternate-angle, or in-context imagery. Product-grounded
   localization may matter more when selecting *among* images than when cropping a single one.
8. **Sensitivity to labelling.** Conclusions shifted materially when the judgement-list definition
   changed during development, which is itself evidence that single-configuration embedding results
   should be treated as provisional.

---

## 8. Conclusion

We set out to test whether image embeddings beat text embeddings for e-commerce search, and whether
product-grounded localization is a precondition for measuring that fairly. The first question we can
answer weakly in the affirmative; the second we cannot answer in the affirmative at all, and that is
the more useful outcome.

Catalogue imagery is genuinely contaminated: a class-agnostic detector localizes a person rather than
the product in 25.5% of a 19,468-image catalogue, and assigns roughly 4,400 sporting-goods images to
classes like `umbrella`, `airplane` and `cake`. MGPL fixes this at the level of the input, resolving
90.7% of the catalogue with correct product types drawn from the items' own titles. But the
three-level ablation shows the downstream ranking benefit is +0.9% over naive cropping at p = 0.39 —
indistinguishable from zero. **A visually obvious, quantitatively large defect in the input produced
an effect we could not resolve with 300 queries.** Practitioners should calibrate expectations for
preprocessing investments accordingly, and we retract the stronger claim made in an earlier draft
that grounding is what makes the modality effect detectable.

The robust findings are elsewhere. Per-query modality selection is worth +16.3% over the best single
system, far exceeding any effect we measured from the modality or its preprocessing, yet fixed fusion
captures under a third of it and two natural routers perform at or below chance — confidence-margin
routing is significantly *worse* than always choosing images, implying that embedding confidence is
anti-correlated with cross-modal correctness. Separately, the inverse-propensity scheme in our
production judgement list leaves a −0.29 position-label correlation, which silently invalidates a
comparison practitioners are likely to make.

For practitioners we draw three conclusions. Fusion, not modality selection and not preprocessing, is
the defensible near-term investment: it is the only intervention here with p < 0.001. Modality
routing is worth research attention because the headroom is large and unclaimed. And the residual
position-label correlation of any click-derived judgement list should be reported as a matter of
course, because a label set can apply inverse-propensity weights and remain substantially
position-determined.

---

## References

Akram, M. K., Sturua, S., Havriushenko, N., Herreros, Q., Günther, M., Werk, M., & Xiao, H. (2026).
*jina-embeddings-v5-text: Task-Targeted Embedding Distillation.* arXiv:2602.15547.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020).
*End-to-End Object Detection with Transformers.* ECCV.

Järvelin, K., & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques.*
ACM TOIS 20(4).

Joachims, T., Swaminathan, A., & Schnabel, T. (2017). *Unbiased Learning-to-Rank with Biased
Feedback.* WSDM.

Minderer, M., Gritsenko, A., Stone, A., Neumann, M., Weissenborn, D., Dosovitskiy, A., et al. (2022).
*Simple Open-Vocabulary Object Detection with Vision Transformers.* ECCV.

Minderer, M., Gritsenko, A., & Houlsby, N. (2023). *Scaling Open-Vocabulary Object Detection.*
NeurIPS.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. (2021).
*Learning Transferable Visual Models From Natural Language Supervision.* ICML.

Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). *Sigmoid Loss for Language Image
Pre-Training.* ICCV. arXiv:2303.15343.

---

## Appendix A — Reproducibility

Pipeline stages, all deterministic given a fixed date window and seed 20260808:

| Stage | Script | Output |
| --- | --- | --- |
| 1 | `01_build_test_set.py` | LTR judgement list |
| 2 | `02_download_images.py` | Catalogue imagery at 512×512 |
| 2b | `02b_crop_objects.py` | MGPL crops + per-image prompt/score manifest |
| 3 | `03_embed.py` | SigLIP query/title/image/crop, Jina query/title |
| 4 | `04_evaluate.py` | Per-query metrics, summary, paired-bootstrap significance |
| 5 | `05_report.py` | Figures and generated report |
| 7 | `07_benchmark_latency.py` | Online and offline latency |
| 8 | `08_modality_analysis.py` | Oracle, fusion, routers, complementarity |

Models: `google/siglip-base-patch16-512` (203M), `jinaai/jina-embeddings-v5-text-nano` (239M, CC
BY-NC 4.0), `google/owlv2-base-patch16-ensemble`, `facebook/detr-resnet-50`. Comparable parameter
counts and identical 768-d output for the two embedding models, so neither has a capacity advantage.

Hardware: Apple M-series, MPS backend, PyTorch 2.13.
