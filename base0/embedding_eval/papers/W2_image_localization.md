# W2 — Is the Image Signal Intact? Product Localization in Catalogue Photography

**Report 2 of 4** · DS eCom Search Ranking · 2026
Scope: audit what a catalogue photograph actually contains, build a product-grounded cropping
method, and measure whether better localization improves ranking.

---

## Summary

W1 left the image system reading whole catalogue frames. This week we asked whether that frame
actually contains the product in a usable form.

It frequently does not. Across the full 19,468-image catalogue, a class-agnostic object detector
selects a **person** as the dominant object in **25.5%** of images — **40.0%** of the images where it
fires at all — because catalogue photography is routinely shot on-model and a human is legitimately
the largest thing in the frame. A further **2,462 images (12.6%)** are assigned to classes that
cannot be the product at all — `airplane`, `vase`, `cake` — because its fixed taxonomy contains no
shoe, jersey or glove. Only 13.4% of detections land on a class that is both plausible and specific.

We built **MGPL** (Metadata-Guided Product Localization), a training-free method that derives a
product-type phrase from the item's own title and taxonomy and conditions an open-vocabulary detector
on it. MGPL resolves 90.7% of the catalogue with qualitatively correct object selection and tightens
median retained frame area from 0.583 to 0.466.

**And it barely moves the metric.** A three-level ablation gives +0.9% NDCG@10 for MGPL over naive
cropping, at p = 0.391. A large, visually unambiguous defect affecting a quarter of inputs produces a
downstream effect we cannot distinguish from zero at n = 300. This is the week's main finding and it
is negative.

---

## 1. Motivation

Benchmark image datasets are object-centric: the subject is centred and dominant. Retail catalogue
imagery is not. It is a mix of studio pack-shots on flat backgrounds and lifestyle photography where
a model wears or holds the merchandise.

Standard practice crops to the dominant detected object before encoding. The concern is mechanical:
if a person occupies most of the frame, "dominant object" selects the person, and the resulting
embedding describes a model rather than the product. If that happens often, an image-embedding
evaluation is measuring a degraded modality and is biased toward a null result.

### Models in this pipeline

Detection and embedding are **entirely separate models**. The detector never touches the embedding;
its only job is deciding which pixels the encoder sees.

```
image → [detector] → crop box → [SigLIP image tower] → 768-d vector → cosine vs query
```

| Role | Checkpoint | Type | Threshold |
| --- | --- | --- | --- |
| Detection — naive baseline | `facebook/detr-resnet-50` | Closed-set, 91 COCO classes | 0.7 |
| Detection — MGPL | `google/owlv2-base-patch16-ensemble` | Open-vocabulary, text-conditioned | 0.12 |
| Embedding — image and text towers | `google/siglip-base-patch16-512` | Dual-tower, shared space | — |
| Embedding — text comparator | `jinaai/jina-embeddings-v5-text-nano` | Text retrieval | — |

SigLIP produces one vector per image and has **no spatial output at all** — it cannot localise
anything, which is why a separate detector is required. Overridable via `DETECTOR_MODEL`, `OWL_MODEL`,
`SIGLIP_MODEL`, `JINA_MODEL`.

OWLv2 is internally CLIP-like, scoring candidate boxes against text embeddings, which is what lets it
be prompted with `"versatile shorts"`. DETR cannot do this — it is a fixed classifier over 91 COCO
classes, the exact limitation the census below exposes. The thresholds differ because open-vocabulary
match scores and closed-set softmax confidences are not calibrated on the same scale.

---

## 2. Measuring the failure mode

**Code:** [`02b_crop_objects.py`](../02b_crop_objects.py) `--method detr_largest` — `detr_box()` with `suppress_person=False` reproduces the naive baseline → `crop_manifest_naive.csv`

We ran class-agnostic largest-area DETR detection over the entire catalogue and recorded the selected
class. This is a census, not a sample.

**How the class is determined.** Nothing here involves our judgement. DETR emits 100 object queries,
each with a box and logits over its fixed taxonomy.
`post_process_object_detection` softmaxes those logits, drops the "no object" slot, takes the argmax
class per box, and discards anything below threshold. The integer class id is mapped through
`model.config.id2label` — a dictionary shipped with the checkpoint — and the winning box is chosen by
area:

```python
names = [model.config.id2label[int(i)] for i in labels]
keep  = list(range(len(names)))          # suppress_person=False in the naive arm
areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
j     = max(keep, key=lambda i: areas[i])
```

`names[j]` is written to the manifest's `prompt` column. The census is a `value_counts()` over that
column filtered to `method == 'detr'`. The reported classes are therefore DETR's own predictions, at
its own threshold, on every image — the only choice we make is "largest box", which is the
industry-standard rule under audit.

> **Manifest caveat.** The `prompt` column is overloaded: for `owlv2` rows it holds *our* conditioning
> phrase, for `detr` rows it holds *DETR's* predicted class. Always filter by `method` before counting.

### Table 1 — which tier resolved each image

**Model:** `facebook/detr-resnet-50`, threshold 0.7, `suppress_person=False`.
**Source:** `crop_manifest_naive.csv`, one row per ecode, `method` column.
**Derivation:** `manifest['method'].value_counts()` over all 19,468 rows. The `method` value records
which cascade tier produced the final box, so this table is a complete partition of the catalogue —
every image appears exactly once.

| Selection outcome | Count | Share of catalogue |
| --- | --- | --- |
| Detector fired | 12,411 | 63.8% |
| — of which **`person`** | **4,967** | **25.5%** |
| Fell through to background trim | 6,341 | 32.6% |
| No usable box | 716 | 3.7% |

**A person is the dominant object in a quarter of all catalogue images, and 40.0% of images where
detection succeeds.**

### Table 2 — what the detector thought it was looking at

**Model:** same DETR run as Table 1 — no second inference pass.
**Source:** `crop_manifest_naive.csv`, `prompt` column filtered to `method == 'detr'` (12,411 rows,
75 distinct classes).
**Derivation:** `value_counts()` on that column, then each class assigned to one of three buckets by
whether the label could describe a real DSG product. **This bucketing is our judgement, not the
model's** — unlike Table 1, it is not reproducible without the class lists, so they are stated in
full:

| Bucket | Membership rule |
| --- | --- |
| Plausibly correct | `baseball glove`, `baseball bat`, `sports ball`, `surfboard`, `skateboard`, `skis`, `snowboard`, `tennis racket`, `frisbee`, `bicycle`, `backpack`, `bottle` |
| Ambiguous | `umbrella`, `suitcase`, `kite`, `handbag`, `tie`, `chair`, `bench`, `clock` |
| Impossible | the remaining 54 classes |

Partitioning all 75 selected classes on that basis:

| Bucket | Images | Share of detections | Examples |
| --- | --- | --- | --- |
| `person` | 4,967 | 40.0% | — |
| **Impossible** — cannot be the product | **2,462** | **19.8%** | airplane 468, cake 362, vase 232, motorcycle 186, bed 103, clock 73, traffic light 58, stop sign 57, parking meter 45 |
| **Ambiguous** — DSG does sell such items | 3,315 | 26.7% | umbrella 1,194, suitcase 717, kite 468, handbag 456, tie 207 |
| Plausibly correct | 1,667 | 13.4% | baseball glove 551, backpack 276, snowboard 248, baseball bat 118 |

**2,462 images — 12.6% of the catalogue — are assigned to classes that cannot possibly be the
product.** These are not near-misses: `airplane`, `cake` and `vase` indicate the label space contains
no representation of the item, so the detector emits the nearest available concept.

The ambiguous bucket is deliberately not counted as error. DSG genuinely sells golf umbrellas, duffel
bags and backpacks, so `umbrella` or `suitcase` may well be correct; the label alone cannot decide,
and resolving it would need per-image inspection we have not done. Only 13.4% of detections land on a
class that is both plausible and specific.

A COCO-trained detector is therefore not merely imperfect on retail imagery — for most of the
catalogue its label space does not contain the products at all.

Selecting by *confidence* instead of area does not help: `person` is also the class such a detector
is most confident about in these images.

> **Reproducing both tables.** `python 02b_crop_objects.py --method detr_largest` writes
> `crop_manifest_naive.csv`; the two tables are `value_counts()` over `method` and over `prompt`
> filtered to `method == 'detr'`. Restrict to the 19,468 W2 ecodes first — the manifests are shared
> across experiment variants via `EMBEDDING_EVAL_IMAGE_STORE` and now contain additional products
> from later runs.

---

## 3. Method: Metadata-Guided Product Localization

**Code:** [`02b_crop_objects.py`](../02b_crop_objects.py)

### 3.1 Core idea

The information needed to disambiguate is already in the catalogue. Every product has a title and a
taxonomy leaf stating what it is. Condition the detector on that text.

### 3.2 Product phrase extraction

**Code:** [`02b_crop_objects.py`](../02b_crop_objects.py) — `product_phrases()`, which strips audience tokens (`Men's`, `Kids'`)

Retail titles place brand and model first, product type last:

```
Nike Men's Dri-FIT Challenger 5" Brief-Lined Versatile Shorts
                                              ^^^^^^^^^^^^^^^^
```

We take the title's final tokens, remove **audience tokens**, and emit the trailing bigram and
unigram plus the taxonomy leaf:

$$P(t,c) = \{\text{bigram}(W(t) \setminus A),\ \text{unigram}(W(t) \setminus A),\ \text{leaf}(c) \setminus A\}$$

$$A = \{\text{men's, women's, boys', girls', kids', youth, junior, toddler, infant, unisex}, \dots\}$$

**Removing $A$ is essential, not cosmetic.** Tokens like `Women's` are semantically *about a person*.
Leaving them in the conditioning text pulls the detector's text embedding toward the human in the
frame — the exact failure the method exists to prevent.

Example output: `["versatile shorts", "shorts"]`.

### 3.3 Selection rule

**Code:** [`02b_crop_objects.py`](../02b_crop_objects.py) — `owlv2_box()` (highest-scoring phrase-conditioned box), `usable()` (area bounds), `square_pad()`

$$b_{\text{MGPL}} = \arg\max_{(b,s) \in D(I, P(t,c))} s$$

Selection is by **text-match score**, not area. Area is precisely the wrong criterion — the person is
legitimately the largest object.

### 3.4 Cascade

**Code:** [`02b_crop_objects.py`](../02b_crop_objects.py) — `main()` tier order `owlv2` → `detr` (person-suppressed) → `background_box()` → full frame

Open-vocabulary detection is expensive and does not always fire, so MGPL takes the first acceptable
result from:

| Tier | Rule |
| --- | --- |
| 1. `owlv2` | Product-conditioned box, highest text-match score |
| 2. `detr` | Fixed taxonomy with **`person` suppressed**; largest remaining box |
| 3. `bgtrim` | Background colour from the border ring; bbox of pixels differing beyond a threshold |
| 4. `full` | Original frame |

Acceptable = box occupies 2–98% of frame. Accepted boxes are padded 8%, squared, clamped, resampled
LANCZOS to 512×512. The manifest records the winning phrase and score per image, so every crop is
auditable.

Tier 3 needs no model and handles flat studio backgrounds; tier 2 exists because suppressing
`person` makes a cheap fixed-taxonomy detector usable as an intermediate.

---

## 4. MGPL behaviour

**Code:** [`02b_crop_objects.py`](../02b_crop_objects.py) → `crop_manifest.csv` (tier, box, area fraction per ecode)

| Tier | Images | Share |
| --- | --- | --- |
| `owlv2` | **17,656** | **90.7%** |
| `bgtrim` | 920 | 4.7% |
| `detr` (person suppressed) | 804 | 4.1% |
| `full` | 88 | 0.5% |

Median retained frame area falls **0.583 → 0.466**. Most frequent conditioning phrases are `cleats`
(1,039) and `shirt` (647) — the method recovers real product types from free text without a curated
taxonomy.

### What the area fractions mean

Every fraction in this section is **box area divided by full image area**, so lower means tighter. A
value of 0.57 means the box covered 57% of the frame. Two different fractions are recoverable from
the manifest and they are not interchangeable:

| Quantity | Manifest column | Definition |
| --- | --- | --- |
| **Detection box** | `area_frac` | `(x1-x0)(y1-y0) / (W·H)` on the box the detector returned |
| **Saved crop** | derived from `box` | the same box after `square_pad()` — expanded 8%, squared to the longer side, clamped to the image |

The encoder never sees the detection box. It sees the saved crop, which is always larger:

| | Detection box | Saved crop |
| --- | --- | --- |
| Median | 0.456 | **1.000** |
| Mean | 0.499 | 0.819 |

**The median saved crop is the entire frame.** Squaring alone expands the median box by 1.62×, and
for any roughly centred object it clamps out to the full image. This has a direct consequence for
Experiment 2 and is picked up in §6.

### Where MGPL relocates — and where it over-crops

On images where largest-area selection had chosen a person, MGPL typically relocates to the garment:

| Product | Person box area % | MGPL box area % | phrase |
| --- | --- | --- | -- |
| Dri-FIT Challenger Shorts | 0.85 | 0.57 | `versatile shorts` |
| Premium Essential T-Shirt | 0.74 | 0.53 | `t shirt` |
| Insulated Golf Vest | 0.63 | 0.44 | `jackets outerwear` |
| Swim Knit Cover-Up Pant | 0.57 | 0.46 | `pant` |
| Pinnacle Crewneck Sweatshirt | 0.55 | 0.37 | `crewneck sweatshirt` |
| Spacer Track Pants | 0.47 | 0.36 | `track pants` |
| Featherweight Golf Polo | 0.44 | 0.26 | `polo` |

**This table is not representative of the tail.** It was selected on moderate area reductions. Sorting
instead by *largest* reduction surfaces a failure mode: MGPL sometimes locks onto a fragment — a patch
of jersey fabric, one glove, a corner of a garment — and discards the product's shape entirely.

| Saved-crop area | Images | Share of `owlv2` crops |
| --- | --- | --- |
| < 5% | 114 | 0.6% |
| < 10% | **294** | **1.7%** |
| < 20% | 812 | 4.6% |

OWLv2's own match score corroborates it: median **0.457** on crops under 10% of frame versus **0.558**
on those at or above 40%. The detector was least confident exactly where it cropped hardest, and `usable()`
only rejects below 2%, so a 3% box passes unchallenged. Raising the lower bound to roughly 15%, or
preferring the larger of several high-scoring boxes, is the obvious fix and is untested.

Side-by-side examples — originals, naive crops and MGPL crops — are rendered to
`results/fig_localization_examples.png`.

At the level of the **input**, MGPL does what it was designed to do on the large majority of images,
with a measured 1.7% over-crop rate as the cost.

---

## 5. Experiment 2 — The localization ladder

**Code:** [`03_embed.py`](../03_embed.py) (three image variants) · [`04_evaluate.py`](../04_evaluate.py) (`siglip_image`, `siglip_image_naive`, `siglip_image_crop`)

### 5.1 Design

Three image arms, identical encoder, identical source photographs, differing only in cropping:

| Arm | Localization |
| --- | --- |
| `image` | None — raw frame |
| `image+naive` | Largest-area detection, `person` **not** suppressed |
| `image+MGPL` | Product-conditioned |

This separates two claims that are easy to conflate: *"cropping helps"* and *"product-grounded
cropping helps"*.

### 5.2 Results

| Arm | NDCG@10 |
| --- | --- |
| `image` | 0.4363 |
| `image+naive` | 0.4393 |
| `image+MGPL` | 0.4430 |

Monotone in localization quality — the ordering the method predicts. But:

| Contrast | Δ | Δ% | 95% CI | p |
| --- | --- | --- | --- | --- |
| `image+naive` vs `image` | +0.0030 | +0.7% | [−0.0042, +0.0095] | 0.373 |
| `image+MGPL` vs `image` | +0.0068 | +1.5% | [−0.0022, +0.0154] | 0.144 |
| **`image+MGPL` vs `image+naive`** | **+0.0038** | **+0.9%** | **[−0.0049, +0.0122]** | **0.391** |

**No contrast is significant, and the one isolating grounding from cropping is the weakest.**

Viewed through the modality contrast against `text-siglip`:

| Localization | Macro Δ% | p | **Weighted Δ** | **p** |
| --- | --- | --- | --- | --- |
| `image` | +5.5% | 0.102 | +0.0627 | **0.006** |
| `image+naive` | +6.2% | 0.057 | +0.0629 | **0.007** |
| `image+MGPL` | +7.1% | **0.033** | +0.0717 | **0.011** |

### 5.3 Reading — including a retraction

Under macro averaging the modality contrast strengthens monotonically and crosses p < 0.05 only under
MGPL. An earlier draft concluded that **product-grounded localization is what makes the modality
effect detectable**.

**That claim is retracted, and impression weighting closes the case against it.** Weighted by traffic,
all three localization levels clear significance — including *no cropping at all* (p = 0.006). The
modality effect does not depend on localization in any form. The macro-average pattern that suggested
otherwise was an artefact of weighting every query equally.

What remains true:

> Ranking quality increases monotonically with localization quality, but the effect is small relative
> to per-query variance, no individual localization step is statistically demonstrable under either
> weighting, and the modality effect it was thought to enable is present without it.

### 5.4 Where cropping helps and hurts

**Code:** [`04_evaluate.py`](../04_evaluate.py) → `results/per_query_metrics.csv`, differenced per `search_term`

| Cropping helps | Δ | Cropping hurts | Δ |
| --- | --- | --- | --- |
| uggs | +0.36 | girls swimsuit | −0.43 |
| walter hagen womens | +0.28 | ankle brace | −0.25 |
| mexico soccer jerseys | +0.23 | nike socks | −0.23 |
| womens nike shorts | +0.23 | umbrella | −0.23 |
| womens one piece swimsuit | +0.21 | girls shorts | −0.19 |

Losses concentrate on apparel, where a tight crop discards silhouette and fit cues that a shopper is
actually selecting on. Gains concentrate on items with a distinctive object or colourway. A
category-conditional policy — crop hard goods and jerseys, leave apparel wide — is the obvious
refinement and is untested.

---

## 6. Why the effect is so small

Three candidate explanations, none verified:

1. **SigLIP may be robust to framing.** Trained on web images with arbitrary composition, it may
   already handle off-centre subjects, making tighter crops redundant.
2. **Head queries may not need fine visual detail.** W1's queries are high-volume terms where coarse
   category signal may suffice; a person wearing running shoes still signals "running shoes".
3. **The design may be underpowered.** At ±6% resolution (W1 §5.3), a +0.9% effect is far below
   threshold. A power calculation for the observed effect size implies several thousand queries.
4. **The rungs of the ladder are closer than the design assumed.** Because `square_pad()` squares and
   clamps every box, the **median saved crop is the full frame** (§4). For more than half the
   catalogue, `image` and `image+MGPL` hand SigLIP identical pixels. The ablation cannot measure a
   difference on images where no difference exists, so the contrast is diluted by construction.

Explanations 3 and 4 are each sufficient on their own, and 4 is the more actionable: it is a defect in
the experiment, not in the method. A re-run that skips squaring, or reports the contrast only on
images whose crops actually differ, would test MGPL far more sharply. The result should be read as
**"not demonstrable here"**, not **"demonstrably absent"**.

---

## 7. Cost

**Code:** [`07_benchmark_latency.py`](../07_benchmark_latency.py) → `results/latency.csv`

| Stage | Throughput | Full catalogue (149,801) |
| --- | --- | --- |
| Image embeddings | 21 items/s | 1.98 h |
| MGPL with OWLv2 | 4 items/s | **10.2 h** |
| MGPL with OWL-ViT v1 | 26 items/s | 1.58 h |

Product-conditioned localization with OWLv2 costs roughly **five times the embeddings it feeds**.
Given a +0.9% unverified benefit, catalogue-wide deployment is not justified on this evidence. If
pursued, it should be scoped to on-model categories where the failure mode actually occurs, or run
with the cheaper detector.

---

## 8. Conclusions

1. Person contamination in catalogue imagery is **real and large**: 25.5% of images, 40% of detector
   hits. This is a census, not an estimate.
2. MGPL **fixes it at the input level** — 90.7% resolution, correct product types, tighter crops.
3. The **downstream ranking benefit is not demonstrable** at n = 300: +0.9% over naive cropping,
   p = 0.391.
4. We **retract** the claim that grounding is what makes the modality effect detectable.
5. **Practical guidance:** do not fund catalogue-wide product-grounded cropping on relevance grounds.
   The input defect is real, but its measured value is bounded and small.

---

## 9. What this hands to W3

Two weeks of work have produced no significant modality result in either direction. The per-query
data, though, shows something the averages hide: image and text win on visibly different query types.
That points away from *choosing* a modality and toward *combining* them.

W3 quantifies how much is available from combination, and whether it can be captured.

---

## Appendix — Reproduction

```bash
python 02b_crop_objects.py                        # MGPL cascade
python 02b_crop_objects.py --method detr_largest  # naive ablation arm
python 03_embed.py --force
python 04_evaluate.py
```

Detector: `google/owlv2-base-patch16-ensemble`, threshold 0.12; fallback `facebook/detr-resnet-50`,
threshold 0.7. Override with `OWL_MODEL` / `DETECTOR_MODEL`.

Manifests `crop_manifest.csv` and `crop_manifest_naive.csv` record method, phrase, score and box per
image.

**Note.** Resolve image paths from configuration, not from stored manifest paths — the latter go
stale whenever the data directory moves, and fail silently as "unreadable image" errors.
