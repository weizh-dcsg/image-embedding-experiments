# W7 — Which Image Encoder? SigLIP vs Jina v5 Omni

Generated 2026-08-28 by `11_image_encoder_experiment.py` / `12_w7_report.py`.

> **TL;DR**
> Best image encoder, macro: **`siglip-image`** (NDCG@10 0.4272).
> Best image encoder, impression-weighted: **`siglip-image`** (NDCG@10 0.4443).
> `siglip-image` is ahead of `omni-nano-image` by 0.0958 NDCG@10 (significant, p = 0.0000); `siglip-image` is ahead of `omni-small-image` by 0.1527 NDCG@10 (significant, p = 0.0000); `siglip-image` is ahead of `siglip-ES version` by 0.1873 NDCG@10 (significant, p = 0.0000).
> Reference: the best image encoder is **ahead of**
> `attr-siglip` (0.3776 macro), whose earlier apparent strength was a tie-ordering
> artefact — see the correction in `papers/w1-section5.md`.

---

## 1. What is being compared

W1–W6 used exactly **one** image encoder, so every image-side conclusion in the series rested on a
single checkpoint. W7 removes that single point of failure by adding two independent image towers.

| System | Query encoder | Document representation | Params | Dim |
| --- | --- | --- | --- | --- |
| `siglip-image` | SigLIP text tower | SigLIP image tower over the photo | 203M | 768 |
| `siglip-ES version` | Elasticsearch-deployed SigLIP text tower | Existing local SigLIP image vectors | 203M | 768 |
| `omni-nano-image` | Jina v5 omni text | Jina v5 omni vision tower | ~1.0B | 768 |
| `omni-small-image` | Jina v5 omni text | Jina v5 omni vision tower | larger | 1024 |

**Each system is self-consistent** — query and document are encoded by the same model, into the same
space. So the contrast isolates the *image encoder*, not the query representation. This is the same
design discipline as W1's `text-siglip` control (SigLIP's own text tower, reused from the image system).

Reference points, shown for scale but not image systems:
`attr-siglip` (best representation found in W1/W6), `production` (incumbent ordering), `random`.

### Retrieval prefixes

The omni models require `Query: ` / `Document: ` prefixes on **every** modality, not just text.
Queries are encoded as `"Query: {term}"`; product images as `images=<photo>, text="Document: <image>"`.
Omitting these puts the model off-distribution.

### Scale and fairness

The omni models are ~1B-param VLMs, 3–5× slower per image than SigLIP, so W7 runs on a seeded,
**tier-stratified subset**: 600 queries
(head 200 / torso 200 / tail 200)
over 28201 products.

A product is scored only if **every** system has a vector for it, and a query only if it retains
≥ 5 candidates and a positive label after that intersection — so all systems rank literally
identical pools. Images are encoded one per forward pass; batching measured only 1.17× on this
hardware while introducing ~3.7e-3 numerical drift, which is not a good trade in a study about
image-encoder fidelity.

Labels are the **corrected** LTR judgement list (τ applied to clicks only — see W4).

---

## 2. Results

Every metric at every cutoff (k = 5, 10, 20, 48, 96, 144), plus MAP. Bold marks the
best-scoring **image** encoder on NDCG@10, tracked consistently across all tables.

### Macro-averaged — every query counts once

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.4034** | **0.4272** | **0.4819** | **0.5779** | **0.6554** | **0.6847** |
| `production (incumbent ordering)` | 0.4025 | 0.4200 | 0.4571 | 0.5429 | 0.6298 | 0.6703 |
| `attr-siglip (text reference)` | 0.3477 | 0.3776 | 0.4335 | 0.5341 | 0.6122 | 0.6472 |
| `omni-nano-image` | 0.3011 | 0.3313 | 0.3879 | 0.4968 | 0.5823 | 0.6205 |
| `omni-small-image` | 0.2414 | 0.2745 | 0.3297 | 0.4394 | 0.5352 | 0.5794 |
| `random` | 0.2392 | 0.2640 | 0.3210 | 0.4357 | 0.5321 | 0.5758 |
| `siglip-ES version` | 0.2107 | 0.2398 | 0.3040 | 0.4244 | 0.5234 | 0.5681 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.1582** | **0.2655** | **0.4451** | **0.7246** | **0.9039** | **0.9698** |
| `production (incumbent ordering)` | 0.1400 | 0.2466 | 0.4189 | 0.6922 | 0.8846 | 0.9646 |
| `attr-siglip (text reference)` | 0.1381 | 0.2483 | 0.4198 | 0.7073 | 0.8872 | 0.9627 |
| `omni-nano-image` | 0.1226 | 0.2278 | 0.4071 | 0.6893 | 0.8800 | 0.9592 |
| `omni-small-image` | 0.1038 | 0.1997 | 0.3776 | 0.6550 | 0.8596 | 0.9501 |
| `random` | 0.0954 | 0.1807 | 0.3567 | 0.6540 | 0.8596 | 0.9509 |
| `siglip-ES version` | 0.0909 | 0.1811 | 0.3524 | 0.6520 | 0.8612 | 0.9528 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.7607** | **0.7383** | **0.7038** | **0.5945** | **0.4438** | **0.3426** |
| `production (incumbent ordering)` | 0.6303 | 0.6435 | 0.6313 | 0.5483 | 0.4265 | 0.3387 |
| `attr-siglip (text reference)` | 0.7020 | 0.7010 | 0.6718 | 0.5741 | 0.4304 | 0.3380 |
| `omni-nano-image` | 0.6433 | 0.6383 | 0.6249 | 0.5517 | 0.4233 | 0.3356 |
| `omni-small-image` | 0.5810 | 0.5772 | 0.5669 | 0.5105 | 0.4073 | 0.3300 |
| `random` | 0.5777 | 0.5777 | 0.5647 | 0.5094 | 0.4078 | 0.3306 |
| `siglip-ES version` | 0.5187 | 0.5348 | 0.5404 | 0.5044 | 0.4079 | 0.3311 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.8427** | **0.8467** | **0.8493** | **0.8500** | **0.8500** | **0.8500** |
| `production (incumbent ordering)` | 0.6940 | 0.7041 | 0.7063 | 0.7071 | 0.7072 | 0.7072 |
| `attr-siglip (text reference)` | 0.7843 | 0.7896 | 0.7913 | 0.7922 | 0.7922 | 0.7922 |
| `omni-nano-image` | 0.7388 | 0.7451 | 0.7481 | 0.7490 | 0.7490 | 0.7490 |
| `omni-small-image` | 0.6776 | 0.6853 | 0.6885 | 0.6893 | 0.6893 | 0.6893 |
| `random` | 0.6917 | 0.6965 | 0.7008 | 0.7022 | 0.7022 | 0.7022 |
| `siglip-ES version` | 0.6200 | 0.6326 | 0.6371 | 0.6386 | 0.6387 | 0.6387 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| **`siglip-image`** | **0.7500** |
| `production (incumbent ordering)` | 0.6808 |
| `attr-siglip (text reference)` | 0.7023 |
| `omni-nano-image` | 0.6664 |
| `omni-small-image` | 0.6079 |
| `random` | 0.6041 |
| `siglip-ES version` | 0.5978 |


### Impression-weighted — every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.4317** | **0.4443** | **0.4863** | **0.5844** | **0.6896** | **0.7387** |
| `attr-siglip (text reference)` | 0.4044 | 0.4185 | 0.4617 | 0.5529 | 0.6596 | 0.7175 |
| `production (incumbent ordering)` | 0.4066 | 0.4123 | 0.4295 | 0.5122 | 0.6384 | 0.7053 |
| `omni-nano-image` | 0.3511 | 0.3677 | 0.4040 | 0.5164 | 0.6310 | 0.6887 |
| `random` | 0.3183 | 0.3208 | 0.3505 | 0.4523 | 0.5797 | 0.6481 |
| `omni-small-image` | 0.2976 | 0.3168 | 0.3543 | 0.4617 | 0.5805 | 0.6492 |
| `siglip-ES version` | 0.2792 | 0.2948 | 0.3399 | 0.4477 | 0.5779 | 0.6448 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.0784** | **0.1552** | **0.2978** | **0.5881** | **0.8361** | **0.9462** |
| `attr-siglip (text reference)` | 0.0737 | 0.1470 | 0.2850 | 0.5719 | 0.8180 | 0.9370 |
| `production (incumbent ordering)` | 0.0609 | 0.1266 | 0.2517 | 0.5200 | 0.8002 | 0.9346 |
| `omni-nano-image` | 0.0674 | 0.1362 | 0.2694 | 0.5551 | 0.8101 | 0.9333 |
| `random` | 0.0645 | 0.1287 | 0.2543 | 0.5199 | 0.7870 | 0.9239 |
| `omni-small-image` | 0.0646 | 0.1292 | 0.2539 | 0.5252 | 0.7888 | 0.9236 |
| `siglip-ES version` | 0.0615 | 0.1235 | 0.2494 | 0.5221 | 0.7881 | 0.9226 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.9408** | **0.9377** | **0.9223** | **0.8429** | **0.6898** | **0.5613** |
| `attr-siglip (text reference)` | 0.8930 | 0.9018 | 0.8883 | 0.8166 | 0.6718 | 0.5535 |
| `production (incumbent ordering)` | 0.7419 | 0.7652 | 0.7678 | 0.7351 | 0.6519 | 0.5507 |
| `omni-nano-image` | 0.8205 | 0.8351 | 0.8369 | 0.7908 | 0.6613 | 0.5497 |
| `random` | 0.7927 | 0.7862 | 0.7858 | 0.7335 | 0.6397 | 0.5425 |
| `omni-small-image` | 0.7913 | 0.7903 | 0.7901 | 0.7422 | 0.6391 | 0.5420 |
| `siglip-ES version` | 0.7445 | 0.7416 | 0.7582 | 0.7319 | 0.6359 | 0.5402 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| **`siglip-image`** | **0.9796** | **0.9798** | **0.9799** | **0.9799** | **0.9799** | **0.9799** |
| `attr-siglip (text reference)` | 0.9431 | 0.9442 | 0.9442 | 0.9442 | 0.9442 | 0.9442 |
| `production (incumbent ordering)` | 0.7692 | 0.7784 | 0.7788 | 0.7788 | 0.7788 | 0.7788 |
| `omni-nano-image` | 0.8847 | 0.8860 | 0.8861 | 0.8862 | 0.8862 | 0.8862 |
| `random` | 0.8806 | 0.8825 | 0.8829 | 0.8829 | 0.8829 | 0.8829 |
| `omni-small-image` | 0.8695 | 0.8709 | 0.8709 | 0.8710 | 0.8710 | 0.8710 |
| `siglip-ES version` | 0.8372 | 0.8431 | 0.8439 | 0.8440 | 0.8440 | 0.8440 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| **`siglip-image`** | **0.9088** |
| `attr-siglip (text reference)` | 0.8753 |
| `production (incumbent ordering)` | 0.8035 |
| `omni-nano-image` | 0.8442 |
| `random` | 0.7970 |
| `omni-small-image` | 0.7981 |
| `siglip-ES version` | 0.7919 |


---

## 3. By query-volume tier

Macro NDCG@10 within each tier. Head queries are short and brand-heavy; tail queries are where
W5 showed every system degrades.

| Tier | siglip-image | omni-nano-image | omni-small-image | siglip-ES version |
| --- | --- | --- | --- | --- |
| head | 0.4300 | 0.3445 | 0.2893 | 0.2529 |
| torso | 0.4643 | 0.3576 | 0.2906 | 0.2655 |
| tail | 0.3872 | 0.2919 | 0.2436 | 0.2011 |

---

## 4. Significance

Paired bootstrap over queries (200 resamples), metric NDCG@10, LTR labels.
All pairwise image-encoder contrasts, plus each image encoder against the reference points.

| Contrast | Macro Δ | Macro p | 95% CI | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- |
| `siglip-image` vs `omni-nano-image` | +0.0958 | 0.0000 **sig.** | [+0.0759, +0.1167] | 62% | +0.0767 | 0.0000 |
| `siglip-image` vs `omni-small-image` | +0.1527 | 0.0000 **sig.** | [+0.1331, +0.1741] | 69% | +0.1275 | 0.0000 |
| `siglip-image` vs `siglip-ES version` | +0.1873 | 0.0000 **sig.** | [+0.1682, +0.2079] | 74% | +0.1495 | 0.0000 |
| `omni-nano-image` vs `omni-small-image` | +0.0568 | 0.0000 **sig.** | [+0.0427, +0.0751] | 56% | +0.0508 | 0.0200 |
| `omni-nano-image` vs `siglip-ES version` | +0.0915 | 0.0000 **sig.** | [+0.0725, +0.1098] | 65% | +0.0728 | 0.0000 |
| `omni-small-image` vs `siglip-ES version` | +0.0347 | 0.0000 **sig.** | [+0.0191, +0.0535] | 56% | +0.0220 | 0.1300 |
| `siglip-image` vs `attr-siglip (text reference)` | +0.0496 | 0.0000 **sig.** | [+0.0275, +0.0716] | 55% | +0.0258 | 0.0500 |
| `siglip-image` vs `production (incumbent ordering)` | +0.0072 | 0.6500 n.s. | [-0.0179, +0.0302] | 50% | +0.0321 | 0.8000 |
| `siglip-image` vs `random` | +0.1632 | 0.0000 **sig.** | [+0.1406, +0.1834] | 70% | +0.1236 | 0.0000 |
| `omni-nano-image` vs `attr-siglip (text reference)` | -0.0462 | 0.0000 **sig.** | [-0.0681, -0.0254] | 42% | -0.0508 | 0.0800 |
| `omni-nano-image` vs `production (incumbent ordering)` | -0.0887 | 0.0000 **sig.** | [-0.1112, -0.0648] | 38% | -0.0446 | 0.0000 |
| `omni-nano-image` vs `random` | +0.0674 | 0.0000 **sig.** | [+0.0482, +0.0872] | 56% | +0.0469 | 0.0000 |
| `omni-small-image` vs `attr-siglip (text reference)` | -0.1031 | 0.0000 **sig.** | [-0.1173, -0.0857] | 26% | -0.1017 | 0.0000 |
| `omni-small-image` vs `production (incumbent ordering)` | -0.1455 | 0.0000 **sig.** | [-0.1675, -0.1188] | 29% | -0.0954 | 0.0000 |
| `omni-small-image` vs `random` | +0.0105 | 0.1800 n.s. | [-0.0040, +0.0246] | 49% | -0.0039 | 0.6400 |
| `siglip-ES version` vs `attr-siglip (text reference)` | -0.1377 | 0.0000 **sig.** | [-0.1567, -0.1177] | 27% | -0.1237 | 0.0000 |
| `siglip-ES version` vs `production (incumbent ordering)` | -0.1801 | 0.0000 **sig.** | [-0.1975, -0.1566] | 24% | -0.1174 | 0.0000 |
| `siglip-ES version` vs `random` | -0.0241 | 0.0000 **sig.** | [-0.0432, -0.0077] | 39% | -0.0259 | 0.2300 |

---

## 5. Reading the result






**Capacity again fails to buy quality.** A ~1B-param multimodal model loses to a 203M dual-tower
model on this task, and by a wide margin. Combined with W1's text-side finding, parameter count is
not the axis that matters here.

**This is the check W1–W6 needed.** Every image-side result in the series had been measured with
SigLIP alone. Adding a second, independently built image tower tests whether those conclusions were
about *images* or about *SigLIP*. The gap measured here is large enough that image-side results
should be read as encoder-specific unless replicated.

**Two harness defects were found while producing this report** — a collapsed encoder that scored
near-perfect, and the tie-ordering leak behind it. Both are fixed and guarded. The broader lesson:
validate that an encoder emits distinct vectors before trusting any metric computed from it.

> **Licensing:** the Jina v5 omni checkpoints are **CC BY-NC 4.0**; commercial use requires a
> licence. They are research comparators here. SigLIP is Apache-2.0.

Source data: `results/w7_summary.csv`, `results/w7_summary_by_tier.csv`,
`results/w7_significance.csv`, `results/w7_per_query.csv`.
