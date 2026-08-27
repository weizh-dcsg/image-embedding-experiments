# Experiment 1 — What Should a Product Embedding Actually Read?

> # ⚠️ RETRACTED CONCLUSION — corrected 2026-08-20
>
> **The headline result below is wrong.** "Structured attributes win decisively" was an artefact
> of a tie-ordering bug in the evaluation harness, not a property of the representation.
>
> **The bug.** `sql/judgement_list.sql` emits each query's candidates ordered by `relevance DESC`.
> `np.argsort(kind="stable")` preserves input order on ties. Any system producing tied scores
> therefore reproduced the label ordering — it was reading the answer key.
>
> **Why it hit attributes specifically.** The Big-4 attribute string has only **17.8% distinct
> values** across the catalogue (257 products share `"Nike Cleats Soccer Adult"`), so **64.1%** of
> candidates in a typical pool receive an identical score — versus **1.2%** for image embeddings.
> The representation's inability to discriminate within a category, described in §3 below as a
> point in its favour, is exactly what let it cheat.
>
> **Corrected numbers** (macro NDCG@10, 6,536-query test set, corrected labels):
>
> | System | As published | Corrected | Δ |
> | --- | --- | --- | --- |
> | `attr-siglip` | 0.5116 | **0.3529** | −0.159 |
> | `attr-jina-small` | 0.4452 | **0.3244** | −0.121 |
> | `attr-jina` | 0.4351 | **0.3289** | −0.106 |
> | every non-attribute system | — | — | < 0.005 |
>
> **Attribute representations now rank LAST among real systems**, below every title-based arm.
> The corrected ordering is `fusion` 0.4289 > `production` 0.4194 > `image` 0.4103 >
> `text-jina` 0.3932 > `text-siglip` 0.3785 > `attr-siglip` 0.3529.
>
> **What still stands:** the same-encoder control design (§1), the model-capacity findings, the
> macro-versus-weighted distinction, and the observation that the labels reward category-level
> matching. **What does not:** every claim that attributes beat titles or photographs, and the
> recommendation to adopt `attr-siglip`.
>
> Fixed by `evaluate.shuffle_pool()`, applied in `04`/`09`/`11`/`13`. See `papers/W6`, `W7`, `W8`
> for corrected downstream results.

---

**Scope:** a controlled test of three ways to represent a product for search relevance — its **photograph**, its **title**, or its **structured attributes**.

> **TL;DR**
> **Structured attributes win decisively.** Encoding the four Big-4 attributes (brand, product type, activity, gender-by-age) instead of the title improves NDCG@10 by **+0.1329 macro / +0.1764 impression-weighted**, p < 0.001 — roughly five times the next largest effect — and closes the gap to the live ranking to statistical indistinguishability on high-traffic queries.
> Images beat titles on a *typical impression* (+0.0627, p = 0.006) but not on a *typical query* (+0.0226, p = 0.102).
> **Model size does not help.** A 203M encoder beats a 677M one on the same attribute text; on titles a 2.8× capacity increase changes nothing at all.

**Source files:** `01b_fetch_attributes.py`, `03_embed.py`, `04_evaluate.py`

---

## 1. Design — removing the usual confound

The naive comparison — a CLIP-family image tower against a separate text retriever — changes the **encoder** and the **modality** at once, and cannot attribute a difference to either. Almost every comparison of this kind in the wild is confounded this way.

We exploit a structural property of dual-tower vision-language models: both towers project into a **shared** embedding space. So the same model can represent a product by its photograph *or* by its title, with the query encoder held completely fixed.

*Implemented in `03_embed.py` — `encode_siglip()` drives both towers into the shared space; `encode_jina()` uses the documented `retrieval.query` / `retrieval.document` prompts.*

| System | Query encoder | Product representation | Role |
| --- | --- | --- | --- |
| `image` | SigLIP text tower | SigLIP **image** tower over the photograph | Treatment |
| `text-siglip` | SigLIP text tower | SigLIP **text** tower over the title | **Control** |
| `text-jina` | Jina `retrieval.query` | Jina `retrieval.document` over the title | External comparator |

Two contrasts fall out of this design:

- **`image` vs `text-siglip`** isolates the *modality*. Same model, same query encoder, same embedding space — only the product representation changes. This is the contrast that actually tests the hypothesis.
- **`text-siglip` vs `text-jina`** isolates the *encoder*. Same modality, different model.

### Models

| Model | Params | Output dim |
| --- | --- | --- |
| `google/siglip-base-patch16-512` | 203M | 768 |
| `jinaai/jina-embeddings-v5-text-nano` | 239M | 768 |
| `jinaai/jina-embeddings-v5-text-small` | 677M | 1024 |

The first two have comparable capacity and identical output dimensionality, so neither has a structural advantage. The third is added as a **capacity control** — though not a clean one, since it changes parameter count and output dimensionality together. The v5 text family publishes only these two size tiers; there is no smaller variant.

> **Licensing note:** both `jinaai/jina-embeddings-v5-text-nano` and `jinaai/jina-embeddings-v5-text-small` are CC BY-NC 4.0. Commercial use requires a licence. They are used here only as research comparators.

---

## 1b. The third representation — structured attributes

A product title is marketing copy: brand, model name, trademarks, size hints, and the product type buried at the end. The catalogue also carries a curated structured description — the **Big-4** attributes the LTR pipeline already uses as features.

Taken from `prod_ml_feature_store_db.products.ecode_attribute` using the same attribute ids as the LTR repo:

| Field | `attr_id` | Example |
| --- | --- | --- |
| brand | `X_BRAND` | `Nike` |
| product_type | `5382` | `Cleats` |
| product_activity | `4285` | `Soccer` |
| gender_by_age | `2101` | `Adult` |

Concatenated in that fixed order, space-separated: `"Nike Cleats Soccer Adult"`. Nothing else — no title, no description. Coverage is **100%** of the test set.

**This representation is far less informative than the title**, which makes the result below harder to explain away:

| Property | Title | Big-4 attributes |
| --- | --- | --- |
| Mean length (words) | 7.0 | 5.5 |
| **Distinct strings** | 18,541 (**94.7%**) | 5,416 (**27.7%**) |

257 products share `Nike Cleats Soccer Adult`. The attribute representation cannot separate products *within* a category at all.

---

## 2. Results

*Produced by `04_evaluate.py` into `results/summary.csv` (both `weighting` rows) and `results/significance.csv`.*

### Macro-averaged — every query counts once

| System | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | MRR@10 | MAP |
| --- | --- | --- | --- | --- | --- | --- |
| `production` (not comparable — see note) | 0.6268 | 0.6176 | 0.6248 | 0.1326 | 0.7784 | 0.7915 |
| **`attr-siglip`** | **0.5763** | **0.5466** | **0.5639** | 0.1557 | 0.9449 | 0.8746 |
| `attr-jina-small` | 0.4937 | 0.4721 | 0.4955 | 0.1524 | 0.9302 | 0.8696 |
| `attr-jina` | 0.4606 | 0.4516 | 0.4932 | 0.1552 | 0.9286 | 0.8735 |
| `image` | 0.4196 | 0.4363 | 0.4942 | 0.1612 | 0.9636 | 0.8960 |
| `text-jina-small` | 0.4014 | 0.4229 | 0.4822 | 0.1645 | 0.9728 | 0.8939 |
| `text-jina` | 0.3989 | 0.4171 | 0.4726 | 0.1633 | 0.9576 | 0.8910 |
| `text-siglip` | 0.4151 | 0.4137 | 0.4535 | 0.1548 | 0.9427 | 0.8503 |
| `random` | 0.2642 | 0.2775 | 0.3255 | 0.1331 | 0.8843 | 0.7740 |

### Impression-weighted — every query counts in proportion to its traffic

| System | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | MRR@10 | MAP |
| --- | --- | --- | --- | --- | --- | --- |
| `production` (not comparable — see note) | 0.6049 | 0.5974 | 0.6063 | 0.1152 | 0.7942 | 0.7985 |
| **`attr-siglip`** | **0.5893** | **0.5630** | **0.5765** | 0.1424 | 0.9676 | 0.8980 |
| `attr-jina-small` | 0.4827 | 0.4677 | 0.4890 | 0.1384 | 0.9246 | 0.8924 |
| `image` | 0.4376 | 0.4492 | 0.4970 | 0.1447 | 0.9751 | 0.9132 |
| `attr-jina` | 0.4349 | 0.4260 | 0.4815 | 0.1393 | 0.9109 | 0.8954 |
| `text-jina-small` | 0.3690 | 0.3970 | 0.4574 | 0.1455 | 0.9704 | 0.9109 |
| `text-jina` | 0.3722 | 0.3946 | 0.4512 | 0.1446 | 0.9616 | 0.9053 |
| `text-siglip` | 0.3971 | 0.3866 | 0.4210 | 0.1359 | 0.9631 | 0.8610 |
| `random` | 0.2630 | 0.2795 | 0.3164 | 0.1206 | 0.9335 | 0.7984 |

Ordering among the title and image arms is identical under both weightings; only the spacing changes. Both text arms lose ground on high-traffic queries while `image` gains, which is what widens the modality gap from +0.0226 to +0.0627.

> **Which metric separates:** `attr-siglip` leads on **NDCG at every k and on MAP**, but not on Recall@k, Precision@k or MRR@10. Those four binarise the label and sit near their ceilings — they measure how many *ever-clicked* products were retrieved, not how well the graded order was reproduced. Only NDCG and MAP use the grades. Contrasts below are computed on NDCG@10.

### Paired-bootstrap contrasts

2,000 resamples over 300 queries, seed 20260808.

| Contrast | Macro delta | p | Weighted delta | p |
| --- | --- | --- | --- | --- |
| **`attr-siglip` vs `text-siglip`** — representation | **+0.1329** | **<0.001** | **+0.1764** | **<0.001** |
| `attr-siglip` vs `image` | +0.1103 | <0.001 | +0.1137 | <0.001 |
| `attr-siglip` vs `attr-jina` — encoder | +0.0951 | <0.001 | +0.1370 | <0.001 |
| **`attr-siglip` vs `attr-jina-small`** — 203M vs 677M | **+0.0745** | **<0.001** | **+0.0952** | **<0.001** |
| **`attr-siglip` vs `production`** | -0.0710 | <0.001 | **-0.0344** | **0.262** |
| `attr-jina-small` vs `text-jina-small` — representation | +0.0493 | 0.007 | +0.0707 | 0.010 |
| `attr-jina` vs `text-jina` — representation | +0.0345 | 0.032 | +0.0314 | 0.184 |
| `image` vs `text-siglip` — **modality** | +0.0226 | 0.102 | **+0.0627** | **0.006** |
| **`attr-jina-small` vs `attr-jina`** — capacity, attributes | +0.0206 | 0.083 | **+0.0418** | **0.015** |
| **`text-jina-small` vs `text-jina`** — capacity, titles | **+0.0058** | **0.517** | **+0.0025** | **0.911** |
| `text-siglip` vs `text-jina` — encoder | -0.0034 | 0.843 | -0.0080 | 0.719 |
| `random` vs `text-jina` | -0.1396 | <0.001 | -0.1150 | <0.001 |

> **Note on `production`:** it is a proxy for the incumbent ordering, reconstructed from mean observed impression position — the LTR ranker was never queried. Its graded score is inflated by position leakage in the labels, so it must not be compared against directly. See the evaluation-methodology and evaluation-validity pages.

---

## 3. Reading the result

### Depth sensitivity — do the conclusions survive deeper cutoffs?

Every contrast was recomputed at six cutoffs. The **system ordering is identical at all six**; only the spacing changes.

| k | `attr-siglip` | `text-siglip` | Gap (macro) | Gap (weighted) | `production` | `random` |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.5763 | 0.4151 | **+0.1612** | **+0.1922** | 0.6268 | 0.2642 |
| 10 | 0.5466 | 0.4137 | **+0.1329** | **+0.1764** | 0.6176 | 0.2775 |
| 20 | 0.5639 | 0.4535 | +0.1104 | +0.1555 | 0.6248 | 0.3255 |
| **48** | 0.6526 | 0.5649 | **+0.0876** | **+0.1218** | 0.6841 | 0.4552 |
| 96 | 0.7398 | 0.6751 | +0.0647 | +0.0851 | 0.7651 | 0.5826 |
| 144 | 0.7806 | 0.7246 | +0.0560 | +0.0706 | 0.8045 | 0.6443 |

The attribute advantage shrinks monotonically with depth but never disappears. That compression is an artefact of the cutoff — as k approaches the pool size every system converges on the same candidate set, which is why `random` climbs from 0.2642 to 0.6443.

The modality contrast behaves differently — it **peaks in the middle**:

| k | 5 | 10 | 20 | 48 | 96 | 144 |
| --- | --- | --- | --- | --- | --- | --- |
| `image` vs `text-siglip` (macro) | +0.0044 | +0.0226 | +0.0407 | **+0.0480** | +0.0361 | +0.0233 |
| same, impression-weighted | +0.0406 | +0.0627 | +0.0760 | **+0.0795** | +0.0549 | +0.0386 |

Images are *not* better at the very top of the ranking (+0.0044 at k = 5) but are better through the middle of the list, peaking at k = 48. Reporting only NDCG@5 would have made the modality effect look like nothing at all.

### Structured attributes beat both the title and the photograph

`attr-siglip` improves on the same-encoder title arm by **+0.1329 macro / +0.1764 weighted**, both p < 0.001, with a 64% per-query win rate. That is roughly **five times** the modality effect and the largest contrast in the series. It also beats the best image arm by +0.11 under both weightings.

Under impression weighting it sits **-0.0344 from `production` at p = 0.262** — not distinguishable. No other embedding system comes near, and `production` retains the position-leakage advantage described in the evaluation-validity report, so its remaining lead is an overstatement.

### The win is not explained by more information

The attribute text is *shorter* (5.5 words vs 7.0) and vastly less unique (27.7% distinct vs 94.7%). It cannot discriminate within a category. That it wins anyway indicates the relevance labels reward **category-level** matching far more than instance-level ordering — a property of the label definition, worth knowing independently of embeddings.

### Model size is not the answer

| Capacity jump | Effect |
| --- | --- |
| 239M → 677M on **titles** | +0.0058, p = 0.517 — nothing |
| 239M → 677M on **attributes** | +0.0418 weighted, p = 0.015 — helps |
| 203M SigLIP vs 677M Jina on **attributes** | **+0.0745, p < 0.001 — the smaller model wins** |

A model 3.3× larger, from a family purpose-built for retrieval, loses decisively on attribute text. This rules out "the bigger model won" and supports the explanation that short keyword strings suit SigLIP's alt-text training distribution. Whether capacity is worth paying for depends on the representation, not the encoder alone.

### The modality contrast depends on the weighting

Under macro averaging it is **inconclusive**: +5.5% relative, interval crossing zero, 53% win rate. Under impression weighting it is **significant**: +0.0627, p = 0.006. Images do not clearly beat titles on a *typical query*, but do on a *typical impression*.

> Neither number is the "true" one. The weighted estimate targets a more business-relevant quantity but rests on about 109 effective queries rather than 300 — better-aimed and noisier at once.

### Encoder choice does not matter for titles

Two independently built encoders land within 0.8% of each other (p = 0.843 macro, p = 0.719 weighted), and a 2.8× capacity increase within one family adds nothing (p = 0.517). Any title-based difference observed later cannot be attributed to encoder quality. **This does not extend to attributes**, where encoder choice is worth +0.0951.

### The harness has signal

Every embedding system beats `random` by 30–50% at p < 0.001, so the setup can detect real effects. A null result elsewhere is evidence about the effect, not the instrument.

### Resolution estimate — the most useful operating number

On the macro metric the confidence-interval half-width for the modality contrast is roughly **±0.026 NDCG, about ±6% relative**. Any proposed improvement expected to deliver less than ~6% relative will not be distinguishable from noise at 300 queries.

---

## 4. What this leaves open

The `image` system is fed a raw catalogue photograph — whole frame, whatever the merchandising crop happened to be. Before concluding anything about modality, we should check whether the image *signal* is even intact by the time it reaches the encoder.

That is the subject of the next report: what is actually inside those frames. The short version is that a quarter of them are dominated by a human model rather than the product.

---

# Summary

| Question | Answer | Evidence |
| --- | --- | --- |
| What should the embedding read? | **Structured attributes**, not the title or the photo | +0.1329 macro / +0.1764 weighted, p < 0.001 |
| Can it match the live ranking? | **Not distinguishable** on traffic-weighted NDCG@10 | -0.0344, p = 0.262 |
| Do images beat titles on a typical query? | **No** — not measurably | +0.0226, p = 0.102 |
| Do images beat titles on a typical impression? | **Yes** | +0.0627, p = 0.006 |
| Does a bigger encoder help? | **No on titles**, yes on attributes — but the smallest model still wins | +0.0058 p = 0.517; 203M beats 677M by +0.0745 |
| What is the smallest detectable effect? | About **6% relative** | CI half-width ±0.026 NDCG |
