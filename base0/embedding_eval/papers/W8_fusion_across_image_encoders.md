# W8 — Does the Fusion Recipe Survive a Change of Image Encoder?

Generated 2026-08-21 by `13_w8_fusion_across_encoders.py` / `14_w8_report.py`.

> **TL;DR**
> RRF is the best fusion method for **2 of 3** image encoders;
> `attr-siglip` is the best text partner for **1 of 3**.
> Best fusion overall: **`z-score average: siglip-image + text-siglip`** (NDCG@10 0.4606 macro).
> Best system overall including standalones: **`z-score average: siglip-image + text-siglip`**
> (0.4606).

---

## 1. Why this report exists

W6 established a fusion recipe — **combine by RRF, partner with `attr-siglip`** — but measured it
with a **single image encoder**. That leaves an unanswered question: is RRF the right *method*, or
just the right method *for SigLIP*? A recipe that only works with one image tower is a much weaker
result than a general one.

W8 re-runs the W6 sweep with each of the three W7 image encoders as the fusion target:
**3 image encoders × 6 text representations × 3 fusion methods
= 36 combinations**, all on the same W7 subset
(600 queries, 28201 products, identical pools for every system).

Labels are the **corrected** LTR judgement list (τ applied to clicks only — see W4).

---

## 2. Does the method conclusion hold? (text partner fixed at `attr-siglip`)

Macro NDCG@10. If RRF wins every row, the method finding generalises beyond SigLIP.

| Image encoder | mean cosine | RRF (k=60) | z-score average | best |
| --- | --- | --- | --- | --- |
| `siglip-image` | 0.4491 | 0.4420 | **0.4516** | z-score average |
| `omni-nano-image` | 0.3839 | **0.3921** | 0.3820 | RRF (k=60) |
| `omni-small-image` | **nan** | nan | nan | mean cosine |

## 3. Does the text-partner conclusion hold? (method fixed at RRF)

Macro NDCG@10. If `attr-siglip` wins every row, the representation finding generalises too.

| Image encoder | text-siglip | text-jina | text-jina-small | attr-siglip | attr-jina | attr-jina-small | best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `siglip-image` | **0.4565** | 0.4514 | 0.4469 | 0.4420 | 0.4217 | 0.4216 | text-siglip |
| `omni-nano-image` | 0.3856 | **0.3924** | 0.3915 | 0.3921 | 0.3658 | 0.3668 | text-jina |
| `omni-small-image` | **nan** | nan | nan | nan | nan | nan | text-siglip |

---

## 4. The W6 recipe applied to each image encoder

Full metric depth for the recommended recipe against each image encoder alone, the best text
system, and the reference points.

### Macro-averaged — every query counts once

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.4149 | 0.4420 | 0.4890 | 0.5826 | 0.6553 | 0.6851 |
| `siglip-image` | 0.4034 | 0.4272 | 0.4819 | 0.5779 | 0.6554 | 0.6847 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.3586 | 0.3921 | 0.4430 | 0.5422 | 0.6204 | 0.6549 |
| `attr-siglip` | 0.3477 | 0.3776 | 0.4335 | 0.5341 | 0.6122 | 0.6472 |
| `omni-nano-image` | 0.3011 | 0.3313 | 0.3879 | 0.4968 | 0.5823 | 0.6205 |
| `random` | 0.2392 | 0.2640 | 0.3210 | 0.4357 | 0.5321 | 0.5758 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.1512 | 0.2730 | 0.4487 | 0.7274 | 0.9020 | 0.9689 |
| `siglip-image` | 0.1582 | 0.2655 | 0.4451 | 0.7246 | 0.9039 | 0.9698 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.1396 | 0.2557 | 0.4283 | 0.7066 | 0.8910 | 0.9641 |
| `attr-siglip` | 0.1381 | 0.2483 | 0.4198 | 0.7073 | 0.8872 | 0.9627 |
| `omni-nano-image` | 0.1226 | 0.2278 | 0.4071 | 0.6893 | 0.8800 | 0.9592 |
| `random` | 0.0954 | 0.1807 | 0.3567 | 0.6540 | 0.8596 | 0.9509 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.7690 | 0.7488 | 0.7049 | 0.5972 | 0.4421 | 0.3418 |
| `siglip-image` | 0.7607 | 0.7383 | 0.7038 | 0.5945 | 0.4438 | 0.3426 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.7247 | 0.7050 | 0.6679 | 0.5714 | 0.4324 | 0.3385 |
| `attr-siglip` | 0.7020 | 0.7010 | 0.6718 | 0.5741 | 0.4304 | 0.3380 |
| `omni-nano-image` | 0.6433 | 0.6383 | 0.6249 | 0.5517 | 0.4233 | 0.3356 |
| `random` | 0.5777 | 0.5777 | 0.5647 | 0.5094 | 0.4078 | 0.3306 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.8466 | 0.8506 | 0.8524 | 0.8529 | 0.8529 | 0.8529 |
| `siglip-image` | 0.8427 | 0.8467 | 0.8493 | 0.8500 | 0.8500 | 0.8500 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.8081 | 0.8135 | 0.8153 | 0.8159 | 0.8159 | 0.8159 |
| `attr-siglip` | 0.7843 | 0.7896 | 0.7913 | 0.7922 | 0.7922 | 0.7922 |
| `omni-nano-image` | 0.7388 | 0.7451 | 0.7481 | 0.7490 | 0.7490 | 0.7490 |
| `random` | 0.6917 | 0.6965 | 0.7008 | 0.7022 | 0.7022 | 0.7022 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.7432 |
| `siglip-image` | 0.7500 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.7057 |
| `attr-siglip` | 0.7023 |
| `omni-nano-image` | 0.6664 |
| `random` | 0.6041 |


### Impression-weighted — every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.4642 | 0.4725 | 0.5160 | 0.5991 | 0.7007 | 0.7525 |
| `siglip-image` | 0.4317 | 0.4443 | 0.4863 | 0.5844 | 0.6896 | 0.7387 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.4155 | 0.4322 | 0.4747 | 0.5724 | 0.6696 | 0.7261 |
| `attr-siglip` | 0.4044 | 0.4185 | 0.4617 | 0.5529 | 0.6596 | 0.7175 |
| `omni-nano-image` | 0.3511 | 0.3677 | 0.4040 | 0.5164 | 0.6310 | 0.6887 |
| `random` | 0.3183 | 0.3208 | 0.3505 | 0.4523 | 0.5797 | 0.6481 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.0797 | 0.1562 | 0.2995 | 0.5854 | 0.8306 | 0.9450 |
| `siglip-image` | 0.0784 | 0.1552 | 0.2978 | 0.5881 | 0.8361 | 0.9462 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.0754 | 0.1486 | 0.2844 | 0.5676 | 0.8211 | 0.9386 |
| `attr-siglip` | 0.0737 | 0.1470 | 0.2850 | 0.5719 | 0.8180 | 0.9370 |
| `omni-nano-image` | 0.0674 | 0.1362 | 0.2694 | 0.5551 | 0.8101 | 0.9333 |
| `random` | 0.0645 | 0.1287 | 0.2543 | 0.5199 | 0.7870 | 0.9239 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.9570 | 0.9530 | 0.9338 | 0.8383 | 0.6847 | 0.5600 |
| `siglip-image` | 0.9408 | 0.9377 | 0.9223 | 0.8429 | 0.6898 | 0.5613 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.9163 | 0.9133 | 0.8896 | 0.8113 | 0.6733 | 0.5543 |
| `attr-siglip` | 0.8930 | 0.9018 | 0.8883 | 0.8166 | 0.6718 | 0.5535 |
| `omni-nano-image` | 0.8205 | 0.8351 | 0.8369 | 0.7908 | 0.6613 | 0.5497 |
| `random` | 0.7927 | 0.7862 | 0.7858 | 0.7335 | 0.6397 | 0.5425 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.9760 | 0.9765 | 0.9765 | 0.9765 | 0.9765 | 0.9765 |
| `siglip-image` | 0.9796 | 0.9798 | 0.9799 | 0.9799 | 0.9799 | 0.9799 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.9465 | 0.9469 | 0.9471 | 0.9471 | 0.9471 | 0.9471 |
| `attr-siglip` | 0.9431 | 0.9442 | 0.9442 | 0.9442 | 0.9442 | 0.9442 |
| `omni-nano-image` | 0.8847 | 0.8860 | 0.8861 | 0.8862 | 0.8862 | 0.8862 |
| `random` | 0.8806 | 0.8825 | 0.8829 | 0.8829 | 0.8829 | 0.8829 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` | 0.9087 |
| `siglip-image` | 0.9088 |
| `RRF (k=60): omni-nano-image + attr-siglip` | 0.8738 |
| `attr-siglip` | 0.8753 |
| `omni-nano-image` | 0.8442 |
| `random` | 0.7970 |


---

## 5. Significance

Paired bootstrap over queries (2000 resamples), NDCG@10, LTR labels.

### Method contrasts — RRF vs the alternatives, per image encoder

| Contrast | Note | Macro Δ | Macro p | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` vs `mean cosine: siglip-image + attr-siglip` | method: rrf vs mean_cosine (image=siglip_image) | -0.0071 | 0.1530 n.s. | -0.0056 | 0.4150 |
| `RRF (k=60): siglip-image + attr-siglip` vs `z-score average: siglip-image + attr-siglip` | method: rrf vs zscore_avg (image=siglip_image) | -0.0096 | 0.0620 n.s. | +0.0026 | 0.7200 |
| `RRF (k=60): omni-nano-image + attr-siglip` vs `mean cosine: omni-nano-image + attr-siglip` | method: rrf vs mean_cosine (image=omni_nano_image) | +0.0082 | 0.0770 n.s. | +0.0221 | 0.0770 |
| `RRF (k=60): omni-nano-image + attr-siglip` vs `z-score average: omni-nano-image + attr-siglip` | method: rrf vs zscore_avg (image=omni_nano_image) | +0.0101 | 0.0270 **sig.** | +0.0229 | 0.0020 |

### Fusion vs its own inputs

| Contrast | Note | Macro Δ | Macro p | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- |
| `RRF (k=60): siglip-image + attr-siglip` vs `siglip-image` | fusion vs siglip_image alone (image=siglip_image) | +0.0149 | 0.0580 n.s. | +0.0282 | 0.1300 |
| `RRF (k=60): siglip-image + attr-siglip` vs `attr-siglip` | fusion vs siglip_attr alone (image=siglip_image) | +0.0645 | 0.0000 **sig.** | +0.0540 | 0.0680 |
| `RRF (k=60): omni-nano-image + attr-siglip` vs `omni-nano-image` | fusion vs omni_nano_image alone (image=omni_nano_image) | +0.0608 | 0.0000 **sig.** | +0.0645 | 0.0000 |
| `RRF (k=60): omni-nano-image + attr-siglip` vs `attr-siglip` | fusion vs siglip_attr alone (image=omni_nano_image) | +0.0145 | 0.0450 **sig.** | +0.0137 | 0.4460 |

---

## 6. Reading the result

**Method generalises, or it does not.** Section 2 is the whole point of this report: RRF winning
for one image encoder is an implementation detail, RRF winning for all three is a design rule.
Reciprocal rank fusion needs no score normalisation because it discards scores entirely, which is
exactly why it should be robust to swapping an encoder whose similarity scale is unknown.

**Watch the mean-cosine column.** Fusing raw, unnormalised cosines from two differently-scaled
spaces is the failure mode W6 identified. Its severity should *vary* by image encoder, since it
depends on the relative score magnitudes of the two towers — a pairing that happens to have
comparable scales will look fine, which is precisely why it is a trap.

**Fusion vs its inputs remains the decisive test.** A fusion that beats the image alone but not the
best text system alone is not an argument for fusion — it is an argument for the text system. W6
found exactly that with SigLIP; section 5 shows whether a better image encoder changes it.

> **Licensing:** the Jina v5 omni checkpoints are **CC BY-NC 4.0**; commercial use requires a
> licence. SigLIP is Apache-2.0.

Source data: `results/w8_summary.csv`, `results/w8_significance.csv`, `results/w8_per_query.csv`.
