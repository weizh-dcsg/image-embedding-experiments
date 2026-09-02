# W9 -- Does a Jina CLIP v2 image tower add anything on top of the live BM25 + E5 hybrid?

_Generated 2026-09-01 from `results/w9_*.csv`._

## Question

Every earlier week compared embedding systems against each other. W9 asks the deployment
question instead: the production stack already fuses Lucene BM25 with multilingual-e5-small by
reciprocal rank fusion, so what is the **marginal** value of adding a product-image tower as a
third arm? A model that wins in isolation can still add nothing once it is fused with signals
that already capture the same information.

- **Baseline** -- `RRF(bm25 + e5)`, the incumbent hybrid.
- **Contrast** -- `RRF(bm25 + e5 + jina-clip-v2-image)`, the same fusion with one extra arm.

## Verdict

Adding the image tower **improves on** the production hybrid on NDCG@10, and the difference is
**significant**: Δ = +0.0203 (95% CI [+0.0173, +0.0232],
p = 0.0000) across 6536 queries, winning on
49.7% of them and losing on 34.5%. Impression-weighted,
Δ = +0.0292 (p = 0.0010).

| Weighting | Metric | Baseline `RRF(bm25 + e5)` | `+ image` | Δ | Δ % |
| --- | --- | --- | --- | --- | --- |
| macro | ndcg@10 | 0.3516 | 0.3718 | +0.0203 | +5.76% |
| macro | ndcg@48 | 0.5035 | 0.5219 | +0.0184 | +3.66% |
| macro | recall@48 | 0.8048 | 0.8138 | +0.0089 | +1.11% |
| macro | mrr@10 | 0.6728 | 0.6889 | +0.0161 | +2.40% |
| macro | map | 0.5672 | 0.5810 | +0.0138 | +2.43% |
| impression | ndcg@10 | 0.3955 | 0.4247 | +0.0292 | +7.40% |
| impression | ndcg@48 | 0.5487 | 0.5747 | +0.0260 | +4.75% |
| impression | recall@48 | 0.6172 | 0.6303 | +0.0131 | +2.13% |
| impression | mrr@10 | 0.9283 | 0.9446 | +0.0163 | +1.76% |
| impression | map | 0.8278 | 0.8409 | +0.0131 | +1.59% |

Two qualifications belong next to that number, not in a footnote.

First, `production` -- the incumbent ordering by mean observed impression position -- still scores
0.4194, above every fusion tested here. It is partly self-fulfilling, since
the labels derive from click behaviour that is itself position-dependent, but it is the bar.

Second, the three-way fusion is **not** significantly better than the image tower on its own
(+0.0028, p = 0.367),
and the incumbent hybrid is not significantly better than BM25 on its own
(-0.0039, p = 0.101).
On this judgement list the E5 arm as currently indexed is close to inert, and most of what the
"three-way fusion" achieves is attributable to the image tower.

## The image encoder matters more than the fusion

W9 was scoped around Jina CLIP v2 because that is the model newly deployed to the cluster. But
SigLIP -- already encoded at full coverage since W1, and already deployed as
`siglip-base-patch16-512-text-v2` -- is the stronger image tower on this test set, and the gap is
larger than the entire benefit of adding a third arm.

| System | macro NDCG@10 |
| --- | --- |
| `siglip-image` alone | 0.4103 |
| `jina-clip-v2-image` alone | 0.3690 |
| `RRF(e5-full + siglip)` | 0.4105 |
| `RRF(e5-full + image)` (Jina) | 0.3882 |
| `RRF(bm25 + e5 + siglip)` | 0.3844 |
| `RRF(bm25 + e5 + image)` (Jina) | 0.3718 |

Swapping the encoder is worth +0.0413 standalone and
+0.0223 inside the best recipe -- both larger than the
+0.0203 the Jina arm buys over the incumbent hybrid. A
recommendation that reads "add Jina CLIP v2" without this table picks the weaker of two models
the cluster already hosts.

## Setup

| | |
| --- | --- |
| Judgement list | same 3-month LTR window as W1-W8 (90 days, banner DSG, channel WEB) |
| Queries | 6,536 |
| Candidates scored | 451,413 |
| Fusion | reciprocal rank fusion, k = 60 |
| Bootstrap | 2,000 paired resamples over queries |
| Seed | 20260808 |

**Signal provenance.** Both baseline arms are read from the live cluster rather than
reimplemented locally.

| Arm | Source | Detail |
| --- | --- | --- |
| `bm25` | `catalog-1` | Lucene BM25, `name^3 keyword^2 attributes longDescription`, pool applied as a filter so IDF still comes from the full 306M-doc index |
| `e5-small` | `catalog_embedding_final`, `catalog-name-embedding`, `e5-vector-index-v2` | `.multilingual-e5-small_linux-x86_64`, 45,575/80,608 products (56.54%) |
| `jina-clip-v2-image` | local encode | `jinaai/jina-clip-v2` vision tower over the product photograph, 1024-d |

**E5 prefix.** The production indices embed `passage: {name}`; queries therefore use
`query: {term}`. This was verified rather than assumed -- inference on the prefixed name
reproduces the stored vector at cosine 1.0, while the unprefixed name gives 0.96. A 0.96 cosine
is close enough to look correct on inspection, so getting this wrong would have silently
degraded the baseline and inflated the apparent value of the image tower.

**Candidate coverage.** Pools are deliberately *not* intersected on signal availability. A
retriever that cannot score a candidate omits it from its ranked list and contributes nothing
for it, which is exactly how RRF behaves in Elasticsearch and how the live stack behaves when a
product is missing from the vector index. Intersecting would have erased the baseline's real
coverage gap and flattered it.

| Arm | Share of candidates it can score |
| --- | --- |
| `bm25` | 95.26% |
| `e5-small` | 53.1% |
| `jina-clip-v2-image` | 100.0% |
| `e5-small (full coverage)` | 100.0% |
| `siglip-image` | 100.0% |

## Results -- macro average, LTR labels

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.3853 | 0.4194 | 0.4699 | 0.5514 | 0.6057 | 0.6261 |
| `RRF(e5-full + siglip)` | 0.3630 | 0.4105 | 0.4687 | 0.5525 | 0.6035 | 0.6206 |
| `siglip-image` | 0.3651 | 0.4103 | 0.4708 | 0.5539 | 0.6040 | 0.6206 |
| `RRF(bm25 + siglip)` | 0.3590 | 0.4055 | 0.4658 | 0.5509 | 0.6019 | 0.6186 |
| `RRF(bm25 + e5-full + image)` | 0.3404 | 0.3891 | 0.4502 | 0.5383 | 0.5911 | 0.6086 |
| `RRF(e5-full + image)` | 0.3397 | 0.3882 | 0.4486 | 0.5360 | 0.5886 | 0.6063 |
| `RRF(bm25 + image)` | 0.3364 | 0.3854 | 0.4468 | 0.5341 | 0.5869 | 0.6045 |
| `RRF(bm25 + e5 + siglip)` | 0.3440 | 0.3844 | 0.4406 | 0.5303 | 0.5850 | 0.6032 |
| **`RRF(bm25 + e5 + image)`** | **0.3297** | **0.3718** | **0.4303** | **0.5219** | **0.5773** | **0.5960** |
| `jina-clip-v2-image` | 0.3205 | 0.3690 | 0.4329 | 0.5211 | 0.5751 | 0.5934 |
| `RRF(e5 + image)` | 0.3234 | 0.3648 | 0.4226 | 0.5106 | 0.5686 | 0.5886 |
| `RRF(bm25 + e5-full)` | 0.3168 | 0.3640 | 0.4275 | 0.5200 | 0.5748 | 0.5934 |
| `e5-small (full coverage)` | 0.3120 | 0.3609 | 0.4238 | 0.5161 | 0.5716 | 0.5911 |
| `bm25` | 0.3082 | 0.3555 | 0.4190 | 0.5139 | 0.5696 | 0.5884 |
| **`RRF(bm25 + e5)`** | **0.3088** | **0.3516** | **0.4111** | **0.5035** | **0.5612** | **0.5819** |
| `e5-small` | 0.2967 | 0.3371 | 0.3969 | 0.4879 | 0.5464 | 0.5692 |
| `random` | 0.2082 | 0.2534 | 0.3232 | 0.4266 | 0.4927 | 0.5182 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.2333 | 0.3783 | 0.5745 | 0.8234 | 0.9437 | 0.9846 |
| `RRF(e5-full + siglip)` | 0.2281 | 0.3850 | 0.5825 | 0.8269 | 0.9461 | 0.9844 |
| `siglip-image` | 0.2250 | 0.3801 | 0.5806 | 0.8264 | 0.9460 | 0.9842 |
| `RRF(bm25 + siglip)` | 0.2225 | 0.3800 | 0.5831 | 0.8307 | 0.9480 | 0.9850 |
| `RRF(bm25 + e5-full + image)` | 0.2158 | 0.3768 | 0.5757 | 0.8255 | 0.9456 | 0.9841 |
| `RRF(e5-full + image)` | 0.2147 | 0.3732 | 0.5729 | 0.8217 | 0.9437 | 0.9834 |
| `RRF(bm25 + image)` | 0.2128 | 0.3711 | 0.5722 | 0.8250 | 0.9458 | 0.9841 |
| `RRF(bm25 + e5 + siglip)` | 0.2069 | 0.3568 | 0.5580 | 0.8174 | 0.9431 | 0.9833 |
| **`RRF(bm25 + e5 + image)`** | **0.2028** | **0.3505** | **0.5510** | **0.8138** | **0.9415** | **0.9823** |
| `jina-clip-v2-image` | 0.2022 | 0.3550 | 0.5597 | 0.8130 | 0.9409 | 0.9821 |
| `RRF(e5 + image)` | 0.2014 | 0.3442 | 0.5441 | 0.8014 | 0.9353 | 0.9798 |
| `RRF(bm25 + e5-full)` | 0.2098 | 0.3619 | 0.5625 | 0.8207 | 0.9427 | 0.9826 |
| `e5-small (full coverage)` | 0.2077 | 0.3593 | 0.5570 | 0.8135 | 0.9390 | 0.9817 |
| `bm25` | 0.2004 | 0.3543 | 0.5536 | 0.8185 | 0.9419 | 0.9822 |
| **`RRF(bm25 + e5)`** | **0.1952** | **0.3423** | **0.5417** | **0.8048** | **0.9366** | **0.9806** |
| `e5-small` | 0.1878 | 0.3272 | 0.5243 | 0.7887 | 0.9240 | 0.9739 |
| `random` | 0.1372 | 0.2735 | 0.4843 | 0.7705 | 0.9184 | 0.9735 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.5294 | 0.5219 | 0.4903 | 0.3880 | 0.2715 | 0.2037 |
| `RRF(e5-full + siglip)` | 0.5931 | 0.5684 | 0.5177 | 0.4004 | 0.2751 | 0.2042 |
| `siglip-image` | 0.5816 | 0.5584 | 0.5129 | 0.3991 | 0.2751 | 0.2041 |
| `RRF(bm25 + siglip)` | 0.5967 | 0.5730 | 0.5231 | 0.4044 | 0.2764 | 0.2045 |
| `RRF(bm25 + e5-full + image)` | 0.5879 | 0.5634 | 0.5163 | 0.4003 | 0.2749 | 0.2040 |
| `RRF(e5-full + image)` | 0.5730 | 0.5525 | 0.5063 | 0.3950 | 0.2734 | 0.2037 |
| `RRF(bm25 + image)` | 0.5826 | 0.5593 | 0.5123 | 0.3994 | 0.2749 | 0.2040 |
| `RRF(bm25 + e5 + siglip)` | 0.5764 | 0.5476 | 0.4980 | 0.3923 | 0.2730 | 0.2035 |
| **`RRF(bm25 + e5 + image)`** | **0.5661** | **0.5401** | **0.4920** | **0.3894** | **0.2718** | **0.2029** |
| `jina-clip-v2-image` | 0.5438 | 0.5278 | 0.4903 | 0.3883 | 0.2716 | 0.2030 |
| `RRF(e5 + image)` | 0.5457 | 0.5236 | 0.4787 | 0.3780 | 0.2676 | 0.2017 |
| `RRF(bm25 + e5-full)` | 0.5752 | 0.5520 | 0.5049 | 0.3954 | 0.2731 | 0.2032 |
| `e5-small (full coverage)` | 0.5576 | 0.5384 | 0.4940 | 0.3888 | 0.2706 | 0.2028 |
| `bm25` | 0.5675 | 0.5464 | 0.5019 | 0.3954 | 0.2728 | 0.2031 |
| **`RRF(bm25 + e5)`** | **0.5525** | **0.5275** | **0.4814** | **0.3816** | **0.2687** | **0.2021** |
| `e5-small` | 0.5293 | 0.5055 | 0.4628 | 0.3670 | 0.2609 | 0.1988 |
| `random` | 0.4250 | 0.4252 | 0.4104 | 0.3468 | 0.2570 | 0.1985 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.6352 | 0.6456 | 0.6492 | 0.6506 | 0.6506 | 0.6506 |
| `RRF(e5-full + siglip)` | 0.7055 | 0.7153 | 0.7192 | 0.7207 | 0.7207 | 0.7207 |
| `siglip-image` | 0.6922 | 0.7019 | 0.7062 | 0.7076 | 0.7076 | 0.7076 |
| `RRF(bm25 + siglip)` | 0.7035 | 0.7135 | 0.7178 | 0.7192 | 0.7193 | 0.7193 |
| `RRF(bm25 + e5-full + image)` | 0.6922 | 0.7035 | 0.7078 | 0.7093 | 0.7094 | 0.7094 |
| `RRF(e5-full + image)` | 0.6786 | 0.6895 | 0.6940 | 0.6954 | 0.6955 | 0.6955 |
| `RRF(bm25 + image)` | 0.6827 | 0.6933 | 0.6977 | 0.6993 | 0.6993 | 0.6993 |
| `RRF(bm25 + e5 + siglip)` | 0.6897 | 0.7001 | 0.7050 | 0.7066 | 0.7066 | 0.7066 |
| **`RRF(bm25 + e5 + image)`** | **0.6782** | **0.6889** | **0.6939** | **0.6956** | **0.6957** | **0.6957** |
| `jina-clip-v2-image` | 0.6424 | 0.6538 | 0.6589 | 0.6605 | 0.6605 | 0.6605 |
| `RRF(e5 + image)` | 0.6557 | 0.6663 | 0.6714 | 0.6732 | 0.6733 | 0.6733 |
| `RRF(bm25 + e5-full)` | 0.6775 | 0.6880 | 0.6925 | 0.6942 | 0.6943 | 0.6943 |
| `e5-small (full coverage)` | 0.6653 | 0.6760 | 0.6806 | 0.6824 | 0.6824 | 0.6824 |
| `bm25` | 0.6748 | 0.6853 | 0.6899 | 0.6919 | 0.6920 | 0.6920 |
| **`RRF(bm25 + e5)`** | **0.6618** | **0.6728** | **0.6779** | **0.6797** | **0.6798** | **0.6798** |
| `e5-small` | 0.6457 | 0.6566 | 0.6621 | 0.6641 | 0.6641 | 0.6641 |
| `random` | 0.5432 | 0.5571 | 0.5639 | 0.5662 | 0.5663 | 0.5663 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| `production` | 0.5979 |
| `RRF(e5-full + siglip)` | 0.6180 |
| `siglip-image` | 0.6119 |
| `RRF(bm25 + siglip)` | 0.6209 |
| `RRF(bm25 + e5-full + image)` | 0.6103 |
| `RRF(e5-full + image)` | 0.5996 |
| `RRF(bm25 + image)` | 0.6022 |
| `RRF(bm25 + e5 + siglip)` | 0.5893 |
| **`RRF(bm25 + e5 + image)`** | **0.5810** |
| `jina-clip-v2-image` | 0.5777 |
| `RRF(e5 + image)` | 0.5637 |
| `RRF(bm25 + e5-full)` | 0.5967 |
| `e5-small (full coverage)` | 0.5891 |
| `bm25` | 0.5931 |
| **`RRF(bm25 + e5)`** | **0.5672** |
| `e5-small` | 0.5429 |
| `random` | 0.4757 |


## Results -- impression weighted, LTR labels

Macro treats every query equally; impression weighting answers "how well does this rank a
typical search impression". They can disagree, and where they do the weighted number is the
business-relevant one.

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `siglip-image` | 0.4312 | 0.4516 | 0.5019 | 0.6065 | 0.7001 | 0.7374 |
| `RRF(bm25 + siglip)` | 0.4182 | 0.4442 | 0.4949 | 0.6012 | 0.6976 | 0.7354 |
| `RRF(e5-full + siglip)` | 0.4120 | 0.4420 | 0.4896 | 0.5958 | 0.6922 | 0.7320 |
| `RRF(bm25 + e5 + siglip)` | 0.4206 | 0.4348 | 0.4784 | 0.5844 | 0.6847 | 0.7239 |
| `RRF(bm25 + image)` | 0.4056 | 0.4285 | 0.4796 | 0.5894 | 0.6864 | 0.7254 |
| **`RRF(bm25 + e5 + image)`** | **0.4025** | **0.4247** | **0.4657** | **0.5747** | **0.6772** | **0.7163** |
| `RRF(bm25 + e5-full + image)` | 0.4053 | 0.4243 | 0.4705 | 0.5868 | 0.6847 | 0.7241 |
| `production` | 0.4190 | 0.4239 | 0.4521 | 0.5459 | 0.6611 | 0.7126 |
| `RRF(e5-full + image)` | 0.4028 | 0.4228 | 0.4706 | 0.5831 | 0.6826 | 0.7216 |
| `RRF(e5 + image)` | 0.3987 | 0.4200 | 0.4601 | 0.5620 | 0.6661 | 0.7094 |
| `jina-clip-v2-image` | 0.3900 | 0.4116 | 0.4671 | 0.5785 | 0.6740 | 0.7147 |
| **`RRF(bm25 + e5)`** | **0.3738** | **0.3955** | **0.4397** | **0.5487** | **0.6541** | **0.6986** |
| `bm25` | 0.3663 | 0.3894 | 0.4395 | 0.5565 | 0.6623 | 0.7027 |
| `RRF(bm25 + e5-full)` | 0.3641 | 0.3866 | 0.4422 | 0.5605 | 0.6640 | 0.7058 |
| `e5-small (full coverage)` | 0.3494 | 0.3773 | 0.4308 | 0.5506 | 0.6563 | 0.6983 |
| `e5-small` | 0.3572 | 0.3745 | 0.4189 | 0.5254 | 0.6307 | 0.6802 |
| `random` | 0.2624 | 0.2840 | 0.3375 | 0.4550 | 0.5768 | 0.6343 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `siglip-image` | 0.0982 | 0.1919 | 0.3537 | 0.6480 | 0.8743 | 0.9604 |
| `RRF(bm25 + siglip)` | 0.1009 | 0.1960 | 0.3592 | 0.6542 | 0.8761 | 0.9625 |
| `RRF(e5-full + siglip)` | 0.1004 | 0.1953 | 0.3578 | 0.6496 | 0.8723 | 0.9606 |
| `RRF(bm25 + e5 + siglip)` | 0.0946 | 0.1834 | 0.3391 | 0.6332 | 0.8668 | 0.9583 |
| `RRF(bm25 + image)` | 0.0996 | 0.1925 | 0.3555 | 0.6489 | 0.8718 | 0.9596 |
| **`RRF(bm25 + e5 + image)`** | **0.0937** | **0.1819** | **0.3370** | **0.6303** | **0.8638** | **0.9559** |
| `RRF(bm25 + e5-full + image)` | 0.1005 | 0.1948 | 0.3575 | 0.6504 | 0.8707 | 0.9600 |
| `production` | 0.0762 | 0.1569 | 0.3051 | 0.5903 | 0.8477 | 0.9533 |
| `RRF(e5-full + image)` | 0.0984 | 0.1920 | 0.3531 | 0.6432 | 0.8684 | 0.9585 |
| `RRF(e5 + image)` | 0.0905 | 0.1773 | 0.3292 | 0.6125 | 0.8519 | 0.9510 |
| `jina-clip-v2-image` | 0.0943 | 0.1842 | 0.3440 | 0.6343 | 0.8632 | 0.9556 |
| **`RRF(bm25 + e5)`** | **0.0920** | **0.1786** | **0.3307** | **0.6172** | **0.8564** | **0.9531** |
| `bm25` | 0.0982 | 0.1899 | 0.3497 | 0.6435 | 0.8682 | 0.9578 |
| `RRF(bm25 + e5-full)` | 0.0987 | 0.1919 | 0.3522 | 0.6447 | 0.8671 | 0.9578 |
| `e5-small (full coverage)` | 0.0971 | 0.1890 | 0.3472 | 0.6369 | 0.8608 | 0.9549 |
| `e5-small` | 0.0891 | 0.1719 | 0.3178 | 0.5940 | 0.8340 | 0.9421 |
| `random` | 0.0749 | 0.1516 | 0.2906 | 0.5721 | 0.8265 | 0.9409 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `siglip-image` | 0.9049 | 0.8981 | 0.8735 | 0.7667 | 0.5952 | 0.4664 |
| `RRF(bm25 + siglip)` | 0.9266 | 0.9197 | 0.8899 | 0.7739 | 0.5969 | 0.4678 |
| `RRF(e5-full + siglip)` | 0.9206 | 0.9149 | 0.8838 | 0.7670 | 0.5937 | 0.4665 |
| `RRF(bm25 + e5 + siglip)` | 0.8944 | 0.8831 | 0.8565 | 0.7509 | 0.5881 | 0.4645 |
| `RRF(bm25 + image)` | 0.9175 | 0.9046 | 0.8791 | 0.7660 | 0.5927 | 0.4656 |
| **`RRF(bm25 + e5 + image)`** | **0.8819** | **0.8731** | **0.8490** | **0.7465** | **0.5854** | **0.4628** |
| `RRF(bm25 + e5-full + image)` | 0.9233 | 0.9105 | 0.8808 | 0.7678 | 0.5922 | 0.4659 |
| `production` | 0.7164 | 0.7371 | 0.7390 | 0.6841 | 0.5694 | 0.4603 |
| `RRF(e5-full + image)` | 0.9009 | 0.8967 | 0.8715 | 0.7582 | 0.5902 | 0.4648 |
| `RRF(e5 + image)` | 0.8534 | 0.8531 | 0.8287 | 0.7257 | 0.5760 | 0.4594 |
| `jina-clip-v2-image` | 0.8677 | 0.8623 | 0.8485 | 0.7475 | 0.5850 | 0.4628 |
| **`RRF(bm25 + e5)`** | **0.8697** | **0.8596** | **0.8320** | **0.7304** | **0.5791** | **0.4612** |
| `bm25` | 0.8983 | 0.8849 | 0.8626 | 0.7607 | 0.5905 | 0.4647 |
| `RRF(bm25 + e5-full)` | 0.9038 | 0.8967 | 0.8685 | 0.7601 | 0.5891 | 0.4643 |
| `e5-small (full coverage)` | 0.8792 | 0.8758 | 0.8516 | 0.7500 | 0.5839 | 0.4622 |
| `e5-small` | 0.8393 | 0.8261 | 0.8008 | 0.7024 | 0.5608 | 0.4536 |
| `random` | 0.7144 | 0.7245 | 0.7187 | 0.6648 | 0.5524 | 0.4523 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `siglip-image` | 0.9541 | 0.9551 | 0.9553 | 0.9553 | 0.9553 | 0.9553 |
| `RRF(bm25 + siglip)` | 0.9647 | 0.9649 | 0.9650 | 0.9650 | 0.9650 | 0.9650 |
| `RRF(e5-full + siglip)` | 0.9651 | 0.9654 | 0.9654 | 0.9654 | 0.9654 | 0.9654 |
| `RRF(bm25 + e5 + siglip)` | 0.9481 | 0.9488 | 0.9491 | 0.9492 | 0.9492 | 0.9492 |
| `RRF(bm25 + image)` | 0.9594 | 0.9597 | 0.9598 | 0.9598 | 0.9598 | 0.9598 |
| **`RRF(bm25 + e5 + image)`** | **0.9437** | **0.9446** | **0.9447** | **0.9449** | **0.9449** | **0.9449** |
| `RRF(bm25 + e5-full + image)` | 0.9623 | 0.9628 | 0.9628 | 0.9628 | 0.9628 | 0.9628 |
| `production` | 0.7561 | 0.7618 | 0.7628 | 0.7628 | 0.7628 | 0.7628 |
| `RRF(e5-full + image)` | 0.9427 | 0.9433 | 0.9434 | 0.9434 | 0.9434 | 0.9434 |
| `RRF(e5 + image)` | 0.9011 | 0.9025 | 0.9032 | 0.9034 | 0.9034 | 0.9034 |
| `jina-clip-v2-image` | 0.9080 | 0.9090 | 0.9092 | 0.9093 | 0.9093 | 0.9093 |
| **`RRF(bm25 + e5)`** | **0.9255** | **0.9283** | **0.9285** | **0.9286** | **0.9286** | **0.9286** |
| `bm25` | 0.9495 | 0.9500 | 0.9505 | 0.9505 | 0.9505 | 0.9505 |
| `RRF(bm25 + e5-full)` | 0.9477 | 0.9484 | 0.9485 | 0.9485 | 0.9485 | 0.9485 |
| `e5-small (full coverage)` | 0.9362 | 0.9369 | 0.9369 | 0.9370 | 0.9370 | 0.9370 |
| `e5-small` | 0.9177 | 0.9189 | 0.9196 | 0.9198 | 0.9198 | 0.9198 |
| `random` | 0.8415 | 0.8442 | 0.8445 | 0.8445 | 0.8445 | 0.8445 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| `siglip-image` | 0.8709 |
| `RRF(bm25 + siglip)` | 0.8828 |
| `RRF(e5-full + siglip)` | 0.8771 |
| `RRF(bm25 + e5 + siglip)` | 0.8472 |
| `RRF(bm25 + image)` | 0.8725 |
| **`RRF(bm25 + e5 + image)`** | **0.8409** |
| `RRF(bm25 + e5-full + image)` | 0.8760 |
| `production` | 0.7692 |
| `RRF(e5-full + image)` | 0.8651 |
| `RRF(e5 + image)` | 0.8192 |
| `jina-clip-v2-image` | 0.8458 |
| **`RRF(bm25 + e5)`** | **0.8278** |
| `bm25` | 0.8649 |
| `RRF(bm25 + e5-full)` | 0.8661 |
| `e5-small (full coverage)` | 0.8536 |
| `e5-small` | 0.7943 |
| `random` | 0.7406 |


## Results -- raw-CTR labels, macro average

A robustness check against the LTR grade construction: same rankings, relevance taken from raw
click-through rate bins instead of the IPW-smoothed weighted-CTR quantiles.

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.4974 | 0.5337 | 0.5779 | 0.6370 | 0.6776 | 0.6945 |
| `RRF(bm25 + siglip)` | 0.3869 | 0.4351 | 0.4969 | 0.5767 | 0.6255 | 0.6428 |
| `RRF(e5-full + siglip)` | 0.3851 | 0.4346 | 0.4950 | 0.5755 | 0.6245 | 0.6423 |
| `siglip-image` | 0.3828 | 0.4300 | 0.4943 | 0.5746 | 0.6229 | 0.6401 |
| `RRF(bm25 + e5-full + image)` | 0.3651 | 0.4150 | 0.4772 | 0.5610 | 0.6120 | 0.6303 |
| `RRF(bm25 + e5 + siglip)` | 0.3681 | 0.4093 | 0.4681 | 0.5548 | 0.6076 | 0.6264 |
| `RRF(bm25 + image)` | 0.3584 | 0.4087 | 0.4724 | 0.5561 | 0.6071 | 0.6255 |
| `RRF(e5-full + image)` | 0.3546 | 0.4057 | 0.4692 | 0.5543 | 0.6057 | 0.6243 |
| **`RRF(bm25 + e5 + image)`** | **0.3532** | **0.3950** | **0.4553** | **0.5445** | **0.5983** | **0.6177** |
| `RRF(bm25 + e5-full)` | 0.3422 | 0.3895 | 0.4538 | 0.5426 | 0.5957 | 0.6152 |
| `bm25` | 0.3349 | 0.3823 | 0.4469 | 0.5368 | 0.5911 | 0.6108 |
| `RRF(e5 + image)` | 0.3389 | 0.3816 | 0.4423 | 0.5295 | 0.5866 | 0.6074 |
| `e5-small (full coverage)` | 0.3314 | 0.3806 | 0.4453 | 0.5357 | 0.5902 | 0.6107 |
| `jina-clip-v2-image` | 0.3263 | 0.3784 | 0.4478 | 0.5360 | 0.5892 | 0.6085 |
| **`RRF(bm25 + e5)`** | **0.3301** | **0.3740** | **0.4353** | **0.5253** | **0.5821** | **0.6033** |
| `e5-small` | 0.3123 | 0.3538 | 0.4154 | 0.5060 | 0.5643 | 0.5880 |
| `random` | 0.2049 | 0.2546 | 0.3306 | 0.4375 | 0.5043 | 0.5311 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.2385 | 0.3870 | 0.5822 | 0.8250 | 0.9438 | 0.9846 |
| `RRF(bm25 + siglip)` | 0.2248 | 0.3835 | 0.5863 | 0.8314 | 0.9483 | 0.9851 |
| `RRF(e5-full + siglip)` | 0.2305 | 0.3886 | 0.5857 | 0.8277 | 0.9464 | 0.9845 |
| `siglip-image` | 0.2272 | 0.3840 | 0.5846 | 0.8275 | 0.9464 | 0.9843 |
| `RRF(bm25 + e5-full + image)` | 0.2178 | 0.3797 | 0.5780 | 0.8260 | 0.9458 | 0.9841 |
| `RRF(bm25 + e5 + siglip)` | 0.2090 | 0.3597 | 0.5602 | 0.8178 | 0.9433 | 0.9834 |
| `RRF(bm25 + image)` | 0.2146 | 0.3741 | 0.5748 | 0.8255 | 0.9460 | 0.9842 |
| `RRF(e5-full + image)` | 0.2166 | 0.3762 | 0.5754 | 0.8224 | 0.9439 | 0.9835 |
| **`RRF(bm25 + e5 + image)`** | **0.2045** | **0.3530** | **0.5527** | **0.8141** | **0.9417** | **0.9823** |
| `RRF(bm25 + e5-full)` | 0.2114 | 0.3639 | 0.5643 | 0.8210 | 0.9428 | 0.9827 |
| `bm25` | 0.2017 | 0.3564 | 0.5551 | 0.8188 | 0.9420 | 0.9822 |
| `RRF(e5 + image)` | 0.2028 | 0.3464 | 0.5461 | 0.8019 | 0.9354 | 0.9798 |
| `e5-small (full coverage)` | 0.2090 | 0.3613 | 0.5587 | 0.8139 | 0.9392 | 0.9818 |
| `jina-clip-v2-image` | 0.2036 | 0.3577 | 0.5625 | 0.8138 | 0.9412 | 0.9821 |
| **`RRF(bm25 + e5)`** | **0.1967** | **0.3444** | **0.5429** | **0.8051** | **0.9367** | **0.9806** |
| `e5-small` | 0.1890 | 0.3287 | 0.5253 | 0.7889 | 0.9241 | 0.9739 |
| `random` | 0.1370 | 0.2737 | 0.4846 | 0.7705 | 0.9184 | 0.9734 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.5259 | 0.5158 | 0.4770 | 0.3714 | 0.2606 | 0.1961 |
| `RRF(bm25 + siglip)` | 0.5794 | 0.5541 | 0.5025 | 0.3870 | 0.2658 | 0.1970 |
| `RRF(e5-full + siglip)` | 0.5766 | 0.5499 | 0.4974 | 0.3831 | 0.2645 | 0.1967 |
| `siglip-image` | 0.5647 | 0.5407 | 0.4938 | 0.3823 | 0.2645 | 0.1966 |
| `RRF(bm25 + e5-full + image)` | 0.5686 | 0.5428 | 0.4941 | 0.3826 | 0.2642 | 0.1964 |
| `RRF(bm25 + e5 + siglip)` | 0.5585 | 0.5276 | 0.4762 | 0.3747 | 0.2623 | 0.1959 |
| `RRF(bm25 + image)` | 0.5625 | 0.5390 | 0.4907 | 0.3818 | 0.2642 | 0.1964 |
| `RRF(e5-full + image)` | 0.5543 | 0.5326 | 0.4848 | 0.3776 | 0.2627 | 0.1961 |
| **`RRF(bm25 + e5 + image)`** | **0.5464** | **0.5191** | **0.4695** | **0.3716** | **0.2611** | **0.1953** |
| `RRF(bm25 + e5-full)` | 0.5537 | 0.5289 | 0.4818 | 0.3775 | 0.2623 | 0.1956 |
| `bm25` | 0.5442 | 0.5231 | 0.4784 | 0.3775 | 0.2620 | 0.1955 |
| `RRF(e5 + image)` | 0.5257 | 0.5027 | 0.4567 | 0.3605 | 0.2569 | 0.1941 |
| `e5-small (full coverage)` | 0.5351 | 0.5155 | 0.4712 | 0.3711 | 0.2599 | 0.1952 |
| `jina-clip-v2-image` | 0.5235 | 0.5074 | 0.4694 | 0.3711 | 0.2609 | 0.1955 |
| **`RRF(bm25 + e5)`** | **0.5310** | **0.5052** | **0.4581** | **0.3637** | **0.2579** | **0.1945** |
| `e5-small` | 0.5073 | 0.4826 | 0.4394 | 0.3490 | 0.2501 | 0.1912 |
| `random` | 0.3985 | 0.3994 | 0.3860 | 0.3286 | 0.2461 | 0.1909 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production` | 0.6341 | 0.6445 | 0.6482 | 0.6496 | 0.6496 | 0.6496 |
| `RRF(bm25 + siglip)` | 0.6960 | 0.7060 | 0.7104 | 0.7118 | 0.7119 | 0.7119 |
| `RRF(e5-full + siglip)` | 0.6964 | 0.7062 | 0.7101 | 0.7116 | 0.7116 | 0.7116 |
| `siglip-image` | 0.6820 | 0.6918 | 0.6961 | 0.6975 | 0.6975 | 0.6975 |
| `RRF(bm25 + e5-full + image)` | 0.6828 | 0.6941 | 0.6983 | 0.6999 | 0.6999 | 0.6999 |
| `RRF(bm25 + e5 + siglip)` | 0.6799 | 0.6904 | 0.6953 | 0.6969 | 0.6969 | 0.6969 |
| `RRF(bm25 + image)` | 0.6728 | 0.6835 | 0.6878 | 0.6894 | 0.6895 | 0.6895 |
| `RRF(e5-full + image)` | 0.6690 | 0.6800 | 0.6845 | 0.6859 | 0.6860 | 0.6860 |
| **`RRF(bm25 + e5 + image)`** | **0.6674** | **0.6782** | **0.6832** | **0.6849** | **0.6850** | **0.6850** |
| `RRF(bm25 + e5-full)` | 0.6658 | 0.6764 | 0.6809 | 0.6827 | 0.6827 | 0.6827 |
| `bm25` | 0.6611 | 0.6717 | 0.6763 | 0.6783 | 0.6784 | 0.6784 |
| `RRF(e5 + image)` | 0.6451 | 0.6558 | 0.6609 | 0.6627 | 0.6628 | 0.6628 |
| `e5-small (full coverage)` | 0.6534 | 0.6642 | 0.6687 | 0.6705 | 0.6706 | 0.6706 |
| `jina-clip-v2-image` | 0.6302 | 0.6417 | 0.6467 | 0.6483 | 0.6484 | 0.6484 |
| **`RRF(bm25 + e5)`** | **0.6490** | **0.6600** | **0.6651** | **0.6670** | **0.6670** | **0.6670** |
| `e5-small` | 0.6315 | 0.6425 | 0.6479 | 0.6499 | 0.6499 | 0.6499 |
| `random` | 0.5293 | 0.5432 | 0.5500 | 0.5523 | 0.5524 | 0.5524 |

**MAP** (no cutoff)

| System | MAP |
| --- | --- |
| `production` | 0.5896 |
| `RRF(bm25 + siglip)` | 0.6037 |
| `RRF(e5-full + siglip)` | 0.6010 |
| `siglip-image` | 0.5955 |
| `RRF(bm25 + e5-full + image)` | 0.5917 |
| `RRF(bm25 + e5 + siglip)` | 0.5707 |
| `RRF(bm25 + image)` | 0.5837 |
| `RRF(e5-full + image)` | 0.5813 |
| **`RRF(bm25 + e5 + image)`** | **0.5617** |
| `RRF(bm25 + e5-full)` | 0.5768 |
| `bm25` | 0.5729 |
| `RRF(e5 + image)` | 0.5444 |
| `e5-small (full coverage)` | 0.5691 |
| `jina-clip-v2-image` | 0.5590 |
| **`RRF(bm25 + e5)`** | **0.5468** |
| `e5-small` | 0.5217 |
| `random` | 0.4524 |


## Query tier breakdown

Head/torso/tail are percentile cut points on total impressions among queries passing the LTR
filters (head above 95%, torso above 70%).

**Macro NDCG@10**

| System | head NDCG@10 | torso NDCG@10 | tail NDCG@10 |
| --- | --- | --- | --- |
| `bm25` | 0.3839 | 0.4011 | 0.3371 |
| `e5-small` | 0.3736 | 0.3869 | 0.3167 |
| `e5-small (full coverage)` | 0.3739 | 0.3999 | 0.3460 |
| `jina-clip-v2-image` | 0.4162 | 0.4154 | 0.3490 |
| **`RRF(bm25 + e5)`** | 0.3911 | 0.4034 | 0.3302 |
| **`RRF(bm25 + e5 + image)`** | 0.4195 | 0.4263 | 0.3489 |
| `RRF(bm25 + image)` | 0.4241 | 0.4358 | 0.3646 |
| `RRF(e5 + image)` | 0.4164 | 0.4173 | 0.3423 |
| `RRF(bm25 + e5-full)` | 0.3786 | 0.4111 | 0.3461 |
| `RRF(bm25 + e5-full + image)` | 0.4149 | 0.4361 | 0.3704 |
| `RRF(e5-full + image)` | 0.4256 | 0.4355 | 0.3685 |
| `siglip-image` | 0.4369 | 0.4586 | 0.3911 |
| `RRF(bm25 + e5 + siglip)` | 0.4270 | 0.4411 | 0.3610 |
| `RRF(bm25 + siglip)` | 0.4364 | 0.4563 | 0.3850 |
| `RRF(e5-full + siglip)` | 0.4362 | 0.4594 | 0.3911 |
| `production` | 0.4218 | 0.4597 | 0.4049 |
| `random` | 0.2932 | 0.2937 | 0.2361 |

**Impression-weighted NDCG@10**

| System | head NDCG@10 | torso NDCG@10 | tail NDCG@10 |
| --- | --- | --- | --- |
| `bm25` | 0.3823 | 0.4017 | 0.3875 |
| `e5-small` | 0.3681 | 0.3892 | 0.3630 |
| `e5-small (full coverage)` | 0.3594 | 0.4014 | 0.3898 |
| `jina-clip-v2-image` | 0.4092 | 0.4192 | 0.4019 |
| **`RRF(bm25 + e5)`** | 0.3939 | 0.4040 | 0.3798 |
| **`RRF(bm25 + e5 + image)`** | 0.4278 | 0.4275 | 0.4044 |
| `RRF(bm25 + image)` | 0.4233 | 0.4388 | 0.4233 |
| `RRF(e5 + image)` | 0.4261 | 0.4197 | 0.3954 |
| `RRF(bm25 + e5-full)` | 0.3687 | 0.4123 | 0.3957 |
| `RRF(bm25 + e5-full + image)` | 0.4152 | 0.4400 | 0.4221 |
| `RRF(e5-full + image)` | 0.4128 | 0.4405 | 0.4185 |
| `siglip-image` | 0.4478 | 0.4572 | 0.4533 |
| `RRF(bm25 + e5 + siglip)` | 0.4345 | 0.4408 | 0.4198 |
| `RRF(bm25 + siglip)` | 0.4379 | 0.4534 | 0.4466 |
| `RRF(e5-full + siglip)` | 0.4297 | 0.4598 | 0.4476 |
| `production` | 0.3938 | 0.4553 | 0.4697 |
| `random` | 0.2802 | 0.2930 | 0.2762 |

## Significance -- paired bootstrap on NDCG@10

All queries:

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | +0.0203 | [+0.0173, +0.0232] | 0.0000 **sig.** | 0.497 | +0.0292 | 0.0010 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0164 | [+0.0110, +0.0218] | 0.0000 **sig.** | 0.509 | +0.0353 | 0.1550 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | +0.0347 | [+0.0305, +0.0387] | 0.0000 **sig.** | 0.522 | +0.0502 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | +0.0028 | [-0.0031, +0.0086] | 0.3670 n.s. | 0.479 | +0.0131 | 0.8250 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | -0.0039 | [-0.0087, +0.0008] | 0.1010 n.s. | 0.457 | +0.0060 | 0.6870 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | +0.0144 | [+0.0108, +0.0180] | 0.0000 **sig.** | 0.458 | +0.0210 | 0.0520 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0679 | [-0.0745, -0.0609] | 0.0000 **sig.** | 0.365 | -0.0285 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0476 | [-0.0551, -0.0407] | 0.0000 **sig.** | 0.391 | +0.0008 | 0.0010 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0338 | [+0.0286, +0.0395] | 0.0000 **sig.** | 0.513 | +0.0330 | 0.0020 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | +0.0132 | [+0.0086, +0.0178] | 0.0000 **sig.** | 0.471 | +0.0246 | 0.1600 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0251 | [+0.0218, +0.0284] | 0.0000 **sig.** | 0.524 | +0.0377 | 0.0000 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | +0.0124 | [+0.0078, +0.0175] | 0.0000 **sig.** | 0.431 | -0.0088 | 0.1930 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | +0.0366 | [+0.0305, +0.0429] | 0.0000 **sig.** | 0.521 | +0.0273 | 0.0040 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0233 | [+0.0185, +0.0281] | 0.0000 **sig.** | 0.461 | +0.0027 | 0.0230 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0413 | [+0.0356, +0.0468] | 0.0000 **sig.** | 0.529 | +0.0400 | 0.0010 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0126 | [+0.0097, +0.0153] | 0.0000 **sig.** | 0.466 | +0.0100 | 0.0370 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0223 | [+0.0186, +0.0263] | 0.0000 **sig.** | 0.504 | +0.0192 | 0.0070 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | +0.0328 | [+0.0298, +0.0358] | 0.0000 **sig.** | 0.550 | +0.0393 | 0.0000 |

By tier:

**head**

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | +0.0283 | [+0.0164, +0.0396] | 0.0000 **sig.** | 0.610 | +0.0339 | 0.0030 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0356 | [+0.0158, +0.0551] | 0.0000 **sig.** | 0.607 | +0.0455 | 0.0500 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | +0.0459 | [+0.0304, +0.0616] | 0.0000 **sig.** | 0.628 | +0.0597 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | +0.0033 | [-0.0197, +0.0274] | 0.7970 n.s. | 0.488 | +0.0186 | 0.8620 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | +0.0073 | [-0.0107, +0.0251] | 0.4170 n.s. | 0.503 | +0.0116 | 0.5990 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | +0.0176 | [+0.0035, +0.0314] | 0.0180 **sig.** | 0.534 | +0.0258 | 0.1490 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0307 | [-0.0631, +0.0022] | 0.0660 n.s. | 0.445 | +0.0001 | 0.2610 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0024 | [-0.0338, +0.0288] | 0.9080 n.s. | 0.509 | +0.0340 | 0.9680 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0330 | [+0.0125, +0.0553] | 0.0010 **sig.** | 0.567 | +0.0294 | 0.0700 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | +0.0253 | [+0.0072, +0.0434] | 0.0040 **sig.** | 0.537 | +0.0322 | 0.1110 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0363 | [+0.0240, +0.0480] | 0.0000 **sig.** | 0.662 | +0.0465 | 0.0000 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | -0.0125 | [-0.0303, +0.0058] | 0.1610 n.s. | 0.436 | -0.0252 | 0.4070 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | +0.0345 | [+0.0112, +0.0563] | 0.0040 **sig.** | 0.567 | +0.0189 | 0.0870 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0092 | [-0.0083, +0.0270] | 0.3050 n.s. | 0.491 | -0.0133 | 0.5480 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0207 | [-0.0003, +0.0410] | 0.0530 n.s. | 0.518 | +0.0386 | 0.2280 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0075 | [-0.0026, +0.0174] | 0.1420 n.s. | 0.515 | +0.0067 | 0.3670 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0106 | [-0.0059, +0.0268] | 0.2040 n.s. | 0.512 | +0.0168 | 0.4510 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | +0.0358 | [+0.0251, +0.0465] | 0.0000 **sig.** | 0.640 | +0.0406 | 0.0000 |

**torso**

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | +0.0229 | [+0.0182, +0.0278] | 0.0000 **sig.** | 0.602 | +0.0235 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0252 | [+0.0161, +0.0340] | 0.0000 **sig.** | 0.572 | +0.0258 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | +0.0394 | [+0.0328, +0.0460] | 0.0000 **sig.** | 0.594 | +0.0384 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | +0.0109 | [+0.0010, +0.0205] | 0.0270 **sig.** | 0.536 | +0.0083 | 0.0510 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | +0.0024 | [-0.0056, +0.0103] | 0.5470 n.s. | 0.516 | +0.0023 | 0.6050 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | +0.0165 | [+0.0107, +0.0223] | 0.0000 **sig.** | 0.519 | +0.0149 | 0.0000 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0562 | [-0.0677, -0.0440] | 0.0000 **sig.** | 0.405 | -0.0513 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0334 | [-0.0452, -0.0215] | 0.0000 **sig.** | 0.434 | -0.0278 | 0.0000 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0324 | [+0.0236, +0.0416] | 0.0000 **sig.** | 0.569 | +0.0348 | 0.0000 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | +0.0139 | [+0.0059, +0.0218] | 0.0000 **sig.** | 0.547 | +0.0157 | 0.0020 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0250 | [+0.0199, +0.0305] | 0.0000 **sig.** | 0.603 | +0.0278 | 0.0000 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | +0.0076 | [+0.0003, +0.0151] | 0.0400 **sig.** | 0.483 | +0.0083 | 0.0650 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | +0.0321 | [+0.0228, +0.0418] | 0.0000 **sig.** | 0.568 | +0.0365 | 0.0000 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0182 | [+0.0108, +0.0264] | 0.0000 **sig.** | 0.525 | +0.0208 | 0.0000 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0433 | [+0.0340, +0.0525] | 0.0000 **sig.** | 0.593 | +0.0380 | 0.0000 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0148 | [+0.0102, +0.0196] | 0.0000 **sig.** | 0.575 | +0.0133 | 0.0000 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0238 | [+0.0170, +0.0304] | 0.0000 **sig.** | 0.572 | +0.0193 | 0.0000 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | +0.0377 | [+0.0325, +0.0427] | 0.0000 **sig.** | 0.661 | +0.0368 | 0.0000 |

**tail**

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | +0.0187 | [+0.0146, +0.0227] | 0.0000 **sig.** | 0.452 | +0.0246 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0118 | [+0.0049, +0.0189] | 0.0010 **sig.** | 0.479 | +0.0169 | 0.0390 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | +0.0322 | [+0.0270, +0.0374] | 0.0000 **sig.** | 0.488 | +0.0415 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | -0.0001 | [-0.0077, +0.0075] | 0.9390 n.s. | 0.458 | +0.0025 | 0.9930 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | -0.0069 | [-0.0130, -0.0011] | 0.0250 **sig.** | 0.432 | -0.0077 | 0.1240 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | +0.0135 | [+0.0089, +0.0182] | 0.0000 **sig.** | 0.431 | +0.0169 | 0.0000 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0747 | [-0.0836, -0.0661] | 0.0000 **sig.** | 0.344 | -0.0899 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0560 | [-0.0650, -0.0469] | 0.0000 **sig.** | 0.367 | -0.0653 | 0.0000 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0344 | [+0.0273, +0.0414] | 0.0000 **sig.** | 0.489 | +0.0434 | 0.0000 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | +0.0121 | [+0.0064, +0.0179] | 0.0000 **sig.** | 0.439 | +0.0156 | 0.0100 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0243 | [+0.0198, +0.0285] | 0.0000 **sig.** | 0.485 | +0.0264 | 0.0000 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | +0.0159 | [+0.0097, +0.0219] | 0.0000 **sig.** | 0.412 | +0.0159 | 0.0000 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | +0.0383 | [+0.0308, +0.0460] | 0.0000 **sig.** | 0.500 | +0.0387 | 0.0000 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0262 | [+0.0197, +0.0322] | 0.0000 **sig.** | 0.435 | +0.0231 | 0.0000 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0420 | [+0.0353, +0.0491] | 0.0000 **sig.** | 0.506 | +0.0514 | 0.0000 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0121 | [+0.0083, +0.0157] | 0.0000 **sig.** | 0.424 | +0.0154 | 0.0000 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0226 | [+0.0173, +0.0276] | 0.0000 **sig.** | 0.479 | +0.0291 | 0.0000 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | +0.0309 | [+0.0271, +0.0350] | 0.0000 **sig.** | 0.504 | +0.0400 | 0.0000 |

## Sensitivity -- is the gain just patching E5's coverage holes?

E5 vectors exist for only 56.54% of the catalog, so a naive reading of
the headline is available: the image tower might be winning simply because it can score
candidates the baseline cannot, rather than because it carries information the baseline lacks.

Queries are split on how much of their pool E5 can actually score. `e5_covered` are the
1,191 queries where E5 reaches at least
80% of the pool -- the baseline is at full strength there,
so any remaining gain cannot be a coverage artefact. `e5_sparse` are the remaining
5,345 queries.

**e5_covered** -- baseline at full strength

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | +0.0223 | [+0.0145, +0.0302] | 0.0000 **sig.** | 0.516 | +0.0281 | 0.0350 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0259 | [+0.0147, +0.0374] | 0.0000 **sig.** | 0.509 | +0.0369 | 0.0990 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | +0.0213 | [+0.0117, +0.0314] | 0.0000 **sig.** | 0.510 | +0.0325 | 0.1210 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | -0.0056 | [-0.0189, +0.0079] | 0.3920 n.s. | 0.468 | +0.0032 | 0.7610 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | +0.0036 | [-0.0062, +0.0137] | 0.4550 n.s. | 0.463 | +0.0088 | 0.7430 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | -0.0010 | [-0.0092, +0.0074] | 0.8030 n.s. | 0.426 | +0.0044 | 0.9310 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0936 | [-0.1101, -0.0773] | 0.0000 **sig.** | 0.327 | -0.1043 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0712 | [-0.0882, -0.0550] | 0.0000 **sig.** | 0.351 | -0.0762 | 0.0020 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0279 | [+0.0162, +0.0401] | 0.0000 **sig.** | 0.512 | +0.0322 | 0.0770 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | +0.0246 | [+0.0137, +0.0361] | 0.0000 **sig.** | 0.500 | +0.0319 | 0.1230 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0258 | [+0.0178, +0.0336] | 0.0000 **sig.** | 0.534 | +0.0322 | 0.0240 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | +0.0136 | [+0.0060, +0.0212] | 0.0000 **sig.** | 0.434 | +0.0081 | 0.1760 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | +0.0450 | [+0.0317, +0.0574] | 0.0000 **sig.** | 0.542 | +0.0530 | 0.0090 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0204 | [+0.0128, +0.0284] | 0.0000 **sig.** | 0.465 | +0.0212 | 0.0390 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0451 | [+0.0318, +0.0580] | 0.0000 **sig.** | 0.546 | +0.0297 | 0.0130 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0210 | [+0.0139, +0.0283] | 0.0000 **sig.** | 0.481 | +0.0078 | 0.0280 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0257 | [+0.0165, +0.0357] | 0.0000 **sig.** | 0.503 | +0.0019 | 0.0490 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | +0.0433 | [+0.0357, +0.0510] | 0.0000 **sig.** | 0.583 | +0.0359 | 0.0000 |

**e5_sparse** -- baseline handicapped by missing vectors

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | +0.0198 | [+0.0166, +0.0231] | 0.0000 **sig.** | 0.493 | +0.0294 | 0.0060 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0142 | [+0.0083, +0.0204] | 0.0000 **sig.** | 0.509 | +0.0350 | 0.2860 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | +0.0377 | [+0.0332, +0.0421] | 0.0000 **sig.** | 0.525 | +0.0533 | 0.0010 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | +0.0047 | [-0.0021, +0.0115] | 0.1830 n.s. | 0.482 | +0.0148 | 0.7410 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | -0.0056 | [-0.0110, +0.0000] | 0.0530 n.s. | 0.455 | +0.0056 | 0.6250 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | +0.0179 | [+0.0139, +0.0221] | 0.0000 **sig.** | 0.465 | +0.0238 | 0.0300 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0621 | [-0.0698, -0.0541] | 0.0000 **sig.** | 0.373 | -0.0155 | 0.0000 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0423 | [-0.0501, -0.0347] | 0.0000 **sig.** | 0.400 | +0.0139 | 0.0100 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0351 | [+0.0290, +0.0416] | 0.0000 **sig.** | 0.514 | +0.0331 | 0.0060 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | +0.0107 | [+0.0058, +0.0154] | 0.0000 **sig.** | 0.464 | +0.0233 | 0.3360 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0249 | [+0.0215, +0.0285] | 0.0000 **sig.** | 0.521 | +0.0386 | 0.0060 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | +0.0122 | [+0.0068, +0.0176] | 0.0000 **sig.** | 0.430 | -0.0117 | 0.2660 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | +0.0347 | [+0.0280, +0.0419] | 0.0000 **sig.** | 0.516 | +0.0229 | 0.0140 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0240 | [+0.0180, +0.0298] | 0.0000 **sig.** | 0.459 | -0.0004 | 0.0480 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0404 | [+0.0341, +0.0464] | 0.0000 **sig.** | 0.525 | +0.0418 | 0.0010 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0107 | [+0.0077, +0.0136] | 0.0000 **sig.** | 0.463 | +0.0104 | 0.0800 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0216 | [+0.0172, +0.0260] | 0.0000 **sig.** | 0.504 | +0.0222 | 0.0200 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | +0.0305 | [+0.0270, +0.0337] | 0.0000 **sig.** | 0.543 | +0.0399 | 0.0000 |

If the effect survives in `e5_covered` at a similar magnitude, the image tower is contributing
genuine signal. If it collapses there and lives entirely in `e5_sparse`, the honest conclusion is
that the cheaper fix is to repair vector coverage, not to add a third tower.

## Where the lexical arm is blind

BM25 returns nothing at all for 9 queries -- plurals (`hokas`,
`uggs`, `sambas`), misspellings (`addidas`, `shoses`), and brand nicknames (`kobes`,
`sabrinas`). These are not index gaps; they are the structural failure mode of exact term
matching, and they are the cleanest test of whether a non-lexical arm rescues cases the lexical
arm cannot reach in principle.

| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `RRF(bm25 + e5 + image)` vs `RRF(bm25 + e5)` | headline: 3-way RRF vs production hybrid baseline | -0.0542 | [-0.1749, +0.0496] | 0.3440 n.s. | 0.444 | -0.0783 | 0.4610 |
| `RRF(bm25 + e5 + image)` vs `bm25` | contrast vs BM25 alone | +0.0045 | [-0.0756, +0.0753] | 0.8970 n.s. | 0.556 | +0.0008 | 0.9200 |
| `RRF(bm25 + e5 + image)` vs `e5-small` | contrast vs E5 alone | -0.0542 | [-0.1663, +0.0480] | 0.3470 n.s. | 0.444 | -0.0783 | 0.4840 |
| `RRF(bm25 + e5 + image)` vs `jina-clip-v2-image` | contrast vs image tower alone | -0.1251 | [-0.3688, +0.0401] | 0.2460 n.s. | 0.333 | -0.0111 | 0.3400 |
| `RRF(bm25 + e5)` vs `bm25` | baseline vs BM25 alone | +0.0587 | [-0.0093, +0.1252] | 0.0940 n.s. | 0.556 | +0.0791 | 0.1760 |
| `RRF(bm25 + e5)` vs `e5-small` | baseline vs E5 alone | +0.0000 | [+0.0000, +0.0000] | 2.0000 n.s. | 0.000 | +0.0000 | 2.0000 |
| `RRF(bm25 + e5)` vs `production` | baseline vs incumbent ordering | -0.0036 | [-0.1393, +0.1302] | 0.9740 n.s. | 0.556 | +0.0466 | 0.9810 |
| `RRF(bm25 + e5 + image)` vs `production` | contrast vs incumbent ordering | -0.0578 | [-0.1147, +0.0030] | 0.0620 n.s. | 0.222 | -0.0317 | 0.1150 |
| `RRF(bm25 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for E5 | +0.0710 | [-0.1595, +0.3408] | 0.6050 n.s. | 0.556 | -0.0672 | 0.7160 |
| `RRF(e5 + image)` vs `RRF(bm25 + e5)` | ablation: image tower substituted for BM25 | -0.0542 | [-0.1675, +0.0501] | 0.3440 n.s. | 0.444 | -0.0783 | 0.4890 |
| `RRF(bm25 + e5-full + image)` vs `RRF(bm25 + e5-full)` | coverage control: same contrast, fully covered E5 arm | +0.0429 | [-0.1074, +0.2000] | 0.6210 n.s. | 0.556 | -0.0521 | 0.7120 |
| `RRF(bm25 + e5-full)` vs `RRF(bm25 + e5)` | coverage control: what full E5 coverage alone is worth | -0.0699 | [-0.1368, -0.0151] | 0.0090 **sig.** | 0.222 | -0.0534 | 0.0300 |
| `RRF(e5-full + image)` vs `RRF(bm25 + e5)` | no lexical arm: fully covered E5 + image vs the hybrid | -0.0270 | [-0.1532, +0.1147] | 0.6700 n.s. | 0.444 | -0.1054 | 0.6800 |
| `RRF(e5-full + image)` vs `RRF(e5 + image)` | no lexical arm: effect of E5 coverage alone | +0.0272 | [-0.0615, +0.1357] | 0.6610 n.s. | 0.556 | -0.0271 | 0.7640 |
| `siglip-image` vs `jina-clip-v2-image` | encoder: SigLIP vs Jina CLIP v2 image tower, alone | +0.0244 | [-0.0863, +0.1294] | 0.6360 n.s. | 0.444 | +0.0288 | 0.6720 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5 + image)` | encoder: same 3-way recipe with SigLIP instead | +0.0174 | [-0.0721, +0.1207] | 0.7360 n.s. | 0.333 | +0.0351 | 0.8590 |
| `RRF(e5-full + siglip)` vs `RRF(e5-full + image)` | encoder: best recipe with SigLIP instead | +0.0220 | [-0.0491, +0.1096] | 0.6090 n.s. | 0.444 | +0.0597 | 0.7650 |
| `RRF(bm25 + e5 + siglip)` vs `RRF(bm25 + e5)` | SigLIP 3-way vs production hybrid baseline | -0.0368 | [-0.0796, +0.0066] | 0.0980 n.s. | 0.333 | -0.0431 | 0.1930 |

## Reading the ablations

`RRF(bm25 + image)` and `RRF(e5 + image)` drop one baseline arm and keep the image tower. They
separate two different claims that the headline number alone cannot: whether the image tower
adds information the existing arms lack, versus whether it merely duplicates one of them well
enough to stand in for it.

## Reproduce

```bash
python 19_embed_jina_clip_v2_full.py     # Jina CLIP v2 towers over the full test set
python 20_fetch_es_baseline.py --what all # BM25 + E5 from the live cluster
python 21_w9_hybrid_experiment.py         # fusion, metrics, bootstrap
python 22_w9_report.py                    # this document
```
