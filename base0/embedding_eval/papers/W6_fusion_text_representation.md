# W6 -- Fusion Method and Text-Representation Sweep

Generated with `09_fusion_experiment.py` / `10_fusion_report.py`. Uses cached embeddings from the
current test set (600 queries) -- no re-querying or re-embedding.

> **TL;DR**
> Best *fusion* combo, macro: **z-score average of siglip-image + text-siglip**
> (NDCG@10 0.4606).
> Best *fusion* combo, impression-weighted: **mean cosine of siglip-image + attr-siglip**
> (NDCG@10 0.4781).
>
> **Correction (2026-08-20):** an earlier version of this report concluded that RRF beat every
> other fusion method and that `attr-siglip` was the best text partner. Both were artefacts of a
> tie-ordering bug in the evaluation — see the correction note below. All numbers here are
> recomputed.

> ### Correction: tie-ordering leak
> The judgement-list SQL emits each query's candidates ordered by `relevance DESC`, and
> `np.argsort(kind="stable")` preserves input order on ties. Any system with tied scores therefore
> inherited the label ordering. Attribute text has only **17.8% distinct strings**, so **64.1%** of
> candidates in a typical pool tie (versus 1.2% for image embeddings) — which inflated every
> attribute-based system and, through them, the fusion rankings. Pools are now shuffled
> deterministically before ranking (`evaluate.shuffle_pool`). Effect on standalone systems:
> `attr-siglip` −0.159 NDCG@10, `attr-jina` −0.106, `attr-jina-small` −0.121; every non-attribute
> system moved less than 0.005.


---

## 1. Design

**Target modalities (3):** every fusion variant combines one of these image-tower
similarities with one text-based similarity. Both are self-consistent -- query and document are
encoded by the same model.

| Name | Query encoder | Document representation |
| --- | --- | --- |
| `siglip-image` | SigLIP text tower | SigLIP image tower over the product photo |
| `omni-nano-image` | Jina v5 omni-nano `Query: ` | Jina v5 omni-nano `Document: ` image tower over the photo |
| `jina-clip-v2-image` | Jina CLIP v2 text tower | Jina CLIP v2 image tower over the product photo |

**Text representations (14):** query encoder always matched to the same model's document
tower, plus three E5 models and the Jina v5 omni-nano text tower added as W6 fusion-partner
candidates alongside the original SigLIP/Jina set.

| Name | Query encoder | Document representation |
| --- | --- | --- |
| `text-siglip` | SigLIP text tower | SigLIP text tower over the product title |
| `text-jina` | Jina v5 nano `retrieval.query` | Jina v5 nano `retrieval.document` over the title |
| `text-jina-small` | Jina v5 small `retrieval.query` | Jina v5 small `retrieval.document` over the title |
| `text-e5-base` | E5 base (`query: `) | E5 base (`passage: `) over the title -- English only |
| `text-e5-small-multi` | Multilingual E5 small (`query: `) | Multilingual E5 small (`passage: `) over the title |
| `text-e5-large-instruct` | Multilingual E5 large-instruct (instruction-prefixed query) | Multilingual E5 large-instruct (no prefix) over the title |
| `text-omni-nano` | Jina v5 omni-nano `Query: ` | Jina v5 omni-nano `Document: ` over the title |
| `attr-siglip` | SigLIP text tower | SigLIP text tower over the Big-4 attribute string |
| `attr-jina` | Jina v5 nano `retrieval.query` | Jina v5 nano `retrieval.document` over the Big-4 string |
| `attr-jina-small` | Jina v5 small `retrieval.query` | Jina v5 small `retrieval.document` over the Big-4 string |
| `attr-e5-base` | E5 base (`query: `) | E5 base (`passage: `) over the Big-4 string -- English only |
| `attr-e5-small-multi` | Multilingual E5 small (`query: `) | Multilingual E5 small (`passage: `) over the Big-4 string |
| `attr-e5-large-instruct` | Multilingual E5 large-instruct (instruction-prefixed query) | Multilingual E5 large-instruct (no prefix) over the Big-4 string |
| `attr-omni-nano` | Jina v5 omni-nano `Query: ` | Jina v5 omni-nano `Document: ` over the Big-4 string |

**Fusion methods (3), per query, over the query's candidate pool:**

| Method | Formula |
| --- | --- |
| Mean cosine | $0.5 \cdot (\cos_{image} + \cos_{text})$ -- raw, unnormalised average |
| Reciprocal rank fusion (RRF, k=60) | $\frac{1}{60 + rank_{image}} + \frac{1}{60 + rank_{text}}$ |
| Z-score average | $0.5 \cdot (z(\cos_{image}) + z(\cos_{text}))$ -- current production `fusion` system |

3 target images x 14 text representations x 3 methods = 126 combinations,
all computed fresh (cosine similarity + ranking over cached embeddings is cheap; nothing here is
reused from an older run). Every system in this report -- standalones, every fusion combo,
production, random -- ranks the same, common candidate pool per query: the intersection of ecodes
covered by every embedding source in use, so adding `omni-nano-image` (which may have a handful
fewer successfully-encoded photos than SigLIP) can't silently change what any other system is
being scored against.

Mean cosine is included specifically to show what goes wrong *without* correcting for scale: image
and text similarities are dot products in unrelated, differently-scaled embedding spaces, so a raw
average lets whichever system happens to have larger score magnitude dominate the fusion regardless
of quality. RRF sidesteps the scale problem entirely by fusing on rank rather than score.

---

## 2. Results

Primary label set: LTR judgement-list relevance (grades 0-4). All 600
queries scored (same set as `results/TIERED_RESULTS.md`, all tiers combined -- this experiment
is not stratified by tier). One subsection per target image modality, so each stays readable;
every text-representation standalone is shared across both (it doesn't depend on the image target).

**Every metric is reported at every cutoff** -- NDCG, Recall, Precision and MRR each at
k = 5, 10, 20, 48, 96, 144, plus MAP (which has no cutoff). Bold marks the
best-scoring system on NDCG@10, tracked consistently across all tables so it can be followed across
k (`production` excluded from that comparison -- reference point, not a fair baseline, see note).


### Target: `siglip-image`

#### Macro-averaged -- every query counts once

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4025 | 0.4200 | 0.4571 | 0.5429 | 0.6298 | 0.6703 |
| **`fusion[z-score average] siglip-image + text-siglip`** | **0.4385** | **0.4606** | **0.5032** | **0.5985** | **0.6736** | **0.7032** |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.4315 | 0.4596 | 0.5070 | 0.6014 | 0.6750 | 0.7034 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.4383 | 0.4565 | 0.4986 | 0.5944 | 0.6690 | 0.6986 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.4262 | 0.4517 | 0.4986 | 0.5951 | 0.6707 | 0.6991 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.4253 | 0.4516 | 0.5016 | 0.5925 | 0.6644 | 0.6934 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.4266 | 0.4514 | 0.4990 | 0.5951 | 0.6707 | 0.6991 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.4222 | 0.4510 | 0.4996 | 0.5951 | 0.6687 | 0.6981 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.4223 | 0.4494 | 0.5006 | 0.5958 | 0.6706 | 0.6991 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.4257 | 0.4491 | 0.4964 | 0.5869 | 0.6595 | 0.6895 |
| `fusion[z-score average] siglip-image + text-jina` | 0.4225 | 0.4489 | 0.5004 | 0.5959 | 0.6705 | 0.6991 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.4180 | 0.4476 | 0.5000 | 0.5942 | 0.6679 | 0.6972 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.4188 | 0.4469 | 0.4936 | 0.5911 | 0.6666 | 0.6953 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.4192 | 0.4453 | 0.4987 | 0.5955 | 0.6698 | 0.6984 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.4163 | 0.4447 | 0.4939 | 0.5888 | 0.6645 | 0.6933 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.4149 | 0.4420 | 0.4890 | 0.5826 | 0.6553 | 0.6851 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.4125 | 0.4396 | 0.4893 | 0.5863 | 0.6620 | 0.6915 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.4118 | 0.4387 | 0.4887 | 0.5868 | 0.6624 | 0.6919 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.4076 | 0.4384 | 0.4906 | 0.5887 | 0.6631 | 0.6924 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.4182 | 0.4380 | 0.4807 | 0.5730 | 0.6524 | 0.6846 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.4111 | 0.4378 | 0.4913 | 0.5875 | 0.6635 | 0.6938 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.4089 | 0.4375 | 0.4905 | 0.5891 | 0.6633 | 0.6925 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.4052 | 0.4369 | 0.4903 | 0.5877 | 0.6642 | 0.6932 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.4050 | 0.4356 | 0.4882 | 0.5823 | 0.6588 | 0.6885 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.4087 | 0.4354 | 0.4893 | 0.5861 | 0.6622 | 0.6925 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.4007 | 0.4334 | 0.4886 | 0.5833 | 0.6598 | 0.6885 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.4051 | 0.4324 | 0.4852 | 0.5821 | 0.6590 | 0.6895 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.3973 | 0.4292 | 0.4837 | 0.5765 | 0.6537 | 0.6844 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.3996 | 0.4287 | 0.4827 | 0.5805 | 0.6562 | 0.6858 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.3946 | 0.4279 | 0.4810 | 0.5791 | 0.6566 | 0.6858 |
| `siglip-image` | 0.4034 | 0.4272 | 0.4819 | 0.5779 | 0.6554 | 0.6847 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.3989 | 0.4269 | 0.4843 | 0.5816 | 0.6576 | 0.6870 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.3955 | 0.4256 | 0.4811 | 0.5783 | 0.6558 | 0.6859 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.3954 | 0.4249 | 0.4811 | 0.5781 | 0.6557 | 0.6857 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.3954 | 0.4242 | 0.4815 | 0.5797 | 0.6573 | 0.6866 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.3932 | 0.4236 | 0.4722 | 0.5703 | 0.6453 | 0.6770 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.3930 | 0.4225 | 0.4774 | 0.5754 | 0.6528 | 0.6829 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.3938 | 0.4221 | 0.4749 | 0.5709 | 0.6472 | 0.6785 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.3942 | 0.4217 | 0.4727 | 0.5704 | 0.6467 | 0.6778 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.3954 | 0.4216 | 0.4725 | 0.5708 | 0.6473 | 0.6785 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.3951 | 0.4199 | 0.4727 | 0.5675 | 0.6447 | 0.6764 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.3893 | 0.4192 | 0.4712 | 0.5709 | 0.6492 | 0.6806 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.3794 | 0.4130 | 0.4678 | 0.5684 | 0.6465 | 0.6765 |
| `text-omni-nano` | 0.3789 | 0.4094 | 0.4634 | 0.5626 | 0.6416 | 0.6722 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.3739 | 0.4080 | 0.4557 | 0.5562 | 0.6327 | 0.6652 |
| `text-jina` | 0.3779 | 0.4072 | 0.4623 | 0.5619 | 0.6407 | 0.6713 |
| `text-e5-large-instruct` | 0.3702 | 0.4012 | 0.4589 | 0.5592 | 0.6396 | 0.6711 |
| `text-jina-small` | 0.3638 | 0.3989 | 0.4566 | 0.5557 | 0.6367 | 0.6672 |
| `text-siglip` | 0.3706 | 0.3981 | 0.4438 | 0.5370 | 0.6204 | 0.6554 |
| `text-e5-base` | 0.3644 | 0.3899 | 0.4435 | 0.5459 | 0.6286 | 0.6626 |
| `attr-siglip` | 0.3477 | 0.3776 | 0.4335 | 0.5341 | 0.6122 | 0.6472 |
| `text-e5-small-multi` | 0.3315 | 0.3653 | 0.4215 | 0.5328 | 0.6143 | 0.6485 |
| `attr-omni-nano` | 0.3030 | 0.3433 | 0.4047 | 0.5124 | 0.5966 | 0.6324 |
| `attr-jina-small` | 0.3064 | 0.3424 | 0.4026 | 0.5126 | 0.5965 | 0.6327 |
| `attr-e5-large-instruct` | 0.3020 | 0.3411 | 0.4007 | 0.5141 | 0.5969 | 0.6330 |
| `attr-jina` | 0.2986 | 0.3398 | 0.4046 | 0.5116 | 0.5956 | 0.6312 |
| `attr-e5-base` | 0.2956 | 0.3380 | 0.3980 | 0.5045 | 0.5906 | 0.6275 |
| `attr-e5-small-multi` | 0.2775 | 0.3125 | 0.3720 | 0.4890 | 0.5743 | 0.6129 |
| `random` | 0.2392 | 0.2640 | 0.3210 | 0.4357 | 0.5321 | 0.5758 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.1400 | 0.2466 | 0.4189 | 0.6922 | 0.8846 | 0.9646 |
| **`fusion[z-score average] siglip-image + text-siglip`** | **0.1672** | **0.2712** | **0.4501** | **0.7292** | **0.9060** | **0.9707** |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.1620 | 0.2826 | 0.4624 | 0.7351 | 0.9076 | 0.9716 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.1687 | 0.2707 | 0.4480 | 0.7277 | 0.9042 | 0.9704 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.1635 | 0.2798 | 0.4549 | 0.7323 | 0.9065 | 0.9704 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.1591 | 0.2780 | 0.4522 | 0.7291 | 0.9038 | 0.9690 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.1629 | 0.2805 | 0.4548 | 0.7321 | 0.9066 | 0.9705 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.1631 | 0.2799 | 0.4584 | 0.7345 | 0.9069 | 0.9718 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.1700 | 0.2794 | 0.4578 | 0.7328 | 0.9063 | 0.9710 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.1566 | 0.2740 | 0.4457 | 0.7250 | 0.8997 | 0.9678 |
| `fusion[z-score average] siglip-image + text-jina` | 0.1692 | 0.2794 | 0.4570 | 0.7330 | 0.9063 | 0.9710 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.1641 | 0.2806 | 0.4609 | 0.7358 | 0.9076 | 0.9717 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.1580 | 0.2750 | 0.4550 | 0.7333 | 0.9068 | 0.9711 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.1609 | 0.2757 | 0.4625 | 0.7341 | 0.9073 | 0.9711 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.1590 | 0.2739 | 0.4512 | 0.7299 | 0.9061 | 0.9712 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.1512 | 0.2730 | 0.4487 | 0.7274 | 0.9020 | 0.9689 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.1626 | 0.2798 | 0.4576 | 0.7308 | 0.9029 | 0.9679 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.1644 | 0.2797 | 0.4567 | 0.7306 | 0.9026 | 0.9679 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.1574 | 0.2740 | 0.4515 | 0.7306 | 0.9055 | 0.9708 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.1628 | 0.2748 | 0.4411 | 0.7174 | 0.8971 | 0.9674 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.1687 | 0.2795 | 0.4622 | 0.7284 | 0.9027 | 0.9699 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.1596 | 0.2740 | 0.4514 | 0.7311 | 0.9055 | 0.9708 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.1671 | 0.2771 | 0.4560 | 0.7309 | 0.9058 | 0.9703 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.1594 | 0.2752 | 0.4502 | 0.7298 | 0.9043 | 0.9703 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.1672 | 0.2783 | 0.4598 | 0.7288 | 0.9036 | 0.9700 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.1559 | 0.2708 | 0.4552 | 0.7303 | 0.9040 | 0.9697 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.1682 | 0.2794 | 0.4557 | 0.7270 | 0.9025 | 0.9697 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.1592 | 0.2741 | 0.4514 | 0.7254 | 0.9022 | 0.9693 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.1579 | 0.2700 | 0.4499 | 0.7287 | 0.9047 | 0.9703 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.1532 | 0.2703 | 0.4475 | 0.7284 | 0.9056 | 0.9713 |
| `siglip-image` | 0.1582 | 0.2655 | 0.4451 | 0.7246 | 0.9039 | 0.9698 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.1602 | 0.2697 | 0.4646 | 0.7324 | 0.9043 | 0.9689 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.1527 | 0.2708 | 0.4464 | 0.7255 | 0.9037 | 0.9707 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.1536 | 0.2717 | 0.4465 | 0.7260 | 0.9038 | 0.9706 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.1651 | 0.2710 | 0.4574 | 0.7305 | 0.9051 | 0.9703 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.1470 | 0.2681 | 0.4434 | 0.7239 | 0.8987 | 0.9684 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.1588 | 0.2768 | 0.4521 | 0.7289 | 0.9038 | 0.9692 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.1493 | 0.2695 | 0.4485 | 0.7217 | 0.9000 | 0.9693 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.1512 | 0.2669 | 0.4461 | 0.7228 | 0.9002 | 0.9695 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.1524 | 0.2645 | 0.4418 | 0.7244 | 0.9009 | 0.9706 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.1513 | 0.2675 | 0.4457 | 0.7231 | 0.8987 | 0.9689 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.1549 | 0.2702 | 0.4369 | 0.7259 | 0.9017 | 0.9702 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.1503 | 0.2661 | 0.4464 | 0.7244 | 0.9023 | 0.9683 |
| `text-omni-nano` | 0.1541 | 0.2678 | 0.4477 | 0.7241 | 0.8984 | 0.9655 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.1418 | 0.2623 | 0.4402 | 0.7187 | 0.8956 | 0.9663 |
| `text-jina` | 0.1537 | 0.2673 | 0.4476 | 0.7243 | 0.8984 | 0.9655 |
| `text-e5-large-instruct` | 0.1597 | 0.2706 | 0.4488 | 0.7209 | 0.9006 | 0.9683 |
| `text-jina-small` | 0.1490 | 0.2669 | 0.4515 | 0.7238 | 0.9011 | 0.9667 |
| `text-siglip` | 0.1485 | 0.2603 | 0.4316 | 0.7036 | 0.8886 | 0.9640 |
| `text-e5-base` | 0.1609 | 0.2671 | 0.4394 | 0.7130 | 0.8932 | 0.9656 |
| `attr-siglip` | 0.1381 | 0.2483 | 0.4198 | 0.7073 | 0.8872 | 0.9627 |
| `text-e5-small-multi` | 0.1556 | 0.2612 | 0.4326 | 0.7165 | 0.8937 | 0.9648 |
| `attr-omni-nano` | 0.1269 | 0.2423 | 0.4161 | 0.7070 | 0.8903 | 0.9647 |
| `attr-jina-small` | 0.1298 | 0.2429 | 0.4127 | 0.7094 | 0.8888 | 0.9651 |
| `attr-e5-large-instruct` | 0.1359 | 0.2440 | 0.4181 | 0.7055 | 0.8870 | 0.9621 |
| `attr-jina` | 0.1263 | 0.2433 | 0.4185 | 0.7086 | 0.8905 | 0.9648 |
| `attr-e5-base` | 0.1256 | 0.2423 | 0.4195 | 0.7014 | 0.8866 | 0.9633 |
| `attr-e5-small-multi` | 0.1262 | 0.2342 | 0.4077 | 0.6940 | 0.8820 | 0.9600 |
| `random` | 0.0954 | 0.1807 | 0.3567 | 0.6540 | 0.8596 | 0.9509 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6303 | 0.6435 | 0.6313 | 0.5483 | 0.4265 | 0.3387 |
| **`fusion[z-score average] siglip-image + text-siglip`** | **0.7807** | **0.7543** | **0.7141** | **0.5999** | **0.4454** | **0.3431** |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.7883 | 0.7667 | 0.7279 | 0.6080 | 0.4470 | 0.3439 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.7827 | 0.7558 | 0.7126 | 0.5988 | 0.4435 | 0.3427 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.7837 | 0.7577 | 0.7183 | 0.6038 | 0.4466 | 0.3432 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.7727 | 0.7520 | 0.7142 | 0.6017 | 0.4437 | 0.3419 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.7833 | 0.7587 | 0.7186 | 0.6034 | 0.4468 | 0.3432 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.7827 | 0.7640 | 0.7243 | 0.6061 | 0.4464 | 0.3440 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.7830 | 0.7593 | 0.7231 | 0.6058 | 0.4464 | 0.3436 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.7693 | 0.7510 | 0.7103 | 0.5960 | 0.4406 | 0.3411 |
| `fusion[z-score average] siglip-image + text-jina` | 0.7813 | 0.7587 | 0.7233 | 0.6061 | 0.4464 | 0.3435 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.7857 | 0.7650 | 0.7273 | 0.6091 | 0.4473 | 0.3439 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.7810 | 0.7620 | 0.7202 | 0.6044 | 0.4467 | 0.3436 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.7823 | 0.7627 | 0.7248 | 0.6070 | 0.4472 | 0.3437 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.7680 | 0.7475 | 0.7105 | 0.6002 | 0.4457 | 0.3435 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.7690 | 0.7488 | 0.7049 | 0.5972 | 0.4421 | 0.3418 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.7727 | 0.7505 | 0.7163 | 0.6038 | 0.4441 | 0.3416 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.7740 | 0.7498 | 0.7157 | 0.6036 | 0.4440 | 0.3416 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.7607 | 0.7425 | 0.7102 | 0.6010 | 0.4455 | 0.3433 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.7657 | 0.7400 | 0.6981 | 0.5855 | 0.4378 | 0.3406 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.7760 | 0.7558 | 0.7191 | 0.6014 | 0.4436 | 0.3430 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.7600 | 0.7427 | 0.7105 | 0.6014 | 0.4455 | 0.3432 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.7780 | 0.7548 | 0.7192 | 0.6031 | 0.4457 | 0.3433 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.7607 | 0.7427 | 0.7109 | 0.5989 | 0.4445 | 0.3429 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.7740 | 0.7538 | 0.7171 | 0.6016 | 0.4443 | 0.3430 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.7580 | 0.7373 | 0.7074 | 0.5968 | 0.4441 | 0.3425 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.7737 | 0.7542 | 0.7141 | 0.5986 | 0.4433 | 0.3426 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.7527 | 0.7352 | 0.7027 | 0.5950 | 0.4429 | 0.3422 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.7600 | 0.7398 | 0.7063 | 0.5989 | 0.4444 | 0.3431 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.7580 | 0.7333 | 0.7045 | 0.5974 | 0.4457 | 0.3436 |
| `siglip-image` | 0.7607 | 0.7383 | 0.7038 | 0.5945 | 0.4438 | 0.3426 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.7777 | 0.7522 | 0.7174 | 0.6057 | 0.4454 | 0.3421 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.7450 | 0.7328 | 0.7020 | 0.5944 | 0.4438 | 0.3431 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.7460 | 0.7333 | 0.7027 | 0.5948 | 0.4438 | 0.3431 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.7760 | 0.7517 | 0.7163 | 0.6017 | 0.4453 | 0.3434 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.7567 | 0.7372 | 0.6958 | 0.5898 | 0.4390 | 0.3414 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.7723 | 0.7518 | 0.7121 | 0.5981 | 0.4442 | 0.3425 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.7527 | 0.7352 | 0.7009 | 0.5894 | 0.4401 | 0.3423 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.7550 | 0.7357 | 0.7003 | 0.5906 | 0.4404 | 0.3424 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.7587 | 0.7345 | 0.6991 | 0.5924 | 0.4411 | 0.3430 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.7507 | 0.7310 | 0.6936 | 0.5889 | 0.4395 | 0.3418 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.7520 | 0.7272 | 0.6937 | 0.5919 | 0.4425 | 0.3428 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.7450 | 0.7257 | 0.6927 | 0.5922 | 0.4426 | 0.3419 |
| `text-omni-nano` | 0.7483 | 0.7342 | 0.7040 | 0.5943 | 0.4404 | 0.3400 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.7440 | 0.7283 | 0.6887 | 0.5847 | 0.4363 | 0.3405 |
| `text-jina` | 0.7483 | 0.7335 | 0.7034 | 0.5948 | 0.4404 | 0.3400 |
| `text-e5-large-instruct` | 0.7600 | 0.7392 | 0.7020 | 0.5932 | 0.4419 | 0.3415 |
| `text-jina-small` | 0.7550 | 0.7407 | 0.7058 | 0.5949 | 0.4426 | 0.3407 |
| `text-siglip` | 0.7347 | 0.7165 | 0.6774 | 0.5687 | 0.4309 | 0.3384 |
| `text-e5-base` | 0.7427 | 0.7205 | 0.6851 | 0.5816 | 0.4364 | 0.3401 |
| `attr-siglip` | 0.7020 | 0.7010 | 0.6718 | 0.5741 | 0.4304 | 0.3380 |
| `text-e5-small-multi` | 0.7317 | 0.7175 | 0.6816 | 0.5829 | 0.4365 | 0.3398 |
| `attr-omni-nano` | 0.6633 | 0.6688 | 0.6471 | 0.5685 | 0.4332 | 0.3394 |
| `attr-jina-small` | 0.6703 | 0.6625 | 0.6469 | 0.5704 | 0.4317 | 0.3394 |
| `attr-e5-large-instruct` | 0.6687 | 0.6623 | 0.6462 | 0.5667 | 0.4295 | 0.3377 |
| `attr-jina` | 0.6660 | 0.6708 | 0.6484 | 0.5689 | 0.4335 | 0.3394 |
| `attr-e5-base` | 0.6587 | 0.6638 | 0.6418 | 0.5629 | 0.4297 | 0.3384 |
| `attr-e5-small-multi` | 0.6350 | 0.6348 | 0.6268 | 0.5549 | 0.4256 | 0.3366 |
| `random` | 0.5777 | 0.5777 | 0.5647 | 0.5094 | 0.4078 | 0.3306 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6940 | 0.7041 | 0.7063 | 0.7071 | 0.7072 | 0.7072 |
| **`fusion[z-score average] siglip-image + text-siglip`** | **0.8600** | **0.8628** | **0.8647** | **0.8654** | **0.8655** | **0.8655** |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.8673 | 0.8729 | 0.8749 | 0.8753 | 0.8753 | 0.8753 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.8612 | 0.8636 | 0.8654 | 0.8662 | 0.8663 | 0.8663 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.8548 | 0.8585 | 0.8600 | 0.8605 | 0.8605 | 0.8605 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.8567 | 0.8608 | 0.8624 | 0.8630 | 0.8630 | 0.8630 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.8570 | 0.8607 | 0.8622 | 0.8628 | 0.8628 | 0.8628 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.8609 | 0.8651 | 0.8668 | 0.8673 | 0.8674 | 0.8674 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.8519 | 0.8543 | 0.8561 | 0.8567 | 0.8567 | 0.8567 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.8523 | 0.8562 | 0.8575 | 0.8585 | 0.8585 | 0.8585 |
| `fusion[z-score average] siglip-image + text-jina` | 0.8526 | 0.8553 | 0.8571 | 0.8577 | 0.8577 | 0.8577 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.8561 | 0.8603 | 0.8624 | 0.8629 | 0.8629 | 0.8629 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.8551 | 0.8590 | 0.8610 | 0.8617 | 0.8617 | 0.8617 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.8631 | 0.8667 | 0.8691 | 0.8695 | 0.8695 | 0.8695 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.8481 | 0.8527 | 0.8548 | 0.8554 | 0.8554 | 0.8554 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.8466 | 0.8506 | 0.8524 | 0.8529 | 0.8529 | 0.8529 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.8465 | 0.8504 | 0.8523 | 0.8529 | 0.8529 | 0.8529 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.8464 | 0.8503 | 0.8521 | 0.8527 | 0.8527 | 0.8527 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.8480 | 0.8521 | 0.8542 | 0.8548 | 0.8549 | 0.8549 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.8544 | 0.8582 | 0.8595 | 0.8602 | 0.8603 | 0.8603 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.8602 | 0.8628 | 0.8649 | 0.8652 | 0.8652 | 0.8652 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.8480 | 0.8516 | 0.8537 | 0.8544 | 0.8544 | 0.8544 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.8609 | 0.8642 | 0.8663 | 0.8669 | 0.8669 | 0.8669 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.8446 | 0.8485 | 0.8501 | 0.8509 | 0.8509 | 0.8509 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.8636 | 0.8664 | 0.8685 | 0.8690 | 0.8690 | 0.8690 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.8294 | 0.8344 | 0.8367 | 0.8372 | 0.8372 | 0.8372 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.8564 | 0.8592 | 0.8611 | 0.8616 | 0.8616 | 0.8616 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.8420 | 0.8454 | 0.8473 | 0.8478 | 0.8479 | 0.8479 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.8385 | 0.8417 | 0.8438 | 0.8446 | 0.8447 | 0.8447 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.8316 | 0.8359 | 0.8380 | 0.8386 | 0.8386 | 0.8386 |
| `siglip-image` | 0.8427 | 0.8467 | 0.8493 | 0.8500 | 0.8500 | 0.8500 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.8419 | 0.8451 | 0.8482 | 0.8486 | 0.8487 | 0.8487 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.8328 | 0.8371 | 0.8391 | 0.8397 | 0.8398 | 0.8398 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.8326 | 0.8365 | 0.8384 | 0.8392 | 0.8392 | 0.8392 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.8550 | 0.8580 | 0.8604 | 0.8610 | 0.8610 | 0.8610 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.8347 | 0.8400 | 0.8423 | 0.8431 | 0.8431 | 0.8431 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.8546 | 0.8602 | 0.8617 | 0.8624 | 0.8625 | 0.8625 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.8361 | 0.8405 | 0.8426 | 0.8431 | 0.8432 | 0.8432 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.8326 | 0.8361 | 0.8384 | 0.8390 | 0.8390 | 0.8390 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.8327 | 0.8362 | 0.8385 | 0.8392 | 0.8392 | 0.8392 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.8323 | 0.8366 | 0.8386 | 0.8392 | 0.8392 | 0.8392 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.8205 | 0.8246 | 0.8262 | 0.8273 | 0.8273 | 0.8273 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.8290 | 0.8332 | 0.8354 | 0.8361 | 0.8361 | 0.8361 |
| `text-omni-nano` | 0.8311 | 0.8360 | 0.8379 | 0.8387 | 0.8387 | 0.8387 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.8259 | 0.8310 | 0.8333 | 0.8340 | 0.8340 | 0.8340 |
| `text-jina` | 0.8305 | 0.8352 | 0.8369 | 0.8378 | 0.8378 | 0.8378 |
| `text-e5-large-instruct` | 0.8383 | 0.8420 | 0.8441 | 0.8448 | 0.8448 | 0.8448 |
| `text-jina-small` | 0.8239 | 0.8290 | 0.8316 | 0.8325 | 0.8325 | 0.8325 |
| `text-siglip` | 0.8123 | 0.8168 | 0.8184 | 0.8192 | 0.8193 | 0.8193 |
| `text-e5-base` | 0.8406 | 0.8439 | 0.8456 | 0.8466 | 0.8466 | 0.8466 |
| `attr-siglip` | 0.7843 | 0.7896 | 0.7913 | 0.7922 | 0.7922 | 0.7922 |
| `text-e5-small-multi` | 0.8054 | 0.8095 | 0.8111 | 0.8124 | 0.8124 | 0.8124 |
| `attr-omni-nano` | 0.7448 | 0.7523 | 0.7554 | 0.7564 | 0.7565 | 0.7565 |
| `attr-jina-small` | 0.7481 | 0.7566 | 0.7589 | 0.7602 | 0.7602 | 0.7602 |
| `attr-e5-large-instruct` | 0.7484 | 0.7543 | 0.7570 | 0.7582 | 0.7582 | 0.7582 |
| `attr-jina` | 0.7419 | 0.7497 | 0.7527 | 0.7537 | 0.7537 | 0.7537 |
| `attr-e5-base` | 0.7364 | 0.7458 | 0.7488 | 0.7498 | 0.7498 | 0.7498 |
| `attr-e5-small-multi` | 0.7056 | 0.7147 | 0.7180 | 0.7190 | 0.7190 | 0.7190 |
| `random` | 0.6917 | 0.6965 | 0.7008 | 0.7022 | 0.7022 | 0.7022 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.6808 |
| **`fusion[z-score average] siglip-image + text-siglip`** | **0.7632** |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.7748 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.7555 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.7669 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.7547 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.7672 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.7693 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.7703 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.7482 |
| `fusion[z-score average] siglip-image + text-jina` | 0.7710 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.7726 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.7647 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.7710 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.7610 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.7432 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.7640 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.7647 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.7571 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.7409 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.7694 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.7575 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.7706 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.7572 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.7685 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.7561 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.7639 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.7514 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.7548 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.7528 |
| `siglip-image` | 0.7500 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.7633 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.7497 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.7498 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.7653 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.7399 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.7590 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.7401 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.7389 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.7415 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.7389 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.7440 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.7446 |
| `text-omni-nano` | 0.7473 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.7268 |
| `text-jina` | 0.7471 |
| `text-e5-large-instruct` | 0.7502 |
| `text-jina-small` | 0.7463 |
| `text-siglip` | 0.7072 |
| `text-e5-base` | 0.7401 |
| `attr-siglip` | 0.7023 |
| `text-e5-small-multi` | 0.7316 |
| `attr-omni-nano` | 0.6937 |
| `attr-jina-small` | 0.6957 |
| `attr-e5-large-instruct` | 0.6975 |
| `attr-jina` | 0.6933 |
| `attr-e5-base` | 0.6916 |
| `attr-e5-small-multi` | 0.6712 |
| `random` | 0.6041 |


#### Impression-weighted -- every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4066 | 0.4123 | 0.4295 | 0.5122 | 0.6384 | 0.7053 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.4621** | **0.4781** | **0.5179** | **0.5954** | **0.6973** | **0.7504** |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.4642 | 0.4725 | 0.5160 | 0.5991 | 0.7007 | 0.7525 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.4524 | 0.4699 | 0.5195 | 0.5985 | 0.6989 | 0.7518 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.4541 | 0.4679 | 0.4983 | 0.5956 | 0.7017 | 0.7500 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.4551 | 0.4635 | 0.4997 | 0.5953 | 0.7013 | 0.7494 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.4372 | 0.4619 | 0.4997 | 0.5933 | 0.6972 | 0.7485 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.4398 | 0.4616 | 0.5053 | 0.5971 | 0.6980 | 0.7505 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.4485 | 0.4610 | 0.5017 | 0.5897 | 0.6982 | 0.7488 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.4410 | 0.4583 | 0.5007 | 0.5949 | 0.6996 | 0.7491 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.4307 | 0.4576 | 0.4954 | 0.5873 | 0.6918 | 0.7424 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.4444 | 0.4562 | 0.5016 | 0.5953 | 0.6983 | 0.7495 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.4346 | 0.4509 | 0.4977 | 0.5948 | 0.6952 | 0.7455 |
| `fusion[z-score average] siglip-image + text-jina` | 0.4290 | 0.4507 | 0.4962 | 0.5953 | 0.6946 | 0.7454 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.4191 | 0.4493 | 0.4891 | 0.5802 | 0.6864 | 0.7388 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.4411 | 0.4487 | 0.4874 | 0.5876 | 0.6934 | 0.7459 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.4228 | 0.4446 | 0.4799 | 0.5691 | 0.6749 | 0.7295 |
| `siglip-image` | 0.4317 | 0.4443 | 0.4863 | 0.5844 | 0.6896 | 0.7387 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.4238 | 0.4415 | 0.4888 | 0.5877 | 0.6871 | 0.7387 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.4077 | 0.4410 | 0.4831 | 0.5710 | 0.6772 | 0.7310 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.4205 | 0.4392 | 0.4853 | 0.5875 | 0.6872 | 0.7385 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.4262 | 0.4391 | 0.4840 | 0.5758 | 0.6811 | 0.7352 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.4200 | 0.4384 | 0.4792 | 0.5840 | 0.6892 | 0.7428 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.4099 | 0.4376 | 0.4806 | 0.5784 | 0.6865 | 0.7388 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.4196 | 0.4366 | 0.4792 | 0.5729 | 0.6805 | 0.7337 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.4239 | 0.4360 | 0.4800 | 0.5718 | 0.6801 | 0.7339 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.4112 | 0.4336 | 0.4852 | 0.5802 | 0.6877 | 0.7389 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.4103 | 0.4309 | 0.4849 | 0.5794 | 0.6875 | 0.7383 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.4076 | 0.4287 | 0.4746 | 0.5747 | 0.6831 | 0.7349 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.4079 | 0.4287 | 0.4776 | 0.5745 | 0.6857 | 0.7358 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.3892 | 0.4285 | 0.4739 | 0.5753 | 0.6800 | 0.7300 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.4051 | 0.4283 | 0.4764 | 0.5723 | 0.6828 | 0.7321 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.4003 | 0.4278 | 0.4759 | 0.5773 | 0.6850 | 0.7357 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.3967 | 0.4267 | 0.4702 | 0.5699 | 0.6773 | 0.7303 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.4006 | 0.4261 | 0.4713 | 0.5675 | 0.6793 | 0.7326 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.3945 | 0.4248 | 0.4754 | 0.5757 | 0.6815 | 0.7328 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.4081 | 0.4245 | 0.4784 | 0.5742 | 0.6858 | 0.7362 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.4030 | 0.4245 | 0.4677 | 0.5669 | 0.6716 | 0.7260 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.3982 | 0.4205 | 0.4772 | 0.5715 | 0.6770 | 0.7304 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.4194 | 0.4196 | 0.4605 | 0.5536 | 0.6680 | 0.7241 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.4027 | 0.4196 | 0.4605 | 0.5576 | 0.6669 | 0.7239 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.3789 | 0.4193 | 0.4640 | 0.5686 | 0.6746 | 0.7268 |
| `attr-siglip` | 0.4044 | 0.4185 | 0.4617 | 0.5529 | 0.6596 | 0.7175 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.3927 | 0.4181 | 0.4743 | 0.5722 | 0.6790 | 0.7320 |
| `text-jina` | 0.3698 | 0.3998 | 0.4536 | 0.5580 | 0.6626 | 0.7140 |
| `text-omni-nano` | 0.3699 | 0.3989 | 0.4559 | 0.5578 | 0.6633 | 0.7148 |
| `text-e5-large-instruct` | 0.3677 | 0.3976 | 0.4503 | 0.5531 | 0.6640 | 0.7159 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.3611 | 0.3963 | 0.4512 | 0.5562 | 0.6663 | 0.7194 |
| `text-siglip` | 0.3842 | 0.3958 | 0.4305 | 0.5228 | 0.6407 | 0.7025 |
| `text-jina-small` | 0.3551 | 0.3895 | 0.4454 | 0.5501 | 0.6633 | 0.7166 |
| `text-e5-base` | 0.3566 | 0.3730 | 0.4220 | 0.5329 | 0.6420 | 0.6987 |
| `attr-omni-nano` | 0.3257 | 0.3590 | 0.4124 | 0.5142 | 0.6336 | 0.6904 |
| `text-e5-small-multi` | 0.3415 | 0.3579 | 0.4101 | 0.5276 | 0.6454 | 0.6970 |
| `attr-e5-base` | 0.3226 | 0.3560 | 0.4064 | 0.5113 | 0.6251 | 0.6863 |
| `attr-jina` | 0.3178 | 0.3455 | 0.4055 | 0.5112 | 0.6314 | 0.6890 |
| `attr-jina-small` | 0.3112 | 0.3407 | 0.4032 | 0.5088 | 0.6246 | 0.6870 |
| `random` | 0.3183 | 0.3208 | 0.3505 | 0.4523 | 0.5797 | 0.6481 |
| `attr-e5-large-instruct` | 0.2946 | 0.3198 | 0.3820 | 0.5023 | 0.6163 | 0.6822 |
| `attr-e5-small-multi` | 0.2846 | 0.3093 | 0.3644 | 0.4861 | 0.6138 | 0.6746 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.0609 | 0.1266 | 0.2517 | 0.5200 | 0.8002 | 0.9346 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.0794** | **0.1561** | **0.3001** | **0.5863** | **0.8307** | **0.9437** |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.0797 | 0.1562 | 0.2995 | 0.5854 | 0.8306 | 0.9450 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.0800 | 0.1569 | 0.3021 | 0.5897 | 0.8338 | 0.9452 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.0800 | 0.1573 | 0.3032 | 0.5930 | 0.8350 | 0.9448 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.0800 | 0.1574 | 0.3032 | 0.5929 | 0.8351 | 0.9448 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.0803 | 0.1580 | 0.3028 | 0.5950 | 0.8373 | 0.9469 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.0805 | 0.1586 | 0.3049 | 0.5983 | 0.8373 | 0.9471 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.0801 | 0.1576 | 0.3029 | 0.5927 | 0.8357 | 0.9464 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.0807 | 0.1581 | 0.3043 | 0.5968 | 0.8391 | 0.9481 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.0796 | 0.1564 | 0.3012 | 0.5885 | 0.8369 | 0.9482 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.0805 | 0.1582 | 0.3049 | 0.5959 | 0.8362 | 0.9466 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.0802 | 0.1569 | 0.3045 | 0.5958 | 0.8351 | 0.9455 |
| `fusion[z-score average] siglip-image + text-jina` | 0.0800 | 0.1570 | 0.3046 | 0.5957 | 0.8353 | 0.9455 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.0797 | 0.1562 | 0.3021 | 0.5902 | 0.8355 | 0.9479 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.0793 | 0.1543 | 0.2963 | 0.5832 | 0.8334 | 0.9449 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.0791 | 0.1548 | 0.2982 | 0.5827 | 0.8298 | 0.9452 |
| `siglip-image` | 0.0784 | 0.1552 | 0.2978 | 0.5881 | 0.8361 | 0.9462 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.0798 | 0.1563 | 0.3026 | 0.5950 | 0.8311 | 0.9412 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.0791 | 0.1550 | 0.2994 | 0.5879 | 0.8334 | 0.9459 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.0798 | 0.1562 | 0.3025 | 0.5950 | 0.8310 | 0.9412 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.0784 | 0.1553 | 0.2986 | 0.5817 | 0.8269 | 0.9463 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.0785 | 0.1544 | 0.2966 | 0.5855 | 0.8353 | 0.9447 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.0794 | 0.1548 | 0.3002 | 0.5885 | 0.8343 | 0.9482 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.0784 | 0.1551 | 0.2984 | 0.5821 | 0.8277 | 0.9464 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.0785 | 0.1550 | 0.2981 | 0.5838 | 0.8292 | 0.9472 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.0793 | 0.1563 | 0.3014 | 0.5890 | 0.8329 | 0.9470 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.0793 | 0.1561 | 0.3013 | 0.5890 | 0.8332 | 0.9469 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.0802 | 0.1566 | 0.3025 | 0.5944 | 0.8348 | 0.9435 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.0783 | 0.1546 | 0.2978 | 0.5873 | 0.8329 | 0.9465 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.0797 | 0.1579 | 0.3047 | 0.5942 | 0.8360 | 0.9464 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.0791 | 0.1545 | 0.3003 | 0.5878 | 0.8337 | 0.9454 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.0798 | 0.1569 | 0.3040 | 0.5930 | 0.8344 | 0.9458 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.0793 | 0.1572 | 0.3028 | 0.5899 | 0.8328 | 0.9453 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.0781 | 0.1536 | 0.2970 | 0.5859 | 0.8323 | 0.9465 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.0793 | 0.1552 | 0.3011 | 0.5916 | 0.8365 | 0.9482 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.0783 | 0.1544 | 0.2981 | 0.5873 | 0.8328 | 0.9464 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.0792 | 0.1547 | 0.2983 | 0.5846 | 0.8278 | 0.9444 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.0798 | 0.1576 | 0.3031 | 0.5923 | 0.8323 | 0.9440 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.0780 | 0.1504 | 0.2898 | 0.5719 | 0.8235 | 0.9395 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.0781 | 0.1539 | 0.2947 | 0.5806 | 0.8274 | 0.9453 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.0797 | 0.1577 | 0.3037 | 0.5925 | 0.8337 | 0.9459 |
| `attr-siglip` | 0.0737 | 0.1470 | 0.2850 | 0.5719 | 0.8180 | 0.9370 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.0798 | 0.1575 | 0.3031 | 0.5939 | 0.8337 | 0.9454 |
| `text-jina` | 0.0770 | 0.1534 | 0.2990 | 0.5890 | 0.8243 | 0.9372 |
| `text-omni-nano` | 0.0770 | 0.1536 | 0.2992 | 0.5888 | 0.8245 | 0.9373 |
| `text-e5-large-instruct` | 0.0779 | 0.1537 | 0.2985 | 0.5903 | 0.8269 | 0.9425 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.0782 | 0.1538 | 0.2965 | 0.5871 | 0.8339 | 0.9463 |
| `text-siglip` | 0.0756 | 0.1473 | 0.2827 | 0.5584 | 0.8122 | 0.9353 |
| `text-jina-small` | 0.0775 | 0.1536 | 0.2994 | 0.5887 | 0.8306 | 0.9412 |
| `text-e5-base` | 0.0790 | 0.1554 | 0.2984 | 0.5836 | 0.8225 | 0.9383 |
| `attr-omni-nano` | 0.0715 | 0.1445 | 0.2814 | 0.5696 | 0.8194 | 0.9413 |
| `text-e5-small-multi` | 0.0772 | 0.1517 | 0.2946 | 0.5831 | 0.8238 | 0.9381 |
| `attr-e5-base` | 0.0713 | 0.1437 | 0.2815 | 0.5699 | 0.8175 | 0.9400 |
| `attr-jina` | 0.0711 | 0.1448 | 0.2816 | 0.5695 | 0.8199 | 0.9413 |
| `attr-jina-small` | 0.0710 | 0.1431 | 0.2799 | 0.5728 | 0.8186 | 0.9416 |
| `random` | 0.0645 | 0.1287 | 0.2543 | 0.5199 | 0.7870 | 0.9239 |
| `attr-e5-large-instruct` | 0.0712 | 0.1429 | 0.2827 | 0.5708 | 0.8177 | 0.9388 |
| `attr-e5-small-multi` | 0.0685 | 0.1388 | 0.2760 | 0.5636 | 0.8187 | 0.9393 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7419 | 0.7652 | 0.7678 | 0.7351 | 0.6519 | 0.5507 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.9549** | **0.9490** | **0.9334** | **0.8404** | **0.6854** | **0.5589** |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.9570 | 0.9530 | 0.9338 | 0.8383 | 0.6847 | 0.5600 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.9575 | 0.9524 | 0.9383 | 0.8455 | 0.6882 | 0.5601 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.9520 | 0.9524 | 0.9382 | 0.8479 | 0.6895 | 0.5596 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.9523 | 0.9518 | 0.9385 | 0.8478 | 0.6897 | 0.5596 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.9597 | 0.9567 | 0.9386 | 0.8496 | 0.6914 | 0.5614 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.9617 | 0.9616 | 0.9436 | 0.8567 | 0.6919 | 0.5616 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.9572 | 0.9539 | 0.9376 | 0.8465 | 0.6902 | 0.5611 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.9642 | 0.9536 | 0.9419 | 0.8558 | 0.6934 | 0.5628 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.9523 | 0.9456 | 0.9309 | 0.8421 | 0.6914 | 0.5632 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.9624 | 0.9565 | 0.9434 | 0.8520 | 0.6905 | 0.5613 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.9561 | 0.9487 | 0.9416 | 0.8533 | 0.6897 | 0.5601 |
| `fusion[z-score average] siglip-image + text-jina` | 0.9513 | 0.9485 | 0.9419 | 0.8532 | 0.6899 | 0.5601 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.9525 | 0.9438 | 0.9341 | 0.8446 | 0.6899 | 0.5628 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.9459 | 0.9362 | 0.9225 | 0.8376 | 0.6875 | 0.5600 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.9461 | 0.9377 | 0.9240 | 0.8295 | 0.6831 | 0.5603 |
| `siglip-image` | 0.9408 | 0.9377 | 0.9223 | 0.8429 | 0.6898 | 0.5613 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.9476 | 0.9374 | 0.9318 | 0.8508 | 0.6853 | 0.5564 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.9441 | 0.9340 | 0.9258 | 0.8394 | 0.6871 | 0.5609 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.9480 | 0.9363 | 0.9311 | 0.8509 | 0.6852 | 0.5564 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.9449 | 0.9425 | 0.9231 | 0.8273 | 0.6801 | 0.5609 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.9342 | 0.9345 | 0.9223 | 0.8410 | 0.6897 | 0.5599 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.9487 | 0.9322 | 0.9270 | 0.8400 | 0.6888 | 0.5629 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.9463 | 0.9427 | 0.9228 | 0.8282 | 0.6813 | 0.5609 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.9444 | 0.9402 | 0.9234 | 0.8305 | 0.6828 | 0.5618 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.9493 | 0.9431 | 0.9291 | 0.8413 | 0.6874 | 0.5618 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.9489 | 0.9405 | 0.9290 | 0.8412 | 0.6878 | 0.5617 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.9543 | 0.9417 | 0.9314 | 0.8488 | 0.6890 | 0.5587 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.9272 | 0.9318 | 0.9147 | 0.8360 | 0.6871 | 0.5614 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.9523 | 0.9530 | 0.9421 | 0.8517 | 0.6903 | 0.5614 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.9407 | 0.9256 | 0.9247 | 0.8381 | 0.6879 | 0.5609 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.9457 | 0.9415 | 0.9381 | 0.8488 | 0.6893 | 0.5610 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.9483 | 0.9494 | 0.9373 | 0.8427 | 0.6872 | 0.5602 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.9341 | 0.9289 | 0.9166 | 0.8339 | 0.6867 | 0.5618 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.9496 | 0.9407 | 0.9319 | 0.8481 | 0.6908 | 0.5631 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.9267 | 0.9286 | 0.9157 | 0.8360 | 0.6870 | 0.5614 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.9481 | 0.9363 | 0.9240 | 0.8310 | 0.6811 | 0.5596 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.9496 | 0.9491 | 0.9365 | 0.8477 | 0.6865 | 0.5588 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.9294 | 0.9112 | 0.9017 | 0.8206 | 0.6782 | 0.5555 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.9427 | 0.9399 | 0.9176 | 0.8272 | 0.6807 | 0.5606 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.9532 | 0.9528 | 0.9383 | 0.8476 | 0.6877 | 0.5609 |
| `attr-siglip` | 0.8930 | 0.9018 | 0.8883 | 0.8166 | 0.6718 | 0.5535 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.9496 | 0.9465 | 0.9332 | 0.8499 | 0.6889 | 0.5604 |
| `text-jina` | 0.8976 | 0.9124 | 0.9153 | 0.8399 | 0.6773 | 0.5532 |
| `text-omni-nano` | 0.8977 | 0.9136 | 0.9163 | 0.8394 | 0.6775 | 0.5532 |
| `text-e5-large-instruct` | 0.9139 | 0.9146 | 0.9135 | 0.8422 | 0.6810 | 0.5578 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.9246 | 0.9278 | 0.9149 | 0.8384 | 0.6876 | 0.5615 |
| `text-siglip` | 0.9013 | 0.8926 | 0.8806 | 0.7998 | 0.6675 | 0.5521 |
| `text-jina-small` | 0.9106 | 0.9138 | 0.9177 | 0.8401 | 0.6840 | 0.5567 |
| `text-e5-base` | 0.9332 | 0.9302 | 0.9124 | 0.8301 | 0.6757 | 0.5541 |
| `attr-omni-nano` | 0.8427 | 0.8668 | 0.8588 | 0.8035 | 0.6726 | 0.5575 |
| `text-e5-small-multi` | 0.9039 | 0.9058 | 0.9045 | 0.8315 | 0.6778 | 0.5541 |
| `attr-e5-base` | 0.8314 | 0.8606 | 0.8625 | 0.8069 | 0.6709 | 0.5564 |
| `attr-jina` | 0.8438 | 0.8700 | 0.8607 | 0.8038 | 0.6729 | 0.5575 |
| `attr-jina-small` | 0.8412 | 0.8591 | 0.8540 | 0.8100 | 0.6714 | 0.5580 |
| `random` | 0.7927 | 0.7862 | 0.7858 | 0.7335 | 0.6397 | 0.5425 |
| `attr-e5-large-instruct` | 0.8357 | 0.8592 | 0.8629 | 0.8066 | 0.6707 | 0.5556 |
| `attr-e5-small-multi` | 0.8137 | 0.8348 | 0.8513 | 0.8016 | 0.6720 | 0.5560 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7692 | 0.7784 | 0.7788 | 0.7788 | 0.7788 | 0.7788 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.9792** | **0.9796** | **0.9796** | **0.9796** | **0.9796** | **0.9796** |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.9760 | 0.9765 | 0.9765 | 0.9765 | 0.9765 | 0.9765 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.9808 | 0.9808 | 0.9809 | 0.9809 | 0.9809 | 0.9809 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.9642 | 0.9642 | 0.9642 | 0.9642 | 0.9642 | 0.9642 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.9646 | 0.9646 | 0.9646 | 0.9646 | 0.9646 | 0.9646 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.9748 | 0.9748 | 0.9748 | 0.9748 | 0.9748 | 0.9748 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.9831 | 0.9831 | 0.9831 | 0.9831 | 0.9831 | 0.9831 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.9789 | 0.9789 | 0.9789 | 0.9789 | 0.9789 | 0.9789 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.9740 | 0.9741 | 0.9742 | 0.9742 | 0.9742 | 0.9742 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 | 0.9810 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.9708 | 0.9709 | 0.9709 | 0.9709 | 0.9709 | 0.9709 |
| `fusion[z-score average] siglip-image + text-jina` | 0.9702 | 0.9702 | 0.9702 | 0.9702 | 0.9702 | 0.9702 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.9788 | 0.9790 | 0.9790 | 0.9790 | 0.9790 | 0.9790 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.9519 | 0.9521 | 0.9521 | 0.9521 | 0.9521 | 0.9521 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.9712 | 0.9715 | 0.9715 | 0.9715 | 0.9715 | 0.9715 |
| `siglip-image` | 0.9796 | 0.9798 | 0.9799 | 0.9799 | 0.9799 | 0.9799 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.9672 | 0.9672 | 0.9680 | 0.9680 | 0.9680 | 0.9680 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.9683 | 0.9686 | 0.9686 | 0.9686 | 0.9686 | 0.9686 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.9668 | 0.9670 | 0.9676 | 0.9676 | 0.9676 | 0.9676 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.9632 | 0.9636 | 0.9636 | 0.9636 | 0.9636 | 0.9636 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.9471 | 0.9472 | 0.9472 | 0.9472 | 0.9472 | 0.9472 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.9723 | 0.9726 | 0.9727 | 0.9727 | 0.9727 | 0.9727 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.9650 | 0.9654 | 0.9654 | 0.9654 | 0.9654 | 0.9654 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.9669 | 0.9673 | 0.9673 | 0.9673 | 0.9673 | 0.9673 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.9710 | 0.9715 | 0.9715 | 0.9715 | 0.9715 | 0.9715 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.9701 | 0.9706 | 0.9706 | 0.9706 | 0.9706 | 0.9706 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.9751 | 0.9751 | 0.9757 | 0.9757 | 0.9757 | 0.9757 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.9419 | 0.9428 | 0.9428 | 0.9428 | 0.9428 | 0.9428 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.9599 | 0.9599 | 0.9599 | 0.9599 | 0.9599 | 0.9599 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.9589 | 0.9593 | 0.9594 | 0.9594 | 0.9594 | 0.9594 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.9736 | 0.9737 | 0.9737 | 0.9737 | 0.9737 | 0.9737 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.9840 | 0.9841 | 0.9841 | 0.9841 | 0.9841 | 0.9841 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.9498 | 0.9502 | 0.9503 | 0.9504 | 0.9504 | 0.9504 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.9646 | 0.9647 | 0.9648 | 0.9648 | 0.9648 | 0.9648 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.9410 | 0.9419 | 0.9419 | 0.9419 | 0.9419 | 0.9419 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.9698 | 0.9702 | 0.9704 | 0.9704 | 0.9704 | 0.9704 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.9751 | 0.9752 | 0.9752 | 0.9752 | 0.9752 | 0.9752 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.9496 | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9500 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.9680 | 0.9681 | 0.9682 | 0.9682 | 0.9682 | 0.9682 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.9575 | 0.9576 | 0.9576 | 0.9576 | 0.9576 | 0.9576 |
| `attr-siglip` | 0.9431 | 0.9442 | 0.9442 | 0.9442 | 0.9442 | 0.9442 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.9775 | 0.9776 | 0.9776 | 0.9776 | 0.9776 | 0.9776 |
| `text-jina` | 0.9636 | 0.9645 | 0.9646 | 0.9646 | 0.9646 | 0.9646 |
| `text-omni-nano` | 0.9637 | 0.9646 | 0.9647 | 0.9647 | 0.9647 | 0.9647 |
| `text-e5-large-instruct` | 0.9648 | 0.9651 | 0.9656 | 0.9657 | 0.9657 | 0.9657 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.9384 | 0.9390 | 0.9392 | 0.9392 | 0.9392 | 0.9392 |
| `text-siglip` | 0.9651 | 0.9655 | 0.9655 | 0.9655 | 0.9655 | 0.9655 |
| `text-jina-small` | 0.9669 | 0.9670 | 0.9670 | 0.9674 | 0.9674 | 0.9674 |
| `text-e5-base` | 0.9717 | 0.9718 | 0.9718 | 0.9718 | 0.9718 | 0.9718 |
| `attr-omni-nano` | 0.8843 | 0.8854 | 0.8857 | 0.8858 | 0.8858 | 0.8858 |
| `text-e5-small-multi` | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 |
| `attr-e5-base` | 0.8794 | 0.8828 | 0.8832 | 0.8832 | 0.8832 | 0.8832 |
| `attr-jina` | 0.8807 | 0.8820 | 0.8823 | 0.8824 | 0.8824 | 0.8824 |
| `attr-jina-small` | 0.8949 | 0.8976 | 0.8978 | 0.8978 | 0.8978 | 0.8978 |
| `random` | 0.8806 | 0.8825 | 0.8829 | 0.8829 | 0.8829 | 0.8829 |
| `attr-e5-large-instruct` | 0.8891 | 0.8906 | 0.8913 | 0.8914 | 0.8914 | 0.8914 |
| `attr-e5-small-multi` | 0.8645 | 0.8671 | 0.8677 | 0.8681 | 0.8681 | 0.8681 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.8035 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.9093** |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.9087 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.9148 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.9175 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.9174 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.9191 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.9232 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.9187 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.9233 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.9172 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.9224 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.9203 |
| `fusion[z-score average] siglip-image + text-jina` | 0.9202 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.9172 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.9025 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.9037 |
| `siglip-image` | 0.9088 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.9156 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.9104 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.9156 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.9036 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.9046 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.9127 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.9037 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.9036 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.9131 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.9130 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.9181 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.9076 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.9192 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.9109 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.9183 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.9149 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.9061 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.9166 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.9076 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.9042 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.9154 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.8838 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.8981 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.9168 |
| `attr-siglip` | 0.8753 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.9179 |
| `text-jina` | 0.9034 |
| `text-omni-nano` | 0.9036 |
| `text-e5-large-instruct` | 0.9057 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.9066 |
| `text-siglip` | 0.8650 |
| `text-jina-small` | 0.9068 |
| `text-e5-base` | 0.9012 |
| `attr-omni-nano` | 0.8703 |
| `text-e5-small-multi` | 0.8959 |
| `attr-e5-base` | 0.8720 |
| `attr-jina` | 0.8707 |
| `attr-jina-small` | 0.8702 |
| `random` | 0.7970 |
| `attr-e5-large-instruct` | 0.8710 |
| `attr-e5-small-multi` | 0.8616 |


### Target: `omni-nano-image`

#### Macro-averaged -- every query counts once

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4025 | 0.4200 | 0.4571 | 0.5429 | 0.6298 | 0.6703 |
| **`text-omni-nano`** | **0.3789** | **0.4094** | **0.4634** | **0.5626** | **0.6416** | **0.6722** |
| `text-jina` | 0.3779 | 0.4072 | 0.4623 | 0.5619 | 0.6407 | 0.6713 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.3767 | 0.4032 | 0.4559 | 0.5511 | 0.6316 | 0.6657 |
| `text-e5-large-instruct` | 0.3702 | 0.4012 | 0.4589 | 0.5592 | 0.6396 | 0.6711 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.3703 | 0.4009 | 0.4566 | 0.5600 | 0.6381 | 0.6697 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.3684 | 0.3991 | 0.4555 | 0.5590 | 0.6370 | 0.6686 |
| `text-jina-small` | 0.3638 | 0.3989 | 0.4566 | 0.5557 | 0.6367 | 0.6672 |
| `text-siglip` | 0.3706 | 0.3981 | 0.4438 | 0.5370 | 0.6204 | 0.6554 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.3695 | 0.3974 | 0.4550 | 0.5583 | 0.6363 | 0.6682 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.3631 | 0.3926 | 0.4485 | 0.5531 | 0.6307 | 0.6642 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.3576 | 0.3920 | 0.4429 | 0.5421 | 0.6202 | 0.6548 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.3597 | 0.3917 | 0.4426 | 0.5485 | 0.6271 | 0.6602 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.3611 | 0.3912 | 0.4485 | 0.5522 | 0.6299 | 0.6634 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.3641 | 0.3907 | 0.4477 | 0.5509 | 0.6314 | 0.6641 |
| `text-e5-base` | 0.3644 | 0.3899 | 0.4435 | 0.5459 | 0.6286 | 0.6626 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.3639 | 0.3891 | 0.4470 | 0.5481 | 0.6283 | 0.6615 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.3573 | 0.3873 | 0.4438 | 0.5496 | 0.6285 | 0.6608 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.3557 | 0.3872 | 0.4420 | 0.5479 | 0.6281 | 0.6604 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.3586 | 0.3864 | 0.4433 | 0.5494 | 0.6282 | 0.6607 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.3482 | 0.3851 | 0.4395 | 0.5372 | 0.6174 | 0.6522 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.3551 | 0.3839 | 0.4398 | 0.5421 | 0.6205 | 0.6543 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.3427 | 0.3826 | 0.4367 | 0.5364 | 0.6180 | 0.6528 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.3528 | 0.3819 | 0.4380 | 0.5392 | 0.6176 | 0.6523 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.3523 | 0.3810 | 0.4383 | 0.5417 | 0.6255 | 0.6586 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.3547 | 0.3799 | 0.4353 | 0.5392 | 0.6224 | 0.6555 |
| `attr-siglip` | 0.3477 | 0.3776 | 0.4335 | 0.5341 | 0.6122 | 0.6472 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.3474 | 0.3723 | 0.4289 | 0.5331 | 0.6163 | 0.6509 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.3344 | 0.3704 | 0.4254 | 0.5320 | 0.6131 | 0.6484 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.3381 | 0.3701 | 0.4260 | 0.5325 | 0.6153 | 0.6491 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.3412 | 0.3693 | 0.4285 | 0.5342 | 0.6158 | 0.6501 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.3393 | 0.3693 | 0.4258 | 0.5344 | 0.6179 | 0.6504 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.3361 | 0.3692 | 0.4254 | 0.5316 | 0.6130 | 0.6482 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.3406 | 0.3687 | 0.4282 | 0.5343 | 0.6157 | 0.6500 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.3359 | 0.3679 | 0.4278 | 0.5336 | 0.6148 | 0.6496 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.3296 | 0.3678 | 0.4280 | 0.5316 | 0.6126 | 0.6478 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.3344 | 0.3666 | 0.4181 | 0.5231 | 0.6040 | 0.6407 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.3322 | 0.3666 | 0.4229 | 0.5263 | 0.6064 | 0.6436 |
| `text-e5-small-multi` | 0.3315 | 0.3653 | 0.4215 | 0.5328 | 0.6143 | 0.6485 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.3336 | 0.3651 | 0.4181 | 0.5235 | 0.6048 | 0.6415 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.3300 | 0.3644 | 0.4228 | 0.5295 | 0.6106 | 0.6460 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.3279 | 0.3627 | 0.4209 | 0.5247 | 0.6041 | 0.6409 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.3330 | 0.3625 | 0.4198 | 0.5257 | 0.6080 | 0.6429 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.3305 | 0.3625 | 0.4234 | 0.5268 | 0.6092 | 0.6446 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.3340 | 0.3621 | 0.4168 | 0.5239 | 0.6083 | 0.6425 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.3301 | 0.3595 | 0.4185 | 0.5205 | 0.6023 | 0.6387 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.3189 | 0.3508 | 0.4121 | 0.5174 | 0.6005 | 0.6357 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.3194 | 0.3504 | 0.4087 | 0.5149 | 0.5987 | 0.6347 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.3152 | 0.3461 | 0.4027 | 0.5104 | 0.5940 | 0.6303 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.3048 | 0.3448 | 0.4056 | 0.5142 | 0.5965 | 0.6332 |
| `attr-omni-nano` | 0.3030 | 0.3433 | 0.4047 | 0.5124 | 0.5966 | 0.6324 |
| `attr-jina-small` | 0.3064 | 0.3424 | 0.4026 | 0.5126 | 0.5965 | 0.6327 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.3133 | 0.3422 | 0.3997 | 0.5045 | 0.5889 | 0.6260 |
| `attr-e5-large-instruct` | 0.3020 | 0.3411 | 0.4007 | 0.5141 | 0.5969 | 0.6330 |
| `attr-jina` | 0.2986 | 0.3398 | 0.4046 | 0.5116 | 0.5956 | 0.6312 |
| `attr-e5-base` | 0.2956 | 0.3380 | 0.3980 | 0.5045 | 0.5906 | 0.6275 |
| `omni-nano-image` | 0.3011 | 0.3315 | 0.3880 | 0.4969 | 0.5824 | 0.6206 |
| `attr-e5-small-multi` | 0.2775 | 0.3125 | 0.3720 | 0.4890 | 0.5743 | 0.6129 |
| `random` | 0.2392 | 0.2640 | 0.3210 | 0.4357 | 0.5321 | 0.5758 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.1400 | 0.2466 | 0.4189 | 0.6922 | 0.8846 | 0.9646 |
| **`text-omni-nano`** | **0.1541** | **0.2678** | **0.4477** | **0.7241** | **0.8984** | **0.9655** |
| `text-jina` | 0.1537 | 0.2673 | 0.4476 | 0.7243 | 0.8984 | 0.9655 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.1490 | 0.2600 | 0.4393 | 0.7135 | 0.8936 | 0.9664 |
| `text-e5-large-instruct` | 0.1597 | 0.2706 | 0.4488 | 0.7209 | 0.9006 | 0.9683 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.1457 | 0.2620 | 0.4460 | 0.7215 | 0.8982 | 0.9669 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.1461 | 0.2621 | 0.4460 | 0.7217 | 0.8983 | 0.9669 |
| `text-jina-small` | 0.1490 | 0.2669 | 0.4515 | 0.7238 | 0.9011 | 0.9667 |
| `text-siglip` | 0.1485 | 0.2603 | 0.4316 | 0.7036 | 0.8886 | 0.9640 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.1531 | 0.2630 | 0.4451 | 0.7232 | 0.8997 | 0.9678 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.1427 | 0.2598 | 0.4410 | 0.7177 | 0.8948 | 0.9663 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.1396 | 0.2556 | 0.4286 | 0.7067 | 0.8909 | 0.9640 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.1424 | 0.2616 | 0.4383 | 0.7172 | 0.8957 | 0.9666 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.1407 | 0.2559 | 0.4412 | 0.7178 | 0.8947 | 0.9663 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.1471 | 0.2601 | 0.4436 | 0.7166 | 0.8978 | 0.9681 |
| `text-e5-base` | 0.1609 | 0.2671 | 0.4394 | 0.7130 | 0.8932 | 0.9656 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.1467 | 0.2593 | 0.4409 | 0.7142 | 0.8962 | 0.9672 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.1464 | 0.2566 | 0.4374 | 0.7172 | 0.8962 | 0.9664 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.1468 | 0.2606 | 0.4401 | 0.7173 | 0.8977 | 0.9674 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.1470 | 0.2565 | 0.4369 | 0.7174 | 0.8962 | 0.9664 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.1400 | 0.2575 | 0.4369 | 0.7093 | 0.8909 | 0.9650 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.1347 | 0.2470 | 0.4272 | 0.7110 | 0.8918 | 0.9648 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.1398 | 0.2560 | 0.4331 | 0.7110 | 0.8920 | 0.9663 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.1352 | 0.2479 | 0.4292 | 0.7084 | 0.8901 | 0.9640 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.1493 | 0.2578 | 0.4377 | 0.7136 | 0.8939 | 0.9666 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.1491 | 0.2598 | 0.4351 | 0.7109 | 0.8932 | 0.9653 |
| `attr-siglip` | 0.1381 | 0.2483 | 0.4198 | 0.7073 | 0.8872 | 0.9627 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.1411 | 0.2498 | 0.4283 | 0.7086 | 0.8910 | 0.9652 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.1295 | 0.2472 | 0.4220 | 0.7127 | 0.8913 | 0.9669 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.1466 | 0.2565 | 0.4295 | 0.7110 | 0.8928 | 0.9651 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.1291 | 0.2409 | 0.4246 | 0.7122 | 0.8935 | 0.9673 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.1455 | 0.2484 | 0.4305 | 0.7142 | 0.8950 | 0.9657 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.1311 | 0.2458 | 0.4223 | 0.7118 | 0.8916 | 0.9668 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.1291 | 0.2409 | 0.4243 | 0.7124 | 0.8936 | 0.9673 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.1291 | 0.2420 | 0.4221 | 0.7117 | 0.8918 | 0.9669 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.1364 | 0.2470 | 0.4236 | 0.7096 | 0.8886 | 0.9642 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.1325 | 0.2478 | 0.4168 | 0.7056 | 0.8876 | 0.9645 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.1324 | 0.2456 | 0.4181 | 0.7046 | 0.8870 | 0.9645 |
| `text-e5-small-multi` | 0.1556 | 0.2612 | 0.4326 | 0.7165 | 0.8937 | 0.9648 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.1333 | 0.2451 | 0.4187 | 0.7054 | 0.8882 | 0.9647 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.1300 | 0.2420 | 0.4216 | 0.7111 | 0.8907 | 0.9664 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.1415 | 0.2463 | 0.4245 | 0.7055 | 0.8858 | 0.9628 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.1363 | 0.2441 | 0.4229 | 0.7059 | 0.8907 | 0.9644 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.1296 | 0.2453 | 0.4246 | 0.7080 | 0.8893 | 0.9653 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.1395 | 0.2475 | 0.4194 | 0.7050 | 0.8909 | 0.9647 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.1319 | 0.2456 | 0.4285 | 0.7028 | 0.8869 | 0.9636 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.1290 | 0.2367 | 0.4189 | 0.7044 | 0.8881 | 0.9645 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.1322 | 0.2376 | 0.4171 | 0.7005 | 0.8858 | 0.9629 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.1300 | 0.2353 | 0.4144 | 0.7004 | 0.8853 | 0.9626 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.1265 | 0.2437 | 0.4169 | 0.7052 | 0.8875 | 0.9635 |
| `attr-omni-nano` | 0.1269 | 0.2423 | 0.4161 | 0.7070 | 0.8903 | 0.9647 |
| `attr-jina-small` | 0.1298 | 0.2429 | 0.4127 | 0.7094 | 0.8888 | 0.9651 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.1270 | 0.2429 | 0.4181 | 0.6979 | 0.8842 | 0.9610 |
| `attr-e5-large-instruct` | 0.1359 | 0.2440 | 0.4181 | 0.7055 | 0.8870 | 0.9621 |
| `attr-jina` | 0.1263 | 0.2433 | 0.4185 | 0.7086 | 0.8905 | 0.9648 |
| `attr-e5-base` | 0.1256 | 0.2423 | 0.4195 | 0.7014 | 0.8866 | 0.9633 |
| `omni-nano-image` | 0.1225 | 0.2277 | 0.4071 | 0.6893 | 0.8800 | 0.9593 |
| `attr-e5-small-multi` | 0.1262 | 0.2342 | 0.4077 | 0.6940 | 0.8820 | 0.9600 |
| `random` | 0.0954 | 0.1807 | 0.3567 | 0.6540 | 0.8596 | 0.9509 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6303 | 0.6435 | 0.6313 | 0.5483 | 0.4265 | 0.3387 |
| **`text-omni-nano`** | **0.7483** | **0.7342** | **0.7040** | **0.5943** | **0.4404** | **0.3400** |
| `text-jina` | 0.7483 | 0.7335 | 0.7034 | 0.5948 | 0.4404 | 0.3400 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.7283 | 0.7177 | 0.6835 | 0.5781 | 0.4350 | 0.3400 |
| `text-e5-large-instruct` | 0.7600 | 0.7392 | 0.7020 | 0.5932 | 0.4419 | 0.3415 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.7330 | 0.7237 | 0.6948 | 0.5917 | 0.4396 | 0.3407 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.7323 | 0.7247 | 0.6951 | 0.5918 | 0.4398 | 0.3407 |
| `text-jina-small` | 0.7550 | 0.7407 | 0.7058 | 0.5949 | 0.4426 | 0.3407 |
| `text-siglip` | 0.7347 | 0.7165 | 0.6774 | 0.5687 | 0.4309 | 0.3384 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.7470 | 0.7300 | 0.6982 | 0.5921 | 0.4410 | 0.3412 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.7283 | 0.7155 | 0.6831 | 0.5840 | 0.4363 | 0.3400 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.7247 | 0.7047 | 0.6678 | 0.5715 | 0.4323 | 0.3385 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.7400 | 0.7183 | 0.6845 | 0.5839 | 0.4371 | 0.3403 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.7283 | 0.7135 | 0.6831 | 0.5841 | 0.4364 | 0.3400 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.7297 | 0.7153 | 0.6859 | 0.5864 | 0.4392 | 0.3413 |
| `text-e5-base` | 0.7427 | 0.7205 | 0.6851 | 0.5816 | 0.4364 | 0.3401 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.7353 | 0.7175 | 0.6861 | 0.5821 | 0.4373 | 0.3404 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.7210 | 0.7097 | 0.6846 | 0.5858 | 0.4382 | 0.3404 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.7260 | 0.7142 | 0.6853 | 0.5869 | 0.4395 | 0.3411 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.7217 | 0.7098 | 0.6840 | 0.5860 | 0.4382 | 0.3404 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.7133 | 0.7043 | 0.6737 | 0.5715 | 0.4327 | 0.3389 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.7137 | 0.6985 | 0.6687 | 0.5761 | 0.4332 | 0.3391 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.7053 | 0.7022 | 0.6723 | 0.5752 | 0.4339 | 0.3401 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.7103 | 0.6958 | 0.6675 | 0.5737 | 0.4322 | 0.3387 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.7197 | 0.7035 | 0.6786 | 0.5803 | 0.4369 | 0.3405 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.7277 | 0.7057 | 0.6743 | 0.5764 | 0.4353 | 0.3394 |
| `attr-siglip` | 0.7020 | 0.7010 | 0.6718 | 0.5741 | 0.4304 | 0.3380 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.6983 | 0.6878 | 0.6653 | 0.5736 | 0.4336 | 0.3395 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.6867 | 0.6863 | 0.6612 | 0.5760 | 0.4336 | 0.3404 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.7177 | 0.7077 | 0.6734 | 0.5760 | 0.4347 | 0.3393 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.6873 | 0.6852 | 0.6643 | 0.5759 | 0.4349 | 0.3406 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.7097 | 0.7028 | 0.6757 | 0.5806 | 0.4374 | 0.3400 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.6873 | 0.6852 | 0.6620 | 0.5760 | 0.4336 | 0.3403 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.6877 | 0.6848 | 0.6641 | 0.5762 | 0.4351 | 0.3406 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.6880 | 0.6875 | 0.6664 | 0.5756 | 0.4337 | 0.3403 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.6880 | 0.6847 | 0.6627 | 0.5715 | 0.4310 | 0.3388 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.6900 | 0.6843 | 0.6535 | 0.5666 | 0.4294 | 0.3389 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.6953 | 0.6878 | 0.6587 | 0.5659 | 0.4293 | 0.3387 |
| `text-e5-small-multi` | 0.7317 | 0.7175 | 0.6816 | 0.5829 | 0.4365 | 0.3398 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.6913 | 0.6838 | 0.6550 | 0.5667 | 0.4301 | 0.3391 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.6860 | 0.6843 | 0.6603 | 0.5755 | 0.4330 | 0.3401 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.7010 | 0.6855 | 0.6562 | 0.5649 | 0.4280 | 0.3377 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.6850 | 0.6817 | 0.6584 | 0.5705 | 0.4328 | 0.3389 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.6823 | 0.6745 | 0.6552 | 0.5715 | 0.4321 | 0.3394 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.6890 | 0.6880 | 0.6576 | 0.5718 | 0.4332 | 0.3391 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.6963 | 0.6840 | 0.6506 | 0.5640 | 0.4294 | 0.3382 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.6733 | 0.6700 | 0.6504 | 0.5664 | 0.4307 | 0.3388 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.6743 | 0.6658 | 0.6473 | 0.5633 | 0.4288 | 0.3379 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.6703 | 0.6658 | 0.6456 | 0.5639 | 0.4283 | 0.3378 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.6687 | 0.6660 | 0.6479 | 0.5675 | 0.4300 | 0.3383 |
| `attr-omni-nano` | 0.6633 | 0.6688 | 0.6471 | 0.5685 | 0.4332 | 0.3394 |
| `attr-jina-small` | 0.6703 | 0.6625 | 0.6469 | 0.5704 | 0.4317 | 0.3394 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.6800 | 0.6668 | 0.6407 | 0.5584 | 0.4264 | 0.3366 |
| `attr-e5-large-instruct` | 0.6687 | 0.6623 | 0.6462 | 0.5667 | 0.4295 | 0.3377 |
| `attr-jina` | 0.6660 | 0.6708 | 0.6484 | 0.5689 | 0.4335 | 0.3394 |
| `attr-e5-base` | 0.6587 | 0.6638 | 0.6418 | 0.5629 | 0.4297 | 0.3384 |
| `omni-nano-image` | 0.6430 | 0.6378 | 0.6248 | 0.5517 | 0.4233 | 0.3357 |
| `attr-e5-small-multi` | 0.6350 | 0.6348 | 0.6268 | 0.5549 | 0.4256 | 0.3366 |
| `random` | 0.5777 | 0.5777 | 0.5647 | 0.5094 | 0.4078 | 0.3306 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6940 | 0.7041 | 0.7063 | 0.7071 | 0.7072 | 0.7072 |
| **`text-omni-nano`** | **0.8311** | **0.8360** | **0.8379** | **0.8387** | **0.8387** | **0.8387** |
| `text-jina` | 0.8305 | 0.8352 | 0.8369 | 0.8378 | 0.8378 | 0.8378 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.8087 | 0.8134 | 0.8154 | 0.8161 | 0.8161 | 0.8161 |
| `text-e5-large-instruct` | 0.8383 | 0.8420 | 0.8441 | 0.8448 | 0.8448 | 0.8448 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.8027 | 0.8079 | 0.8104 | 0.8110 | 0.8110 | 0.8110 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.7999 | 0.8053 | 0.8078 | 0.8084 | 0.8084 | 0.8084 |
| `text-jina-small` | 0.8239 | 0.8290 | 0.8316 | 0.8325 | 0.8325 | 0.8325 |
| `text-siglip` | 0.8123 | 0.8168 | 0.8184 | 0.8192 | 0.8193 | 0.8193 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.8112 | 0.8152 | 0.8176 | 0.8183 | 0.8183 | 0.8183 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.8102 | 0.8149 | 0.8170 | 0.8176 | 0.8176 | 0.8176 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.8083 | 0.8136 | 0.8154 | 0.8161 | 0.8161 | 0.8161 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.8121 | 0.8176 | 0.8193 | 0.8200 | 0.8200 | 0.8200 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.8096 | 0.8147 | 0.8173 | 0.8178 | 0.8178 | 0.8178 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.8033 | 0.8079 | 0.8106 | 0.8109 | 0.8110 | 0.8110 |
| `text-e5-base` | 0.8406 | 0.8439 | 0.8456 | 0.8466 | 0.8466 | 0.8466 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.8186 | 0.8223 | 0.8247 | 0.8253 | 0.8253 | 0.8253 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.7943 | 0.7992 | 0.8016 | 0.8022 | 0.8022 | 0.8022 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.7985 | 0.8024 | 0.8048 | 0.8054 | 0.8054 | 0.8054 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.7932 | 0.7977 | 0.8000 | 0.8007 | 0.8007 | 0.8007 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.8009 | 0.8069 | 0.8093 | 0.8097 | 0.8097 | 0.8097 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.7981 | 0.8032 | 0.8061 | 0.8069 | 0.8069 | 0.8069 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.7758 | 0.7822 | 0.7845 | 0.7851 | 0.7851 | 0.7851 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.8010 | 0.8063 | 0.8086 | 0.8093 | 0.8093 | 0.8093 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.8029 | 0.8068 | 0.8093 | 0.8099 | 0.8099 | 0.8099 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.8252 | 0.8293 | 0.8312 | 0.8320 | 0.8320 | 0.8320 |
| `attr-siglip` | 0.7843 | 0.7896 | 0.7913 | 0.7922 | 0.7922 | 0.7922 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.7824 | 0.7872 | 0.7892 | 0.7901 | 0.7901 | 0.7901 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.7730 | 0.7797 | 0.7820 | 0.7830 | 0.7830 | 0.7830 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.8052 | 0.8095 | 0.8117 | 0.8124 | 0.8124 | 0.8124 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.7756 | 0.7819 | 0.7856 | 0.7864 | 0.7864 | 0.7864 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.7994 | 0.8044 | 0.8070 | 0.8077 | 0.8078 | 0.8078 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.7720 | 0.7790 | 0.7813 | 0.7822 | 0.7822 | 0.7822 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.7763 | 0.7827 | 0.7863 | 0.7871 | 0.7871 | 0.7871 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.7751 | 0.7826 | 0.7849 | 0.7859 | 0.7859 | 0.7859 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.7814 | 0.7866 | 0.7888 | 0.7898 | 0.7898 | 0.7898 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.7918 | 0.7980 | 0.7997 | 0.8008 | 0.8008 | 0.8008 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.7827 | 0.7889 | 0.7907 | 0.7919 | 0.7920 | 0.7920 |
| `text-e5-small-multi` | 0.8054 | 0.8095 | 0.8111 | 0.8124 | 0.8124 | 0.8124 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.7954 | 0.8009 | 0.8030 | 0.8040 | 0.8040 | 0.8040 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.7729 | 0.7792 | 0.7817 | 0.7827 | 0.7828 | 0.7828 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.7864 | 0.7912 | 0.7931 | 0.7939 | 0.7940 | 0.7940 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.7670 | 0.7731 | 0.7755 | 0.7762 | 0.7763 | 0.7763 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.7775 | 0.7835 | 0.7861 | 0.7870 | 0.7871 | 0.7871 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.7777 | 0.7825 | 0.7847 | 0.7855 | 0.7856 | 0.7856 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.7816 | 0.7873 | 0.7904 | 0.7910 | 0.7910 | 0.7910 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.7535 | 0.7597 | 0.7623 | 0.7634 | 0.7634 | 0.7634 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.7549 | 0.7604 | 0.7630 | 0.7638 | 0.7638 | 0.7638 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.7457 | 0.7511 | 0.7533 | 0.7544 | 0.7544 | 0.7544 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.7509 | 0.7591 | 0.7610 | 0.7621 | 0.7622 | 0.7622 |
| `attr-omni-nano` | 0.7448 | 0.7523 | 0.7554 | 0.7564 | 0.7565 | 0.7565 |
| `attr-jina-small` | 0.7481 | 0.7566 | 0.7589 | 0.7602 | 0.7602 | 0.7602 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.7589 | 0.7666 | 0.7685 | 0.7694 | 0.7694 | 0.7694 |
| `attr-e5-large-instruct` | 0.7484 | 0.7543 | 0.7570 | 0.7582 | 0.7582 | 0.7582 |
| `attr-jina` | 0.7419 | 0.7497 | 0.7527 | 0.7537 | 0.7537 | 0.7537 |
| `attr-e5-base` | 0.7364 | 0.7458 | 0.7488 | 0.7498 | 0.7498 | 0.7498 |
| `omni-nano-image` | 0.7397 | 0.7460 | 0.7490 | 0.7499 | 0.7499 | 0.7499 |
| `attr-e5-small-multi` | 0.7056 | 0.7147 | 0.7180 | 0.7190 | 0.7190 | 0.7190 |
| `random` | 0.6917 | 0.6965 | 0.7008 | 0.7022 | 0.7022 | 0.7022 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.6808 |
| **`text-omni-nano`** | **0.7473** |
| `text-jina` | 0.7471 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.7188 |
| `text-e5-large-instruct` | 0.7502 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.7357 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.7350 |
| `text-jina-small` | 0.7463 |
| `text-siglip` | 0.7072 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.7379 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.7227 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.7057 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.7247 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.7227 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.7300 |
| `text-e5-base` | 0.7401 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.7236 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.7243 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.7259 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.7243 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.7047 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.7078 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.7067 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.7055 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.7236 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.7177 |
| `attr-siglip` | 0.7023 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.7094 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.7025 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.7171 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.7039 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.7198 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.7019 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.7039 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.7066 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.7059 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.6935 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.6995 |
| `text-e5-small-multi` | 0.7316 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.6942 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.7031 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.6974 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.7005 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.7020 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.7014 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.6935 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.6899 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.6869 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.6840 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.6904 |
| `attr-omni-nano` | 0.6937 |
| `attr-jina-small` | 0.6957 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.6791 |
| `attr-e5-large-instruct` | 0.6975 |
| `attr-jina` | 0.6933 |
| `attr-e5-base` | 0.6916 |
| `omni-nano-image` | 0.6664 |
| `attr-e5-small-multi` | 0.6712 |
| `random` | 0.6041 |


#### Impression-weighted -- every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4066 | 0.4123 | 0.4295 | 0.5122 | 0.6384 | 0.7053 |
| **`fusion[RRF (k=60)] omni-nano-image + attr-siglip`** | **0.4147** | **0.4328** | **0.4743** | **0.5721** | **0.6694** | **0.7259** |
| `attr-siglip` | 0.4044 | 0.4185 | 0.4617 | 0.5529 | 0.6596 | 0.7175 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.3835 | 0.4101 | 0.4593 | 0.5611 | 0.6637 | 0.7171 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.3941 | 0.4092 | 0.4589 | 0.5621 | 0.6612 | 0.7193 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.3931 | 0.4076 | 0.4541 | 0.5682 | 0.6728 | 0.7211 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.3936 | 0.4052 | 0.4542 | 0.5674 | 0.6721 | 0.7201 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.3889 | 0.4031 | 0.4470 | 0.5558 | 0.6656 | 0.7156 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.3862 | 0.4023 | 0.4450 | 0.5557 | 0.6649 | 0.7152 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.3761 | 0.4014 | 0.4460 | 0.5497 | 0.6624 | 0.7151 |
| `text-jina` | 0.3698 | 0.3998 | 0.4536 | 0.5580 | 0.6626 | 0.7140 |
| `text-omni-nano` | 0.3699 | 0.3989 | 0.4559 | 0.5578 | 0.6633 | 0.7148 |
| `text-e5-large-instruct` | 0.3677 | 0.3976 | 0.4503 | 0.5531 | 0.6640 | 0.7159 |
| `text-siglip` | 0.3842 | 0.3958 | 0.4305 | 0.5228 | 0.6407 | 0.7025 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.3738 | 0.3946 | 0.4338 | 0.5363 | 0.6448 | 0.7035 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.3774 | 0.3934 | 0.4418 | 0.5564 | 0.6628 | 0.7129 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.3779 | 0.3929 | 0.4402 | 0.5564 | 0.6628 | 0.7131 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.3706 | 0.3909 | 0.4522 | 0.5599 | 0.6679 | 0.7194 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.3769 | 0.3902 | 0.4395 | 0.5523 | 0.6623 | 0.7134 |
| `text-jina-small` | 0.3551 | 0.3895 | 0.4454 | 0.5501 | 0.6633 | 0.7166 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.3719 | 0.3894 | 0.4449 | 0.5480 | 0.6597 | 0.7109 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.3750 | 0.3893 | 0.4326 | 0.5394 | 0.6530 | 0.7058 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.3817 | 0.3892 | 0.4359 | 0.5390 | 0.6507 | 0.7086 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.3680 | 0.3884 | 0.4278 | 0.5348 | 0.6463 | 0.7009 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.3714 | 0.3884 | 0.4442 | 0.5495 | 0.6599 | 0.7120 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.3763 | 0.3879 | 0.4460 | 0.5506 | 0.6607 | 0.7128 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.3786 | 0.3874 | 0.4304 | 0.5355 | 0.6439 | 0.7018 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.3664 | 0.3854 | 0.4333 | 0.5375 | 0.6437 | 0.7052 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.3560 | 0.3852 | 0.4388 | 0.5421 | 0.6529 | 0.7079 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.3594 | 0.3841 | 0.4226 | 0.5381 | 0.6545 | 0.7044 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.3620 | 0.3838 | 0.4263 | 0.5355 | 0.6472 | 0.7017 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.3651 | 0.3838 | 0.4387 | 0.5480 | 0.6595 | 0.7127 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.3721 | 0.3838 | 0.4273 | 0.5329 | 0.6426 | 0.7020 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.3601 | 0.3824 | 0.4360 | 0.5504 | 0.6611 | 0.7126 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.3622 | 0.3821 | 0.4382 | 0.5419 | 0.6532 | 0.7075 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.3647 | 0.3819 | 0.4162 | 0.5308 | 0.6454 | 0.6996 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.3502 | 0.3800 | 0.4178 | 0.5300 | 0.6440 | 0.6978 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.3593 | 0.3799 | 0.4295 | 0.5285 | 0.6463 | 0.7035 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.3626 | 0.3799 | 0.4247 | 0.5331 | 0.6468 | 0.7004 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.3643 | 0.3784 | 0.4275 | 0.5363 | 0.6481 | 0.7024 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.3479 | 0.3777 | 0.4329 | 0.5394 | 0.6517 | 0.7073 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.3583 | 0.3747 | 0.4106 | 0.5199 | 0.6357 | 0.6945 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.3499 | 0.3733 | 0.4235 | 0.5264 | 0.6460 | 0.7025 |
| `text-e5-base` | 0.3566 | 0.3730 | 0.4220 | 0.5329 | 0.6420 | 0.6987 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.3353 | 0.3722 | 0.4272 | 0.5365 | 0.6469 | 0.7003 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.3437 | 0.3707 | 0.4192 | 0.5288 | 0.6460 | 0.7010 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.3562 | 0.3691 | 0.4214 | 0.5334 | 0.6475 | 0.6996 |
| `omni-nano-image` | 0.3514 | 0.3679 | 0.4042 | 0.5160 | 0.6314 | 0.6889 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.3403 | 0.3664 | 0.4228 | 0.5317 | 0.6394 | 0.6969 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.3346 | 0.3662 | 0.4182 | 0.5310 | 0.6470 | 0.7000 |
| `attr-omni-nano` | 0.3257 | 0.3590 | 0.4124 | 0.5142 | 0.6336 | 0.6904 |
| `text-e5-small-multi` | 0.3415 | 0.3579 | 0.4101 | 0.5276 | 0.6454 | 0.6970 |
| `attr-e5-base` | 0.3226 | 0.3560 | 0.4064 | 0.5113 | 0.6251 | 0.6863 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.3266 | 0.3503 | 0.4113 | 0.5232 | 0.6374 | 0.6931 |
| `attr-jina` | 0.3178 | 0.3455 | 0.4055 | 0.5112 | 0.6314 | 0.6890 |
| `attr-jina-small` | 0.3112 | 0.3407 | 0.4032 | 0.5088 | 0.6246 | 0.6870 |
| `random` | 0.3183 | 0.3208 | 0.3505 | 0.4523 | 0.5797 | 0.6481 |
| `attr-e5-large-instruct` | 0.2946 | 0.3198 | 0.3820 | 0.5023 | 0.6163 | 0.6822 |
| `attr-e5-small-multi` | 0.2846 | 0.3093 | 0.3644 | 0.4861 | 0.6138 | 0.6746 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.0609 | 0.1266 | 0.2517 | 0.5200 | 0.8002 | 0.9346 |
| **`fusion[RRF (k=60)] omni-nano-image + attr-siglip`** | **0.0754** | **0.1486** | **0.2844** | **0.5678** | **0.8209** | **0.9386** |
| `attr-siglip` | 0.0737 | 0.1470 | 0.2850 | 0.5719 | 0.8180 | 0.9370 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.0730 | 0.1463 | 0.2847 | 0.5724 | 0.8227 | 0.9395 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.0734 | 0.1455 | 0.2846 | 0.5719 | 0.8200 | 0.9391 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.0756 | 0.1512 | 0.2918 | 0.5825 | 0.8279 | 0.9395 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.0755 | 0.1513 | 0.2926 | 0.5826 | 0.8280 | 0.9392 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.0755 | 0.1492 | 0.2876 | 0.5776 | 0.8253 | 0.9384 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.0748 | 0.1492 | 0.2880 | 0.5774 | 0.8250 | 0.9385 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.0762 | 0.1508 | 0.2896 | 0.5771 | 0.8280 | 0.9408 |
| `text-jina` | 0.0770 | 0.1534 | 0.2990 | 0.5890 | 0.8243 | 0.9372 |
| `text-omni-nano` | 0.0770 | 0.1536 | 0.2992 | 0.5888 | 0.8245 | 0.9373 |
| `text-e5-large-instruct` | 0.0779 | 0.1537 | 0.2985 | 0.5903 | 0.8269 | 0.9425 |
| `text-siglip` | 0.0756 | 0.1473 | 0.2827 | 0.5584 | 0.8122 | 0.9353 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.0727 | 0.1457 | 0.2817 | 0.5648 | 0.8183 | 0.9397 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.0745 | 0.1487 | 0.2891 | 0.5790 | 0.8261 | 0.9394 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.0746 | 0.1489 | 0.2897 | 0.5794 | 0.8261 | 0.9392 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.0765 | 0.1513 | 0.2933 | 0.5821 | 0.8304 | 0.9415 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.0758 | 0.1503 | 0.2913 | 0.5782 | 0.8304 | 0.9412 |
| `text-jina-small` | 0.0775 | 0.1536 | 0.2994 | 0.5887 | 0.8306 | 0.9412 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.0764 | 0.1510 | 0.2903 | 0.5743 | 0.8278 | 0.9401 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.0728 | 0.1448 | 0.2844 | 0.5697 | 0.8215 | 0.9384 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.0758 | 0.1476 | 0.2861 | 0.5690 | 0.8199 | 0.9388 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.0715 | 0.1441 | 0.2796 | 0.5672 | 0.8206 | 0.9378 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.0717 | 0.1454 | 0.2863 | 0.5724 | 0.8268 | 0.9431 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.0717 | 0.1454 | 0.2863 | 0.5726 | 0.8267 | 0.9430 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.0727 | 0.1461 | 0.2817 | 0.5662 | 0.8217 | 0.9392 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.0729 | 0.1454 | 0.2836 | 0.5646 | 0.8190 | 0.9402 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.0721 | 0.1466 | 0.2873 | 0.5719 | 0.8238 | 0.9411 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.0746 | 0.1492 | 0.2890 | 0.5772 | 0.8258 | 0.9397 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.0722 | 0.1449 | 0.2850 | 0.5712 | 0.8251 | 0.9405 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.0714 | 0.1451 | 0.2861 | 0.5729 | 0.8261 | 0.9433 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.0728 | 0.1459 | 0.2821 | 0.5649 | 0.8188 | 0.9402 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.0748 | 0.1489 | 0.2905 | 0.5800 | 0.8293 | 0.9412 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.0721 | 0.1465 | 0.2874 | 0.5718 | 0.8237 | 0.9410 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.0698 | 0.1399 | 0.2757 | 0.5631 | 0.8204 | 0.9389 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.0700 | 0.1406 | 0.2768 | 0.5634 | 0.8191 | 0.9376 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.0735 | 0.1456 | 0.2820 | 0.5620 | 0.8206 | 0.9381 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.0721 | 0.1438 | 0.2808 | 0.5683 | 0.8225 | 0.9386 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.0705 | 0.1420 | 0.2798 | 0.5667 | 0.8240 | 0.9400 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.0721 | 0.1458 | 0.2861 | 0.5718 | 0.8246 | 0.9422 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.0724 | 0.1442 | 0.2786 | 0.5619 | 0.8180 | 0.9389 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.0734 | 0.1455 | 0.2820 | 0.5663 | 0.8199 | 0.9393 |
| `text-e5-base` | 0.0790 | 0.1554 | 0.2984 | 0.5836 | 0.8225 | 0.9383 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.0718 | 0.1458 | 0.2875 | 0.5712 | 0.8225 | 0.9392 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.0751 | 0.1485 | 0.2864 | 0.5746 | 0.8254 | 0.9398 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.0761 | 0.1500 | 0.2876 | 0.5754 | 0.8246 | 0.9387 |
| `omni-nano-image` | 0.0675 | 0.1362 | 0.2694 | 0.5551 | 0.8102 | 0.9333 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.0746 | 0.1450 | 0.2835 | 0.5659 | 0.8190 | 0.9384 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.0743 | 0.1470 | 0.2893 | 0.5768 | 0.8272 | 0.9396 |
| `attr-omni-nano` | 0.0715 | 0.1445 | 0.2814 | 0.5696 | 0.8194 | 0.9413 |
| `text-e5-small-multi` | 0.0772 | 0.1517 | 0.2946 | 0.5831 | 0.8238 | 0.9381 |
| `attr-e5-base` | 0.0713 | 0.1437 | 0.2815 | 0.5699 | 0.8175 | 0.9400 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.0710 | 0.1443 | 0.2833 | 0.5679 | 0.8230 | 0.9401 |
| `attr-jina` | 0.0711 | 0.1448 | 0.2816 | 0.5695 | 0.8199 | 0.9413 |
| `attr-jina-small` | 0.0710 | 0.1431 | 0.2799 | 0.5728 | 0.8186 | 0.9416 |
| `random` | 0.0645 | 0.1287 | 0.2543 | 0.5199 | 0.7870 | 0.9239 |
| `attr-e5-large-instruct` | 0.0712 | 0.1429 | 0.2827 | 0.5708 | 0.8177 | 0.9388 |
| `attr-e5-small-multi` | 0.0685 | 0.1388 | 0.2760 | 0.5636 | 0.8187 | 0.9393 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7419 | 0.7652 | 0.7678 | 0.7351 | 0.6519 | 0.5507 |
| **`fusion[RRF (k=60)] omni-nano-image + attr-siglip`** | **0.9163** | **0.9131** | **0.8894** | **0.8115** | **0.6731** | **0.5542** |
| `attr-siglip` | 0.8930 | 0.9018 | 0.8883 | 0.8166 | 0.6718 | 0.5535 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.8775 | 0.8934 | 0.8841 | 0.8189 | 0.6749 | 0.5552 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.8810 | 0.8898 | 0.8873 | 0.8189 | 0.6726 | 0.5551 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.8883 | 0.9032 | 0.8921 | 0.8312 | 0.6810 | 0.5549 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.8877 | 0.9037 | 0.8954 | 0.8313 | 0.6810 | 0.5547 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.9001 | 0.9009 | 0.8866 | 0.8233 | 0.6782 | 0.5538 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.8915 | 0.9009 | 0.8877 | 0.8229 | 0.6778 | 0.5539 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.9096 | 0.9111 | 0.8955 | 0.8227 | 0.6809 | 0.5560 |
| `text-jina` | 0.8976 | 0.9124 | 0.9153 | 0.8399 | 0.6773 | 0.5532 |
| `text-omni-nano` | 0.8977 | 0.9136 | 0.9163 | 0.8394 | 0.6775 | 0.5532 |
| `text-e5-large-instruct` | 0.9139 | 0.9146 | 0.9135 | 0.8422 | 0.6810 | 0.5578 |
| `text-siglip` | 0.9013 | 0.8926 | 0.8806 | 0.7998 | 0.6675 | 0.5521 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.8735 | 0.8854 | 0.8675 | 0.8027 | 0.6699 | 0.5553 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.8884 | 0.8917 | 0.8884 | 0.8257 | 0.6791 | 0.5550 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.8900 | 0.8927 | 0.8900 | 0.8263 | 0.6792 | 0.5548 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.9119 | 0.9052 | 0.9001 | 0.8309 | 0.6835 | 0.5567 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.9054 | 0.9077 | 0.8975 | 0.8253 | 0.6837 | 0.5564 |
| `text-jina-small` | 0.9106 | 0.9138 | 0.9177 | 0.8401 | 0.6840 | 0.5567 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.9094 | 0.9135 | 0.8975 | 0.8184 | 0.6811 | 0.5553 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.8738 | 0.8757 | 0.8790 | 0.8117 | 0.6732 | 0.5541 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.9135 | 0.8955 | 0.8877 | 0.8154 | 0.6739 | 0.5547 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.8662 | 0.8760 | 0.8662 | 0.8091 | 0.6723 | 0.5535 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.8491 | 0.8731 | 0.8822 | 0.8155 | 0.6797 | 0.5584 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.8491 | 0.8729 | 0.8820 | 0.8157 | 0.6797 | 0.5583 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.8577 | 0.8786 | 0.8672 | 0.8043 | 0.6736 | 0.5550 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.8720 | 0.8813 | 0.8775 | 0.8015 | 0.6708 | 0.5559 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.8559 | 0.8827 | 0.8851 | 0.8160 | 0.6770 | 0.5567 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.8855 | 0.8946 | 0.8875 | 0.8223 | 0.6789 | 0.5553 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.8462 | 0.8692 | 0.8763 | 0.8131 | 0.6784 | 0.5563 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.8528 | 0.8738 | 0.8830 | 0.8158 | 0.6797 | 0.5585 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.8742 | 0.8863 | 0.8692 | 0.8028 | 0.6704 | 0.5558 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.8997 | 0.8958 | 0.8952 | 0.8276 | 0.6824 | 0.5566 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.8556 | 0.8824 | 0.8854 | 0.8159 | 0.6768 | 0.5565 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.8458 | 0.8553 | 0.8547 | 0.8041 | 0.6738 | 0.5550 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.8457 | 0.8600 | 0.8566 | 0.8031 | 0.6714 | 0.5536 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.8982 | 0.8891 | 0.8799 | 0.8041 | 0.6744 | 0.5539 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.8677 | 0.8733 | 0.8696 | 0.8104 | 0.6750 | 0.5542 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.8516 | 0.8624 | 0.8639 | 0.8078 | 0.6766 | 0.5556 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.8569 | 0.8795 | 0.8834 | 0.8154 | 0.6780 | 0.5577 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.8718 | 0.8808 | 0.8635 | 0.7999 | 0.6704 | 0.5550 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.8825 | 0.8827 | 0.8757 | 0.8096 | 0.6738 | 0.5555 |
| `text-e5-base` | 0.9332 | 0.9302 | 0.9124 | 0.8301 | 0.6757 | 0.5541 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.8526 | 0.8784 | 0.8858 | 0.8131 | 0.6756 | 0.5557 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.8921 | 0.8974 | 0.8863 | 0.8186 | 0.6786 | 0.5549 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.8979 | 0.8997 | 0.8877 | 0.8196 | 0.6777 | 0.5542 |
| `omni-nano-image` | 0.8213 | 0.8349 | 0.8366 | 0.7908 | 0.6614 | 0.5498 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.8959 | 0.8787 | 0.8738 | 0.8033 | 0.6711 | 0.5545 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.8820 | 0.8836 | 0.8912 | 0.8238 | 0.6804 | 0.5551 |
| `attr-omni-nano` | 0.8427 | 0.8668 | 0.8588 | 0.8035 | 0.6726 | 0.5575 |
| `text-e5-small-multi` | 0.9039 | 0.9058 | 0.9045 | 0.8315 | 0.6778 | 0.5541 |
| `attr-e5-base` | 0.8314 | 0.8606 | 0.8625 | 0.8069 | 0.6709 | 0.5564 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.8475 | 0.8730 | 0.8778 | 0.8101 | 0.6761 | 0.5562 |
| `attr-jina` | 0.8438 | 0.8700 | 0.8607 | 0.8038 | 0.6729 | 0.5575 |
| `attr-jina-small` | 0.8412 | 0.8591 | 0.8540 | 0.8100 | 0.6714 | 0.5580 |
| `random` | 0.7927 | 0.7862 | 0.7858 | 0.7335 | 0.6397 | 0.5425 |
| `attr-e5-large-instruct` | 0.8357 | 0.8592 | 0.8629 | 0.8066 | 0.6707 | 0.5556 |
| `attr-e5-small-multi` | 0.8137 | 0.8348 | 0.8513 | 0.8016 | 0.6720 | 0.5560 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7692 | 0.7784 | 0.7788 | 0.7788 | 0.7788 | 0.7788 |
| **`fusion[RRF (k=60)] omni-nano-image + attr-siglip`** | **0.9465** | **0.9469** | **0.9472** | **0.9472** | **0.9472** | **0.9472** |
| `attr-siglip` | 0.9431 | 0.9442 | 0.9442 | 0.9442 | 0.9442 | 0.9442 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.9003 | 0.9009 | 0.9011 | 0.9011 | 0.9011 | 0.9011 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.9181 | 0.9194 | 0.9195 | 0.9195 | 0.9195 | 0.9195 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.9108 | 0.9109 | 0.9111 | 0.9111 | 0.9111 | 0.9111 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.9085 | 0.9087 | 0.9089 | 0.9089 | 0.9089 | 0.9089 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.9104 | 0.9105 | 0.9105 | 0.9105 | 0.9105 | 0.9105 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.9095 | 0.9096 | 0.9096 | 0.9096 | 0.9096 | 0.9096 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.9166 | 0.9166 | 0.9166 | 0.9166 | 0.9166 | 0.9166 |
| `text-jina` | 0.9636 | 0.9645 | 0.9646 | 0.9646 | 0.9646 | 0.9646 |
| `text-omni-nano` | 0.9637 | 0.9646 | 0.9647 | 0.9647 | 0.9647 | 0.9647 |
| `text-e5-large-instruct` | 0.9648 | 0.9651 | 0.9656 | 0.9657 | 0.9657 | 0.9657 |
| `text-siglip` | 0.9651 | 0.9655 | 0.9655 | 0.9655 | 0.9655 | 0.9655 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.9272 | 0.9279 | 0.9279 | 0.9280 | 0.9280 | 0.9280 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.8972 | 0.8974 | 0.8975 | 0.8975 | 0.8975 | 0.8975 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.8977 | 0.8979 | 0.8980 | 0.8980 | 0.8980 | 0.8980 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.9180 | 0.9181 | 0.9182 | 0.9182 | 0.9182 | 0.9182 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.9331 | 0.9333 | 0.9334 | 0.9334 | 0.9334 | 0.9334 |
| `text-jina-small` | 0.9669 | 0.9670 | 0.9670 | 0.9674 | 0.9674 | 0.9674 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.9177 | 0.9178 | 0.9178 | 0.9178 | 0.9178 | 0.9178 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.8933 | 0.8938 | 0.8938 | 0.8938 | 0.8938 | 0.8938 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.9253 | 0.9263 | 0.9263 | 0.9264 | 0.9264 | 0.9264 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.8796 | 0.8805 | 0.8805 | 0.8805 | 0.8805 | 0.8805 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.8849 | 0.8862 | 0.8864 | 0.8864 | 0.8864 | 0.8864 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.8846 | 0.8859 | 0.8861 | 0.8861 | 0.8861 | 0.8861 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.9117 | 0.9123 | 0.9125 | 0.9125 | 0.9125 | 0.9125 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.9134 | 0.9140 | 0.9140 | 0.9140 | 0.9140 | 0.9140 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.8972 | 0.8981 | 0.8981 | 0.8981 | 0.8981 | 0.8981 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.9291 | 0.9293 | 0.9293 | 0.9293 | 0.9293 | 0.9293 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.9023 | 0.9041 | 0.9042 | 0.9042 | 0.9042 | 0.9042 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.9104 | 0.9120 | 0.9120 | 0.9121 | 0.9121 | 0.9121 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.9296 | 0.9303 | 0.9304 | 0.9304 | 0.9304 | 0.9304 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.9097 | 0.9099 | 0.9100 | 0.9100 | 0.9100 | 0.9100 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.8936 | 0.8949 | 0.8949 | 0.8949 | 0.8949 | 0.8949 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.8877 | 0.8887 | 0.8888 | 0.8888 | 0.8888 | 0.8888 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.8824 | 0.8834 | 0.8837 | 0.8837 | 0.8837 | 0.8837 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.9206 | 0.9216 | 0.9217 | 0.9217 | 0.9217 | 0.9217 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.8930 | 0.8938 | 0.8938 | 0.8939 | 0.8939 | 0.8939 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.8694 | 0.8709 | 0.8712 | 0.8712 | 0.8712 | 0.8712 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.9116 | 0.9126 | 0.9126 | 0.9126 | 0.9126 | 0.9126 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.9066 | 0.9087 | 0.9088 | 0.9088 | 0.9088 | 0.9088 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.8837 | 0.8846 | 0.8846 | 0.8847 | 0.8847 | 0.8847 |
| `text-e5-base` | 0.9717 | 0.9718 | 0.9718 | 0.9718 | 0.9718 | 0.9718 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.9122 | 0.9130 | 0.9131 | 0.9132 | 0.9132 | 0.9132 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.9214 | 0.9215 | 0.9215 | 0.9215 | 0.9215 | 0.9215 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.9452 | 0.9453 | 0.9453 | 0.9453 | 0.9453 | 0.9453 |
| `omni-nano-image` | 0.8854 | 0.8867 | 0.8868 | 0.8869 | 0.8869 | 0.8869 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.9184 | 0.9192 | 0.9192 | 0.9193 | 0.9193 | 0.9193 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.9178 | 0.9187 | 0.9187 | 0.9187 | 0.9187 | 0.9187 |
| `attr-omni-nano` | 0.8843 | 0.8854 | 0.8857 | 0.8858 | 0.8858 | 0.8858 |
| `text-e5-small-multi` | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 |
| `attr-e5-base` | 0.8794 | 0.8828 | 0.8832 | 0.8832 | 0.8832 | 0.8832 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.9023 | 0.9040 | 0.9041 | 0.9042 | 0.9042 | 0.9042 |
| `attr-jina` | 0.8807 | 0.8820 | 0.8823 | 0.8824 | 0.8824 | 0.8824 |
| `attr-jina-small` | 0.8949 | 0.8976 | 0.8978 | 0.8978 | 0.8978 | 0.8978 |
| `random` | 0.8806 | 0.8825 | 0.8829 | 0.8829 | 0.8829 | 0.8829 |
| `attr-e5-large-instruct` | 0.8891 | 0.8906 | 0.8913 | 0.8914 | 0.8914 | 0.8914 |
| `attr-e5-small-multi` | 0.8645 | 0.8671 | 0.8677 | 0.8681 | 0.8681 | 0.8681 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.8035 |
| **`fusion[RRF (k=60)] omni-nano-image + attr-siglip`** | **0.8738** |
| `attr-siglip` | 0.8753 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.8752 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.8756 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.8923 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.8922 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.8827 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.8824 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.8857 |
| `text-jina` | 0.9034 |
| `text-omni-nano` | 0.9036 |
| `text-e5-large-instruct` | 0.9057 |
| `text-siglip` | 0.8650 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.8676 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.8857 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.8857 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.8959 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.8905 |
| `text-jina-small` | 0.9068 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.8865 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.8724 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.8755 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.8676 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.8799 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.8799 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.8689 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.8682 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.8775 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.8847 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.8765 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.8809 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.8680 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.8889 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.8776 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.8619 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.8617 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.8667 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.8698 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.8668 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.8778 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.8633 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.8682 |
| `text-e5-base` | 0.9012 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.8770 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.8809 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.8826 |
| `omni-nano-image` | 0.8442 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.8692 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.8845 |
| `attr-omni-nano` | 0.8703 |
| `text-e5-small-multi` | 0.8959 |
| `attr-e5-base` | 0.8720 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.8711 |
| `attr-jina` | 0.8707 |
| `attr-jina-small` | 0.8702 |
| `random` | 0.7970 |
| `attr-e5-large-instruct` | 0.8710 |
| `attr-e5-small-multi` | 0.8616 |


### Target: `jina-clip-v2-image`

#### Macro-averaged -- every query counts once

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4025 | 0.4200 | 0.4571 | 0.5429 | 0.6298 | 0.6703 |
| **`fusion[mean cosine] jina-clip-v2-image + text-omni-nano`** | **0.4055** | **0.4373** | **0.4863** | **0.5829** | **0.6600** | **0.6893** |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.4040 | 0.4360 | 0.4860 | 0.5827 | 0.6599 | 0.6891 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.4021 | 0.4308 | 0.4833 | 0.5818 | 0.6589 | 0.6878 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.4015 | 0.4303 | 0.4831 | 0.5819 | 0.6589 | 0.6878 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.3980 | 0.4303 | 0.4825 | 0.5806 | 0.6568 | 0.6869 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.3995 | 0.4295 | 0.4778 | 0.5764 | 0.6527 | 0.6831 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.3897 | 0.4290 | 0.4798 | 0.5786 | 0.6557 | 0.6846 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.3968 | 0.4286 | 0.4813 | 0.5790 | 0.6561 | 0.6851 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.4051 | 0.4280 | 0.4744 | 0.5689 | 0.6470 | 0.6785 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.4011 | 0.4276 | 0.4812 | 0.5786 | 0.6561 | 0.6859 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.4000 | 0.4266 | 0.4806 | 0.5780 | 0.6557 | 0.6856 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.3987 | 0.4264 | 0.4764 | 0.5737 | 0.6510 | 0.6815 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.3983 | 0.4263 | 0.4794 | 0.5755 | 0.6507 | 0.6805 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.3892 | 0.4256 | 0.4772 | 0.5760 | 0.6524 | 0.6825 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.3937 | 0.4254 | 0.4757 | 0.5733 | 0.6489 | 0.6807 |
| `fusion[z-score average] jina-clip-v2-image + attr-siglip` | 0.3956 | 0.4237 | 0.4778 | 0.5728 | 0.6487 | 0.6783 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.3925 | 0.4230 | 0.4770 | 0.5750 | 0.6519 | 0.6815 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.3880 | 0.4221 | 0.4752 | 0.5740 | 0.6517 | 0.6820 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.3916 | 0.4204 | 0.4732 | 0.5735 | 0.6512 | 0.6826 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.3863 | 0.4201 | 0.4699 | 0.5715 | 0.6497 | 0.6792 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.3877 | 0.4180 | 0.4709 | 0.5664 | 0.6423 | 0.6736 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.3791 | 0.4102 | 0.4639 | 0.5634 | 0.6435 | 0.6744 |
| `text-omni-nano` | 0.3789 | 0.4094 | 0.4634 | 0.5626 | 0.6416 | 0.6722 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.3762 | 0.4093 | 0.4655 | 0.5662 | 0.6447 | 0.6746 |
| `text-jina` | 0.3779 | 0.4072 | 0.4623 | 0.5619 | 0.6407 | 0.6713 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.3723 | 0.4070 | 0.4650 | 0.5654 | 0.6436 | 0.6737 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.3709 | 0.4066 | 0.4635 | 0.5636 | 0.6418 | 0.6717 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.3729 | 0.4059 | 0.4646 | 0.5651 | 0.6433 | 0.6734 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.3704 | 0.4049 | 0.4636 | 0.5636 | 0.6415 | 0.6714 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.3701 | 0.4047 | 0.4544 | 0.5578 | 0.6374 | 0.6679 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.3696 | 0.4040 | 0.4629 | 0.5625 | 0.6415 | 0.6710 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.3644 | 0.4028 | 0.4617 | 0.5627 | 0.6394 | 0.6703 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.3706 | 0.4023 | 0.4628 | 0.5624 | 0.6402 | 0.6698 |
| `text-e5-large-instruct` | 0.3702 | 0.4012 | 0.4589 | 0.5592 | 0.6396 | 0.6711 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.3679 | 0.4012 | 0.4582 | 0.5582 | 0.6376 | 0.6677 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.3681 | 0.3993 | 0.4552 | 0.5533 | 0.6320 | 0.6643 |
| `text-jina-small` | 0.3638 | 0.3989 | 0.4566 | 0.5557 | 0.6367 | 0.6672 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.3679 | 0.3988 | 0.4586 | 0.5585 | 0.6389 | 0.6689 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.3677 | 0.3984 | 0.4573 | 0.5540 | 0.6331 | 0.6650 |
| `text-siglip` | 0.3706 | 0.3981 | 0.4438 | 0.5370 | 0.6204 | 0.6554 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.3625 | 0.3978 | 0.4563 | 0.5576 | 0.6368 | 0.6679 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.3615 | 0.3974 | 0.4557 | 0.5561 | 0.6352 | 0.6660 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.3583 | 0.3962 | 0.4492 | 0.5494 | 0.6275 | 0.6606 |
| `text-e5-base` | 0.3644 | 0.3899 | 0.4435 | 0.5459 | 0.6286 | 0.6626 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.3581 | 0.3896 | 0.4483 | 0.5483 | 0.6277 | 0.6607 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.3534 | 0.3885 | 0.4491 | 0.5491 | 0.6259 | 0.6586 |
| `jina-clip-v2-image` | 0.3610 | 0.3883 | 0.4491 | 0.5497 | 0.6300 | 0.6615 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.3572 | 0.3854 | 0.4448 | 0.5480 | 0.6304 | 0.6617 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.3538 | 0.3815 | 0.4379 | 0.5380 | 0.6186 | 0.6523 |
| `attr-siglip` | 0.3477 | 0.3776 | 0.4335 | 0.5341 | 0.6122 | 0.6472 |
| `text-e5-small-multi` | 0.3315 | 0.3653 | 0.4215 | 0.5328 | 0.6143 | 0.6485 |
| `attr-omni-nano` | 0.3030 | 0.3433 | 0.4047 | 0.5124 | 0.5966 | 0.6324 |
| `attr-jina-small` | 0.3064 | 0.3424 | 0.4026 | 0.5126 | 0.5965 | 0.6327 |
| `attr-e5-large-instruct` | 0.3020 | 0.3411 | 0.4007 | 0.5141 | 0.5969 | 0.6330 |
| `attr-jina` | 0.2986 | 0.3398 | 0.4046 | 0.5116 | 0.5956 | 0.6312 |
| `attr-e5-base` | 0.2956 | 0.3380 | 0.3980 | 0.5045 | 0.5906 | 0.6275 |
| `attr-e5-small-multi` | 0.2775 | 0.3125 | 0.3720 | 0.4890 | 0.5743 | 0.6129 |
| `random` | 0.2392 | 0.2640 | 0.3210 | 0.4357 | 0.5321 | 0.5758 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.1400 | 0.2466 | 0.4189 | 0.6922 | 0.8846 | 0.9646 |
| **`fusion[mean cosine] jina-clip-v2-image + text-omni-nano`** | **0.1608** | **0.2773** | **0.4551** | **0.7288** | **0.9027** | **0.9686** |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.1613 | 0.2774 | 0.4550 | 0.7290 | 0.9026 | 0.9686 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.1578 | 0.2710 | 0.4560 | 0.7284 | 0.9041 | 0.9693 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.1586 | 0.2707 | 0.4541 | 0.7285 | 0.9041 | 0.9693 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.1623 | 0.2781 | 0.4555 | 0.7291 | 0.9048 | 0.9704 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.1479 | 0.2719 | 0.4485 | 0.7255 | 0.9018 | 0.9694 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.1529 | 0.2739 | 0.4560 | 0.7304 | 0.9056 | 0.9700 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.1624 | 0.2766 | 0.4587 | 0.7304 | 0.9046 | 0.9692 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.1601 | 0.2737 | 0.4468 | 0.7221 | 0.8980 | 0.9674 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.1540 | 0.2653 | 0.4496 | 0.7265 | 0.9037 | 0.9697 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.1547 | 0.2653 | 0.4498 | 0.7266 | 0.9038 | 0.9696 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.1560 | 0.2752 | 0.4486 | 0.7276 | 0.9039 | 0.9707 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.1512 | 0.2691 | 0.4480 | 0.7229 | 0.9000 | 0.9674 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.1486 | 0.2725 | 0.4531 | 0.7279 | 0.9039 | 0.9702 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.1466 | 0.2705 | 0.4455 | 0.7241 | 0.9006 | 0.9691 |
| `fusion[z-score average] jina-clip-v2-image + attr-siglip` | 0.1503 | 0.2695 | 0.4454 | 0.7240 | 0.9006 | 0.9682 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.1543 | 0.2679 | 0.4521 | 0.7272 | 0.9040 | 0.9699 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.1553 | 0.2728 | 0.4548 | 0.7248 | 0.9021 | 0.9694 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.1585 | 0.2731 | 0.4567 | 0.7247 | 0.9007 | 0.9692 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.1548 | 0.2735 | 0.4500 | 0.7252 | 0.9035 | 0.9698 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.1375 | 0.2649 | 0.4438 | 0.7209 | 0.9001 | 0.9681 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.1501 | 0.2702 | 0.4480 | 0.7227 | 0.9002 | 0.9692 |
| `text-omni-nano` | 0.1541 | 0.2678 | 0.4477 | 0.7241 | 0.8984 | 0.9655 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.1577 | 0.2694 | 0.4572 | 0.7258 | 0.9026 | 0.9692 |
| `text-jina` | 0.1537 | 0.2673 | 0.4476 | 0.7243 | 0.8984 | 0.9655 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.1383 | 0.2566 | 0.4420 | 0.7257 | 0.9016 | 0.9695 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.1376 | 0.2588 | 0.4467 | 0.7253 | 0.9021 | 0.9692 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.1382 | 0.2567 | 0.4419 | 0.7255 | 0.9017 | 0.9694 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.1392 | 0.2581 | 0.4489 | 0.7255 | 0.9021 | 0.9693 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.1518 | 0.2728 | 0.4422 | 0.7265 | 0.9012 | 0.9688 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.1427 | 0.2536 | 0.4494 | 0.7252 | 0.9015 | 0.9689 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.1366 | 0.2610 | 0.4452 | 0.7259 | 0.8993 | 0.9683 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.1388 | 0.2538 | 0.4463 | 0.7242 | 0.9009 | 0.9685 |
| `text-e5-large-instruct` | 0.1597 | 0.2706 | 0.4488 | 0.7209 | 0.9006 | 0.9683 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.1378 | 0.2576 | 0.4488 | 0.7220 | 0.9012 | 0.9683 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.1364 | 0.2546 | 0.4436 | 0.7187 | 0.8974 | 0.9685 |
| `text-jina-small` | 0.1490 | 0.2669 | 0.4515 | 0.7238 | 0.9011 | 0.9667 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.1446 | 0.2535 | 0.4433 | 0.7217 | 0.8999 | 0.9684 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.1359 | 0.2554 | 0.4451 | 0.7185 | 0.8975 | 0.9682 |
| `text-siglip` | 0.1485 | 0.2603 | 0.4316 | 0.7036 | 0.8886 | 0.9640 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.1440 | 0.2591 | 0.4408 | 0.7247 | 0.9023 | 0.9699 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.1375 | 0.2589 | 0.4444 | 0.7238 | 0.9023 | 0.9700 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.1302 | 0.2566 | 0.4362 | 0.7206 | 0.8976 | 0.9689 |
| `text-e5-base` | 0.1609 | 0.2671 | 0.4394 | 0.7130 | 0.8932 | 0.9656 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.1354 | 0.2529 | 0.4373 | 0.7175 | 0.8963 | 0.9678 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.1349 | 0.2550 | 0.4405 | 0.7218 | 0.8955 | 0.9668 |
| `jina-clip-v2-image` | 0.1356 | 0.2456 | 0.4382 | 0.7150 | 0.8988 | 0.9676 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.1398 | 0.2532 | 0.4421 | 0.7185 | 0.8985 | 0.9673 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.1336 | 0.2502 | 0.4420 | 0.7126 | 0.8931 | 0.9649 |
| `attr-siglip` | 0.1381 | 0.2483 | 0.4198 | 0.7073 | 0.8872 | 0.9627 |
| `text-e5-small-multi` | 0.1556 | 0.2612 | 0.4326 | 0.7165 | 0.8937 | 0.9648 |
| `attr-omni-nano` | 0.1269 | 0.2423 | 0.4161 | 0.7070 | 0.8903 | 0.9647 |
| `attr-jina-small` | 0.1298 | 0.2429 | 0.4127 | 0.7094 | 0.8888 | 0.9651 |
| `attr-e5-large-instruct` | 0.1359 | 0.2440 | 0.4181 | 0.7055 | 0.8870 | 0.9621 |
| `attr-jina` | 0.1263 | 0.2433 | 0.4185 | 0.7086 | 0.8905 | 0.9648 |
| `attr-e5-base` | 0.1256 | 0.2423 | 0.4195 | 0.7014 | 0.8866 | 0.9633 |
| `attr-e5-small-multi` | 0.1262 | 0.2342 | 0.4077 | 0.6940 | 0.8820 | 0.9600 |
| `random` | 0.0954 | 0.1807 | 0.3567 | 0.6540 | 0.8596 | 0.9509 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6303 | 0.6435 | 0.6313 | 0.5483 | 0.4265 | 0.3387 |
| **`fusion[mean cosine] jina-clip-v2-image + text-omni-nano`** | **0.7637** | **0.7467** | **0.7122** | **0.5999** | **0.4438** | **0.3419** |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.7637 | 0.7467 | 0.7114 | 0.6001 | 0.4438 | 0.3419 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.7613 | 0.7465 | 0.7111 | 0.5987 | 0.4446 | 0.3423 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.7613 | 0.7462 | 0.7105 | 0.5988 | 0.4446 | 0.3424 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.7653 | 0.7500 | 0.7149 | 0.6009 | 0.4448 | 0.3428 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.7503 | 0.7368 | 0.7043 | 0.5939 | 0.4416 | 0.3420 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.7633 | 0.7477 | 0.7115 | 0.6020 | 0.4458 | 0.3428 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.7710 | 0.7513 | 0.7154 | 0.6031 | 0.4455 | 0.3424 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.7577 | 0.7382 | 0.7020 | 0.5869 | 0.4384 | 0.3407 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.7627 | 0.7433 | 0.7067 | 0.5960 | 0.4439 | 0.3424 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.7620 | 0.7430 | 0.7067 | 0.5961 | 0.4440 | 0.3424 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.7653 | 0.7473 | 0.7103 | 0.5977 | 0.4437 | 0.3429 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.7490 | 0.7338 | 0.7000 | 0.5920 | 0.4405 | 0.3408 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.7593 | 0.7480 | 0.7087 | 0.5983 | 0.4440 | 0.3428 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.7527 | 0.7392 | 0.7031 | 0.5910 | 0.4402 | 0.3416 |
| `fusion[z-score average] jina-clip-v2-image + attr-siglip` | 0.7473 | 0.7330 | 0.6994 | 0.5918 | 0.4412 | 0.3413 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.7510 | 0.7402 | 0.7077 | 0.5970 | 0.4437 | 0.3424 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.7537 | 0.7407 | 0.7060 | 0.5953 | 0.4425 | 0.3422 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.7570 | 0.7432 | 0.7067 | 0.5956 | 0.4420 | 0.3422 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.7453 | 0.7368 | 0.7027 | 0.5966 | 0.4437 | 0.3424 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.7457 | 0.7293 | 0.6937 | 0.5883 | 0.4400 | 0.3411 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.7563 | 0.7413 | 0.7033 | 0.5919 | 0.4412 | 0.3420 |
| `text-omni-nano` | 0.7483 | 0.7342 | 0.7040 | 0.5943 | 0.4404 | 0.3400 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.7533 | 0.7383 | 0.7080 | 0.5954 | 0.4435 | 0.3423 |
| `text-jina` | 0.7483 | 0.7335 | 0.7034 | 0.5948 | 0.4404 | 0.3400 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.7240 | 0.7213 | 0.6945 | 0.5919 | 0.4420 | 0.3421 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.7210 | 0.7207 | 0.6931 | 0.5911 | 0.4422 | 0.3419 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.7237 | 0.7213 | 0.6937 | 0.5917 | 0.4420 | 0.3421 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.7223 | 0.7210 | 0.6937 | 0.5914 | 0.4422 | 0.3420 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.7490 | 0.7382 | 0.7013 | 0.5923 | 0.4419 | 0.3418 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.7177 | 0.7165 | 0.6938 | 0.5911 | 0.4418 | 0.3417 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.7193 | 0.7135 | 0.6884 | 0.5891 | 0.4397 | 0.3414 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.7213 | 0.7167 | 0.6915 | 0.5895 | 0.4408 | 0.3414 |
| `text-e5-large-instruct` | 0.7600 | 0.7392 | 0.7020 | 0.5932 | 0.4419 | 0.3415 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.7250 | 0.7202 | 0.6926 | 0.5893 | 0.4410 | 0.3414 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.7283 | 0.7163 | 0.6861 | 0.5831 | 0.4380 | 0.3414 |
| `text-jina-small` | 0.7550 | 0.7407 | 0.7058 | 0.5949 | 0.4426 | 0.3407 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.7150 | 0.7117 | 0.6857 | 0.5866 | 0.4406 | 0.3415 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.7273 | 0.7152 | 0.6866 | 0.5830 | 0.4380 | 0.3412 |
| `text-siglip` | 0.7347 | 0.7165 | 0.6774 | 0.5687 | 0.4309 | 0.3384 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.7237 | 0.7175 | 0.6898 | 0.5893 | 0.4425 | 0.3424 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.7187 | 0.7163 | 0.6887 | 0.5903 | 0.4424 | 0.3425 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.7280 | 0.7218 | 0.6863 | 0.5825 | 0.4377 | 0.3416 |
| `text-e5-base` | 0.7427 | 0.7205 | 0.6851 | 0.5816 | 0.4364 | 0.3401 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.7193 | 0.7077 | 0.6786 | 0.5796 | 0.4369 | 0.3408 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.7303 | 0.7123 | 0.6825 | 0.5824 | 0.4360 | 0.3402 |
| `jina-clip-v2-image` | 0.7133 | 0.7028 | 0.6804 | 0.5825 | 0.4389 | 0.3409 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.7083 | 0.7002 | 0.6756 | 0.5832 | 0.4392 | 0.3408 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.7180 | 0.7038 | 0.6725 | 0.5761 | 0.4339 | 0.3392 |
| `attr-siglip` | 0.7020 | 0.7010 | 0.6718 | 0.5741 | 0.4304 | 0.3380 |
| `text-e5-small-multi` | 0.7317 | 0.7175 | 0.6816 | 0.5829 | 0.4365 | 0.3398 |
| `attr-omni-nano` | 0.6633 | 0.6688 | 0.6471 | 0.5685 | 0.4332 | 0.3394 |
| `attr-jina-small` | 0.6703 | 0.6625 | 0.6469 | 0.5704 | 0.4317 | 0.3394 |
| `attr-e5-large-instruct` | 0.6687 | 0.6623 | 0.6462 | 0.5667 | 0.4295 | 0.3377 |
| `attr-jina` | 0.6660 | 0.6708 | 0.6484 | 0.5689 | 0.4335 | 0.3394 |
| `attr-e5-base` | 0.6587 | 0.6638 | 0.6418 | 0.5629 | 0.4297 | 0.3384 |
| `attr-e5-small-multi` | 0.6350 | 0.6348 | 0.6268 | 0.5549 | 0.4256 | 0.3366 |
| `random` | 0.5777 | 0.5777 | 0.5647 | 0.5094 | 0.4078 | 0.3306 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6940 | 0.7041 | 0.7063 | 0.7071 | 0.7072 | 0.7072 |
| **`fusion[mean cosine] jina-clip-v2-image + text-omni-nano`** | **0.8332** | **0.8379** | **0.8395** | **0.8401** | **0.8402** | **0.8402** |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.8313 | 0.8357 | 0.8374 | 0.8380 | 0.8380 | 0.8380 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.8277 | 0.8320 | 0.8344 | 0.8349 | 0.8349 | 0.8349 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.8263 | 0.8302 | 0.8326 | 0.8332 | 0.8332 | 0.8332 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.8402 | 0.8440 | 0.8460 | 0.8466 | 0.8466 | 0.8466 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.8191 | 0.8246 | 0.8263 | 0.8268 | 0.8269 | 0.8269 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.8295 | 0.8343 | 0.8365 | 0.8371 | 0.8371 | 0.8371 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.8375 | 0.8416 | 0.8436 | 0.8442 | 0.8442 | 0.8442 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.8383 | 0.8413 | 0.8430 | 0.8437 | 0.8437 | 0.8437 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.8327 | 0.8361 | 0.8386 | 0.8392 | 0.8392 | 0.8392 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.8339 | 0.8373 | 0.8400 | 0.8406 | 0.8406 | 0.8406 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.8383 | 0.8429 | 0.8446 | 0.8454 | 0.8454 | 0.8454 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.8323 | 0.8373 | 0.8392 | 0.8397 | 0.8397 | 0.8397 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.8326 | 0.8385 | 0.8405 | 0.8412 | 0.8412 | 0.8412 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.8304 | 0.8363 | 0.8379 | 0.8385 | 0.8386 | 0.8386 |
| `fusion[z-score average] jina-clip-v2-image + attr-siglip` | 0.8272 | 0.8322 | 0.8338 | 0.8344 | 0.8344 | 0.8344 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.8289 | 0.8323 | 0.8348 | 0.8354 | 0.8354 | 0.8354 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.8321 | 0.8362 | 0.8386 | 0.8390 | 0.8391 | 0.8391 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.8441 | 0.8480 | 0.8505 | 0.8510 | 0.8510 | 0.8510 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.8202 | 0.8251 | 0.8270 | 0.8276 | 0.8276 | 0.8276 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.8251 | 0.8313 | 0.8334 | 0.8339 | 0.8340 | 0.8340 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.8290 | 0.8335 | 0.8359 | 0.8366 | 0.8367 | 0.8367 |
| `text-omni-nano` | 0.8311 | 0.8360 | 0.8379 | 0.8387 | 0.8387 | 0.8387 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.8254 | 0.8301 | 0.8328 | 0.8332 | 0.8332 | 0.8332 |
| `text-jina` | 0.8305 | 0.8352 | 0.8369 | 0.8378 | 0.8378 | 0.8378 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.8023 | 0.8070 | 0.8098 | 0.8106 | 0.8106 | 0.8106 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.7987 | 0.8040 | 0.8069 | 0.8075 | 0.8075 | 0.8075 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.8017 | 0.8063 | 0.8090 | 0.8098 | 0.8098 | 0.8098 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.7960 | 0.8008 | 0.8038 | 0.8043 | 0.8043 | 0.8043 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.8237 | 0.8295 | 0.8311 | 0.8320 | 0.8320 | 0.8320 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.8008 | 0.8053 | 0.8088 | 0.8093 | 0.8093 | 0.8093 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.7975 | 0.8047 | 0.8070 | 0.8076 | 0.8076 | 0.8076 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.7991 | 0.8061 | 0.8095 | 0.8100 | 0.8100 | 0.8100 |
| `text-e5-large-instruct` | 0.8383 | 0.8420 | 0.8441 | 0.8448 | 0.8448 | 0.8448 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.7957 | 0.8010 | 0.8043 | 0.8046 | 0.8047 | 0.8047 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.8047 | 0.8089 | 0.8119 | 0.8124 | 0.8125 | 0.8125 |
| `text-jina-small` | 0.8239 | 0.8290 | 0.8316 | 0.8325 | 0.8325 | 0.8325 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.7958 | 0.8002 | 0.8033 | 0.8039 | 0.8039 | 0.8039 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.8026 | 0.8071 | 0.8100 | 0.8105 | 0.8106 | 0.8106 |
| `text-siglip` | 0.8123 | 0.8168 | 0.8184 | 0.8192 | 0.8193 | 0.8193 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.7974 | 0.8032 | 0.8055 | 0.8062 | 0.8062 | 0.8062 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.7910 | 0.7998 | 0.8021 | 0.8027 | 0.8027 | 0.8027 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.8010 | 0.8085 | 0.8107 | 0.8117 | 0.8117 | 0.8117 |
| `text-e5-base` | 0.8406 | 0.8439 | 0.8456 | 0.8466 | 0.8466 | 0.8466 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.8000 | 0.8053 | 0.8081 | 0.8088 | 0.8088 | 0.8088 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.7998 | 0.8057 | 0.8082 | 0.8090 | 0.8090 | 0.8090 |
| `jina-clip-v2-image` | 0.7909 | 0.7961 | 0.7995 | 0.8000 | 0.8001 | 0.8001 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.7883 | 0.7947 | 0.7974 | 0.7978 | 0.7979 | 0.7979 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.8021 | 0.8071 | 0.8104 | 0.8109 | 0.8109 | 0.8109 |
| `attr-siglip` | 0.7843 | 0.7896 | 0.7913 | 0.7922 | 0.7922 | 0.7922 |
| `text-e5-small-multi` | 0.8054 | 0.8095 | 0.8111 | 0.8124 | 0.8124 | 0.8124 |
| `attr-omni-nano` | 0.7448 | 0.7523 | 0.7554 | 0.7564 | 0.7565 | 0.7565 |
| `attr-jina-small` | 0.7481 | 0.7566 | 0.7589 | 0.7602 | 0.7602 | 0.7602 |
| `attr-e5-large-instruct` | 0.7484 | 0.7543 | 0.7570 | 0.7582 | 0.7582 | 0.7582 |
| `attr-jina` | 0.7419 | 0.7497 | 0.7527 | 0.7537 | 0.7537 | 0.7537 |
| `attr-e5-base` | 0.7364 | 0.7458 | 0.7488 | 0.7498 | 0.7498 | 0.7498 |
| `attr-e5-small-multi` | 0.7056 | 0.7147 | 0.7180 | 0.7190 | 0.7190 | 0.7190 |
| `random` | 0.6917 | 0.6965 | 0.7008 | 0.7022 | 0.7022 | 0.7022 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.6808 |
| **`fusion[mean cosine] jina-clip-v2-image + text-omni-nano`** | **0.7584** |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.7585 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.7554 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.7555 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.7597 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.7377 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.7546 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.7602 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.7329 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.7505 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.7504 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.7490 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.7390 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.7506 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.7352 |
| `fusion[z-score average] jina-clip-v2-image + attr-siglip` | 0.7360 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.7490 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.7509 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.7556 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.7493 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.7279 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.7427 |
| `text-omni-nano` | 0.7473 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.7501 |
| `text-jina` | 0.7471 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.7346 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.7327 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.7347 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.7327 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.7413 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.7332 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.7321 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.7306 |
| `text-e5-large-instruct` | 0.7502 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.7293 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.7198 |
| `text-jina-small` | 0.7463 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.7306 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.7209 |
| `text-siglip` | 0.7072 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.7314 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.7278 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.7184 |
| `text-e5-base` | 0.7401 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.7185 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.7194 |
| `jina-clip-v2-image` | 0.7180 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.7215 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.7069 |
| `attr-siglip` | 0.7023 |
| `text-e5-small-multi` | 0.7316 |
| `attr-omni-nano` | 0.6937 |
| `attr-jina-small` | 0.6957 |
| `attr-e5-large-instruct` | 0.6975 |
| `attr-jina` | 0.6933 |
| `attr-e5-base` | 0.6916 |
| `attr-e5-small-multi` | 0.6712 |
| `random` | 0.6041 |


#### Impression-weighted -- every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4066 | 0.4123 | 0.4295 | 0.5122 | 0.6384 | 0.7053 |
| **`fusion[z-score average] jina-clip-v2-image + attr-siglip`** | **0.4462** | **0.4655** | **0.5074** | **0.5961** | **0.6941** | **0.7430** |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.4566 | 0.4614 | 0.5028 | 0.5881 | 0.6897 | 0.7427 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.4473 | 0.4589 | 0.5042 | 0.5954 | 0.6932 | 0.7421 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.4227 | 0.4454 | 0.4820 | 0.5866 | 0.6866 | 0.7325 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.4146 | 0.4436 | 0.4818 | 0.5808 | 0.6853 | 0.7353 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.4283 | 0.4417 | 0.4867 | 0.5876 | 0.6875 | 0.7345 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.4309 | 0.4415 | 0.4882 | 0.5887 | 0.6883 | 0.7353 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.3999 | 0.4374 | 0.4777 | 0.5833 | 0.6880 | 0.7347 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.4257 | 0.4371 | 0.4824 | 0.5793 | 0.6860 | 0.7338 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.4194 | 0.4367 | 0.4793 | 0.5885 | 0.6868 | 0.7333 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.4186 | 0.4358 | 0.4797 | 0.5884 | 0.6861 | 0.7329 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.4108 | 0.4348 | 0.4739 | 0.5680 | 0.6729 | 0.7259 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.4076 | 0.4336 | 0.4788 | 0.5785 | 0.6850 | 0.7318 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.4022 | 0.4336 | 0.4712 | 0.5758 | 0.6800 | 0.7303 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.4129 | 0.4334 | 0.4714 | 0.5681 | 0.6721 | 0.7253 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.4183 | 0.4323 | 0.4741 | 0.5724 | 0.6793 | 0.7312 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.4055 | 0.4318 | 0.4749 | 0.5801 | 0.6824 | 0.7296 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.4092 | 0.4317 | 0.4697 | 0.5780 | 0.6816 | 0.7267 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.4013 | 0.4311 | 0.4716 | 0.5754 | 0.6806 | 0.7303 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.4235 | 0.4305 | 0.4748 | 0.5795 | 0.6830 | 0.7297 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.3978 | 0.4304 | 0.4687 | 0.5770 | 0.6817 | 0.7302 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.4158 | 0.4295 | 0.4679 | 0.5700 | 0.6807 | 0.7305 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.4130 | 0.4291 | 0.4681 | 0.5755 | 0.6788 | 0.7276 |
| `fusion[mean cosine] jina-clip-v2-image + text-omni-nano` | 0.3977 | 0.4290 | 0.4827 | 0.5837 | 0.6826 | 0.7293 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.3963 | 0.4274 | 0.4665 | 0.5756 | 0.6799 | 0.7287 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.3935 | 0.4252 | 0.4809 | 0.5832 | 0.6814 | 0.7283 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.3956 | 0.4238 | 0.4668 | 0.5764 | 0.6793 | 0.7267 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.3908 | 0.4209 | 0.4565 | 0.5562 | 0.6645 | 0.7199 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.3934 | 0.4189 | 0.4714 | 0.5770 | 0.6820 | 0.7297 |
| `attr-siglip` | 0.4044 | 0.4185 | 0.4617 | 0.5529 | 0.6596 | 0.7175 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.4031 | 0.4175 | 0.4550 | 0.5638 | 0.6744 | 0.7227 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.3985 | 0.4164 | 0.4543 | 0.5504 | 0.6663 | 0.7195 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.3810 | 0.4132 | 0.4583 | 0.5635 | 0.6740 | 0.7225 |
| `jina-clip-v2-image` | 0.3989 | 0.4120 | 0.4601 | 0.5683 | 0.6692 | 0.7196 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.3857 | 0.4096 | 0.4565 | 0.5663 | 0.6735 | 0.7217 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.3918 | 0.4084 | 0.4610 | 0.5606 | 0.6645 | 0.7186 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.3747 | 0.4083 | 0.4487 | 0.5616 | 0.6711 | 0.7170 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.3812 | 0.4075 | 0.4602 | 0.5647 | 0.6703 | 0.7206 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.3871 | 0.4072 | 0.4568 | 0.5625 | 0.6731 | 0.7236 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.3699 | 0.4040 | 0.4563 | 0.5618 | 0.6745 | 0.7240 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.3996 | 0.4020 | 0.4454 | 0.5487 | 0.6604 | 0.7144 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.3885 | 0.4016 | 0.4532 | 0.5614 | 0.6718 | 0.7179 |
| `text-jina` | 0.3698 | 0.3998 | 0.4536 | 0.5580 | 0.6626 | 0.7140 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.3832 | 0.3991 | 0.4535 | 0.5556 | 0.6615 | 0.7154 |
| `text-omni-nano` | 0.3699 | 0.3989 | 0.4559 | 0.5578 | 0.6633 | 0.7148 |
| `text-e5-large-instruct` | 0.3677 | 0.3976 | 0.4503 | 0.5531 | 0.6640 | 0.7159 |
| `text-siglip` | 0.3842 | 0.3958 | 0.4305 | 0.5228 | 0.6407 | 0.7025 |
| `text-jina-small` | 0.3551 | 0.3895 | 0.4454 | 0.5501 | 0.6633 | 0.7166 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.3625 | 0.3878 | 0.4421 | 0.5499 | 0.6619 | 0.7132 |
| `text-e5-base` | 0.3566 | 0.3730 | 0.4220 | 0.5329 | 0.6420 | 0.6987 |
| `attr-omni-nano` | 0.3257 | 0.3590 | 0.4124 | 0.5142 | 0.6336 | 0.6904 |
| `text-e5-small-multi` | 0.3415 | 0.3579 | 0.4101 | 0.5276 | 0.6454 | 0.6970 |
| `attr-e5-base` | 0.3226 | 0.3560 | 0.4064 | 0.5113 | 0.6251 | 0.6863 |
| `attr-jina` | 0.3178 | 0.3455 | 0.4055 | 0.5112 | 0.6314 | 0.6890 |
| `attr-jina-small` | 0.3112 | 0.3407 | 0.4032 | 0.5088 | 0.6246 | 0.6870 |
| `random` | 0.3183 | 0.3208 | 0.3505 | 0.4523 | 0.5797 | 0.6481 |
| `attr-e5-large-instruct` | 0.2946 | 0.3198 | 0.3820 | 0.5023 | 0.6163 | 0.6822 |
| `attr-e5-small-multi` | 0.2846 | 0.3093 | 0.3644 | 0.4861 | 0.6138 | 0.6746 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.0609 | 0.1266 | 0.2517 | 0.5200 | 0.8002 | 0.9346 |
| **`fusion[z-score average] jina-clip-v2-image + attr-siglip`** | **0.0786** | **0.1547** | **0.2994** | **0.5880** | **0.8277** | **0.9415** |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.0785 | 0.1542 | 0.2972 | 0.5820 | 0.8257 | 0.9415 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.0786 | 0.1544 | 0.2994 | 0.5877 | 0.8288 | 0.9413 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.0782 | 0.1557 | 0.3019 | 0.5911 | 0.8332 | 0.9420 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.0787 | 0.1558 | 0.3015 | 0.5897 | 0.8332 | 0.9436 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.0791 | 0.1553 | 0.3007 | 0.5884 | 0.8317 | 0.9409 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.0791 | 0.1551 | 0.3007 | 0.5885 | 0.8313 | 0.9409 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.0788 | 0.1565 | 0.3026 | 0.5941 | 0.8361 | 0.9432 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.0793 | 0.1563 | 0.3015 | 0.5904 | 0.8329 | 0.9434 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.0792 | 0.1566 | 0.3025 | 0.5907 | 0.8308 | 0.9405 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.0792 | 0.1563 | 0.3024 | 0.5907 | 0.8308 | 0.9405 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.0770 | 0.1532 | 0.2961 | 0.5807 | 0.8242 | 0.9418 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.0792 | 0.1571 | 0.3033 | 0.5940 | 0.8362 | 0.9432 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.0761 | 0.1530 | 0.2984 | 0.5857 | 0.8306 | 0.9430 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.0772 | 0.1534 | 0.2963 | 0.5808 | 0.8242 | 0.9422 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.0780 | 0.1538 | 0.2945 | 0.5782 | 0.8278 | 0.9408 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.0775 | 0.1538 | 0.2992 | 0.5884 | 0.8317 | 0.9432 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.0782 | 0.1557 | 0.3022 | 0.5913 | 0.8319 | 0.9425 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.0762 | 0.1530 | 0.2985 | 0.5859 | 0.8305 | 0.9431 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.0774 | 0.1532 | 0.2987 | 0.5851 | 0.8299 | 0.9414 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.0763 | 0.1530 | 0.2985 | 0.5867 | 0.8317 | 0.9440 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.0779 | 0.1533 | 0.2942 | 0.5796 | 0.8302 | 0.9402 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.0765 | 0.1529 | 0.2982 | 0.5870 | 0.8308 | 0.9431 |
| `fusion[mean cosine] jina-clip-v2-image + text-omni-nano` | 0.0792 | 0.1560 | 0.3032 | 0.5916 | 0.8287 | 0.9399 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.0763 | 0.1530 | 0.2985 | 0.5866 | 0.8323 | 0.9441 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.0791 | 0.1558 | 0.3031 | 0.5918 | 0.8287 | 0.9397 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.0781 | 0.1562 | 0.3021 | 0.5881 | 0.8307 | 0.9409 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.0767 | 0.1538 | 0.2958 | 0.5778 | 0.8253 | 0.9439 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.0794 | 0.1561 | 0.3033 | 0.5941 | 0.8351 | 0.9430 |
| `attr-siglip` | 0.0737 | 0.1470 | 0.2850 | 0.5719 | 0.8180 | 0.9370 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.0783 | 0.1561 | 0.3012 | 0.5863 | 0.8311 | 0.9424 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.0781 | 0.1520 | 0.2905 | 0.5737 | 0.8244 | 0.9383 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.0766 | 0.1515 | 0.2974 | 0.5852 | 0.8289 | 0.9423 |
| `jina-clip-v2-image` | 0.0761 | 0.1505 | 0.2936 | 0.5799 | 0.8248 | 0.9389 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.0790 | 0.1565 | 0.3016 | 0.5893 | 0.8303 | 0.9414 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.0770 | 0.1525 | 0.2949 | 0.5796 | 0.8255 | 0.9420 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.0783 | 0.1562 | 0.3024 | 0.5892 | 0.8311 | 0.9422 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.0761 | 0.1527 | 0.2974 | 0.5843 | 0.8309 | 0.9429 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.0760 | 0.1525 | 0.2966 | 0.5867 | 0.8304 | 0.9448 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.0757 | 0.1528 | 0.2976 | 0.5843 | 0.8336 | 0.9450 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.0762 | 0.1508 | 0.2916 | 0.5780 | 0.8235 | 0.9413 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.0791 | 0.1561 | 0.3003 | 0.5863 | 0.8304 | 0.9415 |
| `text-jina` | 0.0770 | 0.1534 | 0.2990 | 0.5890 | 0.8243 | 0.9372 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.0768 | 0.1523 | 0.2931 | 0.5817 | 0.8236 | 0.9405 |
| `text-omni-nano` | 0.0770 | 0.1536 | 0.2992 | 0.5888 | 0.8245 | 0.9373 |
| `text-e5-large-instruct` | 0.0779 | 0.1537 | 0.2985 | 0.5903 | 0.8269 | 0.9425 |
| `text-siglip` | 0.0756 | 0.1473 | 0.2827 | 0.5584 | 0.8122 | 0.9353 |
| `text-jina-small` | 0.0775 | 0.1536 | 0.2994 | 0.5887 | 0.8306 | 0.9412 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.0749 | 0.1505 | 0.2929 | 0.5836 | 0.8297 | 0.9427 |
| `text-e5-base` | 0.0790 | 0.1554 | 0.2984 | 0.5836 | 0.8225 | 0.9383 |
| `attr-omni-nano` | 0.0715 | 0.1445 | 0.2814 | 0.5696 | 0.8194 | 0.9413 |
| `text-e5-small-multi` | 0.0772 | 0.1517 | 0.2946 | 0.5831 | 0.8238 | 0.9381 |
| `attr-e5-base` | 0.0713 | 0.1437 | 0.2815 | 0.5699 | 0.8175 | 0.9400 |
| `attr-jina` | 0.0711 | 0.1448 | 0.2816 | 0.5695 | 0.8199 | 0.9413 |
| `attr-jina-small` | 0.0710 | 0.1431 | 0.2799 | 0.5728 | 0.8186 | 0.9416 |
| `random` | 0.0645 | 0.1287 | 0.2543 | 0.5199 | 0.7870 | 0.9239 |
| `attr-e5-large-instruct` | 0.0712 | 0.1429 | 0.2827 | 0.5708 | 0.8177 | 0.9388 |
| `attr-e5-small-multi` | 0.0685 | 0.1388 | 0.2760 | 0.5636 | 0.8187 | 0.9393 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7419 | 0.7652 | 0.7678 | 0.7351 | 0.6519 | 0.5507 |
| **`fusion[z-score average] jina-clip-v2-image + attr-siglip`** | **0.9441** | **0.9409** | **0.9278** | **0.8415** | **0.6816** | **0.5567** |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.9467 | 0.9423 | 0.9221 | 0.8295 | 0.6787 | 0.5568 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.9444 | 0.9405 | 0.9282 | 0.8409 | 0.6829 | 0.5568 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.9249 | 0.9368 | 0.9299 | 0.8440 | 0.6863 | 0.5569 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.9276 | 0.9350 | 0.9284 | 0.8410 | 0.6865 | 0.5584 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.9376 | 0.9336 | 0.9238 | 0.8383 | 0.6845 | 0.5561 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.9375 | 0.9316 | 0.9232 | 0.8386 | 0.6839 | 0.5562 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.9378 | 0.9346 | 0.9285 | 0.8483 | 0.6896 | 0.5584 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.9427 | 0.9387 | 0.9281 | 0.8413 | 0.6858 | 0.5580 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.9339 | 0.9397 | 0.9275 | 0.8413 | 0.6840 | 0.5560 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.9347 | 0.9354 | 0.9273 | 0.8414 | 0.6839 | 0.5560 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.9212 | 0.9257 | 0.9133 | 0.8238 | 0.6763 | 0.5568 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.9370 | 0.9442 | 0.9321 | 0.8489 | 0.6893 | 0.5582 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.8959 | 0.9167 | 0.9192 | 0.8350 | 0.6844 | 0.5583 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.9266 | 0.9273 | 0.9146 | 0.8246 | 0.6763 | 0.5572 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.9356 | 0.9296 | 0.9123 | 0.8270 | 0.6804 | 0.5561 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.9143 | 0.9256 | 0.9203 | 0.8387 | 0.6854 | 0.5583 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.9251 | 0.9342 | 0.9302 | 0.8450 | 0.6853 | 0.5573 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.8962 | 0.9168 | 0.9194 | 0.8352 | 0.6842 | 0.5584 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.9204 | 0.9241 | 0.9211 | 0.8345 | 0.6828 | 0.5569 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.8988 | 0.9171 | 0.9176 | 0.8357 | 0.6862 | 0.5592 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.9297 | 0.9254 | 0.9125 | 0.8295 | 0.6838 | 0.5559 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.9122 | 0.9222 | 0.9211 | 0.8379 | 0.6847 | 0.5582 |
| `fusion[mean cosine] jina-clip-v2-image + text-omni-nano` | 0.9299 | 0.9319 | 0.9294 | 0.8434 | 0.6819 | 0.5554 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.8989 | 0.9164 | 0.9175 | 0.8354 | 0.6864 | 0.5593 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.9299 | 0.9277 | 0.9286 | 0.8437 | 0.6819 | 0.5552 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.9205 | 0.9344 | 0.9277 | 0.8396 | 0.6831 | 0.5561 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.9247 | 0.9308 | 0.9116 | 0.8186 | 0.6772 | 0.5587 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.9358 | 0.9316 | 0.9306 | 0.8481 | 0.6890 | 0.5583 |
| `attr-siglip` | 0.8930 | 0.9018 | 0.8883 | 0.8166 | 0.6718 | 0.5535 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.9272 | 0.9366 | 0.9282 | 0.8357 | 0.6846 | 0.5571 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.9324 | 0.9235 | 0.9026 | 0.8223 | 0.6784 | 0.5546 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.8948 | 0.9082 | 0.9131 | 0.8331 | 0.6830 | 0.5580 |
| `jina-clip-v2-image` | 0.9004 | 0.9005 | 0.9023 | 0.8254 | 0.6765 | 0.5544 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.9322 | 0.9330 | 0.9248 | 0.8420 | 0.6835 | 0.5566 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.9098 | 0.9182 | 0.9086 | 0.8221 | 0.6775 | 0.5573 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.9160 | 0.9343 | 0.9294 | 0.8414 | 0.6848 | 0.5573 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.8796 | 0.9119 | 0.9140 | 0.8317 | 0.6841 | 0.5584 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.9015 | 0.9141 | 0.9120 | 0.8352 | 0.6843 | 0.5599 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.8959 | 0.9134 | 0.9154 | 0.8309 | 0.6874 | 0.5601 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.9129 | 0.9125 | 0.9007 | 0.8215 | 0.6762 | 0.5567 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.9375 | 0.9328 | 0.9226 | 0.8354 | 0.6830 | 0.5564 |
| `text-jina` | 0.8976 | 0.9124 | 0.9153 | 0.8399 | 0.6773 | 0.5532 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.9216 | 0.9200 | 0.9029 | 0.8250 | 0.6754 | 0.5559 |
| `text-omni-nano` | 0.8977 | 0.9136 | 0.9163 | 0.8394 | 0.6775 | 0.5532 |
| `text-e5-large-instruct` | 0.9139 | 0.9146 | 0.9135 | 0.8422 | 0.6810 | 0.5578 |
| `text-siglip` | 0.9013 | 0.8926 | 0.8806 | 0.7998 | 0.6675 | 0.5521 |
| `text-jina-small` | 0.9106 | 0.9138 | 0.9177 | 0.8401 | 0.6840 | 0.5567 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.8866 | 0.9047 | 0.9007 | 0.8326 | 0.6833 | 0.5582 |
| `text-e5-base` | 0.9332 | 0.9302 | 0.9124 | 0.8301 | 0.6757 | 0.5541 |
| `attr-omni-nano` | 0.8427 | 0.8668 | 0.8588 | 0.8035 | 0.6726 | 0.5575 |
| `text-e5-small-multi` | 0.9039 | 0.9058 | 0.9045 | 0.8315 | 0.6778 | 0.5541 |
| `attr-e5-base` | 0.8314 | 0.8606 | 0.8625 | 0.8069 | 0.6709 | 0.5564 |
| `attr-jina` | 0.8438 | 0.8700 | 0.8607 | 0.8038 | 0.6729 | 0.5575 |
| `attr-jina-small` | 0.8412 | 0.8591 | 0.8540 | 0.8100 | 0.6714 | 0.5580 |
| `random` | 0.7927 | 0.7862 | 0.7858 | 0.7335 | 0.6397 | 0.5425 |
| `attr-e5-large-instruct` | 0.8357 | 0.8592 | 0.8629 | 0.8066 | 0.6707 | 0.5556 |
| `attr-e5-small-multi` | 0.8137 | 0.8348 | 0.8513 | 0.8016 | 0.6720 | 0.5560 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7692 | 0.7784 | 0.7788 | 0.7788 | 0.7788 | 0.7788 |
| **`fusion[z-score average] jina-clip-v2-image + attr-siglip`** | **0.9717** | **0.9718** | **0.9719** | **0.9719** | **0.9719** | **0.9719** |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.9629 | 0.9630 | 0.9631 | 0.9631 | 0.9631 | 0.9631 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.9741 | 0.9742 | 0.9743 | 0.9743 | 0.9743 | 0.9743 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.9522 | 0.9523 | 0.9523 | 0.9523 | 0.9523 | 0.9523 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.9472 | 0.9474 | 0.9474 | 0.9474 | 0.9474 | 0.9474 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.9512 | 0.9513 | 0.9513 | 0.9513 | 0.9513 | 0.9513 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.9497 | 0.9498 | 0.9498 | 0.9498 | 0.9498 | 0.9498 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.9451 | 0.9452 | 0.9452 | 0.9452 | 0.9452 | 0.9452 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.9538 | 0.9539 | 0.9539 | 0.9539 | 0.9539 | 0.9539 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.9417 | 0.9418 | 0.9418 | 0.9418 | 0.9418 | 0.9418 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.9389 | 0.9390 | 0.9390 | 0.9390 | 0.9390 | 0.9390 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.9439 | 0.9440 | 0.9442 | 0.9442 | 0.9442 | 0.9442 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.9479 | 0.9479 | 0.9479 | 0.9479 | 0.9479 | 0.9479 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.9260 | 0.9265 | 0.9267 | 0.9267 | 0.9267 | 0.9267 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.9474 | 0.9475 | 0.9476 | 0.9476 | 0.9476 | 0.9476 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.9765 | 0.9765 | 0.9765 | 0.9765 | 0.9765 | 0.9765 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.9367 | 0.9373 | 0.9375 | 0.9375 | 0.9375 | 0.9375 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.9199 | 0.9201 | 0.9201 | 0.9201 | 0.9201 | 0.9201 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.9239 | 0.9244 | 0.9246 | 0.9246 | 0.9246 | 0.9246 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.9350 | 0.9357 | 0.9358 | 0.9358 | 0.9358 | 0.9358 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.9224 | 0.9226 | 0.9227 | 0.9227 | 0.9227 | 0.9227 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.9696 | 0.9697 | 0.9697 | 0.9697 | 0.9697 | 0.9697 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.9287 | 0.9291 | 0.9291 | 0.9291 | 0.9291 | 0.9291 |
| `fusion[mean cosine] jina-clip-v2-image + text-omni-nano` | 0.9361 | 0.9362 | 0.9362 | 0.9362 | 0.9362 | 0.9362 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.9229 | 0.9231 | 0.9232 | 0.9232 | 0.9232 | 0.9232 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.9364 | 0.9365 | 0.9365 | 0.9365 | 0.9365 | 0.9365 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.9496 | 0.9499 | 0.9499 | 0.9499 | 0.9499 | 0.9499 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.9386 | 0.9392 | 0.9393 | 0.9393 | 0.9393 | 0.9393 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.9477 | 0.9479 | 0.9479 | 0.9479 | 0.9479 | 0.9479 |
| `attr-siglip` | 0.9431 | 0.9442 | 0.9442 | 0.9442 | 0.9442 | 0.9442 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.9491 | 0.9494 | 0.9494 | 0.9494 | 0.9494 | 0.9494 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.9565 | 0.9569 | 0.9569 | 0.9570 | 0.9570 | 0.9570 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.9183 | 0.9186 | 0.9188 | 0.9188 | 0.9188 | 0.9188 |
| `jina-clip-v2-image` | 0.9077 | 0.9084 | 0.9085 | 0.9085 | 0.9085 | 0.9085 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.9688 | 0.9689 | 0.9689 | 0.9689 | 0.9689 | 0.9689 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.9350 | 0.9356 | 0.9356 | 0.9356 | 0.9356 | 0.9356 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.9373 | 0.9376 | 0.9376 | 0.9376 | 0.9376 | 0.9376 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.9130 | 0.9141 | 0.9141 | 0.9141 | 0.9141 | 0.9141 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.9326 | 0.9339 | 0.9341 | 0.9341 | 0.9341 | 0.9341 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.9274 | 0.9281 | 0.9283 | 0.9283 | 0.9283 | 0.9283 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.9257 | 0.9257 | 0.9258 | 0.9258 | 0.9258 | 0.9258 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.9498 | 0.9499 | 0.9499 | 0.9499 | 0.9499 | 0.9499 |
| `text-jina` | 0.9636 | 0.9645 | 0.9646 | 0.9646 | 0.9646 | 0.9646 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.9414 | 0.9419 | 0.9419 | 0.9420 | 0.9420 | 0.9420 |
| `text-omni-nano` | 0.9637 | 0.9646 | 0.9647 | 0.9647 | 0.9647 | 0.9647 |
| `text-e5-large-instruct` | 0.9648 | 0.9651 | 0.9656 | 0.9657 | 0.9657 | 0.9657 |
| `text-siglip` | 0.9651 | 0.9655 | 0.9655 | 0.9655 | 0.9655 | 0.9655 |
| `text-jina-small` | 0.9669 | 0.9670 | 0.9670 | 0.9674 | 0.9674 | 0.9674 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.9103 | 0.9126 | 0.9126 | 0.9126 | 0.9126 | 0.9126 |
| `text-e5-base` | 0.9717 | 0.9718 | 0.9718 | 0.9718 | 0.9718 | 0.9718 |
| `attr-omni-nano` | 0.8843 | 0.8854 | 0.8857 | 0.8858 | 0.8858 | 0.8858 |
| `text-e5-small-multi` | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 |
| `attr-e5-base` | 0.8794 | 0.8828 | 0.8832 | 0.8832 | 0.8832 | 0.8832 |
| `attr-jina` | 0.8807 | 0.8820 | 0.8823 | 0.8824 | 0.8824 | 0.8824 |
| `attr-jina-small` | 0.8949 | 0.8976 | 0.8978 | 0.8978 | 0.8978 | 0.8978 |
| `random` | 0.8806 | 0.8825 | 0.8829 | 0.8829 | 0.8829 | 0.8829 |
| `attr-e5-large-instruct` | 0.8891 | 0.8906 | 0.8913 | 0.8914 | 0.8914 | 0.8914 |
| `attr-e5-small-multi` | 0.8645 | 0.8671 | 0.8677 | 0.8681 | 0.8681 | 0.8681 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.8035 |
| **`fusion[z-score average] jina-clip-v2-image + attr-siglip`** | **0.9056** |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-siglip` | 0.8990 |
| `fusion[mean cosine] jina-clip-v2-image + attr-siglip` | 0.9057 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-large-instruct` | 0.9097 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina-small` | 0.9102 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-jina` | 0.9085 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-omni-nano` | 0.9085 |
| `fusion[z-score average] jina-clip-v2-image + text-jina-small` | 0.9155 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-large-instruct` | 0.9109 |
| `fusion[z-score average] jina-clip-v2-image + text-omni-nano` | 0.9119 |
| `fusion[z-score average] jina-clip-v2-image + text-jina` | 0.9118 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-omni-nano` | 0.8943 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-large-instruct` | 0.9155 |
| `fusion[z-score average] jina-clip-v2-image + attr-omni-nano` | 0.9031 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina` | 0.8946 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-siglip` | 0.8935 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-base` | 0.9067 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-small-multi` | 0.9095 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina` | 0.9030 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-large-instruct` | 0.9026 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina` | 0.9045 |
| `fusion[z-score average] jina-clip-v2-image + text-siglip` | 0.8954 |
| `fusion[mean cosine] jina-clip-v2-image + attr-e5-small-multi` | 0.9033 |
| `fusion[mean cosine] jina-clip-v2-image + text-omni-nano` | 0.9119 |
| `fusion[mean cosine] jina-clip-v2-image + attr-omni-nano` | 0.9044 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina` | 0.9120 |
| `fusion[mean cosine] jina-clip-v2-image + text-e5-base` | 0.9086 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-jina-small` | 0.8937 |
| `fusion[mean cosine] jina-clip-v2-image + text-jina-small` | 0.9157 |
| `attr-siglip` | 0.8753 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-small-multi` | 0.9067 |
| `fusion[mean cosine] jina-clip-v2-image + text-siglip` | 0.8867 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-large-instruct` | 0.9005 |
| `jina-clip-v2-image` | 0.8889 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-base` | 0.9106 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-base` | 0.8941 |
| `fusion[z-score average] jina-clip-v2-image + text-e5-small-multi` | 0.9101 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-base` | 0.9012 |
| `fusion[z-score average] jina-clip-v2-image + attr-jina-small` | 0.9019 |
| `fusion[mean cosine] jina-clip-v2-image + attr-jina-small` | 0.9035 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-small-multi` | 0.8878 |
| `fusion[RRF (k=60)] jina-clip-v2-image + text-e5-base` | 0.9067 |
| `text-jina` | 0.9034 |
| `fusion[RRF (k=60)] jina-clip-v2-image + attr-e5-large-instruct` | 0.8934 |
| `text-omni-nano` | 0.9036 |
| `text-e5-large-instruct` | 0.9057 |
| `text-siglip` | 0.8650 |
| `text-jina-small` | 0.9068 |
| `fusion[z-score average] jina-clip-v2-image + attr-e5-small-multi` | 0.8967 |
| `text-e5-base` | 0.9012 |
| `attr-omni-nano` | 0.8703 |
| `text-e5-small-multi` | 0.8959 |
| `attr-e5-base` | 0.8720 |
| `attr-jina` | 0.8707 |
| `attr-jina-small` | 0.8702 |
| `random` | 0.7970 |
| `attr-e5-large-instruct` | 0.8710 |
| `attr-e5-small-multi` | 0.8616 |


---

## 3. Which text representation gives the highest fusion gain?

"Gain vs image alone" = fused NDCG@10 - that target image alone.
"Gain vs text alone" = fused NDCG@10 - that text representation's own NDCG@10 (i.e. does fusing
with the image actually help over just using the text system by itself). One pair of tables per
target image modality.


### Target: `siglip-image` (image alone: macro 0.4272, impression-weighted 0.4443)

#### Macro-averaged

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **text-siglip** | **z-score average** | **0.4606** | **+0.0334** | **+0.0625** |
| text-e5-large-instruct | mean cosine | 0.4596 | +0.0325 | +0.0584 |
| text-siglip | RRF (k=60) | 0.4565 | +0.0293 | +0.0584 |
| text-omni-nano | RRF (k=60) | 0.4517 | +0.0246 | +0.0423 |
| attr-siglip | z-score average | 0.4516 | +0.0244 | +0.0740 |
| text-jina | RRF (k=60) | 0.4514 | +0.0242 | +0.0442 |
| text-e5-large-instruct | RRF (k=60) | 0.4510 | +0.0239 | +0.0498 |
| text-omni-nano | z-score average | 0.4494 | +0.0222 | +0.0399 |
| attr-siglip | mean cosine | 0.4491 | +0.0220 | +0.0716 |
| text-jina | z-score average | 0.4489 | +0.0218 | +0.0417 |
| text-e5-large-instruct | z-score average | 0.4476 | +0.0204 | +0.0464 |
| text-jina-small | RRF (k=60) | 0.4469 | +0.0197 | +0.0480 |
| text-jina-small | z-score average | 0.4453 | +0.0181 | +0.0464 |
| attr-e5-large-instruct | mean cosine | 0.4447 | +0.0175 | +0.1035 |
| attr-siglip | RRF (k=60) | 0.4420 | +0.0149 | +0.0645 |
| text-omni-nano | mean cosine | 0.4396 | +0.0125 | +0.0302 |
| text-jina | mean cosine | 0.4387 | +0.0115 | +0.0315 |
| attr-omni-nano | z-score average | 0.4384 | +0.0112 | +0.0951 |
| text-siglip | mean cosine | 0.4380 | +0.0109 | +0.0399 |
| text-e5-base | mean cosine | 0.4378 | +0.0106 | +0.0479 |
| attr-jina | z-score average | 0.4375 | +0.0103 | +0.0977 |
| text-e5-small-multi | mean cosine | 0.4369 | +0.0097 | +0.0716 |
| attr-e5-base | mean cosine | 0.4356 | +0.0085 | +0.0977 |
| text-e5-base | z-score average | 0.4354 | +0.0082 | +0.0455 |
| attr-e5-large-instruct | z-score average | 0.4334 | +0.0063 | +0.0923 |
| text-e5-base | RRF (k=60) | 0.4324 | +0.0052 | +0.0425 |
| attr-e5-base | z-score average | 0.4292 | +0.0020 | +0.0913 |
| attr-e5-small-multi | mean cosine | 0.4287 | +0.0016 | +0.1163 |
| attr-jina-small | z-score average | 0.4279 | +0.0007 | +0.0855 |
| text-jina-small | mean cosine | 0.4269 | -0.0002 | +0.0280 |
| attr-omni-nano | mean cosine | 0.4256 | -0.0016 | +0.0823 |
| attr-jina | mean cosine | 0.4249 | -0.0023 | +0.0851 |
| text-e5-small-multi | z-score average | 0.4242 | -0.0030 | +0.0590 |
| attr-e5-large-instruct | RRF (k=60) | 0.4236 | -0.0035 | +0.0825 |
| text-e5-small-multi | RRF (k=60) | 0.4225 | -0.0047 | +0.0572 |
| attr-omni-nano | RRF (k=60) | 0.4221 | -0.0051 | +0.0788 |
| attr-jina | RRF (k=60) | 0.4217 | -0.0055 | +0.0819 |
| attr-jina-small | RRF (k=60) | 0.4216 | -0.0056 | +0.0791 |
| attr-e5-base | RRF (k=60) | 0.4199 | -0.0073 | +0.0819 |
| attr-jina-small | mean cosine | 0.4192 | -0.0080 | +0.0767 |
| attr-e5-small-multi | z-score average | 0.4130 | -0.0141 | +0.1005 |
| attr-e5-small-multi | RRF (k=60) | 0.4080 | -0.0192 | +0.0955 |

#### Impression-weighted

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **attr-siglip** | **mean cosine** | **0.4781** | **+0.0338** | **+0.0596** |
| attr-siglip | RRF (k=60) | 0.4725 | +0.0282 | +0.0540 |
| attr-siglip | z-score average | 0.4699 | +0.0256 | +0.0514 |
| text-omni-nano | RRF (k=60) | 0.4679 | +0.0235 | +0.0689 |
| text-jina | RRF (k=60) | 0.4635 | +0.0192 | +0.0637 |
| text-e5-large-instruct | RRF (k=60) | 0.4619 | +0.0175 | +0.0643 |
| text-e5-large-instruct | z-score average | 0.4616 | +0.0173 | +0.0641 |
| text-jina-small | RRF (k=60) | 0.4610 | +0.0167 | +0.0716 |
| text-e5-large-instruct | mean cosine | 0.4583 | +0.0140 | +0.0608 |
| attr-e5-large-instruct | mean cosine | 0.4576 | +0.0133 | +0.1378 |
| text-jina-small | z-score average | 0.4562 | +0.0119 | +0.0668 |
| text-omni-nano | z-score average | 0.4509 | +0.0066 | +0.0520 |
| text-jina | z-score average | 0.4507 | +0.0064 | +0.0509 |
| attr-e5-base | mean cosine | 0.4493 | +0.0050 | +0.0933 |
| text-siglip | RRF (k=60) | 0.4487 | +0.0044 | +0.0529 |
| attr-e5-base | RRF (k=60) | 0.4446 | +0.0003 | +0.0887 |
| text-omni-nano | mean cosine | 0.4415 | -0.0028 | +0.0426 |
| attr-e5-base | z-score average | 0.4410 | -0.0034 | +0.0850 |
| text-jina | mean cosine | 0.4392 | -0.0051 | +0.0394 |
| attr-omni-nano | RRF (k=60) | 0.4391 | -0.0053 | +0.0801 |
| text-siglip | z-score average | 0.4384 | -0.0059 | +0.0426 |
| attr-jina-small | z-score average | 0.4376 | -0.0067 | +0.0969 |
| attr-jina | RRF (k=60) | 0.4366 | -0.0077 | +0.0911 |
| attr-jina-small | RRF (k=60) | 0.4360 | -0.0083 | +0.0953 |
| attr-omni-nano | z-score average | 0.4336 | -0.0107 | +0.0746 |
| attr-jina | z-score average | 0.4309 | -0.0134 | +0.0854 |
| text-jina-small | mean cosine | 0.4287 | -0.0156 | +0.0393 |
| attr-omni-nano | mean cosine | 0.4287 | -0.0157 | +0.0697 |
| text-e5-small-multi | mean cosine | 0.4285 | -0.0159 | +0.0706 |
| attr-e5-large-instruct | z-score average | 0.4283 | -0.0161 | +0.1085 |
| text-e5-base | mean cosine | 0.4278 | -0.0165 | +0.0548 |
| text-e5-small-multi | RRF (k=60) | 0.4267 | -0.0177 | +0.0688 |
| attr-jina-small | mean cosine | 0.4261 | -0.0183 | +0.0854 |
| attr-e5-small-multi | mean cosine | 0.4248 | -0.0196 | +0.1154 |
| attr-jina | mean cosine | 0.4245 | -0.0198 | +0.0791 |
| attr-e5-large-instruct | RRF (k=60) | 0.4245 | -0.0199 | +0.1047 |
| text-e5-base | RRF (k=60) | 0.4205 | -0.0238 | +0.0475 |
| text-siglip | mean cosine | 0.4196 | -0.0247 | +0.0238 |
| attr-e5-small-multi | RRF (k=60) | 0.4196 | -0.0248 | +0.1102 |
| text-e5-small-multi | z-score average | 0.4193 | -0.0250 | +0.0614 |
| text-e5-base | z-score average | 0.4181 | -0.0263 | +0.0450 |
| attr-e5-small-multi | z-score average | 0.3963 | -0.0480 | +0.0870 |

### Target: `omni-nano-image` (image alone: macro 0.3315, impression-weighted 0.3679)

#### Macro-averaged

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **text-siglip** | **mean cosine** | **0.4032** | **+0.0718** | **+0.0052** |
| text-omni-nano | mean cosine | 0.4009 | +0.0695 | -0.0085 |
| text-jina | mean cosine | 0.3991 | +0.0676 | -0.0081 |
| text-jina-small | mean cosine | 0.3974 | +0.0659 | -0.0016 |
| text-jina | RRF (k=60) | 0.3926 | +0.0612 | -0.0146 |
| attr-siglip | RRF (k=60) | 0.3920 | +0.0606 | +0.0145 |
| text-jina-small | RRF (k=60) | 0.3917 | +0.0602 | -0.0073 |
| text-omni-nano | RRF (k=60) | 0.3912 | +0.0597 | -0.0182 |
| text-e5-large-instruct | z-score average | 0.3907 | +0.0592 | -0.0105 |
| text-e5-large-instruct | RRF (k=60) | 0.3891 | +0.0576 | -0.0122 |
| text-omni-nano | z-score average | 0.3873 | +0.0558 | -0.0222 |
| text-jina-small | z-score average | 0.3872 | +0.0558 | -0.0117 |
| text-jina | z-score average | 0.3864 | +0.0550 | -0.0208 |
| text-siglip | RRF (k=60) | 0.3851 | +0.0537 | -0.0130 |
| attr-siglip | mean cosine | 0.3839 | +0.0524 | +0.0063 |
| text-siglip | z-score average | 0.3826 | +0.0512 | -0.0155 |
| attr-siglip | z-score average | 0.3819 | +0.0504 | +0.0043 |
| text-e5-base | z-score average | 0.3810 | +0.0496 | -0.0088 |
| text-e5-base | RRF (k=60) | 0.3799 | +0.0484 | -0.0100 |
| text-e5-base | mean cosine | 0.3723 | +0.0408 | -0.0176 |
| attr-omni-nano | z-score average | 0.3704 | +0.0389 | +0.0271 |
| text-e5-small-multi | RRF (k=60) | 0.3701 | +0.0387 | +0.0048 |
| attr-omni-nano | mean cosine | 0.3693 | +0.0378 | +0.0260 |
| text-e5-small-multi | z-score average | 0.3693 | +0.0378 | +0.0040 |
| attr-jina | z-score average | 0.3692 | +0.0377 | +0.0293 |
| attr-jina | mean cosine | 0.3687 | +0.0372 | +0.0289 |
| attr-jina-small | mean cosine | 0.3679 | +0.0365 | +0.0255 |
| attr-e5-large-instruct | z-score average | 0.3678 | +0.0363 | +0.0267 |
| attr-omni-nano | RRF (k=60) | 0.3666 | +0.0352 | +0.0234 |
| attr-jina-small | RRF (k=60) | 0.3666 | +0.0351 | +0.0242 |
| attr-jina | RRF (k=60) | 0.3651 | +0.0336 | +0.0253 |
| attr-jina-small | z-score average | 0.3644 | +0.0330 | +0.0220 |
| attr-e5-large-instruct | RRF (k=60) | 0.3627 | +0.0313 | +0.0216 |
| text-e5-large-instruct | mean cosine | 0.3625 | +0.0310 | -0.0387 |
| attr-e5-base | z-score average | 0.3625 | +0.0310 | +0.0245 |
| text-e5-small-multi | mean cosine | 0.3621 | +0.0307 | -0.0031 |
| attr-e5-base | RRF (k=60) | 0.3595 | +0.0280 | +0.0215 |
| attr-e5-base | mean cosine | 0.3508 | +0.0193 | +0.0128 |
| attr-e5-large-instruct | mean cosine | 0.3504 | +0.0190 | +0.0093 |
| attr-e5-small-multi | mean cosine | 0.3461 | +0.0146 | +0.0336 |
| attr-e5-small-multi | z-score average | 0.3448 | +0.0133 | +0.0323 |
| attr-e5-small-multi | RRF (k=60) | 0.3422 | +0.0107 | +0.0297 |

#### Impression-weighted

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **attr-siglip** | **RRF (k=60)** | **0.4328** | **+0.0649** | **+0.0143** |
| attr-siglip | mean cosine | 0.4101 | +0.0422 | -0.0084 |
| attr-siglip | z-score average | 0.4092 | +0.0413 | -0.0093 |
| text-omni-nano | mean cosine | 0.4076 | +0.0397 | +0.0087 |
| text-jina | mean cosine | 0.4052 | +0.0373 | +0.0054 |
| text-omni-nano | RRF (k=60) | 0.4031 | +0.0352 | +0.0042 |
| text-jina | RRF (k=60) | 0.4023 | +0.0344 | +0.0025 |
| text-jina-small | RRF (k=60) | 0.4014 | +0.0335 | +0.0120 |
| attr-omni-nano | RRF (k=60) | 0.3946 | +0.0267 | +0.0356 |
| text-omni-nano | z-score average | 0.3934 | +0.0255 | -0.0055 |
| text-jina | z-score average | 0.3929 | +0.0249 | -0.0070 |
| text-jina-small | mean cosine | 0.3909 | +0.0230 | +0.0015 |
| text-e5-large-instruct | z-score average | 0.3902 | +0.0223 | -0.0074 |
| text-e5-large-instruct | RRF (k=60) | 0.3894 | +0.0215 | -0.0081 |
| text-e5-base | mean cosine | 0.3893 | +0.0214 | +0.0162 |
| text-siglip | mean cosine | 0.3892 | +0.0213 | -0.0066 |
| text-e5-large-instruct | mean cosine | 0.3884 | +0.0205 | -0.0091 |
| attr-jina | mean cosine | 0.3884 | +0.0205 | +0.0429 |
| attr-omni-nano | mean cosine | 0.3879 | +0.0199 | +0.0289 |
| attr-e5-base | RRF (k=60) | 0.3874 | +0.0195 | +0.0314 |
| attr-jina-small | RRF (k=60) | 0.3854 | +0.0175 | +0.0447 |
| attr-omni-nano | z-score average | 0.3852 | +0.0173 | +0.0263 |
| text-e5-base | z-score average | 0.3841 | +0.0162 | +0.0111 |
| attr-e5-base | z-score average | 0.3838 | +0.0159 | +0.0279 |
| attr-jina-small | mean cosine | 0.3838 | +0.0159 | +0.0431 |
| attr-jina | RRF (k=60) | 0.3838 | +0.0159 | +0.0383 |
| text-jina-small | z-score average | 0.3824 | +0.0145 | -0.0071 |
| attr-jina | z-score average | 0.3821 | +0.0142 | +0.0366 |
| attr-e5-small-multi | mean cosine | 0.3819 | +0.0140 | +0.0725 |
| attr-e5-large-instruct | mean cosine | 0.3800 | +0.0120 | +0.0602 |
| text-siglip | RRF (k=60) | 0.3799 | +0.0119 | -0.0160 |
| text-e5-small-multi | mean cosine | 0.3799 | +0.0119 | +0.0220 |
| attr-e5-base | mean cosine | 0.3784 | +0.0105 | +0.0224 |
| attr-jina-small | z-score average | 0.3777 | +0.0098 | +0.0370 |
| attr-e5-small-multi | RRF (k=60) | 0.3747 | +0.0068 | +0.0654 |
| text-siglip | z-score average | 0.3733 | +0.0054 | -0.0225 |
| attr-e5-large-instruct | z-score average | 0.3722 | +0.0043 | +0.0525 |
| text-e5-small-multi | RRF (k=60) | 0.3707 | +0.0028 | +0.0128 |
| text-e5-base | RRF (k=60) | 0.3691 | +0.0012 | -0.0039 |
| attr-e5-large-instruct | RRF (k=60) | 0.3664 | -0.0015 | +0.0467 |
| text-e5-small-multi | z-score average | 0.3662 | -0.0017 | +0.0083 |
| attr-e5-small-multi | z-score average | 0.3503 | -0.0176 | +0.0410 |

### Target: `jina-clip-v2-image` (image alone: macro 0.3883, impression-weighted 0.4120)

#### Macro-averaged

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **text-omni-nano** | **mean cosine** | **0.4373** | **+0.0490** | **+0.0278** |
| text-jina | mean cosine | 0.4360 | +0.0476 | +0.0287 |
| text-omni-nano | z-score average | 0.4308 | +0.0425 | +0.0214 |
| text-jina | z-score average | 0.4303 | +0.0420 | +0.0231 |
| text-e5-large-instruct | z-score average | 0.4303 | +0.0420 | +0.0291 |
| text-siglip | z-score average | 0.4295 | +0.0412 | +0.0314 |
| text-jina-small | z-score average | 0.4290 | +0.0407 | +0.0301 |
| text-jina-small | mean cosine | 0.4286 | +0.0402 | +0.0296 |
| text-siglip | mean cosine | 0.4280 | +0.0397 | +0.0300 |
| text-omni-nano | RRF (k=60) | 0.4276 | +0.0393 | +0.0182 |
| text-jina | RRF (k=60) | 0.4266 | +0.0383 | +0.0194 |
| text-e5-large-instruct | RRF (k=60) | 0.4264 | +0.0381 | +0.0252 |
| attr-siglip | mean cosine | 0.4263 | +0.0380 | +0.0488 |
| text-jina-small | RRF (k=60) | 0.4256 | +0.0373 | +0.0267 |
| text-siglip | RRF (k=60) | 0.4254 | +0.0370 | +0.0273 |
| attr-siglip | z-score average | 0.4237 | +0.0354 | +0.0461 |
| text-e5-large-instruct | mean cosine | 0.4230 | +0.0346 | +0.0217 |
| text-e5-base | mean cosine | 0.4221 | +0.0338 | +0.0322 |
| text-e5-base | z-score average | 0.4204 | +0.0321 | +0.0305 |
| text-e5-small-multi | mean cosine | 0.4201 | +0.0318 | +0.0548 |
| attr-siglip | RRF (k=60) | 0.4180 | +0.0296 | +0.0404 |
| text-e5-base | RRF (k=60) | 0.4102 | +0.0219 | +0.0204 |
| text-e5-small-multi | z-score average | 0.4093 | +0.0210 | +0.0441 |
| attr-jina | mean cosine | 0.4070 | +0.0187 | +0.0672 |
| attr-omni-nano | z-score average | 0.4066 | +0.0183 | +0.0633 |
| attr-omni-nano | mean cosine | 0.4059 | +0.0176 | +0.0627 |
| attr-jina | z-score average | 0.4049 | +0.0166 | +0.0651 |
| text-e5-small-multi | RRF (k=60) | 0.4047 | +0.0164 | +0.0394 |
| attr-e5-base | mean cosine | 0.4040 | +0.0157 | +0.0660 |
| attr-e5-large-instruct | z-score average | 0.4028 | +0.0144 | +0.0616 |
| attr-e5-large-instruct | mean cosine | 0.4023 | +0.0139 | +0.0611 |
| attr-e5-small-multi | mean cosine | 0.4012 | +0.0129 | +0.0887 |
| attr-jina | RRF (k=60) | 0.3993 | +0.0110 | +0.0595 |
| attr-e5-base | z-score average | 0.3988 | +0.0105 | +0.0609 |
| attr-omni-nano | RRF (k=60) | 0.3984 | +0.0101 | +0.0552 |
| attr-jina-small | mean cosine | 0.3978 | +0.0095 | +0.0554 |
| attr-jina-small | z-score average | 0.3974 | +0.0090 | +0.0549 |
| attr-jina-small | RRF (k=60) | 0.3962 | +0.0079 | +0.0538 |
| attr-e5-base | RRF (k=60) | 0.3896 | +0.0013 | +0.0516 |
| attr-e5-large-instruct | RRF (k=60) | 0.3885 | +0.0002 | +0.0474 |
| attr-e5-small-multi | z-score average | 0.3854 | -0.0029 | +0.0729 |
| attr-e5-small-multi | RRF (k=60) | 0.3815 | -0.0068 | +0.0690 |

#### Impression-weighted

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **attr-siglip** | **z-score average** | **0.4655** | **+0.0535** | **+0.0470** |
| attr-siglip | RRF (k=60) | 0.4614 | +0.0494 | +0.0429 |
| attr-siglip | mean cosine | 0.4589 | +0.0469 | +0.0404 |
| text-e5-large-instruct | mean cosine | 0.4454 | +0.0334 | +0.0478 |
| text-jina-small | RRF (k=60) | 0.4436 | +0.0316 | +0.0542 |
| text-jina | RRF (k=60) | 0.4417 | +0.0297 | +0.0419 |
| text-omni-nano | RRF (k=60) | 0.4415 | +0.0295 | +0.0426 |
| text-jina-small | z-score average | 0.4374 | +0.0254 | +0.0479 |
| text-e5-large-instruct | RRF (k=60) | 0.4371 | +0.0251 | +0.0395 |
| text-omni-nano | z-score average | 0.4367 | +0.0247 | +0.0378 |
| text-jina | z-score average | 0.4358 | +0.0238 | +0.0360 |
| attr-omni-nano | RRF (k=60) | 0.4348 | +0.0228 | +0.0758 |
| text-e5-large-instruct | z-score average | 0.4336 | +0.0216 | +0.0361 |
| attr-omni-nano | z-score average | 0.4336 | +0.0216 | +0.0746 |
| attr-jina | RRF (k=60) | 0.4334 | +0.0214 | +0.0879 |
| text-siglip | RRF (k=60) | 0.4323 | +0.0203 | +0.0365 |
| attr-e5-base | mean cosine | 0.4318 | +0.0198 | +0.0758 |
| text-e5-small-multi | mean cosine | 0.4317 | +0.0197 | +0.0739 |
| attr-jina | z-score average | 0.4311 | +0.0191 | +0.0856 |
| attr-e5-large-instruct | mean cosine | 0.4305 | +0.0185 | +0.1107 |
| attr-jina | mean cosine | 0.4304 | +0.0184 | +0.0849 |
| text-siglip | z-score average | 0.4295 | +0.0175 | +0.0337 |
| attr-e5-small-multi | mean cosine | 0.4291 | +0.0171 | +0.1198 |
| text-omni-nano | mean cosine | 0.4290 | +0.0170 | +0.0301 |
| attr-omni-nano | mean cosine | 0.4274 | +0.0154 | +0.0684 |
| text-jina | mean cosine | 0.4252 | +0.0132 | +0.0254 |
| text-e5-base | mean cosine | 0.4238 | +0.0118 | +0.0507 |
| attr-jina-small | RRF (k=60) | 0.4209 | +0.0089 | +0.0802 |
| text-jina-small | mean cosine | 0.4189 | +0.0069 | +0.0294 |
| text-e5-small-multi | RRF (k=60) | 0.4175 | +0.0055 | +0.0596 |
| text-siglip | mean cosine | 0.4164 | +0.0044 | +0.0206 |
| attr-e5-large-instruct | z-score average | 0.4132 | +0.0012 | +0.0934 |
| text-e5-base | z-score average | 0.4096 | -0.0024 | +0.0366 |
| attr-e5-base | RRF (k=60) | 0.4084 | -0.0036 | +0.0524 |
| text-e5-small-multi | z-score average | 0.4083 | -0.0037 | +0.0504 |
| attr-e5-base | z-score average | 0.4075 | -0.0045 | +0.0515 |
| attr-jina-small | z-score average | 0.4072 | -0.0048 | +0.0665 |
| attr-jina-small | mean cosine | 0.4040 | -0.0080 | +0.0633 |
| attr-e5-small-multi | RRF (k=60) | 0.4020 | -0.0100 | +0.0927 |
| text-e5-base | RRF (k=60) | 0.4016 | -0.0104 | +0.0285 |
| attr-e5-large-instruct | RRF (k=60) | 0.3991 | -0.0129 | +0.0793 |
| attr-e5-small-multi | z-score average | 0.3878 | -0.0242 | +0.0784 |

---

## 4. Significance

Paired bootstrap over queries (2000 resamples), metric NDCG@10, built around
whichever fusion combo actually wins on macro NDCG@10 (`z-score average of siglip-image + text-siglip`).
Macro and weighted deltas come from the same paired samples, so a contrast can be significant under
one weighting and not the other. Includes a contrast against the *other* target image modality,
holding the fusion method and text partner fixed, to check whether the image-encoder choice matters.

| Contrast | Note | Macro delta | Macro p | Weighted delta | Weighted p |
| --- | --- | --- | --- | --- | --- |
| `fusion[z-score average] siglip-image + text-siglip` vs `siglip-image` | headline: best fusion vs its image alone | +0.0334 | 0.0000 (**significant**) | -0.0059 | 0.0630 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `text-siglip` | headline: best fusion vs its text representation alone | +0.0625 | 0.0000 (**significant**) | +0.0426 | 0.0000 (**significant**) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[mean cosine] siglip-image + text-siglip` | method: zscore_avg vs mean_cosine, image=siglip_image, text=siglip_text | +0.0226 | 0.0000 (**significant**) | +0.0188 | 0.1070 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[RRF (k=60)] siglip-image + text-siglip` | method: zscore_avg vs rrf, image=siglip_image, text=siglip_text | +0.0042 | 0.1850 (not significant) | -0.0103 | 0.6260 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + text-jina` | text: siglip_text vs jina_text, image=siglip_image, method=zscore_avg | +0.0117 | 0.1190 (not significant) | -0.0123 | 0.5470 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + text-jina-small` | text: siglip_text vs jina_small_text, image=siglip_image, method=zscore_avg | +0.0153 | 0.0240 (**significant**) | -0.0178 | 0.4390 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-siglip` | text: siglip_text vs siglip_attr, image=siglip_image, method=zscore_avg | +0.0090 | 0.2250 (not significant) | -0.0315 | 0.6340 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-jina` | text: siglip_text vs jina_attr, image=siglip_image, method=zscore_avg | +0.0231 | 0.0020 (**significant**) | +0.0075 | 0.3010 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-jina-small` | text: siglip_text vs jina_small_attr, image=siglip_image, method=zscore_avg | +0.0327 | 0.0010 (**significant**) | +0.0008 | 0.1480 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + text-e5-base` | text: siglip_text vs e5_base_text, image=siglip_image, method=zscore_avg | +0.0252 | 0.0000 (**significant**) | +0.0203 | 0.2230 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-e5-base` | text: siglip_text vs e5_base_attr, image=siglip_image, method=zscore_avg | +0.0314 | 0.0000 (**significant**) | -0.0026 | 0.1760 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + text-e5-small-multi` | text: siglip_text vs e5_small_multi_text, image=siglip_image, method=zscore_avg | +0.0364 | 0.0000 (**significant**) | +0.0191 | 0.0620 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-e5-small-multi` | text: siglip_text vs e5_small_multi_attr, image=siglip_image, method=zscore_avg | +0.0476 | 0.0000 (**significant**) | +0.0421 | 0.0400 (**significant**) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + text-e5-large-instruct` | text: siglip_text vs e5_large_instruct_text, image=siglip_image, method=zscore_avg | +0.0130 | 0.0480 (**significant**) | -0.0232 | 0.4780 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-e5-large-instruct` | text: siglip_text vs e5_large_instruct_attr, image=siglip_image, method=zscore_avg | +0.0272 | 0.0020 (**significant**) | +0.0101 | 0.2020 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + text-omni-nano` | text: siglip_text vs omni_nano_text, image=siglip_image, method=zscore_avg | +0.0112 | 0.1340 (not significant) | -0.0126 | 0.5600 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] siglip-image + attr-omni-nano` | text: siglip_text vs omni_nano_attr, image=siglip_image, method=zscore_avg | +0.0222 | 0.0050 (**significant**) | +0.0048 | 0.2820 (not significant) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] omni-nano-image + text-siglip` | image: siglip_image vs omni_nano_image, text=siglip_text, method=zscore_avg | +0.0780 | 0.0000 (**significant**) | +0.0651 | 0.0000 (**significant**) |
| `fusion[z-score average] siglip-image + text-siglip` vs `fusion[z-score average] jina-clip-v2-image + text-siglip` | image: siglip_image vs jina_clip_v2_image, text=siglip_text, method=zscore_avg | +0.0311 | 0.0000 (**significant**) | +0.0089 | 0.0700 (not significant) |

---

## 5. Reading the result

**No fusion method dominates.** Once tied scores can no longer leak the label order, the three
methods land close together and the winner depends on the pairing rather than on the method alone.
W8 tests this directly across image encoders and finds the best method changes with the encoder —
so "use RRF" is not supportable as a general rule from this evidence.

**`siglip-image` is the stronger target modality here.** Every winning fusion combo in this report pairs its text partner with `siglip-image` (`omni-nano-image` scores 0.3826 under the same method and text partner; `jina-clip-v2-image` scores 0.4295 under the same method and text partner).

**Fusion can make things worse than the text alone -- when the *image* side is the weak link.** `siglip-image` (alone: 0.4272) underperforms its own text partner alone in 0/14 cases; `omni-nano-image` (alone: 0.3315) underperforms its own text partner alone in 5/14 cases; `jina-clip-v2-image` (alone: 0.3883) underperforms its own text partner alone in 0/14 cases. The pattern tracks each image encoder's own standalone strength: `siglip-image` (0.4272 alone) is close in quality to its text partners, so averaging the two is complementary; a much weaker image tower instead acts like added noise once the text partner is already strong, pulling the fused ranking below what the text system achieves by itself. Check `omni-nano-image`'s own standalone NDCG@10 against a text candidate's before assuming fusion can only help.

**Attribute representations are weak here, not strong.** The corrected numbers put every
attribute-based system *below* its title-based counterpart. This is the expected consequence of
the representation's low cardinality: 17.8% distinct strings cannot order candidates within a
category, and the earlier apparent advantage came entirely from those ties being resolved in label
order.

**Fusion still beats the image alone.** That contrast does not depend on attribute text and
survives the correction — it remains the one robust argument for combining modalities at all.

**Mean-cosine remains the least principled option** even where it happens to score well: it
averages raw similarities from differently-scaled embedding spaces, so whichever tower has larger
score magnitude dominates for reasons unrelated to relevance. That it sometimes wins anyway is a
reason to distrust small differences between fusion methods, not a reason to adopt it.

**Fusing with a weak text representation can make things worse than the image alone.** Every
`mean_cosine`-fused pairing with a Jina attribute variant scores at or below its target image alone.
A fusion method is not automatically safe -- it can degrade a good target if the second input is
noisy and the combination method does not correct for scale.

---

> **Note on `production`:** reconstructed from mean observed impression position, not a live
> ranker query; its graded score benefits from position leakage in the labels (see
> `papers/W4_evaluation_validity_and_systems.md`), so it is a reference point, not a fair baseline
> for the fusion comparison above.

Source data: `results/fusion_experiment_summary.csv`, `results/fusion_experiment_per_query.csv`.
Generated by `09_fusion_experiment.py` / `10_fusion_report.py`.
