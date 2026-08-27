# W6 -- Fusion Method and Text-Representation Sweep

Generated with `09_fusion_experiment.py` / `10_fusion_report.py`. Uses cached embeddings from the
current test set (6536 queries) -- no re-querying or re-embedding.

> **TL;DR**
> Best *fusion* combo, macro: **mean cosine of siglip-image + text-e5-large-instruct**
> (NDCG@10 0.4340).
> Best *fusion* combo, impression-weighted: **mean cosine of siglip-image + attr-siglip**
> (NDCG@10 0.4737).
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

**Target modalities (2):** every fusion variant combines one of these image-tower
similarities with one text-based similarity. Both are self-consistent -- query and document are
encoded by the same model.

| Name | Query encoder | Document representation |
| --- | --- | --- |
| `siglip-image` | SigLIP text tower | SigLIP image tower over the product photo |
| `jina-omni-nano-image` | Jina v5 omni-nano `Query: ` | Jina v5 omni-nano `Document: ` image tower over the photo |

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

2 target images x 14 text representations x 3 methods = 84 combinations,
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

Primary label set: LTR judgement-list relevance (grades 0-4). All 6536
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
| `production (not comparable -- see note)` | 0.3853 | 0.4194 | 0.4699 | 0.5514 | 0.6057 | 0.6261 |
| **`fusion[mean cosine] siglip-image + text-e5-large-instruct`** | **0.3878** | **0.4340** | **0.4905** | **0.5710** | **0.6191** | **0.6351** |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.3845 | 0.4309 | 0.4883 | 0.5695 | 0.6178 | 0.6339 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.3885 | 0.4306 | 0.4853 | 0.5673 | 0.6168 | 0.6331 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.3822 | 0.4291 | 0.4851 | 0.5662 | 0.6149 | 0.6311 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.3836 | 0.4289 | 0.4860 | 0.5679 | 0.6166 | 0.6328 |
| `fusion[z-score average] siglip-image + text-jina` | 0.3835 | 0.4289 | 0.4858 | 0.5677 | 0.6164 | 0.6326 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.3822 | 0.4274 | 0.4851 | 0.5669 | 0.6159 | 0.6320 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.3827 | 0.4272 | 0.4846 | 0.5658 | 0.6150 | 0.6313 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.3855 | 0.4272 | 0.4833 | 0.5648 | 0.6142 | 0.6308 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.3822 | 0.4266 | 0.4843 | 0.5654 | 0.6145 | 0.6308 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.3816 | 0.4262 | 0.4836 | 0.5648 | 0.6144 | 0.6308 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.3751 | 0.4208 | 0.4795 | 0.5617 | 0.6108 | 0.6269 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.3747 | 0.4198 | 0.4782 | 0.5612 | 0.6113 | 0.6280 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.3736 | 0.4189 | 0.4767 | 0.5589 | 0.6076 | 0.6241 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.3752 | 0.4188 | 0.4771 | 0.5607 | 0.6107 | 0.6275 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.3751 | 0.4187 | 0.4771 | 0.5606 | 0.6106 | 0.6274 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.3716 | 0.4185 | 0.4778 | 0.5606 | 0.6106 | 0.6270 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.3706 | 0.4165 | 0.4754 | 0.5589 | 0.6093 | 0.6260 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.3720 | 0.4164 | 0.4746 | 0.5587 | 0.6092 | 0.6259 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.3697 | 0.4147 | 0.4728 | 0.5569 | 0.6074 | 0.6242 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.3698 | 0.4139 | 0.4703 | 0.5533 | 0.6028 | 0.6198 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.3657 | 0.4123 | 0.4718 | 0.5564 | 0.6068 | 0.6235 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.3637 | 0.4114 | 0.4704 | 0.5535 | 0.6039 | 0.6205 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.3630 | 0.4105 | 0.4687 | 0.5525 | 0.6035 | 0.6206 |
| `siglip-image` | 0.3651 | 0.4103 | 0.4708 | 0.5539 | 0.6040 | 0.6206 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.3699 | 0.4096 | 0.4652 | 0.5488 | 0.6017 | 0.6196 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.3643 | 0.4090 | 0.4688 | 0.5519 | 0.6016 | 0.6181 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.3618 | 0.4088 | 0.4691 | 0.5540 | 0.6036 | 0.6200 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.3619 | 0.4084 | 0.4688 | 0.5538 | 0.6034 | 0.6197 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.3611 | 0.4080 | 0.4691 | 0.5528 | 0.6023 | 0.6189 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.3587 | 0.4077 | 0.4679 | 0.5528 | 0.6028 | 0.6194 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.3547 | 0.4035 | 0.4652 | 0.5498 | 0.5999 | 0.6168 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.3549 | 0.4027 | 0.4633 | 0.5474 | 0.5982 | 0.6153 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.3518 | 0.3985 | 0.4605 | 0.5458 | 0.5969 | 0.6138 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.3518 | 0.3985 | 0.4606 | 0.5457 | 0.5969 | 0.6137 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.3526 | 0.3983 | 0.4589 | 0.5429 | 0.5935 | 0.6105 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.3525 | 0.3981 | 0.4590 | 0.5434 | 0.5943 | 0.6116 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.3523 | 0.3981 | 0.4590 | 0.5437 | 0.5945 | 0.6117 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.3479 | 0.3952 | 0.4576 | 0.5423 | 0.5933 | 0.6108 |
| `text-e5-large-instruct` | 0.3488 | 0.3950 | 0.4558 | 0.5422 | 0.5944 | 0.6123 |
| `text-jina` | 0.3480 | 0.3932 | 0.4534 | 0.5407 | 0.5935 | 0.6112 |
| `text-omni-nano` | 0.3479 | 0.3931 | 0.4531 | 0.5405 | 0.5934 | 0.6110 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.3427 | 0.3924 | 0.4547 | 0.5404 | 0.5924 | 0.6099 |
| `text-jina-small` | 0.3438 | 0.3906 | 0.4523 | 0.5386 | 0.5919 | 0.6096 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.3443 | 0.3902 | 0.4523 | 0.5376 | 0.5891 | 0.6066 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.3397 | 0.3887 | 0.4499 | 0.5380 | 0.5897 | 0.6066 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.3321 | 0.3798 | 0.4429 | 0.5296 | 0.5818 | 0.5997 |
| `text-siglip` | 0.3367 | 0.3785 | 0.4380 | 0.5247 | 0.5803 | 0.5998 |
| `text-e5-base` | 0.3289 | 0.3736 | 0.4349 | 0.5252 | 0.5797 | 0.5987 |
| `text-e5-small-multi` | 0.3120 | 0.3609 | 0.4238 | 0.5161 | 0.5716 | 0.5911 |
| `attr-siglip` | 0.3062 | 0.3529 | 0.4179 | 0.5084 | 0.5629 | 0.5826 |
| `attr-jina` | 0.2783 | 0.3289 | 0.3995 | 0.4930 | 0.5505 | 0.5701 |
| `attr-omni-nano` | 0.2769 | 0.3283 | 0.3985 | 0.4924 | 0.5499 | 0.5696 |
| `attr-e5-large-instruct` | 0.2760 | 0.3281 | 0.3989 | 0.4921 | 0.5486 | 0.5685 |
| `attr-jina-small` | 0.2719 | 0.3244 | 0.3959 | 0.4895 | 0.5472 | 0.5674 |
| `attr-e5-base` | 0.2668 | 0.3185 | 0.3882 | 0.4828 | 0.5415 | 0.5623 |
| `attr-e5-small-multi` | 0.2512 | 0.3024 | 0.3728 | 0.4715 | 0.5309 | 0.5524 |
| `random` | 0.2082 | 0.2534 | 0.3232 | 0.4266 | 0.4927 | 0.5182 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.2333 | 0.3783 | 0.5745 | 0.8234 | 0.9437 | 0.9846 |
| **`fusion[mean cosine] siglip-image + text-e5-large-instruct`** | **0.2365** | **0.3965** | **0.5944** | **0.8342** | **0.9486** | **0.9853** |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.2367 | 0.3957 | 0.5924 | 0.8345 | 0.9485 | 0.9853 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.2349 | 0.3881 | 0.5822 | 0.8305 | 0.9482 | 0.9849 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.2343 | 0.3941 | 0.5924 | 0.8326 | 0.9485 | 0.9853 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.2376 | 0.3924 | 0.5891 | 0.8317 | 0.9480 | 0.9850 |
| `fusion[z-score average] siglip-image + text-jina` | 0.2379 | 0.3930 | 0.5895 | 0.8319 | 0.9480 | 0.9850 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.2381 | 0.3937 | 0.5917 | 0.8323 | 0.9483 | 0.9851 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.2349 | 0.3917 | 0.5909 | 0.8312 | 0.9480 | 0.9850 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.2318 | 0.3852 | 0.5837 | 0.8297 | 0.9473 | 0.9849 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.2353 | 0.3916 | 0.5913 | 0.8313 | 0.9479 | 0.9850 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.2341 | 0.3911 | 0.5909 | 0.8305 | 0.9481 | 0.9853 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.2310 | 0.3881 | 0.5850 | 0.8296 | 0.9476 | 0.9848 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.2326 | 0.3873 | 0.5878 | 0.8283 | 0.9464 | 0.9847 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.2280 | 0.3827 | 0.5786 | 0.8275 | 0.9459 | 0.9842 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.2348 | 0.3873 | 0.5829 | 0.8288 | 0.9462 | 0.9841 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.2342 | 0.3869 | 0.5831 | 0.8287 | 0.9461 | 0.9841 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.2345 | 0.3891 | 0.5872 | 0.8295 | 0.9473 | 0.9845 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.2333 | 0.3880 | 0.5858 | 0.8288 | 0.9465 | 0.9842 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.2318 | 0.3850 | 0.5836 | 0.8279 | 0.9466 | 0.9847 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.2285 | 0.3844 | 0.5840 | 0.8269 | 0.9464 | 0.9846 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.2234 | 0.3782 | 0.5733 | 0.8240 | 0.9430 | 0.9828 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.2324 | 0.3875 | 0.5832 | 0.8289 | 0.9467 | 0.9844 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.2273 | 0.3840 | 0.5811 | 0.8271 | 0.9466 | 0.9847 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.2282 | 0.3850 | 0.5825 | 0.8269 | 0.9461 | 0.9844 |
| `siglip-image` | 0.2250 | 0.3801 | 0.5806 | 0.8264 | 0.9460 | 0.9842 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.2257 | 0.3743 | 0.5697 | 0.8188 | 0.9434 | 0.9828 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.2204 | 0.3753 | 0.5785 | 0.8264 | 0.9457 | 0.9843 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.2244 | 0.3788 | 0.5786 | 0.8279 | 0.9470 | 0.9847 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.2249 | 0.3783 | 0.5788 | 0.8279 | 0.9470 | 0.9847 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.2239 | 0.3790 | 0.5807 | 0.8270 | 0.9461 | 0.9842 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.2244 | 0.3825 | 0.5807 | 0.8271 | 0.9469 | 0.9848 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.2220 | 0.3783 | 0.5802 | 0.8262 | 0.9466 | 0.9850 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.2229 | 0.3769 | 0.5777 | 0.8243 | 0.9452 | 0.9839 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.2212 | 0.3729 | 0.5745 | 0.8243 | 0.9458 | 0.9843 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.2222 | 0.3731 | 0.5747 | 0.8242 | 0.9458 | 0.9843 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.2173 | 0.3755 | 0.5786 | 0.8229 | 0.9447 | 0.9838 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.2190 | 0.3737 | 0.5786 | 0.8228 | 0.9453 | 0.9844 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.2167 | 0.3728 | 0.5771 | 0.8228 | 0.9453 | 0.9844 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.2151 | 0.3722 | 0.5770 | 0.8226 | 0.9449 | 0.9847 |
| `text-e5-large-instruct` | 0.2232 | 0.3765 | 0.5749 | 0.8238 | 0.9436 | 0.9834 |
| `text-jina` | 0.2217 | 0.3752 | 0.5732 | 0.8216 | 0.9433 | 0.9828 |
| `text-omni-nano` | 0.2222 | 0.3746 | 0.5731 | 0.8216 | 0.9434 | 0.9828 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.2167 | 0.3726 | 0.5727 | 0.8223 | 0.9447 | 0.9843 |
| `text-jina-small` | 0.2191 | 0.3724 | 0.5736 | 0.8207 | 0.9437 | 0.9829 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.2140 | 0.3687 | 0.5740 | 0.8211 | 0.9440 | 0.9836 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.2156 | 0.3718 | 0.5705 | 0.8223 | 0.9449 | 0.9837 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.2082 | 0.3652 | 0.5699 | 0.8185 | 0.9426 | 0.9829 |
| `text-siglip` | 0.2087 | 0.3549 | 0.5553 | 0.8087 | 0.9388 | 0.9814 |
| `text-e5-base` | 0.2151 | 0.3621 | 0.5603 | 0.8139 | 0.9393 | 0.9818 |
| `text-e5-small-multi` | 0.2077 | 0.3593 | 0.5570 | 0.8135 | 0.9390 | 0.9817 |
| `attr-siglip` | 0.1955 | 0.3429 | 0.5462 | 0.8075 | 0.9349 | 0.9799 |
| `attr-jina` | 0.1859 | 0.3346 | 0.5437 | 0.8056 | 0.9375 | 0.9812 |
| `attr-omni-nano` | 0.1876 | 0.3345 | 0.5441 | 0.8054 | 0.9375 | 0.9812 |
| `attr-e5-large-instruct` | 0.1840 | 0.3353 | 0.5465 | 0.8062 | 0.9354 | 0.9801 |
| `attr-jina-small` | 0.1811 | 0.3316 | 0.5421 | 0.8042 | 0.9361 | 0.9810 |
| `attr-e5-base` | 0.1787 | 0.3256 | 0.5345 | 0.7998 | 0.9344 | 0.9799 |
| `attr-e5-small-multi` | 0.1703 | 0.3176 | 0.5265 | 0.7966 | 0.9319 | 0.9789 |
| `random` | 0.1372 | 0.2735 | 0.4843 | 0.7705 | 0.9184 | 0.9735 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.5294 | 0.5219 | 0.4903 | 0.3880 | 0.2715 | 0.2037 |
| **`fusion[mean cosine] siglip-image + text-e5-large-instruct`** | **0.6071** | **0.5814** | **0.5300** | **0.4075** | **0.2769** | **0.2046** |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.6081 | 0.5821 | 0.5298 | 0.4076 | 0.2769 | 0.2046 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.6010 | 0.5727 | 0.5209 | 0.4039 | 0.2763 | 0.2043 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.6063 | 0.5801 | 0.5275 | 0.4061 | 0.2767 | 0.2046 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.6044 | 0.5775 | 0.5263 | 0.4063 | 0.2767 | 0.2046 |
| `fusion[z-score average] siglip-image + text-jina` | 0.6041 | 0.5776 | 0.5262 | 0.4063 | 0.2767 | 0.2046 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.6061 | 0.5791 | 0.5269 | 0.4063 | 0.2769 | 0.2046 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.6031 | 0.5759 | 0.5246 | 0.4049 | 0.2765 | 0.2046 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.5987 | 0.5710 | 0.5193 | 0.4015 | 0.2755 | 0.2043 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.6035 | 0.5757 | 0.5246 | 0.4049 | 0.2765 | 0.2045 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.6035 | 0.5756 | 0.5244 | 0.4048 | 0.2766 | 0.2047 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.5909 | 0.5679 | 0.5197 | 0.4035 | 0.2764 | 0.2046 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.5968 | 0.5715 | 0.5215 | 0.4031 | 0.2757 | 0.2044 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.5877 | 0.5656 | 0.5176 | 0.4017 | 0.2751 | 0.2041 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.6018 | 0.5720 | 0.5216 | 0.4034 | 0.2755 | 0.2041 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.6014 | 0.5718 | 0.5220 | 0.4034 | 0.2754 | 0.2040 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.5985 | 0.5724 | 0.5220 | 0.4035 | 0.2760 | 0.2044 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.6002 | 0.5729 | 0.5222 | 0.4038 | 0.2757 | 0.2041 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.5962 | 0.5703 | 0.5205 | 0.4028 | 0.2758 | 0.2044 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.5947 | 0.5684 | 0.5187 | 0.4016 | 0.2755 | 0.2043 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.5830 | 0.5609 | 0.5126 | 0.3985 | 0.2733 | 0.2034 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.5954 | 0.5695 | 0.5201 | 0.4025 | 0.2757 | 0.2043 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.5823 | 0.5629 | 0.5156 | 0.4014 | 0.2758 | 0.2045 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.5931 | 0.5684 | 0.5177 | 0.4004 | 0.2751 | 0.2042 |
| `siglip-image` | 0.5816 | 0.5584 | 0.5129 | 0.3991 | 0.2751 | 0.2041 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.5844 | 0.5543 | 0.5052 | 0.3932 | 0.2727 | 0.2032 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.5800 | 0.5584 | 0.5115 | 0.3992 | 0.2747 | 0.2041 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.5801 | 0.5602 | 0.5145 | 0.4026 | 0.2760 | 0.2045 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.5794 | 0.5600 | 0.5145 | 0.4025 | 0.2760 | 0.2045 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.5803 | 0.5591 | 0.5142 | 0.4008 | 0.2753 | 0.2042 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.5816 | 0.5612 | 0.5156 | 0.4020 | 0.2759 | 0.2045 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.5746 | 0.5546 | 0.5116 | 0.4006 | 0.2758 | 0.2046 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.5743 | 0.5556 | 0.5103 | 0.3990 | 0.2748 | 0.2041 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.5733 | 0.5536 | 0.5101 | 0.3998 | 0.2751 | 0.2043 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.5731 | 0.5539 | 0.5104 | 0.3998 | 0.2751 | 0.2043 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.5762 | 0.5543 | 0.5084 | 0.3967 | 0.2739 | 0.2038 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.5754 | 0.5540 | 0.5084 | 0.3971 | 0.2744 | 0.2042 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.5746 | 0.5545 | 0.5091 | 0.3975 | 0.2744 | 0.2042 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.5704 | 0.5497 | 0.5061 | 0.3966 | 0.2744 | 0.2044 |
| `text-e5-large-instruct` | 0.5865 | 0.5618 | 0.5128 | 0.3981 | 0.2734 | 0.2035 |
| `text-jina` | 0.5841 | 0.5574 | 0.5089 | 0.3969 | 0.2733 | 0.2033 |
| `text-omni-nano` | 0.5839 | 0.5572 | 0.5088 | 0.3969 | 0.2733 | 0.2033 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.5666 | 0.5479 | 0.5054 | 0.3968 | 0.2745 | 0.2042 |
| `text-jina-small` | 0.5840 | 0.5586 | 0.5109 | 0.3967 | 0.2737 | 0.2034 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.5669 | 0.5473 | 0.5039 | 0.3943 | 0.2736 | 0.2038 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.5670 | 0.5479 | 0.5058 | 0.3979 | 0.2746 | 0.2040 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.5613 | 0.5431 | 0.5008 | 0.3928 | 0.2726 | 0.2035 |
| `text-siglip` | 0.5576 | 0.5312 | 0.4873 | 0.3836 | 0.2696 | 0.2023 |
| `text-e5-base` | 0.5674 | 0.5447 | 0.4985 | 0.3911 | 0.2711 | 0.2029 |
| `text-e5-small-multi` | 0.5576 | 0.5384 | 0.4940 | 0.3888 | 0.2706 | 0.2028 |
| `attr-siglip` | 0.5270 | 0.5149 | 0.4792 | 0.3827 | 0.2680 | 0.2019 |
| `attr-jina` | 0.5088 | 0.5021 | 0.4743 | 0.3819 | 0.2696 | 0.2026 |
| `attr-omni-nano` | 0.5080 | 0.5008 | 0.4731 | 0.3817 | 0.2695 | 0.2026 |
| `attr-e5-large-instruct` | 0.5097 | 0.5017 | 0.4738 | 0.3807 | 0.2683 | 0.2020 |
| `attr-jina-small` | 0.5050 | 0.4969 | 0.4702 | 0.3799 | 0.2689 | 0.2025 |
| `attr-e5-base` | 0.4959 | 0.4907 | 0.4641 | 0.3765 | 0.2676 | 0.2019 |
| `attr-e5-small-multi` | 0.4856 | 0.4809 | 0.4572 | 0.3737 | 0.2662 | 0.2015 |
| `random` | 0.4250 | 0.4252 | 0.4104 | 0.3468 | 0.2570 | 0.1985 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6352 | 0.6456 | 0.6492 | 0.6506 | 0.6506 | 0.6506 |
| **`fusion[mean cosine] siglip-image + text-e5-large-instruct`** | **0.7164** | **0.7261** | **0.7299** | **0.7312** | **0.7312** | **0.7312** |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.7170 | 0.7262 | 0.7302 | 0.7315 | 0.7316 | 0.7316 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.7111 | 0.7202 | 0.7240 | 0.7254 | 0.7254 | 0.7254 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.7146 | 0.7242 | 0.7282 | 0.7295 | 0.7296 | 0.7296 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.7158 | 0.7251 | 0.7288 | 0.7302 | 0.7303 | 0.7303 |
| `fusion[z-score average] siglip-image + text-jina` | 0.7156 | 0.7249 | 0.7285 | 0.7299 | 0.7299 | 0.7299 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.7188 | 0.7279 | 0.7317 | 0.7330 | 0.7331 | 0.7331 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.7152 | 0.7246 | 0.7284 | 0.7297 | 0.7297 | 0.7297 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.7074 | 0.7169 | 0.7209 | 0.7222 | 0.7223 | 0.7223 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.7152 | 0.7246 | 0.7283 | 0.7296 | 0.7296 | 0.7296 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.7144 | 0.7241 | 0.7280 | 0.7293 | 0.7293 | 0.7293 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.6994 | 0.7087 | 0.7127 | 0.7141 | 0.7141 | 0.7141 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.7105 | 0.7196 | 0.7238 | 0.7251 | 0.7252 | 0.7252 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.6972 | 0.7069 | 0.7109 | 0.7123 | 0.7124 | 0.7124 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.7102 | 0.7196 | 0.7232 | 0.7247 | 0.7248 | 0.7248 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.7108 | 0.7200 | 0.7237 | 0.7252 | 0.7252 | 0.7252 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.7095 | 0.7189 | 0.7229 | 0.7242 | 0.7243 | 0.7243 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.7120 | 0.7214 | 0.7253 | 0.7267 | 0.7268 | 0.7268 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.7091 | 0.7183 | 0.7223 | 0.7238 | 0.7238 | 0.7238 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.7054 | 0.7147 | 0.7189 | 0.7203 | 0.7203 | 0.7203 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.6933 | 0.7033 | 0.7072 | 0.7087 | 0.7088 | 0.7088 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.7074 | 0.7169 | 0.7205 | 0.7220 | 0.7221 | 0.7221 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.6912 | 0.7011 | 0.7052 | 0.7067 | 0.7067 | 0.7067 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.7056 | 0.7153 | 0.7193 | 0.7207 | 0.7207 | 0.7207 |
| `siglip-image` | 0.6922 | 0.7019 | 0.7062 | 0.7076 | 0.7076 | 0.7076 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.6983 | 0.7078 | 0.7118 | 0.7133 | 0.7134 | 0.7134 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.6895 | 0.6994 | 0.7038 | 0.7051 | 0.7052 | 0.7052 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.6877 | 0.6976 | 0.7019 | 0.7033 | 0.7034 | 0.7034 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.6873 | 0.6971 | 0.7013 | 0.7027 | 0.7028 | 0.7028 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.6855 | 0.6955 | 0.7000 | 0.7013 | 0.7014 | 0.7014 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.6904 | 0.7008 | 0.7049 | 0.7063 | 0.7063 | 0.7063 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.6806 | 0.6912 | 0.6956 | 0.6969 | 0.6970 | 0.6970 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.6855 | 0.6954 | 0.6998 | 0.7012 | 0.7012 | 0.7012 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.6770 | 0.6870 | 0.6916 | 0.6931 | 0.6931 | 0.6931 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.6779 | 0.6876 | 0.6922 | 0.6937 | 0.6938 | 0.6938 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.6800 | 0.6907 | 0.6952 | 0.6965 | 0.6965 | 0.6965 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.6819 | 0.6922 | 0.6967 | 0.6980 | 0.6980 | 0.6980 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.6798 | 0.6905 | 0.6950 | 0.6963 | 0.6964 | 0.6964 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.6785 | 0.6893 | 0.6940 | 0.6953 | 0.6954 | 0.6954 |
| `text-e5-large-instruct` | 0.6936 | 0.7036 | 0.7078 | 0.7094 | 0.7095 | 0.7095 |
| `text-jina` | 0.6896 | 0.6997 | 0.7036 | 0.7052 | 0.7053 | 0.7053 |
| `text-omni-nano` | 0.6904 | 0.7004 | 0.7043 | 0.7059 | 0.7060 | 0.7060 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.6707 | 0.6812 | 0.6856 | 0.6872 | 0.6873 | 0.6873 |
| `text-jina-small` | 0.6899 | 0.7001 | 0.7045 | 0.7061 | 0.7061 | 0.7061 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.6757 | 0.6864 | 0.6911 | 0.6924 | 0.6925 | 0.6925 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.6711 | 0.6813 | 0.6857 | 0.6872 | 0.6873 | 0.6873 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.6669 | 0.6780 | 0.6828 | 0.6842 | 0.6842 | 0.6842 |
| `text-siglip` | 0.6689 | 0.6794 | 0.6840 | 0.6856 | 0.6857 | 0.6857 |
| `text-e5-base` | 0.6784 | 0.6883 | 0.6929 | 0.6946 | 0.6947 | 0.6947 |
| `text-e5-small-multi` | 0.6653 | 0.6760 | 0.6806 | 0.6824 | 0.6824 | 0.6824 |
| `attr-siglip` | 0.6346 | 0.6463 | 0.6512 | 0.6531 | 0.6531 | 0.6531 |
| `attr-jina` | 0.6069 | 0.6195 | 0.6252 | 0.6271 | 0.6272 | 0.6272 |
| `attr-omni-nano` | 0.6066 | 0.6192 | 0.6249 | 0.6268 | 0.6268 | 0.6268 |
| `attr-e5-large-instruct` | 0.6027 | 0.6162 | 0.6219 | 0.6238 | 0.6239 | 0.6239 |
| `attr-jina-small` | 0.6039 | 0.6171 | 0.6228 | 0.6247 | 0.6248 | 0.6248 |
| `attr-e5-base` | 0.5934 | 0.6064 | 0.6125 | 0.6146 | 0.6147 | 0.6147 |
| `attr-e5-small-multi` | 0.5797 | 0.5933 | 0.5996 | 0.6018 | 0.6019 | 0.6019 |
| `random` | 0.5432 | 0.5571 | 0.5639 | 0.5662 | 0.5663 | 0.5663 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.5979 |
| **`fusion[mean cosine] siglip-image + text-e5-large-instruct`** | **0.6342** |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.6353 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.6240 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.6299 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.6318 |
| `fusion[z-score average] siglip-image + text-jina` | 0.6315 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.6328 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.6288 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.6188 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.6284 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.6290 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.6218 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.6266 |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.6138 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.6270 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.6269 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.6269 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.6274 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.6246 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.6208 |
| `fusion[mean cosine] siglip-image + attr-siglip` | 0.6081 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.6242 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.6148 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.6180 |
| `siglip-image` | 0.6119 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.6045 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.6044 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.6126 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.6122 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.6121 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.6153 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.6089 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.6072 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.6066 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.6067 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.5993 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.6001 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.5993 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.5991 |
| `text-e5-large-instruct` | 0.6129 |
| `text-jina` | 0.6096 |
| `text-omni-nano` | 0.6094 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.6018 |
| `text-jina-small` | 0.6093 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.5942 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.6002 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.5880 |
| `text-siglip` | 0.5786 |
| `text-e5-base` | 0.5964 |
| `text-e5-small-multi` | 0.5891 |
| `attr-siglip` | 0.5648 |
| `attr-jina` | 0.5564 |
| `attr-omni-nano` | 0.5563 |
| `attr-e5-large-instruct` | 0.5553 |
| `attr-jina-small` | 0.5541 |
| `attr-e5-base` | 0.5481 |
| `attr-e5-small-multi` | 0.5374 |
| `random` | 0.4757 |


#### Impression-weighted -- every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4190 | 0.4239 | 0.4521 | 0.5459 | 0.6611 | 0.7126 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.4586** | **0.4737** | **0.5186** | **0.6121** | **0.7043** | **0.7438** |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.4557 | 0.4736 | 0.5237 | 0.6182 | 0.7087 | 0.7476 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.4548 | 0.4718 | 0.5138 | 0.6169 | 0.7098 | 0.7480 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.4464 | 0.4714 | 0.5160 | 0.6168 | 0.7088 | 0.7475 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.4504 | 0.4713 | 0.5203 | 0.6203 | 0.7122 | 0.7495 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.4550 | 0.4709 | 0.5145 | 0.6171 | 0.7098 | 0.7480 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.4553 | 0.4701 | 0.5159 | 0.6155 | 0.7097 | 0.7483 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.4555 | 0.4697 | 0.5168 | 0.6148 | 0.7066 | 0.7454 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.4474 | 0.4694 | 0.5192 | 0.6190 | 0.7101 | 0.7487 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.4455 | 0.4666 | 0.5150 | 0.6183 | 0.7083 | 0.7468 |
| `fusion[z-score average] siglip-image + text-jina` | 0.4429 | 0.4660 | 0.5137 | 0.6181 | 0.7079 | 0.7465 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.4492 | 0.4657 | 0.5153 | 0.6175 | 0.7095 | 0.7477 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.4427 | 0.4654 | 0.5138 | 0.6140 | 0.7060 | 0.7437 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.4454 | 0.4607 | 0.5045 | 0.6094 | 0.7039 | 0.7435 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.4338 | 0.4568 | 0.5025 | 0.6044 | 0.6988 | 0.7385 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.4376 | 0.4567 | 0.5019 | 0.6078 | 0.7030 | 0.7429 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.4336 | 0.4528 | 0.5036 | 0.6078 | 0.6991 | 0.7390 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.4325 | 0.4518 | 0.5025 | 0.6076 | 0.6992 | 0.7390 |
| `siglip-image` | 0.4312 | 0.4516 | 0.5019 | 0.6065 | 0.7001 | 0.7374 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.4246 | 0.4486 | 0.5017 | 0.6056 | 0.6999 | 0.7386 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.4213 | 0.4483 | 0.4958 | 0.5969 | 0.6921 | 0.7322 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.4246 | 0.4482 | 0.5003 | 0.6028 | 0.6977 | 0.7357 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.4208 | 0.4479 | 0.4975 | 0.6017 | 0.6976 | 0.7369 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.4217 | 0.4478 | 0.4991 | 0.6048 | 0.6995 | 0.7381 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.4240 | 0.4470 | 0.5015 | 0.6052 | 0.6997 | 0.7382 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.4295 | 0.4470 | 0.4952 | 0.5993 | 0.6937 | 0.7342 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.4231 | 0.4449 | 0.4958 | 0.6011 | 0.6965 | 0.7362 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.4274 | 0.4445 | 0.4922 | 0.5982 | 0.6912 | 0.7317 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.4142 | 0.4441 | 0.4979 | 0.6027 | 0.6966 | 0.7349 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.4251 | 0.4438 | 0.4930 | 0.5983 | 0.6932 | 0.7334 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.4167 | 0.4430 | 0.4965 | 0.6035 | 0.6967 | 0.7354 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.4168 | 0.4425 | 0.4962 | 0.6005 | 0.6938 | 0.7343 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.4120 | 0.4420 | 0.4896 | 0.5958 | 0.6922 | 0.7320 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.4171 | 0.4418 | 0.4973 | 0.6013 | 0.6956 | 0.7351 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.4189 | 0.4408 | 0.4863 | 0.5898 | 0.6866 | 0.7269 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.4225 | 0.4402 | 0.4900 | 0.5944 | 0.6908 | 0.7317 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.4170 | 0.4395 | 0.4925 | 0.5972 | 0.6943 | 0.7333 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.4052 | 0.4375 | 0.4884 | 0.5965 | 0.6919 | 0.7309 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.4155 | 0.4373 | 0.4923 | 0.5966 | 0.6939 | 0.7331 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.4271 | 0.4327 | 0.4768 | 0.5789 | 0.6821 | 0.7252 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.4006 | 0.4280 | 0.4795 | 0.5857 | 0.6848 | 0.7265 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.4060 | 0.4250 | 0.4747 | 0.5822 | 0.6800 | 0.7217 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.3929 | 0.4207 | 0.4748 | 0.5859 | 0.6834 | 0.7229 |
| `text-e5-large-instruct` | 0.3919 | 0.4182 | 0.4745 | 0.5818 | 0.6801 | 0.7215 |
| `text-omni-nano` | 0.3912 | 0.4168 | 0.4722 | 0.5813 | 0.6784 | 0.7193 |
| `text-jina` | 0.3905 | 0.4165 | 0.4707 | 0.5810 | 0.6777 | 0.7187 |
| `text-jina-small` | 0.3847 | 0.4139 | 0.4705 | 0.5789 | 0.6784 | 0.7199 |
| `text-siglip` | 0.3953 | 0.4065 | 0.4464 | 0.5499 | 0.6577 | 0.7057 |
| `attr-siglip` | 0.3841 | 0.4055 | 0.4558 | 0.5631 | 0.6622 | 0.7063 |
| `text-e5-base` | 0.3765 | 0.3951 | 0.4476 | 0.5622 | 0.6608 | 0.7055 |
| `text-e5-small-multi` | 0.3494 | 0.3773 | 0.4308 | 0.5506 | 0.6563 | 0.6983 |
| `attr-omni-nano` | 0.3254 | 0.3588 | 0.4191 | 0.5360 | 0.6422 | 0.6871 |
| `attr-e5-base` | 0.3190 | 0.3539 | 0.4102 | 0.5267 | 0.6328 | 0.6806 |
| `attr-jina` | 0.3217 | 0.3510 | 0.4168 | 0.5351 | 0.6418 | 0.6868 |
| `attr-jina-small` | 0.3232 | 0.3509 | 0.4140 | 0.5286 | 0.6345 | 0.6834 |
| `attr-e5-large-instruct` | 0.3220 | 0.3496 | 0.4155 | 0.5362 | 0.6396 | 0.6869 |
| `attr-e5-small-multi` | 0.2912 | 0.3225 | 0.3828 | 0.5098 | 0.6227 | 0.6698 |
| `random` | 0.2624 | 0.2840 | 0.3375 | 0.4550 | 0.5768 | 0.6343 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.0762 | 0.1569 | 0.3051 | 0.5903 | 0.8477 | 0.9533 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.0987** | **0.1922** | **0.3536** | **0.6464** | **0.8678** | **0.9579** |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.0995 | 0.1935 | 0.3561 | 0.6498 | 0.8718 | 0.9602 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.1012 | 0.1962 | 0.3604 | 0.6536 | 0.8739 | 0.9609 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.1018 | 0.1974 | 0.3611 | 0.6557 | 0.8756 | 0.9617 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.1019 | 0.1975 | 0.3626 | 0.6577 | 0.8767 | 0.9623 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.1012 | 0.1963 | 0.3604 | 0.6535 | 0.8740 | 0.9610 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.1013 | 0.1962 | 0.3602 | 0.6540 | 0.8751 | 0.9618 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.0982 | 0.1910 | 0.3525 | 0.6466 | 0.8704 | 0.9602 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.1018 | 0.1978 | 0.3628 | 0.6585 | 0.8757 | 0.9618 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.1013 | 0.1966 | 0.3616 | 0.6553 | 0.8740 | 0.9611 |
| `fusion[z-score average] siglip-image + text-jina` | 0.1012 | 0.1965 | 0.3616 | 0.6552 | 0.8741 | 0.9611 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.1016 | 0.1971 | 0.3620 | 0.6563 | 0.8752 | 0.9617 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.0995 | 0.1941 | 0.3581 | 0.6522 | 0.8759 | 0.9625 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.0997 | 0.1921 | 0.3531 | 0.6457 | 0.8730 | 0.9600 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.0991 | 0.1932 | 0.3565 | 0.6501 | 0.8738 | 0.9623 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.0997 | 0.1931 | 0.3541 | 0.6479 | 0.8738 | 0.9600 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.1011 | 0.1957 | 0.3592 | 0.6519 | 0.8688 | 0.9582 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.1012 | 0.1956 | 0.3591 | 0.6520 | 0.8689 | 0.9583 |
| `siglip-image` | 0.0982 | 0.1919 | 0.3537 | 0.6480 | 0.8743 | 0.9604 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.0982 | 0.1923 | 0.3557 | 0.6512 | 0.8739 | 0.9621 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.0981 | 0.1911 | 0.3535 | 0.6470 | 0.8722 | 0.9610 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.0983 | 0.1918 | 0.3556 | 0.6495 | 0.8733 | 0.9612 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.0978 | 0.1906 | 0.3536 | 0.6488 | 0.8745 | 0.9626 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.1006 | 0.1958 | 0.3602 | 0.6527 | 0.8728 | 0.9611 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.0983 | 0.1923 | 0.3557 | 0.6511 | 0.8739 | 0.9621 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.0973 | 0.1902 | 0.3519 | 0.6449 | 0.8705 | 0.9613 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.1014 | 0.1961 | 0.3594 | 0.6530 | 0.8718 | 0.9597 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.0979 | 0.1901 | 0.3520 | 0.6456 | 0.8702 | 0.9604 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.1010 | 0.1963 | 0.3603 | 0.6535 | 0.8738 | 0.9613 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.0973 | 0.1903 | 0.3521 | 0.6452 | 0.8706 | 0.9614 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.0988 | 0.1928 | 0.3565 | 0.6527 | 0.8755 | 0.9624 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.1006 | 0.1956 | 0.3586 | 0.6520 | 0.8719 | 0.9602 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.1004 | 0.1953 | 0.3578 | 0.6496 | 0.8723 | 0.9606 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.1006 | 0.1960 | 0.3597 | 0.6532 | 0.8729 | 0.9610 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.0970 | 0.1890 | 0.3506 | 0.6423 | 0.8701 | 0.9602 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.0970 | 0.1890 | 0.3502 | 0.6443 | 0.8705 | 0.9620 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.0973 | 0.1906 | 0.3523 | 0.6482 | 0.8711 | 0.9616 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.1006 | 0.1957 | 0.3593 | 0.6519 | 0.8725 | 0.9608 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.0973 | 0.1905 | 0.3526 | 0.6482 | 0.8710 | 0.9616 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.0976 | 0.1872 | 0.3446 | 0.6319 | 0.8632 | 0.9556 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.0967 | 0.1890 | 0.3499 | 0.6444 | 0.8705 | 0.9614 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.0960 | 0.1874 | 0.3477 | 0.6407 | 0.8689 | 0.9595 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.0966 | 0.1888 | 0.3504 | 0.6470 | 0.8729 | 0.9609 |
| `text-e5-large-instruct` | 0.0996 | 0.1933 | 0.3555 | 0.6480 | 0.8659 | 0.9581 |
| `text-omni-nano` | 0.0991 | 0.1925 | 0.3536 | 0.6446 | 0.8632 | 0.9557 |
| `text-jina` | 0.0991 | 0.1924 | 0.3536 | 0.6447 | 0.8631 | 0.9557 |
| `text-jina-small` | 0.0996 | 0.1931 | 0.3551 | 0.6454 | 0.8672 | 0.9576 |
| `text-siglip` | 0.0942 | 0.1813 | 0.3339 | 0.6168 | 0.8534 | 0.9525 |
| `attr-siglip` | 0.0905 | 0.1781 | 0.3332 | 0.6265 | 0.8560 | 0.9522 |
| `text-e5-base` | 0.0988 | 0.1918 | 0.3512 | 0.6404 | 0.8605 | 0.9553 |
| `text-e5-small-multi` | 0.0971 | 0.1890 | 0.3472 | 0.6369 | 0.8608 | 0.9549 |
| `attr-omni-nano` | 0.0886 | 0.1760 | 0.3322 | 0.6273 | 0.8587 | 0.9571 |
| `attr-e5-base` | 0.0875 | 0.1744 | 0.3295 | 0.6235 | 0.8554 | 0.9550 |
| `attr-jina` | 0.0883 | 0.1760 | 0.3327 | 0.6272 | 0.8590 | 0.9571 |
| `attr-jina-small` | 0.0877 | 0.1746 | 0.3298 | 0.6257 | 0.8571 | 0.9564 |
| `attr-e5-large-instruct` | 0.0887 | 0.1757 | 0.3334 | 0.6287 | 0.8580 | 0.9553 |
| `attr-e5-small-multi` | 0.0847 | 0.1696 | 0.3235 | 0.6196 | 0.8562 | 0.9543 |
| `random` | 0.0749 | 0.1516 | 0.2906 | 0.5721 | 0.8265 | 0.9409 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7164 | 0.7371 | 0.7390 | 0.6841 | 0.5694 | 0.4603 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.9154** | **0.9066** | **0.8793** | **0.7654** | **0.5900** | **0.4648** |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.9181 | 0.9114 | 0.8848 | 0.7701 | 0.5934 | 0.4664 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.9287 | 0.9200 | 0.8910 | 0.7729 | 0.5951 | 0.4666 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.9361 | 0.9259 | 0.8937 | 0.7750 | 0.5963 | 0.4671 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.9359 | 0.9241 | 0.8973 | 0.7793 | 0.5974 | 0.4677 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.9287 | 0.9200 | 0.8911 | 0.7728 | 0.5952 | 0.4667 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.9317 | 0.9200 | 0.8906 | 0.7731 | 0.5961 | 0.4673 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.9146 | 0.9052 | 0.8774 | 0.7648 | 0.5916 | 0.4663 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.9343 | 0.9273 | 0.8977 | 0.7800 | 0.5966 | 0.4673 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.9300 | 0.9192 | 0.8938 | 0.7760 | 0.5953 | 0.4669 |
| `fusion[z-score average] siglip-image + text-jina` | 0.9274 | 0.9189 | 0.8938 | 0.7758 | 0.5953 | 0.4669 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.9334 | 0.9227 | 0.8949 | 0.7768 | 0.5962 | 0.4674 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.9197 | 0.9122 | 0.8855 | 0.7717 | 0.5971 | 0.4683 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.9184 | 0.9029 | 0.8748 | 0.7632 | 0.5936 | 0.4659 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.9144 | 0.9053 | 0.8817 | 0.7695 | 0.5955 | 0.4682 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.9149 | 0.9052 | 0.8765 | 0.7665 | 0.5948 | 0.4661 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.9248 | 0.9111 | 0.8849 | 0.7710 | 0.5908 | 0.4646 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.9252 | 0.9106 | 0.8845 | 0.7711 | 0.5909 | 0.4646 |
| `siglip-image` | 0.9049 | 0.8981 | 0.8735 | 0.7667 | 0.5952 | 0.4664 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.9128 | 0.9073 | 0.8815 | 0.7708 | 0.5954 | 0.4679 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.9067 | 0.8971 | 0.8752 | 0.7652 | 0.5939 | 0.4671 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.9088 | 0.8996 | 0.8799 | 0.7678 | 0.5949 | 0.4674 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.9052 | 0.8965 | 0.8757 | 0.7672 | 0.5961 | 0.4683 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.9198 | 0.9131 | 0.8893 | 0.7721 | 0.5945 | 0.4671 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.9135 | 0.9065 | 0.8816 | 0.7707 | 0.5955 | 0.4679 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.9122 | 0.9034 | 0.8739 | 0.7603 | 0.5913 | 0.4670 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.9265 | 0.9133 | 0.8855 | 0.7721 | 0.5934 | 0.4658 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.9139 | 0.8997 | 0.8731 | 0.7612 | 0.5912 | 0.4664 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.9258 | 0.9171 | 0.8899 | 0.7736 | 0.5950 | 0.4672 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.9120 | 0.9038 | 0.8746 | 0.7607 | 0.5915 | 0.4671 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.9141 | 0.9057 | 0.8818 | 0.7739 | 0.5968 | 0.4682 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.9239 | 0.9151 | 0.8862 | 0.7710 | 0.5931 | 0.4659 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.9206 | 0.9149 | 0.8838 | 0.7670 | 0.5937 | 0.4665 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.9226 | 0.9144 | 0.8868 | 0.7729 | 0.5947 | 0.4668 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.9021 | 0.8926 | 0.8680 | 0.7565 | 0.5915 | 0.4663 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.9055 | 0.8944 | 0.8675 | 0.7591 | 0.5915 | 0.4676 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.8995 | 0.8971 | 0.8725 | 0.7665 | 0.5930 | 0.4677 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.9224 | 0.9146 | 0.8865 | 0.7708 | 0.5937 | 0.4667 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.8990 | 0.8961 | 0.8731 | 0.7665 | 0.5929 | 0.4677 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.8978 | 0.8775 | 0.8523 | 0.7452 | 0.5850 | 0.4627 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.8956 | 0.8879 | 0.8643 | 0.7605 | 0.5927 | 0.4676 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.9016 | 0.8940 | 0.8659 | 0.7561 | 0.5901 | 0.4660 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.8932 | 0.8903 | 0.8682 | 0.7663 | 0.5947 | 0.4672 |
| `text-e5-large-instruct` | 0.9035 | 0.8958 | 0.8723 | 0.7645 | 0.5877 | 0.4647 |
| `text-omni-nano` | 0.8947 | 0.8909 | 0.8672 | 0.7603 | 0.5854 | 0.4628 |
| `text-jina` | 0.8945 | 0.8901 | 0.8667 | 0.7604 | 0.5854 | 0.4627 |
| `text-jina-small` | 0.9041 | 0.8948 | 0.8723 | 0.7617 | 0.5893 | 0.4643 |
| `text-siglip` | 0.8683 | 0.8521 | 0.8265 | 0.7249 | 0.5765 | 0.4603 |
| `attr-siglip` | 0.8502 | 0.8522 | 0.8310 | 0.7382 | 0.5798 | 0.4608 |
| `text-e5-base` | 0.9016 | 0.8922 | 0.8614 | 0.7535 | 0.5833 | 0.4627 |
| `text-e5-small-multi` | 0.8792 | 0.8758 | 0.8516 | 0.7500 | 0.5839 | 0.4622 |
| `attr-omni-nano` | 0.8213 | 0.8314 | 0.8182 | 0.7365 | 0.5819 | 0.4645 |
| `attr-e5-base` | 0.8048 | 0.8200 | 0.8093 | 0.7321 | 0.5793 | 0.4632 |
| `attr-jina` | 0.8204 | 0.8320 | 0.8203 | 0.7367 | 0.5820 | 0.4645 |
| `attr-jina-small` | 0.8181 | 0.8242 | 0.8110 | 0.7342 | 0.5804 | 0.4643 |
| `attr-e5-large-instruct` | 0.8219 | 0.8287 | 0.8215 | 0.7386 | 0.5812 | 0.4634 |
| `attr-e5-small-multi` | 0.7901 | 0.8023 | 0.8019 | 0.7307 | 0.5804 | 0.4629 |
| `random` | 0.7144 | 0.7245 | 0.7187 | 0.6648 | 0.5524 | 0.4523 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7561 | 0.7618 | 0.7628 | 0.7628 | 0.7628 | 0.7628 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.9529** | **0.9538** | **0.9539** | **0.9539** | **0.9539** | **0.9539** |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.9561 | 0.9567 | 0.9569 | 0.9569 | 0.9569 | 0.9569 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.9618 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.9689 | 0.9691 | 0.9692 | 0.9692 | 0.9692 | 0.9692 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.9639 | 0.9642 | 0.9642 | 0.9642 | 0.9642 | 0.9642 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.9615 | 0.9618 | 0.9618 | 0.9619 | 0.9619 | 0.9619 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.9680 | 0.9683 | 0.9683 | 0.9684 | 0.9684 | 0.9684 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.9533 | 0.9541 | 0.9542 | 0.9542 | 0.9542 | 0.9542 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.9642 | 0.9644 | 0.9644 | 0.9644 | 0.9644 | 0.9644 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.9623 | 0.9627 | 0.9628 | 0.9628 | 0.9628 | 0.9628 |
| `fusion[z-score average] siglip-image + text-jina` | 0.9626 | 0.9629 | 0.9630 | 0.9630 | 0.9630 | 0.9630 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.9672 | 0.9674 | 0.9675 | 0.9675 | 0.9675 | 0.9675 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.9541 | 0.9546 | 0.9547 | 0.9547 | 0.9547 | 0.9547 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.9493 | 0.9497 | 0.9497 | 0.9497 | 0.9497 | 0.9497 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.9563 | 0.9569 | 0.9569 | 0.9570 | 0.9570 | 0.9570 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.9460 | 0.9463 | 0.9464 | 0.9464 | 0.9464 | 0.9464 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.9562 | 0.9568 | 0.9572 | 0.9572 | 0.9572 | 0.9572 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.9565 | 0.9569 | 0.9572 | 0.9572 | 0.9572 | 0.9572 |
| `siglip-image` | 0.9541 | 0.9551 | 0.9553 | 0.9553 | 0.9553 | 0.9553 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.9486 | 0.9495 | 0.9497 | 0.9497 | 0.9497 | 0.9497 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.9484 | 0.9493 | 0.9494 | 0.9494 | 0.9494 | 0.9494 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.9453 | 0.9464 | 0.9466 | 0.9466 | 0.9466 | 0.9466 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.9466 | 0.9474 | 0.9476 | 0.9477 | 0.9477 | 0.9477 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.9599 | 0.9605 | 0.9606 | 0.9606 | 0.9606 | 0.9606 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.9488 | 0.9497 | 0.9498 | 0.9498 | 0.9498 | 0.9498 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.9489 | 0.9498 | 0.9499 | 0.9499 | 0.9499 | 0.9499 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.9596 | 0.9598 | 0.9601 | 0.9602 | 0.9602 | 0.9602 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.9510 | 0.9517 | 0.9519 | 0.9519 | 0.9519 | 0.9519 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.9543 | 0.9546 | 0.9547 | 0.9547 | 0.9547 | 0.9547 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.9474 | 0.9482 | 0.9483 | 0.9483 | 0.9483 | 0.9483 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.9499 | 0.9505 | 0.9507 | 0.9508 | 0.9508 | 0.9508 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.9588 | 0.9593 | 0.9593 | 0.9594 | 0.9594 | 0.9594 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.9651 | 0.9654 | 0.9654 | 0.9654 | 0.9654 | 0.9654 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.9592 | 0.9596 | 0.9596 | 0.9596 | 0.9596 | 0.9596 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.9465 | 0.9473 | 0.9474 | 0.9474 | 0.9474 | 0.9474 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.9460 | 0.9469 | 0.9471 | 0.9471 | 0.9471 | 0.9471 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.9331 | 0.9341 | 0.9343 | 0.9344 | 0.9344 | 0.9344 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.9531 | 0.9534 | 0.9535 | 0.9535 | 0.9535 | 0.9535 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.9331 | 0.9341 | 0.9343 | 0.9344 | 0.9344 | 0.9344 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.9385 | 0.9395 | 0.9395 | 0.9396 | 0.9396 | 0.9396 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.9274 | 0.9284 | 0.9287 | 0.9288 | 0.9288 | 0.9288 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.9445 | 0.9454 | 0.9456 | 0.9456 | 0.9456 | 0.9456 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.9308 | 0.9317 | 0.9319 | 0.9320 | 0.9320 | 0.9320 |
| `text-e5-large-instruct` | 0.9524 | 0.9530 | 0.9533 | 0.9533 | 0.9533 | 0.9533 |
| `text-omni-nano` | 0.9486 | 0.9497 | 0.9498 | 0.9498 | 0.9498 | 0.9498 |
| `text-jina` | 0.9479 | 0.9491 | 0.9492 | 0.9492 | 0.9492 | 0.9492 |
| `text-jina-small` | 0.9519 | 0.9523 | 0.9524 | 0.9526 | 0.9526 | 0.9526 |
| `text-siglip` | 0.9390 | 0.9399 | 0.9399 | 0.9399 | 0.9399 | 0.9399 |
| `attr-siglip` | 0.9163 | 0.9182 | 0.9185 | 0.9185 | 0.9185 | 0.9185 |
| `text-e5-base` | 0.9502 | 0.9508 | 0.9509 | 0.9509 | 0.9510 | 0.9510 |
| `text-e5-small-multi` | 0.9362 | 0.9369 | 0.9369 | 0.9370 | 0.9370 | 0.9370 |
| `attr-omni-nano` | 0.8732 | 0.8753 | 0.8759 | 0.8760 | 0.8760 | 0.8760 |
| `attr-e5-base` | 0.8682 | 0.8712 | 0.8718 | 0.8719 | 0.8719 | 0.8719 |
| `attr-jina` | 0.8749 | 0.8772 | 0.8778 | 0.8779 | 0.8779 | 0.8779 |
| `attr-jina-small` | 0.8735 | 0.8763 | 0.8768 | 0.8770 | 0.8770 | 0.8770 |
| `attr-e5-large-instruct` | 0.8748 | 0.8773 | 0.8779 | 0.8781 | 0.8781 | 0.8781 |
| `attr-e5-small-multi` | 0.8517 | 0.8550 | 0.8557 | 0.8561 | 0.8561 | 0.8561 |
| `random` | 0.8415 | 0.8442 | 0.8445 | 0.8445 | 0.8445 | 0.8445 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.7692 |
| **`fusion[mean cosine] siglip-image + attr-siglip`** | **0.8687** |
| `fusion[z-score average] siglip-image + attr-siglip` | 0.8750 |
| `fusion[RRF (k=60)] siglip-image + text-omni-nano` | 0.8822 |
| `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | 0.8850 |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` | 0.8883 |
| `fusion[RRF (k=60)] siglip-image + text-jina` | 0.8822 |
| `fusion[RRF (k=60)] siglip-image + text-jina-small` | 0.8833 |
| `fusion[RRF (k=60)] siglip-image + attr-siglip` | 0.8676 |
| `fusion[z-score average] siglip-image + text-e5-large-instruct` | 0.8884 |
| `fusion[z-score average] siglip-image + text-omni-nano` | 0.8849 |
| `fusion[z-score average] siglip-image + text-jina` | 0.8848 |
| `fusion[z-score average] siglip-image + text-jina-small` | 0.8868 |
| `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | 0.8808 |
| `fusion[RRF (k=60)] siglip-image + text-siglip` | 0.8672 |
| `fusion[mean cosine] siglip-image + attr-e5-base` | 0.8775 |
| `fusion[z-score average] siglip-image + text-siglip` | 0.8707 |
| `fusion[mean cosine] siglip-image + text-omni-nano` | 0.8793 |
| `fusion[mean cosine] siglip-image + text-jina` | 0.8793 |
| `siglip-image` | 0.8709 |
| `fusion[z-score average] siglip-image + attr-omni-nano` | 0.8754 |
| `fusion[z-score average] siglip-image + attr-e5-base` | 0.8711 |
| `fusion[z-score average] siglip-image + attr-e5-large-instruct` | 0.8744 |
| `fusion[z-score average] siglip-image + attr-jina-small` | 0.8727 |
| `fusion[mean cosine] siglip-image + text-e5-base` | 0.8819 |
| `fusion[z-score average] siglip-image + attr-jina` | 0.8754 |
| `fusion[RRF (k=60)] siglip-image + attr-omni-nano` | 0.8663 |
| `fusion[mean cosine] siglip-image + text-jina-small` | 0.8819 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-large-instruct` | 0.8664 |
| `fusion[mean cosine] siglip-image + text-e5-small-multi` | 0.8824 |
| `fusion[RRF (k=60)] siglip-image + attr-jina` | 0.8665 |
| `fusion[mean cosine] siglip-image + attr-e5-small-multi` | 0.8786 |
| `fusion[RRF (k=60)] siglip-image + text-e5-base` | 0.8786 |
| `fusion[RRF (k=60)] siglip-image + text-e5-small-multi` | 0.8771 |
| `fusion[z-score average] siglip-image + text-e5-base` | 0.8814 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-base` | 0.8623 |
| `fusion[RRF (k=60)] siglip-image + attr-jina-small` | 0.8636 |
| `fusion[mean cosine] siglip-image + attr-omni-nano` | 0.8702 |
| `fusion[z-score average] siglip-image + text-e5-small-multi` | 0.8797 |
| `fusion[mean cosine] siglip-image + attr-jina` | 0.8702 |
| `fusion[mean cosine] siglip-image + text-siglip` | 0.8466 |
| `fusion[mean cosine] siglip-image + attr-jina-small` | 0.8653 |
| `fusion[RRF (k=60)] siglip-image + attr-e5-small-multi` | 0.8585 |
| `fusion[z-score average] siglip-image + attr-e5-small-multi` | 0.8673 |
| `text-e5-large-instruct` | 0.8701 |
| `text-omni-nano` | 0.8668 |
| `text-jina` | 0.8667 |
| `text-jina-small` | 0.8700 |
| `text-siglip` | 0.8241 |
| `attr-siglip` | 0.8301 |
| `text-e5-base` | 0.8609 |
| `text-e5-small-multi` | 0.8536 |
| `attr-omni-nano` | 0.8299 |
| `attr-e5-base` | 0.8246 |
| `attr-jina` | 0.8300 |
| `attr-jina-small` | 0.8263 |
| `attr-e5-large-instruct` | 0.8310 |
| `attr-e5-small-multi` | 0.8170 |
| `random` | 0.7406 |


### Target: `omni-nano-image`

#### Macro-averaged -- every query counts once

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.3853 | 0.4194 | 0.4699 | 0.5514 | 0.6057 | 0.6261 |
| **`text-e5-large-instruct`** | **0.3488** | **0.3950** | **0.4558** | **0.5422** | **0.5944** | **0.6123** |
| `text-jina` | 0.3480 | 0.3932 | 0.4534 | 0.5407 | 0.5935 | 0.6112 |
| `text-omni-nano` | 0.3479 | 0.3931 | 0.4531 | 0.5405 | 0.5934 | 0.6110 |
| `text-jina-small` | 0.3438 | 0.3906 | 0.4523 | 0.5386 | 0.5919 | 0.6096 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.3337 | 0.3826 | 0.4448 | 0.5325 | 0.5857 | 0.6037 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.3342 | 0.3818 | 0.4437 | 0.5324 | 0.5857 | 0.6034 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.3344 | 0.3814 | 0.4437 | 0.5324 | 0.5857 | 0.6035 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.3349 | 0.3796 | 0.4409 | 0.5277 | 0.5824 | 0.6011 |
| `text-siglip` | 0.3367 | 0.3785 | 0.4380 | 0.5247 | 0.5803 | 0.5998 |
| `text-e5-base` | 0.3289 | 0.3736 | 0.4349 | 0.5252 | 0.5797 | 0.5987 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.3244 | 0.3722 | 0.4355 | 0.5242 | 0.5784 | 0.5968 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.3260 | 0.3720 | 0.4344 | 0.5241 | 0.5785 | 0.5969 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.3241 | 0.3718 | 0.4364 | 0.5257 | 0.5795 | 0.5977 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.3259 | 0.3717 | 0.4342 | 0.5238 | 0.5783 | 0.5966 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.3222 | 0.3714 | 0.4329 | 0.5224 | 0.5766 | 0.5952 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.3214 | 0.3701 | 0.4331 | 0.5230 | 0.5775 | 0.5960 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.3219 | 0.3697 | 0.4336 | 0.5237 | 0.5779 | 0.5962 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.3214 | 0.3693 | 0.4334 | 0.5235 | 0.5776 | 0.5959 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.3160 | 0.3651 | 0.4277 | 0.5161 | 0.5714 | 0.5907 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.3119 | 0.3636 | 0.4266 | 0.5160 | 0.5713 | 0.5904 |
| `text-e5-small-multi` | 0.3120 | 0.3609 | 0.4238 | 0.5161 | 0.5716 | 0.5911 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.3115 | 0.3601 | 0.4246 | 0.5160 | 0.5713 | 0.5902 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.3122 | 0.3591 | 0.4233 | 0.5144 | 0.5697 | 0.5887 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.3088 | 0.3572 | 0.4205 | 0.5127 | 0.5687 | 0.5877 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.3094 | 0.3558 | 0.4193 | 0.5102 | 0.5650 | 0.5844 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.3065 | 0.3547 | 0.4172 | 0.5091 | 0.5654 | 0.5847 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.3064 | 0.3544 | 0.4189 | 0.5102 | 0.5646 | 0.5841 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.3047 | 0.3540 | 0.4189 | 0.5099 | 0.5646 | 0.5839 |
| `attr-siglip` | 0.3062 | 0.3529 | 0.4179 | 0.5084 | 0.5629 | 0.5826 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.2980 | 0.3474 | 0.4127 | 0.5051 | 0.5613 | 0.5808 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.2950 | 0.3464 | 0.4140 | 0.5062 | 0.5618 | 0.5809 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.2946 | 0.3463 | 0.4136 | 0.5059 | 0.5615 | 0.5807 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.2927 | 0.3442 | 0.4102 | 0.5024 | 0.5586 | 0.5782 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.2920 | 0.3440 | 0.4125 | 0.5042 | 0.5602 | 0.5794 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.2919 | 0.3436 | 0.4121 | 0.5038 | 0.5593 | 0.5789 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.2920 | 0.3423 | 0.4070 | 0.5003 | 0.5573 | 0.5771 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.2938 | 0.3421 | 0.4074 | 0.4999 | 0.5565 | 0.5766 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.2944 | 0.3416 | 0.4075 | 0.4998 | 0.5565 | 0.5766 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.2931 | 0.3416 | 0.4081 | 0.5002 | 0.5562 | 0.5761 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.2899 | 0.3415 | 0.4101 | 0.5025 | 0.5584 | 0.5780 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.2902 | 0.3414 | 0.4102 | 0.5026 | 0.5586 | 0.5781 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.2875 | 0.3393 | 0.4079 | 0.5009 | 0.5567 | 0.5766 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.2872 | 0.3393 | 0.4066 | 0.4992 | 0.5554 | 0.5753 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.2896 | 0.3377 | 0.4055 | 0.4979 | 0.5539 | 0.5747 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.2853 | 0.3349 | 0.4002 | 0.4928 | 0.5502 | 0.5706 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.2803 | 0.3309 | 0.3991 | 0.4931 | 0.5501 | 0.5702 |
| `attr-jina` | 0.2783 | 0.3289 | 0.3995 | 0.4930 | 0.5505 | 0.5701 |
| `attr-omni-nano` | 0.2769 | 0.3283 | 0.3985 | 0.4924 | 0.5499 | 0.5696 |
| `attr-e5-large-instruct` | 0.2760 | 0.3281 | 0.3989 | 0.4921 | 0.5486 | 0.5685 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.2761 | 0.3275 | 0.3960 | 0.4905 | 0.5477 | 0.5679 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.2740 | 0.3259 | 0.3948 | 0.4897 | 0.5466 | 0.5672 |
| `attr-jina-small` | 0.2719 | 0.3244 | 0.3959 | 0.4895 | 0.5472 | 0.5674 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.2722 | 0.3237 | 0.3911 | 0.4870 | 0.5445 | 0.5651 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.2729 | 0.3221 | 0.3901 | 0.4845 | 0.5425 | 0.5635 |
| `attr-e5-base` | 0.2668 | 0.3185 | 0.3882 | 0.4828 | 0.5415 | 0.5623 |
| `omni-nano-image` | 0.2578 | 0.3072 | 0.3767 | 0.4736 | 0.5331 | 0.5544 |
| `attr-e5-small-multi` | 0.2512 | 0.3024 | 0.3728 | 0.4715 | 0.5309 | 0.5524 |
| `random` | 0.2082 | 0.2534 | 0.3232 | 0.4266 | 0.4927 | 0.5182 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.2333 | 0.3783 | 0.5745 | 0.8234 | 0.9437 | 0.9846 |
| **`text-e5-large-instruct`** | **0.2232** | **0.3765** | **0.5749** | **0.8238** | **0.9436** | **0.9834** |
| `text-jina` | 0.2217 | 0.3752 | 0.5732 | 0.8216 | 0.9433 | 0.9828 |
| `text-omni-nano` | 0.2222 | 0.3746 | 0.5731 | 0.8216 | 0.9434 | 0.9828 |
| `text-jina-small` | 0.2191 | 0.3724 | 0.5736 | 0.8207 | 0.9437 | 0.9829 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.2132 | 0.3689 | 0.5693 | 0.8207 | 0.9439 | 0.9835 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.2114 | 0.3657 | 0.5664 | 0.8203 | 0.9434 | 0.9833 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.2120 | 0.3650 | 0.5664 | 0.8203 | 0.9434 | 0.9833 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.2090 | 0.3577 | 0.5599 | 0.8135 | 0.9413 | 0.9824 |
| `text-siglip` | 0.2087 | 0.3549 | 0.5553 | 0.8087 | 0.9388 | 0.9814 |
| `text-e5-base` | 0.2151 | 0.3621 | 0.5603 | 0.8139 | 0.9393 | 0.9818 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.2073 | 0.3602 | 0.5632 | 0.8181 | 0.9426 | 0.9833 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.2050 | 0.3585 | 0.5600 | 0.8169 | 0.9421 | 0.9829 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.2111 | 0.3597 | 0.5664 | 0.8193 | 0.9431 | 0.9835 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.2045 | 0.3583 | 0.5601 | 0.8168 | 0.9421 | 0.9829 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.2056 | 0.3603 | 0.5616 | 0.8170 | 0.9424 | 0.9831 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.2095 | 0.3615 | 0.5618 | 0.8166 | 0.9426 | 0.9833 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.2083 | 0.3595 | 0.5621 | 0.8173 | 0.9422 | 0.9830 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.2078 | 0.3587 | 0.5622 | 0.8173 | 0.9422 | 0.9831 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.1994 | 0.3525 | 0.5533 | 0.8119 | 0.9395 | 0.9822 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.2001 | 0.3528 | 0.5538 | 0.8120 | 0.9406 | 0.9824 |
| `text-e5-small-multi` | 0.2077 | 0.3593 | 0.5570 | 0.8135 | 0.9390 | 0.9817 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.2043 | 0.3534 | 0.5550 | 0.8144 | 0.9406 | 0.9826 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.2024 | 0.3518 | 0.5551 | 0.8137 | 0.9404 | 0.9823 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.2041 | 0.3546 | 0.5549 | 0.8130 | 0.9407 | 0.9826 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.1935 | 0.3416 | 0.5464 | 0.8084 | 0.9380 | 0.9813 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.1992 | 0.3522 | 0.5529 | 0.8121 | 0.9398 | 0.9821 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.1916 | 0.3446 | 0.5486 | 0.8091 | 0.9376 | 0.9814 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.1897 | 0.3441 | 0.5501 | 0.8094 | 0.9384 | 0.9815 |
| `attr-siglip` | 0.1955 | 0.3429 | 0.5462 | 0.8075 | 0.9349 | 0.9799 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.1930 | 0.3432 | 0.5485 | 0.8088 | 0.9386 | 0.9817 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.1899 | 0.3429 | 0.5511 | 0.8112 | 0.9403 | 0.9829 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.1902 | 0.3427 | 0.5502 | 0.8112 | 0.9402 | 0.9829 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.1877 | 0.3408 | 0.5491 | 0.8079 | 0.9385 | 0.9815 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.1880 | 0.3411 | 0.5526 | 0.8094 | 0.9396 | 0.9825 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.1887 | 0.3405 | 0.5496 | 0.8088 | 0.9380 | 0.9818 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.1905 | 0.3410 | 0.5450 | 0.8068 | 0.9380 | 0.9816 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.1874 | 0.3380 | 0.5422 | 0.8063 | 0.9378 | 0.9818 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.1864 | 0.3364 | 0.5414 | 0.8063 | 0.9380 | 0.9818 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.1871 | 0.3386 | 0.5453 | 0.8070 | 0.9371 | 0.9811 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.1876 | 0.3392 | 0.5469 | 0.8089 | 0.9393 | 0.9826 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.1875 | 0.3383 | 0.5469 | 0.8088 | 0.9394 | 0.9826 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.1866 | 0.3378 | 0.5478 | 0.8079 | 0.9385 | 0.9824 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.1822 | 0.3395 | 0.5448 | 0.8070 | 0.9375 | 0.9817 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.1863 | 0.3342 | 0.5444 | 0.8050 | 0.9366 | 0.9817 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.1809 | 0.3353 | 0.5400 | 0.8035 | 0.9361 | 0.9809 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.1797 | 0.3309 | 0.5401 | 0.8040 | 0.9369 | 0.9813 |
| `attr-jina` | 0.1859 | 0.3346 | 0.5437 | 0.8056 | 0.9375 | 0.9812 |
| `attr-omni-nano` | 0.1876 | 0.3345 | 0.5441 | 0.8054 | 0.9375 | 0.9812 |
| `attr-e5-large-instruct` | 0.1840 | 0.3353 | 0.5465 | 0.8062 | 0.9354 | 0.9801 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.1777 | 0.3292 | 0.5384 | 0.8023 | 0.9357 | 0.9806 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.1780 | 0.3310 | 0.5391 | 0.8045 | 0.9362 | 0.9812 |
| `attr-jina-small` | 0.1811 | 0.3316 | 0.5421 | 0.8042 | 0.9361 | 0.9810 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.1766 | 0.3261 | 0.5339 | 0.8015 | 0.9355 | 0.9806 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.1749 | 0.3278 | 0.5351 | 0.8006 | 0.9348 | 0.9802 |
| `attr-e5-base` | 0.1787 | 0.3256 | 0.5345 | 0.7998 | 0.9344 | 0.9799 |
| `omni-nano-image` | 0.1675 | 0.3133 | 0.5249 | 0.7945 | 0.9320 | 0.9786 |
| `attr-e5-small-multi` | 0.1703 | 0.3176 | 0.5265 | 0.7966 | 0.9319 | 0.9789 |
| `random` | 0.1372 | 0.2735 | 0.4843 | 0.7705 | 0.9184 | 0.9735 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.5294 | 0.5219 | 0.4903 | 0.3880 | 0.2715 | 0.2037 |
| **`text-e5-large-instruct`** | **0.5865** | **0.5618** | **0.5128** | **0.3981** | **0.2734** | **0.2035** |
| `text-jina` | 0.5841 | 0.5574 | 0.5089 | 0.3969 | 0.2733 | 0.2033 |
| `text-omni-nano` | 0.5839 | 0.5572 | 0.5088 | 0.3969 | 0.2733 | 0.2033 |
| `text-jina-small` | 0.5840 | 0.5586 | 0.5109 | 0.3967 | 0.2737 | 0.2034 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.5697 | 0.5490 | 0.5049 | 0.3947 | 0.2736 | 0.2036 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.5655 | 0.5459 | 0.5019 | 0.3942 | 0.2733 | 0.2035 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.5653 | 0.5457 | 0.5018 | 0.3942 | 0.2733 | 0.2035 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.5565 | 0.5332 | 0.4921 | 0.3874 | 0.2713 | 0.2029 |
| `text-siglip` | 0.5576 | 0.5312 | 0.4873 | 0.3836 | 0.2696 | 0.2023 |
| `text-e5-base` | 0.5674 | 0.5447 | 0.4985 | 0.3911 | 0.2711 | 0.2029 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.5625 | 0.5412 | 0.4974 | 0.3906 | 0.2722 | 0.2033 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.5586 | 0.5376 | 0.4941 | 0.3896 | 0.2720 | 0.2032 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.5586 | 0.5394 | 0.4981 | 0.3925 | 0.2729 | 0.2035 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.5587 | 0.5377 | 0.4938 | 0.3897 | 0.2720 | 0.2032 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.5603 | 0.5388 | 0.4955 | 0.3900 | 0.2722 | 0.2033 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.5548 | 0.5372 | 0.4964 | 0.3913 | 0.2728 | 0.2035 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.5528 | 0.5348 | 0.4949 | 0.3911 | 0.2725 | 0.2034 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.5531 | 0.5350 | 0.4949 | 0.3911 | 0.2725 | 0.2034 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.5441 | 0.5252 | 0.4850 | 0.3840 | 0.2701 | 0.2028 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.5363 | 0.5228 | 0.4856 | 0.3856 | 0.2709 | 0.2030 |
| `text-e5-small-multi` | 0.5576 | 0.5384 | 0.4940 | 0.3888 | 0.2706 | 0.2028 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.5484 | 0.5301 | 0.4913 | 0.3888 | 0.2715 | 0.2032 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.5494 | 0.5296 | 0.4889 | 0.3873 | 0.2710 | 0.2029 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.5465 | 0.5300 | 0.4887 | 0.3874 | 0.2715 | 0.2031 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.5376 | 0.5192 | 0.4794 | 0.3820 | 0.2691 | 0.2024 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.5480 | 0.5294 | 0.4862 | 0.3849 | 0.2705 | 0.2028 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.5316 | 0.5179 | 0.4816 | 0.3836 | 0.2692 | 0.2025 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.5337 | 0.5188 | 0.4827 | 0.3842 | 0.2696 | 0.2026 |
| `attr-siglip` | 0.5270 | 0.5149 | 0.4792 | 0.3827 | 0.2680 | 0.2019 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.5301 | 0.5166 | 0.4809 | 0.3832 | 0.2699 | 0.2027 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.5259 | 0.5165 | 0.4838 | 0.3863 | 0.2712 | 0.2033 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.5259 | 0.5166 | 0.4838 | 0.3862 | 0.2712 | 0.2033 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.5235 | 0.5127 | 0.4772 | 0.3821 | 0.2696 | 0.2026 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.5239 | 0.5155 | 0.4827 | 0.3853 | 0.2708 | 0.2032 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.5233 | 0.5128 | 0.4808 | 0.3833 | 0.2697 | 0.2027 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.5272 | 0.5129 | 0.4761 | 0.3815 | 0.2695 | 0.2027 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.5251 | 0.5114 | 0.4749 | 0.3805 | 0.2690 | 0.2027 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.5247 | 0.5108 | 0.4753 | 0.3807 | 0.2691 | 0.2026 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.5285 | 0.5131 | 0.4764 | 0.3798 | 0.2686 | 0.2023 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.5203 | 0.5110 | 0.4798 | 0.3848 | 0.2705 | 0.2032 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.5204 | 0.5112 | 0.4800 | 0.3848 | 0.2705 | 0.2032 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.5192 | 0.5102 | 0.4777 | 0.3838 | 0.2700 | 0.2031 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.5164 | 0.5080 | 0.4756 | 0.3825 | 0.2696 | 0.2027 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.5229 | 0.5078 | 0.4740 | 0.3796 | 0.2684 | 0.2026 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.5178 | 0.5056 | 0.4701 | 0.3776 | 0.2681 | 0.2022 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.5111 | 0.5013 | 0.4710 | 0.3792 | 0.2687 | 0.2025 |
| `attr-jina` | 0.5088 | 0.5021 | 0.4743 | 0.3819 | 0.2696 | 0.2026 |
| `attr-omni-nano` | 0.5080 | 0.5008 | 0.4731 | 0.3817 | 0.2695 | 0.2026 |
| `attr-e5-large-instruct` | 0.5097 | 0.5017 | 0.4738 | 0.3807 | 0.2683 | 0.2020 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.5076 | 0.4982 | 0.4685 | 0.3776 | 0.2679 | 0.2021 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.5089 | 0.5004 | 0.4717 | 0.3802 | 0.2686 | 0.2024 |
| `attr-jina-small` | 0.5050 | 0.4969 | 0.4702 | 0.3799 | 0.2689 | 0.2025 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.5044 | 0.4959 | 0.4667 | 0.3774 | 0.2678 | 0.2022 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.5088 | 0.4988 | 0.4666 | 0.3754 | 0.2671 | 0.2019 |
| `attr-e5-base` | 0.4959 | 0.4907 | 0.4641 | 0.3765 | 0.2676 | 0.2019 |
| `omni-nano-image` | 0.4828 | 0.4769 | 0.4517 | 0.3698 | 0.2652 | 0.2011 |
| `attr-e5-small-multi` | 0.4856 | 0.4809 | 0.4572 | 0.3737 | 0.2662 | 0.2015 |
| `random` | 0.4250 | 0.4252 | 0.4104 | 0.3468 | 0.2570 | 0.1985 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.6352 | 0.6456 | 0.6492 | 0.6506 | 0.6506 | 0.6506 |
| **`text-e5-large-instruct`** | **0.6936** | **0.7036** | **0.7078** | **0.7094** | **0.7095** | **0.7095** |
| `text-jina` | 0.6896 | 0.6997 | 0.7036 | 0.7052 | 0.7053 | 0.7053 |
| `text-omni-nano` | 0.6904 | 0.7004 | 0.7043 | 0.7059 | 0.7060 | 0.7060 |
| `text-jina-small` | 0.6899 | 0.7001 | 0.7045 | 0.7061 | 0.7061 | 0.7061 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.6743 | 0.6850 | 0.6894 | 0.6910 | 0.6910 | 0.6910 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.6700 | 0.6813 | 0.6857 | 0.6874 | 0.6874 | 0.6874 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.6702 | 0.6812 | 0.6857 | 0.6874 | 0.6875 | 0.6875 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.6671 | 0.6776 | 0.6822 | 0.6838 | 0.6839 | 0.6839 |
| `text-siglip` | 0.6689 | 0.6794 | 0.6840 | 0.6856 | 0.6857 | 0.6857 |
| `text-e5-base` | 0.6784 | 0.6883 | 0.6929 | 0.6946 | 0.6947 | 0.6947 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.6703 | 0.6809 | 0.6854 | 0.6870 | 0.6871 | 0.6871 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.6688 | 0.6796 | 0.6842 | 0.6859 | 0.6859 | 0.6859 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.6625 | 0.6726 | 0.6775 | 0.6790 | 0.6791 | 0.6791 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.6690 | 0.6799 | 0.6846 | 0.6862 | 0.6863 | 0.6863 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.6692 | 0.6803 | 0.6847 | 0.6863 | 0.6864 | 0.6864 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.6609 | 0.6716 | 0.6760 | 0.6776 | 0.6777 | 0.6777 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.6619 | 0.6726 | 0.6772 | 0.6788 | 0.6788 | 0.6788 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.6609 | 0.6717 | 0.6763 | 0.6779 | 0.6780 | 0.6780 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.6550 | 0.6666 | 0.6712 | 0.6728 | 0.6729 | 0.6729 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.6436 | 0.6557 | 0.6602 | 0.6618 | 0.6619 | 0.6619 |
| `text-e5-small-multi` | 0.6653 | 0.6760 | 0.6806 | 0.6824 | 0.6824 | 0.6824 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.6524 | 0.6633 | 0.6679 | 0.6697 | 0.6698 | 0.6698 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.6629 | 0.6735 | 0.6781 | 0.6798 | 0.6799 | 0.6799 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.6525 | 0.6633 | 0.6680 | 0.6697 | 0.6698 | 0.6698 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.6458 | 0.6571 | 0.6620 | 0.6638 | 0.6638 | 0.6638 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.6550 | 0.6658 | 0.6706 | 0.6723 | 0.6724 | 0.6724 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.6408 | 0.6531 | 0.6580 | 0.6597 | 0.6597 | 0.6597 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.6375 | 0.6500 | 0.6550 | 0.6567 | 0.6568 | 0.6568 |
| `attr-siglip` | 0.6346 | 0.6463 | 0.6512 | 0.6531 | 0.6531 | 0.6531 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.6320 | 0.6437 | 0.6488 | 0.6506 | 0.6507 | 0.6507 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.6306 | 0.6427 | 0.6482 | 0.6499 | 0.6500 | 0.6500 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.6299 | 0.6419 | 0.6474 | 0.6491 | 0.6492 | 0.6492 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.6267 | 0.6387 | 0.6442 | 0.6458 | 0.6459 | 0.6459 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.6254 | 0.6381 | 0.6436 | 0.6453 | 0.6454 | 0.6454 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.6285 | 0.6409 | 0.6462 | 0.6479 | 0.6480 | 0.6480 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.6285 | 0.6402 | 0.6453 | 0.6471 | 0.6472 | 0.6472 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.6342 | 0.6463 | 0.6512 | 0.6530 | 0.6531 | 0.6531 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.6340 | 0.6464 | 0.6515 | 0.6533 | 0.6534 | 0.6534 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.6347 | 0.6467 | 0.6520 | 0.6538 | 0.6538 | 0.6538 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.6264 | 0.6384 | 0.6438 | 0.6455 | 0.6456 | 0.6456 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.6263 | 0.6383 | 0.6438 | 0.6455 | 0.6456 | 0.6456 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.6202 | 0.6325 | 0.6382 | 0.6399 | 0.6400 | 0.6400 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.6222 | 0.6357 | 0.6409 | 0.6427 | 0.6428 | 0.6428 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.6267 | 0.6390 | 0.6444 | 0.6461 | 0.6462 | 0.6462 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.6212 | 0.6344 | 0.6398 | 0.6415 | 0.6416 | 0.6416 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.6132 | 0.6260 | 0.6317 | 0.6335 | 0.6336 | 0.6336 |
| `attr-jina` | 0.6069 | 0.6195 | 0.6252 | 0.6271 | 0.6272 | 0.6272 |
| `attr-omni-nano` | 0.6066 | 0.6192 | 0.6249 | 0.6268 | 0.6268 | 0.6268 |
| `attr-e5-large-instruct` | 0.6027 | 0.6162 | 0.6219 | 0.6238 | 0.6239 | 0.6239 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.6094 | 0.6222 | 0.6278 | 0.6296 | 0.6298 | 0.6298 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.6098 | 0.6228 | 0.6282 | 0.6302 | 0.6303 | 0.6303 |
| `attr-jina-small` | 0.6039 | 0.6171 | 0.6228 | 0.6247 | 0.6248 | 0.6248 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.6048 | 0.6176 | 0.6232 | 0.6252 | 0.6253 | 0.6253 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.6115 | 0.6246 | 0.6299 | 0.6319 | 0.6320 | 0.6320 |
| `attr-e5-base` | 0.5934 | 0.6064 | 0.6125 | 0.6146 | 0.6147 | 0.6147 |
| `omni-nano-image` | 0.5869 | 0.5999 | 0.6061 | 0.6080 | 0.6082 | 0.6082 |
| `attr-e5-small-multi` | 0.5797 | 0.5933 | 0.5996 | 0.6018 | 0.6019 | 0.6019 |
| `random` | 0.5432 | 0.5571 | 0.5639 | 0.5662 | 0.5663 | 0.5663 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.5979 |
| **`text-e5-large-instruct`** | **0.6129** |
| `text-jina` | 0.6096 |
| `text-omni-nano` | 0.6094 |
| `text-jina-small` | 0.6093 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.5986 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.5964 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.5965 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.5833 |
| `text-siglip` | 0.5786 |
| `text-e5-base` | 0.5964 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.5864 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.5842 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.5899 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.5840 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.5845 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.5878 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.5863 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.5861 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.5696 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.5712 |
| `text-e5-small-multi` | 0.5891 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.5817 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.5771 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.5785 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.5620 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.5734 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.5646 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.5654 |
| `attr-siglip` | 0.5648 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.5661 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.5671 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.5670 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.5615 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.5654 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.5630 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.5619 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.5568 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.5562 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.5566 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.5615 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.5614 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.5604 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.5586 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.5550 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.5503 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.5515 |
| `attr-jina` | 0.5564 |
| `attr-omni-nano` | 0.5563 |
| `attr-e5-large-instruct` | 0.5553 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.5481 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.5505 |
| `attr-jina-small` | 0.5541 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.5451 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.5427 |
| `attr-e5-base` | 0.5481 |
| `omni-nano-image` | 0.5288 |
| `attr-e5-small-multi` | 0.5374 |
| `random` | 0.4757 |


#### Impression-weighted -- every query counts in proportion to its traffic

**NDCG@k**

| System | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@48 | NDCG@96 | NDCG@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.4190 | 0.4239 | 0.4521 | 0.5459 | 0.6611 | 0.7126 |
| **`text-e5-large-instruct`** | **0.3919** | **0.4182** | **0.4745** | **0.5818** | **0.6801** | **0.7215** |
| `text-omni-nano` | 0.3912 | 0.4168 | 0.4722 | 0.5813 | 0.6784 | 0.7193 |
| `text-jina` | 0.3905 | 0.4165 | 0.4707 | 0.5810 | 0.6777 | 0.7187 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.3945 | 0.4147 | 0.4689 | 0.5832 | 0.6803 | 0.7195 |
| `text-jina-small` | 0.3847 | 0.4139 | 0.4705 | 0.5789 | 0.6784 | 0.7199 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.3940 | 0.4136 | 0.4687 | 0.5828 | 0.6799 | 0.7191 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.3911 | 0.4127 | 0.4620 | 0.5711 | 0.6674 | 0.7116 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.3921 | 0.4103 | 0.4589 | 0.5714 | 0.6730 | 0.7142 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.3888 | 0.4084 | 0.4575 | 0.5702 | 0.6717 | 0.7132 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.3842 | 0.4080 | 0.4698 | 0.5802 | 0.6783 | 0.7192 |
| `text-siglip` | 0.3953 | 0.4065 | 0.4464 | 0.5499 | 0.6577 | 0.7057 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.3816 | 0.4061 | 0.4567 | 0.5675 | 0.6699 | 0.7125 |
| `attr-siglip` | 0.3841 | 0.4055 | 0.4558 | 0.5631 | 0.6622 | 0.7063 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.3913 | 0.4052 | 0.4548 | 0.5621 | 0.6654 | 0.7101 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.3777 | 0.4005 | 0.4562 | 0.5675 | 0.6697 | 0.7109 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.3717 | 0.3991 | 0.4548 | 0.5651 | 0.6644 | 0.7069 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.3778 | 0.3986 | 0.4532 | 0.5684 | 0.6688 | 0.7103 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.3769 | 0.3984 | 0.4531 | 0.5686 | 0.6690 | 0.7105 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.3752 | 0.3977 | 0.4530 | 0.5684 | 0.6691 | 0.7107 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.3760 | 0.3971 | 0.4534 | 0.5651 | 0.6623 | 0.7066 |
| `text-e5-base` | 0.3765 | 0.3951 | 0.4476 | 0.5622 | 0.6608 | 0.7055 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.3662 | 0.3923 | 0.4507 | 0.5652 | 0.6673 | 0.7096 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.3661 | 0.3917 | 0.4427 | 0.5500 | 0.6563 | 0.7015 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.3593 | 0.3881 | 0.4397 | 0.5555 | 0.6603 | 0.7027 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.3638 | 0.3877 | 0.4370 | 0.5487 | 0.6518 | 0.6973 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.3585 | 0.3870 | 0.4410 | 0.5495 | 0.6564 | 0.7011 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.3624 | 0.3856 | 0.4445 | 0.5592 | 0.6608 | 0.7040 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.3611 | 0.3854 | 0.4437 | 0.5588 | 0.6606 | 0.7038 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.3577 | 0.3851 | 0.4440 | 0.5582 | 0.6599 | 0.7040 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.3607 | 0.3847 | 0.4375 | 0.5514 | 0.6555 | 0.6996 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.3635 | 0.3834 | 0.4312 | 0.5440 | 0.6479 | 0.6944 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.3618 | 0.3831 | 0.4394 | 0.5548 | 0.6586 | 0.7013 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.3549 | 0.3829 | 0.4328 | 0.5471 | 0.6508 | 0.6959 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.3640 | 0.3828 | 0.4348 | 0.5470 | 0.6506 | 0.6966 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.3525 | 0.3820 | 0.4399 | 0.5533 | 0.6558 | 0.7005 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.3576 | 0.3820 | 0.4326 | 0.5487 | 0.6550 | 0.6994 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.3594 | 0.3815 | 0.4340 | 0.5469 | 0.6481 | 0.6962 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.3552 | 0.3811 | 0.4398 | 0.5532 | 0.6560 | 0.7004 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.3480 | 0.3789 | 0.4374 | 0.5504 | 0.6542 | 0.6989 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.3480 | 0.3785 | 0.4342 | 0.5508 | 0.6563 | 0.6997 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.3530 | 0.3785 | 0.4341 | 0.5493 | 0.6517 | 0.6964 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.3509 | 0.3781 | 0.4334 | 0.5477 | 0.6510 | 0.6957 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.3459 | 0.3773 | 0.4376 | 0.5519 | 0.6549 | 0.6983 |
| `text-e5-small-multi` | 0.3494 | 0.3773 | 0.4308 | 0.5506 | 0.6563 | 0.6983 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.3508 | 0.3754 | 0.4302 | 0.5451 | 0.6495 | 0.6945 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.3468 | 0.3690 | 0.4157 | 0.5325 | 0.6408 | 0.6876 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.3452 | 0.3689 | 0.4264 | 0.5433 | 0.6468 | 0.6918 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.3376 | 0.3688 | 0.4210 | 0.5388 | 0.6436 | 0.6885 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.3425 | 0.3667 | 0.4174 | 0.5370 | 0.6427 | 0.6881 |
| `attr-omni-nano` | 0.3254 | 0.3588 | 0.4191 | 0.5360 | 0.6422 | 0.6871 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.3300 | 0.3570 | 0.4192 | 0.5361 | 0.6420 | 0.6879 |
| `attr-e5-base` | 0.3190 | 0.3539 | 0.4102 | 0.5267 | 0.6328 | 0.6806 |
| `attr-jina` | 0.3217 | 0.3510 | 0.4168 | 0.5351 | 0.6418 | 0.6868 |
| `attr-jina-small` | 0.3232 | 0.3509 | 0.4140 | 0.5286 | 0.6345 | 0.6834 |
| `attr-e5-large-instruct` | 0.3220 | 0.3496 | 0.4155 | 0.5362 | 0.6396 | 0.6869 |
| `omni-nano-image` | 0.3238 | 0.3483 | 0.4011 | 0.5192 | 0.6272 | 0.6743 |
| `attr-e5-small-multi` | 0.2912 | 0.3225 | 0.3828 | 0.5098 | 0.6227 | 0.6698 |
| `random` | 0.2624 | 0.2840 | 0.3375 | 0.4550 | 0.5768 | 0.6343 |

**Recall@k**

| System | Recall@5 | Recall@10 | Recall@20 | Recall@48 | Recall@96 | Recall@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.0762 | 0.1569 | 0.3051 | 0.5903 | 0.8477 | 0.9533 |
| **`text-e5-large-instruct`** | **0.0996** | **0.1933** | **0.3555** | **0.6480** | **0.8659** | **0.9581** |
| `text-omni-nano` | 0.0991 | 0.1925 | 0.3536 | 0.6446 | 0.8632 | 0.9557 |
| `text-jina` | 0.0991 | 0.1924 | 0.3536 | 0.6447 | 0.8631 | 0.9557 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.0966 | 0.1891 | 0.3479 | 0.6405 | 0.8664 | 0.9566 |
| `text-jina-small` | 0.0996 | 0.1931 | 0.3551 | 0.6454 | 0.8672 | 0.9576 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.0966 | 0.1891 | 0.3482 | 0.6405 | 0.8663 | 0.9565 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.0922 | 0.1799 | 0.3335 | 0.6240 | 0.8590 | 0.9539 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.0952 | 0.1851 | 0.3414 | 0.6344 | 0.8636 | 0.9557 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.0949 | 0.1850 | 0.3418 | 0.6343 | 0.8634 | 0.9557 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.0975 | 0.1896 | 0.3500 | 0.6414 | 0.8686 | 0.9578 |
| `text-siglip` | 0.0942 | 0.1813 | 0.3339 | 0.6168 | 0.8534 | 0.9525 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.0958 | 0.1861 | 0.3432 | 0.6350 | 0.8653 | 0.9572 |
| `attr-siglip` | 0.0905 | 0.1781 | 0.3332 | 0.6265 | 0.8560 | 0.9522 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.0948 | 0.1830 | 0.3391 | 0.6273 | 0.8599 | 0.9546 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.0958 | 0.1863 | 0.3437 | 0.6351 | 0.8656 | 0.9566 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.0914 | 0.1799 | 0.3361 | 0.6274 | 0.8606 | 0.9544 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.0942 | 0.1852 | 0.3435 | 0.6358 | 0.8642 | 0.9563 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.0943 | 0.1852 | 0.3440 | 0.6360 | 0.8642 | 0.9561 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.0952 | 0.1863 | 0.3452 | 0.6382 | 0.8672 | 0.9573 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.0914 | 0.1789 | 0.3348 | 0.6271 | 0.8583 | 0.9542 |
| `text-e5-base` | 0.0988 | 0.1918 | 0.3512 | 0.6404 | 0.8605 | 0.9553 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.0947 | 0.1857 | 0.3450 | 0.6370 | 0.8664 | 0.9573 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.0923 | 0.1801 | 0.3347 | 0.6215 | 0.8586 | 0.9542 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.0943 | 0.1851 | 0.3428 | 0.6349 | 0.8631 | 0.9562 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.0904 | 0.1780 | 0.3315 | 0.6230 | 0.8588 | 0.9555 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.0915 | 0.1799 | 0.3355 | 0.6246 | 0.8593 | 0.9547 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.0904 | 0.1793 | 0.3372 | 0.6316 | 0.8646 | 0.9586 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.0904 | 0.1792 | 0.3372 | 0.6316 | 0.8645 | 0.9586 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.0904 | 0.1790 | 0.3373 | 0.6300 | 0.8635 | 0.9582 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.0917 | 0.1804 | 0.3363 | 0.6263 | 0.8596 | 0.9551 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.0901 | 0.1771 | 0.3304 | 0.6209 | 0.8586 | 0.9544 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.0947 | 0.1845 | 0.3405 | 0.6332 | 0.8620 | 0.9555 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.0905 | 0.1785 | 0.3324 | 0.6236 | 0.8584 | 0.9543 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.0903 | 0.1778 | 0.3317 | 0.6230 | 0.8590 | 0.9557 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.0899 | 0.1786 | 0.3357 | 0.6292 | 0.8622 | 0.9572 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.0947 | 0.1836 | 0.3387 | 0.6304 | 0.8619 | 0.9558 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.0903 | 0.1767 | 0.3315 | 0.6216 | 0.8576 | 0.9550 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.0900 | 0.1786 | 0.3358 | 0.6291 | 0.8621 | 0.9571 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.0902 | 0.1779 | 0.3348 | 0.6279 | 0.8617 | 0.9573 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.0939 | 0.1840 | 0.3412 | 0.6330 | 0.8636 | 0.9560 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.0915 | 0.1780 | 0.3333 | 0.6231 | 0.8597 | 0.9548 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.0903 | 0.1776 | 0.3345 | 0.6268 | 0.8613 | 0.9561 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.0905 | 0.1791 | 0.3373 | 0.6279 | 0.8612 | 0.9561 |
| `text-e5-small-multi` | 0.0971 | 0.1890 | 0.3472 | 0.6369 | 0.8608 | 0.9549 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.0912 | 0.1785 | 0.3335 | 0.6243 | 0.8596 | 0.9549 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.0886 | 0.1747 | 0.3270 | 0.6178 | 0.8568 | 0.9543 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.0884 | 0.1752 | 0.3299 | 0.6212 | 0.8594 | 0.9549 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.0879 | 0.1736 | 0.3277 | 0.6186 | 0.8565 | 0.9531 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.0875 | 0.1725 | 0.3262 | 0.6180 | 0.8570 | 0.9539 |
| `attr-omni-nano` | 0.0886 | 0.1760 | 0.3322 | 0.6273 | 0.8587 | 0.9571 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.0887 | 0.1751 | 0.3309 | 0.6245 | 0.8599 | 0.9556 |
| `attr-e5-base` | 0.0875 | 0.1744 | 0.3295 | 0.6235 | 0.8554 | 0.9550 |
| `attr-jina` | 0.0883 | 0.1760 | 0.3327 | 0.6272 | 0.8590 | 0.9571 |
| `attr-jina-small` | 0.0877 | 0.1746 | 0.3298 | 0.6257 | 0.8571 | 0.9564 |
| `attr-e5-large-instruct` | 0.0887 | 0.1757 | 0.3334 | 0.6287 | 0.8580 | 0.9553 |
| `omni-nano-image` | 0.0844 | 0.1672 | 0.3177 | 0.6074 | 0.8480 | 0.9487 |
| `attr-e5-small-multi` | 0.0847 | 0.1696 | 0.3235 | 0.6196 | 0.8562 | 0.9543 |
| `random` | 0.0749 | 0.1516 | 0.2906 | 0.5721 | 0.8265 | 0.9409 |

**Precision@k**

| System | Precision@5 | Precision@10 | Precision@20 | Precision@48 | Precision@96 | Precision@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7164 | 0.7371 | 0.7390 | 0.6841 | 0.5694 | 0.4603 |
| **`text-e5-large-instruct`** | **0.9035** | **0.8958** | **0.8723** | **0.7645** | **0.5877** | **0.4647** |
| `text-omni-nano` | 0.8947 | 0.8909 | 0.8672 | 0.7603 | 0.5854 | 0.4628 |
| `text-jina` | 0.8945 | 0.8901 | 0.8667 | 0.7604 | 0.5854 | 0.4627 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.8775 | 0.8776 | 0.8512 | 0.7547 | 0.5880 | 0.4634 |
| `text-jina-small` | 0.9041 | 0.8948 | 0.8723 | 0.7617 | 0.5893 | 0.4643 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.8769 | 0.8776 | 0.8525 | 0.7546 | 0.5879 | 0.4633 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.8697 | 0.8592 | 0.8302 | 0.7343 | 0.5810 | 0.4615 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.8751 | 0.8677 | 0.8397 | 0.7452 | 0.5854 | 0.4626 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.8711 | 0.8676 | 0.8410 | 0.7450 | 0.5852 | 0.4627 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.8939 | 0.8825 | 0.8593 | 0.7563 | 0.5901 | 0.4643 |
| `text-siglip` | 0.8683 | 0.8521 | 0.8265 | 0.7249 | 0.5765 | 0.4603 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.8842 | 0.8750 | 0.8452 | 0.7461 | 0.5870 | 0.4638 |
| `attr-siglip` | 0.8502 | 0.8522 | 0.8310 | 0.7382 | 0.5798 | 0.4608 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.8814 | 0.8605 | 0.8375 | 0.7390 | 0.5819 | 0.4618 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.8794 | 0.8737 | 0.8467 | 0.7458 | 0.5870 | 0.4632 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.8541 | 0.8532 | 0.8327 | 0.7401 | 0.5825 | 0.4620 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.8650 | 0.8618 | 0.8423 | 0.7474 | 0.5862 | 0.4633 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.8652 | 0.8619 | 0.8437 | 0.7476 | 0.5863 | 0.4632 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.8750 | 0.8693 | 0.8475 | 0.7506 | 0.5889 | 0.4640 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.8533 | 0.8470 | 0.8301 | 0.7395 | 0.5808 | 0.4620 |
| `text-e5-base` | 0.9016 | 0.8922 | 0.8614 | 0.7535 | 0.5833 | 0.4627 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.8716 | 0.8658 | 0.8469 | 0.7491 | 0.5882 | 0.4640 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.8620 | 0.8509 | 0.8277 | 0.7302 | 0.5810 | 0.4614 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.8632 | 0.8607 | 0.8403 | 0.7460 | 0.5851 | 0.4633 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.8484 | 0.8479 | 0.8202 | 0.7316 | 0.5803 | 0.4627 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.8485 | 0.8458 | 0.8282 | 0.7341 | 0.5817 | 0.4622 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.8371 | 0.8451 | 0.8332 | 0.7444 | 0.5865 | 0.4652 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.8367 | 0.8450 | 0.8330 | 0.7445 | 0.5865 | 0.4652 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.8388 | 0.8439 | 0.8349 | 0.7425 | 0.5859 | 0.4651 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.8445 | 0.8446 | 0.8274 | 0.7350 | 0.5814 | 0.4624 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.8348 | 0.8357 | 0.8148 | 0.7283 | 0.5808 | 0.4620 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.8674 | 0.8634 | 0.8373 | 0.7435 | 0.5839 | 0.4626 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.8373 | 0.8379 | 0.8179 | 0.7318 | 0.5803 | 0.4618 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.8485 | 0.8465 | 0.8205 | 0.7315 | 0.5806 | 0.4629 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.8352 | 0.8439 | 0.8290 | 0.7416 | 0.5845 | 0.4641 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.8704 | 0.8603 | 0.8339 | 0.7400 | 0.5840 | 0.4627 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.8435 | 0.8376 | 0.8210 | 0.7294 | 0.5794 | 0.4625 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.8350 | 0.8440 | 0.8295 | 0.7415 | 0.5844 | 0.4641 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.8367 | 0.8401 | 0.8288 | 0.7399 | 0.5843 | 0.4643 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.8621 | 0.8575 | 0.8381 | 0.7446 | 0.5858 | 0.4630 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.8568 | 0.8430 | 0.8244 | 0.7315 | 0.5814 | 0.4623 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.8314 | 0.8329 | 0.8252 | 0.7374 | 0.5841 | 0.4635 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.8352 | 0.8434 | 0.8333 | 0.7387 | 0.5838 | 0.4635 |
| `text-e5-small-multi` | 0.8792 | 0.8758 | 0.8516 | 0.7500 | 0.5839 | 0.4622 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.8421 | 0.8387 | 0.8205 | 0.7330 | 0.5817 | 0.4622 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.8355 | 0.8376 | 0.8119 | 0.7262 | 0.5793 | 0.4622 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.8221 | 0.8256 | 0.8135 | 0.7305 | 0.5818 | 0.4624 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.8194 | 0.8209 | 0.8081 | 0.7268 | 0.5789 | 0.4610 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.8175 | 0.8177 | 0.8063 | 0.7275 | 0.5801 | 0.4619 |
| `attr-omni-nano` | 0.8213 | 0.8314 | 0.8182 | 0.7365 | 0.5819 | 0.4645 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.8287 | 0.8309 | 0.8213 | 0.7363 | 0.5828 | 0.4633 |
| `attr-e5-base` | 0.8048 | 0.8200 | 0.8093 | 0.7321 | 0.5793 | 0.4632 |
| `attr-jina` | 0.8204 | 0.8320 | 0.8203 | 0.7367 | 0.5820 | 0.4645 |
| `attr-jina-small` | 0.8181 | 0.8242 | 0.8110 | 0.7342 | 0.5804 | 0.4643 |
| `attr-e5-large-instruct` | 0.8219 | 0.8287 | 0.8215 | 0.7386 | 0.5812 | 0.4634 |
| `omni-nano-image` | 0.7892 | 0.7922 | 0.7837 | 0.7116 | 0.5708 | 0.4577 |
| `attr-e5-small-multi` | 0.7901 | 0.8023 | 0.8019 | 0.7307 | 0.5804 | 0.4629 |
| `random` | 0.7144 | 0.7245 | 0.7187 | 0.6648 | 0.5524 | 0.4523 |

**MRR@k**

| System | MRR@5 | MRR@10 | MRR@20 | MRR@48 | MRR@96 | MRR@144 |
| --- | --- | --- | --- | --- | --- | --- |
| `production (not comparable -- see note)` | 0.7561 | 0.7618 | 0.7628 | 0.7628 | 0.7628 | 0.7628 |
| **`text-e5-large-instruct`** | **0.9524** | **0.9530** | **0.9533** | **0.9533** | **0.9533** | **0.9533** |
| `text-omni-nano` | 0.9486 | 0.9497 | 0.9498 | 0.9498 | 0.9498 | 0.9498 |
| `text-jina` | 0.9479 | 0.9491 | 0.9492 | 0.9492 | 0.9492 | 0.9492 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.9208 | 0.9217 | 0.9218 | 0.9218 | 0.9218 | 0.9218 |
| `text-jina-small` | 0.9519 | 0.9523 | 0.9524 | 0.9526 | 0.9526 | 0.9526 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.9201 | 0.9210 | 0.9211 | 0.9211 | 0.9211 | 0.9211 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.9259 | 0.9270 | 0.9273 | 0.9273 | 0.9273 | 0.9273 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.9224 | 0.9235 | 0.9235 | 0.9235 | 0.9235 | 0.9235 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.9214 | 0.9224 | 0.9225 | 0.9225 | 0.9225 | 0.9225 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.9270 | 0.9275 | 0.9276 | 0.9276 | 0.9276 | 0.9276 |
| `text-siglip` | 0.9390 | 0.9399 | 0.9399 | 0.9399 | 0.9399 | 0.9399 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.9279 | 0.9286 | 0.9287 | 0.9287 | 0.9287 | 0.9287 |
| `attr-siglip` | 0.9163 | 0.9182 | 0.9185 | 0.9185 | 0.9185 | 0.9185 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.9253 | 0.9262 | 0.9263 | 0.9263 | 0.9263 | 0.9263 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.9230 | 0.9237 | 0.9238 | 0.9238 | 0.9238 | 0.9238 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.9034 | 0.9046 | 0.9050 | 0.9050 | 0.9050 | 0.9050 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.9052 | 0.9063 | 0.9065 | 0.9066 | 0.9066 | 0.9066 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.9065 | 0.9075 | 0.9078 | 0.9078 | 0.9078 | 0.9078 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.9217 | 0.9227 | 0.9228 | 0.9228 | 0.9228 | 0.9228 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.9097 | 0.9110 | 0.9113 | 0.9113 | 0.9113 | 0.9113 |
| `text-e5-base` | 0.9502 | 0.9508 | 0.9509 | 0.9509 | 0.9510 | 0.9510 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.9130 | 0.9138 | 0.9139 | 0.9139 | 0.9139 | 0.9139 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.9170 | 0.9180 | 0.9182 | 0.9182 | 0.9182 | 0.9182 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.9203 | 0.9212 | 0.9213 | 0.9213 | 0.9213 | 0.9213 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.9136 | 0.9150 | 0.9152 | 0.9152 | 0.9152 | 0.9152 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.8947 | 0.8961 | 0.8962 | 0.8963 | 0.8963 | 0.8963 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.8924 | 0.8940 | 0.8943 | 0.8944 | 0.8944 | 0.8944 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.8931 | 0.8947 | 0.8950 | 0.8951 | 0.8951 | 0.8951 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.8978 | 0.9000 | 0.9002 | 0.9003 | 0.9003 | 0.9003 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.8946 | 0.8959 | 0.8960 | 0.8961 | 0.8961 | 0.8961 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.8991 | 0.9011 | 0.9014 | 0.9014 | 0.9014 | 0.9014 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.9361 | 0.9368 | 0.9368 | 0.9368 | 0.9368 | 0.9368 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.8836 | 0.8848 | 0.8850 | 0.8851 | 0.8851 | 0.8851 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.9165 | 0.9180 | 0.9182 | 0.9182 | 0.9182 | 0.9182 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.8971 | 0.8984 | 0.8988 | 0.8988 | 0.8988 | 0.8988 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.9269 | 0.9275 | 0.9277 | 0.9277 | 0.9277 | 0.9277 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.8974 | 0.8993 | 0.8995 | 0.8995 | 0.8995 | 0.8995 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.8959 | 0.8972 | 0.8976 | 0.8976 | 0.8976 | 0.8976 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.8927 | 0.8942 | 0.8947 | 0.8947 | 0.8947 | 0.8947 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.9171 | 0.9184 | 0.9185 | 0.9185 | 0.9185 | 0.9185 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.9083 | 0.9100 | 0.9103 | 0.9103 | 0.9103 | 0.9103 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.8933 | 0.8952 | 0.8956 | 0.8957 | 0.8957 | 0.8957 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.9014 | 0.9026 | 0.9029 | 0.9030 | 0.9030 | 0.9030 |
| `text-e5-small-multi` | 0.9362 | 0.9369 | 0.9369 | 0.9370 | 0.9370 | 0.9370 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.8935 | 0.8948 | 0.8950 | 0.8950 | 0.8950 | 0.8950 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.8903 | 0.8928 | 0.8930 | 0.8930 | 0.8930 | 0.8930 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.8775 | 0.8793 | 0.8797 | 0.8798 | 0.8798 | 0.8798 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.8794 | 0.8809 | 0.8812 | 0.8813 | 0.8813 | 0.8813 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.8828 | 0.8845 | 0.8848 | 0.8849 | 0.8849 | 0.8849 |
| `attr-omni-nano` | 0.8732 | 0.8753 | 0.8759 | 0.8760 | 0.8760 | 0.8760 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.8869 | 0.8893 | 0.8897 | 0.8898 | 0.8898 | 0.8898 |
| `attr-e5-base` | 0.8682 | 0.8712 | 0.8718 | 0.8719 | 0.8719 | 0.8719 |
| `attr-jina` | 0.8749 | 0.8772 | 0.8778 | 0.8779 | 0.8779 | 0.8779 |
| `attr-jina-small` | 0.8735 | 0.8763 | 0.8768 | 0.8770 | 0.8770 | 0.8770 |
| `attr-e5-large-instruct` | 0.8748 | 0.8773 | 0.8779 | 0.8781 | 0.8781 | 0.8781 |
| `omni-nano-image` | 0.8631 | 0.8652 | 0.8655 | 0.8656 | 0.8656 | 0.8656 |
| `attr-e5-small-multi` | 0.8517 | 0.8550 | 0.8557 | 0.8561 | 0.8561 | 0.8561 |
| `random` | 0.8415 | 0.8442 | 0.8445 | 0.8445 | 0.8445 | 0.8445 |

**MAP** (rank-position metric, no cutoff)

| System | MAP |
| --- | --- |
| `production (not comparable -- see note)` | 0.7692 |
| **`text-e5-large-instruct`** | **0.8701** |
| `text-omni-nano` | 0.8668 |
| `text-jina` | 0.8667 |
| `fusion[mean cosine] omni-nano-image + text-omni-nano` | 0.8571 |
| `text-jina-small` | 0.8700 |
| `fusion[mean cosine] omni-nano-image + text-jina` | 0.8570 |
| `fusion[RRF (k=60)] omni-nano-image + attr-siglip` | 0.8283 |
| `fusion[RRF (k=60)] omni-nano-image + text-omni-nano` | 0.8442 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina` | 0.8440 |
| `fusion[mean cosine] omni-nano-image + text-jina-small` | 0.8603 |
| `text-siglip` | 0.8241 |
| `fusion[RRF (k=60)] omni-nano-image + text-jina-small` | 0.8465 |
| `attr-siglip` | 0.8301 |
| `fusion[mean cosine] omni-nano-image + text-siglip` | 0.8356 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-large-instruct` | 0.8476 |
| `fusion[mean cosine] omni-nano-image + attr-siglip` | 0.8327 |
| `fusion[z-score average] omni-nano-image + text-omni-nano` | 0.8469 |
| `fusion[z-score average] omni-nano-image + text-jina` | 0.8470 |
| `fusion[z-score average] omni-nano-image + text-e5-large-instruct` | 0.8511 |
| `fusion[z-score average] omni-nano-image + attr-siglip` | 0.8306 |
| `text-e5-base` | 0.8609 |
| `fusion[z-score average] omni-nano-image + text-jina-small` | 0.8493 |
| `fusion[RRF (k=60)] omni-nano-image + text-siglip` | 0.8255 |
| `fusion[z-score average] omni-nano-image + text-e5-base` | 0.8449 |
| `fusion[RRF (k=60)] omni-nano-image + attr-omni-nano` | 0.8260 |
| `fusion[z-score average] omni-nano-image + text-siglip` | 0.8277 |
| `fusion[mean cosine] omni-nano-image + attr-omni-nano` | 0.8388 |
| `fusion[mean cosine] omni-nano-image + attr-jina` | 0.8388 |
| `fusion[mean cosine] omni-nano-image + attr-jina-small` | 0.8389 |
| `fusion[mean cosine] omni-nano-image + text-e5-base` | 0.8320 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-base` | 0.8232 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-base` | 0.8416 |
| `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | 0.8266 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina` | 0.8260 |
| `fusion[z-score average] omni-nano-image + attr-omni-nano` | 0.8338 |
| `fusion[RRF (k=60)] omni-nano-image + text-e5-small-multi` | 0.8389 |
| `fusion[RRF (k=60)] omni-nano-image + attr-jina-small` | 0.8243 |
| `fusion[z-score average] omni-nano-image + attr-jina` | 0.8339 |
| `fusion[z-score average] omni-nano-image + attr-jina-small` | 0.8333 |
| `fusion[z-score average] omni-nano-image + text-e5-small-multi` | 0.8432 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-large-instruct` | 0.8273 |
| `fusion[z-score average] omni-nano-image + attr-e5-base` | 0.8320 |
| `fusion[z-score average] omni-nano-image + attr-e5-large-instruct` | 0.8349 |
| `text-e5-small-multi` | 0.8536 |
| `fusion[mean cosine] omni-nano-image + text-e5-small-multi` | 0.8282 |
| `fusion[RRF (k=60)] omni-nano-image + attr-e5-small-multi` | 0.8184 |
| `fusion[mean cosine] omni-nano-image + attr-e5-base` | 0.8224 |
| `fusion[mean cosine] omni-nano-image + attr-e5-large-instruct` | 0.8177 |
| `fusion[mean cosine] omni-nano-image + attr-e5-small-multi` | 0.8169 |
| `attr-omni-nano` | 0.8299 |
| `fusion[z-score average] omni-nano-image + attr-e5-small-multi` | 0.8267 |
| `attr-e5-base` | 0.8246 |
| `attr-jina` | 0.8300 |
| `attr-jina-small` | 0.8263 |
| `attr-e5-large-instruct` | 0.8310 |
| `omni-nano-image` | 0.7970 |
| `attr-e5-small-multi` | 0.8170 |
| `random` | 0.7406 |


---

## 3. Which text representation gives the highest fusion gain?

"Gain vs image alone" = fused NDCG@10 - that target image alone.
"Gain vs text alone" = fused NDCG@10 - that text representation's own NDCG@10 (i.e. does fusing
with the image actually help over just using the text system by itself). One pair of tables per
target image modality.


### Target: `siglip-image` (image alone: macro 0.4103, impression-weighted 0.4516)

#### Macro-averaged

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **text-e5-large-instruct** | **mean cosine** | **0.4340** | **+0.0238** | **+0.0390** |
| text-e5-large-instruct | z-score average | 0.4309 | +0.0206 | +0.0358 |
| text-siglip | z-score average | 0.4306 | +0.0203 | +0.0521 |
| text-e5-large-instruct | RRF (k=60) | 0.4291 | +0.0188 | +0.0341 |
| text-omni-nano | z-score average | 0.4289 | +0.0187 | +0.0358 |
| text-jina | z-score average | 0.4289 | +0.0186 | +0.0357 |
| text-jina-small | z-score average | 0.4274 | +0.0171 | +0.0368 |
| text-jina | RRF (k=60) | 0.4272 | +0.0170 | +0.0340 |
| text-siglip | RRF (k=60) | 0.4272 | +0.0169 | +0.0487 |
| text-omni-nano | RRF (k=60) | 0.4266 | +0.0163 | +0.0335 |
| text-jina-small | RRF (k=60) | 0.4262 | +0.0159 | +0.0356 |
| attr-e5-large-instruct | mean cosine | 0.4208 | +0.0105 | +0.0927 |
| text-e5-base | mean cosine | 0.4198 | +0.0095 | +0.0462 |
| attr-siglip | z-score average | 0.4189 | +0.0086 | +0.0660 |
| text-jina | mean cosine | 0.4188 | +0.0085 | +0.0256 |
| text-omni-nano | mean cosine | 0.4187 | +0.0084 | +0.0255 |
| text-e5-small-multi | mean cosine | 0.4185 | +0.0082 | +0.0576 |
| text-jina-small | mean cosine | 0.4165 | +0.0062 | +0.0259 |
| text-e5-base | z-score average | 0.4164 | +0.0061 | +0.0428 |
| text-e5-base | RRF (k=60) | 0.4147 | +0.0044 | +0.0411 |
| attr-siglip | mean cosine | 0.4139 | +0.0037 | +0.0610 |
| text-e5-small-multi | z-score average | 0.4123 | +0.0020 | +0.0514 |
| attr-e5-base | mean cosine | 0.4114 | +0.0011 | +0.0929 |
| text-e5-small-multi | RRF (k=60) | 0.4105 | +0.0002 | +0.0496 |
| text-siglip | mean cosine | 0.4096 | -0.0007 | +0.0311 |
| attr-siglip | RRF (k=60) | 0.4090 | -0.0013 | +0.0561 |
| attr-jina | z-score average | 0.4088 | -0.0014 | +0.0800 |
| attr-omni-nano | z-score average | 0.4084 | -0.0018 | +0.0801 |
| attr-e5-large-instruct | z-score average | 0.4080 | -0.0023 | +0.0799 |
| attr-e5-small-multi | mean cosine | 0.4077 | -0.0026 | +0.1053 |
| attr-jina-small | z-score average | 0.4035 | -0.0067 | +0.0791 |
| attr-e5-base | z-score average | 0.4027 | -0.0076 | +0.0842 |
| attr-omni-nano | mean cosine | 0.3985 | -0.0118 | +0.0702 |
| attr-jina | mean cosine | 0.3985 | -0.0118 | +0.0696 |
| attr-e5-large-instruct | RRF (k=60) | 0.3983 | -0.0120 | +0.0703 |
| attr-omni-nano | RRF (k=60) | 0.3981 | -0.0121 | +0.0698 |
| attr-jina | RRF (k=60) | 0.3981 | -0.0122 | +0.0692 |
| attr-jina-small | RRF (k=60) | 0.3952 | -0.0150 | +0.0708 |
| attr-jina-small | mean cosine | 0.3924 | -0.0179 | +0.0680 |
| attr-e5-base | RRF (k=60) | 0.3902 | -0.0200 | +0.0718 |
| attr-e5-small-multi | z-score average | 0.3887 | -0.0216 | +0.0862 |
| attr-e5-small-multi | RRF (k=60) | 0.3798 | -0.0305 | +0.0773 |

#### Impression-weighted

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **attr-siglip** | **mean cosine** | **0.4737** | **+0.0220** | **+0.0681** |
| attr-siglip | z-score average | 0.4736 | +0.0220 | +0.0681 |
| text-omni-nano | RRF (k=60) | 0.4718 | +0.0201 | +0.0549 |
| text-e5-large-instruct | RRF (k=60) | 0.4714 | +0.0197 | +0.0532 |
| text-e5-large-instruct | mean cosine | 0.4713 | +0.0197 | +0.0531 |
| text-jina | RRF (k=60) | 0.4709 | +0.0192 | +0.0543 |
| text-jina-small | RRF (k=60) | 0.4701 | +0.0185 | +0.0562 |
| attr-siglip | RRF (k=60) | 0.4697 | +0.0181 | +0.0642 |
| text-e5-large-instruct | z-score average | 0.4694 | +0.0178 | +0.0512 |
| text-omni-nano | z-score average | 0.4666 | +0.0149 | +0.0498 |
| text-jina | z-score average | 0.4660 | +0.0144 | +0.0495 |
| text-jina-small | z-score average | 0.4657 | +0.0141 | +0.0518 |
| attr-e5-large-instruct | mean cosine | 0.4654 | +0.0137 | +0.1158 |
| text-siglip | RRF (k=60) | 0.4607 | +0.0091 | +0.0542 |
| attr-e5-base | mean cosine | 0.4568 | +0.0052 | +0.1029 |
| text-siglip | z-score average | 0.4567 | +0.0051 | +0.0502 |
| text-omni-nano | mean cosine | 0.4528 | +0.0012 | +0.0360 |
| text-jina | mean cosine | 0.4518 | +0.0002 | +0.0353 |
| attr-omni-nano | z-score average | 0.4486 | -0.0030 | +0.0898 |
| attr-e5-base | z-score average | 0.4483 | -0.0033 | +0.0944 |
| attr-e5-large-instruct | z-score average | 0.4482 | -0.0035 | +0.0986 |
| attr-jina-small | z-score average | 0.4479 | -0.0037 | +0.0970 |
| text-e5-base | mean cosine | 0.4478 | -0.0039 | +0.0527 |
| attr-jina | z-score average | 0.4470 | -0.0046 | +0.0961 |
| attr-omni-nano | RRF (k=60) | 0.4470 | -0.0046 | +0.0883 |
| text-jina-small | mean cosine | 0.4449 | -0.0067 | +0.0310 |
| attr-e5-large-instruct | RRF (k=60) | 0.4445 | -0.0071 | +0.0949 |
| text-e5-small-multi | mean cosine | 0.4441 | -0.0076 | +0.0668 |
| attr-jina | RRF (k=60) | 0.4438 | -0.0079 | +0.0928 |
| attr-e5-small-multi | mean cosine | 0.4430 | -0.0086 | +0.1205 |
| text-e5-base | RRF (k=60) | 0.4425 | -0.0091 | +0.0474 |
| text-e5-small-multi | RRF (k=60) | 0.4420 | -0.0097 | +0.0647 |
| text-e5-base | z-score average | 0.4418 | -0.0098 | +0.0467 |
| attr-e5-base | RRF (k=60) | 0.4408 | -0.0109 | +0.0868 |
| attr-jina-small | RRF (k=60) | 0.4402 | -0.0114 | +0.0893 |
| attr-omni-nano | mean cosine | 0.4395 | -0.0121 | +0.0808 |
| text-e5-small-multi | z-score average | 0.4375 | -0.0141 | +0.0603 |
| attr-jina | mean cosine | 0.4373 | -0.0143 | +0.0863 |
| text-siglip | mean cosine | 0.4327 | -0.0189 | +0.0262 |
| attr-jina-small | mean cosine | 0.4280 | -0.0236 | +0.0772 |
| attr-e5-small-multi | RRF (k=60) | 0.4250 | -0.0266 | +0.1025 |
| attr-e5-small-multi | z-score average | 0.4207 | -0.0309 | +0.0982 |

### Target: `omni-nano-image` (image alone: macro 0.3072, impression-weighted 0.3483)

#### Macro-averaged

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **text-jina-small** | **mean cosine** | **0.3826** | **+0.0754** | **-0.0080** |
| text-omni-nano | mean cosine | 0.3818 | +0.0746 | -0.0113 |
| text-jina | mean cosine | 0.3814 | +0.0742 | -0.0118 |
| text-siglip | mean cosine | 0.3796 | +0.0724 | +0.0011 |
| text-e5-large-instruct | RRF (k=60) | 0.3722 | +0.0650 | -0.0228 |
| text-jina | RRF (k=60) | 0.3720 | +0.0648 | -0.0212 |
| text-e5-large-instruct | z-score average | 0.3718 | +0.0646 | -0.0232 |
| text-omni-nano | RRF (k=60) | 0.3717 | +0.0645 | -0.0214 |
| text-jina-small | RRF (k=60) | 0.3714 | +0.0642 | -0.0192 |
| text-jina-small | z-score average | 0.3701 | +0.0629 | -0.0205 |
| text-jina | z-score average | 0.3697 | +0.0625 | -0.0235 |
| text-omni-nano | z-score average | 0.3693 | +0.0621 | -0.0238 |
| text-siglip | RRF (k=60) | 0.3651 | +0.0579 | -0.0134 |
| text-siglip | z-score average | 0.3636 | +0.0564 | -0.0149 |
| text-e5-base | z-score average | 0.3601 | +0.0529 | -0.0134 |
| text-e5-base | RRF (k=60) | 0.3591 | +0.0519 | -0.0145 |
| text-e5-small-multi | z-score average | 0.3572 | +0.0500 | -0.0037 |
| attr-siglip | RRF (k=60) | 0.3558 | +0.0486 | +0.0029 |
| text-e5-small-multi | RRF (k=60) | 0.3547 | +0.0475 | -0.0062 |
| attr-siglip | z-score average | 0.3544 | +0.0472 | +0.0015 |
| attr-siglip | mean cosine | 0.3540 | +0.0468 | +0.0011 |
| text-e5-base | mean cosine | 0.3474 | +0.0402 | -0.0261 |
| attr-jina | mean cosine | 0.3464 | +0.0392 | +0.0176 |
| attr-omni-nano | mean cosine | 0.3463 | +0.0391 | +0.0180 |
| text-e5-large-instruct | mean cosine | 0.3442 | +0.0370 | -0.0508 |
| attr-jina-small | mean cosine | 0.3440 | +0.0368 | +0.0196 |
| attr-e5-large-instruct | z-score average | 0.3436 | +0.0364 | +0.0156 |
| text-e5-small-multi | mean cosine | 0.3423 | +0.0351 | -0.0186 |
| attr-omni-nano | RRF (k=60) | 0.3421 | +0.0349 | +0.0138 |
| attr-jina | RRF (k=60) | 0.3416 | +0.0344 | +0.0127 |
| attr-e5-large-instruct | RRF (k=60) | 0.3416 | +0.0343 | +0.0135 |
| attr-omni-nano | z-score average | 0.3415 | +0.0343 | +0.0132 |
| attr-jina | z-score average | 0.3414 | +0.0342 | +0.0125 |
| attr-jina-small | z-score average | 0.3393 | +0.0321 | +0.0149 |
| attr-e5-base | z-score average | 0.3393 | +0.0321 | +0.0209 |
| attr-jina-small | RRF (k=60) | 0.3377 | +0.0304 | +0.0133 |
| attr-e5-base | RRF (k=60) | 0.3349 | +0.0277 | +0.0165 |
| attr-e5-base | mean cosine | 0.3309 | +0.0237 | +0.0124 |
| attr-e5-large-instruct | mean cosine | 0.3275 | +0.0203 | -0.0005 |
| attr-e5-small-multi | z-score average | 0.3259 | +0.0187 | +0.0235 |
| attr-e5-small-multi | mean cosine | 0.3237 | +0.0165 | +0.0213 |
| attr-e5-small-multi | RRF (k=60) | 0.3221 | +0.0149 | +0.0197 |

#### Impression-weighted

| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |
| --- | --- | --- | --- | --- |
| **text-omni-nano** | **mean cosine** | **0.4147** | **+0.0664** | **-0.0021** |
| text-jina | mean cosine | 0.4136 | +0.0653 | -0.0030 |
| attr-siglip | RRF (k=60) | 0.4127 | +0.0644 | +0.0072 |
| text-omni-nano | RRF (k=60) | 0.4103 | +0.0621 | -0.0065 |
| text-jina | RRF (k=60) | 0.4084 | +0.0601 | -0.0081 |
| text-jina-small | mean cosine | 0.4080 | +0.0598 | -0.0059 |
| text-jina-small | RRF (k=60) | 0.4061 | +0.0578 | -0.0078 |
| text-siglip | mean cosine | 0.4052 | +0.0570 | -0.0013 |
| text-e5-large-instruct | RRF (k=60) | 0.4005 | +0.0523 | -0.0177 |
| attr-siglip | mean cosine | 0.3991 | +0.0508 | -0.0064 |
| text-omni-nano | z-score average | 0.3986 | +0.0504 | -0.0182 |
| text-jina | z-score average | 0.3984 | +0.0501 | -0.0181 |
| text-e5-large-instruct | z-score average | 0.3977 | +0.0494 | -0.0205 |
| attr-siglip | z-score average | 0.3971 | +0.0488 | -0.0085 |
| text-jina-small | z-score average | 0.3923 | +0.0441 | -0.0216 |
| text-siglip | RRF (k=60) | 0.3917 | +0.0434 | -0.0148 |
| text-e5-base | z-score average | 0.3881 | +0.0398 | -0.0070 |
| attr-omni-nano | RRF (k=60) | 0.3877 | +0.0395 | +0.0290 |
| text-siglip | z-score average | 0.3870 | +0.0387 | -0.0195 |
| attr-omni-nano | mean cosine | 0.3856 | +0.0373 | +0.0269 |
| attr-jina | mean cosine | 0.3854 | +0.0371 | +0.0344 |
| attr-jina-small | mean cosine | 0.3851 | +0.0368 | +0.0342 |
| text-e5-base | mean cosine | 0.3847 | +0.0364 | -0.0104 |
| attr-e5-base | RRF (k=60) | 0.3834 | +0.0351 | +0.0294 |
| text-e5-base | RRF (k=60) | 0.3831 | +0.0348 | -0.0120 |
| text-e5-large-instruct | mean cosine | 0.3829 | +0.0346 | -0.0353 |
| attr-jina | RRF (k=60) | 0.3828 | +0.0345 | +0.0318 |
| attr-omni-nano | z-score average | 0.3820 | +0.0337 | +0.0232 |
| text-e5-small-multi | RRF (k=60) | 0.3820 | +0.0337 | +0.0047 |
| attr-jina-small | RRF (k=60) | 0.3815 | +0.0333 | +0.0307 |
| attr-jina | z-score average | 0.3811 | +0.0328 | +0.0301 |
| attr-jina-small | z-score average | 0.3789 | +0.0306 | +0.0280 |
| text-e5-small-multi | z-score average | 0.3785 | +0.0303 | +0.0013 |
| attr-e5-large-instruct | RRF (k=60) | 0.3785 | +0.0303 | +0.0290 |
| attr-e5-base | z-score average | 0.3781 | +0.0299 | +0.0242 |
| attr-e5-large-instruct | z-score average | 0.3773 | +0.0290 | +0.0277 |
| text-e5-small-multi | mean cosine | 0.3754 | +0.0271 | -0.0019 |
| attr-e5-small-multi | RRF (k=60) | 0.3690 | +0.0208 | +0.0465 |
| attr-e5-base | mean cosine | 0.3689 | +0.0206 | +0.0150 |
| attr-e5-large-instruct | mean cosine | 0.3688 | +0.0205 | +0.0192 |
| attr-e5-small-multi | mean cosine | 0.3667 | +0.0184 | +0.0442 |
| attr-e5-small-multi | z-score average | 0.3570 | +0.0087 | +0.0345 |

---

## 4. Significance

Paired bootstrap over queries (2000 resamples), metric NDCG@10, built around
whichever fusion combo actually wins on macro NDCG@10 (`mean cosine of siglip-image + text-e5-large-instruct`).
Macro and weighted deltas come from the same paired samples, so a contrast can be significant under
one weighting and not the other. Includes a contrast against the *other* target image modality,
holding the fusion method and text partner fixed, to check whether the image-encoder choice matters.

| Contrast | Note | Macro delta | Macro p | Weighted delta | Weighted p |
| --- | --- | --- | --- | --- | --- |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `siglip-image` | headline: best fusion vs its image alone | +0.0238 | 0.0000 (**significant**) | +0.0197 | 0.0030 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `text-e5-large-instruct` | headline: best fusion vs its text representation alone | +0.0390 | 0.0000 (**significant**) | +0.0531 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[RRF (k=60)] siglip-image + text-e5-large-instruct` | method: mean_cosine vs rrf, image=siglip_image, text=e5_large_instruct_text | +0.0050 | 0.0000 (**significant**) | -0.0000 | 0.9970 (not significant) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[z-score average] siglip-image + text-e5-large-instruct` | method: mean_cosine vs zscore_avg, image=siglip_image, text=e5_large_instruct_text | +0.0032 | 0.0010 (**significant**) | +0.0019 | 0.5680 (not significant) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + text-siglip` | text: e5_large_instruct_text vs siglip_text, image=siglip_image, method=mean_cosine | +0.0245 | 0.0000 (**significant**) | +0.0386 | 0.0010 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + text-jina` | text: e5_large_instruct_text vs jina_text, image=siglip_image, method=mean_cosine | +0.0153 | 0.0000 (**significant**) | +0.0195 | 0.0100 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + text-jina-small` | text: e5_large_instruct_text vs jina_small_text, image=siglip_image, method=mean_cosine | +0.0175 | 0.0000 (**significant**) | +0.0264 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-siglip` | text: e5_large_instruct_text vs siglip_attr, image=siglip_image, method=mean_cosine | +0.0201 | 0.0000 (**significant**) | -0.0024 | 0.7290 (not significant) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-jina` | text: e5_large_instruct_text vs jina_attr, image=siglip_image, method=mean_cosine | +0.0356 | 0.0000 (**significant**) | +0.0340 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-jina-small` | text: e5_large_instruct_text vs jina_small_attr, image=siglip_image, method=mean_cosine | +0.0417 | 0.0000 (**significant**) | +0.0433 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + text-e5-base` | text: e5_large_instruct_text vs e5_base_text, image=siglip_image, method=mean_cosine | +0.0143 | 0.0000 (**significant**) | +0.0236 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-e5-base` | text: e5_large_instruct_text vs e5_base_attr, image=siglip_image, method=mean_cosine | +0.0227 | 0.0000 (**significant**) | +0.0145 | 0.0180 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + text-e5-small-multi` | text: e5_large_instruct_text vs e5_small_multi_text, image=siglip_image, method=mean_cosine | +0.0155 | 0.0000 (**significant**) | +0.0273 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-e5-small-multi` | text: e5_large_instruct_text vs e5_small_multi_attr, image=siglip_image, method=mean_cosine | +0.0263 | 0.0000 (**significant**) | +0.0283 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-e5-large-instruct` | text: e5_large_instruct_text vs e5_large_instruct_attr, image=siglip_image, method=mean_cosine | +0.0132 | 0.0000 (**significant**) | +0.0059 | 0.3710 (not significant) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + text-omni-nano` | text: e5_large_instruct_text vs omni_nano_text, image=siglip_image, method=mean_cosine | +0.0154 | 0.0000 (**significant**) | +0.0185 | 0.0070 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] siglip-image + attr-omni-nano` | text: e5_large_instruct_text vs omni_nano_attr, image=siglip_image, method=mean_cosine | +0.0355 | 0.0000 (**significant**) | +0.0318 | 0.0000 (**significant**) |
| `fusion[mean cosine] siglip-image + text-e5-large-instruct` vs `fusion[mean cosine] omni-nano-image + text-e5-large-instruct` | image: siglip_image vs omni_nano_image, text=e5_large_instruct_text, method=mean_cosine | +0.0899 | 0.0000 (**significant**) | +0.0884 | 0.0000 (**significant**) |

---

## 5. Reading the result

**No fusion method dominates.** Once tied scores can no longer leak the label order, the three
methods land close together and the winner depends on the pairing rather than on the method alone.
W8 tests this directly across image encoders and finds the best method changes with the encoder —
so "use RRF" is not supportable as a general rule from this evidence.

**A W6 add-on model is now the best fusion partner.** `text-e5-large-instruct` overtakes every original SigLIP/Jina text representation (+0.0153 NDCG@10 vs the same method/image paired with `text-jina`).

**`siglip-image` is the stronger target modality here.** Every winning fusion combo in this report pairs its text partner with `siglip-image` (`omni-nano-image` scores 0.3442 under the same method and text partner).

**Fusion can make things worse than the text alone -- when the *image* side is the weak link.** `siglip-image` (alone: 0.4103) underperforms its own text partner alone in 0/14 cases; `omni-nano-image` (alone: 0.3072) underperforms its own text partner alone in 6/14 cases. The pattern tracks each image encoder's own standalone strength: `siglip-image` (0.4103 alone) is close in quality to its text partners, so averaging the two is complementary; a much weaker image tower instead acts like added noise once the text partner is already strong, pulling the fused ranking below what the text system achieves by itself. Check `omni-nano-image`'s own standalone NDCG@10 against a text candidate's before assuming fusion can only help.

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
