# Weekly Report Series — Image and Text Embeddings for Search Relevance

DS eCom Search Ranking · 2026

Four self-contained reports, each covering roughly one week of work. Read in order; each states what
it hands to the next.

| # | Report | Question | Headline outcome |
| --- | --- | --- | --- |
| [W1](W1_data_and_evaluation_setup.md) | Data collection and evaluation setup | Can we measure embedding quality against production relevance at all? | Harness built; baseline modality contrast **inconclusive** (+5.5%, p = 0.102) · **§5 conclusion retracted — see note** |
| [W2](W2_image_localization.md) | Image localization | Is the image signal being degraded before it reaches the encoder? | 25.5% of images crop to a **person**; fixing it is worth **+0.9%, p = 0.39** |
| [W3](W3_complementarity_and_routing.md) | Modality complementarity and routing | Should we choose a modality, or combine them? | Oracle **+16.3%**; fusion captures 29.5%; both routers **fail** |
| [W4](W4_evaluation_validity_and_systems.md) | Evaluation validity and systems cost | Do we trust the labels, and what does this cost to run? | Position-leakage defect found **and fixed** (−0.29 → −0.18); query cost 7–17 ms |
| [W5](W5%20query%20tier%20results.md) | Query-tier results | Does performance hold on the long tail? | Stratified test set: 328 head / 1,635 torso / 4,573 tail; all systems degrade on tail |
| [W6](W6_fusion_text_representation.md) | Fusion method and text-representation sweep | Which text representation and fusion method give the largest gain? | **No method dominates** after the tie fix; earlier "RRF wins" conclusion **corrected** |
| [W7](W7_image_encoder_comparison.md) | Image-encoder comparison | Do the image-side conclusions hold with a second image encoder? | **SigLIP 0.4272 vs omni-nano 0.3313**; omni-small **degenerate and excluded** |
| [W8](W8_fusion_across_image_encoders.md) | Fusion across image encoders | Does the fusion recipe hold when the image tower changes? | Best method **changes with the encoder** — no general rule |

> ## ⚠️ Correction notice (2026-08-20)
> A tie-ordering bug in the evaluation harness inflated every system that produced tied scores.
> Because the Big-4 attribute string has only 17.8% distinct values, **64.1%** of candidates in a
> typical pool tied, and the label-sorted row order leaked through a stable sort. This invalidated
> W1 §5's headline ("structured attributes win decisively") and W6's fusion conclusions. Fixed by
> shuffling pools before ranking; all results regenerated. `attr-siglip` fell **−0.159 NDCG@10**
> and now ranks last among real systems.

**Overall recommendation, revised:** do **not** adopt the Big-4 attribute representation. Fusion
remains the only intervention that reliably beats its own image input, but no fusion method
generalises across image encoders. Treat image preprocessing and modality routing as unresolved.

All figures trace to `results/` in the parent directory. Reproduction instructions are in each
report's appendix.
