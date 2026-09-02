# W7 — Jina CLIP v2 comparison

Generated 2026-08-28 by `17_embed_jina_clip_v2_w7.py` and `18_compare_jina_clip_v2_w7.py`.

## Scope

This is a scoped W7 comparison using `jinaai/jina-clip-v2` on 100 deterministic W7 queries and the first 500 products by ecode. After the product restriction, 29 queries retained at least five candidates and a positive LTR label. Every system ranked the same candidate pools for those 29 queries.

Jina CLIP v2 uses an XLM-RoBERTa tokenizer, which is supported by the current Eland tokenizer family. Its native text and image towers both produce 1024-dimensional embeddings; both sides were L2-normalized before dot-product ranking.

## Macro LTR results

| System | Queries | NDCG@10 | MAP |
| --- | ---: | ---: | ---: |
| `jina-clip-v2` | 29 | 0.6234 | 0.7105 |
| `siglip-image` | 29 | 0.6140 | 0.7190 |
| `production` | 29 | 0.6123 | 0.6963 |
| `omni-nano-image` | 29 | 0.5497 | 0.6244 |
| `omni-small-image` | 29 | 0.5032 | 0.6327 |
| `random` | 29 | 0.4768 | 0.5773 |

## Impression-weighted LTR results

| System | NDCG@10 | MAP |
| --- | ---: | ---: |
| `jina-clip-v2` | 0.6848 | 0.8446 |
| `omni-nano-image` | 0.6797 | 0.8496 |
| `siglip-image` | 0.6411 | 0.8485 |
| `production` | 0.6333 | 0.7728 |
| `omni-small-image` | 0.5691 | 0.8549 |
| `random` | 0.5046 | 0.8239 |

## Paired significance

Bootstrap uses 500 resamples over the 29 eligible queries.

| Contrast | Macro delta | 95% CI | p | Weighted delta | Weighted p |
| --- | ---: | --- | ---: | ---: | ---: |
| `jina-clip-v2` vs `siglip-image` | +0.0094 | [-0.0524, +0.0695] | 0.756 | +0.0437 | 0.904 |
| `jina-clip-v2` vs `omni-nano-image` | +0.0736 | [+0.0076, +0.1481] | 0.036 | +0.0051 | 0.356 |
| `jina-clip-v2` vs `omni-small-image` | +0.1202 | [+0.0450, +0.1942] | 0.004 | +0.1157 | 0.164 |
| `jina-clip-v2` vs `production` | +0.0111 | [-0.0569, +0.0740] | 0.808 | +0.0515 | 0.956 |
| `jina-clip-v2` vs `random` | +0.1466 | [+0.0759, +0.2202] | 0.000 | +0.1802 | 0.124 |

## Interpretation

On this limited product slice, Jina CLIP v2 is directionally ahead of local SigLIP on macro NDCG@10, but the difference is not statistically significant. It is significantly ahead of the two Omni image systems in the macro comparison. The result is preliminary because the product restriction leaves only 29 eligible queries; a full W7 comparison requires encoding all 28,201 W7 products, which was not completed on the available local MPS hardware.

Artifacts:

- `results/w7_jina_clip_v2_per_query.csv`
- `results/w7_jina_clip_v2_summary.csv`
- `results/w7_jina_clip_v2_significance.csv`
- `results/w7_jina_clip_v2_meta.json`
