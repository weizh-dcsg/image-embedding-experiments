# W7 — Full Jina CLIP v2 comparison

Generated 2026-08-28 by `17_embed_jina_clip_v2_w7.py` and `18_compare_jina_clip_v2_w7.py`.

## Scope

This is the full W7 comparison using `jinaai/jina-clip-v2` on all 600 W7 queries and all 28,201 W7 products. Every system ranked the same candidate pools. Jina CLIP v2 uses an XLM-RoBERTa tokenizer and produces 1024-dimensional text and image embeddings; both sides were L2-normalized before dot-product ranking.

## Macro LTR results

| System | Queries | NDCG@10 | MAP |
| --- | ---: | ---: | ---: |
| `siglip-image` | 600 | 0.4272 | 0.7500 |
| `production` | 600 | 0.4200 | 0.6808 |
| `jina-clip-v2` | 600 | 0.3883 | 0.7180 |
| `omni-nano-image` | 600 | 0.3313 | 0.6664 |
| `omni-small-image` | 600 | 0.2713 | 0.6048 |
| `random` | 600 | 0.2640 | 0.6041 |

## Impression-weighted LTR results

| System | NDCG@10 | MAP |
| --- | ---: | ---: |
| `siglip-image` | 0.4443 | 0.9088 |
| `production` | 0.4123 | 0.8035 |
| `jina-clip-v2` | 0.4120 | 0.8889 |
| `omni-nano-image` | 0.3677 | 0.8442 |
| `omni-small-image` | 0.3168 | 0.7981 |
| `random` | 0.3208 | 0.7970 |

## Paired significance

Bootstrap uses 500 resamples over all 600 queries. Metric is NDCG@10 with LTR labels.

| Contrast | Macro delta | 95% CI | p | Weighted delta | Weighted p |
| --- | ---: | --- | ---: | ---: | ---: |
| `jina-clip-v2` vs `siglip-image` | -0.0388 | [-0.0569, -0.0214] | 0.000 | -0.0323 | 0.120 |
| `jina-clip-v2` vs `omni-nano-image` | +0.0570 | [+0.0376, +0.0751] | 0.000 | +0.0443 | 0.024 |
| `jina-clip-v2` vs `omni-small-image` | +0.1170 | [+0.0978, +0.1370] | 0.000 | +0.0952 | 0.000 |
| `jina-clip-v2` vs `production` | -0.0317 | [-0.0558, -0.0071] | 0.012 | -0.0003 | 0.352 |
| `jina-clip-v2` vs `random` | +0.1243 | [+0.1048, +0.1449] | 0.000 | +0.0913 | 0.000 |

## Interpretation

On the full W7 dataset, Jina CLIP v2 is a healthy and operational cross-modal model, but it trails local SigLIP on macro NDCG@10 by 0.0388. It is significantly better than both Omni image systems and random ordering on the macro comparison. The impression-weighted gap against SigLIP is smaller and not significant at the current bootstrap resolution.

Artifacts:

- `results/w7_jina_clip_v2_full_per_query.csv`
- `results/w7_jina_clip_v2_full_summary.csv`
- `results/w7_jina_clip_v2_full_significance.csv`
- `results/w7_jina_clip_v2_full_meta.json`
