#!/usr/bin/env python3
"""Step 22 (W9): render papers/W9_image_tower_on_production_hybrid.md from results/w9_*.csv.

Full metric depth (NDCG/Recall/Precision/MRR at every k in config.K_VALUES, plus MAP), macro and
impression weighting, tier breakdown, and the paired-bootstrap contrasts.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

LABELS = {
    "bm25": "bm25",
    "e5": "e5-small",
    "e5_local": "e5-small (full coverage)",
    "jina_clip_v2_image": "jina-clip-v2-image",
    "rrf_bm25_e5": "RRF(bm25 + e5)",
    "rrf_bm25_e5_jina": "RRF(bm25 + e5 + image)",
    "rrf_bm25_jina": "RRF(bm25 + image)",
    "rrf_e5_jina": "RRF(e5 + image)",
    "rrf_bm25_e5local": "RRF(bm25 + e5-full)",
    "rrf_bm25_e5local_jina": "RRF(bm25 + e5-full + image)",
    "rrf_e5local_jina": "RRF(e5-full + image)",
    "siglip_image": "siglip-image",
    "rrf_bm25_e5_siglip": "RRF(bm25 + e5 + siglip)",
    "rrf_bm25_siglip": "RRF(bm25 + siglip)",
    "rrf_e5local_siglip": "RRF(e5-full + siglip)",
    "production": "production",
    "random": "random",
}
ORDER = list(LABELS)
BASELINE = "rrf_bm25_e5"
CONTRAST = "rrf_bm25_e5_jina"
K_VALUES = list(config.K_VALUES)
METRIC_NAMES = {"ndcg": "NDCG", "recall": "Recall", "precision": "Precision", "mrr": "MRR"}
PRIMARY = "ltr"
OUT = config.ROOT / "papers" / "W9_image_tower_on_production_hybrid.md"


def label(system: str) -> str:
    return LABELS.get(system, system)


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], highlight: set[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        name = label(row["system"])
        cells = [f"{row[c]:.4f}" for c in cols]
        if row["system"] in highlight:
            name, cells = f"**`{name}`**", [f"**{c}**" for c in cells]
        else:
            name = f"`{name}`"
        lines.append("| " + " | ".join([name] + cells) + " |")
    return "\n".join(lines)


def depth_block(sub: pd.DataFrame) -> str:
    sub = sub.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    highlight = {BASELINE, CONTRAST}
    parts = []
    for metric, lbl in METRIC_NAMES.items():
        parts.append(f"**{lbl}@k**\n")
        parts.append(
            md_table(sub, [f"{metric}@{k}" for k in K_VALUES],
                     ["System"] + [f"{lbl}@{k}" for k in K_VALUES], highlight) + "\n"
        )
    parts.append("**MAP** (no cutoff)\n")
    parts.append(md_table(sub, ["map"], ["System", "MAP"], highlight) + "\n")
    return "\n".join(parts)


def headline_table(summary: pd.DataFrame) -> str:
    rows = []
    for weighting in ("macro", "impression"):
        sub = summary[(summary["label_set"] == PRIMARY) & (summary["weighting"] == weighting)]
        idx = sub.set_index("system")
        if BASELINE not in idx.index or CONTRAST not in idx.index:
            continue
        for metric in ("ndcg@10", "ndcg@48", "recall@48", "mrr@10", "map"):
            base, cont = idx.loc[BASELINE, metric], idx.loc[CONTRAST, metric]
            rows.append((weighting, metric, base, cont, cont - base, 100 * (cont - base) / base))
    lines = ["| Weighting | Metric | Baseline `RRF(bm25 + e5)` | `+ image` | Δ | Δ % |",
             "| --- | --- | --- | --- | --- | --- |"]
    for w, m, b, c, d, p in rows:
        lines.append(f"| {w} | {m} | {b:.4f} | {c:.4f} | {d:+.4f} | {p:+.2f}% |")
    return "\n".join(lines)


def sig_table(sig: pd.DataFrame, tier: str) -> str:
    s = sig[(sig["query_tier"] == tier)]
    lines = ["| Contrast | Note | Macro Δ | 95% CI | p | Win rate | Weighted Δ | Weighted p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in s.iterrows():
        mark = "**sig.**" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "n.s."
        lines.append("| " + " | ".join([
            f"`{label(r['system'])}` vs `{label(r['baseline'])}`",
            r["note"],
            f"{r['delta']:+.4f}",
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
            f"{r['p_value']:.4f} {mark}",
            f"{r['win_rate']:.3f}",
            f"{r['wtd_delta']:+.4f}",
            f"{r['wtd_p_value']:.4f}",
        ]) + " |")
    return "\n".join(lines)


def tier_table(tier_summary: pd.DataFrame, weighting: str) -> str:
    sub = tier_summary[(tier_summary["label_set"] == PRIMARY) & (tier_summary["weighting"] == weighting)]
    tiers = ["head", "torso", "tail"]
    lines = ["| System | " + " | ".join(f"{t} NDCG@10" for t in tiers) + " |",
             "| --- | " + " | ".join("---" for _ in tiers) + " |"]
    for system in ORDER:
        vals = []
        for tier in tiers:
            row = sub[(sub["query_tier"] == tier) & (sub["system"] == system)]
            vals.append(f"{row['ndcg@10'].iloc[0]:.4f}" if len(row) else "-")
        name = f"**`{label(system)}`**" if system in {BASELINE, CONTRAST} else f"`{label(system)}`"
        lines.append("| " + " | ".join([name] + vals) + " |")
    return "\n".join(lines)


def main() -> int:
    summary = pd.read_csv(config.RESULTS_DIR / "w9_summary.csv")
    tier_summary = pd.read_csv(config.RESULTS_DIR / "w9_tier_summary.csv")
    sig = pd.read_csv(config.RESULTS_DIR / "w9_significance.csv")
    meta = json.loads((config.RESULTS_DIR / "w9_meta.json").read_text())
    es_coverage = json.loads((config.RESULTS_DIR / "w9_es_coverage.json").read_text())

    macro = summary[(summary["label_set"] == PRIMARY) & (summary["weighting"] == "macro")]
    wtd = summary[(summary["label_set"] == PRIMARY) & (summary["weighting"] == "impression")]
    raw = summary[(summary["label_set"] == "raw_ctr") & (summary["weighting"] == "macro")]

    head = sig[(sig["query_tier"] == "all") & (sig["system"] == CONTRAST) & (sig["baseline"] == BASELINE)]
    h = head.iloc[0]
    verdict = "significant" if (h["ci_low"] > 0 or h["ci_high"] < 0) else "not significant"
    direction = "improves on" if h["delta"] > 0 else "does not improve on"

    cov = meta["candidate_coverage_pct"]
    image_share = cov.get("jina_clip_v2_image", 0)
    idx_macro = macro.set_index("system")["ndcg@10"]

    def gap(a: str, b: str) -> str:
        return f"{idx_macro[a] - idx_macro[b]:+.4f}"

    encoder_section = ""
    if "siglip_image" in idx_macro.index:
        encoder_section = f"""
## The image encoder matters more than the fusion

W9 was scoped around Jina CLIP v2 because that is the model newly deployed to the cluster. But
SigLIP -- already encoded at full coverage since W1, and already deployed as
`siglip-base-patch16-512-text-v2` -- is the stronger image tower on this test set, and the gap is
larger than the entire benefit of adding a third arm.

| System | macro NDCG@10 |
| --- | --- |
| `siglip-image` alone | {idx_macro['siglip_image']:.4f} |
| `jina-clip-v2-image` alone | {idx_macro['jina_clip_v2_image']:.4f} |
| `RRF(e5-full + siglip)` | {idx_macro['rrf_e5local_siglip']:.4f} |
| `RRF(e5-full + image)` (Jina) | {idx_macro['rrf_e5local_jina']:.4f} |
| `RRF(bm25 + e5 + siglip)` | {idx_macro['rrf_bm25_e5_siglip']:.4f} |
| `RRF(bm25 + e5 + image)` (Jina) | {idx_macro['rrf_bm25_e5_jina']:.4f} |

Swapping the encoder is worth {gap('siglip_image', 'jina_clip_v2_image')} standalone and
{gap('rrf_e5local_siglip', 'rrf_e5local_jina')} inside the best recipe -- both larger than the
{gap('rrf_bm25_e5_jina', 'rrf_bm25_e5')} the Jina arm buys over the incumbent hybrid. A
recommendation that reads "add Jina CLIP v2" without this table picks the weaker of two models
the cluster already hosts.
"""

    banner = ""
    if image_share < 95:
        banner = (
            f"> **PROVISIONAL -- DO NOT CITE.** The Jina CLIP v2 image encode was still running "
            f"when this was generated: the image arm can score only {image_share}% of candidates, "
            f"so every number involving it is a lower bound. Re-run `19_embed_jina_clip_v2_full.py` "
            f"to completion, then `21_w9_hybrid_experiment.py` and `22_w9_report.py`.\n\n"
        )
    doc = f"""# W9 -- Does a Jina CLIP v2 image tower add anything on top of the live BM25 + E5 hybrid?

_Generated {date.today().isoformat()} from `results/w9_*.csv`._

{banner}## Question

Every earlier week compared embedding systems against each other. W9 asks the deployment
question instead: the production stack already fuses Lucene BM25 with multilingual-e5-small by
reciprocal rank fusion, so what is the **marginal** value of adding a product-image tower as a
third arm? A model that wins in isolation can still add nothing once it is fused with signals
that already capture the same information.

- **Baseline** -- `RRF(bm25 + e5)`, the incumbent hybrid.
- **Contrast** -- `RRF(bm25 + e5 + jina-clip-v2-image)`, the same fusion with one extra arm.

## Verdict

Adding the image tower **{direction}** the production hybrid on NDCG@10, and the difference is
**{verdict}**: Δ = {h['delta']:+.4f} (95% CI [{h['ci_low']:+.4f}, {h['ci_high']:+.4f}],
p = {h['p_value']:.4f}) across {int(h['n_queries'])} queries, winning on
{h['win_rate']:.1%} of them and losing on {h['loss_rate']:.1%}. Impression-weighted,
Δ = {h['wtd_delta']:+.4f} (p = {h['wtd_p_value']:.4f}).

{headline_table(summary)}

Two qualifications belong next to that number, not in a footnote.

First, `production` -- the incumbent ordering by mean observed impression position -- still scores
{idx_macro['production']:.4f}, above every fusion tested here. It is partly self-fulfilling, since
the labels derive from click behaviour that is itself position-dependent, but it is the bar.

Second, the three-way fusion is **not** significantly better than the image tower on its own
({gap(CONTRAST, 'jina_clip_v2_image')}, p = {sig[(sig.query_tier=='all') & (sig.system==CONTRAST) & (sig.baseline=='jina_clip_v2_image')]['p_value'].iloc[0]:.3f}),
and the incumbent hybrid is not significantly better than BM25 on its own
({gap(BASELINE, 'bm25')}, p = {sig[(sig.query_tier=='all') & (sig.system==BASELINE) & (sig.baseline=='bm25')]['p_value'].iloc[0]:.3f}).
On this judgement list the E5 arm as currently indexed is close to inert, and most of what the
"three-way fusion" achieves is attributable to the image tower.
{encoder_section}
## Setup

| | |
| --- | --- |
| Judgement list | same 3-month LTR window as W1-W8 ({config.LOOKBACK_DAYS} days, banner {config.BANNER}, channel {config.CHANNEL}) |
| Queries | {meta['n_queries']:,} |
| Candidates scored | {meta['n_candidates']:,} |
| Fusion | reciprocal rank fusion, k = {meta['rrf_k']} |
| Bootstrap | {meta['bootstrap_samples']:,} paired resamples over queries |
| Seed | {meta['random_seed']} |

**Signal provenance.** Both baseline arms are read from the live cluster rather than
reimplemented locally.

| Arm | Source | Detail |
| --- | --- | --- |
| `bm25` | `catalog-1` | Lucene BM25, `name^3 keyword^2 attributes longDescription`, pool applied as a filter so IDF still comes from the full 306M-doc index |
| `e5-small` | {", ".join(f"`{i}`" for i in es_coverage["indices"])} | `{es_coverage['model_id']}`, {es_coverage['n_products_with_e5_vector']:,}/{es_coverage['n_products_test_set']:,} products ({es_coverage['product_coverage_pct']}%) |
| `jina-clip-v2-image` | local encode | `jinaai/jina-clip-v2` vision tower over the product photograph, 1024-d |

**E5 prefix.** The production indices embed `passage: {{name}}`; queries therefore use
`query: {{term}}`. This was verified rather than assumed -- inference on the prefixed name
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
""" + "\n".join(f"| `{label(k)}` | {v}% |" for k, v in cov.items()) + f"""

## Results -- macro average, LTR labels

{depth_block(macro)}

## Results -- impression weighted, LTR labels

Macro treats every query equally; impression weighting answers "how well does this rank a
typical search impression". They can disagree, and where they do the weighted number is the
business-relevant one.

{depth_block(wtd)}

## Results -- raw-CTR labels, macro average

A robustness check against the LTR grade construction: same rankings, relevance taken from raw
click-through rate bins instead of the IPW-smoothed weighted-CTR quantiles.

{depth_block(raw)}

## Query tier breakdown

Head/torso/tail are percentile cut points on total impressions among queries passing the LTR
filters (head above {config.HEAD_PCTL:.0%}, torso above {config.TORSO_PCTL:.0%}).

**Macro NDCG@10**

{tier_table(tier_summary, "macro")}

**Impression-weighted NDCG@10**

{tier_table(tier_summary, "impression")}

## Significance -- paired bootstrap on NDCG@10

All queries:

{sig_table(sig, "all")}

By tier:

""" + "\n\n".join(f"**{t}**\n\n{sig_table(sig, t)}" for t in ("head", "torso", "tail") if (sig["query_tier"] == t).any()) + f"""

## Sensitivity -- is the gain just patching E5's coverage holes?

E5 vectors exist for only {es_coverage['product_coverage_pct']}% of the catalog, so a naive reading of
the headline is available: the image tower might be winning simply because it can score
candidates the baseline cannot, rather than because it carries information the baseline lacks.

Queries are split on how much of their pool E5 can actually score. `e5_covered` are the
{meta['n_queries_e5_covered']:,} queries where E5 reaches at least
{meta['e5_full_coverage_threshold']:.0%} of the pool -- the baseline is at full strength there,
so any remaining gain cannot be a coverage artefact. `e5_sparse` are the remaining
{meta['n_queries_e5_sparse']:,} queries.

**e5_covered** -- baseline at full strength

{sig_table(sig, "e5_covered")}

**e5_sparse** -- baseline handicapped by missing vectors

{sig_table(sig, "e5_sparse")}

If the effect survives in `e5_covered` at a similar magnitude, the image tower is contributing
genuine signal. If it collapses there and lives entirely in `e5_sparse`, the honest conclusion is
that the cheaper fix is to repair vector coverage, not to add a third tower.

## Where the lexical arm is blind

BM25 returns nothing at all for {meta['n_queries_bm25_blind']:,} queries -- plurals (`hokas`,
`uggs`, `sambas`), misspellings (`addidas`, `shoses`), and brand nicknames (`kobes`,
`sabrinas`). These are not index gaps; they are the structural failure mode of exact term
matching, and they are the cleanest test of whether a non-lexical arm rescues cases the lexical
arm cannot reach in principle.

{sig_table(sig, "bm25_blind")}

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
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
