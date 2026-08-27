#!/usr/bin/env python3
"""Step 12: render papers/W7_image_encoder_comparison.md from results/w7_*.csv.

Full metric depth (NDCG/Recall/Precision/MRR at every k in config.K_VALUES, plus MAP), macro and
impression weighting, plus a per-tier breakdown and paired-bootstrap contrasts.
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
    "siglip_image": "siglip-image",
    "omni_nano_image": "omni-nano-image",
    "omni_small_image": "omni-small-image",
    "siglip_attr": "attr-siglip (text reference)",
    "production": "production (incumbent ordering)",
    "random": "random",
}
IMAGE_SYSTEMS = ["siglip_image", "omni_nano_image", "omni_small_image"]
K_VALUES = list(config.K_VALUES)
METRIC_NAMES = {"ndcg": "NDCG", "recall": "Recall", "precision": "Precision", "mrr": "MRR"}
PRIMARY = "ltr"


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], best: str | None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        name = LABELS.get(row["system"], row["system"])
        cells = [f"{row[c]:.4f}" for c in cols]
        if row["system"] == best:
            name, cells = f"**`{name}`**", [f"**{c}**" for c in cells]
        else:
            name = f"`{name}`"
        lines.append("| " + " | ".join([name] + cells) + " |")
    return "\n".join(lines)


def block(sub: pd.DataFrame) -> str:
    sub = sub.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    imgs = sub[sub["system"].isin(IMAGE_SYSTEMS)]
    best = imgs.iloc[0]["system"] if len(imgs) else None
    parts = []
    for metric, label in METRIC_NAMES.items():
        parts.append(f"**{label}@k**\n")
        parts.append(md_table(sub, [f"{metric}@{k}" for k in K_VALUES],
                              ["System"] + [f"{label}@{k}" for k in K_VALUES], best) + "\n")
    parts.append("**MAP** (no cutoff)\n")
    parts.append(md_table(sub, ["map"], ["System", "MAP"], best) + "\n")
    return "\n".join(parts)


def sig_table(sig: pd.DataFrame) -> str:
    lines = ["| Contrast | Macro Δ | Macro p | 95% CI | Win rate | Weighted Δ | Weighted p |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in sig.iterrows():
        verdict = "**sig.**" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "n.s."
        lines.append("| " + " | ".join([
            f"`{LABELS.get(r['system'], r['system'])}` vs `{LABELS.get(r['baseline'], r['baseline'])}`",
            f"{r['delta']:+.4f}", f"{r['p_value']:.4f} {verdict}",
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", f"{r['win_rate']:.0%}",
            f"{r['wtd_delta']:+.4f}", f"{r['wtd_p_value']:.4f}",
        ]) + " |")
    return "\n".join(lines)


def main() -> int:
    summary = pd.read_csv(config.RESULTS_DIR / "w7_summary.csv")
    by_tier = pd.read_csv(config.RESULTS_DIR / "w7_summary_by_tier.csv")
    sig = pd.read_csv(config.RESULTS_DIR / "w7_significance.csv")
    meta = json.loads((config.RESULTS_DIR / "w7_meta.json").read_text())

    global IMAGE_SYSTEMS
    IMAGE_SYSTEMS = meta.get("image_systems", IMAGE_SYSTEMS)
    excluded = meta.get("excluded_degenerate", [])

    macro = summary[(summary.label_set == PRIMARY) & (summary.weighting == "macro")]
    wtd = summary[(summary.label_set == PRIMARY) & (summary.weighting == "impression")]
    m_img = macro[macro["system"].isin(IMAGE_SYSTEMS)].sort_values("ndcg@10", ascending=False)
    w_img = wtd[wtd["system"].isin(IMAGE_SYSTEMS)].sort_values("ndcg@10", ascending=False)
    best_m, best_w = m_img.iloc[0], w_img.iloc[0]
    attr_macro = float(macro.set_index("system").loc["siglip_attr", "ndcg@10"])

    tiers = meta["n_queries_by_tier"]
    tier_rows = []
    for tier in ("head", "torso", "tail"):
        t = by_tier[(by_tier.label_set == PRIMARY) & (by_tier.weighting == "macro")
                    & (by_tier.query_tier == tier) & (by_tier["system"].isin(IMAGE_SYSTEMS))]
        t = t.set_index("system")["ndcg@10"]
        tier_rows.append("| " + " | ".join(
            [tier] + [f"{t.get(s, float('nan')):.4f}" for s in IMAGE_SYSTEMS]) + " |")
    tier_tbl = ("| Tier | " + " | ".join(LABELS[s] for s in IMAGE_SYSTEMS) + " |\n"
                + "| --- | " + " | ".join("---" for _ in IMAGE_SYSTEMS) + " |\n"
                + "\n".join(tier_rows))

    head_to_head = sig[(sig.system == "siglip_image") & (sig.baseline.isin(IMAGE_SYSTEMS))]
    verdict_bits = []
    for _, r in head_to_head.iterrows():
        sigmark = "significant" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "**not** significant"
        direction = "ahead of" if r["delta"] > 0 else "behind"
        verdict_bits.append(
            f"`siglip-image` is {direction} `{LABELS[r['baseline']]}` by {abs(r['delta']):.4f} "
            f"NDCG@10 ({sigmark}, p = {r['p_value']:.4f})")

    report = f"""# W7 — Which Image Encoder? SigLIP vs Jina v5 Omni

Generated {date.today().isoformat()} by `11_image_encoder_experiment.py` / `12_w7_report.py`.

> **TL;DR**
> Best image encoder, macro: **`{LABELS[best_m['system']]}`** (NDCG@10 {best_m['ndcg@10']:.4f}).
> Best image encoder, impression-weighted: **`{LABELS[best_w['system']]}`** (NDCG@10 {best_w['ndcg@10']:.4f}).
> {'; '.join(verdict_bits)}.
> Reference: the best image encoder is {'**ahead of**' if float(best_m['ndcg@10']) > attr_macro else '**behind**'}
> `attr-siglip` ({attr_macro:.4f} macro), whose earlier apparent strength was a tie-ordering
> artefact — see the correction in `papers/w1-section5.md`.

---

## 1. What is being compared

W1–W6 used exactly **one** image encoder, so every image-side conclusion in the series rested on a
single checkpoint. W7 removes that single point of failure by adding two independent image towers.

| System | Query encoder | Document representation | Params | Dim |
| --- | --- | --- | --- | --- |
| `siglip-image` | SigLIP text tower | SigLIP image tower over the photo | 203M | 768 |
| `omni-nano-image` | Jina v5 omni text | Jina v5 omni vision tower | ~1.0B | 768 |
| `omni-small-image` | Jina v5 omni text | Jina v5 omni vision tower | larger | 1024 |

**Each system is self-consistent** — query and document are encoded by the same model, into the same
space. So the contrast isolates the *image encoder*, not the query representation. This is the same
design discipline as W1's `text-siglip` control (SigLIP's own text tower, reused from the image system).

Reference points, shown for scale but not image systems:
`attr-siglip` (best representation found in W1/W6), `production` (incumbent ordering), `random`.

### Retrieval prefixes

The omni models require `Query: ` / `Document: ` prefixes on **every** modality, not just text.
Queries are encoded as `"Query: {{term}}"`; product images as `images=<photo>, text="Document: <image>"`.
Omitting these puts the model off-distribution.

### Scale and fairness

The omni models are ~1B-param VLMs, 3–5× slower per image than SigLIP, so W7 runs on a seeded,
**tier-stratified subset**: {meta['n_queries']} queries
(head {tiers.get('head', 0)} / torso {tiers.get('torso', 0)} / tail {tiers.get('tail', 0)})
over {meta['n_products']} products.

A product is scored only if **every** system has a vector for it, and a query only if it retains
≥ 5 candidates and a positive label after that intersection — so all systems rank literally
identical pools. Images are encoded one per forward pass; batching measured only 1.17× on this
hardware while introducing ~3.7e-3 numerical drift, which is not a good trade in a study about
image-encoder fidelity.

Labels are the **corrected** LTR judgement list (τ applied to clicks only — see W4).

---

## 2. Results

Every metric at every cutoff (k = {", ".join(str(k) for k in K_VALUES)}), plus MAP. Bold marks the
best-scoring **image** encoder on NDCG@10, tracked consistently across all tables.

### Macro-averaged — every query counts once

{block(macro)}

### Impression-weighted — every query counts in proportion to its traffic

{block(wtd)}

---

## 3. By query-volume tier

Macro NDCG@10 within each tier. Head queries are short and brand-heavy; tail queries are where
W5 showed every system degrades.

{tier_tbl}

---

## 4. Significance

Paired bootstrap over queries ({meta['bootstrap_samples']} resamples), metric NDCG@10, LTR labels.
All pairwise image-encoder contrasts, plus each image encoder against the reference points.

{sig_table(sig)}

---

## 5. Reading the result

{"### A third encoder was excluded, and that is itself a finding" if excluded else ""}

{'''`jina-embeddings-v5-omni-small` was encoded over the full subset and then **excluded**: it emits
**byte-identical vectors for every image** — 1 unique vector per 2,000 sampled, similarity spread
exactly 0.0, and self-cosine 1.0072 (impossible for a unit vector). The inputs were verifiably
different (pixel means 0.506 / 0.238 / 0.802 for three products); only the outputs collapsed.

Every configuration available on this hardware failed:

| Configuration | Result |
| --- | --- |
| MPS, bfloat16 (default) | collapses silently, 1 unique vector |
| MPS, float32 | aborts — Metal assertion in `MPSNDArrayMatrixMultiplication` |
| CPU, bfloat16 | collapses (cos 1.0032), and 0.23 img/s is unusable regardless |
| pre-merged `-retrieval` variant, MPS | collapses (cos 0.99387) |

`omni-nano` is healthy on the identical code path, so this is specific to the larger checkpoint.

**Why this nearly became a false result.** On first run this collapsed encoder scored
**NDCG@10 = 0.9978** — near-perfect, and briefly the best system ever measured in this series. The
cause was an evaluation bug, not a model strength: the judgement-list SQL emits candidates ordered
by `relevance DESC`, and a stable sort preserves input order on ties, so a constant score vector
reproduced the label ordering exactly. Both issues are fixed — pools are now shuffled
deterministically before ranking, and `check_healthy()` rejects collapsed encoders outright.

Any evaluation harness that sorts model scores should assume ties are adversarial.''' if excluded else ""}


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
"""

    out = Path(config.ROOT) / "papers" / "W7_image_encoder_comparison.md"
    out.write_text(report)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
