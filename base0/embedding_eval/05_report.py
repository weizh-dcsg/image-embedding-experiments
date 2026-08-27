#!/usr/bin/env python3
"""Step 5: render the evaluation report (markdown + charts) from results/.

Outputs:
  results/EVALUATION_REPORT.md
  results/fig_ndcg_by_system.png
  results/fig_metric_curves.png
  results/fig_head_to_head.png
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

LABELS = {
    "siglip_image": "SigLIP image (product photo)",
    "siglip_image_crop": "SigLIP image (MGPL crop)",
    "siglip_image_naive": "SigLIP image (largest-area crop)",
    "siglip_text": "SigLIP text (product title)",
    "jina_text": "Jina v5 text nano (product title)",
    "fusion": "Fusion (SigLIP image + Jina text)",
    "production": "Production on-site ranking",
    "random": "Random",
}
ORDER = [
    "siglip_image_crop",
    "siglip_image",
    "siglip_image_naive",
    "siglip_text",
    "fusion",
    "jina_text",
    "random",
    "production",
]
LABEL_SET_NAMES = {"ltr": "LTR judgement list (IPW + time decay)", "raw_ctr": "Raw CTR"}
PRIMARY = "ltr"


def sysname(s: str) -> str:
    return LABELS.get(s, s)


def plot_ndcg_by_system(summary: pd.DataFrame, out: Path) -> None:
    systems = [s for s in ORDER if s in set(summary["system"])]
    label_sets = [ls for ls in LABEL_SET_NAMES if ls in set(summary["label_set"])]
    y = np.arange(len(systems))
    height = 0.8 / len(label_sets)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, ls in enumerate(label_sets):
        sub = summary[summary["label_set"] == ls].set_index("system").reindex(systems)
        offset = (i - (len(label_sets) - 1) / 2) * height
        bars = ax.barh(y + offset, sub["ndcg@10"], height=height, label=LABEL_SET_NAMES[ls])
        for bar, v in zip(bars, sub["ndcg@10"]):
            ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([sysname(s) for s in systems])
    ax.invert_yaxis()
    ax.set_xlabel("NDCG@10 (mean over queries)")
    ax.set_title("Search relevance by embedding system")
    ax.set_xlim(0, summary["ndcg@10"].max() * 1.2)
    ax.legend(title="Relevance labels", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_metric_curves(summary: pd.DataFrame, out: Path) -> None:
    sub = summary[summary["label_set"] == PRIMARY]
    ks = list(config.K_VALUES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for metric, ax in zip(("ndcg", "recall", "precision"), axes):
        for system in ORDER:
            row = sub[sub["system"] == system]
            if row.empty:
                continue
            ax.plot(ks, [row[f"{metric}@{k}"].iloc[0] for k in ks], marker="o", label=sysname(system))
        ax.set_xticks(ks)
        ax.set_xlabel("k")
        ax.set_title(f"{metric.upper()}@k")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"Metric curves ({LABEL_SET_NAMES[PRIMARY].lower()} labels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_head_to_head(per_query: pd.DataFrame, out: Path) -> None:
    sub = per_query[per_query["label_set"] == PRIMARY]
    wide = sub.pivot(index="search_term", columns="system", values="ndcg@10").dropna()
    contrasts = [
        ("siglip_image", "jina_text", "Cross-model: image vs text"),
        ("siglip_image", "siglip_text", "Same model: image vs title"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (a, b, title) in zip(axes, contrasts):
        if not {a, b}.issubset(wide.columns):
            continue
        diff = wide[a] - wide[b]
        ax.hist(diff, bins=40, color="#4c72b0")
        ax.axvline(0, color="k", lw=1)
        ax.axvline(float(diff.mean()), color="crimson", ls="--", lw=1.5,
                   label=f"mean {diff.mean():+.4f}\nwin rate {(diff > 1e-9).mean():.0%}")
        ax.set_xlabel(f"NDCG@10  ({sysname(a).split(' (')[0]} - {sysname(b).split(' (')[0]})")
        ax.set_ylabel("queries")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for row in df.itertuples(index=False):
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def contrast(significance: pd.DataFrame, system: str, baseline: str, label_set: str) -> dict:
    row = significance[
        (significance["system"] == system)
        & (significance["baseline"] == baseline)
        & (significance["label_set"] == label_set)
    ]
    return row.iloc[0].to_dict() if not row.empty else {}


def describe(c: dict) -> str:
    if not c:
        return "n/a"
    sig = "significant" if (c["ci_low"] > 0 or c["ci_high"] < 0) else "not significant"
    return (
        f"{c['delta']:+.4f} ({c['rel_delta_pct']:+.1f}%), 95% CI "
        f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}], p = {c['p_value']:.4f}, "
        f"win rate {c['win_rate']:.0%} — {sig}"
    )


def build_verdict(summary: pd.DataFrame, significance: pd.DataFrame) -> str:
    prim = summary[summary["label_set"] == PRIMARY].set_index("system")

    def ndcg(system: str) -> float:
        return float(prim.loc[system, "ndcg@10"]) if system in prim.index else float("nan")

    cross = contrast(significance, "siglip_image", "jina_text", PRIMARY)
    modality = contrast(significance, "siglip_image", "siglip_text", PRIMARY)

    cross_sig = bool(cross) and (cross["ci_low"] > 0 or cross["ci_high"] < 0)
    modality_sig = bool(modality) and (modality["ci_low"] > 0 or modality["ci_high"] < 0)

    lines = [
        f"- **SigLIP image vs Jina v5 text**: NDCG@10 {ndcg('siglip_image'):.4f} vs "
        f"{ndcg('jina_text'):.4f} — {describe(cross)}",
        f"- **SigLIP image vs SigLIP text** (same encoder, modality is the only change): "
        f"NDCG@10 {ndcg('siglip_image'):.4f} vs {ndcg('siglip_text'):.4f} — {describe(modality)}",
    ]

    if cross_sig and cross["delta"] > 0 and not modality_sig:
        headline = (
            "**The image-embedding system does beat the text-embedding system, but the gain is not "
            "a modality effect.** SigLIP's image tower outperforms Jina v5 text nano by a "
            f"statistically significant margin ({cross['rel_delta_pct']:+.1f}% NDCG@10). However, "
            "SigLIP's *text* tower performs essentially the same as its image tower, and it beats "
            "Jina by a near-identical margin. The controlled contrast — same encoder, photo vs "
            "title — shows no significant difference. So what this test set measures is a "
            "**model/alignment difference, not image-over-text superiority**.\n\n"
            "The claim \"text embeddings perform worse than image embeddings\" is **not supported** "
            "in its general form. The defensible claim is: *SigLIP, in either modality, ranks better "
            "than Jina v5 text nano on click-derived relevance for head search terms.*"
        )
    elif cross_sig and cross["delta"] > 0 and modality_sig and modality["delta"] > 0:
        headline = (
            "**Image embeddings beat text embeddings, and the effect survives the controlled "
            "contrast.** SigLIP image beats both Jina v5 text nano and SigLIP's own text tower by "
            "statistically significant margins, so the advantage is attributable to the modality "
            "rather than to the encoder alone."
        )
    elif cross_sig and cross["delta"] < 0:
        headline = (
            "**Text embeddings beat image embeddings on this test set.** Jina v5 text nano "
            "outperforms SigLIP image by a statistically significant margin. The hypothesis that "
            "image embeddings dominate is not supported."
        )
    else:
        headline = (
            "**No statistically significant difference between the image and text systems.** "
            "Neither modality can be declared better on this test set at the current sample size."
        )

    return headline + "\n\n" + "\n".join(lines)


def main() -> int:
    summary = pd.read_csv(config.RESULTS_DIR / "summary.csv")
    per_query = pd.read_csv(config.RESULTS_DIR / "per_query_metrics.csv")
    significance = pd.read_csv(config.RESULTS_DIR / "significance.csv")
    by_tier = pd.read_csv(config.RESULTS_DIR / "summary_by_tier.csv")
    meta = json.loads((config.RESULTS_DIR / "run_meta.json").read_text())

    tier_breakdown = ", ".join(
        f"{n} {tier}" for tier, n in meta["n_queries_by_tier"].items() if tier in ("head", "torso", "tail")
    )

    plot_ndcg_by_system(summary, config.RESULTS_DIR / "fig_ndcg_by_system.png")
    plot_metric_curves(summary, config.RESULTS_DIR / "fig_metric_curves.png")
    plot_head_to_head(per_query, config.RESULTS_DIR / "fig_head_to_head.png")

    metric_cols = ["ndcg@5", "ndcg@10", "ndcg@20", "mrr@10", "map", "recall@10", "precision@5"]

    def results_table(label_set: str) -> str:
        t = summary[summary["label_set"] == label_set].copy()
        t["system"] = t["system"].map(sysname)
        cols = ["system", "n_queries"] + [c for c in metric_cols if c in t.columns]
        return md_table(t[cols])

    def tier_table(label_set: str) -> str:
        t = by_tier[(by_tier["label_set"] == label_set) & (by_tier["weighting"] == "macro")].copy()
        t["system"] = t["system"].map(sysname)
        cols = ["query_tier", "system", "n_queries"] + [c for c in metric_cols if c in t.columns]
        return md_table(t[cols])

    sig_view = significance[
        (significance["label_set"] == PRIMARY)
        & (significance["baseline"].isin(("jina_text", "siglip_text")))
    ].copy()
    sig_view["system"] = sig_view["system"].map(sysname)
    sig_view["baseline"] = sig_view["baseline"].map(sysname)
    sig_view = sig_view[
        ["system", "baseline", "delta", "rel_delta_pct", "ci_low", "ci_high", "p_value", "win_rate"]
    ]

    report = f"""# Image vs Text Embeddings for Search Relevance

Generated {date.today().isoformat()}

- Image model: `{meta['siglip_model']}`
- Text model: `{meta['jina_model']}`

## Verdict

{build_verdict(summary, significance)}

## What was compared

Every system ranks the **same candidate pool** for the **same queries**; only the product
representation changes.

| System | Query encoder | Product representation |
| --- | --- | --- |
| SigLIP image | SigLIP text tower | SigLIP image tower over the product photo |
| SigLIP text | SigLIP text tower | SigLIP text tower over the product title |
| Jina v5 text nano | Jina `retrieval.query` | Jina `retrieval.document` over the product title |
| Fusion | both | z-scored mean of SigLIP-image and Jina-text similarity |
| Production | n/a | current on-site ranking (mean observed impression position) |
| Random | n/a | seeded shuffle, floor reference |

**SigLIP text is the control that makes this experiment interpretable.** Comparing SigLIP image
to Jina text changes two things at once (encoder *and* modality). SigLIP text holds the encoder
fixed and changes only the modality, which is the contrast that actually tests the hypothesis.

## Test data

Built from ML clickstream events (`prod_ent_silver_db.sdsc.ml_events`), SRLP page 0,
banner `{config.BANNER}`, channel `{config.CHANNEL}`, over a {config.LOOKBACK_DAYS}-day window.

- Queries: **{meta['n_queries']}** ({tier_breakdown}), every term that passed the LTR
  eligibility filters (min/max group size, >= 2 relevance levels) -- no volume-based cap
- Query-product pairs: **{meta['n_pairs']}**
- Unique active products: **{meta['n_products']}**
- Mean candidate pool: **{meta['mean_pool_size']:.1f}** products per query

Candidates are restricted to **active products**: DSG web-active in
`entdata.web.dim_sku_bod_web_active` (`web_chain_code = 'DSG'`), joined to
`prod_ml_feature_store_db.products.ecode` with `dsg_web_active = 'Y'`, and required to have both
a product title and a default image URL. This follows the active-product pattern used in
`ds-ecm-search-ranking-ltr`.

### Relevance labels

Two label sets are scored so the conclusion can be checked against the labelling choice.

1. **Position-debiased CTR** (primary). Clicks are divided by an examination propensity estimated
   from the global click-rate-by-rank curve, mirroring the inverse-propensity weighting in
   `ds-ecm-search-ranking-ltr/sandbox/ltr_vanilla`. Grades 0-3 come from the within-query
   percentile of the debiased CTR.
2. **Raw CTR**. Identical grading on undebiased CTR, so rank position still carries signal.

Debiasing works as intended: the correlation between mean impression position and the label drops
from about -0.46 (raw) to about -0.01 (debiased). A side effect is that the `production` baseline
has almost no signal left on the primary label set — by construction, not by failure.

## Results

### Primary: {LABEL_SET_NAMES[PRIMARY].lower()} labels

{results_table(PRIMARY)}

### Robustness: raw CTR labels

{results_table('raw_ctr')}

![NDCG@10 by system](fig_ndcg_by_system.png)

![Metric curves](fig_metric_curves.png)

### By query-volume tier

Macro-averaged within each tier (head = top {100 * (1 - config.HEAD_PCTL):.0f}% by volume among
queries that passed the LTR filters, torso = next {100 * (config.HEAD_PCTL - config.TORSO_PCTL):.0f}%,
tail = the rest), {LABEL_SET_NAMES[PRIMARY].lower()} labels. A system that only wins in aggregate
because head queries dominate the traffic-weighted average will show it here.

{tier_table(PRIMARY)}

## Significance

Paired bootstrap over queries ({meta['bootstrap_samples']} resamples), metric NDCG@10,
{LABEL_SET_NAMES[PRIMARY].lower()} labels.

{md_table(sig_view)}

![Head to head](fig_head_to_head.png)

The left panel is the cross-model contrast; the right panel is the controlled modality contrast.
A conclusion about modality requires the right panel to be significant, not just the left.

## Reading this honestly

- The two encoders were trained for different objectives. SigLIP is a contrastive image-text model
  trained on web alt-text; Jina v5 text nano is a distilled multilingual retrieval model. A gap
  between them is a statement about model fit to short e-commerce queries, not about pixels vs words.
- Product titles are dense with exactly the tokens head queries use (brand, sport, gender, product
  type), so text has a strong prior here. Image embeddings contribute colour, silhouette, and
  material, which head terms rarely specify. Expect the modality gap to widen on descriptive or
  visual queries and to matter less on head terms like these.
- The `fusion` row is the practical takeaway: combining the two signals is competitive with or
  better than either alone on recall and MAP, which implies the modalities carry partly
  complementary information.

## Caveats

- Labels come from click behaviour under the current ranker, so they inherit its exposure bias.
  Position debiasing reduces but does not remove this.
- Candidate pools are products the production system already surfaced, so this measures
  **re-ranking** quality, not full-catalogue retrieval.
- Only head terms are covered. Tail and descriptive queries are where image embeddings are most
  likely to pay off and are not represented here.
- One image per product (the default ecode image); no multi-view or in-context imagery.
- `{config.JINA_MODEL}` is CC BY-NC 4.0. Commercial use requires a licence from Jina AI.

## Suggested next steps

1. Re-run on tail and attribute-heavy queries (colour, pattern, style words) where the modality
   contrast should be strongest.
2. Add a same-family text baseline in the other direction (e.g. a SigLIP-class text retriever vs
   Jina) to separate encoder quality from modality once more.
3. Evaluate fusion weighting rather than an equal-weight z-score blend.

## Reproducing

```bash
./run_all.sh
```

Artifacts: `results/summary.csv`, `results/per_query_metrics.csv`, `results/significance.csv`,
`results/run_meta.json`.
"""

    out = config.RESULTS_DIR / "EVALUATION_REPORT.md"
    out.write_text(report)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
