#!/usr/bin/env python3
"""Generate a standalone head/torso/tail tier report from results/summary_by_tier.csv.

Same table format as papers/w1-section5.md section 2 (Results): one table per weighting
(macro / impression), columns System | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 | MRR@10 | MAP,
plus a full depth-sensitivity breakout (NDCG/Recall/Precision/MRR at every k in
config.K_VALUES = 5, 10, 20, 48, 96, 144 -- w1-section5.md only shows this sweep for NDCG on the
two headline systems; here it's every system, every metric). Repeated per query_tier
(head / torso / tail) instead of only in aggregate, covering every system produced by
04_evaluate.py (title, image, attribute, and capacity-control arms), primary label set (LTR
judgement-list relevance) only.

Output: results/TIERED_RESULTS.md
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
    "siglip_image": "image",
    "siglip_text": "text-siglip",  # SigLIP's own text tower over the title -- same encoder as `image`
    "jina_text": "text-jina",
    "jina_small_text": "text-jina-small",
    "siglip_attr": "attr-siglip",
    "jina_attr": "attr-jina",
    "jina_small_attr": "attr-jina-small",
    "e5_base_text": "text-e5-base",
    "e5_base_attr": "attr-e5-base",
    "e5_small_multi_text": "text-e5-small-multi",
    "e5_small_multi_attr": "attr-e5-small-multi",
    "e5_large_instruct_text": "text-e5-large-instruct",
    "e5_large_instruct_attr": "attr-e5-large-instruct",
    "omni_nano_text": "text-omni-nano",
    "omni_nano_attr": "attr-omni-nano",
    "fusion": "fusion",
    "production": "production (not comparable -- see note)",
    "random": "random",
}
# W7-subset image-encoder systems (600 queries / 28,201 products -- see caveat in that subsection).
IMAGE_ENCODER_LABELS = {
    "siglip_image": "siglip-image (W7 subset)",
    "omni_nano_image": "omni-nano-image (W7 subset)",
    "omni_small_image": "omni-small-image (W7 subset)",
    "production": "production (W7 subset)",
    "random": "random (W7 subset)",
}
COLS = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "mrr@10", "map"]
COL_HEADERS = ["System", "NDCG@5", "NDCG@10", "NDCG@20", "Recall@10", "MRR@10", "MAP"]
K_VALUES = list(config.K_VALUES)
METRIC_NAMES = {"ndcg": "NDCG", "recall": "Recall", "precision": "Precision", "mrr": "MRR"}
TIERS = ["head", "torso", "tail"]
PRIMARY = "ltr"


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], best_idx, labels: dict = LABELS) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for idx, row in df.iterrows():
        name = labels.get(row["system"], row["system"])
        cells = [f"{row[c]:.4f}" for c in cols]
        if idx == best_idx:
            name = f"**`{name}`**"
            cells = [f"**{c}**" for c in cells]
        else:
            name = f"`{name}`"
        lines.append("| " + " | ".join([name] + cells) + " |")
    return "\n".join(lines)


def weighting_block(sub: pd.DataFrame) -> str:
    # production is a reference point, not a competing system; keep it pinned first, and use its
    # exclusion from best-system highlighting consistently across the summary and every depth table
    sub = sub.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    sub = pd.concat([sub[sub["system"] == "production"], sub[sub["system"] != "production"]], ignore_index=True)
    best_idx = sub.loc[sub["system"] != "production", "ndcg@10"].idxmax() if len(sub) else None

    parts = [md_table(sub, COLS, COL_HEADERS, best_idx), ""]
    parts.append("**Depth sensitivity (all k):**\n")
    for metric, label in METRIC_NAMES.items():
        depth_cols = [f"{metric}@{k}" for k in K_VALUES]
        depth_headers = ["System"] + [f"{label}@{k}" for k in K_VALUES]
        parts.append(f"*{label}@k*\n")
        parts.append(md_table(sub, depth_cols, depth_headers, best_idx) + "\n")
    return "\n".join(parts)


def image_encoder_block(w7_by_tier: pd.DataFrame, tier: str) -> str:
    """Macro + impression NDCG@10 summary of the W7 image-encoder comparison, this tier's slice.

    Sourced from results/w7_summary_by_tier.csv (11_image_encoder_experiment.py /
    12_w7_report.py), computed on a 600-query / 28,201-product stratified subset -- NOT the full
    6536-query / 80,608-product W5 set, so these numbers are not directly comparable in scale to
    the tables above; only the relative ordering of the three image encoders is meaningful here.
    """
    parts = []
    for weighting, title in (
        ("macro", "Macro-averaged"),
        ("impression", "Impression-weighted"),
    ):
        sub = w7_by_tier[
            (w7_by_tier["label_set"] == PRIMARY)
            & (w7_by_tier["query_tier"] == tier)
            & (w7_by_tier["weighting"] == weighting)
            & (w7_by_tier["system"].isin(IMAGE_ENCODER_LABELS))
        ]
        if sub.empty:
            continue
        sub = sub.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
        non_ref = sub[~sub["system"].isin(("production", "random"))]
        best_idx = non_ref["ndcg@10"].idxmax() if len(non_ref) else None
        parts.append(f"*{title}*\n")
        parts.append(md_table(sub, COLS, COL_HEADERS, best_idx, labels=IMAGE_ENCODER_LABELS) + "\n")
    return "\n".join(parts)


def tier_section(by_tier: pd.DataFrame, tier: str, n_queries: int, w7_by_tier: pd.DataFrame | None) -> str:
    parts = [f"## {tier.capitalize()} bucket ({n_queries} queries)\n"]
    for weighting, title in (
        ("macro", "Macro-averaged -- every query counts once"),
        ("impression", "Impression-weighted -- every query counts in proportion to its traffic"),
    ):
        sub = by_tier[
            (by_tier["label_set"] == PRIMARY)
            & (by_tier["query_tier"] == tier)
            & (by_tier["weighting"] == weighting)
        ]
        parts.append(f"### {title}\n")
        parts.append(weighting_block(sub))
    if w7_by_tier is not None:
        parts.append("### Image encoder comparison (reference: W7 subset -- see caveat)\n")
        parts.append(image_encoder_block(w7_by_tier, tier))
    return "\n".join(parts)


def main() -> int:
    by_tier = pd.read_csv(config.RESULTS_DIR / "summary_by_tier.csv")
    meta = json.loads((config.RESULTS_DIR / "run_meta.json").read_text())
    tier_counts = meta["n_queries_by_tier"]

    w7_path = config.RESULTS_DIR / "w7_summary_by_tier.csv"
    w7_by_tier = pd.read_csv(w7_path) if w7_path.exists() else None
    w7_meta_path = config.RESULTS_DIR / "w7_meta.json"
    w7_meta = json.loads(w7_meta_path.read_text()) if w7_meta_path.exists() else {}

    tier_defs = "\n".join(
        f"- **{t}**: top {100 * (1 - config.HEAD_PCTL):.0f}% by query-volume percentile among queries "
        f"that passed the LTR eligibility filters"
        if t == "head"
        else (
            f"- **{t}**: next {100 * (config.HEAD_PCTL - config.TORSO_PCTL):.0f}% by volume"
            if t == "torso"
            else f"- **{t}**: the remaining {100 * config.TORSO_PCTL:.0f}% by volume -- the long tail"
        )
        for t in TIERS
    )

    body = "\n\n".join(tier_section(by_tier, t, tier_counts.get(t, 0), w7_by_tier) for t in TIERS)

    image_note = ""
    if w7_by_tier is not None:
        w7_tier_counts = w7_meta.get("n_queries_by_tier", {})
        per_tier = ", ".join(f"{t} {w7_tier_counts[t]}" for t in TIERS if t in w7_tier_counts)
        image_note = (
            "\n\nEach tier section also includes an **image encoder comparison** (SigLIP vs Jina v5 "
            f"omni-nano vs omni-small image towers), reused from `results/w7_summary_by_tier.csv` "
            f"(`11_image_encoder_experiment.py`). That comparison runs on a smaller, seeded "
            f"stratified subset ({w7_meta.get('n_queries', '?')} queries / "
            f"{w7_meta.get('n_products', '?')} products, {per_tier}) rather than the full set "
            "above -- the omni models are ~1B-param VLMs encoded one image per forward pass "
            "(measured at ~4.2 GPU-hours for omni-nano and ~7.1 GPU-hours for omni-small to cover "
            "the full 80,608-product catalogue on this hardware, per `probe_omni_throughput.py`), "
            "so its absolute numbers are not comparable in scale to the tables above; only the "
            "ordering of the three image encoders within it is meaningful. See "
            "`papers/W7_image_encoder_comparison.md` for the full writeup."
        )

    report = f"""# Search Relevance by Query-Volume Tier

Generated {date.today().isoformat()}

Same systems, metrics, and table format as `papers/w1-section5.md` section 2, but broken out by
query-volume tier instead of reported only in aggregate. All {sum(tier_counts.values())} queries
are ones that passed the LTR eligibility filters (min/max candidate-pool group size, >= 2 relevance
levels) -- there is no volume-based cap on which queries are included; head, torso, and tail are
strata of the same evaluable set, not separately-sampled experiments.

Systems now include three E5 text models (base, multilingual-small, multilingual-large-instruct)
and the Jina v5 omni-nano text tower alongside the original SigLIP/Jina title and Big-4
attribute-string representations -- same add-ons as `papers/W6_fusion_text_representation.md`.{image_note}

Tiers (percentile of total query-impression volume, computed within the evaluable set):

{tier_defs}

Query counts: head **{tier_counts.get('head', 0)}**, torso **{tier_counts.get('torso', 0)}**,
tail **{tier_counts.get('tail', 0)}** (total **{sum(tier_counts.values())}**).

Relevance labels: LTR judgement-list grades 0-4 (position-debiased CTR quantile bins), same
definition as the primary label set in `papers/w1-section5.md`.

Bold marks the top-scoring system on NDCG@10 within each table (`production` excluded from that
comparison -- see note below); the same system is bolded consistently across the summary table and
every depth-sensitivity table so it can be tracked across k. The image-encoder subsection bolds
separately among just the three image encoders (production/random are reference points there too).

---

{body}

---

> **Note on `production`:** reconstructed from mean observed impression position, not a live
> ranker query. Its graded score benefits from position leakage in the labels (see
> `papers/W4_evaluation_validity_and_systems.md`), so it is a reference point, not a fair baseline.

Source data: `results/summary_by_tier.csv`, `results/run_meta.json`. Generated by
`06_tier_report.py`.
"""

    out_path = config.RESULTS_DIR / "TIERED_RESULTS.md"
    out_path.write_text(report)
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
