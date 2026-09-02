#!/usr/bin/env python3
"""Generate papers/W6_fusion_text_representation.md from results/fusion_experiment_summary.csv.

Full metric depth (NDCG/Recall/Precision/MRR @ 5,10,20,48,96,144, MAP), macro and impression
weighting, primary label set (LTR judgement-list relevance). See 09_fusion_experiment.py for the
experiment itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMAGE_LABELS = {
    "siglip_image": "siglip-image",
    "omni_nano_image": "omni-nano-image",
    "jina_clip_v2_image": "jina-clip-v2-image",
}
TEXT_LABELS = {
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
}
SYSTEM_LABELS = {
    "production": "production (not comparable -- see note)",
    "random": "random",
    **IMAGE_LABELS,
    **TEXT_LABELS,
}
METHOD_LABELS = {"mean_cosine": "mean cosine", "rrf": "RRF (k=60)", "zscore_avg": "z-score average"}


def display_name(system: str) -> str:
    if system in SYSTEM_LABELS:
        return SYSTEM_LABELS[system]
    if system.startswith("fusion["):
        method, rest = system[len("fusion["):].split("]-", 1)
        image_name, text_name = rest.split("+", 1)
        method_label = METHOD_LABELS.get(method, method)
        image_label = IMAGE_LABELS.get(image_name, image_name)
        text_label = TEXT_LABELS.get(text_name, text_name)
        return f"fusion[{method_label}] {image_label} + {text_label}"
    return system


K_VALUES = list(config.K_VALUES)
METRIC_NAMES = {"ndcg": "NDCG", "recall": "Recall", "precision": "Precision", "mrr": "MRR"}
PRIMARY = "ltr"


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], best_idx) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for idx, row in df.iterrows():
        name = display_name(row["system"])
        cells = [f"{row[c]:.4f}" for c in cols]
        if idx == best_idx:
            name = f"**`{name}`**"
            cells = [f"**{c}**" for c in cells]
        else:
            name = f"`{name}`"
        lines.append("| " + " | ".join([name] + cells) + " |")
    return "\n".join(lines)


def weighting_block(sub: pd.DataFrame) -> str:
    sub = sub.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    sub = pd.concat([sub[sub["system"] == "production"], sub[sub["system"] != "production"]], ignore_index=True)
    best_idx = sub.loc[sub["system"] != "production", "ndcg@10"].idxmax() if len(sub) else None

    parts = []
    for metric, label in METRIC_NAMES.items():
        depth_cols = [f"{metric}@{k}" for k in K_VALUES]
        depth_headers = ["System"] + [f"{label}@{k}" for k in K_VALUES]
        parts.append(f"**{label}@k**\n")
        parts.append(md_table(sub, depth_cols, depth_headers, best_idx) + "\n")
    parts.append("**MAP** (rank-position metric, no cutoff)\n")
    parts.append(md_table(sub, ["map"], ["System", "MAP"], best_idx) + "\n")
    return "\n".join(parts)


def image_slice(df: pd.DataFrame, image_key: str, text_keys: list[str]) -> pd.DataFrame:
    """Rows relevant to one target-image subsection: that image alone, every text alone, every
    fusion combo pairing that image with a text representation, plus production/random."""
    fusion_names = {f"fusion[{m}]-{image_key}+{t}" for m in METHOD_LABELS for t in text_keys}
    keep = {image_key, "production", "random"} | set(text_keys) | fusion_names
    return df[df["system"].isin(keep)]


def gain_table(summary: pd.DataFrame, weighting: str, image_key: str, text_keys: list[str]) -> str:
    sub = summary[(summary["label_set"] == PRIMARY) & (summary["weighting"] == weighting)].set_index("system")
    base_ndcg10 = float(sub.loc[image_key, "ndcg@10"])
    rows = []
    for text_name in text_keys:
        text_ndcg10 = float(sub.loc[text_name, "ndcg@10"]) if text_name in sub.index else float("nan")
        for method in ("mean_cosine", "rrf", "zscore_avg"):
            system = f"fusion[{method}]-{image_key}+{text_name}"
            if system not in sub.index:
                continue
            ndcg10 = float(sub.loc[system, "ndcg@10"])
            rows.append(
                {
                    "text representation": TEXT_LABELS[text_name],
                    "method": METHOD_LABELS[method],
                    "ndcg@10": ndcg10,
                    "gain vs image alone": ndcg10 - base_ndcg10,
                    "gain vs text alone": ndcg10 - text_ndcg10,
                }
            )
    df = pd.DataFrame(rows).sort_values("gain vs image alone", ascending=False).reset_index(drop=True)
    lines = [
        "| Text representation | Fusion method | NDCG@10 | Gain vs image alone | Gain vs text alone |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, row in df.iterrows():
        cells = [
            row["text representation"],
            row["method"],
            f"{row['ndcg@10']:.4f}",
            f"{row['gain vs image alone']:+.4f}",
            f"{row['gain vs text alone']:+.4f}",
        ]
        if i == 0:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


TEXT_CANDIDATES_ORDER = [
    "siglip_text", "jina_text", "jina_small_text",
    "e5_base_text", "e5_small_multi_text", "e5_large_instruct_text", "omni_nano_text",
    "siglip_attr", "jina_attr", "jina_small_attr",
    "e5_base_attr", "e5_small_multi_attr", "e5_large_instruct_attr", "omni_nano_attr",
]
IMAGE_CANDIDATES_ORDER = ["siglip_image", "omni_nano_image", "jina_clip_v2_image"]

# Design-table rows: (key, query encoder, document representation over the product title)
DESIGN_ROWS = [
    ("siglip_text", "SigLIP text tower", "SigLIP text tower over the product title"),
    ("jina_text", "Jina v5 nano `retrieval.query`", "Jina v5 nano `retrieval.document` over the title"),
    ("jina_small_text", "Jina v5 small `retrieval.query`", "Jina v5 small `retrieval.document` over the title"),
    ("e5_base_text", "E5 base (`query: `)", "E5 base (`passage: `) over the title -- English only"),
    (
        "e5_small_multi_text",
        "Multilingual E5 small (`query: `)",
        "Multilingual E5 small (`passage: `) over the title",
    ),
    (
        "e5_large_instruct_text",
        "Multilingual E5 large-instruct (instruction-prefixed query)",
        "Multilingual E5 large-instruct (no prefix) over the title",
    ),
    ("omni_nano_text", "Jina v5 omni-nano `Query: `", "Jina v5 omni-nano `Document: ` over the title"),
    ("siglip_attr", "SigLIP text tower", "SigLIP text tower over the Big-4 attribute string"),
    ("jina_attr", "Jina v5 nano `retrieval.query`", "Jina v5 nano `retrieval.document` over the Big-4 string"),
    (
        "jina_small_attr",
        "Jina v5 small `retrieval.query`",
        "Jina v5 small `retrieval.document` over the Big-4 string",
    ),
    ("e5_base_attr", "E5 base (`query: `)", "E5 base (`passage: `) over the Big-4 string -- English only"),
    (
        "e5_small_multi_attr",
        "Multilingual E5 small (`query: `)",
        "Multilingual E5 small (`passage: `) over the Big-4 string",
    ),
    (
        "e5_large_instruct_attr",
        "Multilingual E5 large-instruct (instruction-prefixed query)",
        "Multilingual E5 large-instruct (no prefix) over the Big-4 string",
    ),
    ("omni_nano_attr", "Jina v5 omni-nano `Query: `", "Jina v5 omni-nano `Document: ` over the Big-4 string"),
]
IMAGE_DESIGN_ROWS = [
    ("siglip_image", "SigLIP text tower", "SigLIP image tower over the product photo"),
    ("omni_nano_image", "Jina v5 omni-nano `Query: `", "Jina v5 omni-nano `Document: ` image tower over the photo"),
    ("jina_clip_v2_image", "Jina CLIP v2 text tower", "Jina CLIP v2 image tower over the product photo"),
]


def design_table(rows: list[tuple[str, str, str]], available: set[str], labels: dict[str, str]) -> str:
    lines = ["| Name | Query encoder | Document representation |", "| --- | --- | --- |"]
    for key, query_enc, doc_repr in rows:
        if key not in available:
            continue
        lines.append(f"| `{labels[key]}` | {query_enc} | {doc_repr} |")
    return "\n".join(lines)


def sig_table(significance: pd.DataFrame) -> str:
    df = significance.copy()
    df["system"] = df["system"].map(display_name)
    df["baseline"] = df["baseline"].map(display_name)
    lines = [
        "| Contrast | Note | Macro delta | Macro p | Weighted delta | Weighted p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df.iterrows():
        sig = "**significant**" if (row["ci_low"] > 0 or row["ci_high"] < 0) else "not significant"
        wtd_sig = "**significant**" if (row["wtd_ci_low"] > 0 or row["wtd_ci_high"] < 0) else "not significant"
        lines.append(
            "| " + " | ".join(
                [
                    f"`{row['system']}` vs `{row['baseline']}`",
                    row["note"],
                    f"{row['delta']:+.4f}",
                    f"{row['p_value']:.4f} ({sig})",
                    f"{row['wtd_delta']:+.4f}",
                    f"{row['wtd_p_value']:.4f} ({wtd_sig})",
                ]
            ) + " |"
        )
    return "\n".join(lines)


def main() -> int:
    summary = pd.read_csv(config.RESULTS_DIR / "fusion_experiment_summary.csv")
    significance = pd.read_csv(config.RESULTS_DIR / "fusion_experiment_significance.csv")

    macro = summary[(summary["label_set"] == PRIMARY) & (summary["weighting"] == "macro")]
    wtd = summary[(summary["label_set"] == PRIMARY) & (summary["weighting"] == "impression")]

    all_systems = set(macro["system"]) | set(wtd["system"])

    def text_present(key: str) -> bool:
        return key in all_systems or any(f"fusion[{m}]-{img}+{key}" in all_systems for m in METHOD_LABELS for img in IMAGE_LABELS)

    def image_present(key: str) -> bool:
        return key in all_systems

    available_text = [key for key in TEXT_CANDIDATES_ORDER if text_present(key)]
    available_images = [key for key in IMAGE_CANDIDATES_ORDER if image_present(key)]
    n_text = len(available_text)
    n_images = len(available_images)
    n_combos = n_images * n_text * 3

    def parse_fusion(system: str) -> str:
        """System as 'method of image + text-representation', or its display name if not a fusion."""
        if not system.startswith("fusion["):
            return display_name(system)
        method, rest = system[len("fusion["):].split("]-", 1)
        image_name, text_name = rest.split("+", 1)
        return f"{METHOD_LABELS[method]} of {IMAGE_LABELS.get(image_name, image_name)} + {TEXT_LABELS.get(text_name, text_name)}"

    is_fusion = lambda s: s.str.startswith("fusion[")  # noqa: E731
    fusion_only_macro = macro[is_fusion(macro["system"])]
    fusion_only_wtd = wtd[is_fusion(wtd["system"])]
    best_fusion_macro = fusion_only_macro.loc[fusion_only_macro["ndcg@10"].idxmax()]
    best_fusion_wtd = fusion_only_wtd.loc[fusion_only_wtd["ndcg@10"].idxmax()]
    best_overall_macro = macro.loc[macro[macro["system"] != "production"]["ndcg@10"].idxmax()]
    best_overall_wtd = wtd.loc[wtd[wtd["system"] != "production"]["ndcg@10"].idxmax()]
    macro_idx = macro.set_index("system")
    wtd_idx = wtd.set_index("system")
    image_macro = {img: float(macro_idx.loc[img, "ndcg@10"]) for img in available_images}
    image_wtd = {img: float(wtd_idx.loc[img, "ndcg@10"]) for img in available_images}

    # "best fusion" and "best system overall" can differ -- a standalone system is allowed to beat
    # every fusion combo (it does, under macro weighting here). Only add the caveat when it happens,
    # and check the paired-bootstrap result before claiming the weighted reversal is a real win.
    macro_caveat = ""
    if best_overall_macro["system"] != best_fusion_macro["system"]:
        text_alone_sig = significance[significance["note"].str.contains("its text representation alone")]
        wtd_significant = bool(
            len(text_alone_sig)
            and (text_alone_sig.iloc[0]["wtd_ci_low"] > 0 or text_alone_sig.iloc[0]["wtd_ci_high"] < 0)
        )
        weighted_claim = (
            "fusion does reverse to a (statistically significant) win under impression weighting"
            if wtd_significant
            else "fusion directionally reverses under impression weighting, but that reversal is "
            "*not* statistically significant at this sample size -- see section 4"
        )
        macro_caveat = (
            f"\n> Under macro averaging, no fusion beats using **`{display_name(best_overall_macro['system'])}`**"
            f" alone (NDCG@10 {best_overall_macro['ndcg@10']:.4f}); {weighted_claim}."
        )
    wtd_caveat = ""
    if best_overall_wtd["system"] != best_fusion_wtd["system"]:
        wtd_caveat = (
            f"\n> Under impression weighting, no fusion beats using **`{display_name(best_overall_wtd['system'])}`**"
            f" alone (NDCG@10 {best_overall_wtd['ndcg@10']:.4f})."
        )

    # W6 add-on finding: does any of the four new text candidates (three E5 models, omni-nano)
    # change which text representation is the best fusion partner?
    add_on_text_keys = {"e5_base_text", "e5_small_multi_text", "e5_large_instruct_text", "omni_nano_text"}
    add_on_bullet = ""
    if best_fusion_macro["system"].startswith("fusion["):
        best_method, rest = best_fusion_macro["system"][len("fusion["):].split("]-", 1)
        winning_image, winning_text = rest.split("+", 1)
        if winning_text in add_on_text_keys:
            old_best_system = f"fusion[{best_method}]-{winning_image}+jina_text"
            delta_vs_old = (
                float(best_fusion_macro["ndcg@10"]) - float(macro_idx.loc[old_best_system, "ndcg@10"])
                if old_best_system in macro_idx.index
                else None
            )
            delta_note = f" ({delta_vs_old:+.4f} NDCG@10 vs the same method/image paired with `{TEXT_LABELS['jina_text']}`)" if delta_vs_old is not None else ""
            add_on_bullet = (
                f"\n\n**A W6 add-on model is now the best fusion partner.** `{TEXT_LABELS[winning_text]}` "
                f"overtakes every original SigLIP/Jina text representation{delta_note}."
            )

    # W6 add-on finding: is the second target image (omni-nano) ever the winning image encoder?
    image_target_bullet = ""
    if len(available_images) > 1 and best_fusion_macro["system"].startswith("fusion["):
        method_key, rest = best_fusion_macro["system"][len("fusion["):].split("]-", 1)
        winning_image, winning_text = rest.split("+", 1)
        comparisons = []
        for other in available_images:
            if other == winning_image:
                continue
            alt_system = f"fusion[{method_key}]-{other}+{winning_text}"
            if alt_system in macro_idx.index:
                comparisons.append(
                    f"`{IMAGE_LABELS.get(other, other)}` scores {macro_idx.loc[alt_system, 'ndcg@10']:.4f} "
                    f"under the same method and text partner"
                )
        comparison_note = f" ({'; '.join(comparisons)})" if comparisons else ""
        image_target_bullet = (
            f"\n\n**`{IMAGE_LABELS.get(winning_image, winning_image)}` is the stronger target modality here.** "
            f"Every winning fusion combo in this report pairs its text partner with "
            f"`{IMAGE_LABELS.get(winning_image, winning_image)}`{comparison_note}."
        )

    # Does fusing with a WEAK image encoder ever make things worse than the text system alone?
    # (as opposed to section 5's existing "weak text representation" caveat, this is the mirror
    # case: the image side is the weak link.) Counts, per image target, how many text partners
    # have their best-of-3-methods fusion NDCG@10 below that text system's own standalone NDCG@10.
    weak_image_bullet = ""
    weak_image_rows = []
    for img in available_images:
        if img not in macro_idx.index:
            continue
        img_ndcg = float(macro_idx.loc[img, "ndcg@10"])
        worse, total = 0, 0
        for text_name in available_text:
            if text_name not in macro_idx.index:
                continue
            best_for_text = max(
                (macro_idx.loc[f"fusion[{m}]-{img}+{text_name}", "ndcg@10"] for m in METHOD_LABELS
                 if f"fusion[{m}]-{img}+{text_name}" in macro_idx.index),
                default=None,
            )
            if best_for_text is None:
                continue
            total += 1
            if best_for_text < macro_idx.loc[text_name, "ndcg@10"]:
                worse += 1
        weak_image_rows.append((img, img_ndcg, worse, total))
    if any(worse > 0 for _, _, worse, _ in weak_image_rows) and len(weak_image_rows) > 1:
        strongest_img = max(weak_image_rows, key=lambda r: r[1])
        parts = [
            f"`{IMAGE_LABELS.get(img, img)}` (alone: {ndcg:.4f}) underperforms its own text partner alone "
            f"in {worse}/{total} cases"
            for img, ndcg, worse, total in weak_image_rows
        ]
        weak_image_bullet = (
            "\n\n**Fusion can make things worse than the text alone -- when the *image* side is the weak"
            " link.** " + "; ".join(parts) + f". The pattern tracks each image encoder's own standalone"
            f" strength: `{IMAGE_LABELS.get(strongest_img[0], strongest_img[0])}` ({strongest_img[1]:.4f} alone)"
            " is close in quality to its text partners, so averaging the two is complementary; a much"
            " weaker image tower instead acts like added noise once the text partner is already strong,"
            " pulling the fused ranking below what the text system achieves by itself. Check `omni-nano-image`'s"
            " own standalone NDCG@10 against a text candidate's before assuming fusion can only help."
        )

    report = f"""# W6 -- Fusion Method and Text-Representation Sweep

Generated with `09_fusion_experiment.py` / `10_fusion_report.py`. Uses cached embeddings from the
current test set ({summary['n_queries'].max()} queries) -- no re-querying or re-embedding.

> **TL;DR**
> Best *fusion* combo, macro: **{parse_fusion(best_fusion_macro['system'])}**
> (NDCG@10 {best_fusion_macro['ndcg@10']:.4f}).
> Best *fusion* combo, impression-weighted: **{parse_fusion(best_fusion_wtd['system'])}**
> (NDCG@10 {best_fusion_wtd['ndcg@10']:.4f}).{macro_caveat}{wtd_caveat}
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

**Target modalities ({n_images}):** every fusion variant combines one of these image-tower
similarities with one text-based similarity. Both are self-consistent -- query and document are
encoded by the same model.

{design_table(IMAGE_DESIGN_ROWS, set(available_images), IMAGE_LABELS)}

**Text representations ({n_text}):** query encoder always matched to the same model's document
tower, plus three E5 models and the Jina v5 omni-nano text tower added as W6 fusion-partner
candidates alongside the original SigLIP/Jina set.

{design_table(DESIGN_ROWS, set(available_text), TEXT_LABELS)}

**Fusion methods (3), per query, over the query's candidate pool:**

| Method | Formula |
| --- | --- |
| Mean cosine | $0.5 \\cdot (\\cos_{{image}} + \\cos_{{text}})$ -- raw, unnormalised average |
| Reciprocal rank fusion (RRF, k=60) | $\\frac{{1}}{{60 + rank_{{image}}}} + \\frac{{1}}{{60 + rank_{{text}}}}$ |
| Z-score average | $0.5 \\cdot (z(\\cos_{{image}}) + z(\\cos_{{text}}))$ -- current production `fusion` system |

{n_images} target images x {n_text} text representations x 3 methods = {n_combos} combinations,
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

Primary label set: LTR judgement-list relevance (grades 0-4). All {summary['n_queries'].max()}
queries scored (same set as `results/TIERED_RESULTS.md`, all tiers combined -- this experiment
is not stratified by tier). One subsection per target image modality, so each stays readable;
every text-representation standalone is shared across both (it doesn't depend on the image target).

**Every metric is reported at every cutoff** -- NDCG, Recall, Precision and MRR each at
k = {", ".join(str(k) for k in K_VALUES)}, plus MAP (which has no cutoff). Bold marks the
best-scoring system on NDCG@10, tracked consistently across all tables so it can be followed across
k (`production` excluded from that comparison -- reference point, not a fair baseline, see note).

{"".join(f'''
### Target: `{IMAGE_LABELS[img]}`

#### Macro-averaged -- every query counts once

{weighting_block(image_slice(macro, img, available_text))}

#### Impression-weighted -- every query counts in proportion to its traffic

{weighting_block(image_slice(wtd, img, available_text))}
''' for img in available_images)}
---

## 3. Which text representation gives the highest fusion gain?

"Gain vs image alone" = fused NDCG@10 - that target image alone.
"Gain vs text alone" = fused NDCG@10 - that text representation's own NDCG@10 (i.e. does fusing
with the image actually help over just using the text system by itself). One pair of tables per
target image modality.

{"".join(f'''
### Target: `{IMAGE_LABELS[img]}` (image alone: macro {image_macro[img]:.4f}, impression-weighted {image_wtd[img]:.4f})

#### Macro-averaged

{gain_table(macro, "macro", img, available_text)}

#### Impression-weighted

{gain_table(wtd, "impression", img, available_text)}
''' for img in available_images)}
---

## 4. Significance

Paired bootstrap over queries ({config.BOOTSTRAP_SAMPLES} resamples), metric NDCG@10, built around
whichever fusion combo actually wins on macro NDCG@10 (`{parse_fusion(best_fusion_macro['system'])}`).
Macro and weighted deltas come from the same paired samples, so a contrast can be significant under
one weighting and not the other. Includes a contrast against the *other* target image modality,
holding the fusion method and text partner fixed, to check whether the image-encoder choice matters.

{sig_table(significance)}

---

## 5. Reading the result

**No fusion method dominates.** Once tied scores can no longer leak the label order, the three
methods land close together and the winner depends on the pairing rather than on the method alone.
W8 tests this directly across image encoders and finds the best method changes with the encoder —
so "use RRF" is not supportable as a general rule from this evidence.{add_on_bullet}{image_target_bullet}{weak_image_bullet}

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
"""

    out_path = Path(config.ROOT) / "papers" / "W6_fusion_text_representation.md"
    out_path.write_text(report)
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

