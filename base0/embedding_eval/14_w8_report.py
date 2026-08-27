#!/usr/bin/env python3
"""Step 14: render papers/W8_fusion_across_image_encoders.md from results/w8_*.csv.

Full metric depth (NDCG/Recall/Precision/MRR at every k in config.K_VALUES, plus MAP), macro and
impression weighting, for every image encoder x fusion method x text partner combination.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMAGE_LABELS = {
    "siglip_image": "siglip-image",
    "omni_nano_image": "omni-nano-image",
    "omni_small_image": "omni-small-image",
}
TEXT_LABELS = {
    "siglip_text": "text-siglip",  # SigLIP's own text tower over the title -- same encoder as `siglip-image`
    "jina_text": "text-jina",
    "jina_small_text": "text-jina-small",
    "siglip_attr": "attr-siglip",
    "jina_attr": "attr-jina",
    "jina_small_attr": "attr-jina-small",
}
METHOD_LABELS = {"mean_cosine": "mean cosine", "rrf": "RRF (k=60)", "zscore_avg": "z-score average"}
K_VALUES = list(config.K_VALUES)
METRIC_NAMES = {"ndcg": "NDCG", "recall": "Recall", "precision": "Precision", "mrr": "MRR"}
PRIMARY = "ltr"


def label(system: str) -> str:
    if system in IMAGE_LABELS:
        return IMAGE_LABELS[system]
    if system in TEXT_LABELS:
        return TEXT_LABELS[system]
    if system.startswith("fusion["):
        method, rest = system[len("fusion["):].split("]-", 1)
        img, txt = rest.split("+", 1)
        return f"{METHOD_LABELS.get(method, method)}: {IMAGE_LABELS.get(img, img)} + {TEXT_LABELS.get(txt, txt)}"
    return system


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str], best: str | None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        name = label(row["system"])
        cells = [f"{row[c]:.4f}" for c in cols]
        if row["system"] == best:
            name, cells = f"**`{name}`**", [f"**{c}**" for c in cells]
        else:
            name = f"`{name}`"
        lines.append("| " + " | ".join([name] + cells) + " |")
    return "\n".join(lines)


def depth_block(sub: pd.DataFrame, best: str | None) -> str:
    sub = sub.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    parts = []
    for metric, lbl in METRIC_NAMES.items():
        parts.append(f"**{lbl}@k**\n")
        parts.append(md_table(sub, [f"{metric}@{k}" for k in K_VALUES],
                              ["System"] + [f"{lbl}@{k}" for k in K_VALUES], best) + "\n")
    parts.append("**MAP** (no cutoff)\n")
    parts.append(md_table(sub, ["map"], ["System", "MAP"], best) + "\n")
    return "\n".join(parts)


def method_matrix(macro: pd.DataFrame) -> str:
    """Rows = image encoder, cols = fusion method, holding the text partner at attr-siglip."""
    idx = macro.set_index("system")["ndcg@10"]
    lines = ["| Image encoder | " + " | ".join(METHOD_LABELS[m] for m in ("mean_cosine", "rrf", "zscore_avg")) + " | best |",
             "| --- | --- | --- | --- | --- |"]
    for img in IMAGE_LABELS:
        vals = {m: idx.get(f"fusion[{m}]-{img}+siglip_attr", float("nan")) for m in METHOD_LABELS}
        best = max(vals, key=lambda m: vals[m])
        cells = [f"**{vals[m]:.4f}**" if m == best else f"{vals[m]:.4f}" for m in ("mean_cosine", "rrf", "zscore_avg")]
        lines.append("| " + " | ".join([f"`{IMAGE_LABELS[img]}`"] + cells + [METHOD_LABELS[best]]) + " |")
    return "\n".join(lines)


def text_matrix(macro: pd.DataFrame) -> str:
    """Rows = image encoder, cols = text partner, holding the method at RRF."""
    idx = macro.set_index("system")["ndcg@10"]
    cols = list(TEXT_LABELS)
    lines = ["| Image encoder | " + " | ".join(TEXT_LABELS[t] for t in cols) + " | best |",
             "| --- | " + " | ".join("---" for _ in cols) + " | --- |"]
    for img in IMAGE_LABELS:
        vals = {t: idx.get(f"fusion[rrf]-{img}+{t}", float("nan")) for t in cols}
        best = max(vals, key=lambda t: vals[t])
        cells = [f"**{vals[t]:.4f}**" if t == best else f"{vals[t]:.4f}" for t in cols]
        lines.append("| " + " | ".join([f"`{IMAGE_LABELS[img]}`"] + cells + [TEXT_LABELS[best]]) + " |")
    return "\n".join(lines)


def sig_table(sig: pd.DataFrame, note_prefix: str) -> str:
    s = sig[sig["note"].str.startswith(note_prefix)]
    lines = ["| Contrast | Note | Macro Δ | Macro p | Weighted Δ | Weighted p |",
             "| --- | --- | --- | --- | --- | --- |"]
    for _, r in s.iterrows():
        mark = "**sig.**" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "n.s."
        lines.append("| " + " | ".join([
            f"`{label(r['system'])}` vs `{label(r['baseline'])}`", r["note"],
            f"{r['delta']:+.4f}", f"{r['p_value']:.4f} {mark}",
            f"{r['wtd_delta']:+.4f}", f"{r['wtd_p_value']:.4f}"]) + " |")
    return "\n".join(lines)


def main() -> int:
    summary = pd.read_csv(config.RESULTS_DIR / "w8_summary.csv")
    sig = pd.read_csv(config.RESULTS_DIR / "w8_significance.csv")
    meta = json.loads((config.RESULTS_DIR / "w8_meta.json").read_text())

    macro = summary[(summary.label_set == PRIMARY) & (summary.weighting == "macro")]
    wtd = summary[(summary.label_set == PRIMARY) & (summary.weighting == "impression")]

    fused_macro = macro[macro["system"].str.startswith("fusion[")]
    best_fusion = fused_macro.loc[fused_macro["ndcg@10"].idxmax()]
    best_overall = macro.loc[macro["ndcg@10"].idxmax()]

    # headline slice: the recipe W6 recommended, applied to each image encoder
    recipe = [f"fusion[rrf]-{img}+siglip_attr" for img in IMAGE_LABELS]
    slice_systems = recipe + list(IMAGE_LABELS) + ["siglip_attr", "production", "random"]
    head_macro = macro[macro["system"].isin(slice_systems)]
    head_wtd = wtd[wtd["system"].isin(slice_systems)]

    rrf_wins = sum(
        1 for img in IMAGE_LABELS
        if macro.set_index("system")["ndcg@10"].get(f"fusion[rrf]-{img}+siglip_attr", -1)
        == max(macro.set_index("system")["ndcg@10"].get(f"fusion[{m}]-{img}+siglip_attr", -1) for m in METHOD_LABELS)
    )
    attr_wins = sum(
        1 for img in IMAGE_LABELS
        if macro.set_index("system")["ndcg@10"].get(f"fusion[rrf]-{img}+siglip_attr", -1)
        == max(macro.set_index("system")["ndcg@10"].get(f"fusion[rrf]-{img}+{t}", -1) for t in TEXT_LABELS)
    )

    report = f"""# W8 — Does the Fusion Recipe Survive a Change of Image Encoder?

Generated {date.today().isoformat()} by `13_w8_fusion_across_encoders.py` / `14_w8_report.py`.

> **TL;DR**
> RRF is the best fusion method for **{rrf_wins} of 3** image encoders;
> `attr-siglip` is the best text partner for **{attr_wins} of 3**.
> Best fusion overall: **`{label(best_fusion['system'])}`** (NDCG@10 {best_fusion['ndcg@10']:.4f} macro).
> Best system overall including standalones: **`{label(best_overall['system'])}`**
> ({best_overall['ndcg@10']:.4f}).

---

## 1. Why this report exists

W6 established a fusion recipe — **combine by RRF, partner with `attr-siglip`** — but measured it
with a **single image encoder**. That leaves an unanswered question: is RRF the right *method*, or
just the right method *for SigLIP*? A recipe that only works with one image tower is a much weaker
result than a general one.

W8 re-runs the W6 sweep with each of the three W7 image encoders as the fusion target:
**{len(IMAGE_LABELS)} image encoders × {len(TEXT_LABELS)} text representations × {len(METHOD_LABELS)} fusion methods
= {meta['n_combinations']} combinations**, all on the same W7 subset
({meta['n_queries']} queries, {meta['n_products']} products, identical pools for every system).

Labels are the **corrected** LTR judgement list (τ applied to clicks only — see W4).

---

## 2. Does the method conclusion hold? (text partner fixed at `attr-siglip`)

Macro NDCG@10. If RRF wins every row, the method finding generalises beyond SigLIP.

{method_matrix(macro)}

## 3. Does the text-partner conclusion hold? (method fixed at RRF)

Macro NDCG@10. If `attr-siglip` wins every row, the representation finding generalises too.

{text_matrix(macro)}

---

## 4. The W6 recipe applied to each image encoder

Full metric depth for the recommended recipe against each image encoder alone, the best text
system, and the reference points.

### Macro-averaged — every query counts once

{depth_block(head_macro, best_fusion['system'])}

### Impression-weighted — every query counts in proportion to its traffic

{depth_block(head_wtd, best_fusion['system'])}

---

## 5. Significance

Paired bootstrap over queries ({meta['bootstrap_samples']} resamples), NDCG@10, LTR labels.

### Method contrasts — RRF vs the alternatives, per image encoder

{sig_table(sig, "method:")}

### Fusion vs its own inputs

{sig_table(sig, "fusion vs")}

---

## 6. Reading the result

**Method generalises, or it does not.** Section 2 is the whole point of this report: RRF winning
for one image encoder is an implementation detail, RRF winning for all three is a design rule.
Reciprocal rank fusion needs no score normalisation because it discards scores entirely, which is
exactly why it should be robust to swapping an encoder whose similarity scale is unknown.

**Watch the mean-cosine column.** Fusing raw, unnormalised cosines from two differently-scaled
spaces is the failure mode W6 identified. Its severity should *vary* by image encoder, since it
depends on the relative score magnitudes of the two towers — a pairing that happens to have
comparable scales will look fine, which is precisely why it is a trap.

**Fusion vs its inputs remains the decisive test.** A fusion that beats the image alone but not the
best text system alone is not an argument for fusion — it is an argument for the text system. W6
found exactly that with SigLIP; section 5 shows whether a better image encoder changes it.

> **Licensing:** the Jina v5 omni checkpoints are **CC BY-NC 4.0**; commercial use requires a
> licence. SigLIP is Apache-2.0.

Source data: `results/w8_summary.csv`, `results/w8_significance.csv`, `results/w8_per_query.csv`.
"""

    out = Path(config.ROOT) / "papers" / "W8_fusion_across_image_encoders.md"
    out.write_text(report)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
