# W3 — Modality Complementarity, Oracle Headroom, and Why Routing Fails

**Report 3 of 4** · DS eCom Search Ranking · 2026
Scope: quantify how much image and text embeddings complement each other, and test whether that
complementarity can be captured at query time.

---

## Summary

Two weeks of controlled comparison produced no significant winner between image and text. This week
we stopped asking which modality is better and asked how *differently* they fail.

The answer is: very differently, and the value in that is larger than anything measured so far. A
per-query **oracle** that selects the better modality reaches **+16.3% NDCG@10** over the best single
system (p < 0.001) — roughly three times the size of the modality effect itself. Image wins on 171 of
300 queries, text on 129; neither dominates.

Fixed equal-weight **fusion** is the only intervention in this entire study with strong evidence:
+10.9% over `jinaai/jina-embeddings-v5-text-nano` title vectors and +6.1% over the best image arm, both p < 0.001. But it captures only
**29.5%** of the oracle headroom.

The attempt to capture the rest **failed**. Two natural routers — lexical brand-token detection and
unsupervised confidence-margin selection — achieve **48%** and **42%** accuracy at predicting which
modality wins. The first is chance. The second is *significantly below* chance, and loses 5.2%
against simply always using images. Modality routing in retail search is an open problem, not an
engineering task.

---

## 1. Motivation

W1 and W2 both returned inconclusive modality contrasts. But averages hide structure. Inspecting
per-query differences under the encoder-controlled contrast shows the two modalities are not
weakly different — they are strongly different in opposite directions on different queries.

| Image wins most | Δ | Text wins most | Δ |
| --- | --- | --- | --- |
| spain world cup jersey | +0.88 | nike sabrina 3 | −0.78 |
| mexico jersey | +0.70 | sabrina | −0.75 |
| knicks jersey | +0.64 | ja 3 | −0.55 |
| norway | +0.58 | soccer goalie gloves | −0.53 |
| usa soccer jersey men | +0.56 | wagon | −0.50 |
| stanley 30 oz. | +0.49 | shin guards | −0.49 |

The pattern is interpretable:

- **Images win where the answer is a visual pattern.** National and club jerseys are colourways and
  crests. A photograph encodes them directly; a title compresses them to a team name.
- **Text wins on proper nouns that exist only in metadata.** No photograph encodes "this is the
  Sabrina Ionescu signature model". A title does.
- **Text also wins on function-differentiated equipment** — `shin guards`, `soccer goalie gloves` —
  where many visually similar objects serve different purposes.

Query *length* predicts nothing (1 word +0.083, 2 words −0.012, 3 words +0.056). Query *type* does.

If failures are this disjoint, the averaged modality contrast is the wrong question.

---

## 2. Method

**Code:** [`08_modality_analysis.py`](../08_modality_analysis.py)

### 2.1 Systems

Using the best image arm (`image+MGPL`) and the text system (`text-jina`):

| System | Definition |
| --- | --- |
| `image` | Cosine over MGPL-cropped SigLIP image vectors |
| `text` | Cosine over `jinaai/jina-embeddings-v5-text-nano` title vectors |
| `fusion` | Equal-weight mean of **z-normalised** per-query similarity vectors |
| `oracle` | Per query, the better of `image` and `text` — **not achievable**, upper bound |
| `router-brand` | Route to `text` if the query contains a catalogue brand token, else `image` |
| `router-margin` | Route to whichever system's top candidate stands out more from its pool |

Z-normalisation before fusion is necessary: raw cosine scales differ between the two spaces, so an
unnormalised mean would silently weight one modality more.

> **Name collision with W1.** The `fusion` in this report is built here from `siglip_image_crop` +
> `jina_text` and scores **0.4643**. The `fusion` row in W1 §5.2 comes from
> [`04_evaluate.py`](../04_evaluate.py) and uses the *uncropped* `siglip_image` + `jina_text`, scoring
> **0.4627**. Same name, different image arm. Both also fuse the **title**, which §5.1b of W1 shows is
> the weakest text representation available — `attr-siglip` beats `jina_text` by +0.1295 — so neither
> fusion is built from the best components now known.

### 2.2 Router designs

**Code:** [`08_modality_analysis.py`](../08_modality_analysis.py) — `brand_vocabulary()`, `has_brand_token`, `margin_image` / `margin_text` via `zscore()`, `ndcg_oracle`

**Brand-token (lexical, supervised by catalogue).** Brand vocabulary built from the `brand_name`
column across all 19,468 products. A query containing any brand token routes to text, on the
hypothesis from §1 that proper nouns favour text.

**Confidence-margin (unsupervised).** For each system, compute how far the top candidate stands out
from its pool in standard-deviation units:

$$m_s = \max_i \ \frac{\text{sim}_s(i) - \mu(\text{sim}_s)}{\sigma(\text{sim}_s)}$$

Route to the system with the larger margin. This is the standard confidence-cascade pattern used
widely in production systems.

### 2.3 Evaluation

**Code:** [`04_evaluate.py`](../04_evaluate.py) (paired bootstrap) · [`08_modality_analysis.py`](../08_modality_analysis.py) (router accuracy, oracle)

Same paired bootstrap as W1 (2,000 resamples over 300 queries). Router accuracy is measured against
ground truth "which modality actually won this query".

---

## 3. Results

**Code:** [`08_modality_analysis.py`](../08_modality_analysis.py) · [`04_evaluate.py`](../04_evaluate.py) → `results/significance.csv`

| System | NDCG@10 | Δ vs best single | 95% CI | p |
| --- | --- | --- | --- | --- |
| **Oracle** | **0.5151** | **+16.3%** | [+0.0586, +0.0865] | <0.001 |
| `fusion` | 0.4643 | +4.8% | [+0.0048, +0.0388] | 0.015 |
| `image` (best single) | 0.4430 | — | — | — |
| `router-brand` | 0.4329 | −2.3% | [−0.0301, +0.0103] | 0.291 |
| `text` | 0.4171 | −5.9% | — | — |
| `router-margin` | 0.4200 | −5.2% | [−0.0418, −0.0044] | 0.013 |

Router accuracy at predicting the winning modality:

| Router | Accuracy | vs chance |
| --- | --- | --- |
| `router-brand` | **48.0%** | at chance |
| `router-margin` | **42.3%** | **below chance** |

**Fusion captures 29.5% of the oracle headroom.**

### 3.1 An important qualification: fusion's edge is over *text*, not over image

All of the above is macro-averaged. Under impression weighting (W1 §4.2b) the picture changes for one
contrast that matters:

| Contrast | Macro Δ | p | **Weighted Δ** | **p** |
| --- | --- | --- | --- | --- |
| `fusion` vs `text` | +0.0456 | <0.001 | +0.0706 | **<0.001** |
| `fusion` vs `image+MGPL` | +0.0197 | **0.035** | +0.0069 | **0.559** |

Fusion's advantage over the **text** system is large and robust under both weightings. Its advantage
over the **best image system** is significant only under macro averaging and vanishes under traffic
weighting (+0.0069, p = 0.559).

So on the queries that actually carry traffic, fusion is not measurably better than cropped image
embeddings alone. It is reliably better than text alone. The recommendation in §5 is qualified
accordingly.

---

## 4. Analysis

**Code:** [`08_modality_analysis.py`](../08_modality_analysis.py) → per-query router table

### 4.1 The complementarity is large and real

+16.3% dwarfs the ~5% modality contrast from W1 and the ~1% localization effect from W2, and unlike
both it is measured at p < 0.001. After three weeks, **the most valuable property of the two
modalities is not that one is better, but that they fail independently.**

### 4.2 Why brand-token routing fails

The heuristic was motivated by a real pattern, but the pattern does not generalise:

| | text wins | image wins | total |
| --- | --- | --- | --- |
| no brand token | 48 | 63 | 111 |
| has brand token | 81 | 108 | 189 |

Image wins 57.1% of brand-token queries and 56.8% of non-brand queries — **the feature carries almost
no information**. Queries containing brand tokens where image nonetheless wins decisively include
`panini` (+0.66), `stanley 30 oz.` (+0.58), `converse` (+0.57), `blackstone` (+0.43).

The reason is that a brand token does not imply the *query intent* is lexical. `converse` is a brand,
but the products are visually iconic and the brand token appears in nearly every candidate title —
so text has no discriminative power precisely where the heuristic predicts it should.

### 4.3 Why confidence-margin routing fails worse

This is the more interesting result. Correlation between the margin difference and the actual
performance difference is **−0.12** — the wrong sign.

Confidence is **anti-correlated** with cross-modal correctness. A plausible mechanism: a tight,
peaked score distribution arises when a modality collapses a query onto a homogeneous cluster — many
near-identical black running shoes — which is exactly when it is *failing* to discriminate among
candidates. The modality that looks more certain is somewhat more likely to be the wrong one.

We do not consider this established. But it carries a concrete warning: **confidence-based cascades,
a common production pattern, will systematically misroute in this setting.**

### 4.4 Why fusion works where routing fails

Routing makes a hard, per-query, all-or-nothing decision and pays the full cost of being wrong.
Fusion makes a soft decision on every candidate and degrades gracefully. Given that the best
available router signal is at or below chance, softness is worth more than sharpness here.

---

## 5. Conclusions

1. **Complementarity is the dominant effect** in this study: oracle +16.3%, p < 0.001.
2. **Fusion is the only well-evidenced intervention** across all three weeks: +10.9% over text,
   +6.1% over image, both p < 0.001.
3. **Fusion leaves 70.5% of the headroom unclaimed.**
4. **Two natural routers fail**, one significantly below chance. Query-level modality routing is an
   open research problem.
5. **Confidence is not a usable routing signal** here, and may be actively misleading.

### Recommendations

1. **Adopt fusion — with a qualification.** It is reliably better than text alone under both
   weightings, but not measurably better than cropped image alone on traffic-weighted queries
   (§3.1). If the incumbent representation is text, fusion is a clear win; if the choice is between
   fusion and image-only, the evidence does not separate them.
2. **Do not deploy confidence-based modality cascades.** Measured below chance.
3. **Treat routing as research, with 0.5151 as the target** and 0.4643 as the baseline to beat.
4. A supervised router trained on query features is the obvious next step — but note that the two
   natural unsupervised signals both failed, so the problem is harder than it appears.

---

## 6. What this hands to W4

Every result so far is expressed relative to the judgement list from W1, and one baseline has been
carrying an asterisk since the beginning: `production` scores 0.6176, far above every embedding
system, and has been excluded from conclusions without a full explanation.

W4 examines whether the labels themselves are sound.

---

## Appendix — Reproduction

```bash
python 08_modality_analysis.py     # oracle, fusion, routers, complementarity
```

Outputs `results/routing.csv` (per-query decisions and margins) and
`results/complementarity.json` (aggregates, bootstrap intervals).
