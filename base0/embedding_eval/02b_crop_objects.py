#!/usr/bin/env python3
"""Step 2b: crop each product image to the product itself, guided by the product's own metadata.

Why metadata-guided: catalogue imagery frequently shows a model wearing or holding the item. A
class-agnostic "largest object" rule picks the person, not the product -- measured at 38% of images
where DETR fired, because COCO has a `person` class but no socks, cleats, or jerseys. Cropping to the
person actively destroys the signal the embedding is meant to capture.

So the product type is derived from the title and category, and detection is conditioned on it.

Tiers, cheapest acceptable result wins:
  owlv2   Open-vocabulary detection prompted with the product phrase ("soccer cleats", "socks").
          Returns the box matching the product, even when a person dominates the frame.
  detr    facebook/detr-resnet-50 with the `person` class suppressed; largest remaining box.
  bgtrim  Background colour from the border ring; bbox of everything differing from it.
  full    Original frame.

Outputs:
  data/images_cropped/<ecode>.jpg
  data/crop_manifest.csv  -- ecode, method, prompt, score, area_frac, box, ok
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

Box = tuple[int, int, int, int]

# Tokens that describe the shopper, not the object, so they must not become the detection prompt.
AUDIENCE_TOKENS = {
    "men", "mens", "men's", "women", "womens", "women's", "boys", "boys'", "girls", "girls'",
    "kids", "kids'", "youth", "junior", "juniors", "toddler", "infant", "baby", "unisex",
    "adult", "adults", "big", "little", "grade", "school", "preschool",
}


def product_phrases(title: str, category: str) -> list[str]:
    """Short noun phrases describing what the product *is*, for open-vocabulary detection."""
    phrases: list[str] = []

    words = [w.lower() for w in re.findall(r"[A-Za-z']+", str(title))]
    tail = [w for w in words[-4:] if w not in AUDIENCE_TOKENS]
    if len(tail) >= 2:
        phrases.append(" ".join(tail[-2:]))
    if tail:
        phrases.append(tail[-1])

    leaf = str(category).split(">")[-1]
    cat_words = [w for w in re.findall(r"[A-Za-z']+", leaf.lower()) if w not in AUDIENCE_TOKENS]
    if cat_words:
        phrases.append(" ".join(cat_words[-2:]))

    seen, out = set(), []
    for p in phrases:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out[:3]


def background_box(img: Image.Image, tolerance: int = 18) -> Box | None:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    ring = np.concatenate([arr[0, :, :], arr[h - 1, :, :], arr[:, 0, :], arr[:, w - 1, :]], axis=0)
    bg = np.median(ring, axis=0).astype(np.uint8)
    flat = Image.new("RGB", img.size, tuple(int(c) for c in bg))
    mask = ImageChops.difference(img.convert("RGB"), flat).convert("L")
    return mask.point(lambda p: 255 if p > tolerance else 0).getbbox()


def owlv2_box(img, model, processor, device, phrases, threshold):
    import torch

    queries = [f"a photo of {p}" for p in phrases]
    inputs = processor(text=[queries], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([img.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes=target, threshold=threshold
    )[0]
    scores = results["scores"].cpu().numpy()
    if not len(scores):
        return None, None, None
    # text-conditioned, so best match to the product phrase -- not the biggest thing in frame
    j = int(np.argmax(scores))
    x0, y0, x1, y1 = results["boxes"].cpu().numpy()[j]
    label_idx = int(results["labels"].cpu().numpy()[j])
    phrase = phrases[label_idx] if label_idx < len(phrases) else phrases[0]
    return (int(x0), int(y0), int(x1), int(y1)), phrase, float(scores[j])


def detr_box(img, model, processor, device, threshold, suppress_person: bool = True):
    """Largest confident box, optionally excluding `person`.

    suppress_person=False reproduces the naive baseline used in the MGPL ablation.
    """
    import torch

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([img.size[::-1]]).to(device)
    res = processor.post_process_object_detection(outputs, target_sizes=target, threshold=threshold)[0]
    boxes = res["boxes"].cpu().numpy()
    labels = res["labels"].cpu().numpy()
    if not len(boxes):
        return None, None
    names = [model.config.id2label[int(i)] for i in labels]
    keep = [i for i, n in enumerate(names) if n != "person"] if suppress_person else list(range(len(names)))
    if not keep:
        return None, None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    j = max(keep, key=lambda i: areas[i])
    x0, y0, x1, y1 = boxes[j]
    return (int(x0), int(y0), int(x1), int(y1)), names[j]


def square_pad(box: Box, size: tuple[int, int], pad_frac: float) -> Box:
    w, h = size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(x1 - x0, y1 - y0) * (1 + pad_frac), min(w, h))
    x0 = int(round(max(0, min(cx - side / 2, w - side))))
    y0 = int(round(max(0, min(cy - side / 2, h - side))))
    return x0, y0, int(round(x0 + side)), int(round(y0 + side))


def usable(box: Box | None, size: tuple[int, int], lo: float, hi: float) -> bool:
    if box is None:
        return False
    frac = (box[2] - box[0]) * (box[3] - box[1]) / float(size[0] * size[1])
    return lo <= frac <= hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("owlv2", "detr", "bgtrim", "hybrid", "detr_largest"),
        default="hybrid",
        help="detr_largest reproduces the naive class-agnostic baseline (person not suppressed)",
    )
    parser.add_argument("--owl-threshold", type=float, default=0.12)
    parser.add_argument("--detr-threshold", type=float, default=0.7)
    parser.add_argument("--pad-frac", type=float, default=0.08)
    parser.add_argument("--min-area", type=float, default=0.02)
    parser.add_argument("--max-area", type=float, default=0.98)
    parser.add_argument("--out-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="process only N images (smoke test)")
    parser.add_argument("--force", action="store_true", help="recrop even if a crop already exists")
    args = parser.parse_args()

    config.ensure_dirs()
    naive = args.method == "detr_largest"
    out_dir = config.CROP_NAIVE_DIR if naive else config.CROP_DIR
    out_manifest = config.CROP_NAIVE_MANIFEST_CSV if naive else config.CROP_MANIFEST_CSV
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)
    manifest = manifest[manifest["ok"] == True]  # noqa: E712
    products = pd.read_csv(config.PRODUCTS_CSV)[["ecode", "product_title", "category_name"]]
    manifest = manifest.merge(products, on="ecode", how="left")
    if args.limit:
        manifest = manifest.head(args.limit)

    # crops are shared across experiment variants; only compute the ones not already present
    prior = pd.DataFrame()
    if out_manifest.exists() and not args.force:
        prior = pd.read_csv(out_manifest)
        done = set(prior.loc[prior["ok"] == True, "ecode"])  # noqa: E712
        done = {e for e in done if (out_dir / f"{e}.jpg").exists()}
        before = len(manifest)
        manifest = manifest[~manifest["ecode"].isin(done)]
        print(f"reusing {before - len(manifest)} existing crops")

    print(f"cropping {len(manifest)} images  method={args.method}")
    if manifest.empty:
        print("nothing to do")
        return 0

    import torch

    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    owl = owl_proc = detr = detr_proc = None
    if args.method in ("owlv2", "hybrid"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        print(f"loading {config.OWL_MODEL} on {device}")
        owl_proc = Owlv2Processor.from_pretrained(config.OWL_MODEL)
        owl = Owlv2ForObjectDetection.from_pretrained(config.OWL_MODEL).to(device).eval()
    if args.method in ("detr", "hybrid", "detr_largest"):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        print(f"loading {config.DETECTOR_MODEL} on {device}")
        detr_proc = AutoImageProcessor.from_pretrained(config.DETECTOR_MODEL)
        detr = AutoModelForObjectDetection.from_pretrained(config.DETECTOR_MODEL).to(device).eval()

    rows = []
    for i, row in enumerate(manifest.itertuples(), start=1):
        dest = out_dir / f"{row.ecode}.jpg"
        # resolve from config, not the manifest: manifest paths go stale if the data dir moves
        src = config.IMAGE_DIR / f"{row.ecode}.jpg"
        try:
            img = Image.open(src).convert("RGB")
        except OSError as exc:
            rows.append({"ecode": row.ecode, "method": "error", "prompt": "", "score": np.nan,
                         "area_frac": np.nan, "box": "", "ok": False, "error": str(exc)})
            continue

        box, method, prompt, score = None, "full", "", np.nan
        phrases = product_phrases(row.product_title, row.category_name)

        if owl is not None and phrases:
            cand, phrase, sc = owlv2_box(img, owl, owl_proc, device, phrases, args.owl_threshold)
            if usable(cand, img.size, args.min_area, args.max_area):
                box, method, prompt, score = cand, "owlv2", phrase, sc
        if box is None and detr is not None:
            cand, name = detr_box(
                img, detr, detr_proc, device, args.detr_threshold, suppress_person=not naive
            )
            if usable(cand, img.size, args.min_area, args.max_area):
                box, method, prompt = cand, "detr", name or ""
        if box is None and args.method in ("bgtrim", "hybrid", "detr_largest"):
            cand = background_box(img)
            if usable(cand, img.size, args.min_area, args.max_area):
                box, method = cand, "bgtrim"
        if box is None:
            box = (0, 0, img.size[0], img.size[1])

        crop_box = square_pad(box, img.size, args.pad_frac)
        img.crop(crop_box).resize((args.out_size, args.out_size), Image.LANCZOS).save(dest, quality=90)

        area = (box[2] - box[0]) * (box[3] - box[1]) / float(img.size[0] * img.size[1])
        rows.append({"ecode": row.ecode, "method": method, "prompt": prompt,
                     "score": round(score, 4) if score == score else np.nan,
                     "area_frac": round(area, 4), "box": str(crop_box), "ok": True, "error": ""})

        if i % 250 == 0 or i == len(manifest):
            print(f"  {i}/{len(manifest)}")

    crop_manifest = pd.DataFrame(rows)
    if not prior.empty:
        crop_manifest = pd.concat([prior, crop_manifest], ignore_index=True)
        crop_manifest = crop_manifest.drop_duplicates(subset="ecode", keep="last")
    crop_manifest.to_csv(out_manifest, index=False)

    print("\nmethod used:")
    print(crop_manifest["method"].value_counts().to_string())
    good = crop_manifest[crop_manifest["ok"]]
    print(f"\nmedian object area fraction: {good['area_frac'].median():.3f}")
    if not naive:
        print("\ntop detection prompts:")
        print(good.loc[good["method"] == "owlv2", "prompt"].value_counts().head(12).to_string())
    else:
        print("\ntop DETR classes selected (person NOT suppressed):")
        print(good.loc[good["method"] == "detr", "prompt"].value_counts().head(12).to_string())
    print(f"\nsaved -> {out_dir}")
    print(f"saved -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
