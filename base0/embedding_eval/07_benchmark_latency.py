#!/usr/bin/env python3
"""Step 7: measure indexing and serving latency for every component of the pipeline.

Two regimes matter for a search deployment:
  online   -- batch size 1, the p50/p95 a query pays at request time
  offline  -- batched throughput, what catalogue re-indexing costs

Outputs: results/latency.csv, results/latency_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CATALOGUE_SIZE = 149_801  # DSG web-active ecodes, current snapshot


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_it(fn, device: torch.device, n: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarize(name: str, stage: str, mode: str, times: list[float], items: int) -> dict:
    times_sorted = sorted(times)
    return {
        "component": name,
        "stage": stage,
        "mode": mode,
        "items_per_call": items,
        "n_calls": len(times),
        "p50_ms": round(statistics.median(times_sorted), 3),
        "p95_ms": round(times_sorted[max(0, int(0.95 * len(times_sorted)) - 1)], 3),
        "mean_ms": round(statistics.fmean(times_sorted), 3),
        "ms_per_item": round(statistics.fmean(times_sorted) / items, 4),
        "items_per_sec": round(items * 1000.0 / statistics.fmean(times_sorted), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForObjectDetection, AutoProcessor

    config.ensure_dirs()
    device = pick_device(args.device)
    print(f"device: {device}  torch {torch.__version__}")

    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    queries = sorted(test_set["search_term"].unique())[: args.batch_size]
    titles = test_set["product_title"].astype(str).drop_duplicates().tolist()[: args.batch_size]
    ecodes = test_set["ecode"].drop_duplicates().tolist()[: args.batch_size]
    img_paths = [config.IMAGE_DIR / f"{e}.jpg" for e in ecodes]
    img_paths = [p for p in img_paths if p.exists()]
    images = [Image.open(p).convert("RGB") for p in img_paths]

    rows: list[dict] = []

    # ---- SigLIP ----
    siglip = AutoModel.from_pretrained(config.SIGLIP_MODEL).to(device).eval()
    sig_proc = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)

    def siglip_text(batch: list[str]):
        def run():
            inputs = sig_proc(
                text=batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                siglip.get_text_features(**inputs)
        return run

    def siglip_image(batch):
        def run():
            inputs = sig_proc(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                siglip.get_image_features(**inputs)
        return run

    rows.append(summarize("SigLIP text tower", "query encode", "online (bs=1)",
                          time_it(siglip_text(queries[:1]), device, args.repeats), 1))
    rows.append(summarize("SigLIP text tower", "title encode", "offline (batched)",
                          time_it(siglip_text(titles), device, 10), len(titles)))
    rows.append(summarize("SigLIP image tower", "image encode", "online (bs=1)",
                          time_it(siglip_image(images[:1]), device, args.repeats), 1))
    rows.append(summarize("SigLIP image tower", "image encode", "offline (batched)",
                          time_it(siglip_image(images), device, 10), len(images)))
    del siglip

    # ---- Jina ----
    jina = AutoModel.from_pretrained(config.JINA_MODEL, trust_remote_code=True).to(device).eval()

    def jina_encode(batch: list[str], prompt: str):
        def run():
            with torch.no_grad():
                jina.encode(texts=batch, task="retrieval", prompt_name=prompt)
        return run

    rows.append(summarize("Jina v5 text nano", "query encode", "online (bs=1)",
                          time_it(jina_encode(queries[:1], "query"), device, args.repeats), 1))
    rows.append(summarize("Jina v5 text nano", "title encode", "offline (batched)",
                          time_it(jina_encode(titles, "document"), device, 10), len(titles)))
    del jina

    # ---- DETR object detection (indexing only) ----
    try:
        det_proc = AutoImageProcessor.from_pretrained(config.DETECTOR_MODEL)
        detr = AutoModelForObjectDetection.from_pretrained(config.DETECTOR_MODEL).to(device).eval()

        def detr_run(batch):
            def run():
                inputs = det_proc(images=batch, return_tensors="pt").to(device)
                with torch.no_grad():
                    detr(**inputs)
            return run

        rows.append(summarize("DETR resnet-50", "object detect", "online (bs=1)",
                              time_it(detr_run(images[:1]), device, args.repeats), 1))
        rows.append(summarize("DETR resnet-50", "object detect", "offline (batched)",
                              time_it(detr_run(images[:8]), device, 5), 8))
        del detr
    except Exception as exc:  # detector is optional
        print(f"skipping DETR benchmark: {exc}")

    # ---- retrieval maths ----
    rng = np.random.default_rng(config.RANDOM_SEED)
    q = rng.standard_normal(768).astype(np.float32)
    q /= np.linalg.norm(q)
    for n in (60, 10_000, CATALOGUE_SIZE):
        docs = rng.standard_normal((n, 768)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)

        def run(d=docs):
            scores = d @ q
            np.argpartition(-scores, min(10, len(scores) - 1))[:10]

        label = "re-rank pool" if n == 60 else f"brute-force scan ({n:,})"
        rows.append(summarize("Cosine + top-10", label, "online (bs=1)",
                              time_it(run, torch.device("cpu"), 50), 1))

    latency = pd.DataFrame(rows)
    latency.to_csv(config.RESULTS_DIR / "latency.csv", index=False)

    def per_item(component: str, stage: str) -> float:
        r = latency[(latency["component"] == component) & (latency["stage"] == stage)
                    & (latency["mode"].str.startswith("offline"))]
        return float(r["ms_per_item"].iloc[0]) if not r.empty else float("nan")

    img_ms = per_item("SigLIP image tower", "image encode")
    txt_ms = per_item("Jina v5 text nano", "title encode")
    det_ms = per_item("DETR resnet-50", "object detect")

    summary = {
        "device": str(device),
        "torch": torch.__version__,
        "catalogue_size": CATALOGUE_SIZE,
        "full_catalogue_hours": {
            "siglip_image": round(CATALOGUE_SIZE * img_ms / 3_600_000, 2),
            "jina_title": round(CATALOGUE_SIZE * txt_ms / 3_600_000, 2),
            "detr_crop": round(CATALOGUE_SIZE * det_ms / 3_600_000, 2) if det_ms == det_ms else None,
        },
        "embedding_bytes_per_product_fp32": 768 * 4,
        "catalogue_index_mb_fp32": round(CATALOGUE_SIZE * 768 * 4 / 1024**2, 1),
    }
    (config.RESULTS_DIR / "latency_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(latency.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {config.RESULTS_DIR / 'latency.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
