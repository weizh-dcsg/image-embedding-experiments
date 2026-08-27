#!/usr/bin/env python3
"""Step 3c: encode the W7 image-encoder comparison set with a Jina v5 omni model.

W7 compares product-image encoders on identical queries and candidate pools:
  siglip_image      SigLIP text tower query  vs SigLIP image tower       (already cached)
  omni_nano_image   omni text query          vs omni image, ~1.0B, 768d
  omni_small_image  omni text query          vs omni image, larger, 1024d

Each system is self-consistent -- the query and the document are encoded by the same model -- so
the contrast isolates the encoder, not the query representation.

Scale: the omni models are ~1B-param VLMs, 3-5x slower per image than SigLIP, so W7 runs on a
seeded, tier-stratified subset of the full test set rather than all 80k products. The subset is
written once to data/w7_subset.csv and reused by every W7 step so all systems see identical
queries and pools.

Images are encoded one per forward pass. Batching was measured at only 1.17x on MPS while
introducing ~3.7e-3 numerical drift versus the per-sample path (probe_omni_batching.py); for a
study about image-encoder fidelity that trade is not worth taking. Text is batched normally.

Resumable: progress is checkpointed, so re-running continues where it stopped.

Usage:
  python 03c_embed_jina_omni.py --variant nano
  python 03c_embed_jina_omni.py --variant small
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

VARIANTS = {
    "nano": "jinaai/jina-embeddings-v5-omni-nano",
    "small": "jinaai/jina-embeddings-v5-omni-small",
}
SUBSET_CSV = config.DATA_DIR / "w7_subset.csv"
QUERIES_PER_TIER = int(__import__("os").environ.get("W7_QUERIES_PER_TIER", 200))


def build_subset() -> pd.DataFrame:
    """Seeded, tier-stratified query subset; built once then reused verbatim."""
    if SUBSET_CSV.exists():
        sub = pd.read_csv(SUBSET_CSV)
        print(f"subset: reusing {SUBSET_CSV} ({sub['search_term'].nunique()} queries)")
        return sub

    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv")
    rng = np.random.default_rng(config.RANDOM_SEED)
    tiers = test_set.drop_duplicates("search_term")[["search_term", "query_tier"]]
    keep: list[str] = []
    for tier, grp in tiers.groupby("query_tier"):
        n = min(QUERIES_PER_TIER, len(grp))
        keep += list(rng.choice(grp["search_term"].to_numpy(), size=n, replace=False))
    sub = test_set[test_set["search_term"].isin(keep)].copy()
    sub.to_csv(SUBSET_CSV, index=False)
    print(f"subset: built {SUBSET_CSV}")
    print(sub.drop_duplicates("search_term")["query_tier"].value_counts().to_string())
    return sub


def load_progress(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not path.exists():
        return {}, {}
    d = np.load(path, allow_pickle=True)
    q = {str(k): v for k, v in zip(d["queries"], d["query_emb"])} if len(d["queries"]) else {}
    e = {str(k): v for k, v in zip(d["ecodes"], d["image_emb"])} if len(d["ecodes"]) else {}
    return q, e


def save_progress(path: Path, q: dict[str, np.ndarray], e: dict[str, np.ndarray]) -> None:
    dim_q = len(next(iter(q.values()))) if q else 0
    dim_e = len(next(iter(e.values()))) if e else 0
    np.savez_compressed(
        path,
        queries=np.array(list(q), dtype=object),
        query_emb=np.array(list(q.values()), dtype=np.float32).reshape(-1, dim_q),
        ecodes=np.array(list(e), dtype=object),
        image_emb=np.array(list(e.values()), dtype=np.float32).reshape(-1, dim_e),
    )


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--text-batch", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    args = ap.parse_args()

    repo = VARIANTS[args.variant]
    out_path = config.EMB_DIR / f"jina_omni_{args.variant}.npz"
    config.ensure_dirs()

    sub = build_subset()
    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)
    ok_ecodes = set(manifest.loc[manifest["ok"] == True, "ecode"].astype(str))  # noqa: E712
    sub = sub[sub["ecode"].astype(str).isin(ok_ecodes)]

    queries = sorted(sub["search_term"].astype(str).unique())
    ecodes = sorted(sub["ecode"].astype(str).unique())

    q_done, e_done = load_progress(out_path)
    todo_q = [q for q in queries if q not in q_done]
    todo_e = [e for e in ecodes if e not in e_done]
    print(f"{repo}\nqueries {len(queries)} ({len(todo_q)} to do)   products {len(ecodes)} ({len(todo_e)} to do)")
    if not todo_q and not todo_e:
        print(f"complete -> {out_path}")
        return 0

    from transformers import AutoModel, AutoProcessor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading on {device} (modality=vision)")
    model = AutoModel.from_pretrained(
        repo, trust_remote_code=True, default_task="retrieval", modality="vision"
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

    # retrieval prefixes are required on every modality, not just text
    for start in range(0, len(todo_q), args.text_batch):
        batch = todo_q[start : start + args.text_batch]
        inp = proc(
            text=[f"Query: {q}" for q in batch], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            vecs = model.embed(**inp.to(device)).float().cpu().numpy()
        for name, v in zip(batch, l2(vecs)):
            q_done[name] = v
        print(f"  query {min(start + args.text_batch, len(todo_q))}/{len(todo_q)}", end="\r")
    print()

    t0 = time.time()
    for i, ecode in enumerate(todo_e, start=1):
        try:
            im = Image.open(config.IMAGE_DIR / f"{ecode}.jpg").convert("RGB")
            inp = proc(images=im, text="Document: <image>", return_tensors="pt")
            with torch.no_grad():
                v = model.embed(**inp.to(device)).float().cpu().numpy()[0]
            e_done[ecode] = l2(v)
        except Exception as exc:
            print(f"\n  skip {ecode}: {type(exc).__name__}: {exc}")
        if i % 50 == 0 or i == len(todo_e):
            rate = i / (time.time() - t0)
            eta = (len(todo_e) - i) / rate / 60
            print(f"  image {i}/{len(todo_e)}  {rate:.2f}/s  eta {eta:.0f}m", end="\r")
        if i % args.checkpoint_every == 0:
            save_progress(out_path, q_done, e_done)
    print()

    save_progress(out_path, q_done, e_done)
    print(f"saved -> {out_path}  ({len(q_done)} queries, {len(e_done)} products)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
