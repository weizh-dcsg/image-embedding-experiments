#!/usr/bin/env python3
"""Step 3: encode queries and products with SigLIP (image + text towers) and Jina v5 text nano.

Produces three retrieval systems that share the same query set and candidate pools:
  siglip_image -- query: SigLIP text tower   | product: SigLIP image tower (product photo)
  siglip_text  -- query: SigLIP text tower   | product: SigLIP text tower (product title)
  jina_text    -- query: Jina v5 retrieval.query | product: Jina v5 retrieval.document (title)

Outputs npz files in data/embeddings/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def load_image(path: Path, retries: int = 6):
    """Open an image, retrying transient reads.

    The data directory lives in OneDrive CloudStorage, which dehydrates cold files to cloud
    placeholders; the first read then blocks on a re-download and can time out.
    """
    from PIL import Image

    for attempt in range(retries):
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except (TimeoutError, OSError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"could not read {path} after {retries} attempts: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not read {path}")


def as_vectors(output) -> torch.Tensor:
    """transformers>=5 returns a model output object from get_*_features; <5 returns a tensor."""
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
        value = getattr(output, attr, None)
        if value is not None:
            return value.mean(dim=1) if value.dim() == 3 else value
    raise TypeError(f"cannot extract embeddings from {type(output)}")


def encode_siglip(
    device: torch.device,
    queries: list[str],
    titles: list[str],
    image_paths: list[Path],
    crop_paths: list[Path] | None,
    naive_paths: list[Path] | None,
    batch_size: int,
    attrs: list[str] | None = None,
) -> dict[str, np.ndarray]:
    from transformers import AutoModel, AutoProcessor

    print(f"loading {config.SIGLIP_MODEL} on {device}")
    model = AutoModel.from_pretrained(config.SIGLIP_MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)

    def encode_text(texts: list[str], label: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = processor(
                text=batch, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                feats = as_vectors(model.get_text_features(**inputs))
            out.append(feats.float().cpu().numpy())
            print(f"  siglip {label}: {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return l2_normalize(np.vstack(out))

    def encode_images(paths: list[Path], label: str) -> np.ndarray:
        if not paths:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [load_image(p) for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = as_vectors(model.get_image_features(**inputs))
            out.append(feats.float().cpu().numpy())
            print(f"  siglip {label}: {min(start + batch_size, len(paths))}/{len(paths)}", end="\r")
        print()
        return l2_normalize(np.vstack(out))

    result = {
        "query": encode_text(queries, "query"),
        "title": encode_text(titles, "title"),
        "image": encode_images(image_paths, "image"),
    }
    if attrs is not None:
        result["attr"] = encode_text(attrs, "attr")
    if crop_paths is not None:
        result["image_crop"] = encode_images(crop_paths, "image_crop")
    if naive_paths is not None:
        result["image_naive"] = encode_images(naive_paths, "image_naive")
    return result


def encode_jina(
    device: torch.device, queries: list[str], titles: list[str], batch_size: int,
    attrs: list[str] | None = None
) -> dict[str, np.ndarray]:
    from transformers import AutoModel

    print(f"loading {config.JINA_MODEL} on {device}")
    model = AutoModel.from_pretrained(config.JINA_MODEL, trust_remote_code=True).to(device).eval()

    def encode(texts: list[str], prompt_name: str, label: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            with torch.no_grad():
                feats = model.encode(texts=batch, task="retrieval", prompt_name=prompt_name)
            if isinstance(feats, torch.Tensor):
                feats = feats.float().cpu().numpy()
            out.append(np.asarray(feats, dtype=np.float32))
            print(f"  jina {label}: {min(start + batch_size, len(texts))}/{len(texts)}", end="\r")
        print()
        return l2_normalize(np.vstack(out))

    return {
        "query": encode(queries, "query", "query"),
        "title": encode(titles, "document", "title"),
        **({"attr": encode(attrs, "document", "attr")} if attrs is not None else {}),
    }


def is_cached(path: Path, queries: list[str], ecodes: list[str], required: tuple[str, ...] = ()) -> bool:
    """A cache covering a superset of what we need is reusable.

    Embeddings depend only on the query string / product title / image, never on the test set, and
    every consumer (04_evaluate, 09_fusion_experiment) looks vectors up by name rather than by
    position. So a relabelling that drops some queries does not require re-encoding the rest.
    """
    if not path.exists():
        return False
    cached = np.load(path, allow_pickle=True)
    if any(key not in cached.files for key in required):
        return False
    have_q = {str(q) for q in cached["queries"]}
    have_e = {str(e) for e in cached["ecodes"]}
    return have_q.issuperset(queries) and have_e.issuperset(ecodes)


def load_cache(path: Path, required: tuple[str, ...]) -> dict | None:
    """Cached arrays keyed by name, or None if unusable for an incremental top-up."""
    if not path.exists():
        return None
    cached = np.load(path, allow_pickle=True)
    if any(key not in cached.files for key in required):
        return None
    return {
        "q_idx": {str(q): i for i, q in enumerate(cached["queries"])},
        "e_idx": {str(e): i for i, e in enumerate(cached["ecodes"])},
        "arrays": {k: cached[k] for k in cached.files if k not in ("queries", "ecodes")},
    }


def missing_items(cache: dict | None, queries: list[str], ecodes: list[str]) -> tuple[list[str], list[str]]:
    if cache is None:
        return list(queries), list(ecodes)
    return (
        [q for q in queries if q not in cache["q_idx"]],
        [e for e in ecodes if e not in cache["e_idx"]],
    )


def assemble(
    cache: dict | None,
    field: str,
    keys_wanted: list[str],
    keys_encoded: list[str],
    encoded: np.ndarray | None,
    idx_name: str,
) -> np.ndarray:
    """Rows for keys_wanted, taken from cache where present and from `encoded` otherwise."""
    new_idx = {k: i for i, k in enumerate(keys_encoded)}
    cached_arr = cache["arrays"][field] if cache is not None and field in cache["arrays"] else None
    width = (encoded if encoded is not None and len(encoded) else cached_arr).shape[1]
    out = np.empty((len(keys_wanted), width), dtype=np.float32)
    for row, key in enumerate(keys_wanted):
        if key in new_idx:
            out[row] = encoded[new_idx[key]]
        else:
            out[row] = cached_arr[cache[idx_name][key]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="re-encode even if cached")
    args = parser.parse_args()

    config.ensure_dirs()
    test_set = pd.read_csv(config.TEST_SET_CSV)
    manifest = pd.read_csv(config.IMAGE_MANIFEST_CSV)

    ok_ecodes = set(manifest.loc[manifest["ok"] == True, "ecode"])  # noqa: E712
    test_set = test_set[test_set["ecode"].isin(ok_ecodes)]

    # a query is only comparable across systems if it still has a positive after image filtering
    keep = test_set.groupby("search_term")["relevance"].max()
    test_set = test_set[test_set["search_term"].isin(keep[keep > 0].index)]
    test_set.to_csv(config.DATA_DIR / "test_set_encoded.csv", index=False)

    queries = sorted(test_set["search_term"].unique())
    products = (
        test_set[["ecode", "product_title"]].drop_duplicates(subset="ecode").sort_values("ecode")
    )
    ecodes = products["ecode"].tolist()
    titles = products["product_title"].astype(str).tolist()
    attr_texts: list[str] | None = None
    if config.ATTRIBUTES_CSV.exists():
        attr_map = (
            pd.read_csv(config.ATTRIBUTES_CSV)
            .assign(attr_text=lambda d: d["attr_text"].fillna("").astype(str))
            .set_index("ecode")["attr_text"]
            .to_dict()
        )
        # Fall back to the title so an ecode missing attributes still yields a comparable vector
        # rather than an empty string, which SigLIP maps to a meaningless constant.
        attr_texts = [attr_map.get(e) or t for e, t in zip(ecodes, titles)]
        n_real = sum(1 for e in ecodes if attr_map.get(e))
        print(f"attribute texts: {n_real}/{len(ecodes)} from Big-4, rest fall back to title")
    image_paths = [config.IMAGE_DIR / f"{e}.jpg" for e in ecodes]
    crop_paths = [config.CROP_DIR / f"{e}.jpg" for e in ecodes]
    naive_paths = [config.CROP_NAIVE_DIR / f"{e}.jpg" for e in ecodes]
    has_crops = all(p.exists() for p in crop_paths)
    has_naive = all(p.exists() for p in naive_paths)

    print(f"queries: {len(queries)}   products: {len(ecodes)}   crops: {has_crops}   naive: {has_naive}")

    device = pick_device(args.device)
    siglip_path = config.EMB_DIR / "siglip.npz"
    jina_path = config.EMB_DIR / "jina.npz"
    required = tuple(
        k
        for k, present in (
            ("image_crop_emb", has_crops),
            ("image_naive_emb", has_naive),
            ("attr_emb", attr_texts is not None),
        )
        if present
    )

    if not args.force and is_cached(siglip_path, queries, ecodes, required):
        print(f"cached  -> {siglip_path}")
    else:
        cache = None if args.force else load_cache(siglip_path, required)
        new_q, new_e = missing_items(cache, queries, ecodes)
        print(f"siglip: encoding {len(new_q)}/{len(queries)} queries, {len(new_e)}/{len(ecodes)} products")
        e_pos = {e: i for i, e in enumerate(ecodes)}
        rows = [e_pos[e] for e in new_e]
        siglip = encode_siglip(
            device,
            new_q,
            [titles[r] for r in rows],
            [image_paths[r] for r in rows],
            [crop_paths[r] for r in rows] if has_crops else None,
            [naive_paths[r] for r in rows] if has_naive else None,
            args.batch_size,
            attrs=[attr_texts[r] for r in rows] if attr_texts is not None else None,
        )
        payload = {
            "queries": np.array(queries, dtype=object),
            "ecodes": np.array(ecodes, dtype=object),
            "query_emb": assemble(cache, "query_emb", queries, new_q, siglip["query"], "q_idx"),
            "title_emb": assemble(cache, "title_emb", ecodes, new_e, siglip["title"], "e_idx"),
            "image_emb": assemble(cache, "image_emb", ecodes, new_e, siglip["image"], "e_idx"),
        }
        if attr_texts is not None:
            payload["attr_emb"] = assemble(cache, "attr_emb", ecodes, new_e, siglip["attr"], "e_idx")
        if has_crops:
            payload["image_crop_emb"] = assemble(cache, "image_crop_emb", ecodes, new_e, siglip["image_crop"], "e_idx")
        if has_naive:
            payload["image_naive_emb"] = assemble(cache, "image_naive_emb", ecodes, new_e, siglip["image_naive"], "e_idx")
        np.savez_compressed(siglip_path, **payload)
        print(f"saved -> {siglip_path}")

    jina_required = ("attr_emb",) if attr_texts else ()
    if not args.force and is_cached(jina_path, queries, ecodes, jina_required):
        print(f"cached  -> {jina_path}")
    else:
        cache = None if args.force else load_cache(jina_path, jina_required)
        new_q, new_e = missing_items(cache, queries, ecodes)
        print(f"jina: encoding {len(new_q)}/{len(queries)} queries, {len(new_e)}/{len(ecodes)} products")
        e_pos = {e: i for i, e in enumerate(ecodes)}
        rows = [e_pos[e] for e in new_e]
        jina = encode_jina(
            device,
            new_q,
            [titles[r] for r in rows],
            args.batch_size,
            attrs=[attr_texts[r] for r in rows] if attr_texts is not None else None,
        )
        payload = {
            "queries": np.array(queries, dtype=object),
            "ecodes": np.array(ecodes, dtype=object),
            "query_emb": assemble(cache, "query_emb", queries, new_q, jina["query"], "q_idx"),
            "title_emb": assemble(cache, "title_emb", ecodes, new_e, jina["title"], "e_idx"),
        }
        if attr_texts is not None:
            payload["attr_emb"] = assemble(cache, "attr_emb", ecodes, new_e, jina["attr"], "e_idx")
        np.savez_compressed(jina_path, **payload)
        print(f"saved -> {jina_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
