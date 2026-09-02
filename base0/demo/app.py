#!/usr/bin/env python3
"""Side-by-side demo: the live BM25 + E5 hybrid, with and without an image tower.

Left column is the baseline that runs in production today -- Lucene BM25 over `catalog-1`
fused with multilingual-e5-small by reciprocal rank fusion. Right column is the same fusion
with a product-image tower added as a third arm. Same query, same candidate universe, same
RRF constant, so the only difference between the columns is the image signal.

Fusion matches 21_w9_hybrid_experiment.py exactly: RRF at k=60 over per-arm ranked lists, with
a rank window per arm. An arm that cannot score a product contributes nothing for it.

Where each signal comes from:
  bm25    live `catalog-1` search, the production index
  e5      query vector from the deployed `.multilingual-e5-small_linux-x86_64` (prefixed
          `query: `, matching how the index was built); document vectors are the ones pulled
          out of the production embedding indices by 20_fetch_es_baseline.py
  image   query vector from a deployed image-tower text encoder; document vectors are the
          local encodes of the product photography

Catalog universe is the 80,608 products of the W9 judgement list, because that is the set for
which image vectors exist.

Usage:
  python app.py                 # http://127.0.0.1:5000
  python app.py --port 8080

Access is behind HTTP Basic Auth. Set DEMO_USER / DEMO_PASSWORD in the environment; if no
password is set one is generated and printed once at startup. Passwords are deliberately not
accepted as command-line flags, which would leak them into shell history and `ps` output.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embedding_eval"))
import config  # noqa: E402

from local_backend import LocalBM25, LocalEncoders  # noqa: E402

ES_URL = "https://dsg-search-prodauth-east.es.eastus.azure.elastic-cloud.com"
BM25_INDEX = "catalog-1"
# Analysed subfields, not the raw ones: `name` uses the default analyser and misses plurals and
# misspellings entirely (`hokas` -> 0 hits), while `name.name-search` searches through
# `user_search_analyzer`, which carries production's synonym and misspelling filters.
BM25_FIELDS = [
    "name.name-search^3",
    "keyword^2",
    "attributes",
    "longDescription.longDescriptionSynonymsEnabled",
]
E5_MODEL = ".multilingual-e5-small_linux-x86_64"

# Both towers are deployed on the cluster, so the demo shows real query-time inference rather
# than a local re-encode. W9 found SigLIP the stronger tower, so it is the default.
# Weights are (bm25, e5, image) for the z-score fusion, grid-searched on half the W9 queries and
# validated on the other half by 24_w9_merge_study.py.
IMAGE_TOWERS = {
    "jina": {
        "label": "Jina CLIP v2",
        "es_model": "jina-clip-v2-text",
        "npz": "jina_clip_v2_full.npz",
        "field": "image_emb",
        "weights": (0.40, 0.10, 0.50),
        "holdout_ndcg10": 0.3984,
    },
    "siglip": {
        "label": "SigLIP",
        "es_model": "siglip-base-patch16-512-text-v2",
        "npz": "siglip.npz",
        "field": "image_emb",
        "weights": (0.15, 0.10, 0.75),
        "holdout_ndcg10": 0.4248,
    },
}

RRF_K = 60
RANK_WINDOW = 200
ROOT = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False  # keep the Big-4 attributes in their intended display order
STATE: dict = {}
AUTH: dict = {}


@app.before_request
def require_auth():
    """Basic auth on every route, compared in constant time."""
    given = request.authorization
    ok = (
        given is not None
        and given.type == "basic"
        and secrets.compare_digest(given.username or "", AUTH["user"])
        and secrets.compare_digest(given.password or "", AUTH["password"])
    )
    if not ok:
        return Response(
            "Authentication required.",
            401,
            {"WWW-Authenticate": 'Basic realm="image-enhanced retrieval demo"'},
        )
    return None


def l2_normalise(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def es_call(fn, *args, attempts: int = 4, **kwargs):
    """Retry Elastic calls through the cluster's intermittent traffic-filter 403s.

    The deployment intermittently rejects a request with 403 "Forbidden due to traffic
    filtering" and serves the identical request moments later. The client does not treat 403 as
    retryable, so it is handled here rather than surfacing as a failed search.
    """
    delay = 0.4
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (es_exceptions.AuthorizationException, es_exceptions.ConnectionError) as exc:
            if attempt == attempts:
                raise
            print(f"  ES {type(exc).__name__}, retry {attempt}/{attempts - 1}")
            time.sleep(delay)
            delay *= 2


def load_state(key_file: Path, mode: str) -> None:
    products = pd.read_csv(config.DATA_DIR / "products.csv")
    products["ecode"] = products["ecode"].astype(str)
    attrs = pd.read_csv(config.DATA_DIR / "product_attributes.csv")
    attrs["ecode"] = attrs["ecode"].astype(str)
    meta = products.merge(
        attrs[["ecode", "brand", "product_type", "product_activity", "gender_by_age"]],
        on="ecode",
        how="left",
    )
    STATE["meta"] = meta.set_index("ecode").to_dict("index")

    e5 = np.load(config.EMB_DIR / "e5_prod.npz", allow_pickle=True)
    STATE["e5_ecodes"] = np.array([str(e) for e in e5["ecodes"]], dtype=object)
    STATE["e5_docs"] = l2_normalise(e5["doc_emb"])
    STATE["e5_row"] = {e: i for i, e in enumerate(STATE["e5_ecodes"])}

    STATE["towers"] = {}
    for name, spec in IMAGE_TOWERS.items():
        path = config.EMB_DIR / spec["npz"]
        if not path.exists():
            print(f"  image tower {name}: {path} not found, skipping")
            continue
        data = np.load(path, allow_pickle=True)
        ecodes = np.array([str(e) for e in data["ecodes"]], dtype=object)
        STATE["towers"][name] = {
            "label": spec["label"],
            "es_model": spec["es_model"],
            "weights": spec["weights"],
            "holdout_ndcg10": spec["holdout_ndcg10"],
            "ecodes": ecodes,
            "docs": l2_normalise(data[spec["field"]]),
            "row": {e: i for i, e in enumerate(ecodes)},
        }
        print(f"  image tower {name}: {len(ecodes):,} products")

    STATE["universe"] = set(STATE["meta"])
    STATE["es"] = Elasticsearch(
        ES_URL,
        api_key=key_file.read_text(encoding="utf-8").strip(),
        request_timeout=60,
        max_retries=1,
    )
    print(f"  catalog: {len(STATE['universe']):,} products, e5 vectors {len(STATE['e5_ecodes']):,}")

    STATE["mode"] = mode
    if mode == "auto":
        try:
            STATE["es"].search(index=BM25_INDEX, size=1, source=False, query={"match_all": {}})
            STATE["mode"] = "elastic"
        except Exception as exc:
            print(f"  cluster unreachable ({type(exc).__name__}); falling back to local backend")
            STATE["mode"] = "local"

    if STATE["mode"] == "local":
        text = (
            meta["product_title"].fillna("").astype(str)
            + " "
            + meta["brand"].fillna("").astype(str)
            + " "
            + meta["product_type"].fillna("").astype(str)
            + " "
            + meta["product_activity"].fillna("").astype(str)
            + " "
            + meta["gender_by_age"].fillna("").astype(str)
        )
        print("  building local BM25 index...")
        STATE["bm25"] = LocalBM25(meta["ecode"].tolist(), text.tolist())
        STATE["encoders"] = LocalEncoders()
    print(f"  mode: {STATE['mode']}")


def embed_query(kind: str, model_id: str, text: str) -> np.ndarray:
    if STATE["mode"] == "local":
        return STATE["encoders"].encode(kind, text)
    prefixed = f"query: {text}" if kind == "e5" else text
    response = es_call(
        STATE["es"].ml.infer_trained_model,
        model_id=model_id,
        docs=[{"text_field": prefixed}],
        timeout="60s",
    ).body
    vector = np.asarray(response["inference_results"][0]["predicted_value"], dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def bm25_ranked(query: str, window: int) -> dict[str, float]:
    """Top in-universe BM25 matches as {ecode: score}, best first (dict preserves order)."""
    if STATE["mode"] == "local":
        return STATE["bm25"].search_scored(query, window)
    response = es_call(
        STATE["es"].search,
        index=BM25_INDEX,
        size=window * 3,  # over-fetch: most catalog-1 hits fall outside the demo universe
        source=False,
        docvalue_fields=["partnumber"],
        query={"multi_match": {"query": query, "fields": BM25_FIELDS, "type": "best_fields"}},
    ).body
    out: dict[str, float] = {}
    for hit in response["hits"]["hits"]:
        ecode = hit["fields"]["partnumber"][0]
        if ecode in STATE["universe"] and ecode not in out:
            out[ecode] = float(hit["_score"])
        if len(out) >= window:
            break
    return out


def vector_ranked(query_vec: np.ndarray, ecodes: np.ndarray, docs: np.ndarray, window: int) -> list[str]:
    scores = docs @ query_vec
    top = np.argpartition(-scores, min(window, len(scores) - 1))[:window]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [ecodes[i] for i in top]


def vector_scores(query_vec: np.ndarray, row: dict, docs: np.ndarray, candidates: list[str]) -> dict[str, float]:
    idx = [(c, row[c]) for c in candidates if c in row]
    if not idx:
        return {}
    rows = np.fromiter((i for _, i in idx), dtype=np.int64, count=len(idx))
    vals = docs[rows] @ query_vec
    return {c: float(v) for (c, _), v in zip(idx, vals)}


def zscore_over(candidates: list[str], scores: dict[str, float]) -> dict[str, float]:
    """Z-score within the candidate set; candidates this arm cannot score take its floor."""
    vals = np.array([scores[c] for c in candidates if c in scores], dtype=float)
    if len(vals) == 0:
        return {c: 0.0 for c in candidates}
    mu, sd = vals.mean(), vals.std()
    z = {c: ((scores[c] - mu) / sd if sd > 1e-9 else 0.0) for c in candidates if c in scores}
    floor = min(z.values()) if z else 0.0
    return {c: z.get(c, floor) for c in candidates}


def rrf(lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, ecode in enumerate(ranked, start=1):
            scores[ecode] = scores.get(ecode, 0.0) + 1.0 / (k + rank)
    return scores


def present(ecode: str, rank: int, baseline_rank: dict[str, int]) -> dict:
    m = STATE["meta"].get(ecode, {})
    prior = baseline_rank.get(ecode)
    return {
        "ecode": ecode,
        "rank": rank,
        "title": m.get("product_title") or ecode,
        "image_url": m.get("image_url") or "",
        "attributes": {
            "Brand": m.get("brand") or m.get("brand_name") or "-",
            "Product type": m.get("product_type") or "-",
            "Activity": m.get("product_activity") or "-",
            "Gender / age": m.get("gender_by_age") or "-",
        },
        "baseline_rank": prior,
        "delta": None if prior is None else prior - rank,
    }


@app.get("/")
def index():
    return send_file(ROOT / "static" / "index.html")


@app.get("/static/<path:name>")
def static_files(name: str):
    return send_from_directory(ROOT / "static", name)


@app.get("/image/<ecode>")
def image(ecode: str):
    """Serve the locally cached product photo; the scene7 URL is the client-side fallback."""
    path = config.IMAGE_DIR / f"{ecode}.jpg"
    if path.exists():
        return send_file(path, mimetype="image/jpeg")
    return ("", 404)


@app.get("/api/towers")
def towers():
    return jsonify(
        {
            "mode": STATE["mode"],
            "towers": [
                {"id": k, "label": v["label"], "model": v["es_model"]}
                for k, v in STATE["towers"].items()
            ],
        }
    )


@app.get("/api/search")
def search():
    query = (request.args.get("q") or "").strip()
    size = int(request.args.get("size", 24))
    tower_id = request.args.get("tower") or next(iter(STATE["towers"]), "")
    if not query:
        return jsonify({"error": "empty query"}), 400
    tower = STATE["towers"].get(tower_id)
    if tower is None:
        return jsonify({"error": f"unknown image tower {tower_id!r}"}), 400

    try:
        bm25 = bm25_ranked(query, RANK_WINDOW)
        e5_vec = embed_query("e5", E5_MODEL, query)
        img_vec = embed_query(tower_id, tower["es_model"], query)
        e5_list = vector_ranked(e5_vec, STATE["e5_ecodes"], STATE["e5_docs"], RANK_WINDOW)
        img_list = vector_ranked(img_vec, tower["ecodes"], tower["docs"], RANK_WINDOW)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {str(exc)[:200]}"}), 502

    # Left column: production today -- uniform RRF over the two text arms.
    baseline_scores = rrf([list(bm25), e5_list])
    baseline = sorted(baseline_scores, key=lambda e: -baseline_scores[e])[:size]

    # Right column: tuned weighted z-score fusion over the union of all three arms' top-N.
    candidates = list(dict.fromkeys([*bm25, *e5_list, *img_list]))
    z_bm25 = zscore_over(candidates, bm25)
    z_e5 = zscore_over(candidates, vector_scores(e5_vec, STATE["e5_row"], STATE["e5_docs"], candidates))
    z_img = zscore_over(candidates, vector_scores(img_vec, tower["row"], tower["docs"], candidates))
    w_bm25, w_e5, w_img = tower["weights"]
    fused = {c: w_bm25 * z_bm25[c] + w_e5 * z_e5[c] + w_img * z_img[c] for c in candidates}
    enhanced = sorted(candidates, key=lambda e: -fused[e])[:size]

    baseline_rank = {e: i + 1 for i, e in enumerate(baseline)}
    return jsonify(
        {
            "query": query,
            "mode": STATE["mode"],
            "tower": {
                "id": tower_id,
                "label": tower["label"],
                "model": tower["es_model"],
                "weights": list(tower["weights"]),
                "holdout_ndcg10": tower["holdout_ndcg10"],
            },
            "baseline": [present(e, i + 1, {}) for i, e in enumerate(baseline)],
            "enhanced": [present(e, i + 1, baseline_rank) for i, e in enumerate(enhanced)],
            "stats": {
                "bm25_hits": len(bm25),
                "candidates": len(candidates),
                "unchanged": len(set(baseline) & set(enhanced)),
                "introduced": len(set(enhanced) - set(baseline)),
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=Path(__file__).resolve().parents[2] / "elastic-key.txt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--mode",
        choices=["auto", "elastic", "local"],
        default="auto",
        help="auto probes the cluster and falls back to the local backend if it is unreachable",
    )
    args = parser.parse_args()

    AUTH["user"] = os.environ.get("DEMO_USER", "demo")
    supplied = os.environ.get("DEMO_PASSWORD")
    AUTH["password"] = supplied or secrets.token_urlsafe(12)

    print("loading catalog and embeddings...")
    load_state(args.key_file, args.mode)
    print(f"ready -> http://{args.host}:{args.port}")
    if supplied:
        print(f"  auth: user {AUTH['user']!r}, password from DEMO_PASSWORD")
    else:
        print(f"  auth: user {AUTH['user']!r}, generated password {AUTH['password']}")
        print("        (set DEMO_PASSWORD to choose your own)")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
