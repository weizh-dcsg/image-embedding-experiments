#!/usr/bin/env python3
"""Step 20 (W9): pull the two production retrieval signals out of Elasticsearch.

W9 asks what a Jina CLIP v2 image tower adds on top of the hybrid that is already live, so the
baseline arms must come from the live cluster rather than from local re-implementations:

  bm25   Lucene BM25 from `catalog-1`, the production catalog. Scored with the pool as a filter,
         so IDF still comes from the full 306M-doc index -- filtering changes which documents are
         returned, not how they are weighted.
  e5     multilingual-e5-small vectors. Document vectors are read from the production embedding
         indices; query vectors come from the deployed `.multilingual-e5-small_linux-x86_64`.

E5 prefix convention: the production indices embed `passage: {name}` (verified -- cosine 1.0
against the stored vector, versus 0.96 for the unprefixed name, which is close enough to look
correct while being wrong). Queries therefore use the matching `query: ` prefix.

Coverage is deliberately NOT patched up with locally computed vectors. The production embedding
indices cover roughly half the judgement-list catalog, and that gap is part of what the baseline
actually is; it is recorded in results/w9_es_coverage.json and carried into the report.

Both stages are resumable.

Usage:
  python 20_fetch_es_baseline.py --what all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ES_URL = "https://dsg-search-prodauth-east.es.eastus.azure.elastic-cloud.com"
BM25_INDEX = "catalog-1"
# Must be the ANALYSED subfields, not the raw ones. `name` is indexed with the default analyser
# and misses plurals and misspellings entirely -- `hokas`, `salomons` and `birkinstock` all return
# zero hits on `name` but tens of thousands on `name.name-search`, whose search analyser
# (`user_search_analyzer`) carries the synonym and misspelling filters production relies on.
BM25_FIELDS = [
    "name.name-search^3",
    "keyword^2",
    "attributes",
    "longDescription.longDescriptionSynonymsEnabled",
]
# Priority order: first index that has a vector for an ecode wins.
E5_INDICES = ["catalog_embedding_final", "catalog-name-embedding", "e5-vector-index-v2"]
E5_MODEL = ".multilingual-e5-small_linux-x86_64"
E5_DIM = 384

BM25_CSV = config.DATA_DIR / "w9_bm25_catalog1.csv"
E5_NPZ = config.EMB_DIR / "e5_prod.npz"
COVERAGE_JSON = config.RESULTS_DIR / "w9_es_coverage.json"


def client(key_file: Path) -> Elasticsearch:
    return Elasticsearch(
        ES_URL,
        api_key=key_file.read_text(encoding="utf-8").strip(),
        request_timeout=300,
        max_retries=3,
        retry_on_timeout=True,
    )


def fetch_bm25(es: Elasticsearch, test_set: pd.DataFrame, batch: int) -> None:
    pools = {term: grp["ecode"].tolist() for term, grp in test_set.groupby("search_term")}

    done: set[str] = set()
    if BM25_CSV.exists():
        done = set(pd.read_csv(BM25_CSV, usecols=["search_term"])["search_term"].astype(str))
        print(f"  resuming: {len(done)} queries already scored")
    todo = [t for t in sorted(pools) if t not in done]
    if not todo:
        print(f"  bm25 complete -> {BM25_CSV}")
        return

    handle = BM25_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if not done:
        writer.writerow(["search_term", "ecode", "bm25_score"])

    t0 = time.time()
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        body: list[dict] = []
        for term in chunk:
            pool = pools[term]
            body.append({"index": BM25_INDEX})
            body.append(
                {
                    "size": len(pool),
                    "_source": False,
                    "docvalue_fields": ["partnumber"],
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": term,
                                        "fields": BM25_FIELDS,
                                        "type": "best_fields",
                                    }
                                }
                            ],
                            "filter": [{"terms": {"partnumber": pool}}],
                        }
                    },
                }
            )
        responses = es.msearch(searches=body).body["responses"]
        for term, response in zip(chunk, responses):
            if "error" in response:
                print(f"\n  ES error for {term!r}: {str(response['error'])[:160]}")
                continue
            for hit in response["hits"]["hits"]:
                ecode = hit["fields"]["partnumber"][0]
                writer.writerow([term, ecode, hit["_score"]])
        handle.flush()
        done_n = start + len(chunk)
        rate = done_n / (time.time() - t0)
        print(f"  bm25 {done_n}/{len(todo)} queries  {rate:.1f}/s", end="\r", flush=True)
    handle.close()
    print(f"\n  bm25 -> {BM25_CSV}")


def fetch_e5_documents(es: Elasticsearch, ecodes: list[str], batch: int) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    if E5_NPZ.exists():
        cached = np.load(E5_NPZ, allow_pickle=True)
        vectors = {str(k): v for k, v in zip(cached["ecodes"], cached["doc_emb"])}
        print(f"  resuming: {len(vectors)} document vectors cached")

    for index in E5_INDICES:
        missing = [e for e in ecodes if e not in vectors]
        if not missing:
            break
        print(f"  {index}: looking up {len(missing)} missing ecodes")
        found = 0
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            response = es.search(
                index=index,
                size=len(chunk) * 2,
                query={
                    "bool": {
                        "filter": [
                            {"terms": {"partnumber": chunk}},
                            {"exists": {"field": "embedding"}},
                        ]
                    }
                },
                source_includes=["partnumber", "embedding"],
            ).body
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                ecode = str(source["partnumber"])
                if ecode not in vectors:
                    vectors[ecode] = np.asarray(source["embedding"], dtype=np.float32)
                    found += 1
            print(f"    {min(start + batch, len(missing))}/{len(missing)}  found {found}", end="\r", flush=True)
        print()
    return vectors


def fetch_e5_queries(es: Elasticsearch, queries: list[str], batch: int) -> np.ndarray:
    out: list[list[float]] = []
    for start in range(0, len(queries), batch):
        chunk = queries[start : start + batch]
        response = es.ml.infer_trained_model(
            model_id=E5_MODEL,
            docs=[{"text_field": f"query: {q}"} for q in chunk],
            timeout="300s",
        ).body["inference_results"]
        if len(response) != len(chunk):
            raise RuntimeError(f"expected {len(chunk)} vectors, got {len(response)}")
        out.extend(r["predicted_value"] for r in response)
        print(f"  e5 queries {min(start + batch, len(queries))}/{len(queries)}", end="\r", flush=True)
    print()
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    global BM25_CSV, BM25_FIELDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=Path("../../elastic-key.txt"))
    parser.add_argument("--what", choices=["bm25", "e5", "all"], default="all")
    parser.add_argument("--bm25-batch", type=int, default=25)
    parser.add_argument("--bm25-csv", type=Path, default=BM25_CSV)
    parser.add_argument("--bm25-fields", nargs="+", default=BM25_FIELDS)
    parser.add_argument("--e5-doc-batch", type=int, default=200)
    parser.add_argument("--e5-query-batch", type=int, default=32)
    args = parser.parse_args()

    BM25_CSV = args.bm25_csv
    BM25_FIELDS = list(args.bm25_fields)

    config.ensure_dirs()
    test_set = pd.read_csv(config.DATA_DIR / "test_set_encoded.csv", usecols=["search_term", "ecode"])
    test_set["ecode"] = test_set["ecode"].astype(str)
    test_set["search_term"] = test_set["search_term"].astype(str)
    queries = sorted(test_set["search_term"].unique())
    ecodes = sorted(test_set["ecode"].unique())
    print(f"test set: {len(queries)} queries, {len(ecodes)} products")

    es = client(args.key_file)

    if args.what in ("bm25", "all"):
        print("== BM25 from", BM25_INDEX, "fields", BM25_FIELDS)
        fetch_bm25(es, test_set, args.bm25_batch)

    if args.what in ("e5", "all"):
        print("== E5 documents from", E5_INDICES)
        doc_vectors = fetch_e5_documents(es, ecodes, args.e5_doc_batch)
        print("== E5 query vectors from", E5_MODEL)
        query_emb = fetch_e5_queries(es, queries, args.e5_query_batch)

        covered = [e for e in ecodes if e in doc_vectors]
        doc_emb = np.asarray([doc_vectors[e] for e in covered], dtype=np.float32)
        np.savez_compressed(
            E5_NPZ,
            queries=np.asarray(queries, dtype=object),
            query_emb=query_emb,
            ecodes=np.asarray(covered, dtype=object),
            doc_emb=doc_emb,
        )
        coverage = {
            "n_queries": len(queries),
            "n_products_test_set": len(ecodes),
            "n_products_with_e5_vector": len(covered),
            "product_coverage_pct": round(100 * len(covered) / len(ecodes), 2),
            "indices": E5_INDICES,
            "model_id": E5_MODEL,
            "document_prefix": "passage: ",
            "query_prefix": "query: ",
        }
        COVERAGE_JSON.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(coverage, indent=2))
        print(f"  e5 -> {E5_NPZ}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
