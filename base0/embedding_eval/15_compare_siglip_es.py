#!/usr/bin/env python3
"""Compare local SigLIP queries with Elasticsearch and collect W7 ES queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

import config

ES_URL = "https://dsg-search-prodauth-east.es.eastus.azure.elastic-cloud.com"
ES_MODEL_ID = "siglip-base-patch16-512-text-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=Path("elastic-key.txt"))
    parser.add_argument("--es-url", default=ES_URL)
    parser.add_argument("--model-id", default=ES_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=config.EMB_DIR / "siglip_es_w7.npz")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=config.RESULTS_DIR / "siglip_es_comparison.json",
    )
    args = parser.parse_args()

    test_set = pd.read_csv(config.DATA_DIR / "w7_subset.csv")
    queries = sorted(test_set["search_term"].astype(str).unique())
    local = np.load(config.EMB_DIR / "siglip.npz", allow_pickle=True)
    local_query_names = local["queries"]
    local_query_vectors = local["query_emb"]
    local_queries = {
        str(query): local_query_vectors[index]
        for index, query in enumerate(local_query_names)
    }
    local.close()
    missing = [query for query in queries if query not in local_queries]
    if missing:
        raise RuntimeError(f"{len(missing)} W7 queries are missing from local siglip.npz")

    key = args.key_file.read_text(encoding="utf-8").strip()
    client = Elasticsearch(args.es_url, api_key=key, request_timeout=120, max_retries=0)
    es_vectors = []
    for start in range(0, len(queries), args.batch_size):
        batch_queries = queries[start : start + args.batch_size]
        response = client.ml.infer_trained_model(
            model_id=args.model_id,
            docs=[{"text_field": query} for query in batch_queries],
        ).body
        results = response["inference_results"]
        if len(results) != len(batch_queries):
            raise RuntimeError(
                f"Elasticsearch returned {len(results)} vectors for {len(batch_queries)} queries"
            )
        es_vectors.extend(result["predicted_value"] for result in results)
        print(
            f"  deployed queries: {min(start + args.batch_size, len(queries))}/{len(queries)}",
            end="\r",
            flush=True,
        )
    print()
    es_vectors = np.asarray(es_vectors, dtype=np.float32)
    local_vectors = np.asarray([local_queries[query] for query in queries], dtype=np.float32)
    if es_vectors.shape != local_vectors.shape:
        raise RuntimeError(f"shape mismatch: local={local_vectors.shape}, ES={es_vectors.shape}")

    absolute_error = np.abs(es_vectors - local_vectors)
    denominator = np.abs(local_vectors)
    nonzero = denominator > 1e-8
    relative_errors = absolute_error[nonzero] / denominator[nonzero]
    local_norms = np.linalg.norm(local_vectors, axis=1)
    vector_l2_relative = np.linalg.norm(es_vectors - local_vectors, axis=1) / local_norms
    cosine = np.sum(es_vectors * local_vectors, axis=1) / (
        np.linalg.norm(es_vectors, axis=1) * local_norms
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, queries=np.asarray(queries, dtype=object), query_emb=es_vectors)
    summary = {
        "model_id": args.model_id,
        "n_queries": len(queries),
        "dimensions": int(es_vectors.shape[1]),
        "mean_absolute_error": float(absolute_error.mean()),
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_percentage_of_local_value": float(relative_errors.mean() * 100),
        "global_absolute_error_percentage_of_local_magnitude": float(
            absolute_error.sum() / denominator.sum() * 100
        ),
        "mean_vector_relative_l2_error_percent": float(vector_l2_relative.mean() * 100),
        "mean_cosine_similarity": float(cosine.mean()),
        "min_cosine_similarity": float(cosine.min()),
        "max_cosine_similarity": float(cosine.max()),
        "output": str(args.output),
    }
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
