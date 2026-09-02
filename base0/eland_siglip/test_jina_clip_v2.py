#!/usr/bin/env python3
"""Smoke-test the deployed Jina CLIP v2 text embedding model."""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

from elasticsearch import Elasticsearch

DEFAULT_ES_URL = "https://dsg-search-prodauth-east.es.eastus.azure.elastic-cloud.com"
DEFAULT_MODEL_ID = "jina-clip-v2-text"
DEFAULT_TEXT = "running shoes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("--key-file", type=Path, default=Path("elastic-key.txt"))
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    api_key = args.key_file.read_text(encoding="utf-8").strip()
    client = Elasticsearch(
        args.es_url,
        api_key=api_key,
        request_timeout=120,
        max_retries=0,
    )

    response = client.ml.infer_trained_model(
        model_id=args.model_id,
        docs=[{"text_field": args.text}],
        # cold start after deployment can exceed the default 10s inference timeout
        timeout="120s",
    ).body
    vector = response["inference_results"][0]["predicted_value"]
    norm = sqrt(sum(value * value for value in vector))

    stats = client.ml.get_trained_models_stats(model_id=args.model_id).body[
        "trained_model_stats"
    ][0]
    deployment = stats.get("deployment_stats", {})

    print(f"model: {args.model_id}")
    print(f"text: {args.text}")
    print(f"vector dimensions: {len(vector)}")
    print(f"vector L2 norm: {norm:.7f}")
    print(f"inference count: {stats['inference_stats']['inference_count']}")
    print(f"deployment state: {deployment.get('state', 'unknown')}")


if __name__ == "__main__":
    main()
