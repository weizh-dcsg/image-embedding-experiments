#!/usr/bin/env python3
"""Sanity-check connectivity to the dsg-search-prodauth-east Elastic Cloud deployment
and run a small test query against an index.

Auth is never hardcoded here -- export one of:
  ELASTIC_API_KEY        (preferred; "id:api_key" or base64-encoded form from Kibana)
  ELASTIC_PASSWORD        (used with ELASTIC_USERNAME, default user "elastic")

Usage:
  export ELASTIC_API_KEY="..."
  ./test_elastic_connection.py [index_name] [row_limit]
"""

from __future__ import annotations

import os
import sys

from elasticsearch import Elasticsearch

CLOUD_ID = "dsg-search-prodauth-east:ZWFzdHVzLmF6dXJlLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQwMDU3MjY4Y2I3NzM0OGVhOGY5ODY1OGY0N2I1NGJmOSQxN2M0Mzg1NjUwOGY0MWZkOTQ3MzA5MmExNTFmNDIwNw=="
ENDPOINT = "https://dsg-search-prodauth-east.es.eastus.azure.elastic-cloud.com"


def build_client() -> Elasticsearch:
    api_key = os.environ.get("ELASTIC_API_KEY")
    username = os.environ.get("ELASTIC_USERNAME", "elastic")
    password = os.environ.get("ELASTIC_PASSWORD")

    if not api_key and not password:
        sys.exit("Set ELASTIC_API_KEY, or ELASTIC_USERNAME/ELASTIC_PASSWORD, before running.")

    if api_key:
        return Elasticsearch(cloud_id=CLOUD_ID, api_key=api_key)
    return Elasticsearch(cloud_id=CLOUD_ID, basic_auth=(username, password))


def main() -> None:
    index = sys.argv[1] if len(sys.argv) > 1 else "*"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    es = build_client()

    # API keys are often scoped to specific indices only, so skip cluster-wide
    # calls (info/health/cat.indices need monitor/manage privileges) and go
    # straight to a search, which is what an index-scoped key is meant for.
    print(f"==> Sample search on {index!r} (size={limit})")
    resp = es.search(index=index, query={"match_all": {}}, size=limit)
    hits = resp["hits"]["hits"]
    print(f"  total hits: {resp['hits']['total']['value']}")
    for hit in hits:
        print(f"    id={hit['_id']}  index={hit['_index']}  source={str(hit['_source'])[:200]}")


if __name__ == "__main__":
    main()
