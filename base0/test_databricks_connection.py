#!/usr/bin/env python3
"""Pull the top N rows from prod_ml_feature_store_db.products.{ecode,ecode_attribute}
via the Databricks SQL Statement Execution API, using the CLI bundled with the
VS Code Databricks extension.

Usage: ./test_databricks_connection.py [row_limit]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CLI = Path.home() / ".vscode/extensions/databricks.databricks-2.13.0-darwin-arm64/bin/databricks"
CATALOG = "prod_ml_feature_store_db"
SCHEMA = "products"
TABLES = ["ecode", "ecode_attribute"]
OUT_DIR = Path(__file__).resolve().parent / "output"


def run_query(cli: Path, profile: str, warehouse_id: str, table: str, limit: int) -> bool:
    out_file = OUT_DIR / f"{table}_top{limit}.json"
    print(f"==> Querying {CATALOG}.{SCHEMA}.{table} (LIMIT {limit})")

    payload = json.dumps(
        {
            "warehouse_id": warehouse_id,
            "statement": f"SELECT * FROM {CATALOG}.{SCHEMA}.{table} LIMIT {limit}",
            "wait_timeout": "30s",
        }
    )

    completed = subprocess.run(
        [str(cli), "api", "post", "/api/2.0/sql/statements", "--profile", profile, "--json", payload],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"  CLI call failed: {completed.stderr.strip()}")
        return False

    resp = json.loads(completed.stdout)
    out_file.write_text(json.dumps(resp, indent=2))

    state = resp.get("status", {}).get("state", "UNKNOWN")
    if state != "SUCCEEDED":
        print(f"  FAILED: {json.dumps(resp.get('status', resp))}")
        return False

    columns = [c["name"] for c in resp["manifest"]["schema"]["columns"]]
    rows = resp.get("result", {}).get("data_array") or []
    print(f"  columns ({len(columns)}): {', '.join(columns)}")
    print(f"  rows returned: {len(rows)}")
    for row in rows[:3]:
        print(f"    {json.dumps(dict(zip(columns, row)))[:200]}")
    print(f"  saved -> {out_file}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row_limit", nargs="?", type=int, default=10)
    args = parser.parse_args()

    if args.row_limit <= 0:
        parser.error("row_limit must be a positive integer")

    cli = Path(os.environ.get("DATABRICKS_CLI", DEFAULT_CLI))
    profile = os.environ.get("DATABRICKS_PROFILE", "test-databricks-config")
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "bbf43c4818c2391a")

    if not os.access(cli, os.X_OK):
        print(f"Databricks CLI not found at: {cli}", file=sys.stderr)
        print("Set DATABRICKS_CLI to the correct path and retry.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Profile:   {profile}")
    print(f"Warehouse: {warehouse_id}\n")

    ok = True
    for table in TABLES:
        ok &= run_query(cli, profile, warehouse_id, table, args.row_limit)
        print()

    print("Done." if ok else "Completed with errors.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
