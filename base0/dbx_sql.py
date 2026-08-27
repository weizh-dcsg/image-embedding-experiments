"""Thin client for the Databricks SQL Statement Execution API.

Uses the `databricks` CLI bundled with the VS Code Databricks extension so it
inherits the editor's OAuth session (no tokens stored in code).

Can also be run directly for ad-hoc queries:
    python dbx_sql.py "SELECT 1"
    python dbx_sql.py --file query.sql --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator

DEFAULT_CLI = Path.home() / ".vscode/extensions/databricks.databricks-2.13.0-darwin-arm64/bin/databricks"
DEFAULT_PROFILE = "test-databricks-config"
DEFAULT_WAREHOUSE = "bbf43c4818c2391a"  # datasci-med-serverless


class DatabricksSQLError(RuntimeError):
    pass


class DatabricksSQL:
    def __init__(
        self,
        cli: str | Path | None = None,
        profile: str | None = None,
        warehouse_id: str | None = None,
        wait_timeout: str = "50s",
    ) -> None:
        self.cli = Path(cli or os.environ.get("DATABRICKS_CLI", DEFAULT_CLI))
        self.profile = profile or os.environ.get("DATABRICKS_PROFILE", DEFAULT_PROFILE)
        self.warehouse_id = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID", DEFAULT_WAREHOUSE)
        self.wait_timeout = wait_timeout

        if not os.access(self.cli, os.X_OK):
            raise DatabricksSQLError(
                f"Databricks CLI not found/executable at {self.cli}. Set DATABRICKS_CLI."
            )

    def _api(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        cmd = [str(self.cli), "api", method, path, "--profile", self.profile]
        if payload is not None:
            cmd += ["--json", json.dumps(payload)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise DatabricksSQLError(f"CLI {method} {path} failed: {proc.stderr.strip()}")
        return json.loads(proc.stdout or "{}")

    def query(self, statement: str, row_limit: int | None = None) -> tuple[list[str], list[list[Any]]]:
        """Run a statement and return (column_names, rows). Fetches all chunks.

        Starts INLINE and transparently re-runs with EXTERNAL_LINKS when the result
        exceeds the 25 MiB inline cap.
        """
        try:
            return self._query(statement, row_limit, disposition="INLINE")
        except DatabricksSQLError as exc:
            if "Inline byte limit exceeded" not in str(exc):
                raise
            return self._query(statement, row_limit, disposition="EXTERNAL_LINKS")

    def _query(
        self, statement: str, row_limit: int | None, disposition: str
    ) -> tuple[list[str], list[list[Any]]]:
        payload: dict[str, Any] = {
            "warehouse_id": self.warehouse_id,
            "statement": statement,
            "wait_timeout": self.wait_timeout,
            "format": "JSON_ARRAY",
            "disposition": disposition,
        }
        if row_limit is not None:
            payload["row_limit"] = row_limit

        resp = self._api("post", "/api/2.0/sql/statements", payload)
        statement_id = resp.get("statement_id")

        while resp.get("status", {}).get("state") in ("PENDING", "RUNNING"):
            resp = self._api("get", f"/api/2.0/sql/statements/{statement_id}")

        state = resp.get("status", {}).get("state")
        if state != "SUCCEEDED":
            raise DatabricksSQLError(json.dumps(resp.get("status", resp), indent=2))

        manifest = resp.get("manifest", {})
        columns = [c["name"] for c in manifest.get("schema", {}).get("columns", [])]
        total_chunks = manifest.get("total_chunk_count", 1)

        if disposition == "EXTERNAL_LINKS":
            rows = self._fetch_external(resp, statement_id, total_chunks)
        else:
            rows = list(resp.get("result", {}).get("data_array") or [])
            for idx in range(1, total_chunks):
                chunk = self._api("get", f"/api/2.0/sql/statements/{statement_id}/result/chunks/{idx}")
                rows.extend(chunk.get("data_array") or [])

        return columns, rows

    def _fetch_external(
        self, resp: dict[str, Any], statement_id: str, total_chunks: int
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        links = list(resp.get("result", {}).get("external_links") or [])
        seen: set[int] = set()
        while links:
            link = links.pop(0)
            idx = link.get("chunk_index", 0)
            if idx in seen:
                continue
            seen.add(idx)
            with urllib.request.urlopen(link["external_link"], timeout=120) as fh:
                rows.extend(json.loads(fh.read().decode("utf-8")))
            nxt = link.get("next_chunk_index")
            if nxt is not None and nxt not in seen:
                more = self._api(
                    "get", f"/api/2.0/sql/statements/{statement_id}/result/chunks/{nxt}"
                )
                links.extend(more.get("external_links") or [])
        if len(seen) < total_chunks:
            raise DatabricksSQLError(f"only fetched {len(seen)}/{total_chunks} result chunks")
        return rows

    def query_dicts(self, statement: str, row_limit: int | None = None) -> list[dict[str, Any]]:
        columns, rows = self.query(statement, row_limit=row_limit)
        return [dict(zip(columns, row)) for row in rows]

    def query_to_csv(self, statement: str, out_path: str | Path, row_limit: int | None = None) -> int:
        columns, rows = self.query(statement, row_limit=row_limit)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
        return len(rows)


def _iter_preview(columns: list[str], rows: list[list[Any]], n: int) -> Iterator[str]:
    for row in rows[:n]:
        yield json.dumps(dict(zip(columns, row)), default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ad-hoc SQL against Databricks.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("statement", nargs="?", help="SQL statement text")
    src.add_argument("--file", help="Path to a .sql file")
    parser.add_argument("--csv", help="Write full results to this CSV path")
    parser.add_argument("--limit", type=int, help="Server-side row limit")
    parser.add_argument("--preview", type=int, default=10, help="Rows to print (default 10)")
    args = parser.parse_args()

    statement = Path(args.file).read_text() if args.file else args.statement

    client = DatabricksSQL()
    try:
        columns, rows = client.query(statement, row_limit=args.limit)
    except DatabricksSQLError as exc:
        print(f"Query failed:\n{exc}", file=sys.stderr)
        return 1

    print(f"columns ({len(columns)}): {', '.join(columns)}")
    print(f"rows: {len(rows)}")
    for line in _iter_preview(columns, rows, args.preview):
        print(f"  {line}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"saved -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
