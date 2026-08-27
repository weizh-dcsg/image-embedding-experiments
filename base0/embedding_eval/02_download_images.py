#!/usr/bin/env python3
"""Step 2: download product images for the test-set products.

Scene7 URLs carry rendering presets after '?'. Those presets are stripped and replaced
with an explicit square render so every image arrives at a consistent size for SigLIP.

Outputs:
  data/images/<ecode>.jpg
  data/image_manifest.csv  -- ecode, path, ok, error
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) embedding-eval/1.0"


def render_url(raw_url: str) -> str:
    base = raw_url.split("?", 1)[0]
    parts = urllib.parse.urlsplit(base)
    # some titles carry accented characters into the asset path; urllib needs them percent-encoded
    path = urllib.parse.quote(parts.path, safe="/%:@!$&'()*+,;=~")
    netloc = parts.netloc.encode("idna").decode("ascii") if not parts.netloc.isascii() else parts.netloc
    base = urllib.parse.urlunsplit((parts.scheme, netloc, path, "", ""))
    return f"{base}{config.IMAGE_RENDER_PRESET}"


def download_one(ecode: str, url: str, dest: Path) -> tuple[str, str, bool, str]:
    if dest.exists() and dest.stat().st_size > 1024:
        return ecode, str(dest), True, ""
    try:
        req = urllib.request.Request(render_url(url), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=config.DOWNLOAD_TIMEOUT) as resp:
            payload = resp.read()
        if len(payload) < 1024:
            return ecode, "", False, f"payload too small ({len(payload)} bytes)"
        dest.write_bytes(payload)
        return ecode, str(dest), True, ""
    except Exception as exc:  # one unusable URL must not abort the batch
        return ecode, "", False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=config.DOWNLOAD_WORKERS)
    args = parser.parse_args()

    config.ensure_dirs()
    products = pd.read_csv(config.PRODUCTS_CSV)
    print(f"downloading {len(products)} product images -> {config.IMAGE_DIR}")

    results: list[tuple[str, str, bool, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row.ecode, row.image_url, config.IMAGE_DIR / f"{row.ecode}.jpg"): row.ecode
            for row in products.itertuples()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 250 == 0 or done == len(futures):
                ok = sum(1 for r in results if r[2])
                print(f"  {done}/{len(futures)} complete ({ok} ok)")

    manifest = pd.DataFrame(results, columns=["ecode", "path", "ok", "error"])
    manifest.to_csv(config.IMAGE_MANIFEST_CSV, index=False)

    failed = manifest[~manifest["ok"]]
    print(f"ok: {int(manifest['ok'].sum())}  failed: {len(failed)}")
    if len(failed):
        print("sample errors:")
        for row in failed.head(5).itertuples():
            print(f"  {row.ecode}: {row.error}")
    print(f"saved -> {config.IMAGE_MANIFEST_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
