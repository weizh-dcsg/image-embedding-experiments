#!/usr/bin/env python3
"""Step 1b: fetch the LTR "Big-4" product attributes for the test-set products.

Source and attribute ids follow ds-ecm-search-ranking-ltr
(sandbox/ltr_vanilla/features/test_experimental_elastic_features.py):

  X_BRAND -> Brand          2101 -> Gender by Age
  5382    -> Product Type   4285 -> Activity

Outputs:
  data/product_attributes.csv -- ecode, brand, product_type, product_activity,
                                 gender_by_age, attr_text
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "base0"))

import config  # noqa: E402
from dbx_sql import DatabricksSQL  # noqa: E402

FIELDS = ["brand", "product_type", "product_activity", "gender_by_age"]

SQL = """
SELECT
    ecode,
    MAX(CASE WHEN attr_id = 'X_BRAND' THEN attr_value END) AS brand,
    MAX(CASE WHEN attr_id = '5382'    THEN attr_value END) AS product_type,
    MAX(CASE WHEN attr_id = '4285'    THEN attr_value END) AS product_activity,
    MAX(CASE WHEN attr_id = '2101'    THEN attr_value END) AS gender_by_age
FROM prod_ml_feature_store_db.products.ecode_attribute
WHERE attr_id IN ('X_BRAND', '5382', '4285', '2101')
GROUP BY ecode
"""


def main() -> int:
    config.ensure_dirs()
    products = pd.read_csv(config.PRODUCTS_CSV)
    wanted = set(products["ecode"].astype(str))
    print(f"test-set products: {len(wanted)}")

    attrs = pd.DataFrame(DatabricksSQL().query_dicts(SQL))
    attrs = attrs[attrs["ecode"].astype(str).isin(wanted)].copy()

    for col in FIELDS:
        attrs[col] = attrs[col].fillna("").astype(str).str.strip()

    # Field order is fixed so the concatenation is deterministic; empties drop out rather than
    # leaving separators that would tokenize as noise.
    attrs["attr_text"] = attrs[FIELDS].apply(
        lambda r: " ".join(v for v in r if v), axis=1
    )

    attrs = attrs[attrs["attr_text"] != ""]
    out = products[["ecode"]].merge(attrs, on="ecode", how="left")
    for col in FIELDS + ["attr_text"]:
        out[col] = out[col].fillna("").astype(str)

    out.to_csv(config.ATTRIBUTES_CSV, index=False)

    have = (out["attr_text"] != "").sum()
    print(f"attribute coverage: {have}/{len(out)}  ({100 * have / len(out):.1f}%)")
    for col in FIELDS:
        n = (out[col] != "").sum()
        print(f"  {col:18s} {n:6d}  {100 * n / len(out):5.1f}%")
    print(f"\nmean attr_text words: {out['attr_text'].str.split().str.len().mean():.1f}")
    print("examples:")
    for t in out.loc[out["attr_text"] != "", "attr_text"].head(5):
        print(f"  {t}")
    print(f"saved -> {config.ATTRIBUTES_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
