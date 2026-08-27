"""Shared configuration for the SigLIP (image) vs Jina v5 (text) embedding evaluation."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Kept outside the repo tree: this workspace lives in OneDrive CloudStorage, which dehydrates cold
# files to cloud placeholders. Reads then block on a re-download and time out mid-run, and every
# image write is throttled by sync.
DATA_DIR = Path(os.environ.get("EMBEDDING_EVAL_DATA", Path.home() / "embedding_eval_data"))
# Imagery is shared across experiment variants so a larger candidate pool only fetches the delta.
IMAGE_STORE = Path(os.environ.get("EMBEDDING_EVAL_IMAGE_STORE", DATA_DIR))
IMAGE_DIR = IMAGE_STORE / "images"
CROP_DIR = IMAGE_STORE / "images_cropped"
# Naive largest-area crops, kept only as the ablation arm for MGPL.
CROP_NAIVE_DIR = IMAGE_STORE / "images_cropped_naive"
EMB_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = Path(os.environ.get("EMBEDDING_EVAL_RESULTS", ROOT / "results"))
SQL_DIR = ROOT / "sql"

TEST_SET_CSV = DATA_DIR / "test_set.csv"
PRODUCTS_CSV = DATA_DIR / "products.csv"
ATTRIBUTES_CSV = DATA_DIR / "product_attributes.csv"
IMAGE_MANIFEST_CSV = DATA_DIR / "image_manifest.csv"
# Crop manifests live with the shared image store so variants reuse crops instead of recomputing.
CROP_MANIFEST_CSV = IMAGE_STORE / "crop_manifest.csv"
CROP_NAIVE_MANIFEST_CSV = IMAGE_STORE / "crop_manifest_naive.csv"

# --- Judgement list settings, aligned to ds-ecm-search-ranking-ltr ---
# Sources:
#   applications/srlp-ltr/services/judgement-list-data-prep/job.py  (job parameter defaults)
#   sandbox/ltr_vanilla/2-judgement_list.py                          (IPW clip, decay, alpha)
# ml_events stores the banner upper-cased; the clickstream tables use lower-case.
BANNER = "DSG"                # LTR: store="DSG"
CHANNEL = "WEB"               # LTR: channel="web"
DAYS_BEFORE_TODAY = int(os.environ.get("DAYS_BEFORE_TODAY", 3))   # LTR: days_before_today=3
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", 90))          # LTR: num_days_train=90
END_DATE = os.environ.get("END_DATE", "")  # empty -> today - DAYS_BEFORE_TODAY
MIN_GROUP_SIZE = int(os.environ.get("MIN_GROUP_SIZE", 10))        # LTR: min_group_size=10
MAX_GROUP_SIZE = int(os.environ.get("MAX_GROUP_SIZE", 240))       # LTR: max_group_size=240 (48*5)
IPW_CLIP_POSITION = int(os.environ.get("IPW_CLIP_POSITION", 48))  # LTR: ipw_clip_position=48
DECAY_FLATNESS = float(os.environ.get("DECAY_FLATNESS", 0.18))    # LTR: decay_flatness=0.18
DECAY_MIDPOINT = int(os.environ.get("DECAY_MIDPOINT", 30))        # LTR: decay_midpoint=30
ALPHA = float(os.environ.get("ALPHA", 1.0))                       # LTR: alpha=1.0 (quantile smoothing)

# Query sampling: every term that survives the LTR eligibility filters is kept (no volume-based
# cap on the final query count). These control the upstream candidate pool only.
TERM_POOL_HEAD = int(os.environ.get("TERM_POOL_HEAD", 6000))         # top-volume candidates
TAIL_POOL_SIZE = int(os.environ.get("TAIL_POOL_SIZE", 6000))         # random long-tail candidates
TAIL_MIN_IMPRESSIONS = int(os.environ.get("TAIL_MIN_IMPRESSIONS", 20))  # floor so tail terms can
                                                                          # still clear min_group_size
# Tier cut points: percentile rank of total impressions among queries that pass every LTR filter.
HEAD_PCTL = float(os.environ.get("HEAD_PCTL", 0.95))
TORSO_PCTL = float(os.environ.get("TORSO_PCTL", 0.70))

# --- Models ---
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip-base-patch16-512")
JINA_MODEL = os.environ.get("JINA_MODEL", "jinaai/jina-embeddings-v5-text-nano")
JINA_SMALL_MODEL = os.environ.get("JINA_SMALL_MODEL", "jinaai/jina-embeddings-v5-text-small")
DETECTOR_MODEL = os.environ.get("DETECTOR_MODEL", "facebook/detr-resnet-50")
# Open-vocabulary detector, prompted with the product's own type phrase.
OWL_MODEL = os.environ.get("OWL_MODEL", "google/owlv2-base-patch16-ensemble")
# W6 text-representation add-ons. E5_BASE_MODEL is English-only; the other two are multilingual.
E5_BASE_MODEL = os.environ.get("E5_BASE_MODEL", "intfloat/e5-base-v2")
E5_SMALL_MULTI_MODEL = os.environ.get("E5_SMALL_MULTI_MODEL", "intfloat/multilingual-e5-small")
E5_LARGE_INSTRUCT_MODEL = os.environ.get(
    "E5_LARGE_INSTRUCT_MODEL", "intfloat/multilingual-e5-large-instruct"
)
# Text tower of the same omni model already used for the W7/W8 image-encoder comparison.
JINA_OMNI_NANO_MODEL = os.environ.get("JINA_OMNI_NANO_MODEL", "jinaai/jina-embeddings-v5-omni-nano")

# --- Image download ---
# scene7 URLs carry sizing presets; request a square render large enough for SigLIP-512.
IMAGE_RENDER_PRESET = "?wid=512&hei=512&fmt=jpeg&qlt=85"
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", 16))
DOWNLOAD_TIMEOUT = 20

# --- Evaluation ---
# 48 is the IPW rank clip; 96 and 144 probe beyond it. Median pool is ~105, so k >= 96 exceeds the
# candidate list for most queries.
K_VALUES = (5, 10, 20, 48, 96, 144)
BOOTSTRAP_SAMPLES = int(os.environ.get("BOOTSTRAP_SAMPLES", 2000))
RANDOM_SEED = 20260808

# LTR grades relevance 0-4 from smoothed weighted-CTR quantile bins; the SQL emits the grade
# directly, so no Python-side binning is needed.
MAX_RELEVANCE = 4


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGE_STORE, IMAGE_DIR, CROP_DIR, EMB_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
