#!/usr/bin/env python3
"""Local stand-ins for the Elasticsearch-backed signals, for demoing without cluster access.

The cluster sits behind an IP allowlist, so the demo cannot reach it from an unlisted network.
This module reproduces the two query-time services locally:

  LocalBM25       Okapi BM25 (k1=1.2, b=0.75, Lucene's defaults) over the same title + Big-4
                  attribute text. It is NOT the production index: IDF comes from the 80k demo
                  catalog rather than the full 306M-document index, and the field boosts and
                  analyzer chain are not replicated. Rankings are close but not identical.

  LocalEncoders   The same checkpoints Elasticsearch hosts, run in-process. These do match:
                  the document vectors in the .npz files were produced by these checkpoints, and
                  for E5 the deployed model is the same `intfloat/multilingual-e5-small`.

The UI labels which mode is in use so a local run is never mistaken for the live stack.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class LocalBM25:
    """Okapi BM25 over an in-memory inverted index."""

    def __init__(self, ecodes: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.ecodes = np.asarray(ecodes, dtype=object)
        self.k1, self.b = k1, b
        n_docs = len(texts)

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(n_docs, dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = defaultdict(int)
            for token in tokenize(text):
                counts[token] += 1
            lengths[i] = sum(counts.values())
            for token, tf in counts.items():
                postings[token].append((i, tf))

        self.avgdl = float(lengths.mean()) if n_docs else 0.0
        self.lengths = lengths
        self.index: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for token, items in postings.items():
            docs = np.fromiter((d for d, _ in items), dtype=np.int32, count=len(items))
            tfs = np.fromiter((t for _, t in items), dtype=np.float32, count=len(items))
            df = len(items)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            self.index[token] = (docs, tfs, idf)

    def search(self, query: str, window: int) -> list[str]:
        return list(self.search_scored(query, window))

    def search_scored(self, query: str, window: int) -> dict[str, float]:
        scores = np.zeros(len(self.ecodes), dtype=np.float32)
        matched = False
        for token in tokenize(query):
            entry = self.index.get(token)
            if entry is None:
                continue
            matched = True
            docs, tfs, idf = entry
            dl = self.lengths[docs]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            np.add.at(scores, docs, idf * tfs * (self.k1 + 1.0) / denom)
        if not matched:
            return {}
        window = min(window, int((scores > 0).sum()))
        if window <= 0:
            return {}
        top = np.argpartition(-scores, window - 1)[:window]
        top = top[np.argsort(-scores[top], kind="stable")]
        return {self.ecodes[i]: float(scores[i]) for i in top}


class LocalEncoders:
    """Lazily loaded local copies of the deployed query encoders."""

    SPECS = {
        "e5": ("intfloat/multilingual-e5-small", "mean_pool"),
        "siglip": ("google/siglip-base-patch16-512", "siglip_text"),
        "jina": ("jinaai/jina-clip-v2", "jina_text"),
    }

    def __init__(self) -> None:
        self._loaded: dict[str, tuple] = {}

    def _load(self, kind: str):
        if kind in self._loaded:
            return self._loaded[kind]
        import torch
        from transformers import AutoModel, AutoTokenizer

        repo, mode = self.SPECS[kind]
        print(f"  loading local encoder {kind} ({repo})")
        model = AutoModel.from_pretrained(repo, trust_remote_code=(kind == "jina")).eval()
        tokenizer = None if mode == "jina_text" else AutoTokenizer.from_pretrained(repo)
        self._loaded[kind] = (model, tokenizer, mode, torch)
        return self._loaded[kind]

    def encode(self, kind: str, text: str) -> np.ndarray:
        model, tokenizer, mode, torch = self._load(kind)
        with torch.no_grad():
            if mode == "jina_text":
                vec = model.encode_text([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return vec.astype(np.float32)
            if mode == "siglip_text":
                batch = tokenizer([text], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                vec = model.get_text_features(**batch)[0].numpy()
            else:
                batch = tokenizer([f"query: {text}"], padding=True, truncation=True, max_length=128, return_tensors="pt")
                out = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = ((out * mask).sum(1) / mask.sum(1))[0].numpy()
        return (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)
