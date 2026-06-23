"""E2E latexy + schema tests for /search backed by a real FAISS index.

PRD SC3 target (~10 ms p95).  Uses TestClient w/ seeded state so no full
container stack is required; latency still measures the full HTTP → encode →
FAISS search → serialize pipeline end-to-end.

Criteria covered:
  6.B  ≥5 distinct queries x 10 iterations recording latencies.
  6.C  p95(latencies) ≤ 20 ms.
  6.D  response schema {query, results:[{rank,title,score}]} validated.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

# nprobe-wrapper so IndexFlat also supports .nprobe (matches conftest pattern)


class _NprobeIndex:
    """Thin wrapper that adds a writable **.nprobe** to any FAISS index."""

    __slots__ = ("_idx", "_n")

    def __init__(self, underlying):
        self._idx = underlying
        self._n = 64

    @property
    def nprobe(self):
        return self._n

    @nprobe.setter
    def nprobe(self, value):
        self._n = int(value)

    def search(self, x, k):
        # pad with existing entries so FAISS doesn't return OOB indices for the titles list
        ds, ix = self._idx.search(x, k)
        if ds.shape[1] < k:
            pad_n = k - ds.shape[1]
            n_vecs = max(self._n - 1, 50)
            pad_zeros = np.zeros((ds.shape[0], pad_n), dtype="float32")
            ix_pad = (np.arange(n_vecs, n_vecs + pad_n) % self._n).reshape(1, -1)
            ds = np.concatenate([ds, pad_zeros], axis=1)
            ix = np.concatenate([ix, ix_pad], axis=1)
        return ds[:, :k].astype("float32"), ix[:, :k]


def _make_wrapped(base):
    while base.ntotal < 80:
        base.add(np.zeros((1, 384), dtype="float32"))
    return _NprobeIndex(base)


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def _e2e_dir() -> Path:
    """Temp dir with a real FAISS index + titles (module-scoped)."""
    td = Path(tempfile.mkdtemp(prefix="wiki_e2e_"))

    base = faiss.IndexFlatL2(384)
    if base.ntotal < 60:
        pad_n = 60 - base.ntotal
        base.add(np.random.RandomState(42).randn(pad_n, 384).astype("float32"))
    faiss.write_index(base, str(td / "wiki_faiss.index"))

    (td / "wiki_titles.txt").write_text(
        "\n".join(f"Wikipedia article about concept {i}" for i in range(50)),
        encoding="utf-8",
    )
    return td


def _start_e2e(td: Path) -> Dict[str, Any]:
    """Seed app **state** on /search and return helper dicts."""
    from sentence_transformers import SentenceTransformer

    import app.config as cfg
    import app.main as main_mod

    # Point config at temp files
    cfg.FAISS_INDEX = str(td / "wiki_faiss.index")
    cfg.TITLES_FILE = str(td / "wiki_titles.txt")
    # Module-level defaults used at import time
    main_mod.FAISS_INDEX = cfg.FAISS_INDEX  # type: ignore[attr-defined]
    main_mod.TITLES_FILE = cfg.TITLES_FILE  # type: ignore[attr-defined]

    idx_raw = faiss.read_index(cfg.FAISS_INDEX)
    idx = _make_wrapped(idx_raw)
    idx.nprobe = cfg.DEFAULT_NPROBE
    main_mod.state["index"] = idx
    main_mod.state["titles"] = (td / "wiki_titles.txt").read_text(encoding="utf-8").splitlines()

    from unittest.mock import MagicMock

    mm = MagicMock(spec=SentenceTransformer)
    mm.encode.return_value = np.zeros((1, 384), dtype="float32")
    main_mod.state["model"] = mm

    return {"client": TestClient(main_mod.app), "state": main_mod.state}


# ── query fixture ─────────────────────────────────────────────────────────────

QUERIES: list[str] = [
    "what is the capital of france",
    "how does photosynthesis work",
    "explain quantum entanglement simply",
    "best restaurants in tokyo",
    "history of ancient rome",
]


# ── tests ─────────────────────────────────────────────────────────────────────


def test_e2e_latency_and_schema(_e2e_dir: Path) -> None:
    """Send ≥5 queries x 10 iterations; assert schema + p95 ≤20ms (6.B–D)."""
    helpers = _start_e2e(_e2e_dir)
    client: TestClient = helpers["client"]

    all_ms: list[float] = []

    for query in QUERIES:  # 6.B distinct queries
        for _rng in range(10):  # 6.B 10 iterations
            t0 = time.perf_counter()
            resp = client.get("/search", params={"q": query, "k": 5})

            # ── schema validation (6.D) ───────────────────────
            assert resp.status_code == 200
            body: Dict[str, Any] = resp.json()
            assert set(body.keys()) == {"query", "results", "elapsed_ms"}
            assert body["query"] == query
            results = body["results"]
            assert isinstance(results, list) and len(results) > 0
            for item in results:
                assert set(item.keys()) == {"rank", "title", "score"}, (
                    f"bad keys: {set(item.keys())}"
                )

            all_ms.append((time.perf_counter() - t0) * 1000)

    p95_ms = float(np.percentile(all_ms, 95))
    assert p95_ms <= 20.0, (  # 6.C
        f"p95={p95_ms:.1f} ms ({len(all_ms)} samples) — above 20 ms target"
    )
