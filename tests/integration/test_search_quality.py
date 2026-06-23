"""Semantic-quality smoke tests for /search (6.E).

Uses a live TestClient + controlled FAISS vectors so quality checks are deterministic.
All labels have ||v||=10 so L2 distances are meaningful when query points at one of them.

Strategy: normalize all labels to unit norm, scale up by 10. Then encoding the target
unit vector toward any label's direction picks that entry as uniquely closest among all
labelled entries (distance zero vs ~10 for all others).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient


class _NprobeIndex:
    """Thin wrapper that adds a writable **.nprobe** to any FAISS index."""

    __slots__ = ("_idx", "_n", "_titles")

    def __init__(self, underlying, titles):
        self._idx = underlying
        self._n = 64
        self._titles = list(titles)

    @property
    def nprobe(self):
        return self._n

    @nprobe.setter
    def nprobe(self, value):
        self._n = int(value)

    def search(self, x, k):
        ds, ix = self._idx.search(x, k)
        if ds.shape[1] < k:
            pad_n = k - ds.shape[1]
            n_vecs = max(len(self._titles), 50)
            pad_zeros = np.zeros((ds.shape[0], pad_n), dtype="float32")
            ix_pad = (np.arange(n_vecs, n_vecs + pad_n) % len(self._titles)).reshape(1, -1)
            ds = np.concatenate([ds, pad_zeros], axis=1)
            ix = np.concatenate([ix, ix_pad], axis=1)
        return ds[:, :k].astype("float32"), ix[:, :k]


# ── fixture: all labels equal-norm so L2 search is fair ─────────────────────


@pytest.fixture(scope="module")
def _pairs_dir() -> Path:
    """Create temp dir w/ 5 labelled vectors (all norm=10) + padding."""
    td = Path(tempfile.mkdtemp(prefix="wiki_qdb_"))
    dim = 384

    rng = np.random.RandomState(9)
    base = faiss.IndexFlatL2(dim)

    labels: np.ndarray = np.zeros((5, dim), dtype="float32")
    for i in range(5):
        labels[i] = rng.randn(dim).astype("float32")
    # Normalize each to unit vector → scale by 10 (||v_z - q||=|q|=1 is far)
    labels /= np.linalg.norm(labels, axis=1, keepdims=True)  # unit norm each
    labels *= 10  # all at ||v||=10

    base.add(labels)  # indices 0–4
    faiss.write_index(base, str(td / "wiki_faiss.index"))

    n_pad = max(20 - len(labels), 7)  # enough to avoid OOB padding
    titles: list[str] = ["Photosynthesis", "Mitochondrion", "Quantum mechanics"]
    for i in range(n_pad):
        titles.append(f"Wiki context entry {i}")
    (td / "wiki_titles.txt").write_text("\n".join(titles), encoding="utf-8")

    np.save(str(td / "_LABELS.npy"), labels)  # save for query generation
    return td


def _make_qc(pairs_dir: Path) -> Dict[str, Any]:
    import app.config as cfg
    import app.main as main_mod

    cfg.FAISS_INDEX = str(pairs_dir / "wiki_faiss.index")
    cfg.TITLES_FILE = str(pairs_dir / "wiki_titles.txt")
    main_mod.FAISS_INDEX = cfg.FAISS_INDEX  # type: ignore[attr-defined]
    main_mod.TITLES_FILE = cfg.TITLES_FILE  # type: ignore[attr-defined]

    raw = faiss.read_index(cfg.FAISS_INDEX)
    tpls = (pairs_dir / "wiki_titles.txt").read_text(encoding="utf-8").splitlines()
    wrap = _NprobeIndex(raw, tpls)
    wrap.nprobe = cfg.DEFAULT_NPROBE  # search all

    main_mod.state["index"] = wrap
    main_mod.state["titles"] = tpls
    labels = np.load(str(pairs_dir / "_LABELS.npy"))  # shape (5, 384), all norm=10
    return {"wrap": wrap, "labels": labels}


# ── quality pairs (6.E) ──────────────────────────────────────────────────────

PAIRS: list[tuple[str, int, str]] = [  # (query_term, target_label_idx, expected_subtitle)
    ("photosynthesis", 0, "Photosynthesis"),
    ("mitochondria", 1, "Mitochondrion"),
    ("quantum mechanics", 2, "Quantum mechanics"),
]


@pytest.mark.parametrize("query,target_lidx,expected_substr", PAIRS)
def test_quality(_pairs_dir: Path, query: str, target_lidx: int, expected_substr: str) -> None:
    """Encode to labelled unit direction → expected title in top hits (6.E)."""
    from unittest.mock import MagicMock

    import app.main as main_mod

    qc = _make_qc(_pairs_dir)

    # encode a unit vector pointing exactly at the target label's direction
    # labels[target_lidx] has ||v||=10, so q = v/|v| is exact direction toward it
    tgt_vec = qc["labels"][target_lidx : target_lidx + 1]  # shape (1, 384)
    norm = float(np.linalg.norm(tgt_vec)) or 1.0
    enc_out = (tgt_vec / norm).astype("float32")  # unit direction

    model_mock = MagicMock()
    model_mock.encode.return_value = enc_out  # controlled vector
    main_mod.state["model"] = model_mock

    client = TestClient(main_mod.app)
    resp = client.get("/search", params={"q": query, "k": 5})
    assert resp.status_code == 200
    top_titles = [r["title"] for r in resp.json()["results"]]

    assert any(expected_substr.lower() in t.lower() for t in top_titles), (
        f"{expected_substr} not found in top results for '{query}': {top_titles}"
    )
    client.close()
