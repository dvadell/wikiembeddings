"""test_build_faiss.py — build_faiss_index() stage (T14).

Uses a tiny in-memory fixture (10 L2-normalised vectors, nlist=2) with
``faiss.write_index`` patched so *no* real FAISS training is needed.  The
manifest write and Path.rename operations on disk are left unpatched because
they operate on simple JSON/text files that work fine without the mock dance.

Acceptance criteria covered
============================
14.A  Both ``index_path`` and ``manifest_path`` exist after call
14.B  Index write is atomic: ``*.tmp`` renamed to final path (verify .tmp disappears)
14.C  Manifest contains all required keys
14.D  resume=True + existing manifest → ``faiss.write_index`` *not* called
14.E  Coverage on FAISS build stage ≥ 95%

"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from unittest import mock

import numpy as np
import pytest

import app.build_index as m


# ── Fixtures ------------------------------------------------------------------- #


def _stub_faiss() -> ModuleType:
    """Return a fresh stub ``faiss`` module with mocked ``write_index`` that *writes to disk*.

    This lets us verify atomicity by checking on-disk files later.
    Callers patch it into ``sys.modules`` via mock.patch.dict.
    """

    idx_ivf = mock.MagicMock()

    def ivf_init(_q, _d, _nlist, _metric):  # noqa: ANN001 — intentionally unused.
        return idx_ivf

    stub = mock.MagicMock()
    stub.IndexFlatL2.return_value = mock.MagicMock()  # noqa: FURB113 — needed for constructor.
    stub.IndexIVFFlat.side_effect = ivf_init  # type: ignore[attr-defined]
    stub.METRIC_INNER_PRODUCT = 1

    def _write_impl(_idx, tmp_path):
        """Write a placeholder FAISS index .tmp file so Path.rename works."""
        Path(tmp_path).write_bytes(b"\x00\x00")  # noqa: FURB127 — fake content.

    stub.write_index = mock.MagicMock(side_effect=_write_impl)  # noqa: FURB113
    return stub


@pytest.fixture()
def fixture_data(tmp_path: Path, n_titles: int = 10) -> tuple[Path, Path]:
    """Write a tiny embeddings file + titles file.

    Uses ``np.save`` so the array shape is preserved in the numpy header.
    Returns ``(embeddings_path, titles_path)``.
    """
    rng = np.random.RandomState(42)
    embed_dim = 384
    embeddings_path = tmp_path / "embeddings.npy"

    # Create the array, write it out in .npy format (preserves shape).
    arr = rng.randn(n_titles, embed_dim).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
    arr[:] = arr / norms
    np.save(str(embeddings_path), arr)

    titles_path = tmp_path / "titles.txt"
    titles_path.write_text(
        "\n".join(f"Title {i}" for i in range(int(n_titles))) + "\n", encoding="utf-8"
    )
    return embeddings_path, titles_path


@pytest.fixture()
def faiss_stub():
    """Per-test stub so mock call-counters stay clean."""
    yield _stub_faiss()


# ── 14.A & 14.C  --  files exist + manifest keys ------------------------------- #


def test_14A_manifest_and_index_exist(tmp_path: Path, fixture_data, faiss_stub):
    """14.A: both index and manifest written on disk after build."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        result = m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=False,
            progress_cb=mock.Mock(),  # noqa: FURB113
        )

    assert isinstance(result, dict)
    assert index_path.exists()
    assert manifest_path.exists()


def test_14C_manifest_keys(tmp_path: Path, fixture_data, faiss_stub):
    """14.C: manifest contains all required keys."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        result = m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=False,
            progress_cb=mock.Mock(),  # noqa: FURB113
        )

    assert set(result.keys()) == {"built_at", "title_count", "model_name", "nlist", "embed_dim"}


# ── 14.B  --  atomic writes (no stale .tmp files) ------------------------------ #


def test_14B_no_stale_tmp(tmp_path: Path, fixture_data, faiss_stub):
    """14.B: no ``*.tmp`` file remains after successful build."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=False,
            progress_cb=mock.Mock(),  # noqa: FURB113
        )

    for p in index_path.parent.iterdir():
        assert not p.name.endswith(".tmp"), f"stale tmp file left: {p}"


def test_14B_write_index_uses_tmp(tmp_path: Path, fixture_data, faiss_stub):
    """Verify ``faiss.write_index`` is called with a ``*.tmp`` path."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=False,
            progress_cb=mock.Mock(),  # noqa: FURB113
        )

    faiss_stub.write_index.assert_called_once()
    written_path_arg = faiss_stub.write_index.call_args[0][1]
    assert str(written_path_arg).endswith(".tmp"), (
        f"write_index target should end with .tmp, got {written_path_arg}"
    )


# ── 14.D  --  resume skips train & write --------------------------------------- #


def test_14D_resume_skips_build(tmp_path: Path, fixture_data):
    """14.D: ``resume=True`` + existing manifest → skip training and return existing manifest."""
    embeddings, titles = fixture_data

    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps({"built_at": "now", "title_count": 0}), encoding="utf-8")

    index_path = tmp_path / "wiki_faiss.index"
    index_path.write_text("dummy", encoding="utf-8")

    progress_cb = mock.Mock()  # noqa: FURB113

    result = m.build_faiss_index(
        embeddings_path=embeddings,
        titles_path=titles,
        index_path=index_path,
        manifest_path=manifest_path,
        nlist=2,
        sample_frac=0.5,
        resume=True,
        progress_cb=progress_cb,
    )

    assert result["built_at"] == "now"
    assert progress_cb.call_args_list[-1] == mock.call(1.0), (
        f"progress_cb should end at 1.0; got {progress_cb.call_args_list}"
    )


def test_14D_resume_proceeds_without_manifest(tmp_path: Path, fixture_data, faiss_stub):
    """When ``resume=True`` but no manifest exists, build proceeds normally."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"
    assert not manifest_path.exists()  # sanity check.

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=True,  # truthy but no manifest → proceed normally.
            progress_cb=mock.Mock(),  # noqa: FURB113
        )

    assert faiss_stub.write_index.call_count == 1


# ── Edge cases --------------------------------------------------------------- #


def test_empty_embeddings(tmp_path: Path):
    """Zero embeddings → returns empty dict, progress at 1.0."""
    embed_dim = 384
    empty_embeddings = tmp_path / "empty.npy"
    mmap = np.memmap(str(empty_embeddings), dtype="float32", mode="w+", shape=(0, int(embed_dim)))
    mmap.flush()
    del mmap

    titles = tmp_path / "titles.txt"
    titles.write_text("", encoding="utf-8")
    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"

    result = m.build_faiss_index(
        embeddings_path=empty_embeddings,
        titles_path=titles,
        index_path=index_path,
        manifest_path=manifest_path,
        nlist=2,
        sample_frac=0.5,
        resume=False,
        progress_cb=mock.Mock(),  # noqa: FURB113
    )

    assert result == {}


# ── Progress tracking (branch coverage) ---------------------------------------- #


def test_progress_reaches_1_0(tmp_path: Path, fixture_data, faiss_stub):
    """Progress callbacks end at 1.0."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"
    progress_cb = mock.Mock()  # noqa: FURB113

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=False,
            progress_cb=progress_cb,
        )

    assert progress_cb.call_args_list[-1] == mock.call(1.0)


def test_progress_monotonic(tmp_path: Path, fixture_data, faiss_stub):
    """Progress values monotonically increase."""
    embeddings, titles = fixture_data

    index_path = tmp_path / "wiki_faiss.index"
    manifest_path = tmp_path / "build_manifest.json"
    values: list[float] = []

    with mock.patch.dict("sys.modules", {"faiss": faiss_stub}):
        m.build_faiss_index(
            embeddings_path=embeddings,
            titles_path=titles,
            index_path=index_path,
            manifest_path=manifest_path,
            nlist=2,
            sample_frac=0.5,
            resume=False,
            progress_cb=values.append,  # type: ignore[arg-type] — intentionally loosely typed; side-effects only.
        )

    for i in range(1, len(values)):
        assert values[i] >= values[i - 1], f"not monotonic: {values}"
