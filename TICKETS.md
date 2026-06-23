# Tickets — Index Build Pipeline

> **Reference:** [PRD.md](./PRD.md) — see §5, §7.3, §7.4, §8.1, §8.2, §9, SC2, SC6–SC8
>
> These tickets add the **auto-build on first start** feature described in PRD v1.2.
> The build runs as a background thread inside the existing FastAPI process — same image,
> same container, no orchestration changes beyond adding a named volume.
>
> Tickets continue from T10. You've completed T0–T7.

---

## Global acceptance criteria (all tickets)

| # | Criterion | How verified |
|---|-----------|--------------|
| G1 | `ruff format` on all modified Python files | CI step or pre-commit hook |
| G2 | Full pytest suite exits 0 on Python 3.12 (`ubuntu-latest`) | `pytest tests/` in CI |
| G3 | `ruff check` passes on all modified and new Python files | `ruff check app/ tests/` in CI |
| G4 | Coverage stays ≥ 95% line coverage for the `app/` package | `--cov-fail-under=95` in CI |

---

## T11 — Build configuration & env vars
git branch: `ai-agent/t11_build_config_env_vars`

Extend `app/config.py` with the new build-stage environment variables introduced in PRD §9.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 11.1 | Add the following config vars to `app/config.py` with documented defaults: `WIKI_DUMP_URL`, `BUILD_BATCH_SIZE` (int, 512), `BUILD_NLIST` (int, 4096), `BUILD_SAMPLE_FRAC` (float, 0.1), `BUILD_RESUME` (bool, **true**), `BUILD_MANIFEST` (str, `"build_manifest.json"`) | Mirror the pattern used for existing vars |
| 11.2 | `WIKI_DUMP_URL` default should resolve to the Wikimedia latest Cirrussearch dump for English Wikipedia. Acceptable: hard-code the pattern URL with `{date}` resolved at runtime, or use the Wikimedia API to discover the latest dump | Document chosen approach in a code comment |
| 11.3 | `BUILD_RESUME` must be coerced from the string `"true"` / `"false"` / `"1"` / `"0"` to a Python `bool` — not a raw string comparison | `os.environ.get("BUILD_RESUME", "true").lower() in ("true", "1")` |
| 11.4 | Write unit tests in `tests/test_build_config.py` covering: all new defaults, env-override of each new var, bool coercion edge cases for `BUILD_RESUME`, int/float coercion for the numeric vars | Extend existing config test patterns |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 11.A | All 6 new config vars present in `app/config.py` with correct types and defaults. | `grep` for each var name returns a match. |
| 11.B | `BUILD_RESUME="true"` and `"1"` → `True`; `"false"` and `"0"` → `False`. | Four parametrized unit test cases. |
| 11.C | `BUILD_BATCH_SIZE` and `BUILD_NLIST` are `int`; `BUILD_SAMPLE_FRAC` is `float`. | `assert isinstance(...)` in unit tests. |
| 11.D | Default for `BUILD_RESUME` is `True` (safe for production restarts). | Unit test asserting default without env override. |

---

## T12 — Wikipedia title downloader (`app/build_index.py` — stage 1)
git branch: `ai-agent/t12_wikipedia_title_downloader`

Implement the download-and-extract stage of the build pipeline.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 12.1 | Create `app/build_index.py` with a `download_titles(output_path: Path, dump_url: str, resume: bool, progress_cb: Callable[[float], None]) -> int` function | Returns number of titles written; `progress_cb` called with a float 0.0–1.0 so the caller can update `app.state.build_progress` |
| 12.2 | Fetch the Cirrussearch dump (gzip-compressed NDJSON) using `httpx` (streaming) or `urllib.request` — stream the download, do not load the whole file into memory | Use `gzip.open` on the streamed bytes |
| 12.3 | For each JSON line, extract the `"title"` field (skip lines where it is absent or empty) and write to `output_path` (one title per line, UTF-8) | |
| 12.4 | If `resume=True` and `output_path` already exists, skip download entirely, log an INFO skip message, and call `progress_cb(1.0)` | Resumability per PRD §7.4 |
| 12.5 | Log progress every 100k titles at INFO level | Operator visibility during multi-hour run |
| 12.6 | Write unit tests in `tests/test_build_download.py` with a tiny synthetic NDJSON fixture (10 lines, 3 valid titles, 2 empty, 5 missing key) — mock HTTP, no network access | |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 12.A | Returns exact count of non-empty titles written. | Unit test: return value == 3 from 10-line fixture. |
| 12.B | Lines without `"title"` or with empty string are skipped. | Unit test: output file has exactly 3 lines. |
| 12.C | `resume=True` + existing file → HTTP client not called; `progress_cb(1.0)` called. | Mock HTTP call count == 0; progress_cb call args asserted. |
| 12.D | Download is streamed (`httpx.stream` or equivalent), not loaded into memory in one call. | Code review. |
| 12.E | Coverage on download stage ≥ 95%. | CI coverage report. |

---

## T13 — Batch embedding generator (`app/build_index.py` — stage 2)
git branch: `ai-agent/t13_batch_embedding_generator`

Implement the embedding stage: read titles from disk, encode in batches, write float32 vectors to a memory-mapped numpy array.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 13.1 | Add `generate_embeddings(titles_path: Path, embeddings_path: Path, model_name: str, batch_size: int, resume: bool, progress_cb: Callable[[float], None]) -> int` to `app/build_index.py` | Returns total titles embedded |
| 13.2 | Open `embeddings_path` as `numpy.memmap` with shape `(n_titles, EMBED_DIM)` dtype `float32` — allocate based on line count of `titles_path` | Avoids holding all vectors in RAM |
| 13.3 | Read titles in chunks of `batch_size`, call `SentenceTransformer.encode(batch, normalize_embeddings=True)`, write each batch into the correct memmap slice; call `progress_cb` after each batch | L2 normalisation converts cosine to inner-product search |
| 13.4 | If `resume=True` and `embeddings_path` exists with size == `(n_titles * EMBED_DIM * 4)` bytes, skip encoding, log INFO skip, call `progress_cb(1.0)` | Resumability |
| 13.5 | Log progress every 10 batches at INFO level | |
| 13.6 | Write unit tests in `tests/test_build_embed.py` mocking `SentenceTransformer.encode` to return deterministic float32 arrays — no real model loaded | |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 13.A | Output is shape `(n_titles, 384)` dtype `float32`. | `assert arr.shape == (n_titles, 384)` and `assert arr.dtype == np.float32`. |
| 13.B | Each row is L2-normalised (`np.linalg.norm(arr[i]) ≈ 1.0`, tol 1e-5). | Unit test iterating all rows. |
| 13.C | `resume=True` + matching file → encode mock not called; `progress_cb(1.0)` called. | Mock call count == 0; progress_cb args asserted. |
| 13.D | Partial file (wrong size) bypasses resume skip, re-encodes. | Unit test with partial file. |
| 13.E | Coverage on embedding stage ≥ 95%. | CI coverage report. |

---

## T14 — FAISS index builder (`app/build_index.py` — stage 3)
git branch: `ai-agent/t14_faiss_index_builder`

Implement the FAISS index construction stage: train on a sample, add all vectors, write index and manifest.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 14.1 | Add `build_faiss_index(embeddings_path, titles_path, index_path, manifest_path, nlist, sample_frac, resume, progress_cb) -> dict` to `app/build_index.py` | Returns manifest dict |
| 14.2 | Load vectors from memmap; draw `int(n * sample_frac)` random rows to train `faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)` | Inner-product + L2-norm = cosine similarity |
| 14.3 | After training, `index.add(all_vectors)` in batches of 100k; call `progress_cb` per batch | |
| 14.4 | Write index with `faiss.write_index` to a `*.tmp` file, then atomically rename to `index_path` | Prevents corrupt index being read on crash |
| 14.5 | Write `build_manifest.json` atomically (tmp → rename) containing: `built_at`, `title_count`, `model_name`, `nlist`, `embed_dim` | Sentinel file for resume logic |
| 14.6 | If `resume=True` and `manifest_path` exists, skip and return the existing manifest; call `progress_cb(1.0)` | |
| 14.7 | Write unit tests in `tests/test_build_faiss.py` using a tiny in-memory fixture (10 vectors, nlist=2), mocking `faiss.write_index` and `Path.rename` | No disk I/O required |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 14.A | Integration smoke test: both `index_path` and `manifest_path` exist after call. | `assert index_path.exists() and manifest_path.exists()`. |
| 14.B | Index write is atomic: `*.tmp` is renamed to final path. | Code review. |
| 14.C | Manifest contains all required keys. | `assert set(manifest.keys()) == {"built_at", "title_count", "model_name", "nlist", "embed_dim"}`. |
| 14.D | `resume=True` + existing manifest → `faiss.write_index` not called. | Mock call count == 0. |
| 14.E | Coverage on FAISS build stage ≥ 95%. | CI coverage report. |

---

## T15 — Build pipeline orchestration (`app/build_index.py`)
git branch: `ai-agent/t15_build_pipeline_orchestration`

Wire stages T12–T14 into a single `run()` function that updates a shared progress state and is designed to be called from a background thread.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 15.1 | Add `run(state, config)` to `app/build_index.py` where `state` is a simple dataclass/object with fields `build_status: str`, `build_progress: float`, `build_error: str \| None` — updated in-place as the pipeline advances | `state` is the same object attached to `app.state` in T16; thread-safe writes are safe here because only one thread writes and FastAPI only reads |
| 15.2 | `run()` calls `download_titles → generate_embeddings → build_faiss_index` in order, mapping each stage's `progress_cb` to a slice of the overall 0.0–1.0 range: download = 0.0–0.3, embed = 0.3–0.8, faiss = 0.8–0.95, load into memory = 0.95–1.0 | Progress ranges are approximate; the important thing is monotonic increase |
| 15.3 | After all three stages, `run()` loads the completed index and titles into `state` (same fields used by `/search`) and sets `state.build_status = "ready"` | No restart required |
| 15.4 | On any unhandled exception, `run()` sets `state.build_status = "error"` and `state.build_error = str(e)`, then returns (does not re-raise — the thread must not crash silently) | `/health` will expose the error; operator can restart the container |
| 15.5 | Write unit tests in `tests/test_build_run.py` patching all three stage functions; assert call order, progress values written to state, and that `state.build_status == "ready"` at the end | Also test exception path: patch `download_titles` to raise, assert `state.build_status == "error"` |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 15.A | `run()` calls stages in order; `state.build_progress` increases monotonically. | Unit test asserting call order and progress values after each mocked stage. |
| 15.B | `state.build_status == "ready"` and `state.index is not None` after successful run. | Unit test. |
| 15.C | Exception in any stage → `state.build_status == "error"`, `state.build_error` is set, thread does not raise. | Unit test: `download_titles` raises `RuntimeError`; assert state after `run()` returns. |
| 15.D | Progress value after each stage falls within its assigned range (download ≤ 0.3, embed ≤ 0.8, faiss ≤ 0.95, final == 1.0). | Unit test asserting progress snapshots. |
| 15.E | Coverage on `run()` ≥ 95%. | CI coverage report. |

---

## T16 — FastAPI lifespan integration
git branch: `ai-agent/t16_fastapi_lifespan_integration`

Integrate the build pipeline into the FastAPI lifespan so the container self-initializes on first start while immediately serving `/health`.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 16.1 | In `app/main.py`, update the `lifespan()` context manager: on startup, check if `build_manifest.json` exists at the configured path | If it exists → load index + titles into `app.state`, set `build_status = "ready"`. If not → set `build_status = "building"`, spawn `threading.Thread(target=build_index.run, args=(app.state, config), daemon=True).start()` |
| 16.2 | Attach a `BuildState` dataclass to `app.state` with fields: `build_status: str`, `build_progress: float`, `index`, `titles`, `build_error: str \| None` — initialised to `("building", 0.0, None, None, None)` at startup | Single source of truth read by both `/health` and `/search` |
| 16.3 | Update `GET /health` to always return `200 OK` with the current `BuildState` fields: `status`, `progress`, `titles_loaded` (len of titles list or null), `error` | Per PRD §6 health spec |
| 16.4 | Update `GET /search` to return `503` with `{"detail": "Index is still building", "status": state.build_status, "progress": state.build_progress}` when `state.build_status != "ready"` | Per PRD §6 search spec |
| 16.5 | Write unit tests in `tests/test_main_lifespan.py` covering: (a) lifespan with existing manifest → no thread spawned, status ready; (b) lifespan without manifest → thread spawned, status building; (c) `/health` returns 200 in both cases; (d) `/search` returns 503 while building, 200 when ready | Use `AsyncClient` from `httpx` and mock `build_index.run` and `threading.Thread` |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 16.A | `GET /health` returns `200 OK` immediately after startup regardless of build state. | Unit test: assert status code 200 before mocked build thread completes. |
| 16.B | `GET /search` returns `503` while `build_status == "building"`; returns `200` once `build_status == "ready"`. | Two unit test cases. |
| 16.C | When manifest exists at startup, no background thread is spawned (index loaded synchronously). | Mock `threading.Thread`; assert it is **not** called when manifest exists. |
| 16.D | When manifest is absent, `threading.Thread` is spawned with `build_index.run` as target. | Mock `threading.Thread`; assert it is called once with the correct target. |
| 16.E | `/health` response body matches PRD §6 schema: `status`, `progress`, `titles_loaded`, and (when errored) `error`. | Unit test asserting response JSON keys. |

---

## T17 — Docker Compose & volume update
git branch: `ai-agent/t17_docker_compose_volume`

Update `docker-compose.yaml` to add the named volume. All configuration lives in `.env` — no `environment:` block in the compose file.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 17.1 | Replace `docker-compose.yaml` with the version below — single `wiki-search` service, `env_file: .env`, named volume `wiki-data`, no `environment:` block | All vars come from `.env`; compose file stays minimal |
| 17.2 | Mount the volume read-write at `/data` (the service both reads and writes — builds on first start, then reads on subsequent starts) | |
| 17.3 | Replace `.env.example` with the version below — all search and build vars in one file, grouped with comments; operators edit `.env` to change behaviour (e.g. set `BUILD_RESUME=false` to force a full rebuild) | Single source of truth for all config |
| 17.4 | Update the comment block at the top of `docker-compose.yaml` with first-time, subsequent-start, and force-rebuild instructions | Inline docs — no separate README change required |

Target `docker-compose.yaml`:

```yaml
services:
  wiki-search:
    build:
      context: .
      dockerfile: Dockerfile
    image: ghcr.io/dvadell/wiki-search:latest
    env_file:
      - .env
    ports:
      - "8001:8000"
    volumes:
      - wiki-data:/data

volumes:
  wiki-data:
```

Target `.env.example`:

```
# --- Search service ---
MODEL_NAME=all-MiniLM-L6-v2
EMBED_DIM=384
DEFAULT_K=5
DEFAULT_NPROBE=64
PORT=8000
WORKERS=1

# --- Paths (must point inside the mounted volume) ---
FAISS_INDEX=/data/wiki_faiss.index
TITLES_FILE=/data/wiki_titles.txt
BUILD_MANIFEST=/data/build_manifest.json

# --- Index build ---
# Set to false to force a full rebuild from scratch
BUILD_RESUME=true
# Leave empty to use the latest Wikimedia Cirrussearch dump
WIKI_DUMP_URL=
BUILD_BATCH_SIZE=512
BUILD_NLIST=4096
BUILD_SAMPLE_FRAC=0.1
```

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 17.A | `docker-compose.yaml` has exactly one service (`wiki-search`), one named volume (`wiki-data`), and no `environment:` block. | `yq '.services.wiki-search.environment'` is null; `yq '.services \| keys'` == `["wiki-search"]`. |
| 17.B | Port mapping is `8001:8000` (unchanged). | `yq '.services.wiki-search.ports[0]'` == `"8001:8000"`. |
| 17.C | `.env.example` contains all path vars (`FAISS_INDEX`, `TITLES_FILE`, `BUILD_MANIFEST`) pointing to `/data/...`. | `grep "/data/" .env.example` returns 3 matches. |
| 17.D | `.env.example` contains all build vars: `BUILD_RESUME`, `WIKI_DUMP_URL`, `BUILD_BATCH_SIZE`, `BUILD_NLIST`, `BUILD_SAMPLE_FRAC`. | `grep -c "BUILD_" .env.example` == 5. |
| 17.E | No `depends_on`, no `builder` service, no `environment:` block in `docker-compose.yaml`. | `grep -c "depends_on\|builder:\|environment:"` == 0. |

---

## T18 — Kubernetes readiness probe update
git branch: `ai-agent/t18_kubernetes_readiness_probe`

Update `k8s/deployment.yaml` to remove any init container (if present) and configure probes correctly for the self-initializing container.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 18.1 | Remove any `initContainers` section from `k8s/deployment.yaml` if present | Single container, no init container |
| 18.2 | **Liveness probe**: `GET /health`, `initialDelaySeconds: 30`, `periodSeconds: 30`, `failureThreshold: 3` — detects crashes only; always returns 200 so won't restart a healthy-but-building pod | |
| 18.3 | **Readiness probe**: `GET /health` with a response body check for `"status": "ready"` — or use `exec` probe: `python -c "import httpx, sys; r=httpx.get('http://localhost:8000/health'); sys.exit(0 if r.json().get('status')=='ready' else 1)"`. Set `initialDelaySeconds: 60`, `periodSeconds: 60`, `failureThreshold: 600` (10 hours — covers a full build) | Pod stays out of the load balancer until index is loaded; generous threshold avoids killing a building pod |
| 18.4 | Set container resource requests to accommodate both build and serve phases: `cpu: "500m"`, `memory: "16Gi"` (build is RAM-heavy); limits: `cpu: "2000m"`, `memory: "20Gi"` | Document in a YAML comment that the high memory request is for the build phase |
| 18.5 | Set `BUILD_RESUME: "true"` in the container `env:` block so pod restarts (e.g. node eviction) resume rather than restart the build | |
| 18.6 | Add an operator runbook comment explaining: how to force a full rebuild (delete `build_manifest.json` from the PVC and restart the pod), how to watch build progress (`kubectl logs -f <pod>`), and expected first-start duration | Inline YAML comment |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 18.A | No `initContainers` key in `k8s/deployment.yaml`. | `yq '.spec.template.spec.initContainers' k8s/deployment.yaml` is null. |
| 18.B | Liveness probe targets `/health` with `failureThreshold: 3`. | `yq '.spec.template.spec.containers[0].livenessProbe'` matches. |
| 18.C | Readiness probe `failureThreshold` is ≥ 600 (10 hours at 60s period). | `yq '.spec.template.spec.containers[0].readinessProbe.failureThreshold'` ≥ 600. |
| 18.D | `BUILD_RESUME: "true"` in container env. | `yq '.spec.template.spec.containers[0].env'` includes match. |
| 18.E | `kubectl apply --dry-run=client -f k8s/deployment.yaml` exits 0. | CI or manual. |

---

## T19 — Integration test for auto-build on first start
git branch: `ai-agent/t19_integration_test_auto_build`

End-to-end test of the full auto-build flow using a synthetic corpus — no real Wikipedia download.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 19.1 | Create `tests/integration/test_auto_build.py` with a fixture that generates a synthetic Cirrussearch NDJSON file (1000 titles, gzip-compressed) in a temp directory and patches `WIKI_DUMP_URL` to point to it | No network access — fixture is pure Python + a local HTTP server or file URL |
| 19.2 | Start the FastAPI app (via `AsyncClient` + `LifespanManager` or `TestClient`) against the temp data directory; assert `GET /health` returns `200` within 5 seconds of startup | Tests SC2: `/health` available immediately |
| 19.3 | Assert `GET /search?q=test` returns `503` with `{"status": "building"}` while the build is in progress | Tests SC8 |
| 19.4 | Wait for `state.build_status == "ready"` (poll `/health` up to 120 seconds for the synthetic corpus); then assert `GET /search?q=test` returns `200` with results | Tests SC6 |
| 19.5 | Simulate a restart: stop the app, restart it pointing at the same temp dir (manifest exists); assert `GET /search` is available within 60 seconds (no rebuild) | Tests that the manifest sentinel works |
| 19.6 | Add this test to CI as a separate job; it runs on every push to `main` and on every PR | Gate: must pass before merge |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 19.A | `GET /health` returns 200 within 5 seconds of app startup. | `assert response.status_code == 200` in timed test. |
| 19.B | `GET /search` returns 503 immediately after startup (before build completes). | Assert 503 on first search call. |
| 19.C | `GET /search` returns 200 after build completes. | Assert 200 after polling `/health` for `status == "ready"`. |
| 19.D | Restart with existing manifest: `/search` ready in < 60 seconds (index load only, no rebuild). | Timed assertion on second app instance. |
| 19.E | CI job present and required for merge. | Review of `.github/workflows/ci.yaml`. |

---

## T20 — Polish & verification
git branch: `ai-agent/t20_polish_verification`

Final cross-check that all PRD success criteria are met.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket and work on it until completion | Per-ticket prerequisite |
| 20.1 | Confirm SC2: `docker compose up` on a clean machine answers `GET /health` within 30 seconds | Manual or CI with ephemeral volume |
| 20.2 | Confirm SC6: full build (synthetic corpus) completes, manifest written, `/search` returns results | CI integration test from T19 |
| 20.3 | Confirm SC7: `BUILD_RESUME=true` on restart skips all completed stages in < 10 seconds | Timed test |
| 20.4 | Confirm SC8: `/search` returns 503 with `progress` field while building | Integration test from T19 |
| 20.5 | `kubectl apply --dry-run=client -f k8s/` exits 0 for all manifests | CI job |
| 20.6 | Update release notes / CHANGELOG with the new auto-build behaviour, the `BUILD_RESUME` env var, and expected first-start duration | PR description or `CHANGELOG.md` entry |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 20.A | SC2, SC6, SC7, SC8 all green in CI. | CI dashboard. |
| 20.B | Single image confirmed: `docker images \| grep wiki` shows one image, not two. | Manual check. |
| 20.C | `docker-compose.yaml` has no `depends_on`, no `builder` service, no separate Dockerfile reference. | `grep -c "depends_on\|Dockerfile.builder\|builder:"` == 0. |
| 20.D | CHANGELOG or release notes document the new behaviour. | PR review. |

---

*Document version: 1.1 (2026-06-23) — Replaced two-container Option B with single-image background-thread Option A. Removed Dockerfile.builder ticket; updated T15 (orchestration), T16 (lifespan integration), T17 (compose — single service), T18 (k8s probes, no init container), T19 (integration test), T20 (polish).*
