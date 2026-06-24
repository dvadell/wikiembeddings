# PRD — Wikipedia Semantic Search Microservice

## 1. Overview

A lightweight, self-contained microservice that performs **semantic (vector similarity) search** over a snapshot of full English Wikipedia page titles. Given a natural-language query, the service returns the most semantically related title(s) from the index in ~10 ms — fast enough for use as a lookup utility or recommendation front-end.

> **Note:** The current implementation searches *titles only*. Searching within article body content is explicitly out of scope for this PRD but may be considered in a future iteration.

## 2. Problem Statement

Users need an instant way to find the most relevant Wikipedia pages for any query concept (topic, phrase, or question). Full-text search misses semantic intent ("plant energy conversion" ≠ "photosynthesis"). A vector-indexed title store solves this gap with near-instantaneous response times (~10 ms vs. ~10 s per query at inference scale via FAISS + ONNX Runtime).

Previously, the service assumed the FAISS index and titles file were already present on disk — requiring manual, out-of-band preparation before the service could start. This PRD adds a first-class **initialization (build) stage** that runs automatically on first startup: the service detects that the index is missing, builds it in a background thread, and begins serving `/health` immediately while the build is in progress. Once the build completes, `/search` becomes available with no restart required.

## 3. Goals

| # | Goal |
|---|------|
| G1 | Serve semantic search over ~5–10M Wikipedia titles from a local FAISS index in ≤ 20 ms p95 |
| G2 | Containerize the service for portability: Docker Compose (local dev) and Kubernetes production |
| G3 | Automate image build & publish via GitHub Actions (GHCR) on every version tag push |
| G4 | Provide a minimal but professional developer experience — one-line start (`docker-compose up`) |
| G5 | On first start with no index on disk, automatically download Wikipedia titles, build the FAISS index in the background, and begin serving search queries — all from a **single image and a single container**, with no manual steps |

## 4. Non-Goals

| # | Scope |
|---|------|
| NG1 | Scheduled/automated index updates / data freshness pipeline (handled in a separate PRD) |
| NG2 | Article body / content search |
| NG3 | Authentication, RBAC, or rate limiting |
| NG4 | Horizontal scaling beyond 1–2 replicas |
| NG5 | Monitoring stack (Prometheus/Grafana), tracing, etc. |
| NG6 | Separate builder image or builder container — one image serves both roles |

## 5. Architecture Overview

```
┌──────────────┐    HTTP/REST     ┌─────────────────────────────────────┐
│   Clients    │ ──────────────► │   FastAPI (uvicorn)                  │
│   curl / UI  │                 │                                      │
└──────────────┘                 │  ┌───────────────────────────────┐   │
                                 │  │ app/main.py                    │   │
                                 │  │   lifespan():                  │   │
                                 │  │     if index missing:          │   │
                                 │  │       spawn build thread       │   │
                                 │  │     else:                      │   │
                                 │  │       load index into memory   │   │
                                 │  │   GET /search  (ready when     │   │
                                 │  │     index loaded)              │   │
                                 │  │   GET /health  (always ready)  │   │
                                 │  └────────────┬──────────────────┘   │
                                 │               │                      │
                                 │  ┌────────────▼──────────────────┐   │
                                 │  │ In-memory state (app.state):   │   │
                                 │  │  • build_status: str           │   │
                                 │  │    "building" | "ready" |      │   │
                                 │  │    "error"                     │   │
                                 │  │  • build_progress: float 0–1   │   │
                                 │  │  • model: SentenceTransformer  │   │
                                 │  │  • index: faiss.Index | None   │   │
                                 │  │  • titles: list[str] | None    │   │
                                 │  └───────────────────────────────┘   │
                                 └─────────────────────────────────────┘
                                                 │
                                 ┌───────────────▼─────────────────────┐
                                 │  Shared data volume                  │
                                 │  wiki_faiss.index  (written once)    │
                                 │  wiki_titles.txt   (written once)    │
                                 │  build_manifest.json                 │
                                 │  model cache (~90 MB)                │
                                 └─────────────────────────────────────┘
```

- **Runtime**: Python 3.12 → FastAPI + Uvicorn
- **ML stack**: `sentence-transformers` (all-MiniLM-L6-v2, 384-dim, ~90 MB)
- **Vector store**: FAISS v1.8 (IVF index on local disk file)
- **Index build**: `app/build_index.py` — module called from a background thread inside the same process on first startup
- **Deployment**: Docker (single image, non-root user, healthcheck) → GitHub Actions → GHCR → Kubernetes deployment manifest

## 6. API Specification

### `GET /search`

| Parameter | Type   | Required | Description                | Default | Constraints       |
|-----------|--------|----------|----------------------------|---------|-------------------|
| q         | string | **Yes**  | The search query           | —       | Must be non-empty |
| k         | int    | No       | Number of results to return| 5       | 1 ≤ k ≤ 100       |
| nprobe    | int    | No       | FAISS IVF cells to examine | 64      | 1 ≤ nProbe ≤ 4096 |

**Success response — `200 OK`:**

```json
{
  "query": "how do plants convert sunlight",
  "results": [
    {"rank": 1, "title": "Photosynthesis",           "score": 0.874},
    {"rank": 2, "title": "Light-dependent reactions", "score": 0.762},
    {"rank": 3, "title": "Photorespiration",          "score": 0.701}
  ]
}
```

**Index not yet ready — `503 Service Unavailable`:**

```json
{"detail": "Index is still building", "status": "building", "progress": 0.42}
```

### `GET /health`

Always returns `200 OK` — even while the index is building. Reflects the current build/serve state so orchestrators and operators can monitor progress.

```json
{"status": "building", "progress": 0.42, "titles_loaded": null}
```

```json
{"status": "ready", "progress": 1.0, "titles_loaded": 5842118}
```

```json
{"status": "error", "progress": null, "error": "Download failed: HTTP 503"}
```

---

## 7. Architecture & Design Decisions

### 7.1 Technology Choices

| Component      | Choice                          | Rationale |
|----------------|---------------------------------|-----------|
| Web framework  | FastAPI + uvicorn               | Modern, auto-generated OpenAPI docs, async support |
| Model          | `all-MiniLM-L6-v2` (384-dim)   | Small footprint (~90 MB), fast inference, good quality for title lookup |
| Index storage  | FAISS IVF4096 on disk           | Fast exact+approximate search at scale; single file portability |
| Language       | Python 3.12                     | Rich ML tooling ecosystem; `sentence-transformers` is Python-first |
| Title source   |  Official Wikipedia titles dump (ns0) | Official snapshot, updated monthly, plain-text title extraction |
| Build execution | Background thread (not async)  | `sentence-transformers` and FAISS are CPU-bound and not async-friendly; `threading.Thread` runs them without blocking the event loop |

### 7.2 Constraints

- Index loads once into memory at process start (~10–30 seconds depending on disk)
- Memory footprint is dominated by the index + titles (rough estimate varies with data size)
- Model weights are cached in `~/.cache/torch/sentence_transformers/` on first load and then re-used across restarts
- **Index build is a long-running, one-time operation** (~2–8 hours for full English Wikipedia depending on hardware). It runs in a background thread and must not block the uvicorn event loop.

### 7.3 Startup Flow

```
Container starts
      │
      ▼
lifespan() runs
      │
      ├─ build_manifest.json exists on volume?
      │         │
      │        YES ──► load index + titles into memory ──► status = "ready"
      │         │
      │        NO  ──► status = "building"
      │                spawn background thread → app/build_index.py::run()
      │                      │
      │                      ├─ stage 1: download titles      (progress 0.0–0.3)
      │                      ├─ stage 2: generate embeddings  (progress 0.3–0.8)
      │                      ├─ stage 3: build FAISS index    (progress 0.8–0.95)
      │                      ├─ stage 4: atomic write + manifest (progress 0.95–1.0)
      │                      └─ load index into memory ──► status = "ready"
      │
      ▼
FastAPI serving (immediately, regardless of build state)
  GET /health   → always 200, reflects status + progress
  GET /search   → 503 while building, 200 when ready
```

### 7.4 Index Build Pipeline Design

The build pipeline lives in `app/build_index.py` and is called from the FastAPI lifespan. It updates `app.state.build_progress` (a `float` 0.0–1.0) as it advances through stages so `/health` can report progress.

Pipeline steps:

1. **Download titles** — fetch `enwiki-latest-all-titles-in-ns0.gz` from Wikimedia — a plain gzip with one title per line — and decompress it into `wiki_titles.txt` (one title per line, UTF-8).
2. **Batch encode** — load `all-MiniLM-L6-v2` and encode titles in configurable batches (default 512), writing float32 embeddings to a memory-mapped numpy array to avoid OOM on large corpora.
3. **Build FAISS index** — train an `IVF4096,Flat` index on a random sample, then add all vectors. L2-normalise before insertion to turn inner-product search into cosine similarity.
4. **Write outputs** — atomically rename temp files to `wiki_faiss.index` and `wiki_titles.txt`; write `build_manifest.json` with timestamp, title count, model name, and index parameters.
5. **Load into memory** — after writing, load the index and titles into `app.state` so `/search` becomes available without a restart.

#### Resumability

If the container is restarted mid-build, the pipeline detects partially completed stages via `BUILD_RESUME=true`. It skips the download step if `wiki_titles.txt` already exists, and skips embedding if the mmap array size matches the expected shape.

## 8. Deployment Strategy

### 8.1 Docker Compose (Local Development)

`docker-compose.yaml` with a **single service**:
- `wiki-search` — the only service; mounts the `wiki-data` named volume at `/data`
- On first `docker compose up`: container starts, FastAPI is immediately up on port 8001, build runs in the background
- On subsequent `docker compose up`: index already on volume, loads in ~10–30s, `/health` goes to `ready`
- No builder service, no `depends_on`, no sequencing needed

### 8.2 Kubernetes (Production)

- Single replica deployment with stateless pod spec
- **No init container** — the main container self-initializes on first start
- **Liveness probe**: `/health` (always 200) — detects crashes
- **Readiness probe**: custom check against `status == "ready"` in `/health` response — keeps the pod out of the load balancer until the index is loaded; generous `initialDelaySeconds` and `failureThreshold` to accommodate the build duration on first start
- Pod resource limits accommodate both build phase (CPU/RAM heavy) and serve phase (lighter)
- `BUILD_RESUME=true` set by default so pod restarts after a crash resume rather than restart the build from scratch

### 8.3 CI/CD (GitHub Actions)

- Workflow `.github/workflows/build-and-push.yml`:
  - Build Docker image using GitHub Actions' Docker build actions
  - Push to GHCR (`ghcr.io/{owner}/{repo}`) on `release/v*.*.*` tags (and main branch for dev builds)
  - Single image tag — no separate builder image

## 9. Configuration

All runtime configuration via environment variables with sensible defaults:

| Var                  | Default                          | Description |
|----------------------|----------------------------------|-------------|
| MODEL_NAME           | `"all-MiniLM-L6-v2"`            | Sentence transformer model name (HF hub) |
| EMBED_DIM            | `384`                            | Embedding dimensionality (must match the model) |
| FAISS_INDEX          | `"wiki_faiss.index"`            | Path to FAISS index file |
| TITLES_FILE          | `"wiki_titles.txt"`             | Path to Wikipedia titles list |
| DEFAULT_K            | `5`                              | Default top-k results |
| DEFAULT_NPROBE       | `64`                             | FAISS IVF search precision (cells) |
| PORT                 | `8000`                           | HTTP listen port |
| WORKERS              | `1`                              | Uvicorn worker count |
| WIKI_DUMP_URL        | *(Wikimedia latest dump URL)*    | Override to use a specific dump date or mirror |
| BUILD_BATCH_SIZE     | `512`                            | Title embedding batch size during index build |
| BUILD_NLIST          | `4096`                           | FAISS IVF `nlist` parameter (number of clusters) |
| BUILD_SAMPLE_FRAC    | `0.1`                            | Fraction of vectors used to train the IVF quantizer |
| BUILD_RESUME         | `true`                           | Skip completed build stages on restart (default true — safe for production) |
| BUILD_MANIFEST       | `"build_manifest.json"`         | Sentinel file written on successful build completion |

## 10. Security Considerations

- Container runs as non-root user
- Build makes one external network call (Wikipedia dump download) on first start only; no external calls at runtime after index is loaded
- Input validation via FastAPI query parameter constraints (max k=100, min/max nprobe range)

## 11. Success Criteria

| # | Criterion | Target | Measurement |
|---|-----------|--------|-------------|
| SC1 | Service builds Docker image with no errors in < 5 minutes | Achieved | GitHub Actions logs |
| SC2 | `docker-compose up` starts service and responds to `GET /health` with `200 OK` in < 30s — even before the index is built | Achieved | `curl http://localhost:8001/health` immediately after `docker compose up` |
| SC3 | Semantic search queries return in ~10 ms (on SSD) once index is ready | Achieved | `curl -s "http://localhost:8001/search?q=test"` timing measurement |
| SC4 | GitHub Action successfully publishes a single image to GHCR on version tag push | Achieved | Logs of `v*.*.*/` release tag |
| SC5 | Kubernetes deployment starts, liveness probe passes immediately, readiness probe passes once index loads | Achieved | `kubectl get pods` shows Running; endpoints populated after build |
| SC6 | On first start with empty volume: build completes, `build_manifest.json` written, `/health` transitions to `{"status": "ready"}`, `/search` returns results | Achieved | End-to-end test with synthetic corpus |
| SC7 | On restart mid-build: `BUILD_RESUME=true` resumes from last completed stage, does not restart from scratch | Achieved | Manual test — interrupt container, restart, verify stage-skip log messages |
| SC8 | `/search` returns `503` with `{"status": "building", "progress": <float>}` while index is building | Achieved | Integration test asserting 503 before build completes |

## 12. Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Readiness probe fails during long build | High | Low | Use `/health` for liveness (always 200) and a separate readiness check on `status == "ready"`; set generous `failureThreshold` |
| Memory pressure during index build (full corpus) | High | High | Memory-mapped numpy array for embeddings; batch encode; document minimum RAM (≥16 GB recommended for build phase) |
| Build thread crashes silently | Medium | High | `app.state.build_status = "error"` on exception; `/health` exposes the error message; liveness probe remains green (process is alive) but operators see the error |
| Build interrupted mid-run (pod eviction, OOM kill) | Medium | Medium | `BUILD_RESUME=true` by default; pipeline resumes from last completed stage on next start |
| Wikipedia dump URL changes / unavailable | Low | Medium | `WIKI_DUMP_URL` env var allows pinning to a known-good URL or mirror |
| Build writes corrupt index then crashes | Low | High | Builder writes to `*.tmp` files and atomically renames; a crash before rename leaves the previous good index (or no index) intact |

## 13. Open Questions / Future Work

| # | Question | Status |
|---|----------|--------|
| OQ1 | Article/body search support? | Consider in v2 PRD |
| OQ2 | Scheduled index refresh (monthly Wikipedia dump)? | Out of scope — candidate for v2 pipeline PRD |
| OQ3 | GPU-accelerated embedding during build? | Not required at current scale; revisit if build time > 8 hours |
| OQ4 | SSE progress stream endpoint (`GET /build/progress`)? | Nice-to-have; `/health` progress field covers the basic case |

---

*Document version: 1.2 (2026-06-23) — Replaced two-container builder approach with single-image, background-thread auto-build on first start (Option A). Updated §2, §3 G5, §4 NG6, §5 architecture diagram, §6 /health and /search specs, §7.1–7.4, §8.1–8.2, §11 SC2/SC6–SC8, §12 risks.*
