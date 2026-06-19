# PRD — Wikipedia Semantic Search Microservice

## 1. Overview

A lightweight, self-contained microservice that performs **semantic (vector similarity) search** over a snapshot of full English Wikipedia page titles. Given a natural-language query, the service returns the most semantically related title(s) from the index in ~10 ms — fast enough for use as a lookup utility or recommendation front-end.

> **Note:** The current implementation searches *titles only*. Searching within article body content is explicitly out of scope for this PRD but may be considered in a future iteration.

## 2. Problem Statement

Users need an instant way to find the most relevant Wikipedia pages for any query concept (topic, phrase, or question). Full-text search misses semantic intent ("plant energy conversion" ≠ "photosynthesis"). A vector-indexed title store solves this gap with near-instantaneous response times (~10 ms vs. ~10 s per query at inference scale via FAISS + ONNX Runtime).

## 3. Goals

| # | Goal |
|---|------|
| G1 | Serve semantic search over ~5–10M Wikipedia titles from a local FAISS index in ≤ 20 ms p95 |
| G2 | Containerize the service for portability: Docker Compose (local dev) and Kubernetes production |
| G3 | Automate image build & publish via GitHub Actions (GHCR) on every version tag push |
| G4 | Provide a minimal but professional developer experience — one-line start (`docker-compose up`) |

## 4. Non-Goals

| # | Scope |
|---|------|
| NG1 | Index updates / data freshness pipeline (handled in a separate PRD) |
| NG2 | Article body / content search |
| NG3 | Authentication, RBAC, or rate limiting |
| NG4 | Horizontal scaling beyond 1–2 replicas |
| NG5 | Monitoring stack (Prometheus/Grafana), tracing, etc. |

## 5. Architecture Overview

```
┌──────────────┐    HTTP/REST     ┌──────────────────────────┐
│   Clients      │ ──────────────► │   FastAPI (uvicorn)      │
│   curl / UI    │                 │                          │
└──────────────┘                   │  ┌──────────────────┐    │
                                   │  │ app/main.py       │    │
                                   │  │   lifespan():     │    │
                                   │  │   GET /search     │    │
                                   │  │   GET /health     │    │
                                   │  │   logging.config  │    │
                                   │  └────────┬─────────┘    │
                                   │           │              │
                                   │  ┌────────▼─────────┐    │
                                   │  │ In-memory state:  │    │
                                   │  │ • SentenceTransform│    │
                                   │  │   rModel          │    │
                                   │  │ • FAISS index     │    │
                                   │  │ • titles list     │    │
                                   │  └────────◄─────────┘    │
                                   └──────────────────────────┘
                                             │
                               ┌───────────────────────────────┐
                               │  wiki_faiss.index (mounted)   │
                               │  wiki_titles.txt              │
                               │  model cache (~90 MB on disk) │
                               └───────────────────────────────┘

```

- **Runtime**: Python 3.12 → FastAPI + Uvicorn
- **ML stack**: `sentence-transformers` (all-MiniLM-L6-v2, 384-dim, ~90 MB)
- **Vector store**: FAISS v1.8 (IVF index on local disk file)
- **Deployment**: Docker (multi-stage build, non-root user, healthcheck) → GitHub Actions → GHCR → Kubernetes deployment manifest

## 6. API Specification

### `GET /search`

| Parameter | Type   | Required | Description                | Default | Constraints      |
|-----------|--------|----------|----------------------------|---------|------------------||
| q         | string | **Yes**  | The search query           | —       | Must be non-empty|
| k         | int    | No       | Number of results to return| 5       | 1 ≤ k ≤ 100      |
| nprobe    | int    | No       | FAISS IVF cells to examine | 64      | 1 ≤ nProbe ≤ 4096|

**Success response — `200 OK`:***json*

```json
{
  "query": "how do plants convert sunlight",
  "results": [
    {"rank":     1,  "title": "Photosynthesis",
                           "score": 0.874},
    {"rank":     2,  "title": "Light-dependent reactions","score": 0.762},
    {"rank":     3,  "title": "Photorespiration"
                        ,
```json
{
  "query": "plant energy conversion",
  "results": [
    {"rank": 1, "title": "Photosynthesis",
```

---

## 7. Architecture & Design Decisions

### 7.1 Technology Choices

| Component | Choice | Rationale |
|-----------|--------|------------||
| Web framework? fastapi | FastAPI + uvicorn | Modern, auto-generated OpenAPI docs, async support |
| Model | `all-MiniLM-L6-v2` (384-dim) | Small footprint (~90 MB), fast inference, good quality for title lookup |
| Index storage | FAISS IVF4096 on disk | Fast exact+approximate search at scale; single file portability |
| Language | Python 3.12 | Rich ML tooling ecosystem; `sentence-transformers` is Python-first |

### 7.2 Constraints

- Index loads once into memory at process start (~10–30 seconds depending on disk)
- Memory footprint is dominated by the index + titles (rough estimate varies with data size)
- Model weights are cached in ~/.cache/torch/sentence_transformers/ on first load and then re-used across restarts |

## 8. Deployment Strategy |

### 8.1 Docker Compose (Local Development

*   `docker-compose.yaml` with:
    *   Build context pointing to current directory
    *   Port mapping (host:8000 → container:8000)
    *   Volume mount for data files (`./data:/app/data`) |
    *   Healthcheck probe on `/health" endpoint
    *   Restart policy (`unless-stopped')
*   Single command start: `docker-compose up`

### 8.2 Kubernetes (Production)

*   Single replica deployment with stateless pod spec
*   Liveness/readiness probes against `/health"` endpoint
*   Pod resource limits (requests: 256Mi/100m, limits: 1Gi/500m)
*   Image pulled from GHCR on each deployment
*   HorizontalPodAutoscaler disabled intentionally (low scale target)

### 8.3 CI/CD (GitHub Actions)

*   Workflow `.github/workflows/build-and-push.yml`:
    *   Build Docker image using GitHub Actions' Docker build actions |
    *   Push to GHCR (ghcr.io/{owner}/{repo}) on `release/v*.*.*` tags (and main branch for dev builds)
    *   Trigger: Manual + auto-payload push to `main`.

## 9. Configuration

All runtime configuration via environment variables with sensible defaults:|

Var | Default | Description |
------|-------|-------------|| MODEL_NAME | "all-MiniLM-L6-v2" | Sentence transformer model name (HF hub) |
EMBED_DIM | 384 | Embedding dimensionality (must match the model)| FAISS_INDEX | "wiki_faiss.index" | Path to FAISS index file
TITLES_FILE | "wiki_titles.txt" | Path to Wikipedia titles list
DEFAULT_K | 5 | Default top-k results
DEFAULT_NPROBE | 64 | FAISS IVF search precision (cells)
PORT | 8000 | HTTP listen port
WORKERS | 1 | Uvicorn worker count |

## 10. Security Considerations

- Container runs as non-root user |
- No external network calls at runtime (all ML model is pre-loaded and cached in ~/.cache/torch/sentence_transformers) |
- Input validation via FastAPI query parameter constraints (max k=100, min/max nprobe range) to prevent denial-of-service or out-of-bounds index access.

## 11. Success Criteria

| # | Criterion | Target | Measurement |
|---|-----------|--------|-------------|
| SC1 | Service builds Docker image with no errors in < 5 minutes | Achieved | GitHub Actions logs |
| SC2 | `docker-compose up` starts service and responds to `/health` in < 30s | Achieved | Manual test
SC3 | Semantic search queries return in ~ 10 ms (on SSD) | Achieved| `curl -s "http://localhost:8000/search?q=test"` timing measurement |
| SC4 | GitHub Action successfully publishes image to GHCR on version tag push | Achieved | Logs of `v*.*.*/` release tag
SC5 | Kubernetes deployment (via manifest from PRD) starts, passes probes with no restarts | Achieved| kubectl get pods/status log

## 12. Risks & Mitigation


| Risk | Likelihood | Impact | Mitigation |
|------|----------|--------|------------|| Index load time at startup | High | Medium | Document expected cold-start time in; add `/health` endpoint for readiness probes |
Model download on first boot | Medium | Low (once) | Document the need to pre-cache model weights or use HF_MODEL_CACHE_DIR env var; add `--no-download` fallback once cached |
Memory usage grows with model loading at each container restart | Medium | Low | Cache model weights to persistent volume mount; document `--reuse-model` flag if needed |

## 13. Open Questions / Future Work

| # | Question | Status |
|---|----------|--------|
| OQ1 | Article/body search support? | Consider in v2 PRD |
| OQ2 | Index rebuild automation (`wiki_search.py --build-index`)? | Out of scope for current scope — add CI pipeline or cron job to update the FAISS index. |

---

*Document version: 1.0 (2026-06-19) — Initial drafting*