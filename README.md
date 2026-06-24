<div align="center">

# Wiki Search

**Wikipedia semantic search microservice — vector similarity over ~5–10M English titles via FAISS + sentence-transformers.**

[![CI](https://github.com/dvadell/wikiembeddings/actions/workflows/ci.yaml/badge.svg)](https://github.com/dvadell/wikiembeddings/actions/workflows/ci.yaml)
[![Docker Build](https://github.com/dvadell/wikiembeddings/actions/workflows/release.yaml/badge.svg)](https://github.com/dvadell/wikiembeddings/actions/workflows/release.yaml)

</div>

## Overview

Instant semantic search over a local index of Wikipedia page titles. Given a natural-language query, the service returns the most relevant titles in ~10 ms — powered by `all-MiniLM-L6-v2` embeddings and a FAISS IVF index.

First start with no index? The container downloads Wikipedia titles, builds the embedding index in the background, and becomes fully operational — zero manual steps.

> **95 % code coverage** enforced on every PR via CI. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Quickstart

```bash
# 1. Clone & start (first start triggers a ~2–8 h index build)
docker compose up -d

# 2. Watch progress
curl http://localhost:8001/health

# 3. Search once ready
curl "http://localhost:8001/search?q=photosynthesis&k=5"
```

See [`.env.example`](.env.example) for configurable environment variables. Copy it to `.env` and edit as needed.

## API Reference

### `GET /search` — Semantic search

| Parameter | Required | Type   | Default | Description            | Constraints       |
|-----------|----------|--------|---------|------------------------|-------------------|
| `q`       | Yes      | string | —       | Search query           | Non-empty          |
| `k`       | No       | int    | 5       | Number of results      | 1 ≤ k ≤ 100       |
| `nprobe`  | No       | int    | 64      | FAISS IVF cells to examine | 1 ≤ nprobe ≤ 4096 |

**Success (200):**

```json
{
  "query": "photosynthesis",
  "results": [
    {"rank": 1, "title": "Photosynthesis",        "score": 0.874},
    {"rank": 2, "title": "Light-dependent reactions", "score": 0.762}
  ]
}
```

**Building (503):**

```json
{"detail": "Index is still building", "status": "building", "progress": 0.42}
```

### `GET /health` — Service health

Always returns `200 OK`:

```json
{"status": "ready", "progress": 1.0, "titles_loaded": 5842118}
```

## Configuration

All configuration is via environment variables (see [`.env.example`](.env.example)):

| Variable           | Default                  | Description                                      |
|--------------------|--------------------------|--------------------------------------------------|
| `MODEL_NAME`       | `all-MiniLM-L6-v2`      | Sentence-transformers model                      |
| `EMBED_DIM`        | `384`                    | Embedding dimensionality                         |
| `FAISS_INDEX`      | `/data/wiki_faiss.index` | Path to FAISS index file                         |
| `TITLES_FILE`      | `/data/wiki_titles.txt`  | Path to Wikipedia titles list                    |
| `DEFAULT_K`        | `5`                      | Default top-k results                            |
| `DEFAULT_NPROBE`   | `64`                     | FAISS IVF search precision                       |
| `PORT`             | `8000`                   | HTTP listen port                                 |
| `WORKERS`          | `1`                      | Uvicorn worker count                             |

### Index build variables

| Variable         | Default       | Description                                      |
|------------------|---------------|--------------------------------------------------|
| `BUILD_RESUME`   | `true`        | Skip completed stages on restart                 |
| `WIKI_DUMP_URL`  | *(auto)*      | Override Wikipedia dump URL                      |
| `BUILD_BATCH_SIZE` | `512`       | Title embedding batch size                       |
| `BUILD_NLIST`    | `4096`        | FAISS IVF cluster count                          |
| `BUILD_SAMPLE_FRAC` | `0.1`      | Fraction of vectors for IVF training             |
| `BUILD_MANIFEST` | `build_manifest.json` | Sentinel file for build completion detection |

## Kubernetes Deploy

Apply the manifests in [`k8s/`](k8s/) to your cluster:

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl get pods -l app=wiki-search
```

The deployment uses a single replica with:

- **Liveness probe** on `/health` (always 200 → crash detection)
- **Readiness probe** on `/health` waiting for `status == ready`: keeps the pod out of service until build completes
- Requested resources: 100 m CPU, 256 Mi memory; limits: 500 m CPU, 1 Gi memory

## Documentation

- [PRD](PRD.md) — Product requirements & architecture
- [TICKETS](TICKETS.md) — Implementation tickets & acceptance criteria
