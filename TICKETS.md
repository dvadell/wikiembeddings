# Tickets — Wikipedia Semantic Search Microservice

> **Reference:** [PRD.md](./PRD.md)

## Pre-requisite

1. Create a new GitHub repository for this project (MIT licensed, Python 3.12 baseline).

---

## Global acceptance criteria (all tickets)

Every ticket must satisfy all of the following before its PR can be merged:

| # | Criterion | How verified |
|---|-----------|--------------|
| G1 | **`ruff format`** — all modified Python files are formatted with `ruff format`. | CI step or pre-commit hook |
| G2 | **Tests pass** — the full pytest suite exits with code 0 on Python 3.12 (`ubuntu-latest`). | `pytest tests/` in CI |
| G3 | **`ruff check`** passes on all modified and new Python files (app/\+tests/\+any new packages). | `ruff check app/ tests/` in CI |
| G4 | Coverage stays at or above 95 % line coverage for the `app/` package. | `--cov-fail-under=95` in CI |

> If a ticket only touches non-Python files (e.g. k8s manifests, docker-compose), criteria G2 and G4 are waived but G1–G3 still apply to any associated scripts or test tooling.

---

## T0 — Testing infrastructure (must be done first)
git branch: `ai-agent/t0_testing_infrastructure`

Tests are required and must pass on every push to `main`. Target: **>95 % line coverage** enforced by CI.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t0_testing_infrastructure`) and work on it until completion | Per-ticket prerequisite |
| 0.1 | Add `pytest`, `pytest-asyncio`, `httpx`, `coverage[toml]` to dev deps | FastAPI test client (`httpx.AsyncClient`) is the core tooling |
| 0.2 | Create `tests/` directory with `__init__.py`, `conftest.py`, and initial structure | See T0 section below for layout |
| 0.3 | Configure `coverage` via `pyproject.toml` (or `tox.ini`) — set `min_line = 95` in `[tool.coverage.run]` and enable `branch = true` so CI gates on the 95 % threshold | Any PR that drops coverage below 95 % will fail CI |
| 0.4 | Write a CI-only pytest run (`python -m pytest --cov=app --cov-report=term-missing tests/`) alongside the build matrix | Do **not** run index-search endpoint tests here (they require FAISS + titles files) — save those for T5 (integration) |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 0.A | `coverage[toml]` is listed under `[project.optional-dependencies]\ndev = …` in `pyproject.toml` and installing it with `uv sync --group dev` succeeds. | `uv run pytest --collect-only tests/` lists >0 tests without import errors |
| 0.B | `tests/__init__.py`, `tests/conftest.py` exist, `conftest.py` contains a live app lifespan fixture or mock helper that lets other test files import from `app`. | Any empty test file in `tests/` such as `test_dummy.py` (with a single `assert True`) passes. |
| 0.C | Coverage config is present: `[tool.coverage.run]` contains `branch = true` and `[tool.coverage.report]` or equivalent sets the fail-under threshold to 95. | `uv run coverage report --fail-under=95` does not exit > 0 when no tests are collected (config itself must parse). |
| 0.D | Running `uv run ruff format tests/ app/ && uv run ruff check tests/ app/ && uv run pytest tests/` succeeds on Python 3.12 (`ubuntu-latest`) in a fresh checkout with no index/search endpoint files referenced by imports. | CI (or manual) execution of the full one-liner above. |

### Test file layout (proposed)

```
tests/
├── __init__.py
├── conftest.py              # fixtures: test app lifespan, temp data dir
├── test_config.py           # validates config loading & env-var overrides
├── test_health_endpoint.py  # mocks state to ensure /health returns correct schema
├── test_search_unit.py      # unit-level: embedding generation, FAISS round-trip (mocked index)
└── test_validation.py       # parameter boundary tests (invalid k/nprobe)
```

---

## T1 — Project restructuring & code quality
git branch: `ai-agent/t1_project_restructuring_code_quality`

| # | Task | Notes | PRD ref |
|---|------|-------|---------||
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t1_project_restructuring_code_quality`) and work on it until completion | Per-ticket prerequisite |
| 1.1 | Create `app/` package (`__init__.py`) | Move all source into this directory |
| 1.2 | Extract config into `app/config.py` | Every PRD §9 env-var (`MODEL_NAME`, `EMBED_DIM`, `FAISS_INDEX`, `TITLES_FILE`, `DEFAULT_K`, `DEFAULT_NPROBE`, `PORT`, `WORKERS`) becomes an `os.environ.get(...)` with documented defaults |
| 1.3 | Replace all `print(...)` calls with Python `logging` | Use `logging.getLogger(__name__)`; configure via config layer so both dev and prod log levels change without code edits |
| 1.4 | Write `.gitignore` | Exclude: `__pycache__/`, `*.egg-info/`, `venv/`, `.env`, `data/*.index`, `.coverage`, `htmlcov/`, `dist/`, `node_modules/` |
| 1.5 | Add a single `requirements.txt` with pinned dependencies (or move to `pyproject.toml`) | Include both runtime (`fastapi`, `uvicorn`, `numpy`, `faiss-cpu`, `sentence-transformers`) and dev (`pytest`, `coverage`, `httpx`, `ruff`) |
| 1.6 | Write tests for every new piece of code (T0 must pass first) | **>95 % line coverage** enforced in CI; any delta that drops below 95 % is a hard fail |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 1.A | `app/__init__.py`, `app/config.py`, and `app/main.py` (or equivalent) exist; no source files remain at the repo root (`wikititles.py` etc. are gone or moved inside `app/`). | `ls *.py` in repo root lists none except `pyproject.toml`, `PRD.md`, `TICKETS.md`. |
| 1.B | Every PRD §9 env-var is surfaced as a config attribute with the documented default. Tests assert both the default and at least one env-override path per var. | Unit tests import each config attr, assert `== expected_default`, set `os.environ` to a distinct value, re-create config, assert `!= original`. |
| 1.C | No `print()` call remains in any `app/` source file. All log output goes through `logging.getLogger(__name__)`. | `grep -rn "print(" app/` returns zero matches (excluding comments/doc-strings). |
| 1.D | `.gitignore` contains every bullet from the task description. A fresh `git status` after `uv sync` shows none of the listed patterns tracked. | `git --no-pager help config` or a dry-run `git add .` and inspect what would be ignored. |
| 1.E | All runtime deps are installable via `pyproject.toml`; installing dev deps also brings in `pytest`, `coverage`, `httpx`, `ruff`. | `uv sync --group dev` succeeds with zero resolution errors. |
| 1.F | Tests covering the new code pass; line coverage on `app/` is ≥ 95 %. | `uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=95 tests/`. |

---

## T2 — Dockerfile (multi-stage, hardened)
git branch: `ai-agent/t2_dockerfile_multi_stage_hardened`

PRD §8.1 target.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t2_dockerfile_multi_stage_hardened`) and work on it until completion | Per-ticket prerequisite |
| 2.1 | Write `Dockerfile` with **two stages**: builder (`python:3.12-slim`, installs deps) → runtime (`python:3.12-alpine`, non-root user, copies only build artifacts) | Reduces final image size; multi-stage enforces this |
| 2.2 | Add a Docker `HEALTHCHECK` polling the `/health` endpoint every 30 s | Mirrors PRD SC2 requirement |
| 2.3 | Run container as non-root (`RUN addgroup -S app && adduser -S app -G app`) | PRD §10 — no root in the image |
| 2.4 | Set `COPY` for `.index`, `*.txt`, and `app/` separately to maximise layer cache | Keeps rebuild fast when only code changes |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 2.A | The Dockerfile has exactly **two** `FROM` stages named `builder` and `runtime` (or equivalent semantic names). | `grep -c "^FROM" Dockerfile == 2`. |
| 2.B | The runtime stage uses `python:3.12-alpine` as its base image. | `grep "alpine" Dockerfile` matches the runtime FROM instruction. |
| 2.C | A `HEALTHCHECK` instruction exists that hits `/health` with an interval of ≤ 30 s. | `docker build -t wiki-search-test . && docker run --rm --entrypoint healthcheck …` or inspect via `docker image inspect`. |
| 2.D | The container runs as a non-root user called `app` (uid ≠ 0). Inspecting the running container shows `whoami app`. | `docker run --rm wiki-search-test whoami` outputs `app`. |
| 2.E | `COPY` instructions for index files, text files, and `app/` appear as three separate layers (not a single `COPY . /`). | Manual review of Dockerfile — distinct COPY lines. |

---

## T3 — docker-compose.yaml (local dev)
git branch: `ai-agent/t3_docker_compose_yaml_local_dev`

PRD §8.0 target.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t3_docker_compose_yaml_local_dev`) and work on it until completion | Per-ticket prerequisite |
| 3.1 | Create `docker-compose.yaml` with service `wiki-search`; build context: `.`; port map `8000:8000` | PRD SC2 |
| 3.2 | Mount volumes for data files (`./wiki_faiss.index:/app/wiki_faiss.index`, `./wiki_titles.txt:/app/wiki_titles.txt`) | Allows dev to swap dumps without rebuilding image |
| 3.3 | Set environment variables (model name, index path, port) with file overrides via env-file mechanism — or at least document how to override them | PRD §9 |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 3.A | A `docker-compose.yaml` (or `docker-compose.yml`) exists in the repo root with a service named `wiki-search`. | `yq '.services/wiki-search' docker-compose.yaml` returns non-empty. |
| 3.B | Port mapping `8000:8000` is declared under `ports`. | `grep "8000" docker-compose.yaml` confirms both host and container sides. |
| 3.C | Volumes section mounts `./wiki_faiss.index:/app/wiki_faiss.index` and `./wiki_titles.txt:/app/wiki_titles.txt` (or config-driven paths matching PRD §9). | `yq '.services/wiki-search.volumes'` contains both mount entries. |
| 3.D | Environment variables for all 8 PRD §9 vars are documented in the compose file as defaults (e.g. `environment:` block) **and** an `.env.example` is provided to show how to override them. | Manual review; both a compose `environment:` section and an example env-file exist. |

---

## T4 — CI: build + test workflow (`.github/workflows/`)
git branch: `ai-agent/t4_ci_build_test_workflow`

PRD §8.3 target. **Tests run on every `main` push.**

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t4_ci_build_test_workflow`) and work on it until completion | Per-ticket prerequisite |
| 4.1 | Create `.github/workflows/ci.yaml`**:** triggers: `push`, `pull_request` to `main`; jobs **(a) unit tests + coverage, (b) Docker build (no push)** | Run the full pytest suite with coverage; gate PR merge on passing CI. The final image is pushed only on tags (see T5) |
| 4.2 | Add a job matrix: Python 3.12 (`ubuntu-latest`) — run `pytest --cov=app --cov-fail-under=95 tests/` | **>95 % line coverage** is the PRD requirement; CI fails if below |
| 4.3 | Add linting step (`ruff check app/ tests/`) as another gate in the same workflow file | One-command quality signal; also part of T1 hygiene |

### Acceptance criteria

| # | Criterion | How verified |
|---|-----------|--------------|
| 4.A | `.github/workflows/ci.yaml` exists and triggers on `push` and `pull_request` to `main`. | `grep -E "(push|pull_request)" .github/workflows/ci.yaml`. |
| 4.B | The workflow contains a job (or matrix) that runs `pytest --cov=app --cov-fail-under=95 tests/` on Python 3.12 (`ubuntu-latest`). | Manual inspection of the workflow YAML for the pytest + coverage invocation. |
| 4.C | A linting step runs `ruff check app/ tests/` and is marked as required (fails the job if non-zero exit). | Manual review — separate `steps:` entry with `ruff check`. |
| 4.D | A Docker build job exists in the same workflow that does **not** push; it only builds locally. | Verify the build job lacks any `docker push` or `ghcr.io/… --push` action. |

---

## T5 — Unit tests (core logic)

PRD §SC1–SC3 target. These tests **do not** need a real FAISS index or full title list — they stub those dependencies.

> **Split into 5 smaller tickets for easier tracking:**

### T5a — `test_config.py`: config loading & env-var overrides
git branch: `ai-agent/t5a_unit_tests_config`

| # | Task | Target area | PRD coverage |
|---|------|-------------|--------------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t5a_unit_tests_config`) and work on it until completion | Per-ticket prerequisite |
| 5a.1 | `test_config_defaults()` — import `app.config`, assert every PRD §9 var returns its documented default (no env vars set). | Unit: load config → assert defaults | T1/PRD §9 |
| 5a.2 | `test_config_env_override()` — for each env var (`MODEL_NAME`, `EMBED_DIM`, `FAISS_INDEX`, `TITLES_FILE`, `DEFAULT_K`, `DEFAULT_NPROBE`, `PORT`), set the env var to a distinct sentinel value, re-create config, assert the new value. | Unit: env-override paths | T1/PRD §9 |
| 5a.3 | Test any custom parsing logic in `app/config.py` (e.g., int coercion for `EMBED_DIM`, `DEFAULT_K`). | Unit: type conversion | PRD §9 |

### Acceptance criteria — T5a

| # | Criterion | How verified |
|---|-----------|--------------|
| 5a.A | Tests exist for every default value in `app/config.py` (MODEL_NAME, EMBED_DIM, FAISS_INDEX, TITLES_FILE, DEFAULT_K, DEFAULT_NPROBE, PORT, WORKERS). | `grep -c "test.*default" tests/test_config.py >= 8`. |
| 5a.B | Tests exist for env-override of every PRD §9 config variable. Each test sets the env var and asserts the new value is returned. | Unit test coverage on `app/config.py` ≥95 % (line). |

### T5b — `test_model_encode.py`: embedding generation
git branch: `ai-agent/t5b_unit_tests_model`

| # | Task | Target area | PRD coverage |
|---|------|-------------|--------------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t5b_unit_tests_model`) and work on it until completion | Per-ticket prerequisite |
| 5b.1 | Mock `SentenceTransformer.encode`. Pass a non-empty string (e.g., "test"), assert returned array shape = `(1, 384)`. | Unit: input shape / dim | PRD §7.1 |
| 5b.2 | Assert output dtype is `float32` (numpy). | Unit: float conversion | PRD §7.1 |
| 5b.3 | Verify L2 normalisation (`q_emb /= np.linalg.norm(q_emb)`) — assert `np.allclose(np.linalg.norm(result), 1.0)` within tol=1e-6. | Unit: L2 norm | PRD §7.1 |
| 5b.4 | Test edge case: single-character / whitespace-only query does not crash (empty array handling). | Unit: boundary | PRD §7.1 |

### Acceptance criteria — T5b

| # | Criterion | How verified |
|---|-----------|--------------|
| 5b.A | Shape assertion test exists (assert shape `(1, 384)` or `(1, EMBED_DIM)`). | `grep "shape" tests/test_model_encode.py`. |
| 5b.B | L2 normalisation is tested (`np.linalg.norm` ≈ 1.0). | `grep "linalg" tests/test_model_encode.py`. |
| 5b.C | Edge-case (empty / whitespace) query test exists and passes. | One test in `test_model_encode.py`. |

### T5c — `test_faiss_roundtrip.py`: mocked index round-trip
git branch: `ai-agent/t5c_unit_tests_faiss`

| # | Task | Target area | PRD coverage |
|---|------|-------------|--------------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t5c_unit_tests_faiss`) and work on it until completion | Per-ticket prerequisite |
| 5c.1 | Mock a FAISS IVF index with known vectors + associated title indices; assert searching the same query returns those exact titles (exact round-trip). | Unit: mock FAISS search | PRD §6 API |
| 5c.2 | Verify response schema `{rank: int, title: str, score: float}` has no missing or extra keys in every result entry. | Unit: schema validation | PRD §6 API |
| 5c.3 | Assert that *k* results yields exactly *k* entries for various values (k=1..100). | Unit: top-k count | PRD §6 API |

### Acceptance criteria — T5c

| # | Criterion | How verified |
|---|-----------|--------------|
| 5c.A | Round-trip test passes: known query → known title returned at rank 1. | Unit test in `test_faiss_roundtrip.py`. |
| 5c.B | Schema test iterates over every result entry asserting keys == `{"rank", "title", "score"}`. | `grep "set(result.keys())" tests/test_faiss_roundtrip.py` or equivalent. |
| 5c.C | k=1 and k=100 boundary tests verify count == exact *k*. | Two test functions (one for each endpoint). |

### T5d — `test_health_endpoint.py` + `test_lifespan.py`: integration-lite
git branch: `ai-agent/t5d_unit_tests_integration_lite`

| # | Task | Target area | PRD coverage |
|---|------|-------------|--------------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t5d_unit_tests_integration_lite`) and work on it until completion | Per-ticket prerequisite |
| 5d.1 | Spin up real FastAPI app via `httpx.AsyncClient`; hit `/health`; assert `{status: "ok", titles_loaded: N}` where *N* ≥ 0 (int). | Integration-lite: /health schema | PRD §SC2 |
| 5d.2 | Verify lifespan populates app state (`state.titles`, `state.index` etc.) on startup with mocked data if temp files are provided, or empty dicts otherwise. | Integration-lite: state population | PRD §7.2 |
| 5d.3 | Assert that after client's async context exits (shutdown), the state dict keys/values corresponding to loaded resources are cleared (`{}`). | Integration-lite: state teardown | PRD §7.2 constraint |

### Acceptance criteria — T5d

| # | Criterion | How verified |
|---|-----------|--------------|
| 5d.A | `/health` endpoint returns `{status: "ok", ...}` with correct schema (keys match exactly). | `assert resp.json()["status"] == "ok"`. |
| 5d.B | State is populated on startup (checked via a test that inspects the app state dict post-initialization). | One assertion on non-empty state dict inside test. |
| 5d.C | State is cleared on shutdown (assert empty state dict after the `AsyncClient` context manager exits). | Assertion in same or separate test file. |

### T5e — `test_validation.py`: parameter boundary tests
git branch: `ai-agent/t5e_unit_tests_validation`

| # | Task | Target area | PRD coverage |
|---|------|-------------|--------------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t5e_unit_tests_validation`) and work on it until completion | Per-ticket prerequisite |
| 5e.1 | `/search?q=` (empty query) → assert **HTTP 422**. | Integration-lite: empty q | PRD §6 + T1/PRD §10 |
| 5e.2 | `?k=0` → assert **HTTP 422**. | Integration-lite: lower k bound | PRD §6 |
| 5e.3 | `?k=101` (above max) → assert **HTTP 422**. | Integration-lite: upper k bound | PRD §6 |
| 5e.4 | Negative / zero nprobe (`?nprobe=-1`, `?nprobe=0`) → assert **HTTP 422**. | Integration-lite: nprobe bounds | PRD §6 |
| 5e.5 | Valid request (`q=test&k=5&nprobe=64`) → assert **HTTP 200** with correct response body shape (`query` str, `results` list of dict). | Integration-lite: happy-path baseline | PRD §6 |

### Acceptance criteria — T5e

| # | Criterion | How verified |
|---|-----------|--------------|
| 5e.A | Five boundary tests exist (empty q, k=0, k=101, nprobe=−1, nprobe=0); each asserts HTTP 422. | Exactly 5 test functions in `test_validation.py`. |
| 5e.B | Valid-request test returns HTTP 200 with a list of dicts under `results`, each dict having keys `{rank, title, score}`. | Assertion on `len(resp.json()["results"]) >= 1` and key check. |

---

## T6 — Integration tests (E2E docker + /search endpoint)
git branch: `ai-agent/t6_integration_tests_e2e_docker`

PRD SC3 target (~10 ms latency check). Run **only in CI on push to `main`** (gate = pass/fail; not coverage).

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t6_integration_tests_e2e_docker`) and work on it until completion | Per-ticket prerequisite |
| 6.1 | Add `tests/integration/` directory with `docker-compose.yaml`-aware e2e suite | Uses real FAISS index + titles files mounted as volumes |
| 6.2 | `search_e2e.py`: boot container via Testcontainers (or just call the compose service HTTP), send ~5 queries, assert response schema & latency < 20 ms p95 over 10 iterations per query | PRD SC3: "≤ 20 ms p95" |
| 6.3 | `search_quality.py`: sanity-check that semantically related queries actually return correct titles (e.g. query "photosynthesis" → first hit is "Photosynthesis") | Coverage must pass on the test module itself; these are smoke tests for correctness, not coverage-bound |

### Acceptance criteria — T6

| # | Criterion | How verified |
|---|-----------|--------------|
| 6.A | `tests/integration/` directory exists with an init file and both test modules (`test_search_e2e.py`, `test_search_quality.py`). | `ls tests/integration/test_*` lists two files. |
| 6.B | `test_search_e2e.py` sends at least 5 distinct queries; each query is asserted over 10 iterations, recording latencies. | Source inspection — loop or repeated assertions covering ≥5 queries × 10 rounds. |
| 6.C | Latency assertion: p95 of the recorded response times across all queries is ≤ 20 ms (measured end-to-end from HTTP request to full response body). | One assertion `assert p95(latencies) <= 20` in `test_search_e2e.py`. |
| 6.D | Response schema per query: `{query, results: [{rank: int, title: str, score: float}]}` — verified on at least one representative query. | Schema assertions in `test_search_e2e.py`. |
| 6.E | `test_search_quality.py` tests the "photosynthesis" → "Photosynthesis" (or near match) relationship and at least two other semantically related pairs. | Three or more quality checks in the file; each asserts `results[0].title` contains a substring of the expected title (case-insensitive). |

---

## T7 — GitHub Action to build & push image to GHCR (on every main push)
git branch: `ai-agent/t7_github_action_build_push_image_to_ghcr`

PRD §8.3 (deploy) target. Runs on **every push to `main`** — test gate first, then build and publish the image. No environment protection or release tags required.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t7_github_action_build_push_image_to_ghcr`) and work on it until completion | Per-ticket prerequisite |
| 7.1 | Create `.github/workflows/release.yaml` triggers: `push` to `main`; jobs **(a) test (reuse T4 CI job), (b) build-push** | Every main push produces a fresh image — CI validation gate stays in place |
| 7.2 | Job **(b)**: run `docker/build-push-action`, tag as `ghcr.io/${{ github.repository_owner }}/wiki-search:${{ github.ref_name }}` (which resolves to `:main`) | PRD SC4 — continuous deployment of every main push image to GHCR |

### Acceptance criteria — T7

| # | Criterion | How verified |
|---|-----------|--------------|
| 7.A | `.github/workflows/release.yaml` exists and triggers **only** on `push` to `main`. | `grep "push:" .github/workflows/release.yaml` confirms the event. |
| 7.B | The workflow contains a test job that re-uses T4's coverage gate before publishing. | One job in release.yaml runs pytest + coverage; its status check gates the build-push job. |
| 7.C | Image is tagged as `ghcr.io/${{ github.repository_owner }}/wiki-search:${{ github.ref_name }}`. | Inspect the `docker/build-push-action` step's `tags:` field in release.yaml. |

---

## T8 — Kubernetes manifests (production, low scale)
git branch: `ai-agent/t8_kubernetes_manifests_production`

PRD §8.2 target. Fixed 1–2 replicas, no HPA, stateless pod spec.

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t8_kubernetes_manifests_production`) and work on it until completion | Per-ticket prerequisite |
| 8.1 | Create `k8s/deployment.yaml` with single-Replica Deployment; read the image from GHCR via PRD SC5 target | kubectl get pods/status log = proof of success (PRD §SC5) |
| 8.2 | Add liveness probe on `/health` and a readiness probe — same port (8000), path (`/health`) | PRD §8.2 – "passes probes with no restarts" |
| 8.3 | Pod resource limits: `requests: 256Mi CPU/100m; limits: 1Gi/500m` (as per PRD) | Fixed scale target = simple spec file, no HPA controller needed |
| 8.4 | Add a ConfigMap for env vars from PRD §9 (text values only; `kubectl create configmap --from-env-file`) | One `ConfigMap`, one `Secret` if any sensitive data later needs to be added
| 8.5 | Create `k8s/pvc.yaml` — PersistentVolumeClaim sized to the FAISS index + titles (~tens of GiB). Mount it in the Deployment so the pod sees `/app/wiki_faiss.index` and `/app/wiki_titles.txt`. Docker Compose achieves this with `./file:/path` bind mounts; Kubernetes has no equivalent, so provision a cluster-level PersistentVolume (S3-backed, EFS/GCS/Fast-SSD, or NFS) and reference it via the PVC. Data files must be copied onto the PVC before pods start — run once: `kubectl exec -i pvc-wiki-data -- tar xf wiki_data.tar.gz`
| 8.6 | Mount the PVC volumes in the Deployment pod spec so the app process sees the index/titles paths matching PRD §9 defaults |

### Acceptance criteria — T8

| # | Criterion | How verified |
|---|-----------|--------------|
| 8.A | `k8s/deployment.yaml` exists; `spec.replicas == 1` (or an explicit single-replica default). | `yq '.spec.replicas' k8s/deployment.yaml` returns `1`. |
| 8.B | `image:` in the pod spec points to `ghcr.io/<repo>/wiki-search:<version>`. | Manual review of the YAML. |
| 8.C | Both `/health` liveness and readiness probes are configured (HTTP GET port 8000, path `/`). | `yq '.spec.template.spec.containers[0].livenessProbe'` and `.readinessProbe` both non-null. |
| 8.D | Resource limits match PRD §8.2: requests `256Mi CPU / 100m`; limits `1Gi / 500m`. (Note: the original PRD lists requests and limits in reversed order — adopt the convention below.) | `yq '.spec.template.spec.containers[0].resources'` matches `requests.cpu: "100m", requests.memory: 256Mi, limits.cpu: "500m", limits.memory: 1Gi`. |
| 8.E | A `ConfigMap` manifest (or embedded `envFrom`) contains the PRD §9 env vars. | `yq '.data' k8s/<configmap-file>.yaml` includes all 8 variables. |
| 8.F | `k8s/pvc.yaml` exists; `spec.accessModes` includes `ReadWriteMany`; `spec.resources.requests.storage` is ≥ the combined size of FAISS index + titles (recommend 50 GiB to start). A corresponding PersistentVolume exists in-cluster. | `yq '.spec.accessModes' k8s/pvc.yaml` contains `ReadWriteMany`; a matching `PersistentVolume` is created with the same `size`. |

---

## T9 — README.md (professional, concise)
git branch: `ai-agent/t9_readme_md_professional_concise`

PRD documentation target (no separate ticket needed — write alongside the other deliverables).

| # | Content elements |
|---|---|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t9_readme_md_professional_concise`) and work on it until completion | Per-ticket prerequisite |
| 9.1 | Project header with badges: `![CI](https://github.com/{owner}/{repo}/actions/workflows/ci.yaml/badge.svg)`, `![Docker Build](release.yaml)`, `![coverage](coverage-badge)` |
| 9.2 | One-line description, overview, quickstart (clone → docker-compose up → curl), API reference, config vars table, Kubernetes deploy section, links to PRD/TICKETS |
| 9.3 | Add license (`LICENSE`) and a small CONTRIBUTING guide (PR template, testing expectations — >95 % coverage gate) |

### Acceptance criteria — T9

| # | Criterion | How verified |
|---|-----------|--------------|
| 9.A | `README.md` contains CI badge pointing to `.github/workflows/ci.yaml`, Docker build badge for release workflow, and a coverage badge (or a line referencing the coverage number). | All three badges render valid URLs when opened in a browser. |
| 9.B | README includes: one-line description, quickstart section (`docker-compose up` → `curl http://localhost:8000/health`), API reference table (all `/search` params), config vars table (all 8 from PRD §9), Kubernetes deploy instructions, links to `PRD.md` and `TICKETS.md`. | Manual review — every listed item present as a section or paragraph. |
| 9.C | A `LICENSE` file exists with the **MIT** license text (matching the pre-requisite). | `head -n 1 LICENSE` starts with `Copyright` followed by the MIT SPDX identifier on an early line. |
| 9.D | A `CONTRIBUTING.md` (or section in README) documents: how to run tests, the >95 % coverage gate expectation, and the PR template location or reference. | Both test instructions and coverage mention are present. |

---

## T10 — Polish & final verification against PRD
git branch: `ai-agent/t10_polish_final_verification_against_prd`

| # | Task | Notes |
|---|------|-------|
| 0.0 | Create a new branch with the name shown in this ticket (`ai-agent/t10_polish_final_verification_against_prd`) and work on it until completion | Per-ticket prerequisite |
| 10.1 | Confirm every PRD goal: G1 ≤20 ms p95, G2 Docker + K8s, G3 GHCR publish, G4 one-line dev start | Ticks all four Goals in one final pass |
| 10.2 | Confirm every PRD success criterion (SC1–SC5) is measurable via the artifacts above | Cross-check each against its ticket |
| 10.3 | Open a single release PR with all of the above; ensure CI + tag-release workflow both succeed end-to-end before merging | The final gate — if this passes, the project is ship-ready |

### Acceptance criteria — T10

| # | Criterion | How verified |
|---|-----------|--------------|
| 10.A | **Goal G1** — latency measured via `TICKETS.md` T6's integration test: p95 ≤ 20 ms. | `pytest tests/integration/test_search_e2e.py -v` records latencies ≤ 20 ms p95. |
| 10.B | **Goal G2** — Docker Compose (SC2) and Kubernetes SC5) both function: compose starts and responds to `/health`; k8s manifest is valid (`kubectl --dry-run=client -f k8s/`). | `docker-compose up -d` + curl /health; `kubectl apply --dry-run=client -k k8s/`. |
| 10.C | **Goal G3** — tag-release workflow exists, triggers on published release, and publishes an image to GHCR. | Push a test tag `v0.0.0-test`; verify the GH run completes and the image appears at `ghcr.io/<owner>/wiki-search:v0.0.0-test`. |
| 10.D | **Goal G4** — `docker-compose up` starts the service and it responds to `/health` within 30s (PRD SC2). | Timing `curl --max-time 30 http://localhost:8000/health` within 30s of compose start. |
| 10.E | Every PRD success criterion (SC1–SC5) maps to at least one measurable artifact: SC1→CI logs, SC2→compose healthcheck log, SC3→T6 latency test, SC4→GHCR image exists, SC5→k8s deployment logs. | A traceability matrix in the release PR description or a comment linking each SC# to its proof. |
| 10.F | CI and tag-release workflows both pass end-to-end on the release branch before merge. | `git push && github-actions run check` or direct GitHub UI inspection shows green checks on both workflows. |
