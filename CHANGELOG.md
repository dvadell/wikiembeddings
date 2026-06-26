# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Self-initialising container** — on first start with no pre-built index, the application
  automatically downloads Wikipedia titles, generates embeddings, and builds a FAISS IVF
  index in a background thread while `/health` stays online immediately with real-time status.
  `/search` returns `503 Service Unavailable` with progress details until the build completes.
- **`BUILD_RESUME` environment variable** — when set to `true` (the default), the build pipeline
  skips any stage whose output files already exist on disk, enabling fast restarts after crashes
  or image updates. Set to `false` to force a full rebuild from scratch.
- **First-start duration estimate** — expected first-start build time for the full English Wikipedia
  dump is approximately **2–8 hours** depending on node storage performance (SSD recommended)
  and CPU capacity. The synthetic corpus used by integration tests (~1k titles) typically finishes
  in tens of seconds.

### Changed

- Single-container architecture: replaced the two-container approach (builder + runtime) with a
  single image that handles both index building and serving (`docker-compose.yaml` no longer
  references an external Dockerfile for building).

<!-- prettier-ignore-start -->
<!-- qwen-code:llm-output-language: English -->
<!-- END-KEEP-CHANGELOG-FORMAT -->
<!-- prettier-ignore-end -->
