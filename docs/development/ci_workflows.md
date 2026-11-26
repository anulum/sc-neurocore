<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — CI/CD workflows
-->

# CI/CD Workflows

This text was part of the repository README until the README became a short landing page.

27 GitHub Actions workflows (`.github/workflows/`), all third-party actions SHA-pinned:

| Workflow | Purpose |
|----------|---------|
| **audit-cadence.yml** | Scheduled repository, release-evidence, and policy audits |
| **ci.yml** | Lint (ruff format + ruff check + bandit) + Test (Python 3.11-3.14, coverage gate enforced in CI) + Build |
| **recorder-coverage.yml** | Native Linux/Windows recorder branch coverage, required artifacts, and portable gap enforcement |
| **compiler-e2e.yml** | Compiler end-to-end lowering, simulation, and evidence gates |
| **v3-engine.yml** | Rust engine `cargo test` + `cargo clippy` |
| **v3-wheels.yml** | Cross-platform wheels (Linux, macOS, Windows × Python 3.11–3.14) |
| **docker.yml** | Build & push Docker image to GHCR on release tags |
| **docs.yml** | MkDocs → GitHub Pages |
| **publish.yml** | Publish `sc-neurocore`, engine wheels, and `engine/` crate releases; retries skip already-published PyPI/crates.io versions |
| **pypi-downloads.yml** | Daily PyPI download history on the serialized `metrics` branch |
| **release.yml** | Build sdist/SBOM/security packet, support tagged-release backfills, and attach release assets to GitHub Release |
| **benchmark.yml** | Performance benchmarks, started by hand with a run reason and change reference |
| **brunel-parity.yml** | Weekly rerun of the dense NumPy Brunel lane against Brian2; fails when mean rates differ by more than the published 5% |
| **codeql.yml** | CodeQL security analysis (weekly + on push) |
| **dependency-audit.yml** | Complete audit of every committed dependency lock, per ecosystem |
| **hdc-reference.yml** | Hyperdimensional-computing reference comparison |
| **fuzz.yml** | Rust engine fuzzing with nightly cargo-fuzz targets |
| **hdl-toolchain.yml** | Shared Yosys, SBY, Z3, Yices and sv2v SDK for CI and synthesis: restores the one published archive the formal receipts bind, checked by its SHA-256; builds from source only on request |
| **mlir-circt.yml** | Python MLIR round-trip and native IR numerical CIRCT/RTL comparisons |
| **nir-canary.yml** | Latest NIR compatibility canary |
| **scorecard.yml** | OpenSSF Scorecard |
| **security-scanners.yml** | pip-audit, cargo-audit, Semgrep, and release security scans |
| **pre-commit.yml** | Pre-commit hook validation |
| **studio-desktop-installation.yml** | Studio Linux installation acceptance on two complete prepared bundles with the real runtime |
| **studio-desktop-runtime.yml** | Shared pinned inputs and standalone runtime assembly checks, required by CI |
| **yosys-synth.yml** | Yosys HDL synthesis verification; a manual run can also run and time every SymbiYosys job |
| **stale.yml** | Auto-label and close stale issues |
