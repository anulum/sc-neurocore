# Contributing to SC-NeuroCore

© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li

Thank you for your interest in SC-NeuroCore. Contributions are welcome under the following guidelines.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork and create a feature branch:
   ```bash
   git clone https://github.com/<your-user>/sc-neurocore.git
   cd sc-neurocore
   git checkout -b feature/your-feature
   ```
3. **Install dev dependencies and git hooks:**
   ```bash
   pip install -e ".[dev]"
   make install-hooks
   ```
4. **Build** the Rust engine (optional, for engine work):
   ```bash
   cd engine && maturin develop --release && cd ..
   ```
5. **Run the preflight gate** to verify your setup:
   ```bash
   make preflight
   ```

## Preflight Gate

Every push is guarded by `tools/preflight.py`, which runs the same checks as CI:

| Gate | What it checks |
|------|----------------|
| **ruff format** | Python formatting (`src/` and `tests/`) |
| **ruff check** | Code quality and import hygiene |
| **bandit** | Security static analysis |
| **spdx-guard** | SPDX license headers on all source files |
| **pytest** | 3 376+ tests with 100% coverage gate |

```bash
make preflight          # full gate (lint + tests)
make preflight-fast     # lint-only (~5s)
```

The `.githooks/pre-push` hook runs `preflight-fast` automatically before every `git push`. Install it with `make install-hooks`.

## Development Guidelines

### Code Style

- **Rust**: `cargo fmt` before committing.
- **Python**: `ruff format` + `ruff check` (both enforced in CI and preflight). Use type hints on public APIs only.
- **SPDX header**: Every `.py`, `.rs`, `.v` file must start with `# SPDX-License-Identifier: AGPL-3.0-or-later`.

### Language-native API documentation

Every new or materially changed public API must be documented in its native
documentation system in the same change. This forward ratchet applies across
the project; existing gaps remain explicit backfill work and are not silently
declared compliant.

- **Python:** module/class/function docstrings, including parameters, returns,
  failure behaviour, state mutation, and scientific or project provenance.
- **Rust:** `//!` module docs and `///` on public items; new public modules
  enable `#![warn(missing_docs)]`, and `cargo doc --no-deps` must pass.
- **Go:** GoDoc comments beginning with the exported identifier; `go test` and
  `go vet` must pass for the touched package.
- **Julia:** docstrings on exported modules, types, and functions, accessible
  through Julia's help/documentation system.
- **Mojo:** module and exported ABI comments recording parameter, ownership,
  error, and mutation contracts until a stable generated-doc gate is available.
- **RTL:** module, parameter, port, fixed-point-format, latency, reset, and
  event-semantics comments beside the maintained Verilog/SystemVerilog source.

Narrative Markdown complements these native surfaces but does not replace
them. Compatibility aliases must point to the canonical documented identity
and must not create duplicate scientific provenance.

### Testing

- All new Rust code must include tests in `engine/tests/` or inline `#[cfg(test)]` modules.
- All new Python APIs must have pytest coverage in `tests/`.
- SIMD paths must include portable fallback — never assume AVX-512 or AVX2 availability.

### Commit Messages

Follow conventional commit format:
```
feat(scope): short description
fix(scope): short description
docs(scope): short description
```

Examples:
```
feat(bitstream): add rotate_left method for HDC permutation
fix(simd): correct AVX2 popcount for non-aligned buffers
docs(readme): update benchmark table for v3.8
```

## What to Contribute

**High-value contributions:**
- New SIMD kernels (ARM SVE, RISC-V Vector)
- Additional HDC similarity metrics
- Jupyter notebook tutorials
- Performance benchmarks on new hardware
- Bug reports with reproducible test cases

**Please discuss first** (open an issue) before:
- Changing the public Python API
- Modifying the IR op set
- Adding new crate dependencies

## Submitting a Pull Request

1. Run `make preflight` — all gates must pass
2. Add a changelog entry if the change is user-visible
3. Ensure SPDX headers are present on new files
4. Open a PR against `main` with a clear description
5. Reference any related issues

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make test` | Python tests with coverage |
| `make test-rust` | Rust engine tests |
| `make test-all` | Both Python and Rust |
| `make lint` | ruff format + ruff check |
| `make fmt` | Auto-format Python + Rust |
| `make bandit` | Security static analysis |
| `make sast` | Alias for bandit |
| `make preflight` | Full CI-equivalent gate |
| `make preflight-fast` | Lint-only (~5s) |
| `make install-hooks` | Install git pre-push hook |
| `make bench` | Python benchmarks |
| `make bench-rust` | Rust Criterion benchmarks |
| `make bridge` | Build Rust bridge via maturin |
| `make docs` | Live docs preview |
| `make docs-build` | Build docs (strict mode) |
| `make build` | Build sdist + wheel |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker image interactively |
| `make clean` | Remove build artifacts |

## Licence

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE). For commercial licensing enquiries, contact protoscience@anulum.li.
