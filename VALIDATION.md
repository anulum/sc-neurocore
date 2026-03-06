# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 911 | `pytest tests/` across 85+ files (887 pass, 46 skip) |
| Rust engine | — | `cargo test --manifest-path engine/Cargo.toml` |
| Bridge (PyO3) | — | Maturin build + Python import smoke test |
| HDL formal verification | 11 | Verilog modules in `hdl/` with testbenches in `tb/` |

CI runs tests on Python 3.10–3.12 (Ubuntu) and Rust on Ubuntu + Windows.

## CI Validation Gates

All gates must pass before merge.

| Gate | Workflow | What it enforces |
|------|----------|------------------|
| black | `ci.yml` | `black --check` (pinned 25.1.0) |
| ruff | `ci.yml` | `ruff check` (import hygiene, code quality) |
| bandit | `ci.yml` | Security static analysis (SAST) |
| test + coverage | `ci.yml` | `pytest --cov-fail-under=98` on Python 3.10, 3.11, 3.12 |
| spdx-guard | `ci.yml` | SPDX license headers on all `.py`, `.rs`, `.v` files |
| build | `ci.yml` | `python -m build` + smoke import |
| rust-lint | `v3-engine.yml` | `cargo fmt --check` + `cargo clippy -D warnings` |
| rust-test | `v3-engine.yml` | `cargo test` on Ubuntu + Windows |
| bridge-build | `v3-engine.yml` | Maturin build + v3 integration tests |
| wheels | `v3-wheels.yml` | Cross-platform wheel builds (Linux, macOS, Windows) |
| pre-commit | `pre-commit.yml` | Trailing whitespace, YAML/TOML, typos, black |
| codeql | `codeql.yml` | GitHub CodeQL security analysis |
| scorecard | `scorecard.yml` | OpenSSF Scorecard supply-chain audit |
| docs | `docs.yml` | MkDocs build verification |

## Coverage Policy

- Threshold: 98% (enforced by `pytest --cov-fail-under=98`)
- Omitted modules (documented in `pyproject.toml [tool.coverage.run] omit`):
  - `experiments/` — research demo scripts
  - `drivers/` — hardware-dependent PYNQ drivers
  - `interfaces/ccw_bridge.py` — CCW integration (tested in CCW repo)
  - `audio/adaptive_engine.py`, `audio/evs_engine.py` — hardware-dependent
  - `sleep/` — hardware-dependent biofeedback
  - `swarm/` — neuroevolution environment
- Excluded lines: `pragma: no cover`, `if __name__`, `raise NotImplementedError`,
  conditional imports (`HAS_MPI`, `HAS_CUPY`, `HAS_NUMBA`)

## Kuramoto Coupling Correctness

The UPDE solver implements `dθ_n/dt = ω_n + Σ_m K_nm sin(θ_m − θ_n)` with
phase-difference coupling. Tests verify:

- Two identical oscillators with K > 0 converge to phase lock
- N oscillators reach order parameter R → 1 for strong coupling
- Coupling term is zero when all phases are equal

## Holonomic Layer Adapters (L1–L16)

Each adapter in `src/sc_neurocore/adapters/holonomic/` has a corresponding
test file. Adapters implement the `HolonomicAdapter` protocol with:

- `adapt()` — transform state through the layer
- `inverse()` — reverse transform (where applicable)
- Round-trip property: `inverse(adapt(x)) ≈ x` within tolerance

## HDL Verification

Verilog modules in `hdl/` are verified via:

- Formal assertions (SystemVerilog `assert property`)
- Testbenches in `tb/` with golden-vector comparison
- Co-simulation parity checks against Python reference
