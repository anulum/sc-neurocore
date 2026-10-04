# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 3 376+ | `pytest tests/` across 145+ files |
| Rust engine | 373 | `cargo test --manifest-path engine/Cargo.toml` |
| Bridge (PyO3) | — | Maturin build + Python import smoke test |
| HDL formal verification | 67 | 7 formal files across `hdl/formal/` (assert + cover properties) |

CI runs tests on Python 3.10–3.14 (Ubuntu) and Rust on Ubuntu + Windows.

## CI Validation Gates

All gates must pass before merge.

| Gate | Workflow | What it enforces |
|------|----------|------------------|
| ruff format | `ci.yml` | `ruff format --check` |
| ruff | `ci.yml` | `ruff check` (import hygiene, code quality) |
| bandit | `ci.yml` | Security static analysis (SAST) |
| test + coverage | `ci.yml` | `pytest --cov-fail-under=100` on Python 3.10, 3.11, 3.12, 3.13, 3.14 |
| spdx-guard | `ci.yml` | SPDX license headers on all `.py`, `.rs`, `.v` files |
| build | `ci.yml` | `python -m build` + smoke import |
| rust-lint | `v3-engine.yml` | `cargo fmt --check` + `cargo clippy -D warnings` |
| rust-test | `v3-engine.yml` | `cargo test` on Ubuntu + Windows |
| bridge-build | `v3-engine.yml` | Maturin build + v3 integration tests |
| wheels | `v3-wheels.yml` | Cross-platform wheel builds (Linux, macOS, Windows) |
| installed engine ABI | `v3-wheels.yml` | Capture facade/extension identities from site-packages outside the checkout and retain the measured inventory |
| default ABI reference | `v3-wheels.yml` | Compare every interface field with the qualified Linux x86-64 / CPython 3.12 / NumPy 2.2.3 default-feature reference |
| ABI proof tool coverage | `ci.yml` | 100% statement and branch coverage of inventory/comparison tools, including actual CLI subprocesses |
| pre-commit | `pre-commit.yml` | Trailing whitespace, YAML/TOML, typos, ruff format |
| codeql | `codeql.yml` | GitHub CodeQL security analysis |
| scorecard | `scorecard.yml` | OpenSSF Scorecard supply-chain audit |
| docs | `docs.yml` | MkDocs build verification |

## Coverage Policy

Engine interface capture and comparison have dedicated tests under
`tests/test_tools/test_engine_abi_inventory*.py` and
`tests/test_tools/test_engine_abi_contracts.py`. These exercise the real CLI,
live global pickle identities, editable-installation refusal and damaged
inventory rejection. [Engine bridge contracts](docs/guides/engine_bridge_contracts.md)
describe the evidence scope and the additional per-binding behavioral proof.

`tests/test_mckean_source_engine_binding.py` exercises the canonical installed
McKean class and complete native batch: numerical traces, array dtype/layout,
input preservation, atomic failure, configured reset, instance-pickle refusal
and the public Rust dispatcher. `tests/test_mckean_engine_binding.py` covers the
distinct retained SC triangular profile. These contracts keep both model
identities explicit; they do not substitute for the other binding owners.

- Threshold: 100% (enforced by `pytest --cov-fail-under=100`)
- Omitted modules (documented in `pyproject.toml [tool.coverage.run] omit`):
  - `experiments/` — research demo scripts
  - `drivers/` — hardware-dependent PYNQ drivers
  - `interfaces/ccw_bridge.py` — CCW integration (tested in CCW repo)
  - `audio/adaptive_engine.py`, `audio/evs_engine.py` — hardware-dependent
  - `sleep/` — hardware-dependent biofeedback

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
