# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 3 376+ | `pytest tests/` across 145+ files |
| Rust engine | 373 | `cargo test --manifest-path engine/Cargo.toml` |
| Bridge (PyO3) | — | Maturin build + Python import smoke test |
| HDL formal verification | 67 | 7 formal files across `hdl/formal/` (assert + cover properties) |

CI runs tests on Python 3.11–3.14 (Ubuntu) and Rust on Ubuntu + Windows.

## CI Validation Gates

All gates must pass before merge.

| Gate | Workflow | What it enforces |
|------|----------|------------------|
| ruff format | `ci.yml` | `ruff format --check` |
| ruff | `ci.yml` | `ruff check` (import hygiene, code quality) |
| bandit | `ci.yml` | Security static analysis (SAST) |
| test + coverage | `ci.yml` | `pytest --cov-fail-under=100` on Python 3.11, 3.12, 3.13, 3.14 |
| installed NIR interoperability | `ci.yml` | Separate hash-pinned snnTorch and Sinabs profiles, real conversions, nonempty JUnit results and zero supported-profile skips |
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

## Studio optional engine typing

The E-I and model simulator getter annotations match the native bridge callable
contract in both dependency profiles. Focused HTTP tests exercise model and
network requests; a separate isolated interpreter without the engine SDK verifies
health, real NumPy E-I simulation and adaptive-threshold model simulation.
The annotation change preserves simulation bodies and numerical behavior.

## Studio input identity

Model-run contracts check exact Python and NumPy constructor integers,
including seeds above 2^53, and reject fractional, boolean and non-finite
inputs. HTTP experiment checks exercise both explicit and legacy parameter
seeds against direct model steps, cache receipts and replay packs.
The installed-wheel browser gate checks Poisson controls and full-domain
seeds through both catalogue and equation workspaces, saved sessions and
a separate replay interpreter. See the [installed browser gate](CONTRIBUTING.md#installed-studio-browser-acceptance).

## Coverage Policy

The installed snnTorch job measures `execution_metadata.py`, `graph_identity.py`
and `snntorch_bridge.py` with 100% statement and branch coverage. It uses an
independent coverage configuration, so the package-wide NIR omission below
does not exclude these modules from that required gate. The separate Sinabs
profile exercises its compatible legacy NIR reader and exporter.

The desktop configuration checks also exercise an actual reader-only FIFO in
a separate interpreter and retain regular-file symlink compatibility. Profile
coverage includes that child through the configured subprocess and immediate-exit
patches; these checks do not qualify the full desktop installation.

`tests/test_studio_desktop_paths.py` and
`tests/test_studio_desktop_configuration.py` exercise the desktop profile
contract on real filesystems: XDG defaults and overrides, private new
directories, legacy project preservation, symlink cycles, bounded schema
refusal and atomic configuration replacement. These focused checks cover
the profile modules; desktop window, installer and lifecycle qualification
remain separate.

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

- Threshold: 100% (enforced by `pytest --cov-fail-under=100`) of the measured
  files. The scope below is generated from `pyproject.toml
  [tool.coverage.run] omit` and checked on every commit; the reason for each
  pattern stands next to it in that file.

<!-- coverage-scope:begin (generated by tools/coverage_scope.py; do not edit) -->

The coverage threshold applies to 1,194 of 1,834 Python files under
`src/sc_neurocore` (214,146 of 355,121 source lines, 60.3 %). The remaining
640 files (140,975 lines) are omitted by the patterns below and are not part of
the percentage. An omitted file may still have tests; the gate does not measure them.

| Omission pattern | Files | Source lines |
| --- | ---: | ---: |
| `*/quantum/hardware_bridge.py` | 1 | 135 |
| `*/quantum_cognition/bridge_adapter.py` | 1 | 551 |
| `*/integrations/lava_bridge.py` | 1 | 164 |
| `*/training/*` | 14 | 4,117 |
| `*/conversion/*` | 31 | 4,786 |
| `*/qat/torch_qat.py` | 1 | 367 |
| `*/qat/lsq.py` | 1 | 298 |
| `*/qat/pact.py` | 1 | 123 |
| `*/qat/observers.py` | 1 | 349 |
| `*/adapters/sonata.py` | 1 | 375 |
| `*/nir_bridge/*` | 32 | 9,680 |
| `*/accel/jax_backend.py` | 1 | 663 |
| `*/layers/jax_dense_layer.py` | 1 | 240 |
| `*/accel/julia/neurons/__init__.py` | 1 | 643 |
| `*/accel/julia/solvers/__init__.py` | 1 | 31 |
| `*/accel/julia/scpn/__init__.py` | 1 | 115 |
| `*/accel/julia/mixed_dense/__init__.py` | 1 | 94 |
| `*/accel/julia/adc_to_spike/__init__.py` | 1 | 103 |
| `*/accel/go/rk4_neurons/__init__.py` | 1 | 163 |
| `*/accel/go/dcls_tent/__init__.py` | 1 | 134 |
| `*/accel/go/mixed_dense/__init__.py` | 1 | 118 |
| `*/accel/go/adc_to_spike/__init__.py` | 1 | 124 |
| `*/accel/mojo/rk4_neurons/__init__.py` | 1 | 163 |
| `*/accel/mojo/dcls_tent/__init__.py` | 1 | 136 |
| `*/accel/mojo/mixed_dense/__init__.py` | 1 | 118 |
| `*/accel/mojo/adc_to_spike/__init__.py` | 1 | 124 |
| `*/neurons/models/*` | 177 | 38,195 |
| `*/network/mpi_runner.py` | 1 | 293 |
| `*/neurons/__main__.py` | 1 | 16 |
| `*/neurons/dsl_cli.py` | 1 | 383 |
| `*/quantum_cognition/__main__.py` | 1 | 463 |
| `*/experiments/*` | 27 | 3,084 |
| `*/quantum_cognition/fs_watcher.py` | 1 | 264 |
| `*/quantum_cognition/tests/*` | 16 | 1,269 |
| `*/viz/plots.py` | 1 | 285 |
| `*/studio/*` | 309 | 71,691 |
| `*/federation/*` | 5 | 1,118 |

<!-- coverage-scope:end -->

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


## Typed scalar native IR

Local Linux qualification exercises the default-feature Rust graph API, normal
release-wheel Python calls, persisted text, direct RTL and pinned CIRCT 1.152.0.
The focused native regression cohort has 69 passing tests. The installed graph
and tool cohort has 113 passing tests, and the retained CPU codec cohort has 67.
The numerical CIRCT corpus contains 94 circuits and 9,095 exact observations.
Direct RTL and CIRCT are each compared with independent encoded-integer references.
Signed/unsigned widths through 128 bits, division by zero, signed minimum,
oversized shifts, rounding ties, casts and saturation bounds are included.
Unused malformed declarations, bounded vector nesting and parser ID exhaustion
are refused before hardware dimensions or wrapped identifiers can be emitted.
Six seeded arithmetic/type/admission faults are caught by public numerical or refusal
gates. Source-leaf instrumentation is separate from default-crate execution;
LLVM branch counts of zero are not branch coverage evidence.

The public API and dedicated tests are documented in [the native IR contract](docs/api/sc-ir.md).
These results cover scalar graph circuits. Equation/state/integrator lowering,
full compiler acceptance and physical board measurements have separate evidence.

## Function-call numeric contracts

The [per-call contract](docs/guides/function_call_domains.md) reports conservative
encoded argument hulls and separate numerical measurements. Tests execute twenty
expressions over every signed input pair at six and eight bits through generated
C with undefined-behaviour checks and Rust with overflow checks. Critical
32-bit Q3.29 operands exercise wide products and narrowing. Clocked RTL checks
banker rounding over 4,096 input pairs, including exact singleton bounds.

Mathematical-domain exclusions and empty domains are explicit. Numerical tests
compare real unit outputs with the model's functions or independent Decimal power
references and reject corrupted bounds. The three numeric-analysis modules have
complete measured statement and branch coverage. The source-bound benchmark
separates a fresh interpreter's first collector invocation from cached calls,
records runtime dependencies and loaded-host context, and checks complete
contract equality outside timing.

These checks establish bounded regression evidence. Conservative hulls do not
prove state reachability, operand correlations or model-level error propagation;
analysis timings do not establish hardware throughput or board behaviour.
