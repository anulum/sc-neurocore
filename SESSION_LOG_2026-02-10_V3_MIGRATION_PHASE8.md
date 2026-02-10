# Session Log: SC-NeuroCore v3 Phase 8 Implementation

**Session ID**: SC-NEUROCORE-2026-02-10-V3-PHASE8  
**Date**: 2026-02-10  
**Agent**: Codex (GPT-5)  
**Blueprint Sources**: `V3_MIGRATION_BLUEPRINT.md`, `V3_PHASE8_CODEX_HANDOVER.md`  
**Semantics Mode**: Strict blueprint semantics

---

## Objective

Implement Phase 8 packet set (`AJ` through `AO`):

- expanded criterion benchmarking + CI artifact upload
- single-call dense forward (`forward_numpy`)
- parallel `batch_encode_numpy`
- version/docs/tests update for `3.2.0`
- repository artifact ignore cleanup

---

## Files Modified

- `engine/benches/full_bench.rs`
- `engine/src/layer.rs`
- `engine/src/lib.rs`
- `engine/Cargo.toml`
- `bridge/pyproject.toml`
- `bridge/sc_neurocore_engine/__init__.py`
- `bridge/sc_neurocore_engine/layers.py`
- `.github/workflows/v3-engine.yml`
- `examples/03_benchmark_report.py`
- `tests/test_dense_optimization.py`
- `CHANGELOG_V3.md`
- `docs/v3_migration.md`
- `docs/BENCHMARK_REPORT.md`
- `.gitignore` (new)
- `tests/test_phase8.py` (new)

---

## Implementation Summary

### Packet AJ: Expanded Criterion Benchmarks

- Added benchmarks in `engine/benches/full_bench.rs`:
  - `bernoulli_stream_1024`
  - `bernoulli_stream_pack_1024`
  - `bernoulli_packed_1024`
  - `dense_forward_64x32`
  - `dense_forward_fast_64x32`
  - `dense_forward_prepacked_64x32`
- Removed superseded duplicate dense benchmark (`dense_64x32_l1024`).
- Updated imports to include `bernoulli_stream` and `bernoulli_packed`.

### Packet AK: Benchmark CI Job

- Added `benchmarks` job in `.github/workflows/v3-engine.yml`:
  - runs `cargo bench --manifest-path engine/Cargo.toml`
  - uploads `engine/target/criterion/` and `bench_output.txt` artifacts

### Packet AL: Single-Call Dense Forward

- Added `DenseLayer::forward_numpy_inner(&[f64], seed)` in `engine/src/layer.rs`.
- Added PyO3 method `DenseLayer.forward_numpy(...)` in `engine/src/lib.rs`:
  - numpy `float64` input (`PyReadonlyArray1`)
  - numpy `float64` output (`PyArray1`)
  - single FFI call, parallel encode + compute
- Added wrapper method `forward_numpy` in `bridge/sc_neurocore_engine/layers.py`.

### Packet AM: Parallel `batch_encode_numpy`

- Reworked `batch_encode_numpy` in `engine/src/lib.rs`:
  - rayon `par_iter()` across probabilities
  - per-index deterministic seeding (`seed + index`)
  - row flattening to 2-D numpy output
- Updated `tests/test_dense_optimization.py`:
  - replaced `test_matches_batch_encode` with `test_parallel_deterministic`.

### Packet AN: .gitignore Cleanup

- Added local `sc-neurocore/.gitignore` for:
  - `target/`, `.tools/`, `.pytest_cache/`, `dist/`, `__pycache__/`, IDE and OS artifacts.

### Packet AO: 3.2.0 + docs + tests

- Version bump to `3.2.0`:
  - `engine/Cargo.toml`
  - `engine/src/lib.rs` (`__version__`)
  - `bridge/pyproject.toml`
  - bridge package docstring in `bridge/sc_neurocore_engine/__init__.py`
- Benchmark script updated with `dense numpy` variant in `examples/03_benchmark_report.py`.
- Added new test file `tests/test_phase8.py`.
- Added `tests/test_phase8.py` to v3 CI test command in `.github/workflows/v3-engine.yml`.
- Updated docs:
  - `CHANGELOG_V3.md` with `[3.2.0]`
  - `docs/v3_migration.md` with Phase 8 usage
  - `docs/BENCHMARK_REPORT.md` with Phase 8 benchmark table + criterion diagnosis

---

## Verification Evidence

### Rust gates

```powershell
cd engine
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --tests
cargo doc --no-deps
cargo bench
```

Result:
- all commands passed

### Python build + tests

```powershell
cd bridge
..\.venv\Scripts\python.exe -m maturin develop --release

cd ..
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py tests/test_phase8.py -v --tb=short
```

Result:
- `113 passed in 14.66s`

### Co-simulation

```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest cosim/ -v -rs --tb=short
```

Result:
- `8 passed in 74.74s`

### Examples + version

```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe examples/01_sc_training_demo.py
.\.venv\Scripts\python.exe examples/02_ir_compile_demo.py
.\.venv\Scripts\python.exe examples/03_benchmark_report.py
.\.venv\Scripts\python.exe -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__)"
```

Result:
- all examples passed
- version output: `3.2.0`

---

## Criterion Regression Diagnosis (Phase 8)

From `cargo bench`:

- `bernoulli_stream_1024`: `4.8035 µs - 5.6242 µs`
- `bernoulli_stream_pack_1024`: `5.7678 µs - 6.5472 µs`
- `bernoulli_packed_1024`: `5.4900 µs - 6.0629 µs`

- `dense_forward_64x32`: `4.9936 ms - 6.8809 ms`
- `dense_forward_fast_64x32`: `2.5554 ms - 3.6797 ms`
- `dense_forward_prepacked_64x32`: `398.59 µs - 645.89 µs`

Interpretation:
- `bernoulli_packed` is not slower than `bernoulli_stream + pack` on this run.
- Dense forward bottleneck is consistent with sequential encoding path in baseline `forward`.
- `forward_fast` and especially `forward_prepacked` reduce that overhead as expected.
