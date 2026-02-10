# SC-NeuroCore v3 — Phase 8 Codex Handover

**From**: Claude (Opus 4.6) — Architect & Reviewer
**To**: Codex (GPT-5) — Implementer
**Date**: 2026-02-10
**Phase**: 8 — Benchmark CI, Single-Call Dense Forward, Parallel Encoding
**Prerequisite**: Phase 7 ACCEPTED (`V3_PHASE7_CODE_REVIEW.md`)

---

## 1. Context & Motivation

Phase 7 delivered three dense forward tiers (original, fast, prepacked) and achieved 7.4x for the prepacked path. However, several quality and performance issues remain:

### Issue 1: forward() Benchmark Regression
Phase 7 reported `forward()` at **0.2x** (v3=20.57ms), down from Phase 6's **1.4x** (v3=2.064ms). This is a 10x wall-clock regression in the same function. Meanwhile, `forward_fast()` achieves **1.0x** (v3=4.3ms) using the same `bernoulli_packed` function but with parallel encoding. This suggests the sequential `bernoulli_packed` path may have different optimization characteristics, or the Phase 7 benchmark run was affected by external factors. Criterion micro-benchmarks will provide statistically rigorous measurements to diagnose this.

### Issue 2: No Automated Regression Detection
The Python benchmark script (`03_benchmark_report.py`) produces ad-hoc measurements with no statistical analysis, warm-up, or baseline comparison. Criterion benchmarks with CI integration would catch regressions automatically.

### Issue 3: Multi-Call Dense Forward
Currently, the fastest dense forward path requires two Python calls:
```python
packed = v3.batch_encode_numpy(inputs, length=1024, seed=42)  # call 1
out = layer.forward_prepacked(packed)                          # call 2
```
A single-call `forward_numpy` that accepts numpy inputs and does parallel encode + compute internally would be simpler and eliminate one FFI crossing.

### Issue 4: Sequential batch_encode_numpy
`batch_encode_numpy` encodes probabilities sequentially despite each being independent. Parallelizing with rayon would speed up large batch encoding.

---

## 2. Packet Summary

| Packet | Deliverable | Files Modified | Files Created |
|--------|------------|---------------|---------------|
| **AJ** | Expanded criterion benchmarks | `engine/benches/full_bench.rs` | — |
| **AK** | Benchmark CI job | `.github/workflows/v3-engine.yml` | — |
| **AL** | `forward_numpy` single-call method | `engine/src/layer.rs`, `engine/src/lib.rs` | — |
| **AM** | Parallel `batch_encode_numpy` | `engine/src/lib.rs` | — |
| **AN** | .gitignore cleanup | — | `.gitignore` (or modify existing) |
| **AO** | Version 3.2.0 + docs + tests | `Cargo.toml`, `lib.rs`, `pyproject.toml`, `__init__.py`, `CHANGELOG_V3.md`, `v3_migration.md`, `BENCHMARK_REPORT.md`, `03_benchmark_report.py`, `v3-engine.yml` | `tests/test_phase8.py` |

**Total**: ~10 files modified, 1-2 files created.

---

## 3. Packet AJ: Expanded Criterion Benchmarks

### 3.1 Update `engine/benches/full_bench.rs`

Add the following benchmarks to the existing `bench_all` function:

```rust
// -- Bernoulli encoding comparison --
{
    use sc_neurocore_engine::bitstream::{bernoulli_packed, bernoulli_stream};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    c.bench_function("bernoulli_stream_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            black_box(bernoulli_stream(0.5, 1024, &mut rng))
        })
    });

    c.bench_function("bernoulli_stream_pack_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let bits = bernoulli_stream(0.5, 1024, &mut rng);
            black_box(pack(&bits).data)
        })
    });

    c.bench_function("bernoulli_packed_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            black_box(bernoulli_packed(0.5, 1024, &mut rng))
        })
    });
}

// -- Dense forward variants --
{
    let layer = DenseLayer::new(64, 32, 1024, 42);
    let inputs = vec![0.5_f64; 64];

    c.bench_function("dense_forward_64x32", |b| {
        b.iter(|| black_box(layer.forward(black_box(&inputs), 42).unwrap()))
    });

    c.bench_function("dense_forward_fast_64x32", |b| {
        b.iter(|| black_box(layer.forward_fast(black_box(&inputs), 42).unwrap()))
    });

    // Pre-pack inputs for prepacked bench
    let packed_inputs: Vec<Vec<u64>> = {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        inputs.iter().enumerate().map(|(idx, &p)| {
            let mut rng = ChaCha8Rng::seed_from_u64(42u64.wrapping_add(idx as u64));
            sc_neurocore_engine::bitstream::bernoulli_packed(p, 1024, &mut rng)
        }).collect()
    };

    c.bench_function("dense_forward_prepacked_64x32", |b| {
        b.iter(|| black_box(layer.forward_prepacked(black_box(&packed_inputs)).unwrap()))
    });
}
```

This adds 6 new benchmarks: 3 encoding comparisons + 3 dense forward variants. The encoding benchmarks will show whether `bernoulli_packed` is slower than `bernoulli_stream + pack`, which would explain the forward() regression.

**Also remove the duplicate `dense_64x32_l1024` benchmark** that already exists (line 57-59) to avoid confusion — the new `dense_forward_64x32` replaces it.

### 3.2 Update imports in full_bench.rs

Add to existing imports:

```rust
use sc_neurocore_engine::bitstream::{bernoulli_packed, bernoulli_stream, pack, popcount_words_portable};
```

(Merge with existing `use sc_neurocore_engine::bitstream::{pack, popcount_words_portable};`)

---

## 4. Packet AK: Benchmark CI Job

### 4.1 New job in `.github/workflows/v3-engine.yml`

Add after the `v2-compat` job:

```yaml
  benchmarks:
    runs-on: ubuntu-latest
    needs: [rust-test]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run criterion benchmarks
        run: cargo bench --manifest-path engine/Cargo.toml -- --output-format bencher 2>&1 | tee bench_output.txt

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: criterion-results
          path: |
            engine/target/criterion/
            bench_output.txt
```

This runs all criterion benchmarks and uploads the HTML reports + raw output as CI artifacts. For now, it does not fail on regressions — that can be added later with `criterion-compare` or `bencher.dev` integration.

---

## 5. Packet AL: `forward_numpy` Single-Call Dense Forward

### 5.1 New method in `engine/src/layer.rs`

Add to `impl DenseLayer` after `forward_prepacked`:

```rust
/// Single-call dense forward with parallel Bernoulli encoding.
///
/// Combines `forward_fast` encoding with `forward_prepacked` compute in
/// one method call, avoiding any intermediate Python/Rust boundary.
/// Each input is encoded with `seed + input_index` as its RNG seed.
///
/// This is functionally identical to calling `forward_fast` but is intended
/// as the target for a numpy-accepting PyO3 binding that eliminates
/// all Python-side marshalling.
pub fn forward_numpy_inner(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
    // forward_fast already does parallel encode + parallel compute
    self.forward_fast(input_values, seed)
}
```

Note: `forward_numpy_inner` is a thin wrapper. The real value is in the PyO3 binding that accepts/returns numpy arrays.

### 5.2 PyO3 binding in `engine/src/lib.rs`

Add to `#[pymethods] impl DenseLayer`:

```rust
/// Dense forward accepting a 1-D numpy float64 input array,
/// returning a 1-D numpy float64 output array.
///
/// Parallel encoding + parallel compute in a single FFI call.
/// This is the recommended high-performance inference API.
#[pyo3(signature = (input_values, seed=44257))]
fn forward_numpy<'py>(
    &self,
    py: Python<'py>,
    input_values: PyReadonlyArray1<'py, f64>,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = input_values
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read input array: {e}")))?;
    let out = self
        .inner
        .forward_fast(slice, seed)
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray_bound(py))
}
```

**Key benefits over existing APIs**:
1. **Zero-copy input**: `PyReadonlyArray1::as_slice()` — no Python list → Vec conversion
2. **Numpy output**: Returns `PyArray1<f64>` — no Vec → Python list conversion
3. **Single FFI call**: Encode + compute in one crossing
4. **Parallel encoding**: Uses `forward_fast` internally (rayon across inputs)

### 5.3 Bridge wrapper

Add to `bridge/sc_neurocore_engine/layers.py` in the `VectorizedSCLayer` class:

```python
def forward_numpy(self, input_values, seed=44257):
    """Dense forward with numpy input/output and parallel encoding."""
    import numpy as np
    arr = np.asarray(input_values, dtype=np.float64)
    return self._engine.forward_numpy(arr, seed)
```

---

## 6. Packet AM: Parallel `batch_encode_numpy`

### 6.1 Modify `batch_encode_numpy` in `engine/src/lib.rs`

Replace the current sequential implementation with a parallel one. The current code (lines ~350-375) does:

```rust
// CURRENT — sequential
let mut rng = ChaCha8Rng::seed_from_u64(seed);
for &p in prob_slice {
    let packed = bernoulli_packed(p, length, &mut rng);
    flat.extend_from_slice(&packed);
}
```

Replace with:

```rust
/// Bernoulli-encode a numpy float64 array into a 2-D numpy uint64 array.
///
/// Each probability is encoded with `seed + index` for deterministic
/// parallel encoding via rayon.
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode_numpy<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    use rayon::prelude::*;

    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);
    let n_probs = prob_slice.len();

    // Parallel encoding: each probability gets its own seeded RNG
    let rows: Vec<Vec<u64>> = prob_slice
        .par_iter()
        .enumerate()
        .map(|(idx, &p)| {
            use rand::SeedableRng;
            let prob_seed = seed.wrapping_add(idx as u64);
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(prob_seed);
            let mut packed = bitstream::bernoulli_packed(p, length, &mut rng);
            packed.resize(words, 0);
            packed
        })
        .collect();

    // Flatten to row-major and reshape to 2-D numpy
    let mut flat = Vec::with_capacity(n_probs * words);
    for row in &rows {
        flat.extend_from_slice(row);
    }

    use numpy::PyArray;
    let arr = PyArray::from_vec_bound(py, flat);
    let arr2d = arr
        .reshape([n_probs, words])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {e}")))?;
    Ok(arr2d)
}
```

**IMPORTANT**: This changes the seeding strategy from single-sequential-RNG to per-index seeds. This means `batch_encode_numpy` output will be **DIFFERENT** from Phase 7 for the same inputs. However:
- Each probability is still deterministically encoded (seed + index)
- The `batch_encode` (non-numpy) function retains the original sequential seeding for backwards compatibility
- The `test_matches_batch_encode` test must be **updated or removed** since the two functions now use different seeding strategies

### 6.2 Update test

In `tests/test_dense_optimization.py`, update the `test_matches_batch_encode` test in `TestBatchEncodeNumpy`:

```python
def test_parallel_deterministic(self):
    """batch_encode_numpy must be deterministic with same seed."""
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    r1 = v3.batch_encode_numpy(probs, length=256, seed=42)
    r2 = v3.batch_encode_numpy(probs, length=256, seed=42)
    np.testing.assert_array_equal(r1, r2)
```

Replace the old `test_matches_batch_encode` (which compared sequential vs parallel output) with this determinism test.

---

## 7. Packet AN: .gitignore Cleanup

### 7.1 Create or update `.gitignore` in the `sc-neurocore` directory

If `.gitignore` already exists, append. If not, create:

```gitignore
# Rust build artifacts
target/

# Python cache
__pycache__/
*.pyc
*.pyo

# Pytest cache
.pytest_cache/

# Local tools (Verilator, etc.)
.tools/

# Maturin build output
dist/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

## 8. Packet AO: Version 3.2.0 + Docs + Tests

### 8.1 Version bump

**`engine/Cargo.toml`** line 3:
```toml
version = "3.2.0"
```

**`engine/src/lib.rs`** line 22:
```rust
m.add("__version__", "3.2.0")?;
```

**`bridge/pyproject.toml`**:
```toml
version = "3.2.0"
```

### 8.2 Updated benchmark script

In `examples/03_benchmark_report.py`, add a `dense_forward_numpy` variant to `bench_dense_forward`:

```python
# After the existing prepacked benchmark, add:
v3_numpy_time = benchmark(lambda: v3_layer.forward_numpy(inputs_f64), n_iters=10)
```

And add a result entry:
```python
{
    "operation": f"dense numpy ({n_in}->{n_out}, L={length})",
    "v2_ms": v2_time / 10 * 1000,
    "v3_ms": v3_numpy_time / 10 * 1000,
    "speedup": fmt_speedup(v2_time, v3_numpy_time),
    "target": "70x",
},
```

The `forward_numpy` variant should benchmark very close to `forward_fast` since it uses the same internal path but with zero-copy numpy input/output.

### 8.3 CHANGELOG

Prepend to `CHANGELOG_V3.md`:

```markdown
## [3.2.0] - 2026-02-10

### Phase 8: Benchmark CI, Single-Call Dense Forward, Parallel Encoding
- **Criterion Benchmarks**: Expanded suite with bernoulli encoding comparison + dense forward variants
- **Benchmark CI**: Automated criterion runs with artifact upload
- **DenseLayer.forward_numpy()**: Single FFI call with numpy input/output + parallel encoding
- **Parallel batch_encode_numpy**: Rayon-parallelized probability encoding
- **Repo cleanup**: .gitignore for build artifacts
```

### 8.4 Migration docs

Append Phase 8 section to `docs/v3_migration.md`:

```markdown
## Phase 8 Features (February 2026)

### Single-Call Dense Forward with NumPy

The recommended high-performance inference API:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
inputs = np.random.uniform(0, 1, 64)

# Single FFI call: numpy in → parallel encode → parallel compute → numpy out
out = layer.forward_numpy(inputs)
# out is a numpy float64 array of shape (32,)
```

### Parallel Batch Encoding

`batch_encode_numpy` now uses rayon-parallel encoding:

```python
probs = np.random.uniform(0, 1, 1000)
packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
# Each probability encoded on its own thread
```

Note: `batch_encode_numpy` now uses per-index seeding (`seed + index`) for
parallelism. Use `batch_encode` for sequential single-RNG seeding.
```

### 8.5 New test file: `tests/test_phase8.py`

```python
"""Tests for Phase 8: forward_numpy, parallel batch_encode_numpy, criterion."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestForwardNumpy:
    """Tests for single-call numpy dense forward."""

    def test_output_shape_and_type(self):
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8,)
        assert out.dtype == np.float64

    def test_output_range(self):
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.3] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_deterministic(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_matches_forward_fast(self):
        """forward_numpy should give same results as forward_fast (same encoding)."""
        layer = v3.DenseLayer(8, 4, 256, seed=42)
        inputs_list = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        inputs_np = np.array(inputs_list, dtype=np.float64)
        out_fast = layer.forward_fast(inputs_list, seed=42)
        out_numpy = layer.forward_numpy(inputs_np, seed=42)
        np.testing.assert_allclose(out_numpy, out_fast)

    def test_wrong_input_length(self):
        layer = v3.DenseLayer(8, 4, 256)
        inputs = np.array([0.5] * 7, dtype=np.float64)
        with pytest.raises(ValueError):
            layer.forward_numpy(inputs)

    def test_different_seed_different_output(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.array([0.5] * 8, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=200)
        assert not np.array_equal(out1, out2)


class TestParallelBatchEncodeNumpy:
    """Tests for parallel batch_encode_numpy."""

    def test_shape_and_dtype(self):
        probs = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        assert packed.shape == (3, 16)
        assert packed.dtype == np.uint64

    def test_deterministic(self):
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed(self):
        probs = np.array([0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=1024, seed=1)
        p2 = v3.batch_encode_numpy(probs, length=1024, seed=2)
        assert not np.array_equal(p1, p2)

    def test_popcount_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10000, seed=42)
        pc0 = sum(bin(w).count('1') for w in packed[0])
        pc1 = sum(bin(w).count('1') for w in packed[1])
        assert abs(pc0 / 10000 - 0.25) < 0.03
        assert abs(pc1 / 10000 - 0.75) < 0.03

    def test_pipeline_encode_then_forward(self):
        """batch_encode_numpy → forward_prepacked still works."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_empty_probs(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=64, seed=42)
        assert packed.shape[0] == 0


class TestCriterionBenchExists:
    """Smoke test that criterion bench binaries compile (run via cargo bench)."""

    def test_version_is_3_2_0(self):
        assert v3.__version__ == "3.2.0"
```

### 8.6 CI test list update

In `.github/workflows/v3-engine.yml`, add `tests/test_phase8.py` to the pytest command in the `equivalence` job's v3-specific test step.

### 8.7 Export forward_numpy

Add `forward_numpy` to the bridge `layers.py` wrapper (Packet AL already covers this).

No changes needed to `__init__.py` since `forward_numpy` is a method on `DenseLayer`, not a standalone function. However, if `batch_encode_numpy` signature changed (parallel seeding), ensure the existing export still works.

---

## 9. Quality Gates

```powershell
# Rust gates
cd 03_CODE/sc-neurocore/engine
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --tests
cargo doc --no-deps
cargo bench  # NEW — verify criterion benchmarks run

# Python build
cd ../bridge
..\.venv\Scripts\python -m maturin develop --release

# Python tests
cd ..
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py tests/test_phase8.py -v --tb=short

# Co-sim
.\.venv\Scripts\python -m pytest cosim/ -v -rs --tb=short

# Examples
.\.venv\Scripts\python examples/01_sc_training_demo.py
.\.venv\Scripts\python examples/02_ir_compile_demo.py
.\.venv\Scripts\python examples/03_benchmark_report.py

# Version check
.\.venv\Scripts\python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__)"
# Expected: 3.2.0
```

### Expected Test Counts

| Suite | Phase 7 | Phase 8 Expected |
|-------|---------|-----------------|
| Rust (`cargo test`) | 57+ | 57+ (unchanged) |
| Python (v3 suite) | 100 | ~118+ (new test file + updated tests) |
| Co-sim | 8 | 8 (unchanged) |
| Criterion benches | 9 | 15+ (6 new) |

---

## 10. Sacred File Rules

**DO NOT MODIFY** any file under:
- `src/sc_neurocore/` (v2 Python source)
- `pyproject.toml` (root — v2 package config)
- `.github/workflows/ci.yml` (v2 CI)

---

## 11. Acceptance Criteria Summary

| # | Criterion | How to Verify |
|---|-----------|--------------|
| 1 | Criterion benches include bernoulli_packed, bernoulli_stream+pack, forward, forward_fast, forward_prepacked | `cargo bench` runs all 15+ benchmarks |
| 2 | Benchmark CI job uploads artifacts | File inspection of v3-engine.yml |
| 3 | `DenseLayer.forward_numpy()` accepts numpy, returns numpy | Python test `test_output_shape_and_type` |
| 4 | `forward_numpy` matches `forward_fast` output | Python test `test_matches_forward_fast` |
| 5 | `batch_encode_numpy` uses rayon parallelism | Code inspection (par_iter in impl) |
| 6 | `batch_encode_numpy` is deterministic | Python test `test_deterministic` |
| 7 | .gitignore covers target/, .tools/, .pytest_cache/ | File inspection |
| 8 | Version = 3.2.0 everywhere | `import sc_neurocore_engine; print(sc_neurocore_engine.__version__)` |
| 9 | CHANGELOG has [3.2.0] section | File inspection |
| 10 | Benchmark script includes forward_numpy variant | `python examples/03_benchmark_report.py` |
| 11 | Sacred files untouched | `git diff -- src/sc_neurocore/ \| grep -v __pycache__` returns nothing |
| 12 | All quality gates pass | See Section 9 |

---

## 12. Notes

### batch_encode_numpy Seeding Change

Phase 8 changes `batch_encode_numpy` from sequential single-RNG to per-index parallel seeding. This means:
- `batch_encode_numpy(probs, seed=42)` in Phase 8 produces **different** output from Phase 7 for the same inputs
- `batch_encode(probs, seed=42)` (non-numpy, sequential) is **unchanged** for backwards compatibility
- The `test_matches_batch_encode` test from Phase 7 must be replaced with `test_parallel_deterministic`

This is an intentional breaking change for `batch_encode_numpy` only. Since v3.2.0 is a minor bump and `batch_encode_numpy` was introduced in v3.1.0 (one release ago), this is acceptable.

### forward() Regression Diagnosis

After running `cargo bench`, examine the criterion output for:
- `bernoulli_packed_1024` vs `bernoulli_stream_pack_1024` — if packed is significantly slower, the function needs optimization
- `dense_forward_64x32` vs `dense_forward_fast_64x32` — if forward is much slower than fast, the issue is sequential vs parallel encoding

Report these numbers in the session log.
