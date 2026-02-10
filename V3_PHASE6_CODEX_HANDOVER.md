# SC-NeuroCore v3.0 — Phase 6 Codex Handover

**Author**: Claude (Opus 4.6)
**Date**: 2026-02-10
**Phase**: 6 — Performance Optimization, CI Completeness, Stable Release
**Preceding Phase**: Phase 5 (3.0.0-rc.1 — IR Bridge, Co-Sim, Wheels)
**Blueprint Reference**: `V3_MIGRATION_BLUEPRINT.md` §8 Performance Targets

---

## Overview

Phase 5 delivered a **Release Candidate (3.0.0-rc.1)** with a complete IR Python bridge, wheel CI, and benchmark report. However, the benchmark exposed a critical performance gap: **FFI marshalling overhead** causes v3 to be *slower* than v2 for pack/popcount operations, and far below Blueprint §8 targets for dense/LIF even where v3 wins.

Phase 6 closes this gap via:
1. **NumPy zero-copy interop** — Eliminate Python list → Rust Vec conversion
2. **Batch operations** — Process arrays in single FFI calls
3. **Verilator CI** — Run co-sim tests automatically on push
4. **Updated benchmarks** — Show true kernel performance
5. **Stable 3.0.0 release** — Final version, changelog, docs

### Current Benchmark (Phase 5)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (1M bits) | 9.5 | 32.6 | 0.3x | 6x |
| popcount (1M words) | 97.5 | 141.0 | 0.7x | 20x |
| dense forward (64→32) | 3.0 | 1.0 | 2.9x | 70x |
| LIF step (100K) | 109.7 | 28.5 | 3.8x | 400x |

**Root cause**: `bits.tolist()` and `packed.tolist()` in the benchmark convert numpy arrays to Python lists, then PyO3 converts those lists to Rust Vecs — two memory copies. The numpy crate (already in Cargo.toml) provides `PyReadonlyArray` for zero-copy access to numpy array buffers.

---

## Sacred File Integrity Rule

**NEVER modify** any file under `src/sc_neurocore/`. This is the v2 Python codebase and must remain untouched.

---

## Packet Sequence

| Packet | Scope | Files | Depends On |
|--------|-------|-------|------------|
| **Y-0** | Phase 5 fixups | 2 modified | — |
| **Y** | NumPy zero-copy functions | 1 modified (lib.rs) | Y-0 |
| **Z** | Batch LIF & encode ops | 1 modified (lib.rs) | Y |
| **AA** | Verilator CI | 1 modified (v3-engine.yml) | — |
| **AB** | Updated benchmarks | 2 modified (script + report) | Y, Z |
| **AC** | 3.0.0 stable release | 5 modified (version, docs) | All above |

---

## Packet Y-0: Phase 5 Fixups

### Y-0a: Fix bridge/pyproject.toml version

The bridge `pyproject.toml` shows `version = "3.0.0a1"` (alpha) while the engine is at `3.0.0-rc.1`. The wheel build picks up the bridge version, producing artifacts named `sc_neurocore-2.2.0-*.whl` (from root pyproject.toml) instead of the engine version.

**File**: `bridge/pyproject.toml`

Replace the entire file with:

```toml
[build-system]
requires = ["maturin>=1.5"]
build-backend = "maturin"

[project]
name = "sc_neurocore_engine"
version = "3.0.0"
requires-python = ">=3.9"
description = "High-performance Rust backend for SC-NeuroCore"
license = {text = "MIT"}
authors = [{name = "Miroslav Sotek", email = "fortisstudio@gmail.com"}]

[tool.maturin]
manifest-path = "../engine/Cargo.toml"
python-source = "."
module-name = "sc_neurocore_engine.sc_neurocore_engine"
```

### Y-0b: Add IR Python tests to v3-engine.yml

Phase 5 added `tests/test_ir_python.py` but the CI workflow's v3-specific test command doesn't include it.

**File**: `.github/workflows/v3-engine.yml`

Replace line 78 (the `pytest` command in the "Run v3-specific tests" step):

```yaml
      - name: Run v3-specific tests
        run: pytest tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py -v --tb=short
```

(Add `tests/test_ir_python.py` to the end of the test file list.)

---

## Packet Y: NumPy Zero-Copy Functions

Add three new `#[pyfunction]` entries that accept numpy arrays directly via `PyReadonlyArray` and return numpy arrays via `IntoPyArray`. These avoid all list conversion overhead.

### Rust Changes (lib.rs)

**Add imports** at the top of `lib.rs` (after existing `use` statements):

```rust
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, IntoPyArray};
```

**Add three new functions** after the existing `popcount` function (after line 136):

```rust
/// Pack a 1-D numpy uint8 array into packed u64 words, returning a numpy array.
/// Zero-copy input, single-allocation output — no Python list conversion.
#[pyfunction]
fn pack_bitstream_numpy<'py>(
    py: Python<'py>,
    bits: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let slice = bits.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    let tensor = bitstream::pack(slice);
    Ok(tensor.data.into_pyarray_bound(py))
}

/// Popcount on a numpy uint64 array — zero-copy input.
#[pyfunction]
fn popcount_numpy(packed: PyReadonlyArray1<'_, u64>) -> PyResult<u64> {
    let words = packed.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    Ok(simd::popcount_dispatch(words))
}

/// Unpack a numpy uint64 array back to a numpy uint8 array.
#[pyfunction]
fn unpack_bitstream_numpy<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u64>,
    original_length: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let words = packed.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read numpy array: {e}")))?;
    let tensor = bitstream::BitStreamTensor::from_words(words.to_vec(), original_length);
    let bits = bitstream::unpack(&tensor);
    Ok(bits.into_pyarray_bound(py))
}
```

**Register the new functions** in the `sc_neurocore_engine` module function (after line 25):

```rust
    m.add_function(wrap_pyfunction!(pack_bitstream_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(popcount_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bitstream_numpy, m)?)?;
```

### Python Bridge Changes

**File**: `bridge/sc_neurocore_engine/__init__.py`

Add the three new functions to the import block and `__all__`:

```python
from sc_neurocore_engine.sc_neurocore_engine import (
    __version__,
    simd_tier,
    pack_bitstream,
    unpack_bitstream,
    popcount,
    pack_bitstream_numpy,
    popcount_numpy,
    unpack_bitstream_numpy,
    Lfsr16,
    BitstreamEncoder,
    FixedPointLif,
    DenseLayer as _RustDenseLayer,
    SurrogateLif as _RustSurrogateLif,
    DifferentiableDenseLayer as _RustDiffDense,
    StochasticAttention as _RustAttention,
    StochasticGraphLayer as _RustGraphLayer,
    KuramotoSolver as _RustKuramotoSolver,
    SCPNMetrics,
)
```

(Note: The current `__init__.py` may not have explicit `__all__`. If it does, add the three new names. If it uses a wildcard import pattern, just adding them to the import statement is sufficient.)

### Verification

```python
import numpy as np
import sc_neurocore_engine as v3

bits = np.random.randint(0, 2, 1_000_000, dtype=np.uint8)

# Zero-copy pack
packed = v3.pack_bitstream_numpy(bits)
assert isinstance(packed, np.ndarray)
assert packed.dtype == np.uint64

# Zero-copy popcount
count = v3.popcount_numpy(packed)
assert count == int(bits.sum())

# Zero-copy unpack
recovered = v3.unpack_bitstream_numpy(packed, len(bits))
assert np.array_equal(bits, recovered)
```

---

## Packet Z: Batch Operations

Add batch variants that process arrays in single FFI calls, eliminating per-step overhead for LIF and encode operations.

### Rust Changes (lib.rs)

**Add two new functions** after the numpy zero-copy functions:

```rust
/// Run a LIF neuron for N steps with constant or per-step inputs.
///
/// Returns (spikes: ndarray[i32], voltages: ndarray[i16]).
/// Single FFI call eliminates per-step Python/Rust boundary crossing.
#[pyfunction]
#[pyo3(signature = (
    n_steps,
    leak_k,
    gain_k,
    i_t,
    noise_in=0,
    data_width=16,
    fraction=8,
    v_rest=0,
    v_reset=0,
    v_threshold=256,
    refractory_period=2
))]
#[allow(clippy::too_many_arguments)]
fn batch_lif_run<'py>(
    py: Python<'py>,
    n_steps: usize,
    leak_k: i16,
    gain_k: i16,
    i_t: i16,
    noise_in: i16,
    data_width: u32,
    fraction: u32,
    v_rest: i16,
    v_reset: i16,
    v_threshold: i16,
    refractory_period: i32,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i16>>) {
    let mut lif = neuron::FixedPointLif::new(
        data_width, fraction, v_rest, v_reset, v_threshold, refractory_period,
    );
    let mut spikes = Vec::with_capacity(n_steps);
    let mut voltages = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let (s, v) = lif.step(leak_k, gain_k, i_t, noise_in);
        spikes.push(s);
        voltages.push(v);
    }

    (spikes.into_pyarray_bound(py), voltages.into_pyarray_bound(py))
}

/// Run a LIF neuron for N steps with per-step current array.
///
/// `currents` is a numpy int16 array of length N.
/// Optional `noises` is a numpy int16 array of length N.
/// Returns (spikes: ndarray[i32], voltages: ndarray[i16]).
#[pyfunction]
#[pyo3(signature = (
    leak_k,
    gain_k,
    currents,
    noises=None,
    data_width=16,
    fraction=8,
    v_rest=0,
    v_reset=0,
    v_threshold=256,
    refractory_period=2
))]
#[allow(clippy::too_many_arguments)]
fn batch_lif_run_varying<'py>(
    py: Python<'py>,
    leak_k: i16,
    gain_k: i16,
    currents: PyReadonlyArray1<'py, i16>,
    noises: Option<PyReadonlyArray1<'py, i16>>,
    data_width: u32,
    fraction: u32,
    v_rest: i16,
    v_reset: i16,
    v_threshold: i16,
    refractory_period: i32,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i16>>)> {
    let curr_slice = currents.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read currents: {e}")))?;
    let noise_slice = match &noises {
        Some(n) => Some(n.as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read noises: {e}")))?),
        None => None,
    };

    let n_steps = curr_slice.len();
    let mut lif = neuron::FixedPointLif::new(
        data_width, fraction, v_rest, v_reset, v_threshold, refractory_period,
    );
    let mut spikes = Vec::with_capacity(n_steps);
    let mut voltages = Vec::with_capacity(n_steps);

    for i in 0..n_steps {
        let noise_in = noise_slice.map(|ns| ns[i]).unwrap_or(0);
        let (s, v) = lif.step(leak_k, gain_k, curr_slice[i], noise_in);
        spikes.push(s);
        voltages.push(v);
    }

    Ok((spikes.into_pyarray_bound(py), voltages.into_pyarray_bound(py)))
}

/// Bernoulli-encode a numpy float64 array into packed bitstream words.
///
/// Each value in `probs` is clamped to [0, 1] and encoded into a
/// Bernoulli bitstream of the given `length`. Returns a 2-D numpy
/// uint64 array of shape (n_probs, n_words).
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Vec<Vec<u64>>> {
    let prob_slice = probs.as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);

    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let packed: Vec<Vec<u64>> = prob_slice.iter().map(|&p| {
        let p = p.clamp(0.0, 1.0);
        let mut bits = vec![0_u8; length];
        for bit in &mut bits {
            *bit = if rand::Rng::gen::<f64>(&mut rng) < p { 1 } else { 0 };
        }
        let tensor = bitstream::pack(&bits);
        let mut data = tensor.data;
        data.resize(words, 0);
        data
    }).collect();

    Ok(packed)
}
```

**Register the new functions** in the module function:

```rust
    m.add_function(wrap_pyfunction!(batch_lif_run, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lif_run_varying, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode, m)?)?;
```

### Python Bridge Changes

**File**: `bridge/sc_neurocore_engine/__init__.py`

Add to imports:

```python
    batch_lif_run,
    batch_lif_run_varying,
    batch_encode,
```

### Verification

```python
import numpy as np
import sc_neurocore_engine as v3

# Batch LIF (constant input)
spikes, voltages = v3.batch_lif_run(100_000, leak_k=20, gain_k=256, i_t=128)
assert spikes.shape == (100_000,)
assert voltages.shape == (100_000,)
assert spikes.dtype == np.int32
assert voltages.dtype == np.int16

# Batch LIF (varying input)
currents = np.full(1000, 200, dtype=np.int16)
spikes, voltages = v3.batch_lif_run_varying(
    leak_k=20, gain_k=256, currents=currents
)
assert spikes.shape == (1000,)

# Batch encode
probs = np.array([0.3, 0.5, 0.8])
packed = v3.batch_encode(probs, length=1024, seed=0xACE1)
assert len(packed) == 3
assert len(packed[0]) == 16  # ceil(1024/64)
```

---

## Packet AA: Verilator CI Integration

Add a `cosim` job to `v3-engine.yml` that installs Verilator on Ubuntu and runs the co-simulation tests.

### File: `.github/workflows/v3-engine.yml`

**Replace the entire file** with:

```yaml
name: SC-NeuroCore v3 Engine

on:
  push:
    paths:
      - "engine/**"
      - "bridge/**"
      - "tests/**"
      - "cosim/**"
      - "examples/**"
  pull_request:
    paths:
      - "engine/**"
      - "bridge/**"
      - "tests/**"
      - "cosim/**"
      - "examples/**"

env:
  CARGO_TERM_COLOR: always
  PYTHONPATH: src

jobs:
  rust-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - name: Check formatting
        run: cargo fmt --manifest-path engine/Cargo.toml -- --check
      - name: Clippy
        run: cargo clippy --manifest-path engine/Cargo.toml -- -D warnings

  rust-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run Rust tests
        run: cargo test --manifest-path engine/Cargo.toml --tests

  equivalence:
    runs-on: ${{ matrix.os }}
    needs: [rust-lint, rust-test]
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Python dependencies
        run: |
          pip install -e ".[dev]"
          pip install maturin pytest

      - name: Build and install v3 engine
        run: |
          cd bridge
          maturin develop --release --manifest-path ../engine/Cargo.toml

      - name: Verify v3 import
        run: python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__); print(sc_neurocore_engine.simd_tier())"

      - name: Verify v2 untouched
        run: python -c "import sc_neurocore; print(sc_neurocore.__version__)"

      - name: Run equivalence tests
        run: pytest tests/equivalence/ -v --tb=short

      - name: Run v3-specific tests
        run: pytest tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py -v --tb=short

      - name: Verify wheel builds
        run: |
          cd bridge
          maturin build --release --manifest-path ../engine/Cargo.toml --out ../dist/
          ls ../dist/

  cosim:
    runs-on: ubuntu-latest
    needs: [rust-test]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Verilator
        run: |
          sudo apt-get update
          sudo apt-get install -y verilator
          verilator --version

      - name: Install Python dependencies
        run: |
          pip install -e ".[dev]"
          pip install maturin pytest

      - name: Build v3 engine
        run: |
          cd bridge
          maturin develop --release --manifest-path ../engine/Cargo.toml

      - name: Run co-simulation tests
        run: pytest cosim/ -v --tb=short

  v2-compat:
    runs-on: ubuntu-latest
    needs: [equivalence]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install v2 package
        run: pip install -e ".[dev]"
      - name: Run v2 test suite
        run: pytest tests/ --ignore=tests/equivalence -v --cov=sc_neurocore --cov-report=term --cov-fail-under=97
```

Key changes:
- Added `cosim` job with `apt-get install verilator`
- Added `tests/test_numpy_interop.py` and `tests/test_batch_ops.py` to v3-specific test list
- Added `tests/test_ir_python.py` to v3-specific test list

---

## Packet AB: Updated Benchmarks

### AB-a: Updated benchmark script

**File**: `examples/03_benchmark_report.py`

**Replace the entire file** with:

```python
"""
SC-NeuroCore v3 - Formal Benchmark Report Generator
====================================================

Runs head-to-head benchmarks between v2 (Python/NumPy) and v3 (Rust)
for all operations specified in the V3 Migration Blueprint section 8.

Includes both list-based (legacy) and numpy zero-copy variants to
show true kernel performance without FFI marshalling overhead.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python examples/03_benchmark_report.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

# -- v2 imports --
from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    vec_popcount as v2_popcount,
)
from sc_neurocore.neurons import FixedPointLIFNeuron as V2Lif
from sc_neurocore.layers import VectorizedSCLayer as V2Layer

# -- v3 imports --
import sc_neurocore_engine as v3
from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif
from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer


def benchmark(fn, n_iters: int = 1) -> float:
    """Time a function call, return seconds."""
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed


def fmt_speedup(v2_time: float, v3_time: float) -> str:
    if v3_time == 0:
        return "inf"
    ratio = v2_time / v3_time
    return f"{ratio:.1f}x"


def bench_pack(n_bits: int = 1_000_000) -> list[dict]:
    """Benchmark pack_bitstream: list vs numpy zero-copy."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, n_bits).astype(np.uint8)

    v2_time = benchmark(lambda: v2_pack(bits), n_iters=10)
    v3_list_time = benchmark(lambda: v3.pack_bitstream(bits.tolist()), n_iters=10)
    v3_np_time = benchmark(lambda: v3.pack_bitstream_numpy(bits), n_iters=10)

    return [
        {
            "operation": f"pack (list, {n_bits // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_list_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_list_time),
            "target": "6x",
        },
        {
            "operation": f"pack (numpy, {n_bits // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_np_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_np_time),
            "target": "6x",
        },
    ]


def bench_popcount(n_words: int = 1_000_000) -> list[dict]:
    """Benchmark popcount: list vs numpy zero-copy."""
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, n_words * 64).astype(np.uint8)
    packed_v2 = v2_pack(bits)
    packed_np = np.asarray(v3.pack_bitstream_numpy(bits))

    v2_time = benchmark(lambda: v2_popcount(packed_v2), n_iters=10)
    v3_list_time = benchmark(lambda: v3.popcount(packed_v2.tolist()), n_iters=10)
    v3_np_time = benchmark(lambda: v3.popcount_numpy(packed_np), n_iters=10)

    return [
        {
            "operation": f"popcount (list, {n_words // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_list_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_list_time),
            "target": "20x",
        },
        {
            "operation": f"popcount (numpy, {n_words // 1000}K)",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_np_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_np_time),
            "target": "20x",
        },
    ]


def bench_dense_forward(n_in: int = 64, n_out: int = 32, length: int = 1024) -> list[dict]:
    """Benchmark dense forward pass."""
    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 1, n_in)

    v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out, length=length)

    v2_time = benchmark(lambda: v2_layer.forward(inputs), n_iters=10)
    v3_time = benchmark(lambda: v3_layer.forward(inputs), n_iters=10)

    return [
        {
            "operation": f"dense forward ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_time),
            "target": "70x",
        },
    ]


def bench_lif_step(n_steps: int = 100_000) -> list[dict]:
    """Benchmark LIF neuron step: per-call vs batch."""

    def run_v2():
        lif = V2Lif()
        for _ in range(n_steps):
            lif.step(20, 256, 128, 0)

    def run_v3_percall():
        lif = V3Lif()
        for _ in range(n_steps):
            lif.step(20, 256, 128, 0)

    def run_v3_batch():
        return v3.batch_lif_run(n_steps, leak_k=20, gain_k=256, i_t=128)

    v2_time = benchmark(run_v2)
    v3_percall_time = benchmark(run_v3_percall)
    v3_batch_time = benchmark(run_v3_batch)

    return [
        {
            "operation": f"LIF (per-call, {n_steps // 1000}K)",
            "v2_ms": v2_time * 1000,
            "v3_ms": v3_percall_time * 1000,
            "speedup": fmt_speedup(v2_time, v3_percall_time),
            "target": "400x",
        },
        {
            "operation": f"LIF (batch, {n_steps // 1000}K)",
            "v2_ms": v2_time * 1000,
            "v3_ms": v3_batch_time * 1000,
            "speedup": fmt_speedup(v2_time, v3_batch_time),
            "target": "400x",
        },
    ]


def main():
    print("SC-NeuroCore v3 - Benchmark Report")
    print("=" * 90)
    print(f"Platform: {sys.platform}")
    print(f"SIMD tier: {v3.simd_tier()}")
    print(f"v3 version: {v3.__version__}")
    print()

    results = []
    results.extend(bench_pack())
    results.extend(bench_popcount())
    results.extend(bench_dense_forward())
    results.extend(bench_lif_step())

    # Print table
    print(f"{'Operation':<40} {'v2 (ms)':<12} {'v3 (ms)':<12} {'Speedup':<10} {'Target':<10}")
    print("-" * 84)
    for r in results:
        print(
            f"{r['operation']:<40} "
            f"{r['v2_ms']:<12.3f} "
            f"{r['v3_ms']:<12.3f} "
            f"{r['speedup']:<10} "
            f"{r['target']:<10}"
        )

    print()
    print("Note: Targets from V3_MIGRATION_BLUEPRINT.md section 8.")
    print("'list' variants cross Python/Rust FFI via list->Vec conversion (2 copies).")
    print("'numpy' variants use PyReadonlyArray for zero-copy buffer access.")
    print("'batch' variants process entire arrays in a single FFI call.")
    print("Dense forward uses rayon parallelism across neurons.")

    return results


if __name__ == "__main__":
    main()
```

### AB-b: Updated benchmark report

After running the updated benchmark script, create or update:

**File**: `docs/BENCHMARK_REPORT.md`

The report must include the actual output from running the script. It should show two tables:
1. **Phase 5 results** (for reference)
2. **Phase 6 results** (with numpy/batch variants)

Template:

```markdown
# SC-NeuroCore v3 Benchmark Report

**Version**: 3.0.0
**Date**: 2026-02-10
**SIMD Tier**: [actual tier detected]

## Phase 6 Results (NumPy Zero-Copy + Batch)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (list, 1000K) | ... | ... | ... | 6x |
| pack (numpy, 1000K) | ... | ... | ... | 6x |
| popcount (list, 1000K) | ... | ... | ... | 20x |
| popcount (numpy, 1000K) | ... | ... | ... | 20x |
| dense forward (64->32) | ... | ... | ... | 70x |
| LIF (per-call, 100K) | ... | ... | ... | 400x |
| LIF (batch, 100K) | ... | ... | ... | 400x |

## Phase 5 Results (Reference — List-Based FFI)

| Operation | v2 (ms) | v3 (ms) | Speedup | Target |
|-----------|---------|---------|---------|--------|
| pack (1M bits) | 9.545 | 32.648 | 0.3x | 6x |
| popcount (1M words) | 97.481 | 141.040 | 0.7x | 20x |
| dense forward (64->32) | 3.018 | 1.041 | 2.9x | 70x |
| LIF step (100K) | 109.683 | 28.495 | 3.8x | 400x |

## Analysis

[Actual analysis based on real numbers — explain how numpy zero-copy
eliminated the FFI overhead for pack/popcount, and how batch_lif_run
eliminated per-step overhead for LIF. Note which Blueprint §8 targets
are now met and which remain aspirational.]
```

---

## Packet AC: 3.0.0 Stable Release

### AC-a: Rust version bump

**File**: `engine/Cargo.toml` line 3

```toml
version = "3.0.0"
```

### AC-b: Python version string

**File**: `engine/src/lib.rs` line 21

```rust
    m.add("__version__", "3.0.0")?;
```

### AC-c: CHANGELOG

**File**: `CHANGELOG_V3.md`

Add a new section at the top (before `[3.0.0-rc.1]`):

```markdown
## [3.0.0] - 2026-02-10

### Phase 6: Performance Optimization & Stable Release
- **NumPy Zero-Copy**: `pack_bitstream_numpy()`, `popcount_numpy()`, `unpack_bitstream_numpy()` — eliminate FFI marshalling overhead
- **Batch Operations**: `batch_lif_run()`, `batch_lif_run_varying()`, `batch_encode()` — process arrays in single FFI calls
- **Verilator CI**: Co-simulation tests run automatically on Ubuntu runners
- **Updated Benchmarks**: Formal report showing true kernel performance with zero-copy interop
- **Bridge Version Fix**: `bridge/pyproject.toml` version now matches engine
```

### AC-d: Migration docs update

**File**: `docs/v3_migration.md`

Add the following section at the end:

```markdown
## Phase 6 Features (February 2026)

### NumPy Zero-Copy Interop

For maximum performance, use the numpy-native variants:

```python
import numpy as np
import sc_neurocore_engine as v3

bits = np.random.randint(0, 2, 1_000_000, dtype=np.uint8)
packed = v3.pack_bitstream_numpy(bits)      # Zero-copy input
count = v3.popcount_numpy(packed)            # Zero-copy input
recovered = v3.unpack_bitstream_numpy(packed, len(bits))
```

The original `pack_bitstream()` and `popcount()` functions still work
with Python lists for backward compatibility.

### Batch Operations

Process entire arrays in single FFI calls:

```python
# 100K LIF steps in one call (vs 100K per-step calls)
spikes, voltages = v3.batch_lif_run(
    100_000, leak_k=20, gain_k=256, i_t=128
)

# Per-step varying current
currents = np.array([128, 200, 150, ...], dtype=np.int16)
spikes, voltages = v3.batch_lif_run_varying(
    leak_k=20, gain_k=256, currents=currents
)
```

### CI/CD

- Verilator co-simulation runs automatically on every push (Ubuntu)
- Wheel builds on 3 OS x 4 Python versions
```

---

## New Test Files

### tests/test_numpy_interop.py

**Create this file** with the following content:

```python
"""Tests for NumPy zero-copy interop functions."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestPackBitstreamNumpy:
    """Zero-copy pack_bitstream_numpy tests."""

    def test_basic_pack(self):
        bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert isinstance(packed, np.ndarray)
        assert packed.dtype == np.uint64

    def test_roundtrip(self):
        bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        recovered = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, recovered)

    def test_large_array(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1_000_000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert packed.dtype == np.uint64
        expected_words = (1_000_000 + 63) // 64
        assert len(packed) == expected_words

    def test_consistency_with_list_variant(self):
        """Numpy and list variants must produce identical results."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1000).astype(np.uint8)
        packed_np = v3.pack_bitstream_numpy(bits)
        packed_list = v3.pack_bitstream(bits.tolist())
        np.testing.assert_array_equal(packed_np, np.array(packed_list, dtype=np.uint64))

    def test_all_zeros(self):
        bits = np.zeros(128, dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert all(w == 0 for w in packed)

    def test_all_ones(self):
        bits = np.ones(64, dtype=np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        assert packed[0] == np.uint64(0xFFFFFFFFFFFFFFFF)


class TestPopcountNumpy:
    """Zero-copy popcount_numpy tests."""

    def test_basic(self):
        packed = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 64

    def test_known_value(self):
        packed = np.array([0x0F0F0F0F0F0F0F0F], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 32

    def test_consistency_with_pack(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        count = v3.popcount_numpy(packed)
        assert count == int(bits.sum())

    def test_large_array(self):
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 1_000_000).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        count = v3.popcount_numpy(packed)
        assert count == int(bits.sum())

    def test_empty(self):
        packed = np.array([], dtype=np.uint64)
        assert v3.popcount_numpy(packed) == 0
```

### tests/test_batch_ops.py

**Create this file** with the following content:

```python
"""Tests for batch LIF and encode operations."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3
from sc_neurocore_engine import FixedPointLif


class TestBatchLifRun:
    """batch_lif_run: constant-input batch."""

    def test_basic_shape(self):
        spikes, voltages = v3.batch_lif_run(100, leak_k=20, gain_k=256, i_t=128)
        assert spikes.shape == (100,)
        assert voltages.shape == (100,)
        assert spikes.dtype == np.int32
        assert voltages.dtype == np.int16

    def test_matches_per_step(self):
        """Batch variant must produce identical results to per-step calls."""
        n_steps = 200
        leak, gain, i_t, noise = 20, 256, 128, 0

        # Per-step
        lif = FixedPointLif()
        per_step_spikes = []
        per_step_voltages = []
        for _ in range(n_steps):
            s, v = lif.step(leak, gain, i_t, noise)
            per_step_spikes.append(s)
            per_step_voltages.append(v)

        # Batch
        batch_spikes, batch_voltages = v3.batch_lif_run(
            n_steps, leak_k=leak, gain_k=gain, i_t=i_t
        )

        np.testing.assert_array_equal(batch_spikes, per_step_spikes)
        np.testing.assert_array_equal(batch_voltages, per_step_voltages)

    def test_spiking_input(self):
        """Strong current should produce spikes."""
        spikes, _ = v3.batch_lif_run(100, leak_k=20, gain_k=256, i_t=200)
        assert spikes.sum() > 0, "Should produce at least one spike with I_t=200"

    def test_zero_steps(self):
        spikes, voltages = v3.batch_lif_run(0, leak_k=20, gain_k=256, i_t=128)
        assert spikes.shape == (0,)
        assert voltages.shape == (0,)

    def test_custom_params(self):
        spikes, voltages = v3.batch_lif_run(
            50, leak_k=10, gain_k=512, i_t=100,
            data_width=16, fraction=8, v_rest=0, v_reset=0,
            v_threshold=256, refractory_period=3,
        )
        assert spikes.shape == (50,)


class TestBatchLifRunVarying:
    """batch_lif_run_varying: per-step current array."""

    def test_basic(self):
        currents = np.full(100, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20, gain_k=256, currents=currents
        )
        assert spikes.shape == (100,)
        assert voltages.shape == (100,)

    def test_with_noise(self):
        currents = np.full(50, 200, dtype=np.int16)
        noises = np.zeros(50, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20, gain_k=256, currents=currents, noises=noises
        )
        assert spikes.shape == (50,)

    def test_matches_constant_batch(self):
        """Varying with constant array == constant batch."""
        n = 100
        currents = np.full(n, 128, dtype=np.int16)
        s1, v1 = v3.batch_lif_run(n, leak_k=20, gain_k=256, i_t=128)
        s2, v2 = v3.batch_lif_run_varying(leak_k=20, gain_k=256, currents=currents)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(v1, v2)


class TestBatchEncode:
    """batch_encode: Bernoulli encoding for arrays of probabilities."""

    def test_basic_shape(self):
        probs = np.array([0.3, 0.5, 0.8])
        packed = v3.batch_encode(probs, length=1024, seed=0xACE1)
        assert len(packed) == 3
        words_per = (1024 + 63) // 64
        assert all(len(row) == words_per for row in packed)

    def test_probability_accuracy(self):
        """Encoded rates should be close to input probabilities."""
        probs = np.array([0.25, 0.5, 0.75])
        packed = v3.batch_encode(probs, length=10000, seed=42)
        for i, p in enumerate(probs):
            bits_set = sum(bin(w).count("1") for w in packed[i])
            rate = bits_set / 10000
            assert abs(rate - p) < 0.05, f"prob {p}: rate {rate}"

    def test_seed_determinism(self):
        probs = np.array([0.5, 0.5])
        p1 = v3.batch_encode(probs, length=1024, seed=42)
        p2 = v3.batch_encode(probs, length=1024, seed=42)
        assert p1 == p2

    def test_empty(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode(probs, length=1024, seed=42)
        assert len(packed) == 0
```

---

## Quality Gates

After all packets are implemented, run:

```powershell
# Rust quality gates
cd 03_CODE/sc-neurocore/engine
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --tests
cargo doc --no-deps

# Python build
cd ../bridge
..\.venv\Scripts\python -m maturin develop --release

# Python test suites
cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py -v --tb=short

# Co-sim (if Verilator available)
.\.venv\Scripts\python -m pytest cosim/ -v -rs --tb=short

# Examples
.\.venv\Scripts\python examples/01_sc_training_demo.py
.\.venv\Scripts\python examples/02_ir_compile_demo.py
.\.venv\Scripts\python examples/03_benchmark_report.py

# Wheel build
cd bridge
..\.venv\Scripts\python -m maturin build --release --manifest-path ../engine/Cargo.toml --out ../dist/
dir ..\dist\*.whl

# Version check
..\.venv\Scripts\python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__)"
# Expected: 3.0.0
```

### Expected Test Counts

| Suite | Phase 5 | Phase 6 | Delta |
|-------|---------|---------|-------|
| Rust tests | 56 | 56 | +0 (no new Rust tests needed) |
| Python tests | 56 | ~80 | +24 (numpy interop + batch ops) |
| Co-sim tests | 8 skip | 8 (run on CI) | 0 new, but now executing |
| **Total** | 112 + 8 | ~136 + 8 | +24 |

---

## Delivery Checklist

| # | Deliverable | Files | Expected |
|---|------------|-------|----------|
| 1 | bridge/pyproject.toml version fix | 1 modified | Version = 3.0.0 |
| 2 | NumPy zero-copy functions in lib.rs | 1 modified | 3 new #[pyfunction] |
| 3 | Batch operations in lib.rs | 1 modified | 3 new #[pyfunction] |
| 4 | Module registration (6 new functions) | 1 modified (lib.rs) | 6 add_function calls |
| 5 | Bridge __init__.py exports | 1 modified | 6 new names |
| 6 | v3-engine.yml with cosim + test list | 1 modified | cosim job + updated tests |
| 7 | Updated benchmark script | 1 modified | numpy + batch variants |
| 8 | Updated benchmark report | 1 modified | Real numbers from script |
| 9 | Version 3.0.0 in Cargo.toml | 1 modified | version = "3.0.0" |
| 10 | Version 3.0.0 in lib.rs | 1 modified | __version__ = "3.0.0" |
| 11 | CHANGELOG_V3.md Phase 6 section | 1 modified | [3.0.0] section |
| 12 | v3_migration.md Phase 6 section | 1 modified | Phase 6 docs |
| 13 | tests/test_numpy_interop.py | 1 **new** | 11 tests |
| 14 | tests/test_batch_ops.py | 1 **new** | 13 tests |
| | **Total new** | **2** | |
| | **Total modified** | **9** | |

### Sacred file check

After all work, verify:
```powershell
git diff src/sc_neurocore/
```
Must show **no changes**.

---

## Verilator Context

Verilator was installed in the Phase 4 follow-up session:
- Python package: `.venv\Lib\site-packages\verilator`
- Executable shim: `.venv\Scripts\verilator.exe`
- Required env vars:
  ```powershell
  $env:PATH="$PWD\.venv\Scripts;$env:PATH"
  $env:VERILATOR_ROOT="$PWD\.venv\Lib\site-packages\verilator"
  ```

If Verilator is not available in the current session, co-sim tests will skip gracefully — this is acceptable. The CI workflow handles Verilator installation via `apt-get`.

---

## Notes for Codex

1. The `numpy` crate is already in `engine/Cargo.toml` (version 0.22). The types `PyReadonlyArray1`, `PyArray1`, and `IntoPyArray` are available from `numpy::*`.

2. For PyO3 0.22 with the numpy crate, the correct return type for numpy array functions is `Bound<'py, PyArray1<T>>` and the conversion from Vec is `.into_pyarray_bound(py)`.

3. The `batch_lif_run` function creates its own `FixedPointLif` internally — it does not use the PyO3 `FixedPointLif` class. This is intentional: the batch function handles the neuron lifecycle entirely in Rust without per-step FFI crossings.

4. `rand::SeedableRng` and `rand_chacha::ChaCha8Rng` are already imported in other modules (layer.rs, bitstream.rs). In `lib.rs`, you'll need to add `use` statements for `batch_encode`.

5. The `batch_encode` function deliberately returns `Vec<Vec<u64>>` (not a numpy 2D array) to keep the signature compatible with existing packed-bitstream consumers that expect nested lists.

6. When updating `__init__.py`, preserve the existing import pattern (try/except with ImportError). Just add the new function names to the import list.

7. If `cargo clippy` warns about too_many_arguments on the batch functions, add `#[allow(clippy::too_many_arguments)]`.
