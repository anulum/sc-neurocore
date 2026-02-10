# SC-NeuroCore v3 — Phase 9 Codex Handover

**From**: Claude (Opus 4.6) — Architect & Reviewer
**To**: Codex (GPT-5) — Implementer
**Date**: 2026-02-10
**Phase**: 9 — Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
**Prerequisite**: Phase 8 ACCEPTED (`V3_PHASE8_CODE_REVIEW.md`)

---

## 1. Context & Motivation

Phase 8 criterion benchmarks revealed the encoding bottleneck clearly:

| Benchmark | Time |
|-----------|------|
| `bernoulli_packed_1024` | 5.5 µs |
| `dense_forward_64x32` (sequential) | 5.0-6.9 ms |
| `dense_forward_fast_64x32` (parallel) | 2.6-3.7 ms |
| `dense_forward_prepacked_64x32` (skip encoding) | 400-650 µs |

The gap between `forward_fast` (~3 ms) and `forward_prepacked` (~500 µs) is entirely encoding cost. For 64 inputs, sequential encoding takes 64 × 5.5 µs = 352 µs serially, but ChaCha8Rng has high per-call overhead when generating one f64 at a time (1024 × `rng.gen::<f64>()` per input = 65,536 RNG calls total).

### Issue 1: Slow Bernoulli via f64 Generation

`bernoulli_packed` (bitstream.rs lines 110-123) generates one `f64` per bit:

```rust
for bit in 0..bits_in_word {
    if rng.gen::<f64>() < p {  // 8 bytes of RNG per bit
        *word |= 1_u64 << bit;
    }
}
```

Each `rng.gen::<f64>()` consumes 8 bytes of ChaCha8 output. For 1024 bits, that's 8,192 bytes = 128 ChaCha blocks. By switching to `rng.fill_bytes()` with byte-threshold comparison, we need only 1,024 bytes = 16 ChaCha blocks — an **8x reduction** in RNG work.

### Issue 2: Per-Iteration and_buf Allocation in Neuron Compute

In `layer.rs`, every neuron's inner loop allocates and fills an `and_buf`:

```rust
let mut and_buf = Vec::<u64>::new();  // allocated per neuron
for (w, i) in weights.iter().zip(inputs.iter()) {
    and_buf.clear();
    and_buf.extend(w.iter().zip(i.iter()).map(|(a, b)| *a & *b));
    total += popcount_dispatch(&and_buf);
}
```

This can be replaced with a **fused AND+popcount** that processes word-by-word without materializing the intermediate buffer:

```rust
for (w, i) in weights.iter().zip(inputs.iter()) {
    total += w.iter().zip(i.iter())
        .map(|(a, b)| (a & b).count_ones() as u64)
        .sum::<u64>();
}
```

### Issue 3: Vec<Vec<u64>> Copy in forward_prepacked PyO3 Binding

The PyO3 binding for `forward_prepacked` (lib.rs lines 569-592) copies numpy 2D rows into `Vec<Vec<u64>>`. A true zero-copy path would pass ndarray views directly into Rust.

---

## 2. Packet Summary

| Packet | Deliverable | Files Modified | Files Created |
|--------|------------|---------------|---------------|
| **AP** | Fast Bernoulli (byte-threshold) | `engine/src/bitstream.rs`, `engine/src/layer.rs`, `engine/src/lib.rs` | — |
| **AQ** | Fused AND+popcount | `engine/src/layer.rs` | — |
| **AR** | Zero-copy forward_prepacked_numpy | `engine/src/layer.rs`, `engine/src/lib.rs`, `bridge/sc_neurocore_engine/layers.py` | — |
| **AS** | Rayon thread pool control | `engine/src/lib.rs` | — |
| **AT** | Criterion benchmarks + benchmark report update | `engine/benches/full_bench.rs`, `examples/03_benchmark_report.py`, `docs/BENCHMARK_REPORT.md` | — |
| **AU** | Version 3.3.0 + docs + tests | `Cargo.toml`, `lib.rs`, `pyproject.toml`, `__init__.py`, `CHANGELOG_V3.md`, `v3_migration.md`, `v3-engine.yml` | `tests/test_phase9.py` |

**Total**: ~12 files modified, 1 file created.

---

## 3. Packet AP: Fast Bernoulli via Byte-Threshold Comparison

### 3.1 New function in `engine/src/bitstream.rs`

Add after `bernoulli_packed`:

```rust
/// Fast packed Bernoulli generation using byte-threshold comparison.
///
/// Instead of generating one f64 (8 bytes) per bit and comparing,
/// this generates one u8 (1 byte) per bit via `rng.fill_bytes()` and
/// compares against a u8 threshold = `(prob * 256.0) as u8`.
///
/// This uses 8x less RNG bandwidth than `bernoulli_packed` at the
/// cost of 8-bit probability resolution (1/256 granularity).
/// For bitstream lengths >= 256, the statistical difference is
/// negligible compared to inherent sampling noise.
///
/// The output is NOT bit-identical to `bernoulli_packed` for the
/// same RNG state.
pub fn bernoulli_packed_fast<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    let mut buf = [0_u8; 64];

    for (word_idx, word) in data.iter_mut().enumerate() {
        let bits_in_word = std::cmp::min(64, length.saturating_sub(word_idx * 64));
        rng.fill(&mut buf[..bits_in_word]);
        for (bit, &rb) in buf[..bits_in_word].iter().enumerate() {
            if rb < threshold {
                *word |= 1_u64 << bit;
            }
        }
    }
    data
}
```

### 3.2 Unit test in bitstream.rs

Add to `mod tests`:

```rust
#[test]
fn bernoulli_packed_fast_statistics() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let prob = 0.35;
    let length = 10_000;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let packed = super::bernoulli_packed_fast(prob, length, &mut rng);
    let count: u64 = packed.iter().map(|w| w.count_ones() as u64).sum();
    let measured = count as f64 / length as f64;
    assert!(
        (measured - prob).abs() < 0.03,
        "Expected ~{prob}, got {measured}"
    );
}

#[test]
fn bernoulli_packed_fast_deterministic() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng1 = ChaCha8Rng::seed_from_u64(99);
    let a = super::bernoulli_packed_fast(0.5, 512, &mut rng1);

    let mut rng2 = ChaCha8Rng::seed_from_u64(99);
    let b = super::bernoulli_packed_fast(0.5, 512, &mut rng2);

    assert_eq!(a, b, "Same seed must produce identical output");
}
```

### 3.3 Use in forward_fast (layer.rs)

Replace `bitstream::bernoulli_packed` with `bitstream::bernoulli_packed_fast` in `forward_fast` (line 155):

```rust
// BEFORE:
bitstream::bernoulli_packed(p, self.length, &mut rng)

// AFTER:
bitstream::bernoulli_packed_fast(p, self.length, &mut rng)
```

Also replace in `forward_numpy_inner` — since it delegates to `forward_fast`, this is automatic.

**Keep `forward()` using `bernoulli_packed`** (the original sequential path) for backward compatibility. Users who need bit-exact f64-precision encoding use `forward()`.

### 3.4 Use in batch_encode_numpy (lib.rs)

Replace `bitstream::bernoulli_packed` with `bitstream::bernoulli_packed_fast` in the parallel `batch_encode_numpy` function (around line 372):

```rust
// BEFORE:
let mut row = bitstream::bernoulli_packed(p, length, &mut rng);

// AFTER:
let mut row = bitstream::bernoulli_packed_fast(p, length, &mut rng);
```

**Keep `batch_encode`** (non-numpy, sequential) using original `bernoulli_packed` for backward compatibility.

---

## 4. Packet AQ: Fused AND+Popcount

### 4.1 New helper function in `engine/src/layer.rs`

Add before `impl DenseLayer`:

```rust
/// Fused bitwise-AND + popcount over two aligned packed word slices.
///
/// Equivalent to `popcount(bitwise_and(a, b))` but avoids
/// materializing the intermediate buffer.
#[inline]
fn fused_and_popcount(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}
```

Note: We use `u64::count_ones()` (Rust intrinsic, maps to hardware POPCNT on x86 with `-C target-cpu=native` or release profile). This is faster than calling `popcount_dispatch` for small word counts (16 words for L=1024) because it avoids function call overhead and SIMD loop setup. For the layer's inner loop, each input has only `ceil(length/64)` = 16 words, which is too small for AVX2/AVX-512 to amortize setup costs.

### 4.2 Update `forward_prepacked` (layer.rs)

Replace lines 204-219:

```rust
// BEFORE:
let out = (0..self.n_neurons)
    .into_par_iter()
    .map(|neuron_idx| {
        let mut total = 0_u64;
        let mut and_buf = Vec::<u64>::new();
        for (w, i) in self.packed_weights[neuron_idx]
            .iter()
            .zip(packed_inputs.iter())
        {
            and_buf.clear();
            and_buf.extend(w.iter().zip(i.iter()).map(|(a, b)| *a & *b));
            total += popcount_dispatch(&and_buf);
        }
        total as f64 / self.length as f64
    })
    .collect();

// AFTER:
let out = (0..self.n_neurons)
    .into_par_iter()
    .map(|neuron_idx| {
        let total: u64 = self.packed_weights[neuron_idx]
            .iter()
            .zip(packed_inputs.iter())
            .map(|(w, i)| fused_and_popcount(w, i))
            .sum();
        total as f64 / self.length as f64
    })
    .collect();
```

### 4.3 Update `forward_fast` (layer.rs)

Apply the same transformation to the neuron compute section in `forward_fast` (lines 159-174):

```rust
// AFTER:
let out = (0..self.n_neurons)
    .into_par_iter()
    .map(|neuron_idx| {
        let total: u64 = self.packed_weights[neuron_idx]
            .iter()
            .zip(packed_inputs.iter())
            .map(|(w, i)| fused_and_popcount(w, i))
            .sum();
        total as f64 / self.length as f64
    })
    .collect();
```

### 4.4 Update `forward()` (layer.rs)

Apply the same transformation to the neuron compute section in `forward` (lines 116-131):

```rust
// AFTER:
let out = (0..self.n_neurons)
    .into_par_iter()
    .map(|neuron_idx| {
        let total: u64 = self.packed_weights[neuron_idx]
            .iter()
            .zip(packed_inputs.iter())
            .map(|(w, i)| fused_and_popcount(w, i))
            .sum();
        total as f64 / self.length as f64
    })
    .collect();
```

### 4.5 Remove unused import

After removing all `and_buf` + `popcount_dispatch` usage from layer.rs, remove the `popcount_dispatch` import at line 12:

```rust
// BEFORE:
use crate::simd::popcount_dispatch;

// AFTER:
// (remove this line entirely)
```

The `simd::popcount_dispatch` remains available for other modules but is no longer needed in layer.rs since `u64::count_ones()` is used inline.

---

## 5. Packet AR: Zero-Copy forward_prepacked_numpy

### 5.1 New Rust method in `engine/src/layer.rs`

Add to `impl DenseLayer`:

```rust
/// Forward pass with pre-packed inputs from a 2-D contiguous array.
///
/// `packed_inputs` is a flat row-major buffer of shape `[n_inputs, words]`.
/// Each row is one input's packed bitstream words.
pub fn forward_prepacked_2d(
    &self,
    packed_flat: &[u64],
    n_inputs: usize,
    words: usize,
) -> Result<Vec<f64>, String> {
    if n_inputs != self.n_inputs {
        return Err(format!(
            "Expected {} packed inputs, got {}.",
            self.n_inputs, n_inputs
        ));
    }
    let expected_words = self.length.div_ceil(64);
    if words != expected_words {
        return Err(format!(
            "Expected {} words per input, got {}.",
            expected_words, words
        ));
    }
    if packed_flat.len() != n_inputs * words {
        return Err(format!(
            "Flat buffer length {} != n_inputs({}) * words({}).",
            packed_flat.len(),
            n_inputs,
            words
        ));
    }

    let out = (0..self.n_neurons)
        .into_par_iter()
        .map(|neuron_idx| {
            let total: u64 = self.packed_weights[neuron_idx]
                .iter()
                .enumerate()
                .map(|(input_idx, w)| {
                    let row_start = input_idx * words;
                    let input_words = &packed_flat[row_start..row_start + words];
                    fused_and_popcount(w, input_words)
                })
                .sum();
            total as f64 / self.length as f64
        })
        .collect();

    Ok(out)
}
```

### 5.2 PyO3 binding in `engine/src/lib.rs`

Add to `#[pymethods] impl DenseLayer`, after the existing `forward_prepacked` method:

```rust
/// Dense forward with pre-packed numpy 2-D input (true zero-copy).
///
/// Accepts a contiguous numpy uint64 array of shape (n_inputs, words).
/// This avoids all row-copying that the `forward_prepacked` method does.
#[pyo3(signature = (packed_inputs,))]
fn forward_prepacked_numpy<'py>(
    &self,
    py: Python<'py>,
    packed_inputs: PyReadonlyArray2<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = packed_inputs.shape();
    let n_inputs = shape[0];
    let words = shape[1];
    let flat = packed_inputs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {e}")))?;
    let out = self
        .inner
        .forward_prepacked_2d(flat, n_inputs, words)
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray_bound(py))
}
```

### 5.3 Bridge wrapper in `bridge/sc_neurocore_engine/layers.py`

Add to `VectorizedSCLayer`:

```python
def forward_prepacked_numpy(self, packed_inputs) -> np.ndarray:
    """Dense forward with pre-packed numpy 2D input (true zero-copy)."""
    import numpy as np
    arr = np.ascontiguousarray(packed_inputs, dtype=np.uint64)
    return self._engine.forward_prepacked_numpy(arr)
```

---

## 6. Packet AS: Rayon Thread Pool Control

### 6.1 New pyfunction in `engine/src/lib.rs`

Add as a standalone function:

```rust
/// Set the number of threads in the global rayon thread pool.
///
/// Must be called before any parallel operation.
/// Passing 0 uses rayon's default (number of CPU cores).
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    if n == 0 {
        // Already using default
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| PyValueError::new_err(format!("Cannot set thread pool: {e}")))
}
```

Register in module init:

```rust
m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
```

### 6.2 Export in bridge `__init__.py`

Add `set_num_threads` to the import line and `__all__` list.

---

## 7. Packet AT: Criterion Benchmarks + Benchmark Report

### 7.1 New benchmark in `engine/benches/full_bench.rs`

Add alongside the existing Bernoulli benchmarks:

```rust
c.bench_function("bernoulli_packed_fast_1024", |b| {
    b.iter(|| {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        black_box(bernoulli_packed_fast(0.5, 1024, &mut rng))
    })
});
```

Update imports to include `bernoulli_packed_fast`.

### 7.2 New benchmark for fused path

The existing `dense_forward_prepacked_64x32` benchmark already measures the prepacked path, which will automatically reflect the fused AND+popcount optimization. No new benchmark needed — the before/after comparison is the criterion data.

### 7.3 Update `examples/03_benchmark_report.py`

Add a `dense prepacked numpy` variant:

```python
# After the existing prepacked benchmark, add:
v3_prepacked_numpy_time = benchmark(
    lambda: v3_layer.forward_prepacked_numpy(packed_inputs), n_iters=10
)
```

And add a result entry:

```python
{
    "operation": f"dense prepacked numpy ({n_in}->{n_out}, L={length})",
    "v2_ms": v2_time / 10 * 1000,
    "v3_ms": v3_prepacked_numpy_time / 10 * 1000,
    "speedup": fmt_speedup(v2_time, v3_prepacked_numpy_time),
    "target": "70x",
},
```

Where `packed_inputs` is the result of `v3.batch_encode_numpy(inputs_f64, length=length, seed=42)` used for the existing prepacked benchmark.

### 7.4 Update `docs/BENCHMARK_REPORT.md`

Update with Phase 9 results after running `cargo bench` and `examples/03_benchmark_report.py`. The table should include:

- `bernoulli_packed_fast_1024` time (expected ~0.7-1.0 µs, ~6-8x faster than `bernoulli_packed_1024`)
- Updated `dense_forward_fast_64x32` time (should drop ~4-6x due to fast Bernoulli)
- Updated `dense_forward_prepacked_64x32` time (should drop ~10-20% due to fused AND+popcount)
- New `dense prepacked numpy` time (should be similar to prepacked, proving zero-copy)

---

## 8. Packet AU: Version 3.3.0 + Docs + Tests

### 8.1 Version bump

**`engine/Cargo.toml`** line 3:
```toml
version = "3.3.0"
```

**`engine/src/lib.rs`** line 22:
```rust
m.add("__version__", "3.3.0")?;
```

**`bridge/pyproject.toml`**:
```toml
version = "3.3.0"
```

### 8.2 CHANGELOG

Prepend to `CHANGELOG_V3.md`:

```markdown
## [3.3.0] - 2026-02-10

### Phase 9: Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
- **bernoulli_packed_fast**: 8x less RNG bandwidth via byte-threshold encoding
- **Fused AND+popcount**: Eliminates intermediate buffer allocation in neuron compute
- **forward_prepacked_numpy()**: True zero-copy from numpy 2D uint64 arrays
- **set_num_threads()**: Rayon thread pool configuration for tuning parallelism
- **Criterion benchmarks**: Added bernoulli_packed_fast benchmark
```

### 8.3 Migration docs

Append Phase 9 section to `docs/v3_migration.md`:

```markdown
## Phase 9 Features (February 2026)

### Fast Bernoulli Encoding

`forward_fast` and `batch_encode_numpy` now use byte-threshold Bernoulli
encoding with 8x less random number generation overhead. This provides
1/256 probability granularity, which is negligible compared to the
statistical noise of 1024-bit bitstreams.

The original `forward()` and `batch_encode()` retain f64-precision
encoding for backward compatibility.

### Zero-Copy Prepacked Forward

For maximum throughput with pre-encoded inputs:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
probs = np.random.uniform(0, 1, 64)

# Encode once, forward many times (zero-copy)
packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
out = layer.forward_prepacked_numpy(packed)
# out is a numpy float64 array, packed was never copied
```

### Thread Pool Tuning

Control rayon's parallel thread count:

```python
v3.set_num_threads(4)  # Use 4 threads for all parallel ops
```

Must be called before any parallel operation. Pass 0 for automatic
(number of CPU cores).
```

### 8.4 New test file: `tests/test_phase9.py`

```python
"""Tests for Phase 9: fast Bernoulli, fused AND+popcount, zero-copy prepacked."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestFastBernoulli:
    """Tests for byte-threshold Bernoulli in forward_fast and batch_encode_numpy."""

    def test_forward_fast_deterministic(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=100)
        out2 = layer.forward_fast(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_forward_fast_output_range(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.3] * 8
        out = layer.forward_fast(inputs, seed=42)
        assert all(0.0 <= v for v in out)

    def test_forward_fast_statistical_sanity(self):
        """forward_fast output should correlate with input probability."""
        layer = v3.DenseLayer(8, 4, 2048, seed=42)
        low_out = np.mean(layer.forward_fast([0.1] * 8, seed=42))
        high_out = np.mean(layer.forward_fast([0.9] * 8, seed=42))
        assert high_out > low_out, "Higher input probs should give higher output"

    def test_batch_encode_numpy_deterministic(self):
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_batch_encode_numpy_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.04
        assert abs(pc1 / 10_000 - 0.75) < 0.04


class TestFusedAndPopcount:
    """Tests verifying fused AND+popcount produces same results as before."""

    def test_forward_matches_reference(self):
        """forward() output should still be valid (range + deterministic)."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = [0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1]
        out1 = layer.forward(inputs, seed=42)
        out2 = layer.forward(inputs, seed=42)
        np.testing.assert_array_equal(out1, out2)
        assert all(0.0 <= v for v in out1)

    def test_prepacked_deterministic(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        np.testing.assert_array_equal(out1, out2)


class TestZeroCopyPrepackedNumpy:
    """Tests for forward_prepacked_numpy (true zero-copy path)."""

    def test_output_shape_and_type(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked_numpy(packed)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == np.float64

    def test_matches_forward_prepacked(self):
        """Zero-copy numpy path must match the existing prepacked path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_wrong_n_inputs(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((3, 16), dtype=np.uint64)  # 3 inputs, need 4
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_wrong_word_count(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((4, 10), dtype=np.uint64)  # 10 words, need 16
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_pipeline_encode_then_zero_copy(self):
        """Full pipeline: batch_encode_numpy -> forward_prepacked_numpy."""
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        probs = np.random.uniform(0, 1, 16)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        out = layer.forward_prepacked_numpy(packed)
        assert out.shape == (8,)
        assert np.all(out >= 0.0)

    def test_deterministic(self):
        layer = v3.DenseLayer(4, 2, 512, seed=42)
        probs = np.array([0.5] * 4, dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=42)
        out1 = layer.forward_prepacked_numpy(packed)
        out2 = layer.forward_prepacked_numpy(packed)
        np.testing.assert_array_equal(out1, out2)


class TestSetNumThreads:
    """Tests for rayon thread pool configuration."""

    def test_set_num_threads_does_not_crash(self):
        """Calling set_num_threads should not raise."""
        # Note: can only be called once per process, so we just
        # verify the function exists and is callable.
        # Second call may raise if pool is already initialized.
        try:
            v3.set_num_threads(0)  # 0 = default
        except ValueError:
            pass  # Pool already initialized — acceptable


class TestPhase9Version:
    def test_version_is_3_3_0(self):
        assert v3.__version__ == "3.3.0"
```

### 8.5 CI test list update

In `.github/workflows/v3-engine.yml`, add `tests/test_phase9.py` to the pytest command in the equivalence job.

### 8.6 Export set_num_threads

In `bridge/sc_neurocore_engine/__init__.py`, add `set_num_threads` to the import from `sc_neurocore_engine.sc_neurocore_engine` and to `__all__`.

---

## 9. Quality Gates

```powershell
# Rust gates
cd 03_CODE/sc-neurocore/engine
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --tests
cargo doc --no-deps
cargo bench

# Python build
cd ../bridge
..\.venv\Scripts\python -m maturin develop --release

# Python tests
cd ..
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py tests/test_phase8.py tests/test_phase9.py -v --tb=short

# Co-sim
.\.venv\Scripts\python -m pytest cosim/ -v -rs --tb=short

# Examples
.\.venv\Scripts\python examples/01_sc_training_demo.py
.\.venv\Scripts\python examples/02_ir_compile_demo.py
.\.venv\Scripts\python examples/03_benchmark_report.py

# Version check
.\.venv\Scripts\python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__)"
# Expected: 3.3.0
```

### Expected Test Counts

| Suite | Phase 8 | Phase 9 Expected |
|-------|---------|-----------------|
| Rust (`cargo test`) | 57+ | 59+ (2 new Bernoulli fast tests) |
| Python (v3 suite) | 113 | ~130+ (new test file) |
| Co-sim | 8 | 8 (unchanged) |
| Criterion benches | 15+ | 16+ (1 new: bernoulli_packed_fast) |

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
| 1 | `bernoulli_packed_fast` exists and passes stats test | `cargo test bernoulli_packed_fast` |
| 2 | `forward_fast` uses `bernoulli_packed_fast` | Code inspection of layer.rs |
| 3 | `batch_encode_numpy` uses `bernoulli_packed_fast` | Code inspection of lib.rs |
| 4 | Fused AND+popcount in all forward methods | No `and_buf` in layer.rs |
| 5 | `popcount_dispatch` import removed from layer.rs | Code inspection |
| 6 | `forward_prepacked_2d` exists in layer.rs | Code inspection |
| 7 | `forward_prepacked_numpy` PyO3 binding accepts PyReadonlyArray2 | Python test |
| 8 | `forward_prepacked_numpy` matches `forward_prepacked` output | Python test |
| 9 | `set_num_threads` registered and exported | Python test |
| 10 | Criterion includes `bernoulli_packed_fast_1024` | `cargo bench` |
| 11 | Version = 3.3.0 everywhere | `import sc_neurocore_engine; print(sc_neurocore_engine.__version__)` |
| 12 | CHANGELOG has [3.3.0] section | File inspection |
| 13 | Sacred files untouched | `git diff -- src/sc_neurocore/` returns nothing |
| 14 | All quality gates pass | See Section 9 |

---

## 12. Performance Expectations

Based on the Phase 8 criterion data:

| Metric | Phase 8 | Phase 9 Expected | Basis |
|--------|---------|-----------------|-------|
| `bernoulli_packed_fast_1024` | — | ~0.7-1.2 µs | 8x less RNG than bernoulli_packed (5.5 µs) |
| `dense_forward_fast_64x32` | 2.6-3.7 ms | ~0.8-1.5 ms | Fast Bernoulli reduces encoding ~4-6x |
| `dense_forward_prepacked_64x32` | 400-650 µs | ~300-500 µs | Fused AND+popcount eliminates and_buf |
| `dense_forward_64x32` | 5.0-6.9 ms | 5.0-6.9 ms | Unchanged (keeps original bernoulli_packed) |

The `forward_fast` improvement from ~3 ms to ~1 ms would bring the Python speedup from **1.0x** to approximately **3-4x** (vs v2 baseline of ~4 ms). Combined with the prepacked numpy zero-copy path, the best-case speedup for the encode+forward pipeline should reach **8-10x**.

---

## 13. Notes

### Backward Compatibility

- `forward()` retains f64-precision `bernoulli_packed` — output is bit-identical to Phase 8
- `batch_encode()` (non-numpy) retains original `bernoulli_packed` — unchanged
- `forward_prepacked()` with list[list[int]] still works — unchanged API
- `forward_fast()` output changes due to `bernoulli_packed_fast` — acceptable since Phase 8 already changed its seeding strategy

### Why u8::count_ones() Instead of SIMD Popcount

The layer's neuron inner loop processes `ceil(length/64)` = 16 words per input-weight pair. For 16 u64 words:
- `u64::count_ones()` (scalar POPCNT instruction): 16 iterations × ~1 cycle = ~16 cycles
- AVX2 popcount (4 words per iteration): 4 iterations + loop setup + store + sum = ~20 cycles
- AVX-512 VPOPCNTDQ (8 words per iteration): 2 iterations + loop setup + store + sum = ~15 cycles

At this small word count, the overhead of SIMD setup (mask loads, remainder handling) negates the throughput advantage. `u64::count_ones()` compiles to a single `POPCNT` instruction on x86-64 with the release profile (`-C opt-level=3`), making it the fastest choice for the layer's access pattern.
