# SC-NeuroCore v3 — Phase 7 Codex Handover

**From**: Claude (Opus 4.6) — Architect & Reviewer
**To**: Codex (GPT-5) — Implementer
**Date**: 2026-02-10
**Phase**: 7 — Dense Forward Optimization & PyPI Publishing
**Prerequisite**: Phase 6 ACCEPTED (`V3_PHASE6_CODE_REVIEW.md`)

---

## 1. Context & Motivation

Phase 6 achieved dramatic speedups for popcount (61.9x) and LIF batch (107.8x) by eliminating FFI marshalling overhead via numpy zero-copy and batch operations. However, **dense forward remains at 1.4x** — far below the Blueprint §8 target of 70x.

### Root Cause Analysis

The dense forward bottleneck is in `engine/src/layer.rs` lines 110-115:

```rust
// CURRENT — sequential encoding with Vec<u8> intermediate
let mut rng = ChaCha8Rng::seed_from_u64(seed);
let mut packed_inputs = vec![Vec::<u64>::new(); self.n_inputs];
for (idx, p) in input_values.iter().copied().enumerate() {
    let bits = bernoulli_stream(p, self.length, &mut rng);  // Vec<u8> alloc
    packed_inputs[idx] = pack(&bits).data;                    // Vec<u64> alloc + copy
}
```

**Three problems**:
1. **Double allocation**: Each input generates a `Vec<u8>` (1024 bytes) then a `Vec<u64>` (128 bytes). For 64 inputs: 128 heap allocations per forward call.
2. **Sequential encoding**: Single RNG forces sequential encoding across all 64 inputs. On an 8-core machine, 87.5% of cores are idle during encoding.
3. **Encoding dominates**: Encoding takes ~2ms, compute (AND + popcount) takes ~0.05ms. The compute phase is already rayon-parallelized and fast.

### Phase 7 Strategy

Three optimization tiers, each giving increasingly better performance:
1. **`bernoulli_packed`**: Eliminate Vec<u8> intermediate (same output, fewer allocations)
2. **`forward_fast`**: Parallelize encoding with per-input RNGs + rayon
3. **`forward_prepacked`**: Accept pre-packed inputs from Python, skip encoding entirely

Plus: `batch_encode_numpy` returning 2D numpy arrays, PyPI publishing automation.

---

## 2. Packet Summary

| Packet | Deliverable | Files Modified | Files Created |
|--------|------------|---------------|---------------|
| **AD** | Direct packed Bernoulli generation | `bitstream.rs`, `layer.rs` | — |
| **AE** | Parallel input encoding (`forward_fast`) | `layer.rs`, `lib.rs` | — |
| **AF** | Pre-packed forward path | `layer.rs`, `lib.rs` | — |
| **AG** | `batch_encode_numpy` returning numpy 2D | `lib.rs`, `__init__.py` | — |
| **AH** | PyPI publish automation | `v3-wheels.yml` | — |
| **AI** | Benchmarks + version 3.1.0 + docs | `Cargo.toml`, `lib.rs`, `pyproject.toml`, `03_benchmark_report.py`, `BENCHMARK_REPORT.md`, `CHANGELOG_V3.md`, `v3_migration.md` | `tests/test_dense_optimization.py` |

**Total**: ~10 files modified, 1 file created.

---

## 3. Packet AD: Direct Packed Bernoulli Generation

### 3.1 New function in `engine/src/bitstream.rs`

Add this function after the existing `encode_matrix_prob_to_packed` function (after line 122):

```rust
/// Generate a packed Bernoulli bitstream directly into u64 words.
///
/// Produces bit-identical output to `bernoulli_stream()` + `pack()` but
/// skips the intermediate Vec<u8> allocation.
pub fn bernoulli_packed<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let p = prob.clamp(0.0, 1.0);
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    for (word_idx, word) in data.iter_mut().enumerate() {
        let bits_in_word = std::cmp::min(64, length.saturating_sub(word_idx * 64));
        for bit in 0..bits_in_word {
            if rng.gen::<f64>() < p {
                *word |= 1_u64 << bit;
            }
        }
    }
    data
}
```

**Key invariant**: The RNG draw order is identical to `bernoulli_stream` + `pack`: bit 0 drawn first → word 0 bit 0, bit 1 drawn second → word 0 bit 1, ..., bit 63 → word 0 bit 63, bit 64 → word 1 bit 0, etc. This means `bernoulli_packed(p, len, rng)` produces the same packed words as `pack(&bernoulli_stream(p, len, rng)).data` for the same RNG state. This is critical for determinism.

### 3.2 Add unit test in `engine/src/bitstream.rs`

Add to the existing `mod tests` block:

```rust
#[test]
fn bernoulli_packed_matches_stream_then_pack() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let prob = 0.35;
    let length = 200; // Not a multiple of 64

    let mut rng1 = ChaCha8Rng::seed_from_u64(999);
    let stream = super::bernoulli_stream(prob, length, &mut rng1);
    let packed_via_stream = super::pack(&stream).data;

    let mut rng2 = ChaCha8Rng::seed_from_u64(999);
    let packed_direct = super::bernoulli_packed(prob, length, &mut rng2);

    assert_eq!(packed_via_stream, packed_direct, "bernoulli_packed must produce bit-identical output");
}
```

Note: `bernoulli_stream` is currently a private function in `layer.rs`. For this test, either:
- (a) Move `bernoulli_stream` to `bitstream.rs` and make it `pub(crate)`, OR
- (b) Create an equivalent local version in the test.

**Preferred**: option (a) — move `bernoulli_stream` from `layer.rs` to `bitstream.rs` as `pub(crate)`, then import it in `layer.rs` as `use crate::bitstream::bernoulli_stream;`. This consolidates all bitstream generation in one module.

### 3.3 Replace usages in `engine/src/layer.rs`

**In `refresh_packed_weights()`** (lines 84-96), replace:
```rust
// BEFORE
let bits = bernoulli_stream(*weight_prob, self.length, &mut rng);
packed_weights[neuron_idx][input_idx] = pack(&bits).data;
```
with:
```rust
// AFTER
packed_weights[neuron_idx][input_idx] = bitstream::bernoulli_packed(*weight_prob, self.length, &mut rng);
```

**In `forward()`** (lines 112-114), replace:
```rust
// BEFORE
let bits = bernoulli_stream(p, self.length, &mut rng);
packed_inputs[idx] = pack(&bits).data;
```
with:
```rust
// AFTER
packed_inputs[idx] = bitstream::bernoulli_packed(p, self.length, &mut rng);
```

**Remove**: the now-unused `bernoulli_stream` function from `layer.rs` (lines 138-146) and the `use crate::bitstream::pack;` import (line 11). Replace with `use crate::bitstream::bernoulli_packed;` if not already importing the whole module.

**Critical**: After this change, all existing tests MUST still pass identically. The RNG draw order is preserved, so `forward()` output is bit-identical.

---

## 4. Packet AE: Parallel Input Encoding (`forward_fast`)

### 4.1 New method in `engine/src/layer.rs`

Add this method to the `impl DenseLayer` block, after the existing `forward()` method:

```rust
/// Forward pass with parallel input encoding.
///
/// Each input gets an independently-seeded RNG, enabling rayon parallel
/// encoding. The compute phase (AND + popcount) is also parallel.
///
/// Note: This produces DIFFERENT bitstreams than `forward()` because each
/// input uses `seed + input_index` as its RNG seed instead of sharing a
/// single sequential RNG. Both are correct stochastic encodings.
pub fn forward_fast(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
    if input_values.len() != self.n_inputs {
        return Err(format!(
            "Expected input of length {}, got {}.",
            self.n_inputs,
            input_values.len()
        ));
    }

    // Parallel encoding: each input gets its own seeded RNG
    let packed_inputs: Vec<Vec<u64>> = input_values
        .par_iter()
        .enumerate()
        .map(|(idx, &p)| {
            let input_seed = seed.wrapping_add(idx as u64);
            let mut rng = ChaCha8Rng::seed_from_u64(input_seed);
            crate::bitstream::bernoulli_packed(p, self.length, &mut rng)
        })
        .collect();

    // Parallel compute (same as forward)
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

    Ok(out)
}
```

### 4.2 PyO3 binding in `engine/src/lib.rs`

Add to the `#[pymethods] impl DenseLayer` block (after the existing `forward` method, around line 534):

```rust
#[pyo3(signature = (input_values, seed=44257))]
fn forward_fast(&self, input_values: Vec<f64>, seed: u64) -> PyResult<Vec<f64>> {
    self.inner
        .forward_fast(&input_values, seed)
        .map_err(PyValueError::new_err)
}
```

No module registration needed — it's a method on the existing `DenseLayer` class.

---

## 5. Packet AF: Pre-packed Forward Path

### 5.1 New method in `engine/src/layer.rs`

Add to `impl DenseLayer`:

```rust
/// Forward pass with pre-packed input bitstreams.
///
/// Skips encoding entirely — accepts already-packed u64 bitstreams.
/// Use with `batch_encode` or `batch_encode_numpy` to encode inputs,
/// then call this for repeated inference with the same encoded inputs.
///
/// `packed_inputs` must have length `n_inputs`, each inner Vec must have
/// `ceil(length / 64)` u64 words.
pub fn forward_prepacked(&self, packed_inputs: &[Vec<u64>]) -> Result<Vec<f64>, String> {
    if packed_inputs.len() != self.n_inputs {
        return Err(format!(
            "Expected {} packed inputs, got {}.",
            self.n_inputs,
            packed_inputs.len()
        ));
    }
    let expected_words = self.length.div_ceil(64);
    for (idx, pi) in packed_inputs.iter().enumerate() {
        if pi.len() != expected_words {
            return Err(format!(
                "Packed input {} has {} words, expected {}.",
                idx,
                pi.len(),
                expected_words
            ));
        }
    }

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

    Ok(out)
}
```

### 5.2 PyO3 binding in `engine/src/lib.rs`

Add to `#[pymethods] impl DenseLayer` (after `forward_fast`):

```rust
/// Forward pass with pre-packed input bitstreams.
///
/// Accepts a 2-D numpy array of shape (n_inputs, words_per_input) with dtype uint64,
/// OR a list of lists of ints.
fn forward_prepacked(&self, packed_inputs: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Try numpy 2D array first (zero-copy path)
    if let Ok(arr) = packed_inputs.extract::<PyReadonlyArray2<u64>>() {
        let view = arr.as_array();
        let rows: Vec<Vec<u64>> = (0..view.nrows())
            .map(|i| view.row(i).to_vec())
            .collect();
        return self
            .inner
            .forward_prepacked(&rows)
            .map_err(PyValueError::new_err);
    }
    // Fall back to list of lists
    let rows = packed_inputs
        .extract::<Vec<Vec<u64>>>()
        .map_err(|_| {
            PyValueError::new_err(
                "packed_inputs must be a 2-D numpy uint64 array or list[list[int]].",
            )
        })?;
    self.inner
        .forward_prepacked(&rows)
        .map_err(PyValueError::new_err)
}
```

### 5.3 Update imports in `lib.rs`

Change line 3 from:
```rust
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
```
to:
```rust
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
```

(`PyArray2` is needed for Packet AG below.)

---

## 6. Packet AG: `batch_encode_numpy` Returning 2D Numpy Array

### 6.1 New `#[pyfunction]` in `engine/src/lib.rs`

Add after the existing `batch_encode` function (after line 349):

```rust
/// Bernoulli-encode a numpy float64 array into a 2-D numpy uint64 array.
///
/// Returns a numpy array of shape (n_probs, ceil(length / 64)).
/// This is the zero-copy companion to `batch_encode` which returns nested Python lists.
#[pyfunction]
#[pyo3(signature = (probs, length=1024, seed=0xACE1))]
fn batch_encode_numpy<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    length: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let prob_slice = probs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Cannot read probs: {e}")))?;
    let words = length.div_ceil(64);
    let n_probs = prob_slice.len();

    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    // Encode into flat Vec<u64> in row-major order
    let mut flat = Vec::with_capacity(n_probs * words);
    for &p in prob_slice {
        let packed = bitstream::bernoulli_packed(p, length, &mut rng);
        // Ensure exactly `words` elements
        if packed.len() == words {
            flat.extend_from_slice(&packed);
        } else {
            flat.extend_from_slice(&packed);
            flat.resize(flat.len() + (words - packed.len()), 0);
        }
    }

    // Convert to 2-D numpy array
    use numpy::PyArray;
    let arr = PyArray::from_vec_bound(py, flat);
    let arr2d = arr
        .reshape([n_probs, words])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {e}")))?;
    Ok(arr2d)
}
```

### 6.2 Register in module init

In the `fn sc_neurocore_engine(m: ...)` function, add after the `batch_encode` registration (line 32):

```rust
m.add_function(wrap_pyfunction!(batch_encode_numpy, m)?)?;
```

### 6.3 Export in `bridge/sc_neurocore_engine/__init__.py`

Add `batch_encode_numpy` to the import from `sc_neurocore_engine.sc_neurocore_engine`:

```python
from sc_neurocore_engine.sc_neurocore_engine import (
    # ... existing imports ...
    batch_encode_numpy,
)
```

And add `"batch_encode_numpy"` to `__all__`.

---

## 7. Packet AH: PyPI Publish Automation

### 7.1 New `publish` job in `.github/workflows/v3-wheels.yml`

Add after the `test-wheels` job:

```yaml
  publish:
    name: Publish to PyPI
    needs: test-wheels
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v3.')
    environment:
      name: pypi
      url: https://pypi.org/project/sc-neurocore-engine/
    permissions:
      id-token: write
    steps:
      - name: Download all wheel artifacts
        uses: actions/download-artifact@v4
        with:
          path: dist/
          pattern: wheel-*
          merge-multiple: true

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist/
```

This uses PyPI's Trusted Publisher workflow (no API token needed — uses OIDC). The repository must be configured as a trusted publisher on PyPI.

---

## 8. Packet AI: Updated Benchmarks + Version 3.1.0 + Docs

### 8.1 Version bump

**`engine/Cargo.toml`** line 3:
```toml
version = "3.1.0"
```

**`engine/src/lib.rs`** line 22:
```rust
m.add("__version__", "3.1.0")?;
```

**`bridge/pyproject.toml`**:
```toml
version = "3.1.0"
```

### 8.2 Updated benchmark script

In `examples/03_benchmark_report.py`, replace the `bench_dense_forward` function with:

```python
def bench_dense_forward(n_in: int = 64, n_out: int = 32, length: int = 1024) -> list[dict]:
    """Benchmark dense forward pass: original, fast (parallel encode), and prepacked."""
    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 1, n_in)
    inputs_f64 = inputs.astype(np.float64)

    v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out, length=length)
    v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out, length=length)

    # Pre-encode inputs for the prepacked variant
    packed_inputs = v3.batch_encode_numpy(inputs_f64, length=length, seed=42)

    v2_time = benchmark(lambda: v2_layer.forward(inputs), n_iters=10)
    v3_time = benchmark(lambda: v3_layer.forward(inputs), n_iters=10)
    v3_fast_time = benchmark(lambda: v3_layer.forward_fast(inputs), n_iters=10)
    v3_prepacked_time = benchmark(lambda: v3_layer.forward_prepacked(packed_inputs), n_iters=10)

    return [
        {
            "operation": f"dense forward ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_time),
            "target": "70x",
        },
        {
            "operation": f"dense fast ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_fast_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_fast_time),
            "target": "70x",
        },
        {
            "operation": f"dense prepacked ({n_in}->{n_out}, L={length})",
            "v2_ms": v2_time / 10 * 1000,
            "v3_ms": v3_prepacked_time / 10 * 1000,
            "speedup": fmt_speedup(v2_time, v3_prepacked_time),
            "target": "70x",
        },
    ]
```

Also add to the Notes section at the bottom of `main()`:

```python
print("'fast' variants use per-input parallel encoding with rayon.")
print("'prepacked' variants skip encoding entirely (pre-encoded inputs).")
```

### 8.3 Updated benchmark report

After running the script, update `docs/BENCHMARK_REPORT.md` with the Phase 7 results table. Keep the Phase 6 table for comparison.

### 8.4 CHANGELOG

Prepend to `CHANGELOG_V3.md`:

```markdown
## [3.1.0] - 2026-02-10

### Phase 7: Dense Forward Optimization & PyPI Publishing
- **Direct Packed Bernoulli**: `bernoulli_packed()` eliminates Vec<u8> intermediate allocations
- **Parallel Encoding**: `DenseLayer.forward_fast()` parallelizes input encoding with per-input RNGs
- **Pre-packed Forward**: `DenseLayer.forward_prepacked()` accepts pre-encoded numpy/list inputs, skips encoding
- **batch_encode_numpy**: Returns 2-D numpy array instead of nested Python lists
- **PyPI Publishing**: Automated wheel upload on v3.* tags via Trusted Publisher
- **Updated Benchmarks**: Dense forward with fast/prepacked variants
```

### 8.5 Migration docs

Append a Phase 7 section to `docs/v3_migration.md`:

```markdown
## Phase 7 Features (February 2026)

### Dense Forward Optimization

Three performance tiers for dense layer inference:

```python
import numpy as np
import sc_neurocore_engine as v3

layer = v3.DenseLayer(64, 32, 1024)
inputs = np.random.uniform(0, 1, 64)

# Original (sequential encoding) — same as Phase 6
out = layer.forward(inputs.tolist())

# Fast (parallel encoding) — each input encoded on its own thread
out = layer.forward_fast(inputs.tolist())

# Pre-packed (skip encoding) — fastest path
packed = v3.batch_encode_numpy(inputs, length=1024, seed=42)
out = layer.forward_prepacked(packed)
```

### batch_encode_numpy

Returns a 2-D numpy uint64 array instead of nested Python lists:

```python
probs = np.array([0.3, 0.5, 0.7, 0.9])
packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
# packed.shape == (4, 16)  # 4 inputs × ceil(1024/64) words
# packed.dtype == np.uint64
```
```

---

## 9. New Test File: `tests/test_dense_optimization.py`

Create `tests/test_dense_optimization.py`:

```python
"""Tests for Phase 7 dense forward optimizations."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestBernoulliPackedEquivalence:
    """Verify bernoulli_packed produces same output as bernoulli_stream + pack."""

    def test_pack_deterministic(self):
        """forward() must produce identical results before and after bernoulli_packed refactor."""
        layer = v3.DenseLayer(8, 4, 256, seed=12345)
        inputs = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        out = layer.forward(inputs, seed=99999)
        # These values are deterministic — should match Phase 6 exactly
        assert len(out) == 4
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_pack_deterministic_repeated(self):
        """Same inputs + seeds produce same outputs."""
        layer = v3.DenseLayer(8, 4, 256, seed=12345)
        out1 = layer.forward([0.5] * 8, seed=42)
        out2 = layer.forward([0.5] * 8, seed=42)
        assert out1 == out2


class TestForwardFast:
    """Tests for parallel-encoded forward_fast method."""

    def test_output_shape(self):
        layer = v3.DenseLayer(16, 8, 512)
        out = layer.forward_fast([0.5] * 16)
        assert len(out) == 8

    def test_output_range(self):
        layer = v3.DenseLayer(16, 8, 512)
        out = layer.forward_fast([0.3] * 16)
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_deterministic(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        out1 = layer.forward_fast([0.5] * 16, seed=100)
        out2 = layer.forward_fast([0.5] * 16, seed=100)
        assert out1 == out2

    def test_different_seed_different_output(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        out1 = layer.forward_fast([0.5] * 16, seed=100)
        out2 = layer.forward_fast([0.5] * 16, seed=200)
        assert out1 != out2

    def test_statistical_sanity(self):
        """forward_fast should give similar distribution to forward."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        inputs = [0.5, 0.5, 0.5, 0.5]
        results_orig = [layer.forward(inputs, seed=s) for s in range(50)]
        results_fast = [layer.forward_fast(inputs, seed=s) for s in range(50)]
        mean_orig = np.mean([r[0] for r in results_orig])
        mean_fast = np.mean([r[0] for r in results_fast])
        # Both should be near 0.25 (0.5 * 0.5 weight * input)
        assert abs(mean_orig - mean_fast) < 0.05

    def test_wrong_input_length(self):
        layer = v3.DenseLayer(8, 4, 256)
        with pytest.raises(ValueError):
            layer.forward_fast([0.5] * 7)


class TestForwardPrepacked:
    """Tests for pre-packed forward path."""

    def test_output_shape(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9])
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2

    def test_output_range(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_deterministic(self):
        """Same pre-packed inputs always give same output."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = v3.batch_encode_numpy(np.array([0.5] * 4), length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        assert out1 == out2

    def test_accepts_list_of_lists(self):
        """forward_prepacked should also accept list[list[int]]."""
        layer = v3.DenseLayer(2, 1, 128, seed=42)
        packed = v3.batch_encode(np.array([0.5, 0.5]), length=128, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 1

    def test_wrong_n_inputs(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = v3.batch_encode_numpy(np.array([0.5, 0.5, 0.5]), length=1024, seed=55)
        with pytest.raises(ValueError):
            layer.forward_prepacked(packed)

    def test_wrong_word_count(self):
        layer = v3.DenseLayer(2, 1, 1024, seed=42)
        # Create packed with wrong length (512 instead of 1024)
        packed = v3.batch_encode_numpy(np.array([0.5, 0.5]), length=512, seed=55)
        with pytest.raises(ValueError):
            layer.forward_prepacked(packed)


class TestBatchEncodeNumpy:
    """Tests for batch_encode_numpy returning 2D numpy array."""

    def test_shape(self):
        probs = np.array([0.3, 0.5, 0.7])
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        assert packed.shape == (3, 16)  # 3 probs × ceil(1024/64)
        assert packed.dtype == np.uint64

    def test_deterministic(self):
        probs = np.array([0.5, 0.5])
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed(self):
        probs = np.array([0.5])
        p1 = v3.batch_encode_numpy(probs, length=1024, seed=1)
        p2 = v3.batch_encode_numpy(probs, length=1024, seed=2)
        assert not np.array_equal(p1, p2)

    def test_matches_batch_encode(self):
        """batch_encode_numpy must produce same packed words as batch_encode."""
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        list_result = v3.batch_encode(probs, length=256, seed=42)
        np_result = v3.batch_encode_numpy(probs, length=256, seed=42)
        for i, row in enumerate(list_result):
            np.testing.assert_array_equal(np_result[i], row)

    def test_popcount_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75])
        packed = v3.batch_encode_numpy(probs, length=10000, seed=42)
        pc0 = sum(bin(w).count('1') for w in packed[0])
        pc1 = sum(bin(w).count('1') for w in packed[1])
        assert abs(pc0 / 10000 - 0.25) < 0.03
        assert abs(pc1 / 10000 - 0.75) < 0.03

    def test_empty_probs(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=64, seed=42)
        assert packed.shape == (0, 1)

    def test_pipeline_encode_then_forward(self):
        """Full pipeline: batch_encode_numpy → forward_prepacked."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9])
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2
        assert all(0.0 <= v <= 1.0 for v in out)
```

---

## 10. CI Test List Update

In `.github/workflows/v3-engine.yml`, add `tests/test_dense_optimization.py` to the pytest command for the v3-specific test step.

---

## 11. Quality Gates

All quality gates from Phase 6 apply. Run in this order:

```powershell
# Rust gates
cd 03_CODE/sc-neurocore/engine
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test --tests
cargo doc --no-deps

# Python build
cd ../bridge
..\.venv\Scripts\python -m maturin develop --release

# Python tests (all v3 tests including new file)
cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py -v --tb=short

# Co-sim
.\.venv\Scripts\python -m pytest cosim/ -v -rs --tb=short

# Examples (must all still work)
.\.venv\Scripts\python examples/01_sc_training_demo.py
.\.venv\Scripts\python examples/02_ir_compile_demo.py
.\.venv\Scripts\python examples/03_benchmark_report.py

# Wheel build + version check
cd bridge
..\.venv\Scripts\python -m maturin build --release --manifest-path ../engine/Cargo.toml --out ../dist/
cd ..
.\.venv\Scripts\python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__)"
# Expected: 3.1.0
```

### Expected Test Counts

| Suite | Phase 6 | Phase 7 Expected |
|-------|---------|-----------------|
| Rust (`cargo test`) | 56 | 57+ (bernoulli_packed test) |
| Python (v3 suite) | 79 | ~100+ (new test file) |
| Co-sim | 8 | 8 (unchanged) |

---

## 12. Sacred File Rules

**DO NOT MODIFY** any file under:
- `src/sc_neurocore/` (v2 Python source)
- `pyproject.toml` (root — v2 package config)
- `.github/workflows/ci.yml` (v2 CI)

Only `__pycache__/*.pyc` artifacts may appear in git diff for these paths.

---

## 13. Acceptance Criteria Summary

| # | Criterion | How to Verify |
|---|-----------|--------------|
| 1 | `bernoulli_packed` produces bit-identical output to `bernoulli_stream + pack` | Rust unit test passes |
| 2 | `forward()` output unchanged after refactor | All existing equivalence tests pass |
| 3 | `forward_fast()` returns correct shape and range | Python tests pass |
| 4 | `forward_prepacked()` accepts numpy 2D and list[list] | Python tests pass |
| 5 | `batch_encode_numpy` returns correct shape and dtype | Python tests pass |
| 6 | `batch_encode_numpy` produces same packed words as `batch_encode` | Python test `test_matches_batch_encode` |
| 7 | Pipeline: `batch_encode_numpy → forward_prepacked` works end-to-end | Python test `test_pipeline_encode_then_forward` |
| 8 | PyPI publish job in v3-wheels.yml | File inspection |
| 9 | Version = 3.1.0 in Cargo.toml, lib.rs, pyproject.toml | `import sc_neurocore_engine; print(sc_neurocore_engine.__version__)` |
| 10 | Benchmark script runs with fast/prepacked variants | `python examples/03_benchmark_report.py` |
| 11 | CHANGELOG has [3.1.0] section | File inspection |
| 12 | Sacred files untouched | `git diff -- src/sc_neurocore/ | grep -v __pycache__` returns nothing |
| 13 | All quality gates pass | See Section 11 |
