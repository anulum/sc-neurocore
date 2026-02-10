# SC-NeuroCore v3 — Phase 11 Codex Handover

**Author**: Claude Opus 4.6 (Architect)
**Date**: 2026-02-10
**Phase**: 11 (Packets BA–BE)
**Target Version**: 3.5.0
**Predecessor**: Phase 10 (v3.4.0) — ACCEPTED
**Theme**: SIMD Pipeline Acceleration + Zero-Allocation Hot Paths

---

## 0. Context & Motivation

### Blueprint Target Status After Phase 10

| Target | Blueprint | Achieved | Status |
|--------|-----------|----------|--------|
| pack | 6x | **127.0x** | EXCEEDED |
| popcount | 20x | **72.4x** | EXCEEDED |
| dense | 70x | **81.6x** | EXCEEDED |
| LIF | 400x | **170.7x** | **IN PROGRESS — Phase 11 primary target** |

### Analysis of Remaining Gap

The LIF 400x target is the only unmet Blueprint goal. Rust-internal throughput is excellent (~3.5 ns/step via criterion), but Python-level measurements show:

| Path | Time (100K steps) | Overhead Source |
|------|-------------------|-----------------|
| Criterion (pure Rust) | 0.342 ms | — |
| `batch_lif_run` (Python) | 0.992 ms | Vec alloc + `into_pyarray_bound` |
| `batch_lif_run_multi` (100 neurons) | 90.5 ms | Vec alloc per neuron + flatten + `Array2::from_shape_vec` |

The gap from 0.342 ms (Rust) to 0.992 ms (Python) is dominated by:
1. Two `Vec::with_capacity(n_steps)` allocations (spikes + voltages)
2. `Vec<i32>.into_pyarray_bound()` + `Vec<i16>.into_pyarray_bound()` conversions

For `batch_lif_run_multi`, the additional overhead is:
1. Per-neuron `Vec` allocations inside the rayon closure (N × 2 allocations)
2. `extend_from_slice` flatten loop (copies all data a second time)
3. `Array2::from_shape_vec` (validates shape, no copy but checks)

### Phase 11 Strategy

Eliminate allocation overhead in LIF and accelerate the dense forward inner loop with SIMD:

1. **Pre-allocate numpy arrays and write directly** — zero intermediate allocation for LIF batch
2. **SIMD fused AND+popcount** — vectorize the dense forward inner loop
3. **SIMD Bernoulli encode** — vectorize the encoding loop that dominates `forward_fast`
4. **Flat contiguous weight storage** — eliminate 3-level Vec indirection in `DenseLayer`

---

## 1. Sacred Files — DO NOT MODIFY

| Path | Reason |
|------|--------|
| `src/sc_neurocore/` | v2 Python golden reference |
| `pyproject.toml` (repo root) | v2 package config |
| `.github/workflows/ci.yml` | v2 CI pipeline |

---

## 2. Packets

### Packet BA: SIMD Fused AND+Popcount

**Goal**: Replace the scalar `fused_and_popcount` in `engine/src/layer.rs` with a SIMD-dispatched version.

**Current code** (`layer.rs` lines 18-23):
```rust
#[inline]
fn fused_and_popcount(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}
```

This compiles to scalar `popcnt` instructions, processing 1 word at a time. For a 1024-bit stream (16 words), this is 16 iterations.

**Changes required**:

1. **`engine/src/simd/avx512.rs`** — Add:
```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
/// Fused AND+popcount: sum of popcount(a[i] & b[i]) for all i.
///
/// Processes 8 words (512 bits) per iteration using VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure CPU supports avx512f + avx512vpopcntdq.
pub unsafe fn fused_and_popcount_avx512(a: &[u64], b: &[u64]) -> u64 {
    let mut total = _mm512_setzero_si512();
    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = _mm512_loadu_si512(ca.as_ptr() as *const __m512i);
        let vb = _mm512_loadu_si512(cb.as_ptr() as *const __m512i);
        let anded = _mm512_and_epi64(va, vb);
        let counts = _mm512_popcnt_epi64(anded);
        total = _mm512_add_epi64(total, counts);
    }

    // Horizontal reduce total
    let mut lanes = [0_u64; 8];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = lanes.iter().sum();

    // Scalar remainder
    for (&wa, &wb) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        sum += (wa & wb).count_ones() as u64;
    }
    sum
}
```

2. **`engine/src/simd/avx2.rs`** — Add:
```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Fused AND+popcount using AVX2 SWAR popcount.
///
/// Processes 4 words (256 bits) per iteration using manual SWAR.
///
/// # Safety
/// Caller must ensure CPU supports avx2.
pub unsafe fn fused_and_popcount_avx2(a: &[u64], b: &[u64]) -> u64 {
    // Same pattern: load 4 x u64, AND, SWAR popcount on 256-bit register
    // Fallback to scalar for remainder
    // ...
}
```

Note: AVX2 SWAR popcount is complex — an acceptable alternative for the AVX2 tier is to use `_mm256_and_si256` for the AND, store back, and use scalar `count_ones()` on the result. This still benefits from wider AND throughput. The performance-critical path is AVX-512 VPOPCNTDQ.

3. **`engine/src/simd/mod.rs`** — Add dispatch:
```rust
/// Fused AND+popcount dispatch: best available SIMD path.
pub fn fused_and_popcount_dispatch(a: &[u64], b: &[u64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            return unsafe { avx512::fused_and_popcount_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::fused_and_popcount_avx2(a, b) };
        }
    }
    // Portable fallback
    a.iter().zip(b.iter()).map(|(&wa, &wb)| (wa & wb).count_ones() as u64).sum()
}
```

4. **`engine/src/layer.rs`** — Replace the current `fused_and_popcount` function with:
```rust
use crate::simd;
```
And change all call sites from `fused_and_popcount(w, i)` to `simd::fused_and_popcount_dispatch(w, i)`.

Remove the local `fn fused_and_popcount` from `layer.rs`.

5. **Tests** — Add in `engine/src/simd/avx512.rs`:
```rust
#[test]
fn fused_and_popcount_avx512_matches_scalar() {
    if !is_x86_feature_detected!("avx512vpopcntdq") { return; }
    // Test with various lengths: 1, 7, 8, 15, 16, 17, 31, 32, 64, 128
    // Compare against scalar: a.iter().zip(b).map(|(a,b)| (a & b).count_ones()).sum()
}
```
And equivalent in `avx2.rs`.

6. **Non-`x86_64` fallbacks** — Add `#[cfg(not(target_arch = "x86_64"))]` stub functions in `avx512.rs` and `avx2.rs` that delegate to the portable scalar path (same pattern as existing `popcount_avx512` fallback).

**Verification**:
- `cargo test --tests` — SWAR equivalence tests pass
- `cargo clippy --all-targets -- -D warnings`
- Criterion: add `fused_and_popcount_16w` benchmark (16 words = 1024 bits, matching dense layer config)

---

### Packet BB: SIMD Bernoulli Encode

**Goal**: Accelerate `bernoulli_packed_fast` by using SIMD comparison on random byte buffers.

**Current code** (`bitstream.rs` lines 160-176):
```rust
pub fn bernoulli_packed_fast<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    // ...
    for (word_idx, word) in data.iter_mut().enumerate() {
        rng.fill(&mut buf[..bits_in_word]);
        for (bit, &rb) in buf[..bits_in_word].iter().enumerate() {
            if rb < threshold { *word |= 1_u64 << bit; }
        }
    }
}
```

The inner comparison loop processes 64 bytes sequentially. With AVX-512BW, `_mm512_cmplt_epu8_mask` can produce a 64-bit k-mask in a single instruction — directly producing the output u64 word.

**Changes required**:

1. **`engine/src/simd/avx512.rs`** — Add:
```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
/// Compare 64 random bytes against a threshold, producing a packed u64.
///
/// Each bit in the output is 1 if `buf[i] < threshold`, else 0.
///
/// # Safety
/// Caller must ensure CPU supports avx512f + avx512bw.
/// `buf` must have at least 64 elements.
pub unsafe fn bernoulli_compare_avx512(buf: &[u8], threshold: u8) -> u64 {
    let data = _mm512_loadu_si512(buf.as_ptr() as *const __m512i);
    let thresh = _mm512_set1_epi8(threshold as i8);
    // _mm512_cmplt_epu8_mask: bit i = 1 if buf[i] < threshold (unsigned compare)
    _mm512_cmplt_epu8_mask(data, thresh)
}
```

2. **`engine/src/simd/avx2.rs`** — Add:
```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Compare 32 random bytes against a threshold, producing a u32 mask.
///
/// # Safety
/// Caller must ensure CPU supports avx2.
/// `buf` must have at least 32 elements.
pub unsafe fn bernoulli_compare_avx2(buf: &[u8], threshold: u8) -> u32 {
    // Load 32 bytes
    let data = _mm256_loadu_si256(buf.as_ptr() as *const __m256i);
    // Subtract 128 to convert unsigned comparison to signed
    // (because _mm256_cmpgt_epi8 is signed)
    let bias = _mm256_set1_epi8(-128_i8);
    let data_biased = _mm256_add_epi8(data, bias);
    let thresh_biased = _mm256_set1_epi8((threshold as i8).wrapping_sub(-128_i8));
    // cmpgt gives 0xFF where data > thresh, we want data < thresh so swap operands
    let gt = _mm256_cmpgt_epi8(thresh_biased, data_biased);
    _mm256_movemask_epi8(gt) as u32
}
```

3. **`engine/src/bitstream.rs`** — Add `bernoulli_packed_simd`:
```rust
/// SIMD-accelerated packed Bernoulli generation.
///
/// Same semantics as `bernoulli_packed_fast` but uses SIMD comparison
/// on full 64-byte random buffers for ~20x faster comparison.
///
/// Output is NOT bit-identical to `bernoulli_packed_fast` or `bernoulli_packed`
/// (different comparison granularity), but is statistically equivalent.
pub fn bernoulli_packed_simd<R: Rng + ?Sized>(prob: f64, length: usize, rng: &mut R) -> Vec<u64> {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    let full_words = length / 64;
    let mut buf = [0_u8; 64];

    for word in data.iter_mut().take(full_words) {
        rng.fill(&mut buf);
        *word = simd_bernoulli_compare(&buf, threshold);
    }

    // Handle partial last word (< 64 bits)
    if full_words < words {
        let remaining = length - full_words * 64;
        rng.fill(&mut buf[..remaining]);
        // Use scalar path for the partial word
        let mut w = 0_u64;
        for (bit, &rb) in buf[..remaining].iter().enumerate() {
            if rb < threshold { w |= 1_u64 << bit; }
        }
        data[full_words] = w;
    }

    data
}

/// Dispatch to best SIMD compare for 64 bytes → u64 mask.
#[inline]
fn simd_bernoulli_compare(buf: &[u8], threshold: u8) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { crate::simd::avx512::bernoulli_compare_avx512(buf, threshold) };
        }
        if is_x86_feature_detected!("avx2") {
            let lo = unsafe { crate::simd::avx2::bernoulli_compare_avx2(&buf[0..32], threshold) };
            let hi = unsafe { crate::simd::avx2::bernoulli_compare_avx2(&buf[32..64], threshold) };
            return (lo as u64) | ((hi as u64) << 32);
        }
    }
    // Portable scalar fallback
    let mut w = 0_u64;
    for (bit, &rb) in buf.iter().enumerate().take(64) {
        if rb < threshold { w |= 1_u64 << bit; }
    }
    w
}
```

4. **`engine/src/layer.rs`** — In `forward_fast()`, replace `bernoulli_packed_fast` with `bernoulli_packed_simd` for the encoding step. Keep `forward()` using `bernoulli_packed` (f64-exact reference path).

5. **`engine/src/lib.rs`** — In `batch_encode_numpy`, replace `bernoulli_packed_fast` with `bernoulli_packed_simd`.

6. **Tests** — Add:
- `bernoulli_packed_simd_statistics` — same pattern as `bernoulli_packed_fast_statistics`: 10K bits at p=0.35, verify measured rate within ±0.03
- `bernoulli_packed_simd_deterministic` — same seed → identical output
- `bernoulli_compare_avx512_matches_scalar` — runtime-gated, 64-byte buffer, compare against scalar loop
- `bernoulli_compare_avx2_matches_scalar` — runtime-gated, 32-byte buffer

**Verification**:
- `cargo test --tests`
- Criterion: add `bernoulli_packed_simd_1024` benchmark alongside existing `bernoulli_packed_fast_1024`

---

### Packet BC: Flat Contiguous Weight Storage

**Goal**: Replace the `Vec<Vec<Vec<u64>>>` weight storage with a single flat `Vec<u64>` for cache-friendly sequential access.

**Current storage** (`layer.rs`):
```rust
pub packed_weights: Vec<Vec<Vec<u64>>>,   // [n_neurons][n_inputs][n_words]
```

Three levels of heap indirection. When computing `fused_and_popcount` for one neuron, the CPU must chase pointers: `packed_weights[neuron]` → `Vec<Vec<u64>>` → `[input]` → `Vec<u64>` → `[word]`. Each level can cause a cache miss.

**New storage**:
```rust
pub packed_weights_flat: Vec<u64>,   // contiguous [n_neurons * n_inputs * words]
pub words_per_input: usize,          // = ceil(length / 64)
```

Layout: `packed_weights_flat[neuron * n_inputs * words + input * words .. + words]`

**Changes required**:

1. **`engine/src/layer.rs`** — Modify `DenseLayer`:
```rust
pub struct DenseLayer {
    pub n_inputs: usize,
    pub n_neurons: usize,
    pub length: usize,
    pub weights: Vec<Vec<f64>>,
    packed_weights_flat: Vec<u64>,   // NEW: flat contiguous storage
    words_per_input: usize,           // NEW: = length.div_ceil(64)
    weight_seed: u64,
}
```

Add helper:
```rust
impl DenseLayer {
    /// Byte offset into flat weight storage.
    #[inline]
    fn weight_slice(&self, neuron: usize, input: usize) -> &[u64] {
        let start = (neuron * self.n_inputs + input) * self.words_per_input;
        &self.packed_weights_flat[start..start + self.words_per_input]
    }
}
```

2. Update `refresh_packed_weights()` to write into the flat buffer.

3. Update ALL forward methods to use `self.weight_slice(neuron_idx, input_idx)` instead of `self.packed_weights[neuron_idx][input_idx]`.

4. Remove the old `packed_weights: Vec<Vec<Vec<u64>>>` field entirely.

5. **Keep `get_weights()` and `set_weights()` signatures unchanged** — these operate on probability-domain `Vec<Vec<f64>>` and are unaffected.

**Tests** — The existing equivalence tests in `tests/equivalence/` will verify correctness. No new test files needed for this change, but add a unit test:
- `flat_weight_roundtrip` — create layer, verify `weight_slice(n, i)` returns same data as would be produced by the old `packed_weights[n][i]`

**Verification**:
- All existing Python tests must pass unchanged (the change is internal, API is identical)
- `cargo test --tests`

---

### Packet BD: Pre-Allocated LIF Output Buffers

**Goal**: Eliminate per-call Vec allocations in `batch_lif_run` and `batch_lif_run_multi` by writing directly into pre-allocated numpy arrays.

**Current pattern** (`lib.rs` lines 246-258):
```rust
fn batch_lif_run(...) -> (PyArray1<i32>, PyArray1<i16>) {
    let mut spikes = Vec::with_capacity(n_steps);     // ALLOC 1
    let mut voltages = Vec::with_capacity(n_steps);   // ALLOC 2
    for _ in 0..n_steps {
        let (s, v) = lif.step(leak_k, gain_k, i_t, noise_in);
        spikes.push(s);     // writes to Vec
        voltages.push(v);   // writes to Vec
    }
    (spikes.into_pyarray_bound(py), voltages.into_pyarray_bound(py))
    // ^^^ copies entire Vec into numpy buffer
}
```

**New pattern**:
```rust
fn batch_lif_run(...) -> (PyArray1<i32>, PyArray1<i16>) {
    // Pre-allocate numpy arrays directly
    let spikes_arr = PyArray1::<i32>::zeros_bound(py, n_steps, false);
    let voltages_arr = PyArray1::<i16>::zeros_bound(py, n_steps, false);

    // Get raw mutable slices — ZERO intermediate allocation
    {
        let spikes_slice = unsafe { spikes_arr.as_slice_mut().unwrap() };
        let voltages_slice = unsafe { voltages_arr.as_slice_mut().unwrap() };

        for i in 0..n_steps {
            let (s, v) = lif.step(leak_k, gain_k, i_t, noise_in);
            spikes_slice[i] = s;
            voltages_slice[i] = v;
        }
    }

    (spikes_arr, voltages_arr)
}
```

**Changes required**:

1. **`engine/src/lib.rs`** — Rewrite `batch_lif_run`:
   - Use `PyArray1::zeros_bound(py, n_steps, false)` to pre-allocate
   - Use `unsafe { arr.as_slice_mut() }` to write directly
   - Eliminate `Vec::with_capacity` + `push` + `into_pyarray_bound`

2. **`engine/src/lib.rs`** — Rewrite `batch_lif_run_multi`:
   - Use `PyArray2::zeros_bound(py, [n_neurons, n_steps], false)` to pre-allocate
   - Use `unsafe { arr.as_array_mut() }` to get ndarray::ArrayViewMut2
   - Each rayon task writes to its row slice via `arr.row_mut(ni)`
   - Eliminate: per-neuron Vec alloc, `extend_from_slice` flatten loop, `Array2::from_shape_vec`

   **IMPORTANT**: The numpy array must be allocated BEFORE the rayon parallel section. The rayon closures receive `&mut` row slices, not owned Vecs. This requires careful unsafe code:
   ```rust
   let spikes_arr = PyArray2::<i32>::zeros_bound(py, [n_neurons, n_steps], false);
   let voltages_arr = PyArray2::<i16>::zeros_bound(py, [n_neurons, n_steps], false);

   // Get raw pointers for rayon-safe parallel writes
   let spikes_ptr = spikes_arr.as_raw_array_mut().as_mut_ptr();
   let voltages_ptr = voltages_arr.as_raw_array_mut().as_mut_ptr();

   // Each neuron writes to its own non-overlapping row — no data races
   (0..n_neurons).into_par_iter().for_each(|ni| {
       let mut lif = FixedPointLif::new(...);
       let spike_row = unsafe {
           std::slice::from_raw_parts_mut(spikes_ptr.add(ni * n_steps), n_steps)
       };
       let voltage_row = unsafe {
           std::slice::from_raw_parts_mut(voltages_ptr.add(ni * n_steps), n_steps)
       };
       for i in 0..n_steps {
           let (s, v) = lif.step(leak_k, gain_k, curr_slice[ni], 0);
           spike_row[i] = s;
           voltage_row[i] = v;
       }
   });
   ```

   The `unsafe` is sound because:
   - Each neuron writes to a disjoint row (no overlap)
   - The numpy arrays are not accessed from Python until after `par_iter` completes
   - C-contiguous layout guarantees row `ni` starts at offset `ni * n_steps`

3. **`engine/src/lib.rs`** — Apply same pattern to `batch_lif_run_varying`.

4. **Tests** — Existing Python tests verify output correctness. Add Rust-side unit test:
   - `batch_lif_multi_output_equivalence` — verify new zero-alloc path produces identical output to a reference Vec-based computation for 10 neurons × 1000 steps

**Verification**:
- All existing Python tests must pass
- `cargo clippy` — verify unsafe blocks don't trigger warnings
- Python benchmark: `batch_lif_run(100_000, ...)` should show measurably lower time (target: < 0.5 ms, was 0.992 ms)

---

### Packet BE: Version 3.5.0 + Benchmarks + Docs + Tests

**Goal**: Version bump, targeted benchmarks, documentation, and Phase 11 test file.

**Changes required**:

1. **Version bump to `3.5.0`** in:
   - `engine/Cargo.toml` — `version = "3.5.0"`
   - `engine/src/lib.rs` — `m.add("__version__", "3.5.0")?;`
   - `bridge/pyproject.toml` — `version = "3.5.0"`
   - `bridge/sc_neurocore_engine/__init__.py` — docstring "SC-NeuroCore v3.5"

2. **Criterion benchmarks** — Add to `engine/benches/full_bench.rs`:
   - `fused_and_popcount_scalar_16w` — 16 u64 words, scalar `(a & b).count_ones()` sum
   - `fused_and_popcount_dispatch_16w` — 16 u64 words via `simd::fused_and_popcount_dispatch`
   - `bernoulli_packed_simd_1024` — SIMD Bernoulli encode 1024 bits
   - `dense_forward_fast_flat_64x32` — dense forward with flat weights + SIMD encode + SIMD fused popcount (same params as existing `dense_forward_fast_64x32`)

   Note: The existing `dense_forward_fast_64x32` benchmark will automatically use the new SIMD paths since it calls `layer.forward_fast()`. The new benchmark is a renamed copy for explicit Phase 11 tracking.

3. **Python benchmark** — Add to `examples/03_benchmark_report.py`:
   - In `bench_dense_forward()`: no changes needed (existing variants already measure all paths)
   - Add optional warm-up: call each function once before the timed loop to stabilize rayon thread pool

4. **`CHANGELOG_V3.md`** — Add `[3.5.0]` entry:
   ```markdown
   ## [3.5.0] - 2026-02-10

   ### Phase 11: SIMD Pipeline Acceleration
   - **SIMD fused AND+popcount**: AVX-512 VPOPCNTDQ accelerated inner loop for dense forward
   - **SIMD Bernoulli encode**: AVX-512BW/AVX2 comparison for ~20x faster encoding
   - **Flat weight storage**: Contiguous memory layout for cache-friendly access
   - **Zero-allocation LIF batch**: Pre-allocated numpy output buffers eliminate copy overhead
   - **Criterion benchmarks**: Added fused_and_popcount, bernoulli_simd diagnostics
   ```

5. **`docs/v3_migration.md`** — Add Phase 11 section (same pattern as Phase 10).

6. **`docs/BENCHMARK_REPORT.md`** — Update header to `Version: 3.5.0`, add Phase 11 results section with measured numbers. Retain Phase 10 and Phase 9 tables as reference.

7. **`tests/test_phase11.py`** — New file with 5 test classes:

   ```python
   class TestSIMDFusedAndPopcount:
       """Verify SIMD fused AND+popcount produces identical results."""
       def test_dense_forward_unchanged(self):
           """Dense forward output matches Phase 10 reference."""
       def test_dense_prepacked_unchanged(self):
           """Prepacked forward identical to non-prepacked."""
       def test_determinism(self):
           """Same seed → identical output."""

   class TestSIMDBernoulliEncode:
       """Verify SIMD Bernoulli encoding is statistically correct."""
       def test_batch_encode_statistics(self):
           """Encoded rate within ±3% of target probability."""
       def test_batch_encode_determinism(self):
           """Same seed → identical packed output."""
       def test_dense_fast_correctness(self):
           """forward_fast() still produces reasonable rates."""

   class TestFlatWeightStorage:
       """Verify flat weight storage preserves behavior."""
       def test_weight_roundtrip(self):
           """set_weights → get_weights → same values."""
       def test_forward_equivalence_vs_prepacked(self):
           """forward() output matches forward_prepacked() for same data."""

   class TestZeroAllocLIF:
       """Verify pre-allocated LIF output is correct."""
       def test_batch_lif_unchanged(self):
           """batch_lif_run output matches Phase 10."""
       def test_batch_lif_multi_unchanged(self):
           """batch_lif_run_multi output matches Phase 10."""
       def test_batch_lif_multi_shape(self):
           """Output arrays have correct shape and dtype."""
       def test_batch_lif_varying_unchanged(self):
           """batch_lif_run_varying output matches Phase 10."""

   class TestPhase11Version:
       def test_version(self):
           assert v3.__version__ == "3.5.0"
   ```

8. **`.github/workflows/v3-engine.yml`** — Add `tests/test_phase11.py` to pytest command.

9. **`tests/test_phase8.py`** and **`tests/test_phase9.py`** and **`tests/test_phase10.py`** — Update version assertions to `"3.5.0"`.

---

## 3. Execution Order

```
BA (SIMD fused AND+popcount)  ─┐
BB (SIMD Bernoulli encode)     ├── Independent, can be parallelized
BC (Flat weight storage)       ─┘
         │
         ▼
BD (Pre-allocated LIF) ─── Independent of BA/BB/BC
         │
         ▼
BE (Version + docs + tests) ─── Depends on all above
```

BA, BB, and BC can be implemented in any order. BD is fully independent of them. BE must be last.

---

## 4. Verification Gates

Run ALL of the following after implementation:

### Rust gates
```powershell
cd 03_CODE/sc-neurocore/engine
C:\Users\forti\.cargo\bin\cargo.exe fmt
C:\Users\forti\.cargo\bin\cargo.exe clippy --all-targets -- -D warnings
C:\Users\forti\.cargo\bin\cargo.exe test --tests
C:\Users\forti\.cargo\bin\cargo.exe doc --no-deps
```

### Python build
```powershell
cd 03_CODE/sc-neurocore/engine
$env:PATH="$env:USERPROFILE\.cargo\bin;$env:PATH"
..\.venv\Scripts\python.exe -m maturin develop --release
```

### Python tests (full v3 suite)
```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py tests/test_phase8.py tests/test_phase9.py tests/test_phase10.py tests/test_phase11.py -v --tb=short
```

### Co-simulation
```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest cosim/ -v -rs --tb=short
```

### Examples + version
```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe examples/01_sc_training_demo.py
.\.venv\Scripts\python.exe examples/02_ir_compile_demo.py
.\.venv\Scripts\python.exe examples/03_benchmark_report.py
.\.venv\Scripts\python.exe -c "import sc_neurocore_engine as v3; print(v3.__version__); print(v3.simd_tier())"
```

### Targeted criterion benchmarks
```powershell
cd 03_CODE/sc-neurocore/engine
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench fused_and_popcount
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench bernoulli_packed_simd
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench dense_forward_fast
```

**Expected version output**: `3.5.0`

**Expected test count**: ~165 Python tests (149 + ~16 new in test_phase11.py)

---

## 5. Performance Targets

| Metric | Phase 10 | Phase 11 Target | Rationale |
|--------|----------|-----------------|-----------|
| `fused_and_popcount` (16 words) | ~50 ns (scalar popcnt) | ~10 ns (AVX-512 VPOPCNTDQ) | 8 words/instruction |
| `bernoulli_packed_simd` (1024 bits) | 2.1 µs (fast) | ~0.5 µs (SIMD compare) | 1 instruction per 64 bytes |
| `dense_forward_fast_64x32` | 5.5-8.0 ms (criterion) | ~1-2 ms | SIMD encode + SIMD popcount + flat weights |
| `batch_lif_run` (100K steps) | 0.992 ms | ~0.4 ms | Zero-alloc direct write |
| `batch_lif_run_multi` (100×100K) | 90.5 ms | ~50 ms | Zero-alloc + no flatten |
| LIF Python speedup | 170.7x | **~300x** (multi-neuron) | Closer to 400x target |

---

## 6. Risk Notes

1. **Unsafe code in Packet BD**: The pre-allocated numpy write pattern requires `unsafe` for raw pointer access across rayon threads. The soundness argument (disjoint non-overlapping rows) is well-established in parallel programming but must be carefully implemented. Consider using `ndarray::Zip` or `ndarray::parallel` as a safer alternative if available.

2. **AVX-512BW `_mm512_cmplt_epu8_mask`**: This instruction requires AVX-512BW (not just AVX-512F). Verify the runtime dispatch checks for `avx512bw` specifically (already done for `pack_dispatch`, same pattern).

3. **Bernoulli statistical equivalence**: The SIMD encode is NOT bit-identical to the scalar path (same as `bernoulli_packed_fast` vs `bernoulli_packed`). The test verifies statistical equivalence (rate within ±3% of target). Determinism is tested separately (same seed → same output).

4. **Flat weight storage migration**: Changing the internal storage layout must not change ANY public API behavior. The `get_weights()`, `set_weights()`, and all `forward*()` return values must remain identical.
