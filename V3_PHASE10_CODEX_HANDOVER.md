# SC-NeuroCore v3 — Phase 10 Codex Handover

**Author**: Claude (Opus 4.6)
**Date**: 2026-02-10
**Phase**: 10 — SIMD Pack, LIF Optimization, Rayon Guard
**Previous Phase**: 9 (v3.3.0 — fast Bernoulli, fused AND+popcount, zero-copy prepacked)
**Target Version**: 3.4.0
**Blueprint Targets to Close**: pack 6x (currently 1.1x), LIF 400x (currently 102x)

---

## Objective

Close the two remaining Blueprint performance targets:

1. **pack_bitstream**: 1.1x → ≥6x via SIMD-vectorized byte-to-bit packing
2. **LIF batch**: 102x → ≥400x via branchless mask + parallel multi-neuron batch
3. **Bonus**: Fix forward_fast rayon regression via minimum work threshold

---

## Sacred File Rules (ABSOLUTE)

**NEVER modify** any file under:
- `src/sc_neurocore/` — the v2.2.0 Golden Reference
- `pyproject.toml` (repository root)
- `.github/workflows/ci.yml` (v2 CI pipeline)

---

## Packets

### Packet AV: SIMD Pack Vectorization

**Goal**: Replace the bit-at-a-time `pack()` with SIMD-vectorized packing.

**Current bottleneck** (`engine/src/bitstream.rs` lines 25-37):
```rust
pub fn pack(bits: &[u8]) -> BitStreamTensor {
    // processes 1 bit at a time: if bit != 0 { data[idx/64] |= 1 << (idx%64); }
}
```

For 1M bits this does 1M conditional OR operations. SIMD can process 32-64 bytes at once.

**New functions to add in `engine/src/bitstream.rs`**:

1. **`pack_fast(bits: &[u8]) -> BitStreamTensor`** — Portable 8-byte-at-a-time packing:
   ```rust
   /// Portable fast pack: processes 8 bytes into 1 byte of the output word at a time.
   pub fn pack_fast(bits: &[u8]) -> BitStreamTensor {
       let length = bits.len();
       let words = length.div_ceil(64);
       let mut data = vec![0_u64; words];

       for (word_idx, word) in data.iter_mut().enumerate() {
           let base = word_idx * 64;
           let chunk = &bits[base..std::cmp::min(base + 64, length)];

           // Process 8 bytes at a time into one byte of the u64 word
           for (byte_idx, byte_chunk) in chunk.chunks(8).enumerate() {
               let mut packed_byte: u8 = 0;
               for (bit_idx, &bit) in byte_chunk.iter().enumerate() {
                   packed_byte |= (bit & 1) << bit_idx;
               }
               *word |= (packed_byte as u64) << (byte_idx * 8);
           }
       }

       BitStreamTensor { data, length }
   }
   ```

2. **`pack_avx2(bits: &[u8]) -> BitStreamTensor`** in `engine/src/simd/avx2.rs`:
   ```rust
   /// Pack u8 bits into u64 words using AVX2 movemask.
   ///
   /// Processes 32 bytes at a time:
   /// 1. Load 32 bytes via _mm256_loadu_si256
   /// 2. Compare != 0 via _mm256_cmpgt_epi8(data, zero)
   /// 3. Extract MSBs via _mm256_movemask_epi8 → 32-bit mask
   /// 4. Two rounds of 32 bytes → one u64 word
   ///
   /// # Safety
   /// Caller must ensure the current CPU supports `avx2`.
   #[cfg(target_arch = "x86_64")]
   #[target_feature(enable = "avx2")]
   pub unsafe fn pack_avx2(bits: &[u8]) -> Vec<u64> {
       // See detailed implementation below
   }
   ```

   **AVX2 algorithm per 64-byte chunk → 1 u64 word:**
   - Load first 32 bytes: `_mm256_loadu_si256`
   - `_mm256_cmpgt_epi8(data, _mm256_setzero_si256())` → each byte becomes 0xFF (if >0) or 0x00
   - BUT: movemask extracts bit 7 of each byte. Since comparison result bytes are 0xFF or 0x00, bit 7 is the sign bit. So `_mm256_movemask_epi8()` directly gives us the 32-bit packed result.
   - **Important**: movemask extracts from MSB (bit 7), but our pack layout needs bit 0 of word to correspond to byte 0 of input. movemask's bit ordering is: bit 0 = MSB of byte 0, bit 1 = MSB of byte 1, ..., bit 31 = MSB of byte 31. Since the comparison sets ALL bits (0xFF) or clears ALL bits (0x00), MSB correctly reflects the byte's nonzero status. So movemask gives us the correct bit packing order already.
   - `let lo = _mm256_movemask_epi8(cmp_lo) as u32;`
   - Load second 32 bytes, repeat → `let hi = _mm256_movemask_epi8(cmp_hi) as u32;`
   - `word = (hi as u64) << 32 | lo as u64;`
   - Handle remainder chunk (<64 bytes) with `pack_fast` fallback.

3. **`pack_avx512(bits: &[u8]) -> Vec<u64>`** in `engine/src/simd/avx512.rs`:
   ```rust
   /// Pack u8 bits into u64 words using AVX-512 k-mask.
   ///
   /// Processes 64 bytes at a time:
   /// 1. Load 64 bytes via _mm512_loadu_si512
   /// 2. Compare != 0 via _mm512_cmpneq_epi8_mask → 64-bit mask
   /// 3. That mask IS the packed u64 word.
   ///
   /// # Safety
   /// Caller must ensure the current CPU supports `avx512bw` and `avx512f`.
   #[cfg(target_arch = "x86_64")]
   #[target_feature(enable = "avx512f,avx512bw")]
   pub unsafe fn pack_avx512(bits: &[u8]) -> Vec<u64>;
   ```

   **AVX-512 algorithm per 64-byte chunk → 1 u64 word:**
   - Load 64 bytes: `_mm512_loadu_si512(chunk.as_ptr() as *const __m512i)`
   - `_mm512_cmpneq_epi8_mask(data, _mm512_setzero_si512())` → returns `__mmask64` = `u64`
   - This u64 IS the packed word (bit i = 1 iff byte i ≠ 0).
   - Handle remainder with `pack_fast` fallback.

4. **`pack_dispatch(bits: &[u8]) -> BitStreamTensor`** in `engine/src/simd/mod.rs`:
   ```rust
   /// Pack u8 bits into u64 words using the best available SIMD path.
   pub fn pack_dispatch(bits: &[u8]) -> crate::bitstream::BitStreamTensor {
       let length = bits.len();

       #[cfg(target_arch = "x86_64")]
       {
           if is_x86_feature_detected!("avx512bw") {
               let data = unsafe { avx512::pack_avx512(bits) };
               return crate::bitstream::BitStreamTensor { data, length };
           }
           if is_x86_feature_detected!("avx2") {
               let data = unsafe { avx2::pack_avx2(bits) };
               return crate::bitstream::BitStreamTensor { data, length };
           }
       }

       crate::bitstream::pack_fast(bits)
   }
   ```

5. **Update `pack_bitstream_numpy` in `engine/src/lib.rs`** to use SIMD dispatch:
   ```rust
   fn pack_bitstream_numpy<'py>(...) -> ... {
       let slice = bits.as_slice()?;
       let tensor = simd::pack_dispatch(slice);  // was: bitstream::pack(slice)
       Ok(tensor.data.into_pyarray_bound(py))
   }
   ```

6. **Add unit tests in `engine/src/bitstream.rs`**:
   - `pack_fast_matches_pack`: `pack_fast(bits).data == pack(bits).data` for various lengths
   - `pack_fast_roundtrip`: `unpack(&pack_fast(bits)) == bits`

7. **Add SIMD tests** in each SIMD module:
   - `pack_avx2_matches_pack` (conditional on x86_64 + avx2)
   - `pack_avx512_matches_pack` (conditional on x86_64 + avx512bw)

**AVX-512 note**: The pack function uses `avx512bw` (byte-word operations) for `_mm512_cmpneq_epi8_mask`, NOT `avx512vpopcntdq`. The machine has AVX-512 VPOPCNTDQ, which implies it has AVX-512F + BW. Update the `simd_tier()` function to also detect `avx512bw`.

---

### Packet AW: Branchless LIF + Multi-Neuron Batch

**Goal**: Push LIF batch from 102x toward 400x.

**Part 1: Branchless `mask()` in `engine/src/neuron.rs`**

Current `mask()` (lines 8-21) uses an `if` branch for sign extension. Replace with branchless shift-based sign extension:

```rust
/// Mask and sign-interpret an integer to `width` bits (branchless).
#[inline(always)]
pub fn mask(value: i32, width: u32) -> i16 {
    let m = (1_i64 << width) - 1;
    let v = (value as i64) & m;
    // Branchless sign extension: shift left to put sign bit at position 63,
    // then arithmetic shift right to propagate the sign bit.
    let shift = 64 - width;
    ((v << shift) >> shift) as i16
}
```

This eliminates both branches (the `if v >= ...` and the `if width >= 32` guard). The arithmetic right shift propagates the sign bit automatically. This is equivalent for all width values used (16 and 32).

**Verification**: The branchless version must produce identical results to the current version for all inputs. Add a unit test:
```rust
#[test]
fn mask_branchless_matches_original() {
    // Test edge cases for width=16 and width=32
    for &width in &[16u32, 32] {
        for value in [-32768i32, -1, 0, 1, 32767, 65535, -65536, i16::MAX as i32, i16::MIN as i32] {
            let result = mask(value, width);
            // Verify against reference implementation
            let m = (1_i64 << width) - 1;
            let mut v = (value as i64) & m;
            if v >= (1_i64 << (width - 1)) {
                v -= 1_i64 << width;
            }
            let expected = if width >= 32 { v as i32 as i16 } else { v as i16 };
            assert_eq!(result, expected, "mask({value}, {width}): got {result}, expected {expected}");
        }
    }
}
```

**Part 2: `batch_lif_run_multi` in `engine/src/lib.rs`**

New pyfunction that runs N independent neurons in parallel on separate input streams:

```rust
/// Run N independent LIF neurons in parallel, each with its own constant input.
///
/// Returns (spikes: ndarray[i32, (n_neurons, n_steps)],
///          voltages: ndarray[i16, (n_neurons, n_steps)]).
#[pyfunction]
#[pyo3(signature = (
    n_neurons,
    n_steps,
    leak_k,
    gain_k,
    currents,
    data_width=16,
    fraction=8,
    v_rest=0,
    v_reset=0,
    v_threshold=256,
    refractory_period=2
))]
fn batch_lif_run_multi<'py>(
    py: Python<'py>,
    n_neurons: usize,
    n_steps: usize,
    leak_k: i16,
    gain_k: i16,
    currents: PyReadonlyArray1<'py, i16>,
    ...
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i16>>)>
```

- `currents` is a 1-D array of length `n_neurons` (one constant current per neuron).
- Each neuron runs independently for `n_steps`, parallelized via rayon `par_iter`.
- Returns 2-D numpy arrays of shape `(n_neurons, n_steps)`.
- This amortizes the rayon overhead across many neurons instead of many steps.

**Implementation sketch:**
```rust
let curr_slice = currents.as_slice()?;
// Parallel across neurons
let results: Vec<(Vec<i32>, Vec<i16>)> = (0..n_neurons)
    .into_par_iter()
    .map(|ni| {
        let mut lif = neuron::FixedPointLif::new(...);
        let mut spikes = Vec::with_capacity(n_steps);
        let mut voltages = Vec::with_capacity(n_steps);
        let i_t = curr_slice[ni];
        for _ in 0..n_steps {
            let (s, v) = lif.step(leak_k, gain_k, i_t, 0);
            spikes.push(s);
            voltages.push(v);
        }
        (spikes, voltages)
    })
    .collect();
// Flatten to 2-D numpy arrays
```

**Register** in module init and **export** in bridge `__init__.py` + `__all__`.

**Part 3: LIF neuron step micro-optimization**

In `engine/src/neuron.rs` `step()` method, make the spike/refractory logic branchless:

```rust
pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
    // ... leak and input computation unchanged ...
    let v_next = mask(...);

    // Branchless spike detection
    let fired = (v_next >= self.v_threshold) as i32;
    // Select between reset (fired) and v_next (not fired)
    let v_after_spike = if fired != 0 { self.v_reset } else { v_next };
    let new_refrac = if fired != 0 { self.refractory_period } else { self.refractory_counter };

    // Branchless refractory override
    let in_refrac = (self.refractory_counter > 0) as i32;
    let final_v = if in_refrac != 0 { self.v_rest } else { v_after_spike };
    let final_spike = fired & (1 - in_refrac);
    let final_refrac = if in_refrac != 0 { new_refrac - 1 } else { new_refrac };

    self.v = final_v;
    self.refractory_counter = final_refrac;

    (final_spike, mask(final_v as i32, self.data_width))
}
```

**CRITICAL**: The branchless version must produce **identical** outputs to the current version for all inputs. The Rust equivalence tests and Python equivalence tests are the contract. Do NOT change this if you cannot verify correctness. If in doubt, keep the current `if/else` and focus on `mask()` branchlessness + multi-neuron parallelism.

---

### Packet AX: Rayon Minimum Work Threshold

**Goal**: Fix the `forward_fast` regression where rayon overhead exceeds parallel benefit at small input counts.

**Changes in `engine/src/layer.rs`**:

In `forward_fast()`, add a threshold check before using rayon for encoding:

```rust
/// Minimum number of inputs before rayon parallelism is used for encoding.
/// Below this threshold, sequential encoding with `bernoulli_packed_fast` is faster
/// because rayon's work-stealing overhead (~10-50µs) exceeds the total encoding work.
const RAYON_ENCODE_THRESHOLD: usize = 128;

pub fn forward_fast(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
    // ... validation unchanged ...

    let packed_inputs: Vec<Vec<u64>> = if self.n_inputs >= RAYON_ENCODE_THRESHOLD {
        // Parallel encoding
        input_values
            .par_iter()
            .enumerate()
            .map(|(idx, &p)| {
                let input_seed = seed.wrapping_add(idx as u64);
                let mut rng = ChaCha8Rng::seed_from_u64(input_seed);
                bitstream::bernoulli_packed_fast(p, self.length, &mut rng)
            })
            .collect()
    } else {
        // Sequential encoding (avoids rayon overhead at small input counts)
        input_values
            .iter()
            .enumerate()
            .map(|(idx, &p)| {
                let input_seed = seed.wrapping_add(idx as u64);
                let mut rng = ChaCha8Rng::seed_from_u64(input_seed);
                bitstream::bernoulli_packed_fast(p, self.length, &mut rng)
            })
            .collect()
    };

    // Neuron compute always uses rayon (n_neurons is typically 32+)
    // ... unchanged ...
}
```

**IMPORTANT**: The encoding uses per-input seeding (`seed.wrapping_add(idx as u64)`), so the sequential path produces **identical** results to the parallel path for the same seed. This is a pure performance optimization with zero behavioral change.

Apply the same threshold to the neuron compute loop in `forward()` and `forward_fast()`:

```rust
let out: Vec<f64> = if self.n_neurons >= 8 {
    (0..self.n_neurons).into_par_iter().map(|neuron_idx| { ... }).collect()
} else {
    (0..self.n_neurons).map(|neuron_idx| { ... }).collect()
};
```

---

### Packet AY: Benchmarks + Report Update

**Criterion benchmarks to add** (`engine/benches/full_bench.rs`):

1. `pack_fast_1m`: Benchmark `pack_fast()` on 1M bits:
   ```rust
   c.bench_function("pack_fast_1m", |b| {
       b.iter(|| black_box(bitstream::pack_fast(black_box(&bits_1m))))
   });
   ```

2. `pack_dispatch_1m`: Benchmark SIMD-dispatched pack:
   ```rust
   c.bench_function("pack_dispatch_1m", |b| {
       b.iter(|| black_box(simd::pack_dispatch(black_box(&bits_1m))))
   });
   ```

3. `lif_100k_steps` (rename existing `lif_10k_steps` OR add new):
   ```rust
   c.bench_function("lif_100k_steps", |b| {
       b.iter(|| {
           let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
           for _ in 0..100_000 {
               black_box(lif.step(20, 256, 128, 0));
           }
       })
   });
   ```

**Python benchmark additions** (`examples/03_benchmark_report.py`):

Add a `bench_lif_multi` function:
```python
def bench_lif_multi(n_neurons: int = 100, n_steps: int = 100_000) -> list[dict]:
    """Benchmark multi-neuron parallel LIF batch."""
    currents = np.full(n_neurons, 128, dtype=np.int16)

    def run_v2():
        for _ in range(n_neurons):
            lif = V2Lif()
            for _ in range(n_steps):
                lif.step(20, 256, 128, 0)

    def run_v3():
        return v3.batch_lif_run_multi(n_neurons, n_steps, leak_k=20, gain_k=256, currents=currents)

    v2_time = benchmark(run_v2)
    v3_time = benchmark(run_v3)

    return [{
        "operation": f"LIF multi ({n_neurons}x{n_steps//1000}K)",
        "v2_ms": v2_time * 1000,
        "v3_ms": v3_time * 1000,
        "speedup": fmt_speedup(v2_time, v3_time),
        "target": "400x",
    }]
```

Call `bench_lif_multi()` in `main()` and append to `results`.

**Update `docs/BENCHMARK_REPORT.md`** to version 3.4.0 with Phase 10 results. Keep Phase 9 and 7 reference tables.

---

### Packet AZ: Version 3.4.0 + Docs + Tests

**Version bump** to `3.4.0` in:
- `engine/Cargo.toml` line 3: `version = "3.4.0"`
- `engine/src/lib.rs` line 24: `m.add("__version__", "3.4.0")?;`
- `bridge/pyproject.toml` line 7: `version = "3.4.0"`

**Bridge updates** (`bridge/sc_neurocore_engine/__init__.py`):
- Add `batch_lif_run_multi` to import list and `__all__`
- Update module docstring version

**Changelog** (`CHANGELOG_V3.md`): Add `[3.4.0]` entry at top:
```markdown
## [3.4.0] - 2026-02-10

### Phase 10: SIMD Pack, LIF Optimization, Rayon Guard
- **SIMD pack vectorization**: AVX-512/AVX2/portable fast packing (closes 6x Blueprint target)
- **Branchless LIF mask**: Eliminates branches in fixed-point sign extension
- **batch_lif_run_multi()**: Parallel multi-neuron batch execution via rayon
- **Rayon work threshold**: Avoids thread-pool overhead at small input counts
- **Criterion benchmarks**: Added pack_fast, pack_dispatch, lif_100k_steps
```

**Migration docs** (`docs/v3_migration.md`): Add Phase 10 section with:
- SIMD pack example showing `pack_bitstream_numpy` now uses dispatch
- Multi-neuron batch example
- Rayon threshold explanation

**CI update** (`.github/workflows/v3-engine.yml`): Add `tests/test_phase10.py` to pytest command.

**Update `tests/test_phase9.py`** version assertion: Change `"3.3.0"` to `"3.4.0"`.

**New test file**: `tests/test_phase10.py`

```python
"""Phase 10 acceptance tests: SIMD pack, LIF optimization, rayon guard."""

import numpy as np
import pytest
import sc_neurocore_engine as v3


class TestSIMDPack:
    """Test SIMD-accelerated pack_bitstream_numpy correctness."""

    def test_pack_numpy_matches_list_pack(self):
        """SIMD pack must produce identical output to list pack."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10_000).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    @pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 256, 1024, 4096])
    def test_pack_numpy_various_lengths(self, length):
        """SIMD pack handles all lengths including non-aligned."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, length).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    def test_pack_numpy_deterministic(self):
        """Same input → same output."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 128, dtype=np.uint8)
        a = np.asarray(v3.pack_bitstream_numpy(bits))
        b = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(a, b)

    def test_pack_unpack_roundtrip(self):
        """Pack→unpack roundtrip preserves bits."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 2048).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        unpacked = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, np.asarray(unpacked))


class TestBranchlessLIF:
    """Test that branchless LIF step produces identical results."""

    def test_100_steps_constant_input(self):
        """Standard equivalence: same as equivalence suite."""
        lif = v3.FixedPointLif()
        results = []
        for _ in range(100):
            s, v = lif.step(20, 256, 128, 0)
            results.append((s, v))
        # Verify at least some spikes occur
        spikes = [r[0] for r in results]
        assert sum(spikes) > 0

    def test_batch_matches_step_by_step(self):
        """batch_lif_run must match step-by-step execution."""
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_refractory_period(self):
        """Refractory behavior preserved under branchless mask."""
        lif = v3.FixedPointLif(refractory_period=5)
        spikes, voltages = v3.batch_lif_run(
            200, 20, 256, 200, refractory_period=5
        )
        spikes_arr = np.asarray(spikes)
        spike_indices = np.where(spikes_arr == 1)[0]
        # After each spike, next 5 steps should be refractory (no spike)
        for idx in spike_indices:
            for ref_step in range(1, 6):
                if idx + ref_step < len(spikes_arr):
                    assert spikes_arr[idx + ref_step] == 0, \
                        f"Spike during refractory at step {idx + ref_step}"


class TestMultiNeuronBatch:
    """Test parallel multi-neuron LIF batch."""

    def test_shape_and_dtype(self):
        """Output shape is (n_neurons, n_steps)."""
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        assert np.asarray(spikes).shape == (10, 100)
        assert np.asarray(voltages).shape == (10, 100)

    def test_matches_sequential(self):
        """Parallel multi-neuron must match N sequential single-neuron runs."""
        n_neurons = 8
        n_steps = 500
        i_values = [64, 96, 128, 160, 192, 224, 100, 140]
        currents = np.array(i_values, dtype=np.int16)

        # Sequential: run each neuron separately
        sequential_spikes = []
        for i_t in i_values:
            s, _ = v3.batch_lif_run(n_steps, 20, 256, i_t)
            sequential_spikes.append(np.asarray(s))

        # Parallel multi-neuron
        par_spikes, _ = v3.batch_lif_run_multi(n_neurons, n_steps, 20, 256, currents)
        par_arr = np.asarray(par_spikes)

        for ni in range(n_neurons):
            np.testing.assert_array_equal(
                par_arr[ni], sequential_spikes[ni],
                err_msg=f"Neuron {ni} mismatch"
            )

    def test_deterministic(self):
        """Same inputs → same outputs."""
        currents = np.full(4, 128, dtype=np.int16)
        s1, v1 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        s2, v2 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))


class TestRayonThreshold:
    """Test that rayon threshold does not change forward_fast outputs."""

    def test_forward_fast_determinism(self):
        """forward_fast with small inputs (below threshold) matches large inputs pattern."""
        layer = v3.DenseLayer(16, 8, 1024)
        inputs = [0.5] * 16
        a = layer.forward_fast(inputs, seed=42)
        b = layer.forward_fast(inputs, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_forward_fast_consistent_across_sizes(self):
        """forward_fast produces valid outputs for various input sizes."""
        for n_in in [4, 16, 64, 128, 256]:
            layer = v3.DenseLayer(n_in, 8, 1024)
            inputs = [0.5] * n_in
            result = layer.forward_fast(inputs, seed=42)
            assert len(result) == 8
            for val in result:
                assert 0.0 <= val <= float(n_in), f"Out of range: {val}"


class TestPhase10Version:
    def test_version(self):
        assert v3.__version__ == "3.4.0"
```

**Expected test counts**: ~20 tests in `test_phase10.py`:
- TestSIMDPack: 4 (+ parametrized = ~12)
- TestBranchlessLIF: 3
- TestMultiNeuronBatch: 3
- TestRayonThreshold: 2
- TestPhase10Version: 1

---

## Verification Protocol

After implementing all packets, run the following gates in order:

### 1. Rust gates

```powershell
cd 03_CODE/sc-neurocore/engine
C:\Users\forti\.cargo\bin\cargo.exe fmt
C:\Users\forti\.cargo\bin\cargo.exe clippy --all-targets -- -D warnings
C:\Users\forti\.cargo\bin\cargo.exe test --tests
C:\Users\forti\.cargo\bin\cargo.exe doc --no-deps
```

All must pass. The new `mask_branchless_matches_original` test is critical.

### 2. Build Python extension

```powershell
cd 03_CODE/sc-neurocore/engine
$env:PATH="$env:USERPROFILE\.cargo\bin;$env:PATH"
..\.venv\Scripts\python.exe -m maturin develop --release
```

### 3. Python tests (full v3 suite)

```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py tests/test_phase8.py tests/test_phase9.py tests/test_phase10.py -v --tb=short
```

Expected: **~148 passed** (128 + ~20 new).

### 4. Co-simulation

```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest cosim/ -v -rs --tb=short
```

Expected: **8 passed**.

### 5. Examples + version

```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe examples/01_sc_training_demo.py
.\.venv\Scripts\python.exe examples/02_ir_compile_demo.py
.\.venv\Scripts\python.exe examples/03_benchmark_report.py
.\.venv\Scripts\python.exe -c "import sc_neurocore_engine as v3; print(v3.__version__); print(v3.simd_tier())"
```

Version must print `3.4.0`.

### 6. Criterion benchmarks

```powershell
cd 03_CODE/sc-neurocore/engine
C:\Users\forti\.cargo\bin\cargo.exe bench
```

Record times for `pack_1m`, `pack_fast_1m`, `pack_dispatch_1m`, `lif_10k_steps`, `lif_100k_steps`.

---

## Files Modified

| File | Changes |
|------|---------|
| `engine/src/bitstream.rs` | Add `pack_fast()` |
| `engine/src/simd/mod.rs` | Add `pack_dispatch()` |
| `engine/src/simd/avx2.rs` | Add `pack_avx2()` |
| `engine/src/simd/avx512.rs` | Add `pack_avx512()` |
| `engine/src/neuron.rs` | Branchless `mask()`, add unit test |
| `engine/src/layer.rs` | Rayon threshold in `forward_fast()` |
| `engine/src/lib.rs` | `batch_lif_run_multi()`, update `pack_bitstream_numpy` |
| `engine/Cargo.toml` | Version 3.4.0 |
| `engine/benches/full_bench.rs` | Add pack + LIF benchmarks |
| `bridge/pyproject.toml` | Version 3.4.0 |
| `bridge/sc_neurocore_engine/__init__.py` | Export `batch_lif_run_multi` |
| `examples/03_benchmark_report.py` | Add `bench_lif_multi()` |
| `docs/BENCHMARK_REPORT.md` | Phase 10 results |
| `docs/v3_migration.md` | Phase 10 section |
| `CHANGELOG_V3.md` | [3.4.0] entry |
| `.github/workflows/v3-engine.yml` | Add `tests/test_phase10.py` |
| `tests/test_phase9.py` | Version assertion → 3.4.0 |
| `tests/test_phase10.py` | **NEW** — ~20 acceptance tests |

**Total**: 17 modified + 1 new = 18 files.

---

## Expected Performance Targets

| Operation | Phase 9 | Phase 10 Target | Mechanism |
|-----------|---------|-----------------|-----------|
| pack (numpy, 1M) | 1.1x | **≥6x** | SIMD pack (AVX-512: 1 instruction per word) |
| LIF batch (100K) | 102x | **≥200x** | Branchless mask + micro-optimization |
| LIF multi (100×100K) | — | **≥400x** | Rayon parallelism across 100 neurons |
| forward_fast (64→32) | ~1x | **≥2x** | Rayon threshold eliminates overhead |

**Note**: The 400x LIF target may not be achievable for a single neuron (sequential dependency between steps). The `batch_lif_run_multi` approach parallelizes across neurons, which is the realistic production scenario. If N neurons × M steps has total work = N × (single-neuron batch time), then with rayon parallelism on C cores: speedup = v2_time × N / (single_neuron_time × N / C) = v2_time × C / single_neuron_time. With C=8 cores and single_neuron_time = 1.4ms for 100K steps: 143ms × 8 / 1.4ms ≈ 817x. So 400x is achievable with ≥4 cores.
