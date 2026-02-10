# SC-NeuroCore v3 — Phase 12 Codex Handover

**Author**: Claude Opus 4.6
**Date**: 2026-02-10
**Baseline**: v3.5.0 (tag `v3.5.0-phase11`, commit `07594bdd3`)
**Target Version**: 3.6.0
**Semantics Mode**: Strict blueprint semantics

---

## Objective

Push beyond Blueprint targets with three high-impact architectural optimizations:

1. **Fused encode+AND+popcount kernel** — eliminate materialized input bitstreams from memory
2. **Fast PRNG (xoshiro256++)** — replace cryptographic ChaCha8 with fast statistical PRNG
3. **Batched multi-sample forward** — amortize FFI overhead across N input vectors

These three changes are complementary and compound multiplicatively. Expected combined effect: 2-5x improvement on dense forward throughput at the Python level.

---

## Backup

Current state is tagged: `v3.5.0-phase11`

Restore command if needed:
```powershell
git checkout v3.5.0-phase11 -- 03_CODE/sc-neurocore/
```

---

## Packets

### Packet BF: Fused Encode+AND+Popcount Kernel

**Problem**: `forward_fast()` in `layer.rs:184-204` encodes all inputs into `Vec<Vec<u64>>` (heap allocation per input), then reads them back in the neuron accumulation loop (lines 206-239). For a 64-input, L=1024 layer, this materializes 64 × 16 × 8 = 8 KB of intermediate data that is written once and read once — a pure waste of memory bandwidth.

**Solution**: A new function that generates random bytes, compares against threshold, ANDs with weight words, and popcounts — all in one pass per (neuron, input) pair. The encoded bitstream never touches memory.

**New function in `bitstream.rs`**:
```rust
/// Fused encode+AND+popcount: generate Bernoulli bits on-the-fly, AND with
/// weight words, accumulate popcount. Returns the total popcount.
///
/// This is semantically equivalent to:
///   let encoded = bernoulli_packed_simd(prob, length, rng);
///   fused_and_popcount_dispatch(&weight_words, &encoded)
/// but avoids materializing the encoded bitstream.
pub fn encode_and_popcount<R: Rng + ?Sized>(
    weight_words: &[u64],
    prob: f64,
    length: usize,
    rng: &mut R,
) -> u64 {
    let threshold = (prob.clamp(0.0, 1.0) * 256.0).min(255.0) as u8;
    let full_words = length / 64;
    let mut total = 0_u64;
    let mut buf = [0_u8; 64];

    for (word_idx, &w_word) in weight_words.iter().take(full_words).enumerate() {
        rng.fill(&mut buf);
        let encoded = simd_bernoulli_compare(&buf, threshold);
        total += (encoded & w_word).count_ones() as u64;
    }

    // Tail (< 64 bits)
    let remaining = length.saturating_sub(full_words * 64);
    if remaining > 0 && full_words < weight_words.len() {
        rng.fill(&mut buf[..remaining]);
        let mut encoded = 0_u64;
        for (bit, &rb) in buf[..remaining].iter().enumerate() {
            if rb < threshold {
                encoded |= 1_u64 << bit;
            }
        }
        total += (encoded & weight_words[full_words]).count_ones() as u64;
    }

    total
}
```

**SIMD-accelerated variant** — add to `simd/mod.rs`:
```rust
/// Fused encode+AND+popcount with SIMD Bernoulli compare.
/// Generates random bytes, compares threshold via SIMD, ANDs with weight, popcounts.
pub fn encode_and_popcount_dispatch<R: Rng + ?Sized>(
    weight_words: &[u64],
    prob: f64,
    length: usize,
    rng: &mut R,
) -> u64
```
This calls `simd_bernoulli_compare` (already exists, `#[inline]`) for the compare stage, then uses `count_ones()` for the AND+popcount. The `count_ones()` compiles to hardware `popcnt` on all target platforms.

**Note on AVX-512 full fusion**: A fully-fused AVX-512 version that does AND+POPCNT in-register on 8 words at once would require generating 8×64=512 random bytes per iteration. Since the PRNG is the bottleneck (not the compare/AND/popcount), the word-at-a-time approach with SIMD compare is the correct design.

**New `forward_fused()` in `layer.rs`**:
```rust
/// Forward pass with fused encode+AND+popcount — zero intermediate allocation.
///
/// Each input is encoded on-the-fly and immediately consumed.
/// No `Vec<Vec<u64>>` materialized.
pub fn forward_fused(&self, input_values: &[f64], seed: u64) -> Result<Vec<f64>, String> {
    // validation...

    let out: Vec<f64> = if self.n_neurons >= RAYON_NEURON_THRESHOLD {
        (0..self.n_neurons)
            .into_par_iter()
            .map(|neuron_idx| {
                let total: u64 = input_values
                    .iter()
                    .enumerate()
                    .map(|(input_idx, &p)| {
                        let input_seed = seed.wrapping_add(input_idx as u64);
                        let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                        bitstream::encode_and_popcount(
                            self.weight_slice(neuron_idx, input_idx),
                            p,
                            self.length,
                            &mut rng,
                        )
                    })
                    .sum();
                total as f64 / self.length as f64
            })
            .collect()
    } else {
        // sequential variant with identical logic
    };

    Ok(out)
}
```

**Critical semantic note**: `forward_fused()` re-creates the RNG for each (neuron, input) pair. This means each neuron sees the same encoded input (same seed per input index). This matches `forward_fast()` semantics where inputs are encoded once and shared across neurons. The difference: `forward_fast` materializes; `forward_fused` regenerates. Since the PRNG is deterministic, the results are **bit-identical** for the same seed.

**Determinism verification**: `forward_fused(inputs, seed)` MUST produce the same output as `forward_fast(inputs, seed)` with the same PRNG type. Add a unit test confirming bit-exact equivalence.

**Integration in `layer.rs`**:
- Add `forward_fused()` as a new public method
- Update `forward_numpy_inner()` to call `forward_fused()` instead of `forward_fast()`
- Keep `forward_fast()` unchanged (backward compatibility)

**Integration in `lib.rs`**:
- `DenseLayer.forward_fast` Python binding calls `forward_fused()` internally
- `DenseLayer.forward_numpy` Python binding calls `forward_fused()` internally

---

### Packet BG: Fast PRNG (xoshiro256++)

**Problem**: ChaCha8 costs ~4 ns per u64 of output. For L=1024 (16 words × 64 bytes = 1024 bytes per input), encoding one input costs ~512 ns in PRNG alone. For 64 inputs, that's ~33 µs just in random number generation — a significant fraction of the ~165 µs dense forward time.

**Solution**: Replace ChaCha8 with xoshiro256++ for all encoding paths. xoshiro256++ costs ~0.7 ns per u64, a 5.7x improvement on PRNG throughput.

**Dependency**: Add to `engine/Cargo.toml`:
```toml
rand_xoshiro = "0.6"
```

**Changes in `layer.rs`**:
```rust
use rand_xoshiro::Xoshiro256PlusPlus;
// Replace all:  ChaCha8Rng::seed_from_u64(input_seed)
// With:         Xoshiro256PlusPlus::seed_from_u64(input_seed)
```

**Affected functions in `layer.rs`**:
- `forward()` — input encoding RNG
- `forward_fast()` — input encoding RNG
- `forward_fused()` — per-(neuron, input) RNG (Packet BF)
- `forward_numpy_inner()` — delegates to fused

**Affected functions in `lib.rs`**:
- `batch_encode_numpy()` — per-probability RNG (line 504)

**NOT affected** (keep ChaCha8 for backward compatibility):
- `refresh_packed_weights()` — weight encoding uses `weight_seed`, must remain ChaCha8 for bit-exact weight compatibility with existing models
- `bernoulli_packed()` — the f64-per-bit original function (reference implementation)
- `batch_encode()` — the list-based batch encode (non-performance path)

**Determinism**: xoshiro256++ is fully deterministic from seed. The output will differ from ChaCha8 for the same seed, but:
1. `forward_fast()` already produces different results from `forward()` (different seeding strategy)
2. The only contract is: same function + same seed = same output (intra-function determinism)
3. All existing tests that compare `forward_fast` output use fixed seeds and will need updated expected values

**Test strategy**:
- Update existing `forward_fast` tests to compare against themselves (determinism), not against specific expected values
- Add `forward_fused_matches_forward_fast` equivalence test (both use xoshiro, must be bit-identical)
- Add statistical correctness tests (same pattern as bernoulli_packed_simd_statistics)

**Benchmark note**: Add a new criterion benchmark `bernoulli_packed_simd_xoshiro_1024` to measure the PRNG improvement in isolation.

---

### Packet BH: Batched Multi-Sample Forward

**Problem**: Python training/inference loops call `forward_fast()` once per sample. Each call crosses the FFI boundary, acquires/releases numpy arrays, and creates/destroys thread pool work items. For a batch of 100 samples, this overhead is paid 100 times.

**Solution**: A new `forward_batch()` method that processes N samples in one FFI call.

**New method in `layer.rs`**:
```rust
/// Batched forward pass: process N input vectors in one call.
///
/// `inputs_flat` is row-major: `[n_samples, n_inputs]`.
/// Returns flat output: `[n_samples, n_neurons]`.
pub fn forward_batch(
    &self,
    inputs_flat: &[f64],
    n_samples: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    if inputs_flat.len() != n_samples * self.n_inputs {
        return Err(format!(
            "Expected {} values ({}×{}), got {}.",
            n_samples * self.n_inputs, n_samples, self.n_inputs, inputs_flat.len()
        ));
    }

    let mut output = vec![0.0_f64; n_samples * self.n_neurons];

    // Parallel over samples (each sample is independent)
    output
        .par_chunks_mut(self.n_neurons)
        .enumerate()
        .for_each(|(sample_idx, out_row)| {
            let input_row = &inputs_flat[sample_idx * self.n_inputs..(sample_idx + 1) * self.n_inputs];
            let sample_seed = seed.wrapping_add((sample_idx as u64) * 1_000_000);

            for (neuron_idx, out_val) in out_row.iter_mut().enumerate() {
                let total: u64 = input_row
                    .iter()
                    .enumerate()
                    .map(|(input_idx, &p)| {
                        let input_seed = sample_seed.wrapping_add(input_idx as u64);
                        let mut rng = Xoshiro256PlusPlus::seed_from_u64(input_seed);
                        bitstream::encode_and_popcount(
                            self.weight_slice(neuron_idx, input_idx),
                            p,
                            self.length,
                            &mut rng,
                        )
                    })
                    .sum();
                *out_val = total as f64 / self.length as f64;
            }
        });

    Ok(output)
}
```

**Seed strategy**: `sample_seed = seed + sample_idx * 1_000_000` ensures non-overlapping seed spaces across samples (each sample has 1M seed slots for its inputs). This prevents seed collision for up to 1M inputs per sample.

**Python binding in `lib.rs`**:
```rust
/// Batched dense forward: process N samples in one FFI call.
///
/// Accepts numpy float64 array of shape (n_samples, n_inputs).
/// Returns numpy float64 array of shape (n_samples, n_neurons).
#[pyfunction]
#[pyo3(signature = (packed_inputs, seed=44257))]
fn forward_batch_numpy<'py>(
    &self,
    py: Python<'py>,
    inputs: PyReadonlyArray2<'py, f64>,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>>
```

**Pre-allocated output**: Use `PyArray2::zeros_bound` + `as_slice_mut` (same pattern as `batch_lif_run_multi` in Packet BD), write results directly via `par_chunks_mut`.

**Integration in `lib.rs`**:
- Add `forward_batch_numpy` as a method on the `DenseLayer` PyO3 class
- Register in module init (already happens via `#[pymethods]`)

---

### Packet BI: Version 3.6.0 + Benchmarks + Docs + Tests

**Version bump to 3.6.0** in all 6 sites:
1. `engine/Cargo.toml` — `version = "3.6.0"`
2. `engine/src/lib.rs` — `m.add("__version__", "3.6.0")`
3. `bridge/pyproject.toml` — `version = "3.6.0"`
4. `bridge/sc_neurocore_engine/__init__.py` — docstring `v3.6`
5. `CHANGELOG_V3.md` — `[3.6.0]` entry
6. `tests/test_phase12.py` — `assert v3.__version__ == "3.6.0"`

**Update prior phase version assertions**:
- `tests/test_phase8.py` — "3.6.0"
- `tests/test_phase9.py` — "3.6.0"
- `tests/test_phase10.py` — "3.6.0"
- `tests/test_phase11.py` — "3.6.0"

**New criterion benchmarks** in `engine/benches/full_bench.rs`:
1. `dense_forward_fused_64x32` — fused kernel (no intermediate allocation)
2. `bernoulli_encode_and_popcount_1024` — fused encode+AND+popcount single pair
3. `dense_forward_batch_64x32_x100` — 100 samples batched
4. `prng_xoshiro_fill_1024` — xoshiro256++ fill 1024 bytes (vs ChaCha8 baseline)

**New Python tests** in `tests/test_phase12.py`:
```
class TestFusedKernel:
    test_fused_matches_forward_fast      — bit-exact equivalence (same PRNG)
    test_fused_determinism               — same seed → same output
    test_fused_statistical_correctness   — high/low probability ordering preserved

class TestFastPRNG:
    test_xoshiro_determinism             — same seed → same output
    test_xoshiro_statistical_quality     — 10K-bit runs, ±3% tolerance
    test_forward_fast_determinism_new    — forward_fast still deterministic

class TestBatchForward:
    test_batch_vs_sequential             — batch output matches N sequential calls
    test_batch_shape                     — output shape (n_samples, n_neurons)
    test_batch_determinism               — same seed → same output
    test_batch_numpy_output              — returns proper numpy array

class TestPhase12Version:
    test_version                         — "3.6.0"
```

**Update Python benchmark** `examples/03_benchmark_report.py`:
- Add `bench_dense_fused()` — fused forward timing
- Add `bench_dense_batch()` — batched forward (100 samples) timing
- Report fused vs fast speedup ratio

**Update docs**:
- `CHANGELOG_V3.md` — [3.6.0] entry
- `docs/v3_migration.md` — Phase 12 section
- `docs/BENCHMARK_REPORT.md` — Phase 12 results

**Update CI**:
- `.github/workflows/v3-engine.yml` — add `tests/test_phase12.py` to pytest command

---

## Execution Order

1. **BG first** (PRNG swap) — minimal code change, establishes the fast PRNG that BF and BH depend on
2. **BF second** (fused kernel) — uses xoshiro from BG, adds the core new function
3. **BH third** (batch forward) — uses fused kernel from BF
4. **BI last** (version/docs/tests) — references all three features

---

## Verification Gates

Run in order after all packets are implemented:

### Gate 1: Rust
```powershell
cd 03_CODE/sc-neurocore/engine
C:\Users\forti\.cargo\bin\cargo.exe fmt
C:\Users\forti\.cargo\bin\cargo.exe clippy --all-targets -- -D warnings
C:\Users\forti\.cargo\bin\cargo.exe test --tests
C:\Users\forti\.cargo\bin\cargo.exe doc --no-deps
```

### Gate 2: Build
```powershell
cd 03_CODE/sc-neurocore/engine
$env:PATH="$env:USERPROFILE\.cargo\bin;$env:PATH"
..\.venv\Scripts\python.exe -m maturin develop --release
```

### Gate 3: Python tests (full suite)
```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest tests/equivalence tests/test_surrogate_python.py tests/test_kuramoto_python.py tests/test_kuramoto_ssgf_python.py tests/test_multihead_attention.py tests/test_gnn_sc_mode.py tests/test_ir_python.py tests/test_numpy_interop.py tests/test_batch_ops.py tests/test_dense_optimization.py tests/test_phase8.py tests/test_phase9.py tests/test_phase10.py tests/test_phase11.py tests/test_phase12.py -v --tb=short
```

### Gate 4: Co-simulation
```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe -m pytest cosim/ -v -rs --tb=short
```

### Gate 5: Examples + version
```powershell
$env:PYTHONPATH='src;bridge'
.\.venv\Scripts\python.exe examples/01_sc_training_demo.py
.\.venv\Scripts\python.exe examples/02_ir_compile_demo.py
.\.venv\Scripts\python.exe examples/03_benchmark_report.py
.\.venv\Scripts\python.exe -c "import sc_neurocore_engine as v3; print(v3.__version__); print(v3.simd_tier())"
```

Expected: version `3.6.0`, all tests pass, all examples run.

### Gate 6: Criterion benchmarks
```powershell
cd 03_CODE/sc-neurocore/engine
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench dense_forward_fused
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench encode_and_popcount
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench dense_forward_batch
C:\Users\forti\.cargo\bin\cargo.exe bench --bench full_bench prng_xoshiro
```

---

## Sacred Files — DO NOT MODIFY

| Path | Reason |
|------|--------|
| `src/sc_neurocore/` | v2 Python package (dual-stack coexistence) |
| `pyproject.toml` (repo root) | v2 package config |
| `.github/workflows/ci.yml` | v2 CI pipeline |

---

## Expected Performance Targets

| Operation | Phase 11 Best | Phase 12 Target | Rationale |
|-----------|--------------|----------------|-----------|
| dense forward fast (64→32, L=1024) | 0.171 ms | < 0.08 ms | Fused kernel + fast PRNG |
| dense prepacked numpy | 0.033 ms | 0.033 ms (unchanged) | Pre-packed path unaffected |
| dense batch (64→32, L=1024, 100 samples) | ~17 ms (100× forward_fast) | < 5 ms | Batched + fused + fast PRNG |
| Criterion: dense_forward_fast | 163 µs | < 80 µs | Fused kernel eliminates allocation |
| Criterion: encode_and_popcount (single pair, 16w) | N/A (new) | < 300 ns | Single encode+AND+popcount |

---

## Files to Modify

| File | Packets | Changes |
|------|---------|---------|
| `engine/Cargo.toml` | BG, BI | Add `rand_xoshiro`, version 3.6.0 |
| `engine/src/bitstream.rs` | BF | Add `encode_and_popcount()` |
| `engine/src/simd/mod.rs` | BF | Add `encode_and_popcount_dispatch()` (optional, if SIMD accumulation is added) |
| `engine/src/layer.rs` | BF, BG, BH | Add `forward_fused()`, `forward_batch()`, replace ChaCha8 with xoshiro in forward paths |
| `engine/src/lib.rs` | BH, BI | Add `forward_batch_numpy` binding, version 3.6.0 |
| `engine/benches/full_bench.rs` | BI | 4 new benchmarks |
| `bridge/pyproject.toml` | BI | version 3.6.0 |
| `bridge/sc_neurocore_engine/__init__.py` | BI | docstring v3.6 |
| `examples/03_benchmark_report.py` | BI | fused + batch benchmarks |
| `CHANGELOG_V3.md` | BI | [3.6.0] entry |
| `docs/v3_migration.md` | BI | Phase 12 section |
| `docs/BENCHMARK_REPORT.md` | BI | Phase 12 results |
| `.github/workflows/v3-engine.yml` | BI | Add test_phase12.py |
| `tests/test_phase8.py` | BI | Version → "3.6.0" |
| `tests/test_phase9.py` | BI | Version → "3.6.0" |
| `tests/test_phase10.py` | BI | Version → "3.6.0" |
| `tests/test_phase11.py` | BI | Version → "3.6.0" |
| `tests/test_phase12.py` (NEW) | BI | 11 tests across 4 classes |

---

## Critical Correctness Constraint

The fused kernel (BF) regenerates the PRNG per (neuron, input) pair instead of encoding once and sharing. This is correct ONLY if:
1. The same PRNG type is used in both `forward_fast` and `forward_fused`
2. The same seed derivation is used: `seed.wrapping_add(input_idx as u64)`

After Packet BG swaps both to xoshiro256++, `forward_fused` and `forward_fast` MUST produce bit-identical output. The `test_fused_matches_forward_fast` test MUST pass. If it does not, the fused kernel has a seeding or RNG consumption bug.

---

## Session Log

Write results to `SESSION_LOG_2026-02-10_V3_MIGRATION_PHASE12.md` following the same format as Phase 11.
