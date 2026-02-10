# SC-NeuroCore v3.0 — "Metal Engine" Migration Blueprint

**Status:** Phase 1 Architecture & Codex Work Packets
**Author:** Principal Systems Architect
**Date:** February 2026
**Constraint:** v2.2.0 Python code is SACRED. Zero modifications.

---

## 1. THE ARCHITECTURE DECISION: Rust + PyO3

### Verdict: **Rust (2024 Edition) with PyO3 + maturin**

### Justification

| Criterion | Rust + PyO3 | C++26 + pybind11 | Winner |
|-----------|-------------|-------------------|--------|
| **Memory Safety** | Borrow checker eliminates use-after-free, double-free, buffer overflows at compile time. Zero unsafe needed for bitstream math. | Manual memory management. Undefined behavior from pointer arithmetic on packed arrays is a real risk. | **Rust** |
| **AVX-512 SIMD** | `core::arch::x86_64` has full AVX-512 intrinsics. `std::simd` (nightly) provides portable SIMD. Stable `_mm512_and_epi64`, `_mm512_popcnt_epi64` via `#[target_feature]`. | Intel Intrinsics are first-class. Identical intrinsic coverage. | **Tie** |
| **NEON (ARM)** | `core::arch::aarch64` has full NEON intrinsics. Same pattern as x86. | Identical via `<arm_neon.h>`. | **Tie** |
| **Python Binding** | PyO3 is ergonomic: `#[pyclass]`, `#[pymethods]`, numpy interop via `numpy` crate. `maturin` builds wheels in one command. | pybind11 is mature but requires CMake + setuptools dance. Header-only but compilation is fragile. | **Rust** |
| **Build System** | `cargo` — single command, reproducible, cross-platform. `maturin develop` for dev iteration. | CMake — powerful but complex. Windows/Linux/macOS differences cause CI pain. | **Rust** |
| **Type Safety** | `u64` is `u64`. No implicit conversions. Bit-width errors caught at compile time. The 5-stage SWAR popcount cannot silently overflow. | Implicit integer promotions (e.g., `uint64_t >> int` can sign-extend). Silent bugs in bitwise code. | **Rust** |
| **Concurrency** | `rayon` for data-parallel `par_iter()` over neuron batches. Send/Sync enforced at compile time. | OpenMP `#pragma omp parallel for`. Race conditions not checked at compile time. | **Rust** |
| **Ecosystem** | `ndarray` (NumPy-like), `rand` (PRNG), `criterion` (benchmarks), `proptest` (property testing). | Eigen, Intel MKL, Google Benchmark. More mature numerical libs. | **Tie** |
| **Deployment** | `maturin build --release` produces a `.whl` with no runtime dependencies. Ships as a single `.pyd`/`.so`. | Requires compiler toolchain on target or prebuilt binary. Shared libs (.dll/.so) need careful RPATH. | **Rust** |

**Final Score: Rust wins 5-0 with 4 ties.** The decisive factors are memory safety (critical for bitwise operations where off-by-one errors are catastrophic), PyO3 ergonomics, and the `maturin` build story.

### Specific SIMD Strategy

The hot path is: **encode (Bernoulli) -> pack (uint8 -> uint64) -> AND (multiply) -> POPCOUNT (accumulate)**.

**Tier 1 — Portable (all platforms):** Rust `u64` operations with SWAR popcount. This is the baseline that always works. Already faster than Python/NumPy because no interpreter overhead.

**Tier 2 — AVX2 (x86_64, ~95% of desktop/server):** Process 4 x u64 = 256 bits per instruction via `_mm256_and_si256` and manual SWAR popcount.

**Tier 3 — AVX-512 (Intel Xeon, Ice Lake+):** Process 8 x u64 = 512 bits per instruction via `_mm512_and_epi64` + native `_mm512_popcnt_epi64` (VPOPCNTDQ). This is the maximum throughput path.

**Tier 4 — NEON (Apple Silicon, ARM servers):** Process 2 x u64 = 128 bits per instruction via `vand_u8` + `vcnt_u8` (hardware popcount on ARM).

Runtime detection via `is_x86_feature_detected!("avx512vpopcntdq")` selects the fastest available path at startup.

---

## 2. THE DUAL-STACK DIRECTORY STRUCTURE

```
sc-neurocore/                          # Repository root
├── pyproject.toml                     # Existing v2.2.0 config (UNTOUCHED)
├── Cargo.toml                         # NEW — Rust workspace root
├── src/
│   └── sc_neurocore/                  # EXISTING v2.2.0 Python (UNTOUCHED)
│       ├── __init__.py                # Current public API
│       ├── neurons/
│       ├── layers/
│       ├── synapses/
│       ├── accel/
│       └── ...                        # All 42+ packages preserved
│
├── engine/                            # NEW — Rust crate ("The Metal")
│   ├── Cargo.toml                     # Rust crate config with PyO3
│   ├── src/
│   │   ├── lib.rs                     # PyO3 module entry point
│   │   ├── bitstream.rs               # BitStreamTensor (SIMD kernels)
│   │   ├── neuron.rs                  # LIF neuron (fixed-point, bit-true)
│   │   ├── layer.rs                   # Dense layer (MAC pipeline)
│   │   ├── encoder.rs                 # LFSR + Bernoulli encoder
│   │   ├── popcount.rs                # SWAR + AVX-512 popcount
│   │   ├── simd/
│   │   │   ├── mod.rs                 # Runtime dispatch
│   │   │   ├── avx2.rs               # AVX2 kernels
│   │   │   ├── avx512.rs             # AVX-512 kernels
│   │   │   └── neon.rs               # ARM NEON kernels
│   │   └── grad/
│   │       ├── mod.rs                 # Surrogate gradient framework
│   │       └── surrogate.rs           # FastSigmoid, ArcTan surrogates
│   ├── benches/
│   │   └── bitstream_bench.rs         # criterion benchmarks
│   └── tests/
│       └── equivalence.rs             # Rust-side equivalence checks
│
├── bridge/                            # NEW — Python bridge package
│   ├── __init__.py                    # `from sc_neurocore_engine import ...`
│   └── sc_neurocore_engine/
│       ├── __init__.py                # Re-exports from Rust .so/.pyd
│       ├── layers.py                  # Python wrapper: EngineVectorizedSCLayer
│       ├── neurons.py                 # Python wrapper: EngineLIFNeuron
│       └── compat.py                  # Duck-type adapter (v3 API = v2 API)
│
├── tests/
│   ├── ...                            # EXISTING v2.2.0 tests (UNTOUCHED)
│   └── equivalence/                   # NEW — v2 vs v3 comparison tests
│       ├── conftest.py                # Shared fixtures (seeded RNG)
│       ├── test_bitstream_equiv.py    # pack/unpack/AND/popcount
│       ├── test_neuron_equiv.py       # LIF neuron step-by-step
│       ├── test_layer_equiv.py        # Dense layer forward pass
│       └── test_encoder_equiv.py      # LFSR sequence matching
│
├── scripts/
│   ├── ...                            # EXISTING scripts (UNTOUCHED)
│   └── bench_v2_vs_v3.py             # NEW — Head-to-head benchmark
│
└── docs/
    └── v3_migration.md                # NEW — Migration guide
```

### Import Pattern (User-Facing)

```python
# ---- Using v2 (Legacy Golden Reference) ----
from sc_neurocore.layers import VectorizedSCLayer
from sc_neurocore.accel import pack_bitstream, vec_popcount

# ---- Using v3 (Metal Engine) ----
from sc_neurocore_engine.layers import VectorizedSCLayer  # Same API!
from sc_neurocore_engine import pack_bitstream, vec_popcount

# ---- Side-by-side verification ----
from sc_neurocore.layers import VectorizedSCLayer as V2Layer
from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer

v2 = V2Layer(n_inputs=32, n_neurons=16, length=1024)
v3 = V3Layer(n_inputs=32, n_neurons=16, length=1024)
# ... run same inputs, compare outputs
```

---

## 3. THE EQUIVALENCE ENGINE

### Philosophy

Every v3 function must produce **identical output** to v2 given the same inputs and RNG seed. The test suite is the contract:

```
FAIL if v3_output != v2_output  (within floating-point tolerance for rates)
FAIL if v3_spike_train != v2_spike_train  (exact match for integer outputs)
FAIL if v3_lfsr_sequence != v2_lfsr_sequence  (bit-exact for 65,535 steps)
```

### Test Matrix

| Component | v2 Reference | v3 Under Test | Comparison |
|-----------|-------------|---------------|------------|
| LFSR step | `FixedPointLFSR(seed=0xACE1).step()` x 65535 | `engine::Lfsr16::new(0xACE1).step()` x 65535 | Bit-exact sequence |
| Bitstream encode | `FixedPointBitstreamEncoder(seed=0xACE1).step(x)` | `engine::BitstreamEncoder::step(x)` | Bit-exact output |
| pack_bitstream | `accel.pack_bitstream(bits)` | `engine.pack_bitstream(bits)` | Bit-exact uint64 array |
| vec_popcount | `accel.vec_popcount(packed)` | `engine.popcount(packed)` | Exact integer match |
| LIF neuron | `FixedPointLIFNeuron.step(leak, gain, I, noise)` | `engine::FixedPointLif::step(...)` | `(spike, v_out)` exact match |
| Dense forward | `VectorizedSCLayer.forward(inputs)` | `engine::DenseLayer::forward(inputs)` | `|rate_v2 - rate_v3| < 1e-10` |

---

## 4. SURROGATE GRADIENTS (New Capability in v3)

The v2 forward pass is:
```
input_prob -> Bernoulli -> bitstream -> AND(weight_bitstream) -> popcount -> rate
```

The `AND` gate and `popcount` are non-differentiable (discrete). For backpropagation, v3 introduces **surrogate gradients** on the forward pass:

**Forward:** Unchanged. Bit-true identical to v2.
**Backward:** Replace the step-function derivative with a smooth surrogate:

- **Fast Sigmoid:** `d/dx = 1 / (1 + k|x|)^2` where `k` = steepness
- **SuperSpike:** `d/dx = 1 / (k * |x| + 1)^2`
- **ArcTan:** `d/dx = 1 / (1 + (kx)^2)`

The surrogate gradient is applied ONLY during backpropagation. The forward pass remains bit-true. This is the standard approach in spiking neural network training (Neftci et al. 2019, Zenke & Ganguli 2018).

Implementation in Rust:
```rust
pub struct SurrogateLif {
    lif: FixedPointLif,          // Bit-true forward
    surrogate: SurrogateType,     // Gradient approximation
    membrane_trace: Vec<f32>,     // Saved for backward pass
}

impl SurrogateLif {
    pub fn forward(&mut self, ...) -> (i32, i32) {
        // Identical to v2 FixedPointLIFNeuron.step()
        let (spike, v_out) = self.lif.step(leak_k, gain_k, i_t, noise);
        self.membrane_trace.push(v_out as f32);
        (spike, v_out)
    }

    pub fn backward(&self, grad_output: f32) -> f32 {
        // Surrogate gradient: smooth approximation of Heaviside derivative
        let v = *self.membrane_trace.last().unwrap();
        let threshold = self.lif.v_threshold as f32;
        let x = v - threshold;
        match self.surrogate {
            SurrogateType::FastSigmoid { k } => {
                grad_output / (1.0 + k * x.abs()).powi(2)
            }
            SurrogateType::ArcTan { k } => {
                grad_output / (1.0 + (k * x).powi(2))
            }
        }
    }
}
```

---

## 5. MLIR/CIRCT COMPILER (Future Phase)

The current v2 `VerilogGenerator` emits HDL via string templates. v3 will optionally support an MLIR-based compilation flow:

```
User Python API
    │
    ▼
sc_neurocore_engine (Rust)
    │
    ▼
MLIR IR (custom "sc" dialect)
    │ Lowering passes:
    │ sc.dense_layer → hw.module (CIRCT)
    │ sc.lif_neuron → comb.and, seq.compreg
    ▼
CIRCT (Circuit IR Compiler Toolkit)
    │
    ▼
SystemVerilog / Verilog / FIRRTL
    │
    ▼
Vivado / Yosys synthesis
```

This replaces the string-template approach with a proper compiler that can:
1. Optimize the compute graph (fuse operations, eliminate dead code)
2. Target multiple backends (FPGA, ASIC, simulation)
3. Insert pipeline registers automatically based on clock constraints

**This is Phase 3+ work.** Not part of initial v3 delivery.

---

## 6. CODEX WORK PACKETS — PHASE 1

---

### PACKET A: PROJECT SCAFFOLDING

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET A: Project Scaffolding
═══════════════════════════════════════════════════════════════

CONTEXT:
You are setting up a new Rust crate inside an existing Python project.

- Repository root: sc-neurocore/
- Existing Python package: src/sc_neurocore/ (DO NOT MODIFY)
- Existing pyproject.toml (DO NOT MODIFY)
- New Rust crate location: engine/
- New Python bridge package: bridge/sc_neurocore_engine/

GOAL:
Create the scaffolding for a Rust crate that compiles to a Python
extension module (.pyd on Windows, .so on Linux/macOS) using PyO3
and maturin.

FILES TO CREATE:

1. sc-neurocore/Cargo.toml (workspace root)
   Contents:
   ```toml
   [workspace]
   members = ["engine"]
   resolver = "2"
   ```

2. sc-neurocore/engine/Cargo.toml
   Contents:
   ```toml
   [package]
   name = "sc_neurocore_engine"
   version = "3.0.0-alpha.1"
   edition = "2021"
   authors = ["Miroslav Sotek <fortisstudio@gmail.com>"]
   description = "High-performance Rust backend for SC-NeuroCore"

   [lib]
   name = "sc_neurocore_engine"
   crate-type = ["cdylib"]

   [dependencies]
   pyo3 = { version = "0.22", features = ["extension-module"] }
   numpy = "0.22"          # PyO3 numpy interop
   ndarray = "0.16"        # N-dimensional arrays
   rand = "0.8"            # PRNG
   rand_chacha = "0.3"     # Deterministic seeded RNG
   rayon = "1.10"          # Data parallelism

   [dev-dependencies]
   criterion = { version = "0.5", features = ["html_reports"] }

   [build-dependencies]
   pyo3-build-config = "0.22"

   [[bench]]
   name = "bitstream_bench"
   harness = false

   [profile.release]
   opt-level = 3
   lto = "fat"
   codegen-units = 1
   target-cpu = "native"
   ```

3. sc-neurocore/engine/src/lib.rs
   Contents: A minimal PyO3 module with one test function:
   ```rust
   use pyo3::prelude::*;

   /// SC-NeuroCore v3.0 — High-Performance Rust Engine
   #[pymodule]
   fn sc_neurocore_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add("__version__", "3.0.0-alpha.1")?;
       m.add_function(wrap_pyfunction!(simd_tier, m)?)?;
       Ok(())
   }

   /// Returns the highest SIMD tier available on this CPU.
   #[pyfunction]
   fn simd_tier() -> &'static str {
       #[cfg(target_arch = "x86_64")]
       {
           if is_x86_feature_detected!("avx512vpopcntdq") {
               return "avx512-vpopcntdq";
           }
           if is_x86_feature_detected!("avx512f") {
               return "avx512f";
           }
           if is_x86_feature_detected!("avx2") {
               return "avx2";
           }
           if is_x86_feature_detected!("popcnt") {
               return "popcnt";
           }
       }
       #[cfg(target_arch = "aarch64")]
       {
           return "neon";
       }
       "portable"
   }
   ```

4. sc-neurocore/engine/rust-toolchain.toml
   Contents:
   ```toml
   [toolchain]
   channel = "stable"
   components = ["rustfmt", "clippy"]
   ```

5. sc-neurocore/bridge/sc_neurocore_engine/__init__.py
   Contents:
   ```python
   """SC-NeuroCore v3.0 — High-Performance Engine (Rust Backend)."""
   try:
       from sc_neurocore_engine.sc_neurocore_engine import (
           __version__,
           simd_tier,
       )
   except ImportError:
       raise ImportError(
           "sc_neurocore_engine native module not found. "
           "Build with: cd engine && maturin develop --release"
       )
   ```

6. sc-neurocore/bridge/pyproject.toml
   Contents:
   ```toml
   [build-system]
   requires = ["maturin>=1.5"]
   build-backend = "maturin"

   [project]
   name = "sc_neurocore_engine"
   version = "3.0.0a1"
   requires-python = ">=3.9"

   [tool.maturin]
   manifest-path = "../engine/Cargo.toml"
   python-source = "."
   module-name = "sc_neurocore_engine.sc_neurocore_engine"
   ```

CONSTRAINTS:
- Do NOT modify any file under src/sc_neurocore/.
- Do NOT modify pyproject.toml in the repo root.
- The Rust toolchain must be stable (no nightly features).
- Target x86_64 and aarch64.

VERIFICATION:
After creating all files, run:
```bash
cd engine
maturin develop --release
python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__); print(sc_neurocore_engine.simd_tier())"
```
Expected output:
```
3.0.0-alpha.1
avx2      (or avx512f, or neon, depending on CPU)
```

Also verify legacy is untouched:
```bash
python -c "import sc_neurocore; print(sc_neurocore.__version__)"
```
Expected: 2.1.0 (the existing version string)

═══════════════════════════════════════════════════════════════
```

---

### PACKET B: THE BITSTREAM KERNEL

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET B: BitStream Kernel
═══════════════════════════════════════════════════════════════

CONTEXT:
You are implementing the core BitStream computation kernel for
SC-NeuroCore v3.0 in Rust. This is THE performance-critical path.

Repository: sc-neurocore/engine/src/
Depends on: Packet A (scaffolding) must be complete.
Reference: The Python "ground truth" is in:
  - src/sc_neurocore/accel/vector_ops.py (pack, unpack, and, popcount)
  - src/sc_neurocore/neurons/fixed_point_lif.py (LFSR, encoder, LIF)
  - src/sc_neurocore/layers/vectorized_layer.py (dense layer forward)

GOAL:
Implement the following Rust modules that are BIT-TRUE identical
to the Python v2.2.0 reference.

═════════════════════════════════════════════════════════════
FILE 1: engine/src/bitstream.rs
═════════════════════════════════════════════════════════════

/// A packed bitstream stored as a Vec<u64>.
/// Each u64 holds 64 time steps.
/// Bit 0 of word 0 = time step 0.
/// Bit 63 of word 0 = time step 63.
/// Bit 0 of word 1 = time step 64.
pub struct BitStreamTensor {
    pub data: Vec<u64>,
    pub length: usize,    // Original bitstream length (before padding)
}

REQUIRED FUNCTIONS:

1. pub fn pack(bits: &[u8]) -> BitStreamTensor
   - Input: slice of {0, 1} u8 values
   - Output: packed u64 array
   - Algorithm: Identical to Python pack_bitstream():
     pad to multiple of 64, reshape to chunks of 64,
     multiply by powers of 2, sum.
   - MUST produce identical output to Python for all inputs.

2. pub fn unpack(tensor: &BitStreamTensor) -> Vec<u8>
   - Inverse of pack.
   - Extract bits via (word >> bit_idx) & 1.

3. pub fn bitwise_and(a: &BitStreamTensor, b: &BitStreamTensor) -> BitStreamTensor
   - Element-wise AND on the u64 arrays.
   - Panics if lengths differ.

4. pub fn popcount(tensor: &BitStreamTensor) -> u64
   - Count total set bits across all words.
   - PORTABLE implementation: SWAR algorithm (must match Python):
     x -= (x >> 1) & 0x5555555555555555;
     x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
     x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;
     x = (x * 0x0101010101010101) >> 56;
   - Sum across all words.

5. SIMD ACCELERATED VERSION (same results, faster):
   - Create engine/src/simd/mod.rs with runtime dispatch.
   - Create engine/src/simd/avx2.rs:
     #[target_feature(enable = "avx2")]
     pub unsafe fn popcount_avx2(data: &[u64]) -> u64
     Uses _mm256_and_si256, manual SWAR on 256-bit registers.
   - Create engine/src/simd/avx512.rs:
     #[target_feature(enable = "avx512vpopcntdq")]
     pub unsafe fn popcount_avx512(data: &[u64]) -> u64
     Uses _mm512_popcnt_epi64 (native hardware popcount).
   - All SIMD paths MUST return identical results to the portable path.

═════════════════════════════════════════════════════════════
FILE 2: engine/src/encoder.rs
═════════════════════════════════════════════════════════════

/// 16-bit Galois LFSR matching Python FixedPointLFSR.
/// Polynomial: x^16 + x^14 + x^13 + x^11 + 1
/// Taps (0-indexed): 15, 13, 12, 10
pub struct Lfsr16 {
    pub reg: u16,
    pub width: u32,
}

REQUIRED:
- new(seed: u16) -> Self  (panics if seed == 0)
- step(&mut self) -> u16  (advance one cycle, return new reg)

Algorithm (MUST match Python exactly):
  feedback = (reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10) & 1
  reg = ((reg << 1) & 0xFFFF) | feedback

/// Bitstream encoder: LFSR comparison.
pub struct BitstreamEncoder {
    lfsr: Lfsr16,
    data_width: u32,
}

REQUIRED:
- new(data_width: u32, seed: u16) -> Self
- step(&mut self, x_value: u16) -> u8
  Returns 1 if lfsr.reg < x_value, else 0.
  Calls lfsr.step() before comparison.

CRITICAL: The LFSR sequence must be identical to Python's
FixedPointLFSR for ALL 65,535 steps of the cycle.

═════════════════════════════════════════════════════════════
FILE 3: engine/src/neuron.rs
═════════════════════════════════════════════════════════════

/// Fixed-point LIF neuron matching Python FixedPointLIFNeuron.
/// Q8.8 format: 16-bit signed, 8 fractional bits.
pub struct FixedPointLif {
    pub v: i16,
    pub refractory_counter: i32,
    pub data_width: u32,     // 16
    pub fraction: u32,       // 8
    pub v_rest: i16,         // 0
    pub v_reset: i16,        // 0
    pub v_threshold: i16,    // 256 (= 1.0 in Q8.8)
    pub refractory_period: i32, // 2
}

REQUIRED:
- new(defaults matching Python) -> Self
- step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16)
  Returns (spike: 0 or 1, v_out: i16)

CRITICAL: The bit-masking logic must match Python exactly:
  fn mask(value: i32, width: u32) -> i16 {
      let m = (1i32 << width) - 1;
      let v = value & m;
      if v >= (1i32 << (width - 1)) { (v - (1i32 << width)) as i16 }
      else { v as i16 }
  }

  // Leak term:
  diff = mask((v_rest as i32 - v as i32), 2 * data_width)
  leak_mul = diff * leak_k as i32
  dv_leak = mask(leak_mul >> fraction, data_width)

  // Input term:
  in_mul = i_t as i32 * gain_k as i32
  dv_in = mask(in_mul >> fraction, data_width)

  // Next state:
  v_next = mask(v as i32 + dv_leak as i32 + dv_in as i32 + noise_in as i32, data_width)

  // Spike detection:
  if v_next >= v_threshold { spike=1, v=v_reset, refractory=period }
  else { spike=0, v=v_next }

  // Refractory override:
  if refractory_counter > 0 { counter--, v=v_rest, spike=0 }

═════════════════════════════════════════════════════════════
FILE 4: engine/src/layer.rs
═════════════════════════════════════════════════════════════

/// High-performance dense SC layer.
/// Matches VectorizedSCLayer.forward() from Python.
pub struct DenseLayer {
    pub n_inputs: usize,
    pub n_neurons: usize,
    pub length: usize,
    pub weights: Vec<Vec<f64>>,       // [n_neurons][n_inputs] probabilities
    pub packed_weights: Vec<Vec<Vec<u64>>>,  // [n_neurons][n_inputs][n_words]
}

REQUIRED:
- new(n_inputs, n_neurons, length, seed) -> Self
  Initialize weights uniformly in [0,1] using ChaCha8Rng seeded.
  Generate weight bitstreams and pack them.

- forward(&self, input_values: &[f64], seed: u64) -> Vec<f64>
  1. Generate input bitstreams: Bernoulli(p_i) for each input
  2. Pack input bitstreams to uint64
  3. For each neuron: AND(packed_weight, packed_input), popcount
  4. Return counts / length as firing rates

  The seed parameter controls the input Bernoulli RNG.
  Given the same seed + same weights, output MUST match Python.

PARALLELISM:
- Use rayon::par_iter() over neurons for the AND+popcount loop.
- Do NOT parallelize the encoding step (must be deterministic).

CONSTRAINTS:
- No unsafe blocks except in SIMD dispatch functions.
- All SIMD functions must be wrapped in #[target_feature] and
  called via runtime dispatch.
- The portable (non-SIMD) path must always be available.

VERIFICATION (Python):
```python
import numpy as np
# Test pack equivalence
bits = np.array([1,0,1,1,0,0,1,0] * 8, dtype=np.uint8)  # 64 bits
from sc_neurocore.accel import pack_bitstream
v2_packed = pack_bitstream(bits)
# Compare with: sc_neurocore_engine.pack_bitstream(bits)
# Must be identical.

# Test LFSR equivalence
from sc_neurocore.neurons import FixedPointLFSR
lfsr = FixedPointLFSR(width=16, seed=0xACE1)
v2_sequence = [lfsr.step() for _ in range(65535)]
# Compare with: engine Lfsr16 sequence
# Must be identical for all 65535 values.

# Test LIF equivalence
from sc_neurocore.neurons import FixedPointLIFNeuron
lif = FixedPointLIFNeuron()
v2_results = []
for t in range(100):
    spike, v = lif.step(leak_k=20, gain_k=256, I_t=128, noise_in=0)
    v2_results.append((spike, v))
# Compare with: engine FixedPointLif step-by-step
# Must be identical for all 100 steps.
```

═══════════════════════════════════════════════════════════════
```

---

### PACKET C: THE PYTHON BRIDGE

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET C: Python Bridge
═══════════════════════════════════════════════════════════════

CONTEXT:
You are creating Python wrapper classes that expose the Rust
engine (Packet B) with an API IDENTICAL to the v2.2.0 Python
classes. The user must be able to swap v2 for v3 by changing
one import line.

Repository: sc-neurocore/bridge/sc_neurocore_engine/
Depends on: Packets A + B must be complete.

GOAL:
Create Python wrappers that delegate to the Rust engine while
presenting the EXACT same interface as v2.2.0.

═════════════════════════════════════════════════════════════
FILE 1: bridge/sc_neurocore_engine/__init__.py
═════════════════════════════════════════════════════════════

"""SC-NeuroCore Engine v3.0 — Drop-in replacement for v2 hot paths."""

from sc_neurocore_engine.sc_neurocore_engine import (
    __version__,
    simd_tier,
    # Bitstream ops
    pack_bitstream,
    unpack_bitstream,
    popcount,
    # LFSR
    Lfsr16,
    BitstreamEncoder,
    # Neuron
    FixedPointLif,
)

from .layers import VectorizedSCLayer
from .neurons import FixedPointLIFNeuron

__all__ = [
    "__version__",
    "simd_tier",
    "pack_bitstream",
    "unpack_bitstream",
    "popcount",
    "Lfsr16",
    "BitstreamEncoder",
    "FixedPointLif",
    "VectorizedSCLayer",
    "FixedPointLIFNeuron",
]

═════════════════════════════════════════════════════════════
FILE 2: bridge/sc_neurocore_engine/layers.py
═════════════════════════════════════════════════════════════

"""Drop-in replacement for sc_neurocore.layers.VectorizedSCLayer."""

import numpy as np
from dataclasses import dataclass
from sc_neurocore_engine.sc_neurocore_engine import DenseLayer as _RustDenseLayer

@dataclass
class VectorizedSCLayer:
    """
    High-Performance SC Layer using Rust SIMD backend.

    API-compatible with sc_neurocore.layers.VectorizedSCLayer.
    Delegates all computation to the Rust engine.
    """
    n_inputs: int
    n_neurons: int
    length: int = 1024
    use_gpu: bool = False   # Ignored (Rust handles acceleration)

    def __post_init__(self):
        self._engine = _RustDenseLayer(
            self.n_inputs, self.n_neurons, self.length
        )
        self.weights = np.array(self._engine.get_weights())
        self.packed_weights = None  # Managed by Rust

    def _refresh_packed_weights(self):
        """Regenerate packed weight bitstreams (delegates to Rust)."""
        self._engine.set_weights(self.weights.tolist())
        self._engine.refresh_packed_weights()

    def forward(self, input_values) -> np.ndarray:
        """Compute output firing rates. Signature matches v2."""
        in_probs = np.asarray(input_values, dtype=np.float64)
        if in_probs.ndim != 1 or in_probs.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected 1-D input of length {self.n_inputs}, "
                f"got shape {in_probs.shape}"
            )
        result = self._engine.forward(in_probs.tolist())
        return np.array(result, dtype=np.float64)

═════════════════════════════════════════════════════════════
FILE 3: bridge/sc_neurocore_engine/neurons.py
═════════════════════════════════════════════════════════════

"""Drop-in replacement for sc_neurocore.neurons.FixedPointLIFNeuron."""

from sc_neurocore_engine.sc_neurocore_engine import FixedPointLif as _RustLif

class FixedPointLIFNeuron:
    """
    Fixed-point LIF neuron using Rust backend.
    API-compatible with sc_neurocore.neurons.FixedPointLIFNeuron.
    """
    def __init__(self, data_width=16, fraction=8,
                 v_rest=0, v_reset=0, v_threshold=256,
                 refractory_period=2):
        self._engine = _RustLif(
            data_width, fraction,
            v_rest, v_reset, v_threshold,
            refractory_period
        )

    def step(self, leak_k, gain_k, I_t, noise_in=0):
        """Returns (spike: int, v_out: int). Matches v2 exactly."""
        return self._engine.step(leak_k, gain_k, I_t, noise_in)

    def reset(self):
        self._engine.reset()

    def reset_state(self):
        self.reset()

    def get_state(self):
        return self._engine.get_state()

═════════════════════════════════════════════════════════════
FILE 4: tests/equivalence/conftest.py
═════════════════════════════════════════════════════════════

"""Shared fixtures for v2-vs-v3 equivalence testing."""

import pytest
import numpy as np

@pytest.fixture
def deterministic_rng():
    """Return a seeded NumPy RNG for reproducible tests."""
    return np.random.RandomState(42)

@pytest.fixture
def sample_bitstream():
    """A known 1024-bit bitstream for testing."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, 1024).astype(np.uint8)

═════════════════════════════════════════════════════════════
FILE 5: tests/equivalence/test_bitstream_equiv.py
═════════════════════════════════════════════════════════════

"""Equivalence: v2 pack/unpack/and/popcount vs v3."""

import numpy as np
import pytest

from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    unpack_bitstream as v2_unpack,
    vec_and as v2_and,
    vec_popcount as v2_popcount,
)
import sc_neurocore_engine as v3


class TestPackEquivalence:
    @pytest.mark.parametrize("length", [64, 128, 256, 1024, 1025, 4096])
    def test_pack_1d(self, length):
        bits = np.random.RandomState(42).randint(0, 2, length).astype(np.uint8)
        v2 = v2_pack(bits)
        v3_result = v3.pack_bitstream(bits)
        np.testing.assert_array_equal(v2, v3_result)

    def test_popcount(self, sample_bitstream):
        packed = v2_pack(sample_bitstream)
        v2_count = v2_popcount(packed)
        v3_count = v3.popcount(packed)
        assert v2_count == v3_count

═════════════════════════════════════════════════════════════
FILE 6: tests/equivalence/test_neuron_equiv.py
═════════════════════════════════════════════════════════════

"""Equivalence: v2 FixedPointLIFNeuron vs v3 engine."""

from sc_neurocore.neurons import FixedPointLIFNeuron as V2Lif
from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif


class TestLIFEquivalence:
    def test_100_steps_constant_input(self):
        v2 = V2Lif()
        v3 = V3Lif()
        for t in range(100):
            v2_spike, v2_v = v2.step(leak_k=20, gain_k=256, I_t=128, noise_in=0)
            v3_spike, v3_v = v3.step(leak_k=20, gain_k=256, I_t=128, noise_in=0)
            assert v2_spike == v3_spike, f"Spike mismatch at step {t}"
            assert v2_v == v3_v, f"Voltage mismatch at step {t}: v2={v2_v}, v3={v3_v}"

    def test_refractory_period(self):
        v2 = V2Lif(refractory_period=5)
        v3 = V3Lif(refractory_period=5)
        for t in range(200):
            v2_spike, v2_v = v2.step(20, 256, 200, 0)
            v3_spike, v3_v = v3.step(20, 256, 200, 0)
            assert v2_spike == v3_spike
            assert v2_v == v3_v

═════════════════════════════════════════════════════════════
FILE 7: tests/equivalence/test_encoder_equiv.py
═════════════════════════════════════════════════════════════

"""Equivalence: v2 FixedPointLFSR vs v3 Lfsr16."""

from sc_neurocore.neurons import FixedPointLFSR as V2Lfsr
from sc_neurocore_engine import Lfsr16 as V3Lfsr


class TestLFSREquivalence:
    def test_full_cycle(self):
        """LFSR must produce identical sequence for all 65535 steps."""
        v2 = V2Lfsr(width=16, seed=0xACE1)
        v3 = V3Lfsr(seed=0xACE1)
        for i in range(65535):
            v2_val = v2.step()
            v3_val = v3.step()
            assert v2_val == v3_val, (
                f"LFSR divergence at step {i}: v2={v2_val:#06x}, v3={v3_val:#06x}"
            )

    def test_multiple_seeds(self):
        """Test decorrelation seeds used in dense layers."""
        for seed in [0xACE1, 0xBEEF, 0xACE1 + 7, 0xBEEF + 13]:
            v2 = V2Lfsr(width=16, seed=seed)
            v3 = V3Lfsr(seed=seed)
            for i in range(1000):
                assert v2.step() == v3.step()

CONSTRAINTS:
- Wrapper classes must pass isinstance checks where possible
  (use @dataclass for layers, standard class for neurons).
- All v2 public methods must exist on v3 wrappers.
- No performance overhead in the bridge (pure delegation).
- Equivalence tests must run in < 30 seconds total.

VERIFICATION:
```bash
# Build engine
cd engine && maturin develop --release && cd ..

# Run equivalence suite
pytest tests/equivalence/ -v --tb=short

# All tests must pass. ANY failure means the engine
# is not bit-true to the Golden Reference.
```

═══════════════════════════════════════════════════════════════
```

---

## 7. PHASE 2 PREVIEW (Future Packets)

| Packet | Scope | Depends On |
|--------|-------|------------|
| **D** | Surrogate Gradient LIF (backward pass) | B |
| **E** | Stochastic Attention block (Transformer) | B, D |
| **F** | SCPN 7-Layer Stack (full consciousness model) | B, C |
| **G** | MLIR Dialect definition (`sc.dense_layer`, `sc.lif_neuron`) | B |
| **H** | CIRCT lowering passes (MLIR → SystemVerilog) | G |
| **I** | Benchmark suite (v2 vs v3 head-to-head) | B, C |
| **J** | CI/CD pipeline (GitHub Actions: build Rust, test equiv) | A, C |

---

## 8. EXPECTED PERFORMANCE TARGETS

| Operation | v2 (Python/NumPy) | v3 Target (Rust) | Speedup |
|-----------|-------------------|-------------------|---------|
| pack_bitstream (1M bits) | 10 Gbit/s | 60 Gbit/s (AVX-512) | 6x |
| popcount (1M words) | 5 Gbit/s | 100 Gbit/s (VPOPCNTDQ) | 20x |
| Dense forward (64in, 32out, L=1024) | 3.5 ms | 0.05 ms | 70x |
| LIF neuron step (100K steps) | 850 ms | 2 ms | 400x |
| Full pipeline (encode+AND+popcount) | 12 ms | 0.08 ms | 150x |

These targets assume single-threaded Rust with AVX-512. With `rayon` parallelism across neurons, the dense layer forward pass could achieve an additional 4-16x on multi-core CPUs.

---

Anulum CH&LI / Anulum Institute
Miroslav Sotek
ORCID: 0009-0009-3560-0851

(c) 1998-2026 Anulum Institute. All rights reserved.
