# SC-NeuroCore v3.0 — Phase 2 Codex Work Packets (D-0 through J)

**Status:** Phase 2 — Ready for Codex
**Author:** Principal Systems Architect
**Date:** February 9, 2026
**Depends On:** Phase 1 (Packets A-C) fully delivered and verified
**Constraint:** v2.2.0 Python code under `src/sc_neurocore/` is SACRED. Zero modifications.

---

## PHASE 1 REVIEW SUMMARY

Phase 1 delivered 24 source files (~809 lines), all tests passing:
- 13 Python equivalence tests (v2-vs-v3)
- 7 Rust integration/unit tests
- SIMD tier detected: `avx512-vpopcntdq`
- Legacy v2.1.0 untouched

**Issues found during code review (to be fixed in Packet D-0):**
1. `target-cpu = "native"` in `engine/Cargo.toml` is invalid (Cargo ignores it)
2. Missing DenseLayer equivalence test (critical gap)
3. NEON no-op mask (`vandq_u8(v, vdupq_n_u8(0xff))`)
4. Benchmark only tests portable popcount path
5. `crate-type` should include `"rlib"` for Rust tests (already present — confirmed)

---

## PACKET D-0: PHASE 1 FIXUPS

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET D-0: Phase 1 Fixups
═══════════════════════════════════════════════════════════════

CONTEXT:
Phase 1 was delivered and verified. Code review found 4 issues
that need fixing before Phase 2 work begins. These are small,
targeted fixes.

Repository: sc-neurocore/
Working directory: 03_CODE/sc-neurocore/
Do NOT modify anything under src/sc_neurocore/.

═════════════════════════════════════════════════════════════
FIX 1: Remove invalid target-cpu from Cargo.toml
═════════════════════════════════════════════════════════════

File: engine/Cargo.toml

`target-cpu = "native"` is NOT a valid Cargo profile key. Cargo
ignores it with a warning. To actually pass this flag to rustc,
create a new file instead.

DELETE this line from engine/Cargo.toml [profile.release]:
  target-cpu = "native"

CREATE: engine/.cargo/config.toml
  ```toml
  [target.'cfg(target_arch = "x86_64")']
  rustflags = ["-C", "target-cpu=native"]

  [target.'cfg(target_arch = "aarch64")']
  rustflags = ["-C", "target-cpu=native"]
  ```

═════════════════════════════════════════════════════════════
FIX 2: Add DenseLayer equivalence test (critical gap)
═════════════════════════════════════════════════════════════

There is NO test comparing v2 VectorizedSCLayer.forward() output
against v3 DenseLayer.forward() output. This is the primary hot
path and MUST have an equivalence test.

CREATE: tests/equivalence/test_layer_equiv.py
  ```python
  """Equivalence: v2 VectorizedSCLayer.forward vs v3 DenseLayer."""

  import numpy as np
  import pytest

  from sc_neurocore.layers import VectorizedSCLayer as V2Layer
  from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer


  class TestDenseLayerEquivalence:
      """
      Both v2 and v3 use Bernoulli sampling to encode probabilities
      as bitstreams. Because they use DIFFERENT RNG implementations
      (NumPy vs ChaCha8), they cannot produce bit-identical bitstreams.

      Instead, we verify STATISTICAL equivalence: given the same
      weights and inputs, the output firing rates must converge to
      the same expected values within tolerance.
      """

      @pytest.mark.parametrize("n_inputs,n_neurons", [
          (4, 2),
          (16, 8),
          (32, 16),
          (64, 32),
      ])
      def test_forward_statistical_equivalence(self, n_inputs, n_neurons):
          length = 4096  # Long bitstreams reduce variance

          v2 = V2Layer(n_inputs=n_inputs, n_neurons=n_neurons,
                       length=length, use_gpu=False)

          v3 = V3Layer(n_inputs=n_inputs, n_neurons=n_neurons,
                       length=length)

          # Use v2's random weights in v3
          v3._engine.set_weights(v2.weights.tolist())
          v3._engine.refresh_packed_weights()

          # Same input probabilities
          rng = np.random.RandomState(42)
          inputs = rng.uniform(0.1, 0.9, n_inputs)

          v2_out = v2.forward(inputs)
          v3_out = v3.forward(inputs)

          # Both should approximate the same expected value:
          # E[output_j] = sum_i(w_ji * p_i) / n_inputs (approximately)
          # With length=4096 the stochastic error is small.
          np.testing.assert_allclose(v2_out, v3_out, atol=0.05,
              err_msg="Dense layer outputs diverge beyond tolerance")

      def test_output_shape(self):
          v3 = V3Layer(n_inputs=8, n_neurons=4, length=1024)
          inputs = np.full(8, 0.5)
          out = v3.forward(inputs)
          assert out.shape == (4,)
          assert np.all(out >= 0.0) and np.all(out <= 1.0)

      def test_deterministic_with_same_seed(self):
          """Same v3 layer with same seed must produce identical output."""
          v3a = V3Layer(n_inputs=8, n_neurons=4, length=1024)
          v3b = V3Layer(n_inputs=8, n_neurons=4, length=1024)
          inputs = np.full(8, 0.5)
          np.testing.assert_array_equal(v3a.forward(inputs),
                                         v3b.forward(inputs))
  ```

═════════════════════════════════════════════════════════════
FIX 3: Remove NEON no-op mask
═════════════════════════════════════════════════════════════

File: engine/src/simd/neon.rs

CHANGE line:
  let masked = vandq_u8(v, vdupq_n_u8(0xff));
  let byte_counts = vcntq_u8(masked);
TO:
  let byte_counts = vcntq_u8(v);

The AND with 0xFF is an identity operation (no-op).

═════════════════════════════════════════════════════════════
FIX 4: Add SIMD dispatch to benchmark
═════════════════════════════════════════════════════════════

File: engine/benches/bitstream_bench.rs

ADD a new benchmark function that uses popcount_dispatch()
(the SIMD-accelerated path) alongside the existing portable
benchmark. This measures actual production performance.

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sc_neurocore_engine::bitstream::{pack, popcount_words_portable};
use sc_neurocore_engine::simd::popcount_dispatch;

fn bench_pack_and_popcount(c: &mut Criterion) {
    let bits: Vec<u8> = (0..(1024 * 1024))
        .map(|idx| if idx % 3 == 0 { 1 } else { 0 })
        .collect();

    c.bench_function("pack_bitstream_1m", |b| {
        b.iter(|| {
            let packed = pack(black_box(&bits));
            black_box(packed);
        })
    });

    let packed = pack(&bits);

    c.bench_function("popcount_portable_1m", |b| {
        b.iter(|| {
            let count = popcount_words_portable(black_box(&packed.data));
            black_box(count);
        })
    });

    c.bench_function("popcount_simd_dispatch_1m", |b| {
        b.iter(|| {
            let count = popcount_dispatch(black_box(&packed.data));
            black_box(count);
        })
    });
}

criterion_group!(benches, bench_pack_and_popcount);
criterion_main!(benches);
```

═════════════════════════════════════════════════════════════

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore

# Build engine
cd bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..

# Run full test suite (existing + new layer equiv test)
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/equivalence -v --tb=short

# Rust tests
cd engine && cargo test --tests && cd ..

# Benchmark (optional — visual confirmation)
cd engine && cargo bench --bench bitstream_bench && cd ..
```

Expected: All existing 13 tests pass + new layer equivalence tests
pass (4 parametrized + 2 = 6 new tests, total 19).

═══════════════════════════════════════════════════════════════
```

---

## PACKET D: SURROGATE GRADIENT LIF (Backward Pass)

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET D: Surrogate Gradient Engine
═══════════════════════════════════════════════════════════════

CONTEXT:
SC-NeuroCore v2.2.0 has no backpropagation support. The forward
pass (bitstream → AND → popcount) is discrete and non-differentiable.
SC-NeuroCore v3.0 introduces SURROGATE GRADIENTS — the standard
approach for training spiking neural networks (Neftci et al. 2019,
Zenke & Ganguli 2018).

Key principle:
- FORWARD PASS: Unchanged. Bit-true identical to v2.
- BACKWARD PASS: Replace the Heaviside step derivative with a
  smooth surrogate function. The forward pass uses the true
  threshold; the backward pass pretends the threshold was smooth.

Repository: sc-neurocore/engine/src/
Depends on: Packet B (bitstream/neuron/layer modules).

GOAL:
Implement surrogate gradient primitives in Rust and expose them
to Python. This is NEW functionality (not in v2) so there is
no equivalence test — only correctness tests.

═════════════════════════════════════════════════════════════
FILE 1: engine/src/grad/mod.rs
═════════════════════════════════════════════════════════════

```rust
pub mod surrogate;

pub use surrogate::{SurrogateType, SurrogateLif, DifferentiableDenseLayer};
```

═════════════════════════════════════════════════════════════
FILE 2: engine/src/grad/surrogate.rs
═════════════════════════════════════════════════════════════

REQUIRED TYPES:

```rust
/// Surrogate gradient function type.
#[derive(Clone, Debug)]
pub enum SurrogateType {
    /// Fast Sigmoid: d/dx = 1 / (1 + k|x|)^2
    /// Default k = 25.0 (Zenke & Ganguli 2018)
    FastSigmoid { k: f32 },

    /// SuperSpike: d/dx = 1 / (k * |x| + 1)^2
    /// Default k = 100.0 (Zenke & Ganguli 2018)
    SuperSpike { k: f32 },

    /// ArcTan: d/dx = 1 / (1 + (kx)^2)
    /// Default k = 10.0 (Fang et al. 2021)
    ArcTan { k: f32 },

    /// Straight-Through Estimator: d/dx = 1 when |x| < 0.5, else 0
    StraightThrough,
}
```

REQUIRED: `impl SurrogateType`

```rust
impl SurrogateType {
    /// Compute the surrogate gradient at a given membrane potential.
    /// x = (v_membrane - v_threshold), normalized.
    pub fn grad(&self, x: f32) -> f32 {
        match self {
            Self::FastSigmoid { k } => {
                1.0 / (1.0 + k * x.abs()).powi(2)
            }
            Self::SuperSpike { k } => {
                1.0 / (k * x.abs() + 1.0).powi(2)
            }
            Self::ArcTan { k } => {
                1.0 / (1.0 + (k * x).powi(2))
            }
            Self::StraightThrough => {
                if x.abs() < 0.5 { 1.0 } else { 0.0 }
            }
        }
    }
}
```

REQUIRED: `SurrogateLif`

```rust
/// LIF neuron with surrogate gradient support.
/// Forward pass is BIT-TRUE identical to FixedPointLif.
/// Backward pass uses a smooth surrogate.
pub struct SurrogateLif {
    /// The bit-true LIF (unchanged from Packet B).
    pub lif: crate::neuron::FixedPointLif,
    /// The surrogate function used for backward pass.
    pub surrogate: SurrogateType,
    /// Saved membrane traces for backward pass.
    /// Each entry is (v_before_threshold, spike_output).
    membrane_trace: Vec<(f32, f32)>,
}

impl SurrogateLif {
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: SurrogateType,
    ) -> Self;

    /// Forward pass — IDENTICAL to FixedPointLif.step().
    /// Saves (v_normalized, spike) to membrane_trace for backward.
    pub fn forward(
        &mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16,
    ) -> (i32, i16) {
        // Get pre-threshold membrane potential
        // (compute v_next using the same mask logic as neuron.rs)
        let (spike, v_out) = self.lif.step(leak_k, gain_k, i_t, noise_in);

        // Save trace for backward pass
        let v_norm = (v_out as f32 - self.lif.v_threshold as f32)
            / (1 << self.lif.fraction) as f32;
        self.membrane_trace.push((v_norm, spike as f32));

        (spike, v_out)
    }

    /// Backward pass — surrogate gradient.
    /// Given upstream gradient dL/d(spike), returns dL/d(v_membrane).
    /// Processes the LAST saved trace entry.
    pub fn backward(&mut self, grad_output: f32) -> f32 {
        let (v_norm, _spike) = self.membrane_trace.pop()
            .expect("backward() called without forward()");
        grad_output * self.surrogate.grad(v_norm)
    }

    /// Clear all saved traces (call between training batches).
    pub fn clear_trace(&mut self);

    /// Reset neuron state AND traces.
    pub fn reset(&mut self);

    /// Number of saved trace entries.
    pub fn trace_len(&self) -> usize;
}
```

REQUIRED: `DifferentiableDenseLayer`

```rust
/// Dense SC layer with surrogate gradient backward pass.
/// Forward: identical to DenseLayer (bit-true).
/// Backward: surrogate gradient through spike functions,
/// true gradient through weight encoding (Bernoulli → prob).
pub struct DifferentiableDenseLayer {
    pub layer: crate::layer::DenseLayer,
    pub surrogate: SurrogateType,
    /// Saved input probabilities for backward pass.
    input_cache: Vec<f64>,
    /// Saved output activations for backward pass.
    output_cache: Vec<f64>,
}

impl DifferentiableDenseLayer {
    pub fn new(
        n_inputs: usize, n_neurons: usize, length: usize,
        seed: u64, surrogate: SurrogateType,
    ) -> Self;

    /// Forward pass — identical to DenseLayer.forward().
    /// Caches inputs and outputs for backward.
    pub fn forward(&mut self, input_values: &[f64], seed: u64)
        -> Result<Vec<f64>, String>;

    /// Backward pass.
    /// Given dL/d(output) of shape [n_neurons], returns:
    /// - dL/d(input) of shape [n_inputs]
    /// - dL/d(weights) of shape [n_neurons][n_inputs]
    ///
    /// The gradient through the SC forward pass (Bernoulli→AND→popcount)
    /// is approximated as:
    ///   d(output_j)/d(w_ji) ≈ input_i  (expectation of AND gate)
    ///   d(output_j)/d(input_i) ≈ w_ji   (expectation of AND gate)
    ///
    /// This is the "expected value" gradient — correct in expectation
    /// even though the actual forward pass is stochastic.
    pub fn backward(&self, grad_output: &[f64])
        -> Result<(Vec<f64>, Vec<Vec<f64>>), String>;

    /// Apply weight gradient update with learning rate.
    /// w_ji -= lr * dL/dw_ji
    /// Clamps weights to [0, 1] (probability range).
    pub fn update_weights(&mut self, weight_grads: &[Vec<f64>], lr: f64);

    /// Clear caches.
    pub fn clear_cache(&mut self);
}
```

═════════════════════════════════════════════════════════════
PyO3 BINDINGS: Add to engine/src/lib.rs
═════════════════════════════════════════════════════════════

Add `pub mod grad;` to the module declarations.

Expose the following to Python:

```rust
// In the #[pymodule] function:
m.add_class::<PySurrogateLif>()?;
m.add_class::<PyDifferentiableDenseLayer>()?;
```

Create PyO3 wrapper classes that mirror the Rust API.
Use string-based surrogate type selection for Python ergonomics:

```python
# Python usage:
from sc_neurocore_engine import SurrogateLif, DifferentiableDenseLayer

lif = SurrogateLif(surrogate="fast_sigmoid", k=25.0)
spike, v = lif.forward(leak_k=20, gain_k=256, i_t=128)
grad_in = lif.backward(grad_output=1.0)

layer = DifferentiableDenseLayer(
    n_inputs=32, n_neurons=16, length=1024,
    surrogate="arctan", k=10.0
)
out = layer.forward([0.5] * 32)
grad_input, grad_weights = layer.backward([1.0] * 16)
layer.update_weights(grad_weights, lr=0.01)
```

═════════════════════════════════════════════════════════════
BRIDGE WRAPPERS: Add to bridge/
═════════════════════════════════════════════════════════════

CREATE: bridge/sc_neurocore_engine/grad.py
  Wrapper classes that expose SurrogateLif and DifferentiableDenseLayer
  with NumPy array I/O (convert lists ↔ np.ndarray at boundaries).

UPDATE: bridge/sc_neurocore_engine/__init__.py
  Add imports: SurrogateLif, DifferentiableDenseLayer

═════════════════════════════════════════════════════════════
TESTS
═════════════════════════════════════════════════════════════

CREATE: engine/tests/test_surrogate.rs

```rust
// Rust-side tests:

#[test]
fn fast_sigmoid_gradient_at_zero_is_one() {
    let sg = SurrogateType::FastSigmoid { k: 25.0 };
    assert!((sg.grad(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn fast_sigmoid_gradient_decays_away_from_zero() {
    let sg = SurrogateType::FastSigmoid { k: 25.0 };
    let g0 = sg.grad(0.0);
    let g1 = sg.grad(0.1);
    let g2 = sg.grad(1.0);
    assert!(g0 > g1);
    assert!(g1 > g2);
    assert!(g2 > 0.0);
}

#[test]
fn superspike_gradient_symmetric() {
    let sg = SurrogateType::SuperSpike { k: 100.0 };
    assert!((sg.grad(0.5) - sg.grad(-0.5)).abs() < 1e-6);
}

#[test]
fn arctan_gradient_is_lorentzian() {
    let sg = SurrogateType::ArcTan { k: 10.0 };
    // At x=0: grad = 1 / (1 + 0) = 1.0
    assert!((sg.grad(0.0) - 1.0).abs() < 1e-6);
    // At x=0.1: grad = 1 / (1 + 1.0) = 0.5
    assert!((sg.grad(0.1) - 0.5).abs() < 1e-6);
}

#[test]
fn straight_through_is_unit_box() {
    let sg = SurrogateType::StraightThrough;
    assert_eq!(sg.grad(0.0), 1.0);
    assert_eq!(sg.grad(0.3), 1.0);
    assert_eq!(sg.grad(0.5), 0.0);
    assert_eq!(sg.grad(-0.5), 0.0);
    assert_eq!(sg.grad(1.0), 0.0);
}

#[test]
fn surrogate_lif_forward_matches_plain_lif() {
    // Forward pass of SurrogateLif MUST equal FixedPointLif.
    let mut plain = FixedPointLif::new(16, 8, 0, 0, 256, 2);
    let mut surr = SurrogateLif::new(
        16, 8, 0, 0, 256, 2,
        SurrogateType::FastSigmoid { k: 25.0 },
    );

    for _ in 0..50 {
        let (s1, v1) = plain.step(20, 256, 128, 0);
        let (s2, v2) = surr.forward(20, 256, 128, 0);
        assert_eq!(s1, s2);
        assert_eq!(v1, v2);
    }
}

#[test]
fn backward_produces_nonzero_gradient() {
    let mut surr = SurrogateLif::new(
        16, 8, 0, 0, 256, 2,
        SurrogateType::FastSigmoid { k: 25.0 },
    );
    surr.forward(20, 256, 128, 0);
    let grad = surr.backward(1.0);
    assert!(grad.abs() > 0.0);
}

#[test]
fn differentiable_layer_backward_shapes() {
    let mut layer = DifferentiableDenseLayer::new(
        8, 4, 1024, 42,
        SurrogateType::FastSigmoid { k: 25.0 },
    );
    let out = layer.forward(&[0.5; 8], 42).unwrap();
    assert_eq!(out.len(), 4);

    let (grad_in, grad_w) = layer.backward(&[1.0; 4]).unwrap();
    assert_eq!(grad_in.len(), 8);
    assert_eq!(grad_w.len(), 4);
    assert_eq!(grad_w[0].len(), 8);
}

#[test]
fn weight_update_changes_weights() {
    let mut layer = DifferentiableDenseLayer::new(
        4, 2, 1024, 42,
        SurrogateType::FastSigmoid { k: 25.0 },
    );
    let w_before = layer.layer.get_weights();
    let _ = layer.forward(&[0.5; 4], 42).unwrap();
    let (_, grad_w) = layer.backward(&[1.0; 2]).unwrap();
    layer.update_weights(&grad_w, 0.1);
    let w_after = layer.layer.get_weights();
    assert_ne!(w_before, w_after);
}
```

CREATE: tests/test_surrogate_python.py

```python
"""Tests for surrogate gradient engine (Python bridge)."""

import numpy as np
from sc_neurocore_engine import SurrogateLif, DifferentiableDenseLayer


class TestSurrogateLif:
    def test_forward_backward_cycle(self):
        lif = SurrogateLif(surrogate="fast_sigmoid", k=25.0)
        spike, v = lif.forward(leak_k=20, gain_k=256, i_t=128)
        grad = lif.backward(1.0)
        assert isinstance(grad, float)
        assert grad != 0.0

    def test_clear_trace(self):
        lif = SurrogateLif(surrogate="arctan", k=10.0)
        for _ in range(10):
            lif.forward(20, 256, 128)
        assert lif.trace_len() == 10
        lif.clear_trace()
        assert lif.trace_len() == 0


class TestDifferentiableDenseLayer:
    def test_train_step(self):
        layer = DifferentiableDenseLayer(
            n_inputs=8, n_neurons=4, length=1024,
            surrogate="fast_sigmoid", k=25.0,
        )
        out1 = np.array(layer.forward([0.5] * 8))

        # Backward + weight update
        grad_in, grad_w = layer.backward([1.0] * 4)
        layer.update_weights(grad_w, lr=0.01)

        # Output should change after weight update
        out2 = np.array(layer.forward([0.5] * 8))
        assert not np.allclose(out1, out2)
```

CONSTRAINTS:
- Forward pass of SurrogateLif MUST be bit-identical to FixedPointLif.
  Use the SAME step() implementation internally. Do not re-implement
  the LIF logic. Just wrap it.
- Surrogate gradients are f32 (single precision is sufficient for
  gradient approximation — no need for f64).
- DifferentiableDenseLayer backward uses the "expected value" gradient
  approximation. This is NOT bit-true — it is a smooth approximation
  of the stochastic forward pass.
- Weight clamping to [0, 1] is mandatory after update_weights()
  since weights represent probabilities.

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo test --tests
cd ../bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/test_surrogate_python.py -v
```

═══════════════════════════════════════════════════════════════
```

---

## PACKET E: STOCHASTIC ATTENTION + GNN (Rust Accelerated)

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET E: Attention + GNN Kernels
═══════════════════════════════════════════════════════════════

CONTEXT:
SC-NeuroCore v2.2.0 has two key layer types that are compute-
intensive but currently pure NumPy:

1. StochasticAttention (src/sc_neurocore/layers/attention.py)
   — Approximates Softmax(Q·K^T)·V using SC dot products.

2. StochasticGraphLayer (src/sc_neurocore/graphs/gnn.py)
   — GCN-style message passing: A·X·W with degree normalization.

GOAL:
Implement Rust-accelerated versions of both, exposed via PyO3.
The v2 Python implementations serve as the specification:

v2 StochasticAttention.forward(Q, K, V):
  scores = Q @ K.T           # (N, M)
  row_sums = scores.sum(1)   # (N,)  — normalize per query
  row_sums[row_sums == 0] = 1.0
  attn_weights = scores / row_sums[:, None]
  output = attn_weights @ V   # (N, Dim_V)

v2 StochasticGraphLayer.forward(node_features):
  agg = adj @ node_features   # message passing
  degrees = adj.sum(1)
  degrees[degrees == 0] = 1.0
  agg /= degrees[:, None]    # degree normalization
  output = tanh(agg @ weights) # linear transform + activation

Both are matrix-multiply-heavy. Rust + rayon parallelism should
provide significant speedup, especially for large graphs.

Repository: sc-neurocore/engine/src/
Depends on: Packet B (bitstream primitives).

═════════════════════════════════════════════════════════════
FILE 1: engine/src/attention.rs
═════════════════════════════════════════════════════════════

```rust
/// Stochastic Computing Attention Block.
///
/// Approximates: Output = Softmax(Q · K^T) · V
/// In SC interpretation: Q, K, V are probability matrices.
/// Dot products are computed via bitstream AND+popcount
/// when operating in "full SC mode", or via f64 matmul
/// in "rate mode" (matching v2 exactly).
pub struct StochasticAttention {
    pub dim_k: usize,
}

impl StochasticAttention {
    pub fn new(dim_k: usize) -> Self;

    /// Rate-mode forward pass.
    /// Matches v2 StochasticAttention.forward() exactly.
    ///
    /// q: (N, dim_k) row-major
    /// k: (M, dim_k) row-major
    /// v: (M, dim_v) row-major
    /// Returns: (N, dim_v) row-major
    pub fn forward(
        &self,
        q: &[f64], q_rows: usize, q_cols: usize,
        k: &[f64], k_rows: usize, k_cols: usize,
        v: &[f64], v_rows: usize, v_cols: usize,
    ) -> Result<Vec<f64>, String>;

    /// SC-mode forward pass (new v3 capability).
    /// Encodes Q, K, V as bitstreams, uses AND+popcount for
    /// all matrix multiplies. Returns rate-coded result.
    /// This is approximate but hardware-friendly.
    pub fn forward_sc(
        &self,
        q: &[f64], q_rows: usize, q_cols: usize,
        k: &[f64], k_rows: usize, k_cols: usize,
        v: &[f64], v_rows: usize, v_cols: usize,
        length: usize, seed: u64,
    ) -> Result<Vec<f64>, String>;
}
```

ALGORITHM for `forward()` (rate-mode, MUST match v2):
```
1. scores[i][j] = dot(q[i], k[j])  for i=0..N, j=0..M
2. row_sums[i] = sum(scores[i])
3. if row_sums[i] == 0.0: row_sums[i] = 1.0
4. attn_weights[i][j] = scores[i][j] / row_sums[i]
5. output[i][d] = sum_j(attn_weights[i][j] * v[j][d])
```

ALGORITHM for `forward_sc()` (SC bitstream mode):
```
1. Encode each Q[i,k], K[j,k], V[j,d] as Bernoulli bitstreams
   of the given length.
2. Pack to uint64.
3. scores_bits[i][j] = AND(q_bits[i], k_bits[j]) -> popcount
   (this computes the stochastic inner product)
4. Normalize scores per row -> attn_weights (rate-coded)
5. Re-encode attn_weights as bitstreams
6. output_bits[i][d] = AND(attn_bits[i][j], v_bits[j][d])
   for all j -> popcount -> sum
7. Return output / length
```

Use rayon to parallelize the outer loops (over queries and output dims).

═════════════════════════════════════════════════════════════
FILE 2: engine/src/graph.rs
═════════════════════════════════════════════════════════════

```rust
/// Stochastic Graph Convolution Layer.
///
/// Implements: output = tanh(D^{-1} · A · X · W)
/// where A is adjacency, D is degree matrix, X is node features,
/// W is learnable weight matrix.
pub struct StochasticGraphLayer {
    pub n_nodes: usize,
    pub n_features: usize,
    /// Adjacency matrix stored as flat row-major: (n_nodes * n_nodes)
    pub adj: Vec<f64>,
    /// Weight matrix stored as flat row-major: (n_features * n_features)
    pub weights: Vec<f64>,
    /// Precomputed degree vector: (n_nodes,)
    pub degrees: Vec<f64>,
}

impl StochasticGraphLayer {
    /// adj_flat: row-major (n_nodes * n_nodes) adjacency matrix.
    /// n_features: feature dimension per node.
    /// seed: RNG seed for weight initialization.
    pub fn new(
        adj_flat: Vec<f64>, n_nodes: usize, n_features: usize, seed: u64,
    ) -> Self;

    /// Rate-mode forward pass.
    /// Matches v2 StochasticGraphLayer.forward() exactly.
    ///
    /// node_features: flat row-major (n_nodes * n_features)
    /// Returns: flat row-major (n_nodes * n_features)
    pub fn forward(
        &self, node_features: &[f64],
    ) -> Result<Vec<f64>, String>;

    pub fn get_weights(&self) -> Vec<f64>;
    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<(), String>;
}
```

ALGORITHM for `forward()` (MUST match v2):
```
1. agg[i][f] = sum_j(adj[i][j] * node_features[j][f])  // message passing
2. agg[i][f] /= degrees[i]  (where degrees[i] = sum_j adj[i][j])
   if degrees[i] == 0: skip normalization (leave agg[i] as 0)
3. output[i][f] = tanh(sum_g(agg[i][g] * weights[g][f]))  // transform
```

Use rayon to parallelize over nodes (step 1 and 3).

═════════════════════════════════════════════════════════════
PyO3 BINDINGS: Add to engine/src/lib.rs
═════════════════════════════════════════════════════════════

```rust
pub mod attention;
pub mod graph;

// In #[pymodule]:
m.add_class::<PyStochasticAttention>()?;
m.add_class::<PyStochasticGraphLayer>()?;
```

Python-facing API:

```python
from sc_neurocore_engine import StochasticAttention, StochasticGraphLayer
import numpy as np

# Attention
attn = StochasticAttention(dim_k=16)
Q = np.random.uniform(0, 1, (10, 16))
K = np.random.uniform(0, 1, (20, 16))
V = np.random.uniform(0, 1, (20, 32))
output = attn.forward(Q, K, V)  # Returns np.ndarray (10, 32)

# Graph
adj = np.eye(5) + np.roll(np.eye(5), 1, axis=0)  # ring graph
gnn = StochasticGraphLayer(adj, n_features=8)
X = np.random.uniform(0, 1, (5, 8))
output = gnn.forward(X)  # Returns np.ndarray (5, 8)
```

═════════════════════════════════════════════════════════════
BRIDGE WRAPPERS
═════════════════════════════════════════════════════════════

CREATE: bridge/sc_neurocore_engine/attention.py

```python
"""Drop-in replacement for sc_neurocore.layers.StochasticAttention."""

import numpy as np
from sc_neurocore_engine.sc_neurocore_engine import (
    StochasticAttention as _RustAttention,
)


class StochasticAttention:
    """API-compatible with sc_neurocore.layers.StochasticAttention."""

    def __init__(self, dim_k: int):
        self.dim_k = dim_k
        self._engine = _RustAttention(dim_k)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray
                ) -> np.ndarray:
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim == 1: Q = Q[None, :]
        if K.ndim == 1: K = K[None, :]
        if V.ndim == 1: V = V[None, :]
        result = self._engine.forward(Q, K, V)
        return np.asarray(result, dtype=np.float64)
```

CREATE: bridge/sc_neurocore_engine/graphs.py

```python
"""Drop-in replacement for sc_neurocore.graphs.StochasticGraphLayer."""

import numpy as np
from sc_neurocore_engine.sc_neurocore_engine import (
    StochasticGraphLayer as _RustGraphLayer,
)


class StochasticGraphLayer:
    """API-compatible with sc_neurocore.graphs.StochasticGraphLayer."""

    def __init__(self, adj_matrix: np.ndarray, n_features: int,
                 seed: int = 42):
        adj = np.asarray(adj_matrix, dtype=np.float64)
        self.n_nodes = adj.shape[0]
        self.n_features = n_features
        self._engine = _RustGraphLayer(adj, n_features, seed)
        self.weights = np.array(
            self._engine.get_weights()
        ).reshape(n_features, n_features)

    def forward(self, node_features: np.ndarray) -> np.ndarray:
        X = np.asarray(node_features, dtype=np.float64)
        result = self._engine.forward(X)
        return np.asarray(result, dtype=np.float64).reshape(
            self.n_nodes, self.n_features
        )
```

UPDATE: bridge/sc_neurocore_engine/__init__.py
  Add: from .attention import StochasticAttention
  Add: from .graphs import StochasticGraphLayer

═════════════════════════════════════════════════════════════
EQUIVALENCE TESTS
═════════════════════════════════════════════════════════════

CREATE: tests/equivalence/test_attention_equiv.py

```python
"""Equivalence: v2 StochasticAttention vs v3."""

import numpy as np
import pytest

from sc_neurocore.layers.attention import StochasticAttention as V2Attn
from sc_neurocore_engine.attention import StochasticAttention as V3Attn


class TestAttentionEquivalence:
    @pytest.mark.parametrize("n,m,dk,dv", [
        (1, 1, 4, 4),
        (5, 10, 8, 16),
        (10, 10, 16, 32),
        (20, 50, 32, 64),
    ])
    def test_forward_matches_v2(self, n, m, dk, dv):
        rng = np.random.RandomState(42)
        Q = rng.uniform(0, 1, (n, dk))
        K = rng.uniform(0, 1, (m, dk))
        V = rng.uniform(0, 1, (m, dv))

        v2 = V2Attn(dim_k=dk)
        v3 = V3Attn(dim_k=dk)

        v2_out = v2.forward(Q, K, V)
        v3_out = v3.forward(Q, K, V)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12,
            err_msg="Attention output mismatch (rate mode)")

    def test_1d_input_expansion(self):
        """v2 handles 1-D inputs by expanding to 2-D. v3 must too."""
        rng = np.random.RandomState(42)
        q = rng.uniform(0, 1, 8)   # 1-D
        k = rng.uniform(0, 1, 8)   # 1-D
        v = rng.uniform(0, 1, 4)   # 1-D

        v2 = V2Attn(dim_k=8)
        v3 = V3Attn(dim_k=8)

        v2_out = v2.forward(q, k, v)
        v3_out = v3.forward(q, k, v)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12)

    def test_zero_scores_handling(self):
        """When Q and K are zero, row_sums = 0. Must not crash."""
        Q = np.zeros((3, 4))
        K = np.zeros((5, 4))
        V = np.random.RandomState(42).uniform(0, 1, (5, 8))

        v2 = V2Attn(dim_k=4)
        v3 = V3Attn(dim_k=4)

        v2_out = v2.forward(Q, K, V)
        v3_out = v3.forward(Q, K, V)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12)
```

CREATE: tests/equivalence/test_gnn_equiv.py

```python
"""Equivalence: v2 StochasticGraphLayer vs v3."""

import numpy as np
import pytest

from sc_neurocore.graphs.gnn import StochasticGraphLayer as V2GNN
from sc_neurocore_engine.graphs import StochasticGraphLayer as V3GNN


class TestGraphLayerEquivalence:
    @pytest.mark.parametrize("n_nodes,n_features", [
        (5, 4),
        (10, 8),
        (20, 16),
    ])
    def test_forward_matches_v2(self, n_nodes, n_features):
        rng = np.random.RandomState(42)
        # Random adjacency (symmetric, no self-loops)
        adj = rng.randint(0, 2, (n_nodes, n_nodes)).astype(np.float64)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0.0)

        X = rng.uniform(0, 1, (n_nodes, n_features))

        v2 = V2GNN(adj, n_features)
        v3 = V3GNN(adj, n_features)

        # Sync weights: v2 → v3
        v3._engine.set_weights(v2.weights.flatten().tolist())

        v2_out = v2.forward(X)
        v3_out = v3.forward(X)

        np.testing.assert_allclose(v2_out, v3_out, atol=1e-12,
            err_msg="GNN output mismatch")

    def test_isolated_node(self):
        """Node with no edges should produce tanh(0) = 0."""
        adj = np.zeros((3, 3))
        adj[0, 1] = adj[1, 0] = 1.0  # Only nodes 0-1 connected
        # Node 2 is isolated

        X = np.ones((3, 4))
        v3 = V3GNN(adj, 4)

        out = v3.forward(X)
        # Node 2 gets zero aggregation → tanh(0·W) = 0
        np.testing.assert_allclose(out[2], 0.0, atol=1e-12)
```

CONSTRAINTS:
- Rate-mode forward() MUST produce results identical to v2 within
  1e-12 (float64 precision). These are deterministic f64 matmuls.
- SC-mode forward_sc() is a NEW v3 capability, so no v2 equivalence
  required — only internal consistency tests.
- Use flat f64 slices for Rust API. The Python bridge handles
  reshape to/from np.ndarray.
- Parallelize with rayon where beneficial (queries in attention,
  nodes in GNN).

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo test --tests
cd ../bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/equivalence/test_attention_equiv.py tests/equivalence/test_gnn_equiv.py -v --tb=short
```

Expected: All tests pass with atol=1e-12.

═══════════════════════════════════════════════════════════════
```

---

## PACKET F: SCPN 7-LAYER STACK BRIDGE

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET F: SCPN Stack Orchestration
═══════════════════════════════════════════════════════════════

CONTEXT:
The SCPN (Self-Consistent Phenomenological Network) has 7 layers:
  L1: Quantum Biological (microtubules, NV centers)
  L2: Neurochemical (receptors, neurotransmitters)
  L3: Genomic-Epigenomic (CISS, bioelectric)
  L4: Cellular-Tissue (Kuramoto oscillators, gap junctions)
  L5: Organismal-Psychoemotional (HRV, emotions)
  L6: Ecological-Planetary (Schumann resonances, geomagnetic)
  L7: Geometric-Symbolic (sacred geometry, E8, acupuncture)

The v2 Python implementation is in:
  src/sc_neurocore/scpn/layers/l1_quantum.py through l7_symbolic.py
  src/sc_neurocore/scpn/layers/__init__.py (create_full_stack,
    run_integrated_step, get_global_metrics)

The layers are biophysical simulations — they are NOT the hot-path
for inference (that's VectorizedSCLayer). Porting them entirely
to Rust would be a massive effort with little performance benefit
since they're already vectorized NumPy.

GOAL:
Instead of rewriting all 7 layers in Rust, create a RUST
ORCHESTRATOR that:
1. Calls the existing Python layers via PyO3 callbacks.
2. Accelerates the HOT INNER LOOPS (Kuramoto coupling in L4,
   receptor binding in L2) in Rust.
3. Provides a unified run_integrated_step() in Rust that manages
   the data flow between layers.

This is a "selective acceleration" strategy.

═════════════════════════════════════════════════════════════
FILE 1: engine/src/scpn/mod.rs
═════════════════════════════════════════════════════════════

```rust
pub mod kuramoto;
pub mod metrics;

pub use kuramoto::KuramotoSolver;
pub use metrics::SCPNMetrics;
```

═════════════════════════════════════════════════════════════
FILE 2: engine/src/scpn/kuramoto.rs
═════════════════════════════════════════════════════════════

This is the hot loop in L4_CellularLayer and the UPDE solver.
Porting it to Rust with SIMD gives the biggest bang for the buck.

```rust
use rayon::prelude::*;

/// High-performance Kuramoto oscillator solver.
///
/// Implements: dθ_n/dt = ω_n + Σ_m K_nm sin(θ_m - θ_n) + noise
///
/// This is the core equation for L4 (cellular synchronization)
/// and the UPDE (Unified Phase Dynamics Equation).
pub struct KuramotoSolver {
    pub n: usize,
    /// Natural frequencies ω_n: (n,)
    pub omega: Vec<f64>,
    /// Coupling matrix K_nm: flat row-major (n * n)
    pub coupling: Vec<f64>,
    /// Current phases θ_n: (n,)
    pub phases: Vec<f64>,
    /// Noise amplitude
    pub noise_amp: f64,
    /// Scratch arrays (pre-allocated for performance)
    dtheta: Vec<f64>,
    sin_diff: Vec<f64>,
}

impl KuramotoSolver {
    pub fn new(
        omega: Vec<f64>,
        coupling_flat: Vec<f64>,
        initial_phases: Vec<f64>,
        noise_amp: f64,
    ) -> Self;

    /// Advance one Euler step of size dt.
    /// Updates self.phases in-place.
    /// Returns the Kuramoto order parameter R ∈ [0, 1].
    pub fn step(&mut self, dt: f64, seed: u64) -> f64;

    /// Advance N steps, returning R after each step.
    pub fn run(&mut self, n_steps: usize, dt: f64, seed: u64) -> Vec<f64>;

    /// Compute the Kuramoto order parameter:
    /// R = |1/N Σ_n exp(i θ_n)|
    pub fn order_parameter(&self) -> f64;

    /// Get current phases.
    pub fn get_phases(&self) -> &[f64];

    /// Set phases (for synchronization with Python layers).
    pub fn set_phases(&mut self, phases: Vec<f64>);

    /// Set coupling matrix (for dynamic coupling updates).
    pub fn set_coupling(&mut self, coupling_flat: Vec<f64>);
}
```

ALGORITHM for `step()`:
```
for n in 0..N:
    coupling_sum = 0.0
    for m in 0..N:
        coupling_sum += K[n*N + m] * sin(phases[m] - phases[n])
    dtheta[n] = omega[n] + coupling_sum + noise_amp * random_normal()
phases[n] += dtheta[n] * dt

R = |mean(exp(i * phases))|
  = sqrt(mean(cos(phases))^2 + mean(sin(phases))^2)
```

OPTIMIZATION:
- Use rayon to parallelize the outer loop over n.
- Pre-compute sin_diff[n*N + m] = sin(phases[m] - phases[n]) using
  SIMD if available.
- The coupling_sum inner loop can use SIMD FMA (fused multiply-add).

═════════════════════════════════════════════════════════════
FILE 3: engine/src/scpn/metrics.rs
═════════════════════════════════════════════════════════════

```rust
/// SCPN-wide metrics computed from the 7-layer outputs.
pub struct SCPNMetrics;

impl SCPNMetrics {
    /// Compute weighted global coherence across all layers.
    /// weights: per-layer importance weights (7,)
    /// metrics: per-layer global metric values (7,)
    /// Returns: weighted average coherence ∈ [0, 1]
    pub fn global_coherence(weights: &[f64; 7], metrics: &[f64; 7]) -> f64;

    /// Compute the "consciousness index" — a composite score
    /// based on cross-layer synchronization.
    /// phases_l4: Kuramoto phases from L4
    /// glyph_l7: Glyph vector from L7
    /// Returns: index ∈ [0, 1]
    pub fn consciousness_index(
        phases_l4: &[f64], glyph_l7: &[f64; 6],
    ) -> f64;
}
```

═════════════════════════════════════════════════════════════
PyO3 BINDINGS: Add to engine/src/lib.rs
═════════════════════════════════════════════════════════════

```rust
pub mod scpn;

// In #[pymodule]:
m.add_class::<PyKuramotoSolver>()?;
```

Python API:
```python
from sc_neurocore_engine import KuramotoSolver
import numpy as np

omega = np.ones(400)  # 20x20 grid
K = np.random.uniform(0, 0.5, (400, 400))
K = (K + K.T) / 2  # symmetric
phases = np.random.uniform(0, 2*np.pi, 400)

solver = KuramotoSolver(omega, K, phases, noise_amp=0.1)
R_values = solver.run(n_steps=1000, dt=0.01)
print(f"Final R = {R_values[-1]:.4f}")
```

═════════════════════════════════════════════════════════════
BRIDGE WRAPPER
═════════════════════════════════════════════════════════════

CREATE: bridge/sc_neurocore_engine/scpn.py

```python
"""Accelerated SCPN components."""

import numpy as np
from sc_neurocore_engine.sc_neurocore_engine import (
    KuramotoSolver as _RustKuramoto,
)


class KuramotoSolver:
    """
    Drop-in replacement for the Kuramoto coupling loop in
    L4_CellularLayer and the UPDE solver.
    """

    def __init__(self, omega, coupling, phases, noise_amp=0.1):
        self._engine = _RustKuramoto(
            np.asarray(omega, dtype=np.float64).tolist(),
            np.asarray(coupling, dtype=np.float64).ravel().tolist(),
            np.asarray(phases, dtype=np.float64).tolist(),
            float(noise_amp),
        )

    def step(self, dt: float, seed: int = 0) -> float:
        return self._engine.step(dt, seed)

    def run(self, n_steps: int, dt: float, seed: int = 0) -> np.ndarray:
        return np.array(self._engine.run(n_steps, dt, seed))

    def order_parameter(self) -> float:
        return self._engine.order_parameter()

    @property
    def phases(self) -> np.ndarray:
        return np.array(self._engine.get_phases())

    @phases.setter
    def phases(self, new_phases):
        self._engine.set_phases(
            np.asarray(new_phases, dtype=np.float64).tolist()
        )
```

UPDATE: bridge/sc_neurocore_engine/__init__.py
  Add: from .scpn import KuramotoSolver

═════════════════════════════════════════════════════════════
TESTS
═════════════════════════════════════════════════════════════

CREATE: engine/tests/test_kuramoto.rs

```rust
#[test]
fn identical_phases_give_R_equals_one() {
    let n = 16;
    let omega = vec![1.0; n];
    let coupling = vec![0.0; n * n];
    let phases = vec![0.5; n];  // All identical
    let solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let r = solver.order_parameter();
    assert!((r - 1.0).abs() < 1e-10, "R should be 1.0 for identical phases");
}

#[test]
fn uniform_phases_give_R_near_zero() {
    let n = 1000;
    let omega = vec![1.0; n];
    let coupling = vec![0.0; n * n];
    // Uniformly distributed phases cancel out
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
        .collect();
    let solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let r = solver.order_parameter();
    assert!(r < 0.01, "R should be near 0 for uniform phases, got {r}");
}

#[test]
fn strong_coupling_increases_R() {
    let n = 50;
    let omega = vec![1.0; n];
    // Strong all-to-all coupling
    let coupling = vec![2.0; n * n];
    let mut rng_phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * ((i * 37 % n) as f64) / (n as f64))
        .collect();
    let mut solver = KuramotoSolver::new(
        omega, coupling, rng_phases, 0.0,
    );
    let r_initial = solver.order_parameter();
    let r_values = solver.run(500, 0.01, 42);
    let r_final = *r_values.last().unwrap();
    assert!(r_final > r_initial + 0.1,
        "Strong coupling should increase R: initial={r_initial}, final={r_final}");
}

#[test]
fn step_preserves_phase_count() {
    let n = 10;
    let omega = vec![1.0; n];
    let coupling = vec![0.1; n * n];
    let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.step(0.01, 0);
    assert_eq!(solver.get_phases().len(), n);
}
```

CREATE: tests/test_kuramoto_python.py

```python
"""Tests for Rust-accelerated Kuramoto solver."""

import numpy as np
from sc_neurocore_engine import KuramotoSolver


class TestKuramotoSolver:
    def test_synchronization(self):
        n = 100
        omega = np.ones(n)
        K = np.full((n, n), 1.0)
        phases = np.random.RandomState(42).uniform(0, 2*np.pi, n)

        solver = KuramotoSolver(omega, K, phases, noise_amp=0.0)
        R_values = solver.run(n_steps=500, dt=0.01)

        assert R_values[-1] > 0.8, (
            f"Strong coupling should synchronize: R={R_values[-1]:.4f}"
        )

    def test_order_parameter_range(self):
        solver = KuramotoSolver(
            np.ones(50), np.zeros((50, 50)),
            np.random.RandomState(42).uniform(0, 2*np.pi, 50),
        )
        R = solver.order_parameter()
        assert 0.0 <= R <= 1.0

    def test_phase_roundtrip(self):
        phases = np.array([0.1, 0.2, 0.3, 0.4])
        solver = KuramotoSolver(
            np.ones(4), np.zeros((4, 4)), phases,
        )
        np.testing.assert_allclose(solver.phases, phases, atol=1e-12)
```

CONSTRAINTS:
- The Kuramoto step() MUST use the CORRECT phase-difference coupling:
  Σ_m K_nm * sin(θ_m - θ_n)
  NOT: K @ sin(θ) (which was the bug fixed in v2.2.0)
- Noise generation: use ChaCha8Rng seeded from the provided seed
  parameter to generate N normal-distributed samples per step.
  If seed=0, use no noise.
- Pre-allocate scratch arrays (dtheta, sin_diff) in the constructor.
  Do NOT allocate per step.
- Order parameter formula: R = sqrt(mean(cos(θ))^2 + mean(sin(θ))^2)

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore/engine
cargo test --tests
cd ../bridge && ..\.venv\Scripts\python -m maturin develop --release && cd ..
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/test_kuramoto_python.py -v
```

═══════════════════════════════════════════════════════════════
```

---

## PACKET I: COMPREHENSIVE BENCHMARK SUITE

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET I: Benchmark Suite
═══════════════════════════════════════════════════════════════

CONTEXT:
Phase 1 includes a minimal benchmark (scripts/bench_v2_vs_v3.py)
that only tests pack+popcount. Phase 2 needs a comprehensive
head-to-head benchmark covering ALL v3 components.

Repository: sc-neurocore/
Depends on: All Phase 2 packets.

GOAL:
Create a benchmark suite that measures v2 vs v3 performance for
every accelerated operation and produces a formatted report.

═════════════════════════════════════════════════════════════
FILE 1: scripts/bench_v2_vs_v3.py (REPLACE existing)
═════════════════════════════════════════════════════════════

```python
#!/usr/bin/env python
"""
SC-NeuroCore v2 vs v3 Head-to-Head Benchmark Suite
===================================================

Measures wall-clock time for every operation that has both
a Python (v2) and Rust (v3) implementation.

Usage:
    cd 03_CODE/sc-neurocore
    $env:PYTHONPATH='src'
    .\.venv\Scripts\python scripts/bench_v2_vs_v3.py
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

# ── v2 imports ──
from sc_neurocore.accel.vector_ops import (
    pack_bitstream as v2_pack,
    unpack_bitstream as v2_unpack,
    vec_and as v2_and,
    vec_popcount as v2_popcount,
)
from sc_neurocore.neurons import FixedPointLIFNeuron as V2Lif
from sc_neurocore.layers import VectorizedSCLayer as V2Layer
from sc_neurocore.layers.attention import StochasticAttention as V2Attn

# ── v3 imports ──
import sc_neurocore_engine as v3
from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer
from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif
from sc_neurocore_engine.attention import StochasticAttention as V3Attn
from sc_neurocore_engine import KuramotoSolver


@dataclass
class BenchResult:
    name: str
    v2_ms: float
    v3_ms: float

    @property
    def speedup(self) -> float:
        return self.v2_ms / self.v3_ms if self.v3_ms > 0 else float("inf")


def bench(name: str, v2_fn: Callable, v3_fn: Callable,
          warmup: int = 3, repeats: int = 10) -> BenchResult:
    """Benchmark v2 and v3 implementations."""
    # Warmup
    for _ in range(warmup):
        v2_fn()
        v3_fn()

    # v2
    t0 = time.perf_counter()
    for _ in range(repeats):
        v2_fn()
    v2_ms = (time.perf_counter() - t0) * 1000 / repeats

    # v3
    t0 = time.perf_counter()
    for _ in range(repeats):
        v3_fn()
    v3_ms = (time.perf_counter() - t0) * 1000 / repeats

    return BenchResult(name, v2_ms, v3_ms)


def main():
    rng = np.random.RandomState(42)
    results: list[BenchResult] = []

    print("SC-NeuroCore v2 vs v3 Benchmark Suite")
    print(f"SIMD tier: {v3.simd_tier()}")
    print("=" * 60)

    # ── 1. Pack Bitstream ──
    bits_1m = rng.randint(0, 2, 1_000_000).astype(np.uint8)
    results.append(bench(
        "pack_bitstream (1M bits)",
        lambda: v2_pack(bits_1m),
        lambda: v3.pack_bitstream(bits_1m),
    ))

    # ── 2. Popcount ──
    packed_1m = v2_pack(bits_1m)
    v3_packed = v3.pack_bitstream(bits_1m)
    results.append(bench(
        "popcount (1M bits)",
        lambda: v2_popcount(packed_1m),
        lambda: v3.popcount(v3_packed),
    ))

    # ── 3. LIF Neuron (10K steps) ──
    def v2_lif_10k():
        lif = V2Lif()
        for _ in range(10_000):
            lif.step(20, 256, 128, 0)

    def v3_lif_10k():
        lif = V3Lif()
        for _ in range(10_000):
            lif.step(20, 256, 128, 0)

    results.append(bench("LIF neuron (10K steps)", v2_lif_10k, v3_lif_10k))

    # ── 4. Dense Layer Forward ──
    for n_in, n_out in [(16, 8), (64, 32), (128, 64)]:
        length = 1024
        v2_layer = V2Layer(n_inputs=n_in, n_neurons=n_out,
                           length=length, use_gpu=False)
        v3_layer = V3Layer(n_inputs=n_in, n_neurons=n_out,
                           length=length)
        inp = rng.uniform(0.1, 0.9, n_in)

        results.append(bench(
            f"Dense forward ({n_in}→{n_out}, L={length})",
            lambda i=inp, ly=v2_layer: ly.forward(i),
            lambda i=inp, ly=v3_layer: ly.forward(i),
        ))

    # ── 5. Attention ──
    for n, m, dk, dv in [(10, 20, 16, 32), (50, 100, 32, 64)]:
        Q = rng.uniform(0, 1, (n, dk))
        K = rng.uniform(0, 1, (m, dk))
        V = rng.uniform(0, 1, (m, dv))

        v2_attn = V2Attn(dim_k=dk)
        v3_attn = V3Attn(dim_k=dk)

        results.append(bench(
            f"Attention ({n}×{dk} → {m}×{dv})",
            lambda q=Q, k=K, v=V, a=v2_attn: a.forward(q, k, v),
            lambda q=Q, k=K, v=V, a=v3_attn: a.forward(q, k, v),
        ))

    # ── 6. Kuramoto Solver (v3 only — no direct v2 equivalent) ──
    n_osc = 400  # 20x20 grid
    omega = np.ones(n_osc)
    K_mat = rng.uniform(0, 0.5, (n_osc, n_osc))
    K_mat = (K_mat + K_mat.T) / 2
    phases = rng.uniform(0, 2 * np.pi, n_osc)

    def v3_kuramoto_1000():
        solver = KuramotoSolver(omega, K_mat, phases, noise_amp=0.0)
        solver.run(1000, 0.01)

    t0 = time.perf_counter()
    for _ in range(3):
        v3_kuramoto_1000()
    v3_k_ms = (time.perf_counter() - t0) * 1000 / 3

    # ── Print Report ──
    print()
    print(f"{'Operation':<45} {'v2 (ms)':>10} {'v3 (ms)':>10} {'Speedup':>10}")
    print("-" * 77)
    for r in results:
        print(f"{r.name:<45} {r.v2_ms:>10.3f} {r.v3_ms:>10.3f} {r.speedup:>9.1f}x")
    print("-" * 77)
    print(f"{'Kuramoto 400 osc × 1000 steps (v3 only)':<45} {'N/A':>10} {v3_k_ms:>10.3f} {'—':>10}")
    print()

    # ── Geometric Mean Speedup ──
    speedups = [r.speedup for r in results if r.speedup < float("inf")]
    if speedups:
        geo_mean = np.exp(np.mean(np.log(speedups)))
        print(f"Geometric mean speedup: {geo_mean:.1f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

═════════════════════════════════════════════════════════════
FILE 2: engine/benches/full_bench.rs (NEW)
═════════════════════════════════════════════════════════════

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sc_neurocore_engine::bitstream::{pack, popcount_words_portable};
use sc_neurocore_engine::encoder::BitstreamEncoder;
use sc_neurocore_engine::neuron::FixedPointLif;
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::simd::popcount_dispatch;
use sc_neurocore_engine::scpn::KuramotoSolver;

fn bench_all(c: &mut Criterion) {
    // ── Bitstream ──
    let bits_1m: Vec<u8> = (0..(1024 * 1024))
        .map(|i| if i % 3 == 0 { 1 } else { 0 })
        .collect();

    c.bench_function("pack_1m", |b| {
        b.iter(|| black_box(pack(black_box(&bits_1m))))
    });

    let packed = pack(&bits_1m);

    c.bench_function("popcount_portable_1m", |b| {
        b.iter(|| black_box(popcount_words_portable(black_box(&packed.data))))
    });

    c.bench_function("popcount_simd_1m", |b| {
        b.iter(|| black_box(popcount_dispatch(black_box(&packed.data))))
    });

    // ── Encoder ──
    c.bench_function("encoder_64k_steps", |b| {
        b.iter(|| {
            let mut enc = BitstreamEncoder::new(16, 0xACE1);
            for _ in 0..65535 {
                black_box(enc.step(32768));
            }
        })
    });

    // ── LIF Neuron ──
    c.bench_function("lif_10k_steps", |b| {
        b.iter(|| {
            let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
            for _ in 0..10_000 {
                black_box(lif.step(20, 256, 128, 0));
            }
        })
    });

    // ── Dense Layer ──
    let layer = DenseLayer::new(64, 32, 1024, 42);
    let inputs = vec![0.5_f64; 64];
    c.bench_function("dense_64x32_L1024", |b| {
        b.iter(|| black_box(layer.forward(black_box(&inputs), 42).unwrap()))
    });

    // ── Kuramoto ──
    let n = 100;
    let omega = vec![1.0; n];
    let coupling = vec![0.3; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
        .collect();

    c.bench_function("kuramoto_100_osc_1000_steps", |b| {
        b.iter(|| {
            let mut solver = KuramotoSolver::new(
                omega.clone(), coupling.clone(), phases.clone(), 0.0,
            );
            black_box(solver.run(1000, 0.01, 42));
        })
    });
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
```

Add to engine/Cargo.toml:
```toml
[[bench]]
name = "full_bench"
harness = false
```

VERIFICATION:
```powershell
cd 03_CODE/sc-neurocore
# Python benchmark
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts/bench_v2_vs_v3.py

# Rust benchmark
cd engine && cargo bench && cd ..
```

═══════════════════════════════════════════════════════════════
```

---

## PACKET J: CI/CD PIPELINE (Rust + Equivalence Tests)

```
═══════════════════════════════════════════════════════════════
HANDOVER PROMPT FOR CODEX — PACKET J: CI/CD Integration
═══════════════════════════════════════════════════════════════

CONTEXT:
The existing CI/CD pipeline (.github/workflows/ci.yml) only tests
the Python v2.2.0 package. It does NOT build the Rust engine or
run equivalence tests. Phase 2 needs a comprehensive CI that:

1. Builds the Rust engine on Linux/macOS/Windows.
2. Runs Rust unit/integration tests.
3. Installs the Python bridge (maturin develop).
4. Runs all equivalence tests (v2 vs v3).
5. Runs the existing v2 test suite (826 tests, 97% coverage).
6. Runs clippy and rustfmt on the Rust code.

DO NOT modify the existing ci.yml — create a NEW workflow file.

═════════════════════════════════════════════════════════════
FILE 1: .github/workflows/v3-engine.yml (NEW)
═════════════════════════════════════════════════════════════

```yaml
name: SC-NeuroCore v3 Engine

on:
  push:
    paths:
      - "engine/**"
      - "bridge/**"
      - "tests/equivalence/**"
      - "Cargo.toml"
  pull_request:
    paths:
      - "engine/**"
      - "bridge/**"
      - "tests/equivalence/**"
      - "Cargo.toml"

env:
  CARGO_TERM_COLOR: always
  PYTHONPATH: src

jobs:
  # ── Rust lint ──
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

  # ── Rust tests ──
  rust-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run Rust tests
        run: cargo test --manifest-path engine/Cargo.toml --tests

  # ── Python equivalence tests (matrix) ──
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
        run: pytest tests/test_surrogate_python.py tests/test_kuramoto_python.py -v --tb=short

  # ── Full v2 test suite (ensure no regressions) ──
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

═════════════════════════════════════════════════════════════
FILE 2: engine/.cargo/config.toml (if not created in D-0)
═════════════════════════════════════════════════════════════

NOTE: For CI, we do NOT want target-cpu=native because the
CI runner may have different CPU features than the target.
Only apply target-cpu=native for LOCAL builds.

Create engine/.cargo/config.toml with target-cpu=native ONLY
if it was not already created in Packet D-0. If it was already
created, leave it as-is. The CI runners will use default target
CPU (which enables SSE2/popcnt but not AVX-512).

═════════════════════════════════════════════════════════════
FILE 3: Update engine/Cargo.toml — ensure rlib for tests
═════════════════════════════════════════════════════════════

Verify that `crate-type` includes both "cdylib" and "rlib":
```toml
[lib]
name = "sc_neurocore_engine"
crate-type = ["cdylib", "rlib"]
```

This is needed so that `cargo test` can link against the crate
(rlib) while maturin produces the Python extension (cdylib).
This should already be the case from Phase 1.

═════════════════════════════════════════════════════════════

CONSTRAINTS:
- Do NOT modify the existing .github/workflows/ci.yml.
- The v3-engine.yml triggers ONLY on engine/bridge/equivalence changes.
- The equivalence job depends on rust-lint + rust-test passing first.
- The v2-compat job depends on equivalence passing.
- macOS runner will test NEON (if Apple Silicon) or SSE2 (Intel).
- Windows runner tests the .pyd build path.
- All equivalence tests MUST pass on all 3 OS × 2 Python versions.

VERIFICATION:
Verify the YAML is valid:
```powershell
python -c "import yaml; yaml.safe_load(open('.github/workflows/v3-engine.yml'))"
```

The workflow will be tested on the next push to the relevant paths.

═══════════════════════════════════════════════════════════════
```

---

## DELIVERY CHECKLIST

### Packet D-0 (Phase 1 Fixups)
- [ ] Remove `target-cpu = "native"` from `engine/Cargo.toml`
- [ ] Create `engine/.cargo/config.toml` with target-cpu=native
- [ ] Create `tests/equivalence/test_layer_equiv.py` (6 new tests)
- [ ] Fix NEON no-op in `engine/src/simd/neon.rs`
- [ ] Update `engine/benches/bitstream_bench.rs` (add SIMD dispatch bench)
- [ ] All 19+ equivalence tests pass

### Packet D (Surrogate Gradients)
- [ ] Create `engine/src/grad/mod.rs`
- [ ] Create `engine/src/grad/surrogate.rs` (SurrogateType, SurrogateLif, DifferentiableDenseLayer)
- [ ] Add PyO3 bindings in `engine/src/lib.rs`
- [ ] Create `bridge/sc_neurocore_engine/grad.py`
- [ ] Update `bridge/sc_neurocore_engine/__init__.py`
- [ ] Create `engine/tests/test_surrogate.rs` (10 tests)
- [ ] Create `tests/test_surrogate_python.py` (3 tests)
- [ ] SurrogateLif forward == FixedPointLif.step (bit-identical)

### Packet E (Attention + GNN)
- [ ] Create `engine/src/attention.rs`
- [ ] Create `engine/src/graph.rs`
- [ ] Add PyO3 bindings in `engine/src/lib.rs`
- [ ] Create `bridge/sc_neurocore_engine/attention.py`
- [ ] Create `bridge/sc_neurocore_engine/graphs.py`
- [ ] Update `bridge/sc_neurocore_engine/__init__.py`
- [ ] Create `tests/equivalence/test_attention_equiv.py` (6 tests)
- [ ] Create `tests/equivalence/test_gnn_equiv.py` (4 tests)
- [ ] Rate-mode matches v2 within 1e-12

### Packet F (SCPN Kuramoto)
- [ ] Create `engine/src/scpn/mod.rs`
- [ ] Create `engine/src/scpn/kuramoto.rs`
- [ ] Create `engine/src/scpn/metrics.rs`
- [ ] Add PyO3 bindings in `engine/src/lib.rs`
- [ ] Create `bridge/sc_neurocore_engine/scpn.py`
- [ ] Update `bridge/sc_neurocore_engine/__init__.py`
- [ ] Create `engine/tests/test_kuramoto.rs` (4 tests)
- [ ] Create `tests/test_kuramoto_python.py` (3 tests)
- [ ] Uses correct phase-difference coupling (NOT the v2.1.0 bug)

### Packet I (Benchmarks)
- [ ] Replace `scripts/bench_v2_vs_v3.py` (comprehensive)
- [ ] Create `engine/benches/full_bench.rs`
- [ ] Add bench to `engine/Cargo.toml`

### Packet J (CI/CD)
- [ ] Create `.github/workflows/v3-engine.yml`
- [ ] Existing ci.yml NOT modified
- [ ] 3 OS × 2 Python versions matrix

---

## EXECUTION ORDER

```
D-0 (fixups) ──────────────────────────────→ done
                                              │
          ┌───────────────────────────────────┤
          │               │                   │
          ▼               ▼                   ▼
    D (surrogates)   E (attn+gnn)      F (kuramoto)
          │               │                   │
          └───────┬───────┘                   │
                  │                           │
                  ▼                           │
            I (benchmarks) ◄──────────────────┘
                  │
                  ▼
            J (CI/CD)
```

Packets D, E, F are independent and can be parallelized.
Packet I depends on D+E+F (needs all components for benchmarking).
Packet J depends on everything (CI must test all components).

---

## STRICT RULES FOR CODEX

1. **NEVER modify any file under `src/sc_neurocore/`.** That is the v2.2.0
   Golden Reference. Zero changes.
2. **NEVER modify `pyproject.toml` in the repo root.** v2.2.0 packaging
   is sacred.
3. **NEVER modify `.github/workflows/ci.yml`.** Create a NEW workflow file.
4. **NEVER modify existing tests under `tests/`.** Only ADD new test files.
5. All Rust code must compile with `cargo clippy -- -D warnings` (no warnings).
6. All Rust code must pass `cargo fmt -- --check` (consistent formatting).
7. All Python wrappers must pass `black --check`.
8. The Kuramoto solver MUST use the correct phase-difference coupling formula.
9. SurrogateLif.forward() MUST be bit-identical to FixedPointLif.step().
10. Rate-mode attention and GNN MUST match v2 within 1e-12.

---

Anulum CH&LI / Anulum Institute
Miroslav Sotek
ORCID: 0009-0009-3560-0851

(c) 1998-2026 Anulum Institute. All rights reserved.
