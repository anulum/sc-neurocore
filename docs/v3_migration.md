# SC-NeuroCore v3 Migration Guide

## Status

Phase 1 scaffolding is in place:

- Rust engine crate in `engine/`
- Python bridge package in `bridge/sc_neurocore_engine/`
- v2-vs-v3 equivalence tests in `tests/equivalence/`

## Build (Local)

```powershell
cd 03_CODE/sc-neurocore/engine
maturin develop --release
```

## Quick Sanity Check

```powershell
python -c "import sc_neurocore_engine; print(sc_neurocore_engine.__version__); print(sc_neurocore_engine.simd_tier())"
```

## Equivalence Tests

```powershell
cd 03_CODE/sc-neurocore
$env:PYTHONPATH=\"src;bridge\"
python -m pytest tests/equivalence -v --tb=short
```

## Notes

- v2 package under `src/sc_neurocore/` remains untouched.
- v3 bridge is a drop-in import path for hot kernels and fixed-point neuron APIs.
- Encoder and LIF in v3 currently follow strict blueprint operation ordering
  (step-then-compare encoder, refractory override after threshold evaluation).

## Phase 2 Features (February 2026)

### Surrogate Gradients

SC-NeuroCore v3 introduces backpropagation support for stochastic
computing layers via surrogate gradients:

- `SurrogateLif` - LIF neuron with differentiable backward pass
- `DifferentiableDenseLayer` - SC layer with weight gradient computation
- Supported surrogates: FastSigmoid, SuperSpike, ArcTan, StraightThrough

### Stochastic Attention

- Rate-mode: bit-exact match with v2 (atol < 1e-12)
- SC-mode: bitstream-based matrix multiply (new v3 capability)
- Multi-head support (Phase 3)

### Graph Neural Network

- Rate-mode: bit-exact match with v2 (atol < 1e-12)
- SC-mode: bitstream-based message passing (Phase 3)

### Kuramoto Oscillator Solver

- High-performance phase-difference coupling
- SSGF-compatible extended solver with geometry + PGBO terms
- Pre-allocated scratch arrays, rayon parallelism
- Box-Muller noise generation with ChaCha8Rng

## Phase 3 Features (February 2026)

### SSGF Integration

- `step_ssgf()` - Extended Kuramoto with geometry (`W`), PGBO (`h_munu`),
  and field pressure (`F*cos`) coupling terms
- Direct integration with SSGF MicroCycleEngine pipeline
- Single `sin_diff` computation shared across all coupling terms

### Property-Based Testing

- proptest coverage for all numeric modules
- Catches edge cases: overflows, NaN, extreme values

## Phase 4 Features (February 2026)

### SC Compute Graph IR

A Rust-native intermediate representation for SC pipelines:

- `ScGraph`: Directed acyclic graph of SC operations
- `ScGraphBuilder`: Fluent API for graph construction
- `verify()`: Static verification (SSA, type checking, acyclicity)
- `print()` / `parse()`: Stable text format with round-trip fidelity
- 11 operation types mapping to HDL primitives

### SystemVerilog Emitter

Compile IR graphs to synthesizable RTL:

- Direct instantiation of existing `hdl/` modules
- Automatic clock/reset distribution
- Constant folding for Q8.8 fixed-point parameters

### Co-Simulation Harness

Verify generated HDL against Rust golden model:

- LFSR full-cycle equivalence
- LIF neuron bit-exact comparison
- Encoder probability convergence
- Synapse AND operation verification

## Phase 5 Features (February 2026)

### IR Python Bridge

Construct SC compute graphs from Python and compile to SystemVerilog:

```python
from sc_neurocore_engine.ir import ScGraphBuilder

b = ScGraphBuilder("my_synapse")
x = b.input("x_prob", "rate")
w = b.input("w_prob", "rate")
x_enc = b.encode(x, length=1024, seed=0xACE1)
w_enc = b.encode(w, length=1024, seed=0xBEEF)
product = b.bitwise_and(x_enc, w_enc)
count = b.popcount(product)
rate = b.div_const(count, 1024)
b.output("firing_rate", rate)

graph = b.build()
assert graph.verify() is None
sv_code = graph.emit_sv()
```

### Co-Simulation

When Verilator is installed, co-sim tests compile HDL and compare
against the Rust golden model bit-by-bit. Without Verilator,
tests skip gracefully.

### Distributable Wheels

Pre-built wheels available for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x86_64)
- Python 3.9, 3.10, 3.11, 3.12
