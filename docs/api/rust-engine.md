# Rust Engine API (sc_neurocore_engine)

The high-performance Rust engine provides SIMD-accelerated stochastic computing
with SIMD-accelerated bitstream operations, IR compilation, and HDC support.

**[Browse the full Rust API documentation →](https://anulum.github.io/sc-neurocore/rust-api/sc_neurocore_engine/)**

## Key Modules

| Module | Description |
|--------|-------------|
| `bitstream` | Packed bitstream types and SIMD operations (AND, popcount, rotate) |
| `encoder` | LFSR-based stochastic encoders with decorrelated seeds |
| `neuron` | Fixed-point LIF neuron with Q8.8 arithmetic |
| `layer` | Dense layer pipeline with vectorised forward pass |
| `ir` | Intermediate representation for graph compilation |
| `graph` | Computational graph builder and verifier |
| `attention` | Stochastic attention mechanism |
| `grad` | Surrogate gradient training support |
| `scpn` | SCPN layer primitives (Petri net places/transitions) |
| `simd` | Platform-adaptive SIMD kernels (AVX2, SSE4.1, NEON, portable) |

## Building from Source

```bash
cd engine
cargo build --release
cargo test
cargo doc --open
```

## Python Bindings (PyO3)

The engine is exposed to Python via the `sc_neurocore_engine` wheel:

```bash
cd bridge
maturin develop --release
```

```python
import sc_neurocore_engine as engine

# Compile an IR graph
graph = engine.IRGraph()
graph.add_encode(0, 1024, 0xACE1)
graph.verify()
sv_code = graph.emit_sv()
```
