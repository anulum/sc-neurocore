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
| `analysis` | **22 spike train analysis modules** (see [Rust Analysis Engine](rust-analysis-engine.md)) |
| `neurons` | **100+ neuron models** — biophysical, maps, hardware, interneurons |

**[Criterion Benchmarks →](rust-benchmarks.md)** — all measured latencies for engine, neurons, and analysis.

## Building from Source

```bash
cd engine
cargo build --release
cargo test
cargo doc --open
```

## Python Bindings (PyO3)

The engine is exposed to Python via the local `sc_neurocore_engine` bridge.
For source checkouts:

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

## Installed-wheel interface evidence

The [engine bridge contracts](../guides/engine_bridge_contracts.md#capturing-the-installed-interface)
document `tools/engine_abi_inventory.py`, which captures the facade and compiled
module after wheel installation. Its installed-origin guard refuses a checkout
facade; retained captures include module hashes, complete interface metadata and
exact alias identities. Compare inventories under an equal Python/NumPy/feature
profile, then use per-binding behavioral tests for input layouts, exception and
mutation contracts, numerical goldens and supported pickle state.

The qualified default interface reference is
`tests/fixtures/engine_abi_default.json`, for Linux x86-64, CPython 3.12 and
NumPy 2.2.3 with default engine features. Its matching wheel job compares all
interface fields; other runtime profiles retain separate measured captures.
See [model compatibility and migration](../guides/engine_bridge_contracts.md#model-compatibility-and-migration)
for canonical source models and their explicit retained project profiles.
