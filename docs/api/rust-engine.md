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

## Fallible dense storage

`layer::DenseLayer::try_new` and the `try_get_weights`, `try_set_weights`,
`try_refresh_packed_weights`, `try_forward`, `try_forward_fast`,
`try_forward_fused`, `try_forward_batch`, `try_forward_batch_into`,
`try_forward_prepacked` and `try_forward_prepacked_2d` methods return
`layer::DenseLayerError`. Its `Invalid` variant retains authored shape refusals;
`Allocation` identifies storage that cannot be represented or reserved.
Weight replacements and caller-owned batch output commit only after complete
success. The existing Rust convenience signatures remain available; constructor,
weight-copy and refresh convenience methods retain their documented panic
behavior. Python bindings use the fallible paths and Python-owned outputs.

## Fallible GPU dense storage

GPU-feature wheels re-export `GpuDenseLayer`, `GpuLifBatch`, `GpuKuramoto` and
`GpuIzhikevichBatch` from `sc_neurocore_engine`; these are the native class
objects registered in `sc_neurocore_engine.sc_neurocore_engine`. Default wheels
omit the GPU exports.

With the `gpu` feature, `gpu::GpuDenseLayer::try_new_checked` returns a typed
`gpu::dense::GpuDenseError` or `Ok(None)` for unavailable GPU hardware.
`try_forward_gpu` and `try_forward_batch_gpu` validate inputs and reserve host
storage before submitting the original Philox shader kernels. Construction
checks shader index widths, device buffer limits and dispatch limits. A zero
sample batch returns an empty result. Calls sharing persistent GPU buffers
serialize submission and readback; rejected shapes leave packed weights intact.
The original Rust convenience signatures remain available with documented panic
behavior. Python uses the checked methods and separately owned list outputs.

GPU feature qualification requires the dedicated Rust integration surface and
an installed GPU-feature wheel for the Python surface:

```bash
cargo test --locked --features gpu --test test_gpu_dense -- --test-threads=1
python -m pytest engine/tests/python/test_gpu_dense_layer_engine_binding.py
```

These GPU tests require real hardware; a default wheel does not provide this
optional binding. The shader's Philox inputs remain distinct from the CPU
encoder's random streams, so exact same-seed CPU outputs are not its oracle.

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
