# Tutorial 20: Holonomic Adapter Ecosystem

SC-NeuroCore provides 16 holonomic adapters that map between SCPN
consciousness layers (L1–L16) and the stochastic computing substrate.
Each adapter encodes domain-specific state into bitstreams, runs a
JAX-compatible simulation step, and decodes the result back into
domain quantities.

## Prerequisites

```bash
pip install sc-neurocore[jax]  # JAX backend for step_jax()
```

## Using the Factory

The simplest way to get an adapter is via `create_adapter()`:

```python
from sc_neurocore.adapters.holonomic import create_adapter

# Create the L1 Quantum adapter
adapter = create_adapter(1)

# All 16 layers
adapters = [create_adapter(i) for i in range(1, 17)]
```

## Adapter Lifecycle

Every adapter follows a three-phase lifecycle:

```python
from sc_neurocore.adapters.holonomic import create_adapter

adapter = create_adapter(7)  # L7 Symbolic

# 1. Encode domain state into JAX-compatible arrays
state = adapter.encode(initial_state)

# 2. Step the simulation (dt in seconds)
new_state = adapter.step_jax(dt=0.001, inputs=external_input)

# 3. Decode back to domain quantities
result = adapter.decode(new_state)
metrics = adapter.get_metrics()
```

## Layer Directory

| Layer | Adapter | Domain |
|-------|---------|--------|
| L1 | `L1_QuantumAdapter` | Quantum field dynamics |
| L2 | `L2_NeurochemicalAdapter` | Neurotransmitter kinetics |
| L3 | `L3_GenomicAdapter` | Gene regulatory networks |
| L4 | `L4_CellularAdapter` | Cellular signalling |
| L5 | `L5_OrganismalAdapter` | Autonomic nervous system |
| L6 | `L6_PlanetaryAdapter` | Geophysical coupling |
| L7 | `L7_SymbolicAdapter` | Symbolic/Vibrana resonance |
| L8 | `L8_CosmicAdapter` | Cosmological phase fields |
| L9 | `L9_MemoryAdapter` | Persistent memory traces |
| L10 | `L10_FirewallAdapter` | Boundary/firewall dynamics |
| L11 | `L11_NoosphericAdapter` | Collective intelligence |
| L12 | `L12_GaianAdapter` | Earth-system feedback |
| L13 | `L13_SourceAdapter` | Source field coupling |
| L14 | `L14_TransdimensionalAdapter` | Cross-dimensional mapping |
| L15 | `L15_ConsiliumAdapter` | Consilience integration |
| L16 | `L16_MetaAdapter` | Meta-cognitive director |

## Registry Integration

All 16 adapters are registered in the global `ComponentRegistry`
at import time:

```python
from sc_neurocore.utils.registry import registry

# Triggers registration on first import
import sc_neurocore.adapters.holonomic

# List all registered adapters
print(registry.list("adapter"))
# ['L10_Firewall', 'L11_Noospheric', ..., 'L9_Memory', 'neuroml', ...]

# Retrieve by name
cls = registry.get("adapter", "L7_Symbolic")
adapter = cls()
```

## Creating a Custom Adapter

Extend `BaseStochasticAdapter`:

```python
from sc_neurocore.adapters.base import BaseStochasticAdapter
import numpy as np

class MyCustomAdapter(BaseStochasticAdapter):
    def encode(self, state):
        return np.array(state, dtype=float)

    def step_jax(self, dt, inputs=None):
        # Your dynamics here
        return self.state * np.exp(-dt)

    def decode(self, bitstream):
        return float(bitstream.mean())

    def get_metrics(self):
        return {"energy": 0.0}
```

Register it:

```python
from sc_neurocore.utils.registry import registry

registry.register("adapter", "MyCustom")(MyCustomAdapter)
```

## Plugin Discovery

First-party importer adapter classes are declared in SC-NeuroCore's
`pyproject.toml` under the same entry-point group used by third-party plugins:

```python
from sc_neurocore.utils.adapter_discovery import discover_adapters

found = discover_adapters(include_entry_points=False)
print(found["neuroml"])  # sc_neurocore.adapters.importers.NeuroMLImporter
```

Third-party adapters can be discovered via Python entry points. Add to your
package's `pyproject.toml`:

```toml
[project.entry-points."sc_neurocore.adapters"]
MyAdapter = "my_package.adapters:MyCustomAdapter"
```

Then discover at runtime:

```python
found = discover_adapters()  # auto-registers into global registry
```

Discovery is idempotent: repeated calls return the discovered classes and leave
existing registry entries intact.

## Benchmarking

Run the built-in benchmark suite:

```bash
python benchmarks/adapter_benchmark.py
```

This measures per-adapter latency, peak memory, and throughput
(steps/sec), outputting JSON and markdown reports to
`benchmarks/results/`.

## Next Steps

- [Tutorial 05: Rust Engine Performance](05_rust_engine_performance.md) — SIMD acceleration
- [Tutorial 15: Quantum-SC Hybrid](15_quantum_sc_hybrid.md) — quantum backend integration
- [API Reference: Adapters](../api/adapters.md) — full class signatures
