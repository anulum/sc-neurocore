# Adapters

Domain-specific adapter layer. The base adapter defines the interface;
holonomic adapters map between SCPN layers and external coordinate systems.
All 16 L1-L16 adapters are registered in the global ComponentRegistry
and accessible via the `create_adapter(layer)` factory.

## Base Adapter

::: sc_neurocore.adapters.base

## Holonomic Atlas (L1-L16)

::: sc_neurocore.adapters.holonomic

### L9 Memory Adapter Contract

`sc_neurocore.adapters.holonomic.l9_mem.L9_MemoryAdapter` accepts optional
rank-2 upstream bitstream batches shaped `(N, bitstream_length)`. The adapter
now validates positive integer memory dimensions, finite non-negative retrieval
gain, bounded weak-measurement strength, positive finite `dt`, non-empty input
rows, exact bitstream width, and finite input values before mutating its
forward/backward TSVF state. If `N` differs from `n_memory_slots`, rows are
tiled deterministically by slot index instead of relying on backend modulo or
broadcasting errors.

The Python/JAX adapter owns the runtime TSVF update. The Rust safety mirror,
Julia mirror, and Mojo validation shim expose the same parameter, timestep, and
input-projection boundaries for downstream generated-kernel checks; they are
not benchmark-dispatched acceleration paths.

## SpikeInterface / Neo Adapter

Import experimental spike data into SC-NeuroCore. Converts between
SpikeInterface sorting results and SC-NeuroCore representations
(bitstream matrices, Population inputs, SC probabilities).

::: sc_neurocore.adapters.spikeinterface

## Plugin Discovery

Community-contributed adapters can be discovered via `importlib.metadata`
entry points in group `sc_neurocore.adapters`.

::: sc_neurocore.utils.adapter_discovery
