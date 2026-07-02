# Adapters

Domain-specific adapter layer. The base adapter defines the interface;
holonomic adapters map between SCPN layers and external coordinate systems.
All 16 L1-L16 adapters are registered in the global ComponentRegistry
and accessible via the `create_adapter(layer)` factory.

## Base Adapter

::: sc_neurocore.adapters.base

## Holonomic Atlas (L1-L16)

::: sc_neurocore.adapters.holonomic

### L6 Planetary Adapter Contract

`sc_neurocore.adapters.holonomic.l6_plan.L6_PlanetaryAdapter` accepts optional
rank-2 upstream bitstream batches shaped `(N, bitstream_length)`. The adapter
validates positive integer region dimensions, positive finite Schumann
frequency, cavity quality, Gaia coupling, bounded percolation threshold,
positive finite `dt`, non-empty input rows, exact bitstream width, and finite
input values before mutating planetary field state. If `N` differs from
`n_regions`, the mean regional drive is broadcast deterministically across all
configured regions.

The Python/JAX adapter owns the runtime Gaia-field update. The Rust safety
mirror, Julia mirror, and Mojo validation shim expose the same parameter,
timestep, and input-projection boundaries for downstream generated-kernel
checks; they are not benchmark-dispatched acceleration paths.

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

### L13 Source-Field Adapter Contract

`sc_neurocore.adapters.holonomic.l13_source.L13_SourceAdapter` accepts optional
L16 cybernetic-closure feedback as a scalar, rank-1 vector, or rank-2 batch.
The adapter validates positive integer vacuum dimensions, finite primordial
coupling and source bias, finite non-negative scission drive, positive finite
`dt`, non-empty feedback vectors/batches, rank at most 2, and finite input
values before mutating vacuum or Fisher-metric state. Scalars broadcast across
all nodes; mismatched vector lengths or batch row counts broadcast the mean
feedback drive deterministically across the configured vacuum lattice.

The Python/JAX adapter owns the runtime source-field update. The Rust safety
mirror, Julia mirror, and Mojo validation shim expose the same parameter,
timestep, feedback-projection, and decode boundaries for downstream
generated-kernel checks; they are not benchmark-dispatched acceleration paths.

## SpikeInterface / Neo Adapter

Import experimental spike data into SC-NeuroCore. Converts between
SpikeInterface sorting results and SC-NeuroCore representations
(bitstream matrices, Population inputs, SC probabilities).

::: sc_neurocore.adapters.spikeinterface

## Plugin Discovery

Community-contributed adapters can be discovered via `importlib.metadata`
entry points in group `sc_neurocore.adapters`.

::: sc_neurocore.utils.adapter_discovery
