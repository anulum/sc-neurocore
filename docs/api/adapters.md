# Adapters

Domain-specific adapter layer. The base adapter defines the interface;
holonomic adapters map between SCPN layers and external coordinate systems.
All 16 L1-L16 adapters are registered in the global ComponentRegistry
and accessible via the `create_adapter(layer)` factory.

## Base Adapter

::: sc_neurocore.adapters.base

## Holonomic Atlas (L1-L16)

::: sc_neurocore.adapters.holonomic

## SpikeInterface / Neo Adapter

Import experimental spike data into SC-NeuroCore. Converts between
SpikeInterface sorting results and SC-NeuroCore representations
(bitstream matrices, Population inputs, SC probabilities).

::: sc_neurocore.adapters.spikeinterface

## Plugin Discovery

Community-contributed adapters can be discovered via `importlib.metadata`
entry points in group `sc_neurocore.adapters`.

::: sc_neurocore.utils.adapter_discovery
