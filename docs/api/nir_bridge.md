# NIR Bridge API

::: sc_neurocore.nir_bridge
    options:
      show_root_heading: true
      members_order: source

## FPGA Network Compiler

::: sc_neurocore.nir_bridge.fpga_compiler
    options:
      show_root_heading: true
      members:
        - compile_network_to_fpga
        - NetworkCompilationResult
        - FoldedResourceMetrics

`compile_network_to_fpga()` remains the stable public composition boundary.
Its implementation delegates to responsibility-specific modules for connection
validation, neuron RTL, weight ROMs, SC-NIR hierarchy boundaries, and the
direct, address-event, and folded interconnects. Result classes retain their
historical `sc_neurocore.nir_bridge.fpga_compiler` serialisation path.

Malformed direct `NeuronGraph` inputs fail before SC-NIR conversion or HDL
emission: the compiler rejects empty networks, non-matrix weights, inconsistent
source/destination widths, invalid bias or threshold vectors, and invalid
connection delays with stable `ValueError` contracts.

## Hardware Neuron Graph

::: sc_neurocore.nir_bridge.neuron_graph
    options:
      show_root_heading: true
      members:
        - NeuronSpec
        - ConnectionSpec
        - HierarchyInstanceSpec
        - NeuronGraph
        - from_scnetwork

`from_scnetwork()` is the boundary between the executable parser graph and FPGA
lowering. The historical module remains a compatibility facade: graph records
and the converter retain their established qualified names and pickle paths,
while focused modules own contracts, node classification, nested hierarchy,
dense operators, connection traversal, metadata, and conversion orchestration.

The lowering path preserves source and destination scales, scalar or
per-channel delays, thresholds, flatten widths, nested-instance provenance,
and recurrent delayed edges. Ambiguous pass-through fan-in/fan-out, malformed
flatten dimensions, non-finite dense parameters, unsupported nested boundaries,
and hierarchy cycles fail before an incomplete hardware graph can be emitted.
The resulting `NeuronGraph` is consumed directly by
`compile_network_to_fpga()` and SC-NIR conversion.

## Parser

::: sc_neurocore.nir_bridge.parser
    options:
      show_root_heading: true
      members:
        - from_nir
        - SCNetwork
        - SCSubgraphNode
        - SCMultiPortSubgraphNode

### Recurrent Edge Handling

Graphs with cycles (feedback connections) are automatically handled by
inserting unit-delay nodes on back edges. The delay node buffers the
previous timestep's value, breaking algebraic loops while preserving
temporal dynamics. See `_UnitDelayNode`.

### Multi-Port Subgraphs

Nested NIR graphs with multiple inputs/outputs use `SCMultiPortSubgraphNode`,
which exposes `forward_multi(inputs_dict) → outputs_dict` for named I/O ports.

### Import Boundary Validation

`from_nir()` accepts a `nir.NIRGraph`, string path, or `Path`. File reads are
wrapped as `ValueError` on malformed or unreadable NIR payloads. Parsed graphs
must expose mapping-like nodes and sequence-like edges, all node names and edge
endpoints must be non-empty strings, and every edge endpoint must reference an
existing node before the graph is lowered.

### High-Level Hardware Path

`SCNNetwork` is the public alias for the parsed NIR `SCNetwork`. It supports
`SCNNetwork.from_nir(...)` for import and `network.to_hardware(...)` for
lowering through the same `from_scnetwork()` and `compile_network_to_fpga()`
pipeline used by the lower-level compiler API.

## Node Map

::: sc_neurocore.nir_bridge.node_map
    options:
      show_root_heading: true
      members:
        - map_node
        - SCLIFNode
        - SCIFNode
        - SCLINode
        - SCIntegratorNode
        - SCAffineNode
        - SCLinearNode
        - SCScaleNode
        - SCThresholdNode
        - SCFlattenNode
        - SCInputNode
        - SCOutputNode
        - SCDelayNode
        - SCCubaLIFNode
        - SCCubaLINode
        - SCConv1dNode
        - SCConv2dNode
        - SCSumPool2dNode
        - SCAvgPool2dNode
        - NODE_MAP

## Export

::: sc_neurocore.nir_bridge.export
    options:
      show_root_heading: true
      members:
        - to_nir

## Hardware Target Manifests

::: sc_neurocore.nir_bridge.hardware_targets
    options:
      show_root_heading: true
      members:
        - SCMappingConstraints
        - NeuromorphicHardwareProfile
        - HardwareNoiseAnnotation
        - available_hardware_profiles
        - get_hardware_profile
        - build_nir_hardware_manifest
        - build_noise_annotation

`build_nir_hardware_manifest()` records capability manifests for Akida,
Loihi 2, BrainScaleS-3, SpiNNaker2, and DYNAP-SE. These entries are planning
metadata, not live SDK integrations: each profile carries `backend_status:
capability_manifest` and only records NIR node support, SC bitstream ranges,
stream transport, stochastic sources, and noise channels that can be measured
and replayed in simulation.

```python
from sc_neurocore.nir_bridge import build_nir_hardware_manifest, build_noise_annotation

manifest = build_nir_hardware_manifest(("loihi2", "spinnaker2", "akida"))
noise = build_noise_annotation("loihi2", {"spike_drop_rate": 0.001})
```

Noise annotations validate channel names and reject non-finite or negative
measurements before they can influence simulation.

## Loihi 2 / SpiNNaker2 Adapter Packages

::: sc_neurocore.nir_bridge.neuromorphic_adapters
    options:
      show_root_heading: true
      members:
        - NeuromorphicAdapterPackage
        - build_neuromorphic_adapter_package
        - build_neuromorphic_adapter_bundle
        - write_neuromorphic_adapter_bundle

`build_neuromorphic_adapter_package()` turns a parsed NIR graph into a
deterministic handoff package for either `loihi2` or `spinnaker2`. The package
contains:

- `adapter_manifest.json` with lowering status, fallback requirements, selected
  bitstream length, and noise back-annotation hooks;
- `nir_silicon_mapping_report.json`, the full mapping report used to build the
  manifest;
- `README.md` documenting the vendor SDK boundary.

The adapter package is intentionally SDK-free. Loihi 2 execution still requires
Lava/Loihi access, and SpiNNaker2 execution still requires the SpiNNaker2 SDK
and board access. The package is therefore a reproducible planning and handoff
artefact, not a hardware-execution claim.

```python
from sc_neurocore.nir_bridge import write_neuromorphic_adapter_bundle

write_neuromorphic_adapter_bundle(
    "build/neuromorphic_targets",
    nir_graph,
    targets=("loihi2", "spinnaker2"),
)
```
