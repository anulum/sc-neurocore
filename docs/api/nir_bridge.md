# NIR Bridge API

::: sc_neurocore.nir_bridge
    options:
      show_root_heading: true
      members_order: source

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
