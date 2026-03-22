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
