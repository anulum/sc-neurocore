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
        - NODE_MAP

## Export

::: sc_neurocore.nir_bridge.export
    options:
      show_root_heading: true
      members:
        - to_nir
