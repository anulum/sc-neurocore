# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical hardware neuron graph facade

"""Preserve the historical neuron-graph API over responsibility modules.

The graph contracts and :func:`from_scnetwork` remain import-compatible while
node classification, hierarchy flattening, dense lowering, metadata handling,
connection resolution, and conversion orchestration live in focused modules.
"""

from __future__ import annotations

from sc_neurocore.nir_bridge.neuron_graph_builder import (
    from_scnetwork as from_scnetwork,
    logger as logger,
)
from sc_neurocore.nir_bridge.neuron_graph_connections import (
    _fold_connection_scales as _fold_connection_scales,
    _resolve_weight_destination as _resolve_weight_destination,
    _resolve_weight_source as _resolve_weight_source,
)
from sc_neurocore.nir_bridge.neuron_graph_contracts import (
    ConnectionSpec as ConnectionSpec,
    DelaySteps as DelaySteps,
    HierarchyInstanceSpec as HierarchyInstanceSpec,
    NeuronGraph as NeuronGraph,
    NeuronSpec as NeuronSpec,
)
from sc_neurocore.nir_bridge.neuron_graph_dense import (
    _conv1d_to_dense_matrix as _conv1d_to_dense_matrix,
    _conv2d_to_dense_matrix as _conv2d_to_dense_matrix,
    _pool2d_to_dense_matrix as _pool2d_to_dense_matrix,
    _weight_matrix_and_bias as _weight_matrix_and_bias,
)
from sc_neurocore.nir_bridge.neuron_graph_hierarchy import (
    _hdl_identifier_fragment as _hdl_identifier_fragment,
    _inline_single_port_subgraphs as _inline_single_port_subgraphs,
    _topological_order as _topological_order,
)
from sc_neurocore.nir_bridge.neuron_graph_metadata import (
    _broadcast_scale as _broadcast_scale,
    _broadcast_threshold as _broadcast_threshold,
    _compose_delay_steps as _compose_delay_steps,
    _compose_scale as _compose_scale,
    _delay_steps as _delay_steps,
    _delay_steps_array as _delay_steps_array,
    _fit_delay_steps_to_width as _fit_delay_steps_to_width,
    _flatten_widths as _flatten_widths,
    _scale_vector as _scale_vector,
    _shape_width as _shape_width,
    _threshold_vector as _threshold_vector,
)
from sc_neurocore.nir_bridge.neuron_graph_nodes import (
    _DELAY_NODE_NAME as _DELAY_NODE_NAME,
    _FLATTEN_NODE_NAME as _FLATTEN_NODE_NAME,
    _MULTIPORT_SUBGRAPH_NODE as _MULTIPORT_SUBGRAPH_NODE,
    _SCALE_NODE_NAME as _SCALE_NODE_NAME,
    _SC_NODE_TO_TYPE as _SC_NODE_TO_TYPE,
    _SC_PASSTHROUGH_NODES as _SC_PASSTHROUGH_NODES,
    _SC_WEIGHT_NODES as _SC_WEIGHT_NODES,
    _SINGLE_PORT_SUBGRAPH_NODE as _SINGLE_PORT_SUBGRAPH_NODE,
    _THRESHOLD_NODE_NAME as _THRESHOLD_NODE_NAME,
    _extract_neuron_params as _extract_neuron_params,
    _node_logical_width as _node_logical_width,
)

_HISTORICAL_DEFINITIONS = (
    ConnectionSpec,
    HierarchyInstanceSpec,
    NeuronGraph,
    NeuronSpec,
    from_scnetwork,
)

for _definition in _HISTORICAL_DEFINITIONS:
    _definition.__module__ = __name__

del _definition
del _HISTORICAL_DEFINITIONS
