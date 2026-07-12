# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Weighted address-event FPGA interconnect emission."""

from typing import Sequence

import numpy as np

from ..hdl_gen._ident import sanitize_ident
from ..ir.scnir_schema import SCNIRHierarchyInstance
from .fpga_connection_routing import (
    _ceil_log2_at_least_one,
    _connection_sources_are_analogue,
    _external_input_layout,
    _signed_hex,
)
from .fpga_neuron_rtl import _neuron_param_override, _type_default_qparams
from .fpga_scnir_hierarchy import (
    build_scnir_hierarchy_instance_block,
    _hierarchy_output_wires_by_stream,
    _hierarchy_weight_expr,
    _scnir_connection_stream_id,
)
from .neuron_graph import NeuronSpec
from .quantise_params import QuantisedGraph


def build_aer_interconnect(
    module_name: str,
    qgraph: QuantisedGraph,
    *,
    data_width: int = 16,
    fraction: int = 8,
    bitstream_length: int = 256,
    scnir_stream_count: int = 0,
    scnir_source_module_count: int = 0,
    scnir_hierarchy: Sequence[SCNIRHierarchyInstance] = (),
    scnir_semantic_hierarchy_stream_ids: frozenset[str] = frozenset(),
) -> str:
    """Generate weighted event-bus top-level interconnect.

    The emitted datapath keeps the same population instances as the direct
    interconnect but routes spike-producing source populations through an
    address-event fan-out block.  All active source spikes in a cycle contribute
    their signed fixed-point weights to every destination accumulator, so
    simultaneous events preserve the dense affine semantics of the NIR graph.
    External analogue inputs and analogue source populations remain direct
    fixed-point multiply-accumulate terms because they are not sparse events.

    Parameters
    ----------
    module_name : str
        Top-level module name.
    qgraph : QuantisedGraph
        Quantised graph.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits for analogue multiply downshift.
    bitstream_length : int
        SC-NIR bitstream length recorded in the generated top module.
    scnir_stream_count : int
        Number of semantic streams recorded in the SC-NIR document.
    scnir_source_module_count : int
        Number of materialised stochastic source modules.
    scnir_hierarchy : Sequence[SCNIRHierarchyInstance]
        Typed hierarchy boundaries instantiated by the network top.
    scnir_semantic_hierarchy_stream_ids : frozenset[str]
        Hierarchy stream identifiers that provide semantic connection weights.

    Returns
    -------
    str
        Verilog top-level module source.

    Raises
    ------
    ValueError
        If hierarchy metadata cannot be represented by the address-event
        interconnect contract.
    """
    pops = qgraph.populations
    conns = qgraph.connections
    type_defaults = _type_default_qparams(pops, data_width)
    safe_module = sanitize_ident(module_name, context="module name")
    pop_by_name = {pop.name: pop for pop in pops}
    pop_index = {pop.name: idx for idx, pop in enumerate(pops)}
    pop_offsets: dict[str, int] = {}

    offset = 0
    for pop in pops:
        pop_offsets[pop.name] = offset
        offset += pop.n_neurons

    external_width, external_offsets, _ = _external_input_layout(
        conns,
        pop_by_name,
        pops,
    )
    hierarchy_output_wires = _hierarchy_output_wires_by_stream(
        scnir_hierarchy,
        semantic_stream_ids=set(scnir_semantic_hierarchy_stream_ids),
    )

    max_terms = external_width if pops else 1
    for conn in conns:
        weights = np.asarray(conn.weights)
        max_terms = max(max_terms, weights.shape[1] + (1 if conn.bias is not None else 0))

    acc_width = max(
        data_width + 2,
        (2 * data_width) + _ceil_log2_at_least_one(max_terms + 1),
    )
    product_width = 2 * data_width
    input_bus_width = max(1, external_width * data_width)
    spike_width = max(1, qgraph.total_neurons)

    def neuron_prefix(pop: NeuronSpec, neuron_idx: int) -> str:
        return f"p{pop_index[pop.name]}_n{neuron_idx}"

    event_sources: list[tuple[NeuronSpec, int, str]] = []
    for pop in pops:
        if _connection_sources_are_analogue(pop):
            continue
        for neuron_idx in range(pop.n_neurons):
            event_sources.append((pop, neuron_idx, neuron_prefix(pop, neuron_idx)))

    aer_src_count = max(1, len(event_sources))
    aer_addr_width = _ceil_log2_at_least_one(aer_src_count)

    lines = [
        f"// Auto-generated top-level network: {safe_module}",
        "// SC-NeuroCore NIR → FPGA compiler",
        f"// Interconnect: weighted event routing ({qgraph.total_neurons} neurons)",
        f"// Populations: {len(pops)}, Connections: {len(conns)}",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module} (",
        "    input  wire clk,",
        "    input  wire rst_n,",
        "    input  wire en,",
        f"    input  wire signed [{input_bus_width - 1}:0] I_ext_flat,",
        f"    output wire [{spike_width - 1}:0] spike_bus",
        ");",
        "",
        f"    localparam integer DATA_WIDTH = {data_width};",
        f"    localparam integer ACC_WIDTH = {acc_width};",
        f"    localparam integer SCNIR_BITSTREAM_LENGTH = {bitstream_length};",
        f"    localparam integer SCNIR_STREAM_COUNT = {scnir_stream_count};",
        f"    localparam integer SCNIR_SOURCE_MODULE_COUNT = {scnir_source_module_count};",
        f"    localparam integer AER_SRC_COUNT = {aer_src_count};",
        f"    localparam integer AER_ADDR_WIDTH = {aer_addr_width};",
        "    localparam signed [DATA_WIDTH - 1:0] Q_MAX = {1'b0, {(DATA_WIDTH - 1){1'b1}}};",
        "    localparam signed [DATA_WIDTH - 1:0] Q_MIN = {1'b1, {(DATA_WIDTH - 1){1'b0}}};",
        "",
        "    function signed [DATA_WIDTH - 1:0] sat_acc;",
        "        input signed [ACC_WIDTH - 1:0] x;",
        "        begin",
        "            if (x > $signed({{(ACC_WIDTH - DATA_WIDTH){Q_MAX[DATA_WIDTH - 1]}}, Q_MAX}))",
        "                sat_acc = Q_MAX;",
        "            else if (x < $signed({{(ACC_WIDTH - DATA_WIDTH){Q_MIN[DATA_WIDTH - 1]}}, Q_MIN}))",
        "                sat_acc = Q_MIN;",
        "            else",
        "                sat_acc = x[DATA_WIDTH - 1:0];",
        "        end",
        "    endfunction",
        "",
    ]
    lines.extend(build_scnir_hierarchy_instance_block(scnir_hierarchy, data_width=data_width))

    lines.append("    // External analogue input vector")
    for idx in range(external_width):
        base = idx * data_width
        lines.append(
            f"    wire signed [DATA_WIDTH - 1:0] ext_input_{idx} = "
            f"I_ext_flat[{base} +: DATA_WIDTH];"
        )
    lines.append("")

    for pop in pops:
        mod = sanitize_ident(f"sc_nir_{pop.neuron_type}", context="module name")
        lines.append(
            f"    // Population {pop_index[pop.name]}: {pop.name} "
            f"({pop.neuron_type} x {pop.n_neurons})"
        )
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            lines.extend(
                [
                    f"    wire signed [DATA_WIDTH - 1:0] {prefix}_I;",
                    f"    wire {prefix}_spike;",
                    f"    wire signed [DATA_WIDTH - 1:0] {prefix}_v;",
                    f"    {mod}{_neuron_param_override(pop, neuron_idx, type_defaults, data_width)} {prefix}_inst (",
                    "        .clk(clk),",
                    "        .rst_n(rst_n),",
                    f"        .I_t({prefix}_I),",
                    f"        .spike_out({prefix}_spike),",
                    f"        .v_out({prefix}_v)",
                    "    );",
                    "",
                ]
            )

    lines.append("    // Weighted address-event source vector")
    if event_sources:
        event_concat = ", ".join(prefix + "_spike" for _, _, prefix in reversed(event_sources))
        lines.append(f"    wire [AER_SRC_COUNT - 1:0] aer_event_valid = {{{event_concat}}};")
        addr_terms = [
            f"aer_event_valid[{idx}] ? {aer_addr_width}'d{idx}" for idx in range(len(event_sources))
        ]
        lines.append(
            f"    wire [AER_ADDR_WIDTH - 1:0] aer_addr = {' : '.join(addr_terms)} : {aer_addr_width}'d0;"
        )
    else:
        lines.append("    wire [AER_SRC_COUNT - 1:0] aer_event_valid = 1'b0;")
        lines.append("    wire [AER_ADDR_WIDTH - 1:0] aer_addr = 1'b0;")
    lines.append("    wire aer_valid = |aer_event_valid;")
    lines.append("")

    term_defs: list[str] = []
    init_expr: dict[str, str] = {}
    event_adds: dict[str, list[str]] = {prefix: [] for _, _, prefix in event_sources}

    for pop in pops:
        feeding = [conn for conn in conns if conn.dst == pop.name]
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            acc_name = f"{prefix}_I_acc_next"
            lines.append(f"    reg signed [ACC_WIDTH - 1:0] {acc_name};")
            terms: list[str] = []

            if not feeding and pop_index[pop.name] == 0 and neuron_idx < external_width:
                terms.append(
                    f"{{{{(ACC_WIDTH - DATA_WIDTH){{ext_input_{neuron_idx}"
                    f"[DATA_WIDTH - 1]}}}}, ext_input_{neuron_idx}}}"
                )

            for conn_idx, conn in enumerate(feeding):
                weights = np.asarray(conn.weights, dtype=np.int64)
                if conn.bias is not None:
                    bias = int(np.asarray(conn.bias, dtype=np.int64).reshape(-1)[neuron_idx])
                    terms.append(_signed_hex(bias, acc_width))

                src_pop = pop_by_name.get(conn.src)
                for src_idx in range(weights.shape[1]):
                    weight = int(weights[neuron_idx, src_idx])
                    if weight == 0:
                        continue
                    term_base = f"{prefix}_c{conn_idx}_s{src_idx}"
                    weight_stream_id = _scnir_connection_stream_id(str(conn.src), str(conn.dst))
                    weight_expr = _hierarchy_weight_expr(
                        hierarchy_output_wires,
                        weight_stream_id,
                        weight=weight,
                        data_width=data_width,
                        weight_index=(neuron_idx * weights.shape[1]) + src_idx,
                    )

                    if src_pop is None:
                        external_idx = external_offsets[conn.src] + src_idx
                        mul = f"{term_base}_mul"
                        term = f"{term_base}_term"
                        term_defs.extend(
                            [
                                f"    wire signed [{product_width - 1}:0] {mul} = "
                                f"ext_input_{external_idx} * {weight_expr};",
                                f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};",
                            ]
                        )
                        terms.append(term)
                        continue

                    src_prefix = neuron_prefix(src_pop, src_idx)
                    if _connection_sources_are_analogue(src_pop):
                        mul = f"{term_base}_mul"
                        term = f"{term_base}_term"
                        term_defs.extend(
                            [
                                f"    wire signed [{product_width - 1}:0] {mul} = "
                                f"{src_prefix}_v * {weight_expr};",
                                f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};",
                            ]
                        )
                        terms.append(term)
                    else:
                        event_adds[src_prefix].append(
                            f"            {acc_name} = {acc_name} + {_signed_hex(weight, acc_width)};"
                        )

            init_expr[acc_name] = " + ".join(terms) if terms else f"{acc_width}'sd0"

    if term_defs:
        lines.append("")
        lines.append("    // Direct analogue multiply-accumulate terms")
        lines.extend(term_defs)
    lines.append("")
    lines.append("    // weighted event fan-out accumulation")
    lines.append("    always @(*) begin")
    for acc_name, expr in init_expr.items():
        lines.append(f"        {acc_name} = {expr};")
    for _, _, src_prefix in event_sources:
        additions = event_adds.get(src_prefix, [])
        if not additions:
            continue
        lines.append(f"        if ({src_prefix}_spike) begin")
        lines.extend(additions)
        lines.append("        end")
    lines.append("    end")
    lines.append("")

    for pop in pops:
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            lines.append(
                f"    assign {prefix}_I = en ? sat_acc({prefix}_I_acc_next) : {data_width}'sd0;"
            )

    lines.append("")
    for pop in pops:
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            bus_idx = pop_offsets[pop.name] + neuron_idx
            lines.append(f"    assign spike_bus[{bus_idx}] = {prefix}_spike;")

    lines.extend(["", "endmodule", ""])
    return "\n".join(lines)
