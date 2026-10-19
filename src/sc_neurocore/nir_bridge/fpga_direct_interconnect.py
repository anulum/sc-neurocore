# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Explicit per-neuron FPGA interconnect emission."""

from typing import Callable, Sequence

import numpy as np

from ..hdl_gen._ident import sanitize_ident
from ..ir.scnir_schema import SCNIRHierarchyInstance
from .fpga_connection_routing import (
    DelayVector,
    _ceil_log2_at_least_one,
    _connection_sources_are_analogue,
    _external_input_layout,
    _normalise_connection_delay_steps,
    _signed_hex,
)
from .fpga_neuron_rtl import (
    _neuron_param_override,
    _type_default_qparams,
    population_draws,
    population_graded,
    population_stream,
)
from .fpga_scnir_hierarchy import (
    build_scnir_hierarchy_instance_block,
    _hierarchy_output_wires_by_stream,
    _hierarchy_weight_expr,
    _scnir_connection_stream_id,
)
from .neuron_graph import ConnectionSpec, NeuronSpec
from .quantise_params import QuantisedGraph
from .stdp import QuantisedSTDP


def check_plastic_connection(
    conn: ConnectionSpec, pop_by_name: dict[str, NeuronSpec], graded: set[str]
) -> None:
    """Refuse a plastic connection outside what the STDP datapath implements.

    STDP needs spike-timed pre and post neurons: the source must be a binary
    spiking population (not external, analogue or graded), and the connection
    may carry no delay and no NIR threshold.

    Raises
    ------
    ValueError
        Naming the connection and what is not supported.
    """
    name = f"Connection {conn.src}->{conn.dst}"
    src_pop = pop_by_name.get(conn.src)
    if src_pop is None or _connection_sources_are_analogue(src_pop) or conn.src in graded:
        raise ValueError(f"{name}: STDP needs a binary spiking source population")
    if any(_normalise_connection_delay_steps(conn.delay_steps, src_pop.n_neurons, name)):
        raise ValueError(f"{name}: STDP is not implemented on a delayed connection")
    if conn.source_threshold is not None or conn.destination_threshold is not None:
        raise ValueError(f"{name}: STDP is not implemented with NIR thresholds")


def stdp_trace_lines(
    name: str, trace: str, spike: str, rule: QuantisedSTDP, acc_width: int, data_width: int
) -> list[str]:
    """Return the wires of one trace's next value: ``sat((x * D) >>> f + spike · one)``."""
    wide = f"{{{{(ACC_WIDTH - DATA_WIDTH){{{trace}[DATA_WIDTH - 1]}}}}, {trace}}}"
    return [
        f"    wire signed [ACC_WIDTH - 1:0] {name}_raw = "
        f"(({wide} * {_signed_hex(rule.decay, acc_width)}) >>> {rule.fraction})"
        f" + ({spike} ? {_signed_hex(rule.one, acc_width)} : {acc_width}'sd0);",
        f"    wire signed [DATA_WIDTH - 1:0] {name} = sat_acc({name}_raw);",
    ]


def stdp_weight_lines(
    name: str,
    weight: str,
    pre_spike: str,
    post_spike: str,
    pre_next: str,
    post_next: str,
    rule: QuantisedSTDP,
    acc_width: int,
) -> list[str]:
    """Return the wires of one synapse's next weight, clipped to ``[w_min, w_max]``."""

    def wide(signal: str) -> str:
        return f"{{{{(ACC_WIDTH - DATA_WIDTH){{{signal}[DATA_WIDTH - 1]}}}}, {signal}}}"

    zero = f"{acc_width}'sd0"
    depress = f"(({_signed_hex(rule.a_minus, acc_width)} * {wide(post_next)}) >>> {rule.fraction})"
    potentiate = f"(({_signed_hex(rule.a_plus, acc_width)} * {wide(pre_next)}) >>> {rule.fraction})"
    w_min = _signed_hex(rule.w_min, acc_width)
    w_max = _signed_hex(rule.w_max, acc_width)
    return [
        f"    wire signed [ACC_WIDTH - 1:0] {name}_raw = {wide(weight)}"
        f" - ({pre_spike} ? {depress} : {zero}) + ({post_spike} ? {potentiate} : {zero});",
        f"    wire signed [ACC_WIDTH - 1:0] {name}_clip = ({name}_raw < {w_min}) ? {w_min} : "
        f"(({name}_raw > {w_max}) ? {w_max} : {name}_raw);",
        f"    wire signed [DATA_WIDTH - 1:0] {name} = {name}_clip[DATA_WIDTH - 1:0];",
    ]


def build_direct_interconnect(
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
    """Generate direct-wired per-neuron top-level interconnect.

    Every neuron gets its own instance of the type-specific module.  NIR
    affine weights are emitted explicitly in fixed-point arithmetic:

    * external analogue inputs use ``(input * weight) >>> fraction``;
    * analogue source populations use ``(v_out * weight) >>> fraction``;
    * spiking source populations contribute their fixed-point weight on
      spikes and zero otherwise;
    * graded spiking source populations (a template with ``spike_value``)
      contribute ``(spike_value_out * weight) >>> fraction``, the value their
      module emits on a spike and zero otherwise, so a value of 1.0 is the
      binary case; a source threshold compares that value;
    * all fan-in terms and biases accumulate in a widened signed accumulator
      before saturation back to the neuron input Q-format.

    Parameters
    ----------
    module_name : str
        Top-level module name.
    qgraph : QuantisedGraph
        Quantised graph.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits for fixed-point multiply downshift.
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
        If the graph or hierarchy metadata cannot be represented by the direct
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

    graded_pops = {pop.name for pop in pops if population_graded(pop)}
    # Plastic connections (trace STDP): global index and the rule's words.
    plastic: dict[int, tuple[int, QuantisedSTDP]] = {}
    for g, conn in enumerate(conns):
        if conn.plasticity is not None:
            check_plastic_connection(conn, pop_by_name, graded_pops)
            plastic[id(conn)] = (g, conn.plasticity.quantised(data_width, fraction))
    max_terms = external_width if pops else 1
    delayed_source_depths: dict[tuple[str, int], int] = {}
    delay_vectors: dict[int, DelayVector] = {}
    for conn in conns:
        weights = np.asarray(conn.weights)
        src_pop = pop_by_name.get(conn.src)
        expected_src = weights.shape[1]
        delay_vector = _normalise_connection_delay_steps(
            getattr(conn, "delay_steps", 0),
            expected_src,
            f"Connection {conn.src}->{conn.dst}",
        )
        delay_vectors[id(conn)] = delay_vector
        if any(delay_vector) and src_pop is None:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} is delayed but does not originate from "
                "a neuron population"
            )
        if any(delay_vector) and src_pop is not None:
            for src_idx in range(src_pop.n_neurons):
                delay_steps = delay_vector[src_idx]
                if delay_steps <= 0:
                    continue  # an undelayed column needs no register chain
                key = (src_pop.name, src_idx)
                delayed_source_depths[key] = max(delayed_source_depths.get(key, 0), delay_steps)
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

    lines = [
        f"// Auto-generated top-level network: {safe_module}",
        "// SC-NeuroCore NIR → FPGA compiler",
        f"// Interconnect: exact direct wiring ({qgraph.total_neurons} neurons)",
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

    first_neuron = 0
    for pop in pops:
        mod = sanitize_ident(f"sc_nir_{pop.neuron_type}", context="module name")
        lines.append(
            f"    // Population {pop_index[pop.name]}: {pop.name} "
            f"({pop.neuron_type} x {pop.n_neurons})"
        )
        draws = population_draws(pop)
        graded = pop.name in graded_pops
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            stream = population_stream(pop, first_neuron + neuron_idx) if draws else None
            override = _neuron_param_override(
                pop, neuron_idx, type_defaults, data_width, rng_stream=stream
            )
            lines.extend(
                [
                    f"    wire signed [DATA_WIDTH - 1:0] {prefix}_I;",
                    f"    wire {prefix}_spike;",
                    f"    wire signed [DATA_WIDTH - 1:0] {prefix}_v;",
                    *([f"    wire signed [DATA_WIDTH - 1:0] {prefix}_val;"] if graded else []),
                    f"    {mod}{override} {prefix}_inst (",
                    "        .clk(clk),",
                    "        .rst_n(rst_n),",
                    f"        .I_t({prefix}_I),",
                    f"        .spike_out({prefix}_spike),",
                    *([f"        .spike_value_out({prefix}_val),"] if graded else []),
                    f"        .v_out({prefix}_v)",
                    "    );",
                    "",
                ]
            )
        first_neuron += pop.n_neurons

    if delayed_source_depths:
        lines.append(
            "    // Delayed source register chains for recurrent and explicit NIR Delay paths"
        )
        for pop_name, neuron_idx in sorted(
            delayed_source_depths,
            key=lambda item: (pop_index[item[0]], item[1]),
        ):
            pop = pop_by_name[pop_name]
            prefix = neuron_prefix(pop, neuron_idx)
            for step in range(1, delayed_source_depths[(pop_name, neuron_idx)] + 1):
                lines.append(f"    reg {prefix}_spike_d{step};")
                lines.append(f"    reg signed [DATA_WIDTH - 1:0] {prefix}_v_d{step};")
                if pop_name in graded_pops:
                    lines.append(f"    reg signed [DATA_WIDTH - 1:0] {prefix}_val_d{step};")
        lines.extend(
            [
                "    always @(posedge clk or negedge rst_n) begin",
                "        if (!rst_n) begin",
            ]
        )
        for pop_name, neuron_idx in sorted(
            delayed_source_depths,
            key=lambda item: (pop_index[item[0]], item[1]),
        ):
            pop = pop_by_name[pop_name]
            prefix = neuron_prefix(pop, neuron_idx)
            for step in range(1, delayed_source_depths[(pop_name, neuron_idx)] + 1):
                lines.append(f"            {prefix}_spike_d{step} <= 1'b0;")
                lines.append(f"            {prefix}_v_d{step} <= {data_width}'sd0;")
                if pop_name in graded_pops:
                    lines.append(f"            {prefix}_val_d{step} <= {data_width}'sd0;")
        lines.extend(
            [
                "        end else if (en) begin",
            ]
        )
        for pop_name, neuron_idx in sorted(
            delayed_source_depths,
            key=lambda item: (pop_index[item[0]], item[1]),
        ):
            pop = pop_by_name[pop_name]
            prefix = neuron_prefix(pop, neuron_idx)
            depth = delayed_source_depths[(pop_name, neuron_idx)]
            lines.append(f"            {prefix}_spike_d1 <= {prefix}_spike;")
            lines.append(f"            {prefix}_v_d1 <= {prefix}_v;")
            if pop_name in graded_pops:
                lines.append(f"            {prefix}_val_d1 <= {prefix}_val;")
            for step in range(2, depth + 1):
                lines.append(f"            {prefix}_spike_d{step} <= {prefix}_spike_d{step - 1};")
                lines.append(f"            {prefix}_v_d{step} <= {prefix}_v_d{step - 1};")
                if pop_name in graded_pops:
                    lines.append(f"            {prefix}_val_d{step} <= {prefix}_val_d{step - 1};")
        lines.extend(["        end", "    end", ""])

    if plastic:
        lines.extend(_stdp_block(conns, plastic, pop_by_name, neuron_prefix, acc_width, data_width))

    lines.append("    // Weighted fixed-point input accumulation")
    for pop in pops:
        feeding = [conn for conn in conns if conn.dst == pop.name]
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            term_names: list[str] = []
            term_defs: list[str] = []

            if not feeding and pop_index[pop.name] == 0 and neuron_idx < external_width:
                term_names.append(
                    f"{{{{(ACC_WIDTH - DATA_WIDTH){{ext_input_{neuron_idx}"
                    f"[DATA_WIDTH - 1]}}}}, ext_input_{neuron_idx}}}"
                )

            for conn_idx, conn in enumerate(feeding):
                weights = np.asarray(conn.weights, dtype=np.int64)
                conn_terms: list[str] = []
                if conn.bias is not None:
                    bias = int(np.asarray(conn.bias, dtype=np.int64).reshape(-1)[neuron_idx])
                    conn_terms.append(_signed_hex(bias, acc_width))

                src_pop = pop_by_name.get(conn.src)
                source_thresholds = (
                    None
                    if conn.source_threshold is None
                    else np.asarray(conn.source_threshold, dtype=np.int64).reshape(-1)
                )
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
                        if source_thresholds is not None:
                            threshold = int(source_thresholds[src_idx])
                            conn_terms.append(
                                f"(ext_input_{external_idx} > {_signed_hex(threshold, data_width)} "
                                f"? {_signed_hex(weight, acc_width)} : {acc_width}'sd0)"
                            )
                        else:
                            mul = f"{term_base}_mul"
                            term = f"{term_base}_term"
                            term_defs.extend(
                                [
                                    f"    wire signed [{product_width - 1}:0] {mul} = "
                                    f"ext_input_{external_idx} * {weight_expr};",
                                    f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};",
                                ]
                            )
                            conn_terms.append(term)
                        continue

                    src_prefix = neuron_prefix(src_pop, src_idx)
                    delay_steps = delay_vectors[id(conn)][src_idx]
                    graded_src = src_pop.name in graded_pops
                    if _connection_sources_are_analogue(src_pop) or graded_src:
                        # A voltage (analogue source) or the graded spike's value: both
                        # multiply the weight, or are compared under a source threshold.
                        signal = "val" if graded_src else "v"
                        src_value = (
                            f"{src_prefix}_{signal}_d{delay_steps}"
                            if delay_steps
                            else f"{src_prefix}_{signal}"
                        )
                        if source_thresholds is not None:
                            threshold = int(source_thresholds[src_idx])
                            conn_terms.append(
                                f"({src_value} > {_signed_hex(threshold, data_width)} "
                                f"? {_signed_hex(weight, acc_width)} : {acc_width}'sd0)"
                            )
                        else:
                            mul = f"{term_base}_mul"
                            term = f"{term_base}_term"
                            term_defs.extend(
                                [
                                    f"    wire signed [{product_width - 1}:0] {mul} = "
                                    f"{src_value} * {weight_expr};",
                                    f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};",
                                ]
                            )
                            conn_terms.append(term)
                    else:
                        src_spike = (
                            f"{src_prefix}_spike_d{delay_steps}"
                            if delay_steps
                            else f"{src_prefix}_spike"
                        )
                        if source_thresholds is not None:
                            threshold = int(source_thresholds[src_idx])
                            spike_value = f"({src_spike} ? {_signed_hex(1 << fraction, data_width)} : {data_width}'sd0)"
                            conn_terms.append(
                                f"({spike_value} > {_signed_hex(threshold, data_width)} "
                                f"? {_signed_hex(weight, acc_width)} : {acc_width}'sd0)"
                            )
                        elif id(conn) in plastic:
                            # The weight this step uses is the one the previous step's
                            # spikes just updated: the next value of the weight register.
                            g = plastic[id(conn)][0]
                            w_next = f"stdp{g}_w{neuron_idx}_{src_idx}_next"
                            conn_terms.append(
                                f"({src_spike} ? {{{{(ACC_WIDTH - DATA_WIDTH){{{w_next}"
                                f"[DATA_WIDTH - 1]}}}}, {w_next}}} : {acc_width}'sd0)"
                            )
                        else:
                            conn_terms.append(
                                f"({src_spike} ? {_signed_hex(weight, acc_width)} : {acc_width}'sd0)"
                            )

                if conn.destination_threshold is not None:
                    threshold = int(
                        np.asarray(conn.destination_threshold, dtype=np.int64).reshape(-1)[
                            neuron_idx
                        ]
                    )
                    raw_name = f"{prefix}_c{conn_idx}_raw"
                    out_name = f"{prefix}_c{conn_idx}_threshold_out"
                    raw_expr = " + ".join(conn_terms) if conn_terms else f"{acc_width}'sd0"
                    term_defs.extend(
                        [
                            f"    wire signed [ACC_WIDTH - 1:0] {raw_name} = {raw_expr};",
                            f"    wire {out_name} = ({raw_name} > {_signed_hex(threshold, acc_width)});",
                        ]
                    )
                    term_names.append(
                        f"({out_name} ? {_signed_hex(1 << fraction, acc_width)} : {acc_width}'sd0)"
                    )
                else:
                    term_names.extend(conn_terms)

            lines.extend(term_defs)
            acc_expr = " + ".join(term_names) if term_names else f"{acc_width}'sd0"
            lines.append(f"    wire signed [ACC_WIDTH - 1:0] {prefix}_I_acc = {acc_expr};")
            lines.append(
                f"    assign {prefix}_I = en ? sat_acc({prefix}_I_acc) : {data_width}'sd0;"
            )

    lines.append("")
    for pop in pops:
        for neuron_idx in range(pop.n_neurons):
            prefix = neuron_prefix(pop, neuron_idx)
            bus_idx = pop_offsets[pop.name] + neuron_idx
            lines.append(f"    assign spike_bus[{bus_idx}] = {prefix}_spike;")

    lines.extend(["", "endmodule", ""])
    return "\n".join(lines)


def _stdp_block(
    conns: Sequence[ConnectionSpec],
    plastic: dict[int, tuple[int, QuantisedSTDP]],
    pop_by_name: dict[str, NeuronSpec],
    neuron_prefix: Callable[[NeuronSpec, int], str],
    acc_width: int,
    data_width: int,
) -> list[str]:
    """Emit the traces and weights of every plastic connection of the direct top.

    Each trace and weight register holds the value after the previous step; its
    ``_next`` wire applies the spikes now on the neurons' outputs (this step's),
    and the input currents read the ``_next`` weights, so the step after a spike
    already uses the updated weight, as the folded interconnect and
    :func:`sc_neurocore.nir_bridge.stdp.stdp_tick` do.
    """
    lines = ["    // Trace-based STDP: traces and weights, updated every step"]
    resets: list[str] = []
    latches: list[str] = []
    for conn in conns:
        if id(conn) not in plastic:
            continue
        g, rule = plastic[id(conn)]
        src, dst = pop_by_name[conn.src], pop_by_name[conn.dst]
        weights = np.asarray(conn.weights, dtype=np.int64)
        for role, pop in (("pre", src), ("post", dst)):
            for k in range(pop.n_neurons):
                reg = f"stdp{g}_{role}{k}"
                spike = f"{neuron_prefix(pop, k)}_spike"
                lines.append(f"    reg signed [DATA_WIDTH - 1:0] {reg};")
                lines.extend(
                    stdp_trace_lines(f"{reg}_next", reg, spike, rule, acc_width, data_width)
                )
                resets.append(f"            {reg} <= {data_width}'sd0;")
                latches.append(f"            {reg} <= {reg}_next;")
        for j in range(dst.n_neurons):
            for i in range(src.n_neurons):
                if int(weights[j, i]) == 0:
                    continue  # not a synapse: never learns
                reg = f"stdp{g}_w{j}_{i}"
                lines.append(f"    reg signed [DATA_WIDTH - 1:0] {reg};")
                lines.extend(
                    stdp_weight_lines(
                        f"{reg}_next",
                        reg,
                        f"{neuron_prefix(src, i)}_spike",
                        f"{neuron_prefix(dst, j)}_spike",
                        f"stdp{g}_pre{i}_next",
                        f"stdp{g}_post{j}_next",
                        rule,
                        acc_width,
                    )
                )
                resets.append(
                    f"            {reg} <= {_signed_hex(int(weights[j, i]), data_width)};"
                )
                latches.append(f"            {reg} <= {reg}_next;")
    lines.extend(
        [
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            *resets,
            "        end else if (en) begin",
            *latches,
            "        end",
            "    end",
            "",
        ]
    )
    return lines
