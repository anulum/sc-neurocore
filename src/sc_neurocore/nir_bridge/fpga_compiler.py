# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler

"""Compile a NeuronGraph to synthesisable Verilog RTL.

End-to-end pipeline: NeuronGraph → quantisation → per-type neuron modules
+ weight ROM artefact + top-level interconnect.

Interconnect Strategy
~~~~~~~~~~~~~~~~~~~~~
Small networks emit an explicit per-neuron direct interconnect.  Larger
networks emit a weighted address-event fan-out block for spike-producing source
populations, while external analogue inputs and analogue source populations
remain exact fixed-point multiply-accumulate terms.  Both paths preserve the
NIR weighted affine semantics.

Usage
~~~~~
::

    from sc_neurocore.nir_bridge.neuron_graph import from_scnetwork
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    graph = from_scnetwork(network)
    result = compile_network_to_fpga(graph, module_name="my_snn")
    # result.top_module   → top-level Verilog
    # result.neuron_modules → per-type Verilog dict
    # result.weight_rom   → weight ROM Verilog
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..hdl_gen._ident import sanitize_ident
from ..compiler.equation_compiler import Q88, compile_to_verilog
from ..ir.scnir_convert import SCNIRConversionConfig, build_scnir_from_neuron_graph
from ..ir.scnir_hdl import (
    SCNIRHDLSourceManifestEntry,
    build_scnir_source_bundle,
)
from ..ir.scnir_schema import SCNIRDocument
from ..neurons.equation_builder import from_equations
from .neuron_graph import NeuronGraph, NeuronSpec
from .quantise_params import QuantisedGraph, quantise_graph

logger = logging.getLogger(__name__)

DelayVector = tuple[int, ...]

# Threshold above which the compiler records that event-bus RTL would be useful.
# The default emitter still uses exact direct wiring until weighted routing is
# implemented and verified.
_AER_THRESHOLD = 64
_MAX_SYNTHESISABLE_DELAY_STEPS = 1024


# ═══════════════════════════════════════════════════════════════════════
# Canonical ODE Templates
# ═══════════════════════════════════════════════════════════════════════

_NEURON_TEMPLATES: dict[str, dict[str, Any]] = {
    "lif": {
        "equations": ["dv/dt = -(v - v_leak) / tau + I * r / tau"],
        "threshold": "v > v_threshold",
        "reset": "v = v_reset",
        "default_params": {
            "tau": 20.0,
            "r": 1.0,
            "v_leak": 0.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
        },
    },
    "if": {
        "equations": ["dv/dt = I * r"],
        "threshold": "v > v_threshold",
        "reset": "v = v_reset",
        "default_params": {"r": 1.0, "v_threshold": 1.0, "v_reset": 0.0},
    },
    "li": {
        "equations": ["dv/dt = -(v - v_leak) / tau + I * r / tau"],
        "threshold": None,
        "reset": None,
        "default_params": {"tau": 20.0, "r": 1.0, "v_leak": 0.0},
    },
    "cuba_lif": {
        "equations": [
            "di_syn/dt = -i_syn / tau_syn + I * w_in",
            "dv/dt = -(v - v_leak) / tau_mem + i_syn * r / tau_mem",
        ],
        "threshold": "v > v_threshold",
        "reset": "v = v_reset",
        "default_params": {
            "tau_syn": 5.0,
            "tau_mem": 20.0,
            "r": 1.0,
            "v_leak": 0.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "w_in": 1.0,
        },
    },
    "cuba_li": {
        "equations": [
            "di_syn/dt = -i_syn / tau_syn + I * w_in",
            "dv/dt = -(v - v_leak) / tau_mem + i_syn * r / tau_mem",
        ],
        "threshold": None,
        "reset": None,
        "default_params": {"tau_syn": 5.0, "tau_mem": 20.0, "r": 1.0, "v_leak": 0.0, "w_in": 1.0},
    },
    "integrator": {
        "equations": ["dv/dt = I * r"],
        "threshold": None,
        "reset": None,
        "default_params": {"r": 1.0},
    },
}


def _require_homogeneous_param(values: np.ndarray, label: str) -> float:
    """Return the scalar value of a per-neuron parameter or fail closed."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty parameter array for {label}")
    if not np.allclose(arr, arr[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Heterogeneous per-neuron parameter {label} cannot be represented "
            "by a shared RTL neuron module"
        )
    return float(arr[0])


def _resolved_population_params(neuron_type: str, pop: NeuronSpec) -> dict[str, float]:
    """Resolve population parameters without averaging per-neuron values."""
    template = _NEURON_TEMPLATES.get(neuron_type)
    if template is None:
        raise ValueError(f"No ODE template for neuron type: {neuron_type!r}")

    params: dict[str, float] = {
        name: float(value) for name, value in template["default_params"].items()
    }
    for pname, pval in pop.params.items():
        if pname not in params:
            raise ValueError(
                f"Parameter {pop.name}.{pname} is not supported by the "
                f"{neuron_type!r} FPGA template"
            )
        params[pname] = _require_homogeneous_param(pval, f"{pop.name}.{pname}")
    return params


def _population_module_signature(pop: NeuronSpec) -> tuple[Any, ...]:
    """Build the exact parameter signature for shared module reuse."""
    params = _resolved_population_params(pop.neuron_type, pop)
    return (
        pop.neuron_type,
        float(pop.dt),
        tuple((name, params[name]) for name in sorted(params)),
    )


def _signed_hex(value: int, width: int) -> str:
    """Emit a width-limited signed Verilog literal."""
    if width < 1:
        raise ValueError("Verilog literal width must be positive")
    n_hex = max(1, (width + 3) // 4)
    return f"{width}'sh{int(value) & ((1 << width) - 1):0{n_hex}x}"


def _ceil_log2_at_least_one(value: int) -> int:
    """Return ceil(log2(value)) with a lower bound of 1."""
    return max(1, math.ceil(math.log2(max(1, value))))


def _connection_sources_are_analogue(pop: NeuronSpec) -> bool:
    """Whether a population output should be routed as an analogue state."""
    return pop.neuron_type in {"li", "cuba_li", "integrator"}


def _connection_has_thresholds(conn: Any) -> bool:
    """Whether a connection carries explicit NIR Threshold metadata."""

    return (
        getattr(conn, "source_threshold", None) is not None
        or getattr(conn, "destination_threshold", None) is not None
    )


def _normalise_connection_delay_steps(
    delay_steps: Any,
    source_width: int,
    label: str,
) -> DelayVector:
    """Return one validated delay value per source column."""

    if source_width <= 0:
        raise ValueError(f"{label} source width must be positive")
    if isinstance(delay_steps, int) and not isinstance(delay_steps, bool):
        value = delay_steps
        if value < 0:
            raise ValueError(f"{label} delay_steps must be non-negative")
        if value > _MAX_SYNTHESISABLE_DELAY_STEPS:
            raise ValueError(
                f"{label} has delay_steps={value}, above "
                f"the synthesis guard {_MAX_SYNTHESISABLE_DELAY_STEPS}"
            )
        return tuple(value for _ in range(source_width))

    raw = np.atleast_1d(np.asarray(delay_steps, dtype=np.int64)).reshape(-1)
    if raw.size != source_width:
        raise ValueError(
            f"{label} delay_steps vector length {raw.size} does not match "
            f"source width {source_width}"
        )
    if np.any(raw < 0):
        raise ValueError(f"{label} delay_steps must be non-negative")
    max_delay = int(np.max(raw)) if raw.size else 0
    if max_delay > _MAX_SYNTHESISABLE_DELAY_STEPS:
        raise ValueError(
            f"{label} has delay_steps={max_delay}, above "
            f"the synthesis guard {_MAX_SYNTHESISABLE_DELAY_STEPS}"
        )
    return tuple(int(value) for value in raw)


# ═══════════════════════════════════════════════════════════════════════
# Result Container
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class NetworkCompilationResult:
    """All artefacts from a network-level FPGA compilation.

    Attributes
    ----------
    neuron_modules : dict[str, str]
        Mapping from neuron type to Verilog source.
    weight_rom : str
        Weight ROM Verilog source.
    top_module : str
        Top-level interconnect Verilog source.
    module_name : str
        Top-level module name.
    total_neurons : int
        Total neuron count.
    total_synapses : int
        Total synapse count.
    q_format : str
        Q-format label (e.g. ``"Q8.8"``).
    interconnect : str
        ``"direct"`` or ``"aer"``.
    warnings : list[str]
        Quantisation and compilation warnings.
    scnir_document : SCNIRDocument
        SC-aware metadata document consumed by the compilation artefacts.
    scnir_source_modules : dict[str, str]
        Concrete stochastic source HDL modules keyed by Verilog module name.
    scnir_source_manifest : tuple[SCNIRHDLSourceManifestEntry, ...]
        Deterministic manifest mapping SC-NIR streams to source modules.
    """

    neuron_modules: dict[str, str]
    weight_rom: str
    top_module: str
    module_name: str
    total_neurons: int
    total_synapses: int
    q_format: str
    interconnect: str
    scnir_document: SCNIRDocument
    scnir_source_modules: dict[str, str]
    scnir_source_manifest: tuple[SCNIRHDLSourceManifestEntry, ...]
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Neuron Module Generation
# ═══════════════════════════════════════════════════════════════════════


def _build_neuron_module(
    neuron_type: str,
    pop: NeuronSpec,
    *,
    data_width: int = 16,
    fraction: int = 8,
) -> str:
    """Build a Verilog module for one canonical neuron type.

    Uses the existing ``equation_compiler.compile_to_verilog()`` with
    canonical ODE templates.

    Parameters
    ----------
    neuron_type : str
        Canonical neuron type (``"lif"``, ``"if"``, etc.).
    pop : NeuronSpec
        Representative population (for parameter defaults).
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.

    Returns
    -------
    str
        Synthesisable Verilog module source.
    """
    template = _NEURON_TEMPLATES.get(neuron_type)
    if template is None:
        raise ValueError(f"No ODE template for neuron type: {neuron_type!r}")

    # Shared modules are only emitted when every per-neuron parameter is
    # identical.  Heterogeneous populations need generated per-neuron modules
    # and are rejected rather than averaged.
    params = _resolved_population_params(neuron_type, pop)

    # Build initial state
    init: dict[str, float] = {}
    for eq_str in template["equations"]:
        var_name = eq_str.split("/")[0].replace("d", "", 1).strip()
        if var_name == "v":
            init["v"] = params.get("v_leak", 0.0)
        else:
            init[var_name] = 0.0

    module_name = f"sc_nir_{neuron_type}"

    neuron = from_equations(
        *template["equations"],
        threshold=template["threshold"],
        reset=template["reset"],
        params=params,
        init=init,
        dt=pop.dt,
    )

    return compile_to_verilog(
        neuron,
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )


# ═══════════════════════════════════════════════════════════════════════
# Weight ROM Generation
# ═══════════════════════════════════════════════════════════════════════


def _build_weight_rom(
    qgraph: QuantisedGraph,
    *,
    data_width: int = 16,
) -> str:
    """Generate a combined weight ROM for all connections.

    All connection weight matrices are flattened into a single ROM
    addressed by a global index.  Each connection gets a base address
    offset.

    Parameters
    ----------
    qgraph : QuantisedGraph
        Quantised graph with integer weight matrices.
    data_width : int
        Weight data width.

    Returns
    -------
    str
        Verilog weight ROM module source.
    """
    if not qgraph.connections:
        # Empty ROM
        return (
            "// Auto-generated weight ROM (empty — no connections)\n"
            "module sc_nir_weight_rom (\n"
            "    input  wire [0:0] addr,\n"
            f"    output wire signed [{data_width - 1}:0] data\n"
            ");\n"
            f"    assign data = {data_width}'sd0;\n"
            "endmodule\n"
        )

    # Flatten all weights into a single list
    all_weights: list[int] = []
    conn_offsets: list[tuple[str, str, int, int]] = []  # src, dst, offset, count

    for conn in qgraph.connections:
        offset = len(all_weights)
        flat = conn.weights.flatten().tolist()
        count = len(flat)
        all_weights.extend(int(w) for w in flat)
        conn_offsets.append((conn.src, conn.dst, offset, count))

    total = len(all_weights)
    addr_w = max(1, (total - 1).bit_length())

    lines = [
        "// Auto-generated combined weight ROM",
        "// SC-NeuroCore NIR → FPGA compiler",
        f"// Total entries: {total}, Address width: {addr_w}",
        "",
    ]

    # Connection offset comments
    for src, dst, offset, count in conn_offsets:
        lines.append(f"// {src} → {dst}: offset={offset}, count={count}")
    lines.append("")

    lines.extend(
        [
            "module sc_nir_weight_rom (",
            f"    input  wire [{addr_w - 1}:0] addr,",
            f"    output reg  signed [{data_width - 1}:0] data",
            ");",
            "",
            "    always @(*) begin",
            "        case (addr)",
        ]
    )

    mask = (1 << data_width) - 1
    for i, w in enumerate(all_weights):
        val = w & mask
        lines.append(f"            {addr_w}'d{i}: data = {data_width}'sh{val:0{data_width // 4}x};")

    lines.extend(
        [
            f"            default: data = {data_width}'sd0;",
            "        endcase",
            "    end",
            "",
            "endmodule",
        ]
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Top-Level Interconnect
# ═══════════════════════════════════════════════════════════════════════


def _build_top_direct(
    module_name: str,
    qgraph: QuantisedGraph,
    *,
    data_width: int = 16,
    fraction: int = 8,
    bitstream_length: int = 256,
    scnir_stream_count: int = 0,
    scnir_source_module_count: int = 0,
) -> str:
    """Generate direct-wired per-neuron top-level interconnect.

    Every neuron gets its own instance of the type-specific module.  NIR
    affine weights are emitted explicitly in fixed-point arithmetic:

    * external analogue inputs use ``(input * weight) >>> fraction``;
    * analogue source populations use ``(v_out * weight) >>> fraction``;
    * spiking source populations contribute their fixed-point weight on
      spikes and zero otherwise;
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

    Returns
    -------
    str
        Verilog top-level module source.
    """
    pops = qgraph.populations
    conns = qgraph.connections
    safe_module = sanitize_ident(module_name, context="module name")
    pop_by_name = {pop.name: pop for pop in pops}
    pop_index = {pop.name: idx for idx, pop in enumerate(pops)}
    pop_offsets: dict[str, int] = {}

    offset = 0
    for pop in pops:
        pop_offsets[pop.name] = offset
        offset += pop.n_neurons

    external_connections = [conn for conn in conns if conn.src not in pop_by_name]
    if external_connections:
        external_width = max(int(conn.weights.shape[1]) for conn in external_connections)
    elif pops:
        external_width = max(1, pops[0].n_neurons)
    else:
        external_width = 1

    max_terms = external_width if pops else 1
    delayed_source_depths: dict[tuple[str, int], int] = {}
    delay_vectors: dict[int, DelayVector] = {}
    for conn in conns:
        weights = np.asarray(conn.weights)
        if weights.ndim != 2:
            raise ValueError(f"Connection {conn.src}->{conn.dst} weights must be a 2-D matrix")
        dst_pop = pop_by_name.get(conn.dst)
        if dst_pop is None:
            raise ValueError(f"Connection destination {conn.dst!r} is not a neuron population")
        if weights.shape[0] != dst_pop.n_neurons:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} has {weights.shape[0]} "
                f"destination rows for {dst_pop.n_neurons} destination neurons"
            )
        src_pop = pop_by_name.get(conn.src)
        expected_src = src_pop.n_neurons if src_pop is not None else external_width
        if weights.shape[1] != expected_src:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} has {weights.shape[1]} "
                f"source columns for {expected_src} source signals"
            )
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
                key = (src_pop.name, src_idx)
                delayed_source_depths[key] = max(delayed_source_depths.get(key, 0), delay_steps)
        if conn.bias is not None and np.asarray(conn.bias).reshape(-1).size != dst_pop.n_neurons:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} bias length does not match "
                f"{dst_pop.n_neurons} destination neurons"
            )
        if conn.source_threshold is not None:
            source_threshold = np.asarray(conn.source_threshold).reshape(-1)
            if source_threshold.size != weights.shape[1]:
                raise ValueError(
                    f"Connection {conn.src}->{conn.dst} source_threshold length "
                    f"does not match {weights.shape[1]} source columns"
                )
        if conn.destination_threshold is not None:
            destination_threshold = np.asarray(conn.destination_threshold).reshape(-1)
            if destination_threshold.size != dst_pop.n_neurons:
                raise ValueError(
                    f"Connection {conn.src}->{conn.dst} destination_threshold length "
                    f"does not match {dst_pop.n_neurons} destination neurons"
                )
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
                    f"    {mod} {prefix}_inst (",
                    "        .clk(clk),",
                    "        .rst_n(rst_n),",
                    f"        .I_t({prefix}_I),",
                    f"        .spike_out({prefix}_spike),",
                    f"        .v_out({prefix}_v)",
                    "    );",
                    "",
                ]
            )

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
            for step in range(2, depth + 1):
                lines.append(f"            {prefix}_spike_d{step} <= {prefix}_spike_d{step - 1};")
                lines.append(f"            {prefix}_v_d{step} <= {prefix}_v_d{step - 1};")
        lines.extend(["        end", "    end", ""])

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

                    if src_pop is None:
                        if source_thresholds is not None:
                            threshold = int(source_thresholds[src_idx])
                            conn_terms.append(
                                f"(ext_input_{src_idx} > {_signed_hex(threshold, data_width)} "
                                f"? {_signed_hex(weight, acc_width)} : {acc_width}'sd0)"
                            )
                        else:
                            mul = f"{term_base}_mul"
                            term = f"{term_base}_term"
                            term_defs.extend(
                                [
                                    f"    wire signed [{product_width - 1}:0] {mul} = "
                                    f"ext_input_{src_idx} * {_signed_hex(weight, data_width)};",
                                    f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};",
                                ]
                            )
                            conn_terms.append(term)
                        continue

                    src_prefix = neuron_prefix(src_pop, src_idx)
                    delay_steps = delay_vectors[id(conn)][src_idx]
                    if _connection_sources_are_analogue(src_pop):
                        src_value = (
                            f"{src_prefix}_v_d{delay_steps}" if delay_steps else f"{src_prefix}_v"
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
                                    f"{src_value} * {_signed_hex(weight, data_width)};",
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


def _build_top_aer(
    module_name: str,
    qgraph: QuantisedGraph,
    *,
    data_width: int = 16,
    fraction: int = 8,
    bitstream_length: int = 256,
    scnir_stream_count: int = 0,
    scnir_source_module_count: int = 0,
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

    Returns
    -------
    str
        Verilog top-level module source.
    """
    pops = qgraph.populations
    conns = qgraph.connections
    safe_module = sanitize_ident(module_name, context="module name")
    pop_by_name = {pop.name: pop for pop in pops}
    pop_index = {pop.name: idx for idx, pop in enumerate(pops)}
    pop_offsets: dict[str, int] = {}

    offset = 0
    for pop in pops:
        pop_offsets[pop.name] = offset
        offset += pop.n_neurons

    external_connections = [conn for conn in conns if conn.src not in pop_by_name]
    if external_connections:
        external_width = max(int(conn.weights.shape[1]) for conn in external_connections)
    elif pops:
        external_width = max(1, pops[0].n_neurons)
    else:
        external_width = 1

    max_terms = external_width if pops else 1
    for conn in conns:
        weights = np.asarray(conn.weights)
        if weights.ndim != 2:
            raise ValueError(f"Connection {conn.src}->{conn.dst} weights must be a 2-D matrix")
        dst_pop = pop_by_name.get(conn.dst)
        if dst_pop is None:
            raise ValueError(f"Connection destination {conn.dst!r} is not a neuron population")
        if weights.shape[0] != dst_pop.n_neurons:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} has {weights.shape[0]} "
                f"destination rows for {dst_pop.n_neurons} destination neurons"
            )
        src_pop = pop_by_name.get(conn.src)
        expected_src = src_pop.n_neurons if src_pop is not None else external_width
        if weights.shape[1] != expected_src:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} has {weights.shape[1]} "
                f"source columns for {expected_src} source signals"
            )
        if conn.bias is not None and np.asarray(conn.bias).reshape(-1).size != dst_pop.n_neurons:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} bias length does not match "
                f"{dst_pop.n_neurons} destination neurons"
            )
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
                    f"    {mod} {prefix}_inst (",
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

                    if src_pop is None:
                        mul = f"{term_base}_mul"
                        term = f"{term_base}_term"
                        term_defs.extend(
                            [
                                f"    wire signed [{product_width - 1}:0] {mul} = "
                                f"ext_input_{src_idx} * {_signed_hex(weight, data_width)};",
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
                                f"{src_prefix}_v * {_signed_hex(weight, data_width)};",
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


# ═══════════════════════════════════════════════════════════════════════
# Main Compiler Entry Point
# ═══════════════════════════════════════════════════════════════════════


def compile_network_to_fpga(
    graph: NeuronGraph,
    *,
    module_name: str = "sc_nir_network",
    data_width: int = 16,
    fraction: int = 8,
    bitstream_length: int = 256,
    source_kind: Literal["lfsr", "sobol"] = "lfsr",
    base_seed: int = 1,
    target: str = "artix7",
) -> NetworkCompilationResult:
    """Compile a NeuronGraph to synthesisable Verilog RTL.

    End-to-end pipeline:

    1. Quantise all parameters to the target Q-format.
    2. Generate one Verilog module per unique neuron type.
    3. Generate a combined weight ROM.
    4. Generate a top-level interconnect module (direct or AER).

    Parameters
    ----------
    graph : NeuronGraph
        Network description (from ``from_scnetwork()``).
    module_name : str
        Top-level Verilog module name.
    data_width : int
        Fixed-point total width (16 for Q8.8, 32 for Q16.16).
    fraction : int
        Fractional bits.
    bitstream_length : int
        SC-NIR bitstream length metadata propagated into compilation artefacts.
    source_kind : {"lfsr", "sobol"}
        Hardware stochastic source family materialised from SC-NIR metadata.
    base_seed : int
        First deterministic source seed; stream index increments from this base.
    target : str
        FPGA target for resource estimation hints.

    Returns
    -------
    NetworkCompilationResult
        All generated Verilog sources and compilation metadata.

    Raises
    ------
    ValueError
        If the graph is empty or contains unsupported neuron types.
    """
    q = Q88(data_width=data_width, fraction=fraction)
    if source_kind == "lfsr" or source_kind == "sobol":
        resolved_source_kind: Literal["lfsr", "sobol"] = source_kind
    else:
        raise ValueError("source_kind must be 'lfsr' or 'sobol' for FPGA source emission")

    scnir_config = SCNIRConversionConfig(
        bitstream_length=bitstream_length,
        data_width=data_width,
        fraction=fraction,
        base_seed=base_seed,
        source_kind=resolved_source_kind,
    )
    scnir_document = build_scnir_from_neuron_graph(graph, config=scnir_config)
    scnir_source_bundle = build_scnir_source_bundle(scnir_document)
    warnings: list[str] = []

    # Step 1: Quantise
    qgraph = quantise_graph(graph, q)
    warnings.extend(qgraph.warnings)

    # Step 2: Generate per-type neuron modules (cached by exact parameter set)
    neuron_modules: dict[str, str] = {}
    type_representative: dict[str, NeuronSpec] = {}
    type_signature: dict[str, tuple[Any, ...]] = {}

    for pop in graph.populations:
        signature = _population_module_signature(pop)
        if pop.neuron_type not in type_representative:
            type_representative[pop.neuron_type] = pop
            type_signature[pop.neuron_type] = signature
        elif type_signature[pop.neuron_type] != signature:
            raise ValueError(
                f"Neuron type {pop.neuron_type!r} appears with different "
                "parameters across populations; per-population RTL modules are "
                "required before this can be compiled faithfully"
            )

    for ntype, rep_pop in type_representative.items():
        if ntype not in _NEURON_TEMPLATES:
            warnings.append(f"Unsupported neuron type '{ntype}' — skipping module generation")
            continue
        try:
            verilog = _build_neuron_module(
                ntype,
                rep_pop,
                data_width=data_width,
                fraction=fraction,
            )
            neuron_modules[ntype] = verilog
            logger.info("Generated Verilog for neuron type: %s", ntype)
        except (ValueError, KeyError) as exc:
            warnings.append(f"Failed to compile neuron type '{ntype}': {exc}")
            logger.error("Neuron compilation failed for %s: %s", ntype, exc)

    # Step 3: Weight ROM
    weight_rom = _build_weight_rom(qgraph, data_width=data_width)

    # Step 4: Top-level interconnect.  Small networks use explicit direct
    # wiring.  Larger networks use weighted address-event fan-out while
    # preserving dense affine accumulation semantics.
    total_neurons = graph.total_neurons
    interconnect = "direct"
    has_delayed_connections = any(
        any(
            _normalise_connection_delay_steps(
                getattr(conn, "delay_steps", 0),
                int(np.asarray(conn.weights).shape[1]),
                f"Connection {conn.src}->{conn.dst}",
            )
        )
        for conn in qgraph.connections
    )
    has_threshold_connections = any(_connection_has_thresholds(conn) for conn in qgraph.connections)
    if (
        total_neurons > _AER_THRESHOLD
        and not has_delayed_connections
        and not has_threshold_connections
    ):
        interconnect = "aer"
        top_module = _build_top_aer(
            module_name,
            qgraph,
            data_width=data_width,
            fraction=fraction,
            bitstream_length=bitstream_length,
            scnir_stream_count=len(scnir_document.streams),
            scnir_source_module_count=len(scnir_source_bundle.manifest),
        )
    else:
        if total_neurons > _AER_THRESHOLD and has_delayed_connections:
            warnings.append(
                "Using direct interconnect because delayed recurrent connections require "
                "registered one-step source semantics"
            )
        if total_neurons > _AER_THRESHOLD and has_threshold_connections:
            warnings.append(
                "Using direct interconnect because NIR Threshold transforms require exact "
                "fixed-point comparator semantics"
            )
        top_module = _build_top_direct(
            module_name,
            qgraph,
            data_width=data_width,
            fraction=fraction,
            bitstream_length=bitstream_length,
            scnir_stream_count=len(scnir_document.streams),
            scnir_source_module_count=len(scnir_source_bundle.manifest),
        )

    q_label = f"Q{data_width - fraction}.{fraction}"

    result = NetworkCompilationResult(
        neuron_modules=neuron_modules,
        weight_rom=weight_rom,
        top_module=top_module,
        module_name=module_name,
        total_neurons=total_neurons,
        total_synapses=graph.total_synapses,
        q_format=q_label,
        interconnect=interconnect,
        scnir_document=scnir_document,
        scnir_source_modules=dict(scnir_source_bundle.modules),
        scnir_source_manifest=scnir_source_bundle.manifest,
        warnings=warnings,
    )

    logger.info(
        "Network compilation complete: %s, %d neurons, %d synapses, %s interconnect, %d warnings",
        q_label,
        total_neurons,
        graph.total_synapses,
        interconnect,
        len(warnings),
    )

    return result
