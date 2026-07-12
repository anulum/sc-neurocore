# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Time-multiplexed FPGA interconnect and resource estimation."""

from typing import Sequence

import numpy as np

from ..compiler.equation_compiler import Q88, compile_to_datapath
from ..hdl_gen._ident import sanitize_ident
from ..neurons.equation_builder import EquationNeuron
from .fpga_compilation_result import FoldedResourceMetrics
from .fpga_connection_routing import (
    _ceil_log2_at_least_one,
    _connection_sources_are_analogue,
    _external_input_layout,
    _normalise_connection_delay_steps,
    _signed_hex,
)
from .fpga_neuron_rtl import (
    _NEURON_TEMPLATES,
    _dequantised_pop,
    _heterogeneous_param_names,
    _param_neuron_literal,
    _population_neuron,
)
from .neuron_graph import ConnectionSpec, NeuronSpec
from .quantise_params import QuantisedGraph


def _cond_mux(terms: Sequence[tuple[str, str]], default: str) -> str:
    """Fold ``(condition, value)`` pairs into a nested Verilog ternary expression.

    The pairs are evaluated in order — the first true condition wins — so
    ``[("s==0", "a"), ("s==1", "b")]`` with default ``"z"`` becomes
    ``(s==0 ? a : (s==1 ? b : z))``. A single term still emits the ternary so the
    selector stays in the expression even when only one population is folded.
    """
    expr = default
    for cond, value in reversed(list(terms)):
        expr = f"({cond} ? {value} : {expr})"
    return expr


def _folded_population_input(
    pop: NeuronSpec,
    feeding: list[ConnectionSpec],
    *,
    data_width: int,
    fraction: int,
    acc_width: int,
    ext_offsets: dict[str, int],
    spike_offsets: dict[str, int],
    analogue_offsets: dict[str, int],
    pop_names: set[str],
    is_first_pop: bool,
    idx_signal: str,
    idx_w: int,
    suffix: str,
) -> tuple[list[str], str]:
    """Build one folded population's per-neuron input-current datapath.

    Returns ``(decl_lines, cur_i_expr)``, where ``cur_i_expr`` is the input current
    for the neuron currently addressed by ``idx_signal`` and ``decl_lines`` declare
    the supporting wires/registers. Three fan-in shapes are emitted, matching the
    direct interconnect bit-for-bit:

    * **connection-less** — the first population draws one external ``I_ext`` lane
      per neuron; any other connection-less population has no drive (zero current);
    * **external-weighted** — external-source columns multiply a per-neuron weight
      row (selected from a ``case`` ROM over ``idx_signal``), shifted by ``fraction``;
    * **spiking fan-in** — recurrent (self) or inter-population spikes read the
      prior-tick global ``spike_bus`` at the source population's bit offset and gate
      the sign-extended weight, exactly like the direct path's registered spikes. A
      per-source synaptic delay of ``d`` ticks instead reads ``spike_bus_hist_d`` (the
      ``spike_bus`` committed ``d`` ticks ago), mirroring direct's ``*_spike_d{d}``
      register chain.
    * **analogue fan-in** — a source population of an analogue type (``li`` /
      ``cuba_li`` / ``integrator``, whose output is the membrane voltage rather than a
      spike) reads the prior-tick global ``v_bus`` (one ``DATA_WIDTH`` word per analogue
      source neuron, committed once per tick like ``spike_bus``) at the source's word
      offset and multiplies it by the per-neuron weight (shifted by ``fraction``), or
      threshold-gates the sign-extended weight on that voltage, exactly like the direct
      path's registered ``v_out``. A delay of ``d`` ticks instead reads
      ``v_bus_hist_d`` (the voltage bus committed ``d`` ticks ago), mirroring direct's
      ``*_v_d{d}`` register chain.

    NIR ``Threshold`` transforms fold too: a **source threshold** gates the full
    sign-extended weight on the source value (spike magnitude or external input)
    exceeding the per-column threshold; a **destination threshold** sums the
    connection's terms into a per-neuron ``raw`` accumulator and replaces them with a
    fixed spike-magnitude term when ``raw`` exceeds the per-neuron threshold (selected
    from the same ``case`` ROM over ``idx_signal``).

    A connection **bias** adds a per-destination-neuron constant (held in the same
    per-neuron ``case`` ROM, ACC_WIDTH) to that connection's term list before its
    weighted fan-in, so a destination threshold wraps the bias along with the weights.

    The accumulator width (``ACC_WIDTH``), the saturating cast (``sat_acc``), the
    ``ext_input_*`` lane wires, and the ``spike_bus_hist_*`` delay shift-register are
    module-scope and emitted once by the caller.
    """
    n = pop.n_neurons
    product_width = 2 * data_width
    if not feeding:
        if is_first_pop:
            return [], f"I_ext_flat[{idx_signal} * {data_width} +: {data_width}]"
        return [], f"{data_width}'sd0"

    decls: list[str] = []
    rom_regs: list[tuple[str, int]] = []  # (reg name, bit width) for the per-neuron ROM
    rows: dict[int, list[str]] = {nrow: [] for nrow in range(n)}
    term_lines: list[str] = []
    pop_term_names: list[str] = []
    spike_mag = _signed_hex(1 << fraction, data_width)
    for ci, conn in enumerate(feeding):
        weights = np.asarray(conn.weights, dtype=np.int64)
        src_is_analogue = conn.src in analogue_offsets
        src_is_pop = conn.src in pop_names and not src_is_analogue
        ext_off = ext_offsets.get(conn.src, 0)
        spike_base = spike_offsets.get(conn.src, 0)
        analogue_base = analogue_offsets.get(conn.src, 0)
        delay_vector = _normalise_connection_delay_steps(
            getattr(conn, "delay_steps", 0),
            int(weights.shape[1]),
            f"Connection {conn.src}->{conn.dst}",
        )
        source_thresholds = (
            None
            if conn.source_threshold is None
            else np.asarray(conn.source_threshold, dtype=np.int64).reshape(-1)
        )
        destination_thresholds = (
            None
            if conn.destination_threshold is None
            else np.asarray(conn.destination_threshold, dtype=np.int64).reshape(-1)
        )
        conn_terms: list[str] = []
        if conn.bias is not None:
            # A per-destination-neuron constant, held in the same per-neuron case ROM
            # (ACC_WIDTH, like a destination threshold) and added to the connection's
            # term list before the weighted fan-in, exactly like the direct path. When
            # the connection is destination-thresholded the bias therefore participates
            # in the per-neuron ``raw`` sum.
            biases = np.asarray(conn.bias, dtype=np.int64).reshape(-1)
            rb = f"rb{suffix}_{ci}"
            rom_regs.append((rb, acc_width))
            for nrow in range(n):
                rows[nrow].append(f"{rb} = {_signed_hex(int(biases[nrow]), acc_width)};")
            conn_terms.append(rb)
        for src in range(weights.shape[1]):
            rw = f"rw{suffix}_{ci}_{src}"
            rom_regs.append((rw, data_width))
            for nrow in range(n):
                rows[nrow].append(f"{rw} = {_signed_hex(int(weights[nrow, src]), data_width)};")
            # Full weight, sign-extended to ACC_WIDTH (gated paths contribute the whole
            # weight; only the un-thresholded external path multiplies by the input).
            rw_sext = f"{{{{(ACC_WIDTH - DATA_WIDTH){{{rw}[DATA_WIDTH - 1]}}}}, {rw}}}"
            if src_is_analogue:
                # Analogue source: the prior-tick committed membrane voltage from the
                # global v_bus (one DATA_WIDTH word per analogue source neuron). A delay
                # of d ticks instead reads v_bus_hist_d (the voltage bus committed d ticks
                # ago). It multiplies the weight (shifted) or, under a source threshold,
                # gates the sign-extended weight on the voltage — exactly like the direct
                # path's registered v_out (or its v_d{d} delay-chain) term.
                delay = delay_vector[src]
                v_signal = "v_bus" if delay == 0 else f"v_bus_hist_{delay}"
                v_word = f"{v_signal}[{(analogue_base + src) * data_width} +: DATA_WIDTH]"
                if source_thresholds is not None:
                    thr = _signed_hex(int(source_thresholds[src]), data_width)
                    conn_terms.append(f"({v_word} > {thr} ? {rw_sext} : {acc_width}'sd0)")
                else:
                    mul = f"vmul{suffix}_{ci}_{src}"
                    term = f"vterm{suffix}_{ci}_{src}"
                    term_lines.append(
                        f"    wire signed [{product_width - 1}:0] {mul} = {v_word} * {rw};"
                    )
                    term_lines.append(
                        f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};"
                    )
                    conn_terms.append(term)
            elif src_is_pop:
                # A delay of d ticks reads the d-tick-old snapshot spike_bus_hist_d.
                delay = delay_vector[src]
                spike_signal = "spike_bus" if delay == 0 else f"spike_bus_hist_{delay}"
                spike_bit = f"{spike_signal}[{spike_base + src}]"
                if source_thresholds is not None:
                    thr = _signed_hex(int(source_thresholds[src]), data_width)
                    spike_value = f"({spike_bit} ? {spike_mag} : {data_width}'sd0)"
                    conn_terms.append(f"({spike_value} > {thr} ? {rw_sext} : {acc_width}'sd0)")
                else:
                    conn_terms.append(f"({spike_bit} ? {rw_sext} : {acc_width}'sd0)")
            else:
                ext_in = f"ext_input_{ext_off + src}"
                if source_thresholds is not None:
                    thr = _signed_hex(int(source_thresholds[src]), data_width)
                    conn_terms.append(f"({ext_in} > {thr} ? {rw_sext} : {acc_width}'sd0)")
                else:
                    mul = f"fmul{suffix}_{ci}_{src}"
                    term = f"fterm{suffix}_{ci}_{src}"
                    term_lines.append(
                        f"    wire signed [{product_width - 1}:0] {mul} = {ext_in} * {rw};"
                    )
                    term_lines.append(
                        f"    wire signed [ACC_WIDTH - 1:0] {term} = {mul} >>> {fraction};"
                    )
                    conn_terms.append(term)

        if destination_thresholds is not None:
            # The connection's whole fan-in is thresholded per destination neuron: a
            # per-neuron threshold ROM, then the connection emits a fixed spike-magnitude.
            dthr = f"dthr{suffix}_{ci}"
            rom_regs.append((dthr, acc_width))
            for nrow in range(n):
                rows[nrow].append(
                    f"{dthr} = {_signed_hex(int(destination_thresholds[nrow]), acc_width)};"
                )
            raw = f"raw{suffix}_{ci}"
            out = f"thr_out{suffix}_{ci}"
            raw_expr = " + ".join(conn_terms) if conn_terms else f"{acc_width}'sd0"
            term_lines.append(f"    wire signed [ACC_WIDTH - 1:0] {raw} = {raw_expr};")
            term_lines.append(f"    wire {out} = ({raw} > {dthr});")
            pop_term_names.append(
                f"({out} ? {_signed_hex(1 << fraction, acc_width)} : {acc_width}'sd0)"
            )
        else:
            pop_term_names.extend(conn_terms)

    decls.extend(f"    reg signed [{w - 1}:0] {name};" for name, w in rom_regs)
    default_assigns = " ".join(f"{name} = {w}'sd0;" for name, w in rom_regs)
    decls.append("    always @(*) begin")
    decls.append(f"        case ({idx_signal})")
    for nrow in range(n):
        decls.append(f"            {idx_w}'d{nrow}: begin {' '.join(rows[nrow])} end")
    decls.append(f"            default: begin {default_assigns} end")
    decls.append("        endcase")
    decls.append("    end")
    decls.extend(term_lines)
    acc_expr = " + ".join(pop_term_names) if pop_term_names else f"{acc_width}'sd0"
    decls.append(f"    wire signed [ACC_WIDTH - 1:0] fold_i_acc{suffix} = {acc_expr};")
    return decls, f"sat_acc(fold_i_acc{suffix})"


def can_fold(qgraph: QuantisedGraph, *, data_width: int) -> bool:
    """Return True if the graph is in the folded interconnect's supported subset.

    The folded interconnect time-multiplexes any number of populations of supported
    neuron types over a per-type PE pool and one global spike bus. A graph folds when
    every population has an ODE template, every population's heterogeneous per-neuron
    parameters are datapath parameters the PE can carry on a port (streamed from a
    per-neuron parameter ROM — a parameter varying in something other than an ODE
    parameter cannot be streamed and falls back to the direct path), and every
    connection is one of:

    * **connection-less** — neurons driven only by their own external ``I_ext`` lane;
    * **external-weighted** — fed by external (non-population) source columns;
    * **spiking fan-in** — recurrent (self) or inter-population spikes from another
      population, read from the prior-tick global spike bus, optionally delayed or
      gated by a source/destination NIR ``Threshold`` transform;
    * **analogue fan-in** — an analogue source population (``li``/``cuba_li``/
      ``integrator``, whose output is the membrane voltage), read from the prior-tick
      global voltage bus (optionally delayed via a voltage-bus history register) and
      multiplied (or threshold-gated) by the weight.

    Connections may also carry a per-destination-neuron bias constant. Only a *delayed
    external* (non-population) source connection is not folded — a synaptic delay has
    registered semantics only from a neuron population — and falls back to the direct
    interconnect.

    Parameters
    ----------
    qgraph : QuantisedGraph
        Quantised network graph to classify.
    data_width : int
        Fixed-point data width used to compare per-neuron parameters.

    Returns
    -------
    bool
        ``True`` when the graph can use the shared-datapath interconnect.
    """
    pops = qgraph.populations
    if not pops:
        return False
    pop_by_name = {p.name: p for p in pops}
    pop_names = set(pop_by_name)
    if any(p.neuron_type not in _NEURON_TEMPLATES for p in pops):
        return False
    for pop in pops:
        het = _heterogeneous_param_names(pop, data_width)
        if het:
            # Population construction validates that every varying key belongs to the
            # canonical template. All accepted template parameters are datapath inputs.
            _population_neuron(pop.neuron_type, pop)
    for conn in qgraph.connections:
        if conn.dst not in pop_names:
            return False
        src_pop = pop_by_name.get(conn.src)
        delay_vector = _normalise_connection_delay_steps(
            getattr(conn, "delay_steps", 0),
            int(np.asarray(conn.weights).shape[1]),
            f"Connection {conn.src}->{conn.dst}",
        )
        if any(delay_vector) and src_pop is None:
            # A synaptic delay has registered semantics only from a neuron population
            # source (spiking via spike_bus_hist or analogue via v_bus_hist); a delayed
            # external input is left to the direct path.
            return False
    return True


def folded_resource_metrics(qgraph: QuantisedGraph, *, data_width: int) -> FoldedResourceMetrics:
    """Summarise the shared-datapath resources of a foldable graph.

    Counts one PE per distinct neuron type (the per-type pool), the shared
    weighted-fan-in multipliers (external-source and analogue-voltage-source columns —
    spiking recurrent or inter-population fan-in is spike-gated and uses none), the BRAM
    state-word storage summed over populations, the per-neuron parameter-ROM storage for
    heterogeneous populations, the cycles-per-tick, and the direct-path instance count the
    fold collapses.

    Parameters
    ----------
    qgraph : QuantisedGraph
        A graph satisfying :func:`can_fold`.
    data_width : int
        Fixed-point data width (BRAM word sizing).

    Returns
    -------
    FoldedResourceMetrics
        The architectural fold summary.
    """
    pops = qgraph.populations
    pop_names = {p.name for p in pops}
    n_total = sum(p.n_neurons for p in pops)
    state_ram_bits = 0
    max_state_vars = 0
    for pop in pops:
        neuron = _population_neuron(pop.neuron_type, pop)
        n_vars = len(neuron.equations)
        max_state_vars = max(max_state_vars, n_vars)
        state_ram_bits += pop.n_neurons * n_vars * data_width
    analogue_src_names = {p.name for p in pops if _connection_sources_are_analogue(p)}
    shared_multipliers = sum(
        int(np.asarray(conn.weights).shape[1])
        for conn in qgraph.connections
        # External-weighted and analogue (voltage) sources multiply; spiking sources are
        # spike-gated and use no multiplier.
        if conn.src not in pop_names or conn.src in analogue_src_names
    )
    distinct_types = len({pop.neuron_type for pop in pops})
    # Per-neuron parameter ROM: each population contributes one data-width word per neuron
    # for every parameter that varies across its neurons (a uniform parameter is baked, no
    # ROM). The parameter-space analogue of the state BRAM.
    param_rom_bits = sum(
        pop.n_neurons * len(_heterogeneous_param_names(pop, data_width)) * data_width
        for pop in pops
    )
    return FoldedResourceMetrics(
        neurons=n_total,
        state_vars_per_neuron=max_state_vars,
        pe_instances=distinct_types,
        shared_multipliers=shared_multipliers,
        state_ram_bits=state_ram_bits,
        cycles_per_tick=n_total + 1,
        direct_neuron_instances=n_total,
        populations=len(pops),
        param_rom_bits=param_rom_bits,
    )


def build_folded_interconnect(
    module_name: str,
    qgraph: QuantisedGraph,
    *,
    data_width: int = 16,
    fraction: int = 8,
) -> tuple[dict[str, str], str]:
    """Generate a time-multiplexed (folded) top plus its per-type datapath PE pool.

    One combinational PE (:func:`compile_to_datapath`) per distinct neuron type and
    one BRAM-backed state array per population are shared across every neuron: a
    single sequencer steps one neuron per cycle, walking each population in turn,
    reading the addressed neuron's packed state from its BRAM, driving the population's
    PE with that state and the neuron's input current, and writing the next state back.
    Spikes accumulate over a tick into a global accumulator and commit to a single
    ``spike_bus`` in a dedicated cycle (``tick_done`` pulses), so the bus is race-free
    and stable for the whole next tick. Recurrent and inter-population spiking fan-in
    read that prior-tick ``spike_bus`` at the source population's bit offset — the same
    double-buffer the direct interconnect's registered spikes provide. An analogue
    source population's membrane voltage is committed the same way to a global ``v_bus``
    (one ``DATA_WIDTH`` word per analogue source neuron), so analogue fan-in reads the
    prior-tick voltage exactly like the direct path's registered ``v_out``.

    Restricted to the :func:`can_fold` subset. Returns
    ``({neuron_module_key: pe_source}, top_module_source)`` with one PE source per
    distinct neuron type, keyed ``"{neuron_type}_pe"`` for the compilation artefacts.

    Parameters
    ----------
    module_name : str
        Verilog module name for the folded network top.
    qgraph : QuantisedGraph
        Quantised graph accepted by :func:`can_fold`.
    data_width : int
        Fixed-point word width.
    fraction : int
        Number of fractional bits in each fixed-point word.

    Returns
    -------
    tuple[dict[str, str], str]
        Per-type processing-element sources and the folded top-level Verilog.

    Raises
    ------
    ValueError
        If the graph is outside the supported folded subset.
    """
    if not can_fold(qgraph, data_width=data_width):
        raise ValueError("graph is outside the folded interconnect's supported subset")

    pops = list(qgraph.populations)
    conns = list(qgraph.connections)
    pop_by_name = {p.name: p for p in pops}
    pop_names = set(pop_by_name)
    safe_module = sanitize_ident(module_name, context="module name")
    q = Q88(data_width=data_width, fraction=fraction)

    # Global spike-bus layout: each population owns a contiguous slice, matching the
    # direct interconnect's pop_offsets so the golden raster compares bit-for-bit.
    spike_offsets: dict[str, int] = {}
    cursor = 0
    for pop in pops:
        spike_offsets[pop.name] = cursor
        cursor += pop.n_neurons
    n_total = cursor

    # Global voltage-bus layout: each analogue source population (one whose membrane
    # voltage feeds another population) owns a contiguous DATA_WIDTH-word slice. Empty
    # when no analogue source is used, in which case no v_bus is emitted.
    analogue_src_names = {
        conn.src
        for conn in conns
        if (sp := pop_by_name.get(conn.src)) is not None and _connection_sources_are_analogue(sp)
    }
    analogue_offsets: dict[str, int] = {}
    v_cursor = 0
    for pop in pops:
        if pop.name in analogue_src_names:
            analogue_offsets[pop.name] = v_cursor
            v_cursor += pop.n_neurons
    v_total = v_cursor

    ext_width, ext_offsets, _ext_src_widths = _external_input_layout(conns, pop_by_name, pops)
    input_bus_width = max(1, ext_width * data_width)

    # Global accumulator width chosen exactly as the direct path (seeded by the external
    # vector width, widened by the largest single connection) so saturation matches.
    max_terms = ext_width if pops else 1
    for conn in conns:
        max_terms = max(max_terms, int(np.asarray(conn.weights).shape[1]))
    acc_width = max(data_width + 2, (2 * data_width) + _ceil_log2_at_least_one(max_terms + 1))

    # Deepest synaptic delay over spiking vs analogue connections, kept separate so each
    # bus holds only the history depth it needs. A delay of d ticks reads the snapshot
    # from d ticks ago, so a depth-D shift-register of committed buses is held; D == 0
    # means no delayed connection of that kind and no history register is emitted.
    max_delay = 0  # spiking-source history depth (spike_bus_hist)
    max_analogue_delay = 0  # analogue-source history depth (v_bus_hist)
    for conn in conns:
        delay_vector = _normalise_connection_delay_steps(
            getattr(conn, "delay_steps", 0),
            int(np.asarray(conn.weights).shape[1]),
            f"Connection {conn.src}->{conn.dst}",
        )
        if conn.src in analogue_offsets:
            max_analogue_delay = max(max_analogue_delay, max(delay_vector))
        else:
            max_delay = max(max_delay, max(delay_vector))

    idx_w = max(1, (max(p.n_neurons for p in pops) - 1).bit_length())
    pidx_w = max(1, (len(pops) - 1).bit_length())

    def slice_of(k: int) -> str:
        return f"[{k * data_width} +: {data_width}]"

    # One PE module per distinct neuron type (params are uniform per type within a
    # compile, so same-type populations instantiate the same module — a true PE pool).
    pe_modules: dict[str, str] = {}
    pe_module_name: dict[str, str] = {}
    type_neuron: dict[str, EquationNeuron] = {}
    type_svars: dict[str, list[str]] = {}
    type_state_w: dict[str, int] = {}
    type_init_packed: dict[str, str] = {}
    # Parameters heterogeneous in ANY population of a type become that type's PE input
    # ports, streamed per-neuron from a parameter ROM (the parameter-space analogue of the
    # state BRAM); a type whose populations are all uniform keeps every parameter baked.
    type_het_param_sets: dict[str, set[str]] = {}
    for pop in pops:
        parameter_names = _heterogeneous_param_names(pop, data_width)
        if parameter_names:
            type_het_param_sets.setdefault(pop.neuron_type, set()).update(parameter_names)
    type_het_params = {
        neuron_type: sorted(parameter_names)
        for neuron_type, parameter_names in type_het_param_sets.items()
    }
    for pop in pops:
        ntype = pop.neuron_type
        if ntype in pe_module_name:
            continue
        # Build the PE from real-valued parameters: the population carries quantised
        # integers, which Q88.encode would encode a second time (see _dequantised_pop).
        neuron = _population_neuron(ntype, _dequantised_pop(pop, fraction))
        type_neuron[ntype] = neuron
        mod = sanitize_ident(f"sc_nir_{ntype}_pe", context="module name")
        pe_module_name[ntype] = mod
        pe_modules[f"{ntype}_pe"] = compile_to_datapath(
            neuron,
            module_name=mod,
            data_width=data_width,
            fraction=fraction,
            param_ports=type_het_params.get(ntype, []),
        )
        svars = [sanitize_ident(v, context="state variable") for v in neuron.equations]
        type_svars[ntype] = svars
        type_state_w[ntype] = len(svars) * data_width
        init_words = [
            f"{data_width}'h{q.encode(neuron.initial_state.get(var, 0.0)) & ((1 << data_width) - 1):0{max(1, data_width // 4)}x}"
            for var in neuron.equations
        ]
        type_init_packed[ntype] = "{" + ", ".join(reversed(init_words)) + "}"

    # ----- module header + saturating accumulator (module-scope, shared) -----
    lines: list[str] = [
        f"// Auto-generated folded (time-multiplexed) top-level network: {safe_module}",
        "// SC-NeuroCore NIR → FPGA compiler — per-type shared datapath PE pool + BRAM state.",
        f"// Populations: {len(pops)} ({n_total} neurons); one neuron per cycle, shared spike bus.",
        "`timescale 1ns / 1ps",
        "",
        f"module {safe_module} (",
        "    input  wire clk,",
        "    input  wire rst_n,",
        "    input  wire en,",
        f"    input  wire signed [{input_bus_width - 1}:0] I_ext_flat,",
        f"    output reg  [{n_total - 1}:0] spike_bus,",
        "    output reg  tick_done",
        ");",
        "",
        f"    localparam integer DATA_WIDTH = {data_width};",
        f"    localparam integer ACC_WIDTH = {acc_width};",
        f"    localparam integer N_TOTAL = {n_total};",
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
        f"    reg [{pidx_w - 1}:0] pidx;  // active population",
        f"    reg [{idx_w - 1}:0] nidx;  // neuron within the active population",
        "    reg phase;  // 0 = process one neuron, 1 = commit tick",
        f"    reg [{n_total - 1}:0] spike_acc;",
        "",
    ]

    # Delayed spiking fan-in: a depth-`max_delay` shift-register of committed spike buses,
    # so a connection delayed by d ticks reads the bus from d ticks ago.
    for d in range(1, max_delay + 1):
        lines.append(f"    reg [{n_total - 1}:0] spike_bus_hist_{d};")
    if max_delay:
        lines.append("")

    # Global analogue voltage bus: one DATA_WIDTH word per analogue source neuron, the
    # membrane voltage committed once per tick (double-buffered through v_acc), so an
    # analogue fan-in reads the prior-tick voltage exactly like the direct path's v_out.
    if v_total:
        lines.append(f"    reg signed [{v_total * data_width - 1}:0] v_bus;")
        lines.append(f"    reg signed [{v_total * data_width - 1}:0] v_acc;")
        # A delayed analogue fan-in reads v_bus_hist_d (the voltage bus committed d ticks
        # ago), the exact double-buffer analogue of spike_bus_hist for delayed spikes.
        for d in range(1, max_analogue_delay + 1):
            lines.append(f"    reg signed [{v_total * data_width - 1}:0] v_bus_hist_{d};")
        lines.append("")

    # Per-population state BRAMs (word width set by the population's neuron type).
    for pi, pop in enumerate(pops):
        sw = type_state_w[pop.neuron_type]
        lines.extend(
            [
                f"    // Population {pi}: {pop.name} ({pop.neuron_type} x {pop.n_neurons}); "
                f"spike_bus[{spike_offsets[pop.name]} +: {pop.n_neurons}]",
                '    (* ram_style = "block" *)',
                f"    reg [{sw - 1}:0] state_bram_{pi} [0:{pop.n_neurons - 1}];",
            ]
        )
    lines.append("")

    # Per-neuron parameter ROMs — the parameter-space analogue of the state BRAM. For each
    # parameter carried on a type's PE ports, every population of that type supplies a
    # per-neuron value addressed by `nidx`: a combinational case-ROM when the population is
    # heterogeneous in that parameter, a constant when it is uniform. The active population's
    # value is selected by `pidx` (exactly like the state read) and drives the shared PE port,
    # so each neuron sees its own parameters — bit-for-bit the direct path's per-instance
    # ``#(.P_X(...))`` overrides.
    for ntype, het_names in type_het_params.items():
        same_type = [pi for pi, pop in enumerate(pops) if pop.neuron_type == ntype]
        for pname in het_names:
            pkey = sanitize_ident(pname, context="parameter name").lower()
            for pi in same_type:
                pop = pops[pi]
                sig = f"param_{pkey}_{pi}"
                if pname in _heterogeneous_param_names(pop, data_width):
                    lines.append(f"    reg signed [DATA_WIDTH - 1:0] {sig};")
                    lines.append(f"    always @(*) begin  // {pop.name}.{pname} per-neuron ROM")
                    lines.append("        case (nidx)")
                    for j in range(pop.n_neurons):
                        lit = _param_neuron_literal(pop, pname, j, data_width)
                        lines.append(f"            {idx_w}'d{j}: {sig} = {lit};")
                    lines.append(
                        f"            default: {sig} = {_param_neuron_literal(pop, pname, 0, data_width)};"
                    )
                    lines.append("        endcase")
                    lines.append("    end")
                else:
                    lit = _param_neuron_literal(pop, pname, 0, data_width)
                    lines.append(f"    wire signed [DATA_WIDTH - 1:0] {sig} = {lit};")
            mux = _cond_mux(
                [(f"pidx == {pidx_w}'d{pi}", f"param_{pkey}_{pi}") for pi in same_type],
                default=f"param_{pkey}_{same_type[0]}",
            )
            lines.append(f"    wire signed [DATA_WIDTH - 1:0] param_{pkey}_{ntype} = {mux};")
    if type_het_params:
        lines.append("")

    # Shared external-input lane wires (only when external sources exist).
    if ext_offsets:
        for k in range(ext_width):
            lines.append(
                f"    wire signed [DATA_WIDTH - 1:0] ext_input_{k} = "
                f"I_ext_flat[{k * data_width} +: {data_width}];"
            )
        lines.append("")

    # Per-population input current, addressed by the shared sequencer index nidx.
    for pi, pop in enumerate(pops):
        feeding = [conn for conn in conns if conn.dst == pop.name]
        decls, cur_i_expr = _folded_population_input(
            pop,
            feeding,
            data_width=data_width,
            fraction=fraction,
            acc_width=acc_width,
            ext_offsets=ext_offsets,
            spike_offsets=spike_offsets,
            analogue_offsets=analogue_offsets,
            pop_names=pop_names,
            is_first_pop=(pi == 0),
            idx_signal="nidx",
            idx_w=idx_w,
            suffix=f"_{pi}",
        )
        lines.append(f"    // --- population {pi} ({pop.name}) input current ---")
        lines.extend(decls)
        lines.append(f"    wire signed [DATA_WIDTH - 1:0] cur_I_{pi} = {cur_i_expr};")
    active_i_mux = _cond_mux(
        [(f"pidx == {pidx_w}'d{pi}", f"cur_I_{pi}") for pi in range(len(pops))],
        default=f"{data_width}'sd0",
    )
    lines.append("")
    lines.append(f"    wire signed [DATA_WIDTH - 1:0] active_I = {active_i_mux};")
    lines.append("")

    # Per-type PE pool. Each PE's current state is the active same-type population's
    # BRAM word; only the active population's PE outputs are written back each cycle.
    for ntype, mod in pe_module_name.items():
        svars = type_svars[ntype]
        sw = type_state_w[ntype]
        same_type = [pi for pi, pop in enumerate(pops) if pop.neuron_type == ntype]
        state_mux = _cond_mux(
            [(f"pidx == {pidx_w}'d{pi}", f"state_bram_{pi}[nidx]") for pi in same_type],
            default=f"state_bram_{same_type[0]}[nidx]",
        )
        lines.extend(
            [
                f"    // PE for neuron type '{ntype}' (shared across {len(same_type)} population(s))",
                f"    wire [{sw - 1}:0] cur_state_{ntype} = {state_mux};",
                f"    wire [{sw - 1}:0] next_state_{ntype};",
                f"    wire pe_spike_{ntype};",
                f"    {mod} pe_inst_{ntype} (",
                "        .I_t(active_I),",
            ]
        )
        # Heterogeneous parameters streamed from the per-neuron ROM into the PE's ports.
        for pname in type_het_params.get(ntype, []):
            pkey = sanitize_ident(pname, context="parameter name").lower()
            vname = f"P_{sanitize_ident(pname, context='parameter name').upper()}"
            lines.append(f"        .{vname}(param_{pkey}_{ntype}),")
        lines.extend(
            f"        .{svars[k]}_reg(cur_state_{ntype}{slice_of(k)})," for k in range(len(svars))
        )
        lines.append(f"        .spike_out(pe_spike_{ntype}),")
        lines.extend(
            f"        .{svars[k]}_next_out(next_state_{ntype}{slice_of(k)}),"
            for k in range(len(svars))
        )
        lines[-1] = lines[-1].rstrip(",")
        lines.extend(["    );", ""])

    # Last-neuron index of the active population (sequencer roll-over bound).
    last_mux = _cond_mux(
        [
            (f"pidx == {pidx_w}'d{pi}", f"{idx_w}'d{pop.n_neurons - 1}")
            for pi, pop in enumerate(pops)
        ],
        default=f"{idx_w}'d{pops[-1].n_neurons - 1}",
    )
    lines.append(f"    wire [{idx_w - 1}:0] cur_pop_last = {last_mux};")
    lines.append("")

    # ----- sequencer FSM -----
    lines.extend(
        [
            "    integer i;",
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            "            pidx <= 0;",
            "            nidx <= 0;",
            "            phase <= 1'b0;",
            "            spike_acc <= 0;",
            "            spike_bus <= 0;",
            "            tick_done <= 1'b0;",
        ]
    )
    for d in range(1, max_delay + 1):
        lines.append(f"            spike_bus_hist_{d} <= 0;")
    if v_total:
        lines.append("            v_bus <= 0;")
        lines.append("            v_acc <= 0;")
        for d in range(1, max_analogue_delay + 1):
            lines.append(f"            v_bus_hist_{d} <= 0;")
    for pi, pop in enumerate(pops):
        lines.append(
            f"            for (i = 0; i < {pop.n_neurons}; i = i + 1) "
            f"state_bram_{pi}[i] <= {type_init_packed[pop.neuron_type]};"
        )
    lines.extend(
        [
            "        end else if (en) begin",
            "            if (phase == 1'b0) begin",
            "                tick_done <= 1'b0;",
            "                case (pidx)",
        ]
    )
    for pi, pop in enumerate(pops):
        ntype = pop.neuron_type
        off = spike_offsets[pop.name]
        bus_idx = "nidx" if off == 0 else f"{off} + nidx"
        writeback = (
            f"spike_acc[{bus_idx}] <= pe_spike_{ntype}; "
            f"state_bram_{pi}[nidx] <= next_state_{ntype};"
        )
        if pop.name in analogue_offsets:
            # Snapshot this analogue source neuron's next membrane voltage into v_acc;
            # it commits to v_bus at the tick boundary for the next tick's readers.
            voff = analogue_offsets[pop.name]
            v_word_off = type_svars[ntype].index("v") * data_width
            v_base = "nidx" if voff == 0 else f"({voff} + nidx)"
            writeback += (
                f" v_acc[{v_base} * DATA_WIDTH +: DATA_WIDTH] <= "
                f"next_state_{ntype}[{v_word_off} +: DATA_WIDTH];"
            )
        lines.append(f"                    {pidx_w}'d{pi}: begin {writeback} end")
    lines.extend(
        [
            "                    default: ;",
            "                endcase",
            "                if (nidx == cur_pop_last) begin",
            f"                    if (pidx == {pidx_w}'d{len(pops) - 1}) phase <= 1'b1;",
            "                    else begin pidx <= pidx + 1'b1; nidx <= 0; end",
            "                end else nidx <= nidx + 1'b1;",
            "            end else begin",
            "                spike_bus <= spike_acc;",
            "                tick_done <= 1'b1;",
            "                phase <= 1'b0;",
            "                pidx <= 0;",
            "                nidx <= 0;",
        ]
    )
    if v_total:
        # Commit the analogue voltage snapshot alongside the spike bus.
        lines.append("                v_bus <= v_acc;")
    # Advance the delay shift-registers on the same commit edge (nonblocking, so every
    # stage samples its old source): hist_d <- hist_{d-1}, hist_1 <- the bus being retired.
    for d in range(max_delay, 0, -1):
        source = "spike_bus" if d == 1 else f"spike_bus_hist_{d - 1}"
        lines.append(f"                spike_bus_hist_{d} <= {source};")
    for d in range(max_analogue_delay, 0, -1):
        source = "v_bus" if d == 1 else f"v_bus_hist_{d - 1}"
        lines.append(f"                v_bus_hist_{d} <= {source};")
    lines.extend(
        [
            "            end",
            "        end",
            "    end",
            "",
            "endmodule",
            "",
        ]
    )
    return pe_modules, "\n".join(lines)
