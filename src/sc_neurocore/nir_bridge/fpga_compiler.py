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

import logging
from typing import Any, Literal, Mapping

import numpy as np

from ..compiler.equation_compiler import Q88
from ..ir.scnir_convert import SCNIRConversionConfig, build_scnir_from_neuron_graph
from ..ir.scnir_hdl import build_scnir_source_bundle
from .fpga_aer_interconnect import build_aer_interconnect as _build_top_aer
from .fpga_compilation_result import (
    FoldedResourceMetrics as FoldedResourceMetrics,
    NetworkCompilationResult as NetworkCompilationResult,
    SCNIRExternalInputManifestEntry as SCNIRExternalInputManifestEntry,
)
from .fpga_connection_routing import (
    DelayVector as DelayVector,
    _MAX_SYNTHESISABLE_DELAY_STEPS as _MAX_SYNTHESISABLE_DELAY_STEPS,
    _ceil_log2_at_least_one as _ceil_log2_at_least_one,
    _connection_has_thresholds as _connection_has_thresholds,
    _connection_sources_are_analogue as _connection_sources_are_analogue,
    _external_input_layout as _external_input_layout,
    _external_input_manifest as _external_input_manifest,
    _normalise_connection_delay_steps as _normalise_connection_delay_steps,
    _signed_hex as _signed_hex,
    validate_connection_routing as _validate_connection_routing,
)
from .fpga_direct_interconnect import build_direct_interconnect as _build_top_direct
from .fpga_folded_interconnect import (
    _cond_mux as _cond_mux,
    _folded_population_input as _folded_population_input,
    build_folded_interconnect as _build_top_folded,
    can_fold as _can_fold,
    folded_resource_metrics as _folded_resource_metrics,
)
from .fpga_neuron_rtl import (
    _NEURON_TEMPLATES as _NEURON_TEMPLATES,
    _dequantised_pop as _dequantised_pop,
    _heterogeneous_param_names as _heterogeneous_param_names,
    _neuron_param_override as _neuron_param_override,
    _param_neuron_literal as _param_neuron_literal,
    _population_module_signature as _population_module_signature,
    _population_neuron as _population_neuron,
    _population_params_are_uniform as _population_params_are_uniform,
    _representative_param as _representative_param,
    _resolved_population_params as _resolved_population_params,
    _type_default_qparams as _type_default_qparams,
    build_neuron_module as _build_neuron_module,
)
from .fpga_scnir_hierarchy import (
    _SCNIR_STREAM_FRAGMENT_RE as _SCNIR_STREAM_FRAGMENT_RE,
    _build_scnir_hierarchy_module as _build_scnir_hierarchy_module,
    _hierarchy_output_wires_by_stream as _hierarchy_output_wires_by_stream,
    _hierarchy_port_declaration as _hierarchy_port_declaration,
    _hierarchy_top_wire_declaration as _hierarchy_top_wire_declaration,
    _hierarchy_weight_expr as _hierarchy_weight_expr,
    _hierarchy_zero_literal as _hierarchy_zero_literal,
    _scnir_connection_stream_id as _scnir_connection_stream_id,
    _scnir_stream_fragment as _scnir_stream_fragment,
    build_scnir_hierarchy_modules as _build_scnir_hierarchy_modules,
    resolve_hierarchy_weight_literals as _hierarchy_weight_literals,
)
from .fpga_weight_rom import build_weight_rom as _build_weight_rom
from .neuron_graph import NeuronGraph, NeuronSpec
from .quantise_params import quantise_graph

logger = logging.getLogger(__name__)

_AER_THRESHOLD = 64
_MAX_UNROLLED_NEURONS = 8192
_MAX_FOLDED_NEURONS = 262144
_MAX_SYNTHESISABLE_DATA_WIDTH = 64
_MAX_SYNTHESISABLE_SYNAPSES = 1_048_576


def _check_synthesis_resource_bounds(
    *,
    total_neurons: int,
    total_synapses: int,
    data_width: int,
    fraction: int,
    interconnect: str | None,
) -> None:
    """Reject IR that would exhaust synthesis resources or is malformed, before any RTL.

    ``data_width`` must fit a hardware-plausible fixed-point datapath and ``fraction`` must
    leave at least one integer or sign bit, so the signed Q-format is well formed (a
    ``fraction >= data_width`` would give negative integer bits and silently emit broken
    RTL). The direct and AER
    interconnects instantiate one module per neuron, so their neuron count is capped at
    ``_MAX_UNROLLED_NEURONS``; the folded interconnect shares a single processing element
    and is capped higher at ``_MAX_FOLDED_NEURONS`` (its state-RAM depth). Independently,
    every interconnect flattens all weight matrices into one ROM, so the total synapse count
    is capped at ``_MAX_SYNTHESISABLE_SYNAPSES`` regardless of interconnect. Raising here
    means a pathological network fails closed rather than exhausting memory or the
    downstream synthesis tool.
    """
    if total_neurons <= 0:
        raise ValueError("network must contain at least one neuron")
    if not 1 <= data_width <= _MAX_SYNTHESISABLE_DATA_WIDTH:
        raise ValueError(
            f"data_width {data_width} is outside the synthesisable range "
            f"[1, {_MAX_SYNTHESISABLE_DATA_WIDTH}]"
        )
    if not 0 <= fraction < data_width:
        raise ValueError(
            f"fraction {fraction} must satisfy 0 <= fraction < data_width ({data_width}); "
            f"a signed Q-format needs at least one integer or sign bit"
        )
    if total_synapses > _MAX_SYNTHESISABLE_SYNAPSES:
        raise ValueError(
            f"network has {total_synapses} synapses, exceeding the weight-ROM synthesis "
            f"guard {_MAX_SYNTHESISABLE_SYNAPSES}; reduce connection density or fan-out"
        )
    if interconnect == "folded":
        if total_neurons > _MAX_FOLDED_NEURONS:
            raise ValueError(
                f"network has {total_neurons} neurons, exceeding the folded synthesis "
                f"guard {_MAX_FOLDED_NEURONS} (the shared processing element's state-RAM "
                f"depth)"
            )
    elif total_neurons > _MAX_UNROLLED_NEURONS:
        raise ValueError(
            f"network has {total_neurons} neurons, exceeding the per-neuron synthesis "
            f"guard {_MAX_UNROLLED_NEURONS} for the direct/AER interconnect; use "
            f"interconnect='folded' to share one processing element across neurons"
        )


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
    online_learning: Mapping[str, Mapping[str, Any]] | None = None,
    interconnect: str | None = None,
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
    online_learning : Mapping[str, Mapping[str, Any]] | None
        Optional validated per-weight-stream SC-NIR online-learning annotations,
        keyed by deterministic stream id such as ``"conn.src_to_dst.weight"``.
    interconnect : str | None
        ``None`` (default) auto-selects direct (small) or AER (large) wiring;
        ``"direct"`` forces direct; ``"folded"`` opts into the time-multiplexed
        shared-datapath interconnect (one PE per neuron type + per-population BRAM
        state, swept by a single sequencer), which supports the :func:`_can_fold`
        subset: any number of populations with external-weighted, recurrent, or
        inter-population spiking fan-in.

    Returns
    -------
    NetworkCompilationResult
        All generated Verilog sources and compilation metadata.

    Raises
    ------
    ValueError
        If the graph is empty or contains unsupported neuron types.
    """
    _check_synthesis_resource_bounds(
        total_neurons=graph.total_neurons,
        total_synapses=graph.total_synapses,
        data_width=data_width,
        fraction=fraction,
        interconnect=interconnect,
    )
    _validate_connection_routing(graph)
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
        online_learning=dict(online_learning or {}),
    )
    scnir_document = build_scnir_from_neuron_graph(graph, config=scnir_config)
    scnir_source_bundle = build_scnir_source_bundle(scnir_document)
    warnings: list[str] = []

    # Step 1: Quantise
    qgraph = quantise_graph(graph, q)
    warnings.extend(qgraph.warnings)
    hierarchy_weight_literals = _hierarchy_weight_literals(scnir_document, qgraph)

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
        verilog = _build_neuron_module(
            ntype,
            rep_pop,
            data_width=data_width,
            fraction=fraction,
        )
        neuron_modules[ntype] = verilog
        logger.info("Generated Verilog for neuron type: %s", ntype)

    # Step 3: Weight ROM
    weight_rom = _build_weight_rom(qgraph, data_width=data_width)

    # Step 4: Top-level interconnect.  Small networks use explicit direct
    # wiring.  Larger networks use weighted address-event fan-out while
    # preserving dense affine accumulation semantics.
    total_neurons = graph.total_neurons
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
    folded_metrics: FoldedResourceMetrics | None = None

    if interconnect == "folded":
        # Opt-in time-multiplexed interconnect; never auto-selected. Restricted to the
        # _can_fold subset (any number of populations of supported types with
        # external-weighted, recurrent, inter-population, delayed, thresholded, biased
        # spiking, or analogue-voltage fan-in).
        if not _can_fold(qgraph, data_width=data_width):
            raise ValueError(
                "interconnect='folded' supports populations of supported neuron types with "
                "external-weighted, recurrent, inter-population, delayed, NIR-thresholded, "
                "biased spiking, or analogue source connections (the folded subset), and only "
                "when every population's per-neuron parameters are uniform (the shared PE has no "
                "per-neuron parameter RAM); a delayed external (non-population) source connection "
                "or a heterogeneous population is not folded — use 'direct' or auto otherwise"
            )
        selected_interconnect = "folded"
        pe_modules, top_module = _build_top_folded(
            module_name, qgraph, data_width=data_width, fraction=fraction
        )
        neuron_modules.update(pe_modules)
        folded_metrics = _folded_resource_metrics(qgraph, data_width=data_width)
    elif interconnect not in (None, "direct"):
        raise ValueError(
            f"unknown interconnect {interconnect!r}; choose 'folded', 'direct', or None (auto)"
        )
    elif (
        interconnect is None
        and total_neurons > _AER_THRESHOLD
        and not has_delayed_connections
        and not has_threshold_connections
    ):
        selected_interconnect = "aer"
        top_module = _build_top_aer(
            module_name,
            qgraph,
            data_width=data_width,
            fraction=fraction,
            bitstream_length=bitstream_length,
            scnir_stream_count=len(scnir_document.streams),
            scnir_source_module_count=len(scnir_source_bundle.manifest),
            scnir_hierarchy=scnir_document.hierarchy,
            scnir_semantic_hierarchy_stream_ids=frozenset(hierarchy_weight_literals),
        )
    else:
        selected_interconnect = "direct"
        if interconnect is None and total_neurons > _AER_THRESHOLD and has_delayed_connections:
            warnings.append(
                "Using direct interconnect because delayed recurrent connections require "
                "registered one-step source semantics"
            )
        if interconnect is None and total_neurons > _AER_THRESHOLD and has_threshold_connections:
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
            scnir_hierarchy=scnir_document.hierarchy,
            scnir_semantic_hierarchy_stream_ids=frozenset(hierarchy_weight_literals),
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
        interconnect=selected_interconnect,
        scnir_document=scnir_document,
        scnir_source_modules=dict(scnir_source_bundle.modules),
        scnir_source_manifest=scnir_source_bundle.manifest,
        scnir_external_inputs=_external_input_manifest(qgraph),
        scnir_hierarchy_modules=_build_scnir_hierarchy_modules(
            scnir_document,
            weight_literals=hierarchy_weight_literals,
        ),
        folded_metrics=folded_metrics,
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
