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
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hdl_gen._ident import sanitize_ident
from ..compiler.equation_compiler import Q88, compile_to_datapath, compile_to_verilog
from ..ir.scnir_convert import SCNIRConversionConfig, build_scnir_from_neuron_graph
from ..ir.scnir_hdl import (
    SCNIRHDLSourceManifestEntry,
    build_scnir_source_bundle,
)
from ..ir.scnir_schema import SCNIRDocument, SCNIRHierarchyInstance, SCNIRHierarchyPort
from ..neurons.equation_builder import EquationNeuron, from_equations
from .neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec
from .neuron_templates import NEURON_TEMPLATES
from .quantise_params import QuantisedGraph, quantise_graph

logger = logging.getLogger(__name__)

DelayVector = tuple[int, ...]

# Threshold above which the compiler records that event-bus RTL would be useful.
# The default emitter still uses exact direct wiring until weighted routing is
# implemented and verified.
_AER_THRESHOLD = 64
_MAX_SYNTHESISABLE_DELAY_STEPS = 1024
# Fail-closed synthesis-resource guards, checked before any RTL is emitted. The direct and
# AER interconnects instantiate one module per neuron, so an unbounded neuron count would
# inflate the netlist without limit (a synthesis-time denial of service); the folded
# interconnect shares one processing element and is bounded instead by its state- and
# parameter-RAM depth, so it is allowed a higher ceiling. Fixed-point data paths wider than
# 64 bits are not hardware-plausible for these kernels.
_MAX_UNROLLED_NEURONS = 8192
_MAX_FOLDED_NEURONS = 262144
_MAX_SYNTHESISABLE_DATA_WIDTH = 64
# Every interconnect flattens all connection weight matrices into one shared weight ROM, so
# the total synapse count is a second blow-up axis independent of the neuron count: a dense
# N x N connection is N**2 ROM entries. Cap it so pathological connectivity fails closed.
_MAX_SYNTHESISABLE_SYNAPSES = 1_048_576
_SCNIR_STREAM_FRAGMENT_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


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


# ═══════════════════════════════════════════════════════════════════════
# Canonical ODE Templates
# ═══════════════════════════════════════════════════════════════════════

_NEURON_TEMPLATES = NEURON_TEMPLATES


def _representative_param(values: np.ndarray[Any, Any], label: str) -> float:
    """Return the reference (first-neuron) value of a per-neuron parameter.

    This value becomes the default of the shared, parameterised RTL neuron module.
    A heterogeneous population is no longer rejected: every neuron whose own
    quantised parameter differs from this default is instantiated with an explicit
    Verilog parameter override at the top level (see :func:`_neuron_param_override`).
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty parameter array for {label}")
    return float(arr[0])


def _type_default_qparams(pops: Sequence[NeuronSpec], data_width: int) -> dict[str, dict[str, int]]:
    """First-population, first-neuron quantised parameters per neuron type.

    Values are stored as the unsigned two's-complement bit pattern the neuron
    module declares as its parameter default, so the top level only overrides a
    neuron whose quantised parameter differs.
    """
    mask = (1 << data_width) - 1
    defaults: dict[str, dict[str, int]] = {}
    for pop in pops:
        if pop.neuron_type in defaults:
            continue
        entry: dict[str, int] = {}
        for pname, pval in pop.params.items():
            arr = np.atleast_1d(np.asarray(pval).reshape(-1))
            entry[pname] = int(arr[0]) & mask
        defaults[pop.neuron_type] = entry
    return defaults


def _neuron_param_override(
    pop: NeuronSpec,
    neuron_idx: int,
    type_defaults: dict[str, dict[str, int]],
    data_width: int,
) -> str:
    """Return a Verilog parameter-override clause for a single neuron instance.

    Empty when this neuron's per-neuron quantised parameters all equal the shared
    module defaults (the homogeneous case, so the emitted RTL is unchanged).
    Otherwise emits ``#(.P_X(W'sdN), ...)`` so a heterogeneous population reuses
    the same parameterised module with each neuron's own quantised parameters. The
    literal is the unsigned two's-complement bit pattern, matching how the module
    declares each parameter default (negative fixed-point values included).
    """
    mask = (1 << data_width) - 1
    defaults = type_defaults.get(pop.neuron_type, {})
    fragments: list[str] = []
    for pname in sorted(pop.params):
        if pname not in defaults:
            continue
        arr = np.atleast_1d(np.asarray(pop.params[pname]).reshape(-1))
        raw = int(arr[neuron_idx]) if arr.shape[0] == pop.n_neurons else int(arr[0])
        qval = raw & mask
        if qval != defaults[pname]:
            vname = f"P_{sanitize_ident(pname, context='parameter name').upper()}"
            fragments.append(f".{vname}({data_width}'sd{qval})")
    if not fragments:
        return ""
    return " #(" + ", ".join(fragments) + ")"


def _heterogeneous_param_names(pop: NeuronSpec, data_width: int) -> list[str]:
    """Return the sorted parameter names whose per-neuron quantised values vary in ``pop``.

    A name is heterogeneous when its per-neuron array (length ``n_neurons``) holds more
    than one distinct value at the quantised data width — exactly the set the direct path
    emits per-neuron ``#(.P_X(...))`` overrides for (see :func:`_neuron_param_override`),
    and the set the folded interconnect must stream through a per-neuron parameter ROM.

    Uniformity is decided at the *quantised* data width (the same mask the override
    detection uses), so two float parameters that round to the same fixed-point literal
    are not heterogeneous. A scalar / population-shared parameter (an array that is not
    per-neuron) is never heterogeneous.
    """
    mask = (1 << data_width) - 1
    names: list[str] = []
    for pname, pval in pop.params.items():
        arr = np.atleast_1d(np.asarray(pval).reshape(-1))
        if arr.shape[0] != pop.n_neurons:
            continue  # scalar / shared parameter — identical for every neuron
        if len({int(v) & mask for v in arr}) > 1:
            names.append(pname)
    return sorted(names)


def _population_params_are_uniform(pop: NeuronSpec, data_width: int) -> bool:
    """Return True when every per-neuron quantised parameter is identical across ``pop``.

    Equivalent to ``not _heterogeneous_param_names(pop, data_width)``. A heterogeneous
    population — one the direct path reproduces via per-neuron ``#(.P_X(...))`` overrides —
    folds only when the folded interconnect streams its varying parameters through a
    per-neuron parameter ROM (see :func:`_build_top_folded`).
    """
    return not _heterogeneous_param_names(pop, data_width)


def _param_neuron_literal(pop: NeuronSpec, pname: str, neuron_idx: int, data_width: int) -> str:
    """Return the Verilog signed literal of ``pop``'s ``pname`` for one neuron (quantised).

    The literal is the unsigned two's-complement bit pattern the module declares its
    parameter default with — the same form the direct path's per-neuron overrides use
    (see :func:`_neuron_param_override`) — so the folded parameter ROM feeds the PE the
    identical value the direct instance would receive.
    """
    mask = (1 << data_width) - 1
    arr = np.atleast_1d(np.asarray(pop.params[pname]).reshape(-1))
    raw = int(arr[neuron_idx]) if arr.shape[0] == pop.n_neurons else int(arr[0])
    return f"{data_width}'sd{raw & mask}"


def _dequantised_pop(pop: NeuronSpec, fraction: int) -> NeuronSpec:
    """Return ``pop`` with its quantised parameter values scaled back to real units.

    A :class:`QuantisedGraph` population stores fixed-point *integer* parameters
    (``value × 2**fraction``). The folded PE and the per-instance module both encode
    real-valued parameters with :meth:`Q88.encode`, so they must be handed the *real*
    value — feeding the already-quantised integer encodes it a second time (a 16-bit
    ``tau = 5120`` re-encodes to ``5120 × 256 mod 2**16 = 0``, silently baking a broken
    parameter into the shared PE). The rescale is lossless for genuine fixed-point values
    (``5120 / 256 = 20.0`` re-encodes to ``5120``). Parameters absent from ``pop.params``
    are untouched (they fall back to the template default, already a real value).
    """
    if not pop.params:
        return pop
    scale = float(1 << fraction)
    rescaled = {
        name: np.asarray(values, dtype=np.float64) / scale for name, values in pop.params.items()
    }
    return replace(pop, params=rescaled)


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
        params[pname] = _representative_param(pval, f"{pop.name}.{pname}")
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


def _external_input_layout(
    conns: list[ConnectionSpec],
    pop_by_name: dict[str, NeuronSpec],
    pops: list[NeuronSpec],
) -> tuple[int, dict[str, int], dict[str, int]]:
    """Assign stable flattened input-bus lanes to each external source name."""

    offsets: dict[str, int] = {}
    widths: dict[str, int] = {}
    cursor = 0
    for conn in conns:
        if conn.src in pop_by_name:
            continue
        width = int(np.asarray(conn.weights).shape[1])
        if width <= 0:
            raise ValueError(f"Connection {conn.src}->{conn.dst} has no external source columns")
        existing = widths.get(conn.src)
        if existing is not None:
            if existing != width:
                raise ValueError(
                    f"External source {conn.src!r} is used with inconsistent widths "
                    f"{existing} and {width}"
                )
            continue
        offsets[conn.src] = cursor
        widths[conn.src] = width
        cursor += width

    if offsets:
        return cursor, offsets, widths
    if pops:
        return max(1, pops[0].n_neurons), {}, {}
    return 1, {}, {}


def _external_input_manifest(graph: QuantisedGraph) -> tuple[SCNIRExternalInputManifestEntry, ...]:
    """Return the flattened input-bus layout used by generated top-level RTL."""

    pop_by_name = {pop.name: pop for pop in graph.populations}
    _, offsets, widths = _external_input_layout(graph.connections, pop_by_name, graph.populations)
    return tuple(
        SCNIRExternalInputManifestEntry(source=source, offset=offsets[source], width=widths[source])
        for source in sorted(offsets, key=lambda item: offsets[item])
    )


def _scnir_stream_fragment(value: str) -> str:
    cleaned = _SCNIR_STREAM_FRAGMENT_RE.sub("_", value.strip())
    cleaned = cleaned.strip("_.:-")
    if not cleaned:
        cleaned = "stream"
    if not cleaned[0].isalpha():
        cleaned = f"s_{cleaned}"
    return cleaned[:96]


def _scnir_connection_stream_id(src: str, dst: str) -> str:
    return f"conn.{_scnir_stream_fragment(src)}_to_{_scnir_stream_fragment(dst)}.weight"


def _hierarchy_weight_literals(
    document: SCNIRDocument,
    qgraph: QuantisedGraph,
) -> dict[str, tuple[int, ...]]:
    """Return flattened quantised weight literals owned by hierarchy output ports."""

    weights_by_stream: dict[str, np.ndarray[Any, Any]] = {
        _scnir_connection_stream_id(str(conn.src), str(conn.dst)): np.asarray(
            conn.weights,
            dtype=np.int64,
        )
        for conn in qgraph.connections
    }
    literals: dict[str, tuple[int, ...]] = {}
    for instance in document.hierarchy:
        for port in instance.ports:
            if port.direction != "output" or port.signal_kind != "weight":
                continue
            weights = weights_by_stream.get(port.stream_id)
            if weights is None:
                raise ValueError(
                    f"SC-NIR hierarchy port {port.port_name!r} references unknown "
                    f"weight stream {port.stream_id!r}"
                )
            flat = weights.reshape(-1)
            if port.bit_width % int(flat.size) != 0:
                raise ValueError(
                    f"SC-NIR hierarchy weight port {port.port_name!r} bit width "
                    f"{port.bit_width} is not divisible by flattened weight count {flat.size}"
                )
            literals[port.stream_id] = tuple(int(value) for value in flat)
    return literals


def _hierarchy_output_wires_by_stream(
    hierarchy: Sequence[SCNIRHierarchyInstance],
    *,
    semantic_stream_ids: set[str],
) -> dict[str, tuple[str, int]]:
    """Return top-level hierarchy output wire names keyed by SC-NIR stream id."""

    wires: dict[str, tuple[str, int]] = {}
    for instance in hierarchy:
        module_name = sanitize_ident(instance.module_name, context="hierarchy module name")
        for port in instance.ports:
            if port.direction != "output" or port.stream_id not in semantic_stream_ids:
                continue
            port_name = sanitize_ident(port.port_name, context="hierarchy port name")
            wire_name = sanitize_ident(
                f"{module_name}__{port_name}",
                context="hierarchy top-level wire name",
            )
            if port.stream_id in wires:
                raise ValueError(f"duplicate hierarchy output for stream {port.stream_id!r}")
            wires[port.stream_id] = (wire_name, port.bit_width)
    return wires


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
class SCNIRExternalInputManifestEntry:
    """Stable flattened input-bus layout entry for one external source."""

    source: str
    offset: int
    width: int

    def as_dict(self) -> dict[str, int | str]:
        """Return deterministic JSON-ready external input metadata."""

        return {
            "source": self.source,
            "offset": self.offset,
            "width": self.width,
        }


@dataclass(frozen=True)
class FoldedResourceMetrics:
    """Architectural resource summary of a folded (time-multiplexed) interconnect.

    Quantifies what the shared-datapath fold buys versus the direct interconnect's
    one-module-instance-per-neuron unrolling: one processing element per distinct
    neuron type is reused across every neuron of that type (a per-type PE pool), with
    per-neuron state held in BRAM, at the cost of ``cycles_per_tick`` cycles to advance
    the whole network by one timestep.

    Attributes
    ----------
    neurons : int
        Total neurons sharing the datapath across all folded populations.
    state_vars_per_neuron : int
        Widest per-neuron state-variable count across the folded types (the largest
        BRAM word = ``state_vars_per_neuron`` × data width). Equal to the single type's
        count for a homogeneous network.
    pe_instances : int
        Physical processing elements instantiated: one per distinct neuron type. A
        single population, or several populations all of one type, share one PE.
    shared_multipliers : int
        Multipliers in the shared weighted fan-in, summed over external-source columns
        across all populations and reused across each population's neurons. Spiking
        fan-in (recurrent or inter-population) is spike-gated and uses none.
    state_ram_bits : int
        Total BRAM-backed neuron-state storage, in bits, summed over populations
        (each population contributes ``neurons`` × its type's state-var count × data width).
    cycles_per_tick : int
        Clock cycles to advance the whole network by one timestep
        (``neurons`` process cycles + 1 commit cycle).
    direct_neuron_instances : int
        Neuron module instances the direct interconnect would unroll (= ``neurons``);
        the count the fold collapses to ``pe_instances``.
    populations : int
        Number of folded populations sharing the one sequencer and the global spike bus.
    param_rom_bits : int
        Total per-neuron parameter-ROM storage, in bits, for heterogeneous populations
        (each contributes ``neurons`` × its count of per-neuron-varying parameters × data
        width). Zero for a network whose populations all have uniform parameters (the PE
        bakes them). The parameter-space analogue of ``state_ram_bits``.
    """

    neurons: int
    state_vars_per_neuron: int
    pe_instances: int
    shared_multipliers: int
    state_ram_bits: int
    cycles_per_tick: int
    direct_neuron_instances: int
    populations: int = 1
    param_rom_bits: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a deterministic plain-``int`` mapping for manifests/JSON."""
        return {
            "neurons": self.neurons,
            "state_vars_per_neuron": self.state_vars_per_neuron,
            "pe_instances": self.pe_instances,
            "shared_multipliers": self.shared_multipliers,
            "state_ram_bits": self.state_ram_bits,
            "cycles_per_tick": self.cycles_per_tick,
            "direct_neuron_instances": self.direct_neuron_instances,
            "populations": self.populations,
            "param_rom_bits": self.param_rom_bits,
        }


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
        ``"direct"``, ``"aer"``, or ``"folded"`` (the time-multiplexed shared datapath).
    folded_metrics : FoldedResourceMetrics | None
        Architectural fold resource summary when ``interconnect == "folded"``; ``None``
        for the direct/AER paths.
    warnings : list[str]
        Quantisation and compilation warnings.
    scnir_document : SCNIRDocument
        SC-aware metadata document consumed by the compilation artefacts.
    scnir_source_modules : dict[str, str]
        Concrete stochastic source HDL modules keyed by Verilog module name.
    scnir_source_manifest : tuple[SCNIRHDLSourceManifestEntry, ...]
        Deterministic manifest mapping SC-NIR streams to source modules.
    scnir_external_inputs : tuple[SCNIRExternalInputManifestEntry, ...]
        Deterministic flattened input-bus layout for external source names.
    scnir_hierarchy_modules : dict[str, str]
        Standalone SC-NIR hierarchy boundary modules keyed by module name.
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
    scnir_external_inputs: tuple[SCNIRExternalInputManifestEntry, ...]
    scnir_hierarchy_modules: dict[str, str]
    folded_metrics: FoldedResourceMetrics | None = None
    warnings: list[str] = field(default_factory=list)


def _build_scnir_hierarchy_modules(
    document: SCNIRDocument,
    *,
    weight_literals: dict[str, tuple[int, ...]],
) -> dict[str, str]:
    """Emit standalone boundary modules for preserved SC-NIR hierarchy instances."""

    modules: dict[str, str] = {}
    for instance in document.hierarchy:
        module_name = sanitize_ident(instance.module_name, context="hierarchy module name")
        if module_name in modules:
            raise ValueError(f"duplicate SC-NIR hierarchy module name {module_name!r}")
        modules[module_name] = _build_scnir_hierarchy_module(
            instance,
            module_name=module_name,
            weight_literals=weight_literals,
        )
    return modules


def _build_scnir_hierarchy_module(
    instance: SCNIRHierarchyInstance,
    *,
    module_name: str,
    weight_literals: dict[str, tuple[int, ...]],
) -> str:
    """Emit one synthesisable hierarchy boundary module from typed SC-NIR ports."""

    if not instance.ports:
        raise ValueError(f"SC-NIR hierarchy instance {instance.instance_id!r} has no ports")
    port_lines = [f"    {_hierarchy_port_declaration(port)}" for port in instance.ports]
    lines = [
        f"// Auto-generated SC-NIR hierarchy boundary: {module_name}",
        "// Scalar weight outputs may feed the generated top-level MAC path.",
        "`timescale 1ns / 1ps",
        "",
        f"module {module_name} (",
        ",\n".join(port_lines),
        ");",
        "",
    ]
    for port in instance.ports:
        lines.append(f"    // stream_id: {port.stream_id}")
        lines.append(f"    // signal_kind: {port.signal_kind}")
        if port.direction == "output":
            port_name = sanitize_ident(port.port_name, context="hierarchy port name")
            literals = weight_literals.get(port.stream_id)
            if literals is None:
                lines.append(f"    assign {port_name} = {_hierarchy_zero_literal(port)};")
            elif len(literals) == 1:
                lines.append(
                    f"    assign {port_name} = {_signed_hex(literals[0], port.bit_width)};"
                )
            else:
                if port.bit_width % len(literals) != 0:
                    raise ValueError(
                        f"SC-NIR hierarchy port {port.port_name!r} bit width "
                        f"{port.bit_width} is not divisible by literal count {len(literals)}"
                    )
                element_width = port.bit_width // len(literals)
                for index, literal in enumerate(literals):
                    offset = index * element_width
                    lines.append(
                        f"    assign {port_name}[{offset} +: {element_width}] = "
                        f"{_signed_hex(literal, element_width)};"
                    )
        lines.append("")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def _hierarchy_port_declaration(port: SCNIRHierarchyPort) -> str:
    direction = port.direction
    port_name = sanitize_ident(port.port_name, context="hierarchy port name")
    if port.bit_width <= 0:
        raise ValueError(f"SC-NIR hierarchy port {port.port_name!r} has non-positive bit width")
    if port.bit_width == 1:
        return f"{direction} wire {port_name}"
    return f"{direction} wire signed [{port.bit_width - 1}:0] {port_name}"


def _hierarchy_zero_literal(port: SCNIRHierarchyPort) -> str:
    if port.bit_width == 1:
        return "1'b0"
    return f"{port.bit_width}'sd0"


def _build_scnir_hierarchy_instance_block(
    hierarchy: Sequence[SCNIRHierarchyInstance],
    *,
    data_width: int,
) -> list[str]:
    """Emit top-level hierarchy contract instances for preserved SC-NIR boundaries."""

    if not hierarchy:
        return []
    lines = ["    // SC-NIR hierarchy contract instances"]
    for instance in hierarchy:
        module_name = sanitize_ident(instance.module_name, context="hierarchy module name")
        instance_name = sanitize_ident(
            f"{module_name}_hierarchy_inst",
            context="hierarchy instance name",
        )
        port_bindings: list[str] = []
        for port in instance.ports:
            port_name = sanitize_ident(port.port_name, context="hierarchy port name")
            wire_name = sanitize_ident(
                f"{module_name}__{port_name}",
                context="hierarchy top-level wire name",
            )
            lines.append(_hierarchy_top_wire_declaration(wire_name, port, data_width=data_width))
            if port.direction == "input":
                lines.append(f"    assign {wire_name} = {_hierarchy_zero_literal(port)};")
            port_bindings.append(f"        .{port_name}({wire_name})")
        lines.append(f"    {module_name} {instance_name} (")
        lines.append(",\n".join(port_bindings))
        lines.append("    );")
        lines.append("")
    return lines


def _hierarchy_top_wire_declaration(
    wire_name: str,
    port: SCNIRHierarchyPort,
    *,
    data_width: int,
) -> str:
    if port.bit_width <= 0:
        raise ValueError(f"SC-NIR hierarchy port {port.port_name!r} has non-positive bit width")
    if port.bit_width == 1:
        return f"    wire {wire_name};"
    if port.bit_width == data_width:
        return f"    wire signed [DATA_WIDTH - 1:0] {wire_name};"
    return f"    wire signed [{port.bit_width - 1}:0] {wire_name};"


def _hierarchy_weight_expr(
    hierarchy_output_wires: Mapping[str, tuple[str, int]],
    stream_id: str,
    *,
    weight: int,
    data_width: int,
    weight_index: int,
) -> str:
    wire = hierarchy_output_wires.get(stream_id)
    if wire is None:
        return _signed_hex(weight, data_width)
    wire_name, bit_width = wire
    if bit_width == data_width:
        return wire_name
    offset = weight_index * data_width
    if offset + data_width > bit_width:
        raise ValueError(
            f"SC-NIR hierarchy stream {stream_id!r} width {bit_width} cannot provide "
            f"weight index {weight_index}"
        )
    return f"{wire_name}[{offset} +: DATA_WIDTH]"


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
    neuron = _population_neuron(neuron_type, pop)
    return compile_to_verilog(
        neuron,
        module_name=f"sc_nir_{neuron_type}",
        data_width=data_width,
        fraction=fraction,
    )


def _population_neuron(neuron_type: str, pop: NeuronSpec) -> EquationNeuron:
    """Build the canonical :class:`EquationNeuron` for one population's type.

    Single source of truth for the ODE/threshold/reset/params/init/dt used by
    both the per-instance module (:func:`_build_neuron_module`) and the folded
    datapath PE (:func:`_build_top_folded`), so the two share identical dynamics.
    """
    template = _NEURON_TEMPLATES.get(neuron_type)
    if template is None:
        raise ValueError(f"No ODE template for neuron type: {neuron_type!r}")

    # Shared modules are only emitted when every per-neuron parameter is
    # identical.  Heterogeneous populations need generated per-neuron modules
    # and are rejected rather than averaged.
    params = _resolved_population_params(neuron_type, pop)

    init: dict[str, float] = {}
    for eq_str in template["equations"]:
        var_name = eq_str.split("/")[0].replace("d", "", 1).strip()
        if var_name == "v":
            init["v"] = params.get("v_leak", 0.0)
        else:
            init[var_name] = 0.0

    return from_equations(
        *template["equations"],
        threshold=template["threshold"],
        reset=template["reset"],
        params=params,
        init=init,
        dt=pop.dt,
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
    type_defaults = _type_default_qparams(pops, data_width)
    safe_module = sanitize_ident(module_name, context="module name")
    pop_by_name = {pop.name: pop for pop in pops}
    pop_index = {pop.name: idx for idx, pop in enumerate(pops)}
    pop_offsets: dict[str, int] = {}

    offset = 0
    for pop in pops:
        pop_offsets[pop.name] = offset
        offset += pop.n_neurons

    external_width, external_offsets, external_source_widths = _external_input_layout(
        conns,
        pop_by_name,
        pops,
    )
    hierarchy_output_wires = _hierarchy_output_wires_by_stream(
        scnir_hierarchy,
        semantic_stream_ids=set(scnir_semantic_hierarchy_stream_ids),
    )

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
        expected_src = (
            src_pop.n_neurons if src_pop is not None else external_source_widths[conn.src]
        )
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
                if delay_steps <= 0:
                    continue  # an undelayed column needs no register chain
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
    lines.extend(_build_scnir_hierarchy_instance_block(scnir_hierarchy, data_width=data_width))

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


def _can_fold(qgraph: QuantisedGraph, *, data_width: int) -> bool:
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
            # Heterogeneous parameters stream through a per-neuron parameter ROM into the
            # PE's ports (see :func:`_build_top_folded`), but only datapath parameters can
            # be carried on a port; a parameter varying in something the PE does not take
            # (not in its ODE parameters/constants) cannot be streamed — use the direct path.
            neuron = _population_neuron(pop.neuron_type, pop)
            pe_params = set(neuron.parameters) | set(neuron.constants)
            if not set(het) <= pe_params:
                return False
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


def _folded_resource_metrics(qgraph: QuantisedGraph, *, data_width: int) -> FoldedResourceMetrics:
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
        A graph satisfying :func:`_can_fold`.
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


def _build_top_folded(
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

    Restricted to the :func:`_can_fold` subset. Returns
    ``({neuron_module_key: pe_source}, top_module_source)`` with one PE source per
    distinct neuron type, keyed ``"{neuron_type}_pe"`` for the compilation artefacts.
    """
    if not _can_fold(qgraph, data_width=data_width):
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
        if delay_vector:
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
    type_het_params: dict[str, list[str]] = {}
    for pop in pops:
        for pname in _heterogeneous_param_names(pop, data_width):
            names = type_het_params.setdefault(pop.neuron_type, [])
            if pname not in names:
                names.append(pname)
    for names in type_het_params.values():
        names.sort()
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


def _build_top_aer(
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

    Returns
    -------
    str
        Verilog top-level module source.
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

    external_width, external_offsets, external_source_widths = _external_input_layout(
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
        expected_src = (
            src_pop.n_neurons if src_pop is not None else external_source_widths[conn.src]
        )
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
    lines.extend(_build_scnir_hierarchy_instance_block(scnir_hierarchy, data_width=data_width))

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
