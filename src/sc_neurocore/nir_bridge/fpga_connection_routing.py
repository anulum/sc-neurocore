# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Connection layout, delay validation, and fixed-point RTL literals."""

import math
from typing import Any

import numpy as np

from .fpga_compilation_result import SCNIRExternalInputManifestEntry
from .neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec
from .quantise_params import QuantisedGraph

DelayVector = tuple[int, ...]
_MAX_SYNTHESISABLE_DELAY_STEPS = 1024


def validate_connection_routing(graph: NeuronGraph) -> None:
    """Validate connection shapes, external lanes, and delay vectors before lowering.

    The SC-NIR conversion and all three interconnect emitters index connection
    matrices by source column. Validating the matrix rank first converts malformed
    direct ``NeuronGraph`` input into a stable ``ValueError`` instead of leaking an
    ``IndexError`` from a downstream conversion step.

    Parameters
    ----------
    graph : NeuronGraph
        Network graph whose connection layout will be lowered to RTL.

    Raises
    ------
    ValueError
        If a connection matrix, endpoint width, bias, threshold, or delay is
        inconsistent with its source and destination populations.
    """
    pop_by_name = {pop.name: pop for pop in graph.populations}
    for conn in graph.connections:
        weights = np.asarray(conn.weights)
        if weights.ndim != 2:
            raise ValueError(f"Connection {conn.src}->{conn.dst} weights must be a 2-D matrix")
    _, _, external_widths = _external_input_layout(
        graph.connections,
        pop_by_name,
        graph.populations,
    )
    for conn in graph.connections:
        weights = np.asarray(conn.weights)
        dst_pop = pop_by_name.get(conn.dst)
        if dst_pop is None:
            raise ValueError(f"Connection destination {conn.dst!r} is not a neuron population")
        if weights.shape[0] != dst_pop.n_neurons:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} has {weights.shape[0]} "
                f"destination rows for {dst_pop.n_neurons} destination neurons"
            )
        src_pop = pop_by_name.get(conn.src)
        expected_src = src_pop.n_neurons if src_pop is not None else external_widths[conn.src]
        if weights.shape[1] != expected_src:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} has {weights.shape[1]} "
                f"source columns for {expected_src} source signals"
            )
        _normalise_connection_delay_steps(
            conn.delay_steps,
            expected_src,
            f"Connection {conn.src}->{conn.dst}",
        )
        if conn.bias is not None and np.asarray(conn.bias).reshape(-1).size != dst_pop.n_neurons:
            raise ValueError(
                f"Connection {conn.src}->{conn.dst} bias length does not match "
                f"{dst_pop.n_neurons} destination neurons"
            )
        if conn.source_threshold is not None:
            source_threshold = np.asarray(conn.source_threshold).reshape(-1)
            if source_threshold.size != expected_src:
                raise ValueError(
                    f"Connection {conn.src}->{conn.dst} source_threshold length "
                    f"does not match {expected_src} source columns"
                )
        if conn.destination_threshold is not None:
            destination_threshold = np.asarray(conn.destination_threshold).reshape(-1)
            if destination_threshold.size != dst_pop.n_neurons:
                raise ValueError(
                    f"Connection {conn.src}->{conn.dst} destination_threshold length "
                    f"does not match {dst_pop.n_neurons} destination neurons"
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
    raise ValueError("interconnect requires at least one neuron population")


def _external_input_manifest(graph: QuantisedGraph) -> tuple[SCNIRExternalInputManifestEntry, ...]:
    """Return the flattened input-bus layout used by generated top-level RTL."""
    pop_by_name = {pop.name: pop for pop in graph.populations}
    _, offsets, widths = _external_input_layout(graph.connections, pop_by_name, graph.populations)
    return tuple(
        SCNIRExternalInputManifestEntry(source=source, offset=offsets[source], width=widths[source])
        for source in sorted(offsets, key=lambda item: offsets[item])
    )


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
