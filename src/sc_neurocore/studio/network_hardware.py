# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio network graph lowered to the hardware network compiler

"""Lower a Studio network graph to the hardware network compiler, or refuse by name.

The lowering reproduces the network the Studio simulates, not an approximation
of it. Each population keeps its catalogue model's own step:

* ``SCLapicqueLIFNeuron`` (profile ``sc_lif``) becomes the ``sc_lif`` template,
  the exact step ``v <- v*decay + (v_rest + R*I)*(1 - decay)`` with
  ``decay = exp(-dt/tau)``, firing at ``v >= v_threshold``;
* ``PerfectIntegratorNeuron`` becomes ``sc_if`` (profile ``sc_inclusive``,
  ``v >= v_threshold``) or NIR's ``if`` (profile ``naud_gerstner_2012``,
  ``v > v_threshold``), with ``r = 1/c_m``.

A projection keeps the connectivity the runtime realises and its delay in
steps. The hardware's neurons register their spikes, which is the runtime's
one-step propagation latency, so a projection of ``d`` delay steps arrives
``d + 1`` steps after the spike in both. A constant drive enters through an
external input lane per neuron at unit weight, so the hardware takes the
drive as its input rather than baking it into the design.

Everything else is refused before any hardware is generated, every reason at
once: another model, a Poisson drive (the hardware has no source for the
Studio's generator), a membrane that does not start at rest, a value outside
the fixed-point range or one that quantises to zero, a delay beyond the
synthesis guard, and two populations of one model with different parameters
(the compiler shares one module per neuron type).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.nir_bridge.fpga_connection_routing import _MAX_SYNTHESISABLE_DELAY_STEPS
from sc_neurocore.nir_bridge.neuron_graph_contracts import (
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
)
from sc_neurocore.studio.network_graph_spec import GraphSpec, PopulationSpec, resolve_graph
from sc_neurocore.studio.network_nir import dense_weight

NETWORK_HARDWARE_SCHEMA_VERSION = "sc-neurocore.studio.network-hardware.v1"
_LIF_MODEL = "SCLapicqueLIFNeuron"
_IF_MODEL = "PerfectIntegratorNeuron"
_IF_TYPES = {"sc_inclusive": "sc_if", "naud_gerstner_2012": "if"}


class HardwareLoweringRefused(ValueError):
    """The graph holds something the hardware lowering cannot reproduce exactly."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(reasons))


@dataclass(frozen=True, slots=True)
class LoweredNetwork:
    """A Studio network as the hardware compiler's graph, with its drive lanes."""

    graph: NeuronGraph
    spec: GraphSpec
    drive_lanes: tuple[tuple[str, tuple[float, ...]], ...]
    data_width: int
    fraction: int
    notes: tuple[str, ...]

    def input_sha256(self) -> str:
        """Return the digest every later artefact of this lowering is bound to.

        It covers the resolved graph (its ``graph_sha256``), the fixed-point
        format and the lowering schema, so two lowerings share it only when they
        compile the same network in the same format.
        """
        body = {
            "schema_version": NETWORK_HARDWARE_SCHEMA_VERSION,
            "graph_sha256": self.spec.to_public_dict()["graph_sha256"],
            "data_width": self.data_width,
            "fraction": self.fraction,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def lower_graph(graph: object, *, data_width: int = 16, fraction: int = 8) -> LoweredNetwork:
    """Lower a Studio network graph to the hardware compiler's graph.

    Raises
    ------
    GraphRejected
        When the graph does not validate.
    HardwareLoweringRefused
        When any population, drive or projection cannot be reproduced exactly;
        the exception lists every reason.
    """
    spec = resolve_graph(graph)
    q = Q88(data_width=data_width, fraction=fraction)
    reasons: list[str] = []
    notes: list[str] = []
    populations: list[NeuronSpec] = []
    connections: list[ConnectionSpec] = []
    drive_lanes: list[tuple[str, tuple[float, ...]]] = []
    signatures: dict[str, tuple[str, dict[str, float]]] = {}

    for population in spec.populations:
        lowered = _population(population, spec.dt, reasons)
        if lowered is None:
            continue
        neuron_type, params = lowered
        _check_values(f"population {population.id}", params, q, reasons, notes)
        first = signatures.setdefault(neuron_type, (population.id, params))
        if first[1] != params:
            reasons.append(
                f"populations {first[0]} and {population.id} are both {neuron_type} with different "
                "parameters; the compiler shares one module per neuron type"
            )
        populations.append(
            NeuronSpec(
                name=population.id,
                neuron_type=neuron_type,
                n_neurons=population.count,
                params={name: np.full(population.count, value) for name, value in params.items()},
                dt=spec.dt,
            )
        )
        drive = population.drive
        if drive.kind == "constant" and drive.current is not None:
            source = f"drive_{population.id}"
            _check_values(
                f"population {population.id} drive", {"current": drive.current}, q, reasons, notes
            )
            drive_lanes.append(
                (source, tuple(float(drive.current) for _ in range(population.count)))
            )
            connections.append(
                ConnectionSpec(
                    src=source,
                    dst=population.id,
                    weights=np.eye(population.count),
                    bias=None,
                    delay_steps=0,
                )
            )
        elif drive.kind == "poisson":
            reasons.append(
                f"population {population.id}: a Poisson drive has no hardware source that "
                "reproduces the Studio's generator"
            )

    counts = {population.id: population.count for population in spec.populations}
    for projection in spec.projections:
        if projection.delay_steps > _MAX_SYNTHESISABLE_DELAY_STEPS:
            reasons.append(
                f"projection {projection.id}: {projection.delay_steps} delay steps exceed the "
                f"synthesis guard of {_MAX_SYNTHESISABLE_DELAY_STEPS}"
            )
            continue
        _check_values(
            f"projection {projection.id}", {"weight": projection.weight}, q, reasons, notes
        )
        connections.append(
            ConnectionSpec(
                src=projection.source,
                dst=projection.target,
                weights=dense_weight(
                    projection, counts[projection.source], counts[projection.target]
                ),
                bias=None,
                delay_steps=projection.delay_steps,
            )
        )

    if reasons:
        raise HardwareLoweringRefused(reasons)
    network = NeuronGraph(
        populations=populations,
        connections=connections,
        input_pop=drive_lanes[0][0] if drive_lanes else populations[0].name,
        output_pop=populations[-1].name,
        dt=spec.dt,
    )
    return LoweredNetwork(
        graph=network,
        spec=spec,
        drive_lanes=tuple(drive_lanes),
        data_width=data_width,
        fraction=fraction,
        notes=tuple(notes),
    )


def _population(
    population: PopulationSpec, dt: float, reasons: list[str]
) -> tuple[str, dict[str, float]] | None:
    """Return the template and parameters of one population, or record why not."""
    neuron = population.inputs.instantiate()
    if population.model == _LIF_MODEL and neuron.profile == "sc_lif":
        resting = float(neuron.v_rest)
        decay = math.exp(-dt / float(neuron.tau))
        lowered = (
            "sc_lif",
            {
                "decay": decay,
                "gain": 1.0 - decay,
                "v_rest": resting,
                "r": float(neuron.resistance),
                "v_threshold": float(neuron.v_threshold),
                "v_reset": float(neuron.v_reset),
            },
        )
    elif population.model == _IF_MODEL:
        resting = 0.0
        lowered = (
            _IF_TYPES[neuron.profile],
            {
                "r": 1.0 / float(neuron.c_m),
                "v_threshold": float(neuron.v_threshold),
                "v_reset": float(neuron.v_reset),
            },
        )
    else:
        profile = getattr(neuron, "profile", None)
        reasons.append(
            f"population {population.id}: model {population.model}"
            + (f" (profile {profile})" if profile is not None else "")
            + f" has no hardware lowering; the lowering reproduces {_LIF_MODEL} (profile sc_lif) "
            f"and {_IF_MODEL}"
        )
        return None
    if float(neuron.v) != resting:
        reasons.append(
            f"population {population.id}: its membrane starts at {float(neuron.v)!r}, not at rest "
            f"({resting!r}); the hardware neuron starts at rest"
        )
    return lowered


def _check_values(
    where: str, values: Mapping[str, float], q: Q88, reasons: list[str], notes: list[str]
) -> None:
    """Refuse values the fixed-point format cannot hold; note what it rounds."""
    for name, value in values.items():
        if not q.min_value <= value <= q.max_value:
            reasons.append(
                f"{where}: {name} = {value!r} is outside the {_q_label(q)} range "
                f"[{q.min_value!r}, {q.max_value!r}]"
            )
            continue
        encoded = round(value * (1 << q.fraction)) / (1 << q.fraction)
        if value != 0.0 and encoded == 0.0:
            reasons.append(f"{where}: {name} = {value!r} quantises to zero in {_q_label(q)}")
        elif encoded != value:
            notes.append(
                f"{where}: {name} = {value!r} is held as {encoded!r} "
                f"(relative error {abs(encoded - value) / abs(value):.2e})"
            )


def _q_label(q: Q88) -> str:
    return f"Q{q.data_width - q.fraction}.{q.fraction}"


def lowering_public_dict(lowered: LoweredNetwork) -> dict[str, Any]:
    """Return the JSON projection of a lowering: what was compiled, and what it rounds."""
    return {
        "schema_version": NETWORK_HARDWARE_SCHEMA_VERSION,
        "input_sha256": lowered.input_sha256(),
        "graph_sha256": lowered.spec.to_public_dict()["graph_sha256"],
        "q_format": f"Q{lowered.data_width - lowered.fraction}.{lowered.fraction}",
        "populations": [
            {"id": pop.name, "template": pop.neuron_type, "count": pop.n_neurons}
            for pop in lowered.graph.populations
        ],
        "drive_lanes": [
            {"source": source, "currents": list(currents)}
            for source, currents in lowered.drive_lanes
        ],
        "notes": list(lowered.notes),
    }


__all__ = [
    "NETWORK_HARDWARE_SCHEMA_VERSION",
    "HardwareLoweringRefused",
    "LoweredNetwork",
    "lower_graph",
    "lowering_public_dict",
]
