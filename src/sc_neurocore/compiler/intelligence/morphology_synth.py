# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morphological synthesizer

"""Auto-synthesizer for optimal interconnect topologies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Morphology:
    """Interconnect topology morphology.

    Attributes
    ----------
    topology : str
        ``"Hypercube"``, ``"3D Torus"``, or ``"2D Mesh"``.
    bisection_bandwidth_gbps : float
    routing_latency_ns : float
    dimensions : int
    """

    topology: str
    bisection_bandwidth_gbps: float
    routing_latency_ns: float
    dimensions: int


def synthesize_morphology(equations: dict[str, str], max_generations: int = 10) -> Morphology:
    """Auto-synthesizer for optimal interconnect topologies."""
    inter_dependencies = sum(
        1 for v, e in equations.items() for v2 in equations if v2 in e and v != v2
    )

    if inter_dependencies > len(equations) * 1.5:
        topology = "Hypercube"
        dims = 4
        bw = 512.0
        lat = 2.5
    elif inter_dependencies > len(equations):
        topology = "3D Torus"
        dims = 3
        bw = 256.0
        lat = 5.0
    else:
        topology = "2D Mesh"
        dims = 2
        bw = 128.0
        lat = 10.0

    bw *= 1.0 + (max_generations * 0.05)
    lat *= 1.0 - (max_generations * 0.01)

    return Morphology(
        topology=topology,
        bisection_bandwidth_gbps=round(bw, 1),
        routing_latency_ns=round(max(0.1, lat), 1),
        dimensions=dims,
    )
