# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet routing and package-link analysis

"""AER routing tables, path selection, timing, congestion, and link energy."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from sc_neurocore.chiplet.topology import ChipletTopology, InterposerLink, InterposerTech


@dataclass
class RoutingEntry:
    """Map one source neuron to a remote die, neuron, and Q8.8 weight."""

    src_neuron: int
    dst_die: int
    dst_neuron: int
    weight_q88: int = 256


@dataclass
class RoutingTable:
    """Store AER routes originating on one die."""

    die_id: int
    entries: list[RoutingEntry] = field(default_factory=list)

    def add_route(self, src: int, dst_die: int, dst_neuron: int, weight: int = 256) -> None:
        """Append one source-to-destination route."""
        self.entries.append(RoutingEntry(src, dst_die, dst_neuron, weight))

    def routes_to_die(self, target_die: int) -> list[RoutingEntry]:
        """Return entries whose destination is ``target_die``."""
        return [entry for entry in self.entries if entry.dst_die == target_die]

    @property
    def num_entries(self) -> int:
        """Return the number of routes in the table."""
        return len(self.entries)

    @property
    def target_dies(self) -> list[int]:
        """Return sorted unique destination die identifiers."""
        return sorted({entry.dst_die for entry in self.entries})


def compute_decorrelation_seeds(topology: ChipletTopology) -> dict[tuple[int, int], int]:
    """Assign deterministic non-zero LFSR seeds to directed links.

    The golden-ratio sequence spreads adjacent link indices over the 16-bit
    state space while preserving reproducibility.
    """
    phi_inverse = 0.6180339887498949
    return {
        (link.src_die, link.dst_die): int((index * phi_inverse * 65535) % 65535) + 1
        for index, link in enumerate(topology.links)
    }


_ENERGY_PJ_PER_BIT: dict[InterposerTech, float] = {
    InterposerTech.UCIE: 0.5,
    InterposerTech.BOW: 0.3,
    InterposerTech.EMIB: 0.2,
    InterposerTech.COWOS: 0.1,
    InterposerTech.ORGANIC: 2.0,
    InterposerTech.CUSTOM: 0.5,
}


def link_energy_pj(link: InterposerLink, bits: int) -> float:
    """Return energy in picojoules for ``bits`` transmitted over ``link``."""
    if bits < 0:
        raise ValueError("bits must be >= 0")
    return _ENERGY_PJ_PER_BIT[link.technology] * bits


@dataclass
class PackageEnergyReport:
    """Record per-link and aggregate communication energy."""

    per_link_pj: dict[tuple[int, int], float] = field(default_factory=dict)
    total_pj: float = 0.0

    @property
    def total_nj(self) -> float:
        """Return aggregate energy in nanojoules."""
        return self.total_pj / 1000.0


def estimate_package_energy(
    topology: ChipletTopology,
    bits_per_link: int = 256,
) -> PackageEnergyReport:
    """Estimate communication energy for uniform traffic on every link."""
    if bits_per_link < 0:
        raise ValueError("bits_per_link must be >= 0")
    report = PackageEnergyReport()
    for link in topology.links:
        key = (link.src_die, link.dst_die)
        energy = link_energy_pj(link, bits_per_link)
        report.per_link_pj[key] = energy
        report.total_pj += energy
    return report


@dataclass
class CongestionReport:
    """Record per-link utilisation and the highest-utilisation link."""

    utilisation: dict[tuple[int, int], float] = field(default_factory=dict)
    bottleneck: tuple[int, int] | None = None
    max_utilisation: float = 0.0


def estimate_congestion(
    topology: ChipletTopology,
    routing_tables: dict[int, RoutingTable],
    events_per_cycle: int = 100,
) -> CongestionReport:
    """Estimate directed-link utilisation from AER routing tables."""
    if events_per_cycle < 0:
        raise ValueError("events_per_cycle must be >= 0")
    report = CongestionReport()
    link_traffic: dict[tuple[int, int], int] = {}
    for die_id, table in routing_tables.items():
        for entry in table.entries:
            key = (die_id, entry.dst_die)
            link_traffic[key] = link_traffic.get(key, 0) + events_per_cycle
    for link in topology.links:
        key = (link.src_die, link.dst_die)
        traffic = link_traffic.get(key, 0)
        bits_per_second = traffic * link.data_width * 200e6
        utilisation = bits_per_second / (link.bandwidth_gbps * 1e9)
        report.utilisation[key] = utilisation
        if utilisation > report.max_utilisation:
            report.max_utilisation = utilisation
            report.bottleneck = key
    return report


def _bfs_path(
    topology: ChipletTopology,
    src: int,
    dst: int,
    excluded: set[tuple[int, int]],
) -> list[int] | None:
    visited = {src: [src]}
    queue: deque[int] = deque([src])
    while queue:
        current = queue.popleft()
        for link in topology.get_links_from(current):
            next_die = link.dst_die
            if (current, next_die) in excluded or next_die in visited:
                continue
            visited[next_die] = [*visited[current], next_die]
            if next_die == dst:
                return visited[next_die]
            queue.append(next_die)
    return None


def find_disjoint_paths(
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    max_paths: int = 2,
) -> list[list[int]]:
    """Find up to ``max_paths`` directed link-disjoint paths."""
    if max_paths < 0:
        raise ValueError("max_paths must be >= 0")
    if src_die == dst_die:
        return [[src_die]]
    paths: list[list[int]] = []
    excluded_links: set[tuple[int, int]] = set()
    for _ in range(max_paths):
        path = _bfs_path(topology, src_die, dst_die, excluded_links)
        if path is None:
            break
        paths.append(path)
        excluded_links.update(zip(path, path[1:]))
    return paths


@dataclass
class TimingSimResult:
    """Summarise accumulated timing and reliability along one path."""

    total_latency_ns: float
    max_jitter_ns: float
    min_bandwidth_gbps: float
    worst_ber: float
    path: list[int]


def simulate_timing(
    topology: ChipletTopology, src_die: int, dst_die: int
) -> TimingSimResult | None:
    """Return the lowest-latency reachable path between two dies."""
    if src_die == dst_die:
        return TimingSimResult(0.0, 0.0, float("inf"), 0.0, [src_die])
    state: dict[int, tuple[float, float, float, float, list[int]]] = {
        src_die: (0.0, 0.0, float("inf"), 0.0, [src_die])
    }
    queue: deque[int] = deque([src_die])
    while queue:
        current = queue.popleft()
        latency, jitter, bandwidth, ber, path = state[current]
        for link in topology.get_links_from(current):
            next_die = link.dst_die
            candidate = (
                latency + link.latency_ns,
                max(jitter, link.jitter_ns),
                min(bandwidth, link.bandwidth_gbps),
                max(ber, link.bit_error_rate),
                [*path, next_die],
            )
            if next_die not in state or state[next_die][0] > candidate[0]:
                state[next_die] = candidate
                queue.append(next_die)
    result = state.get(dst_die)
    return TimingSimResult(*result) if result is not None else None


def adaptive_route(
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    congestion: CongestionReport,
    congestion_threshold: float = 0.8,
) -> list[int] | None:
    """Find a path avoiding links above ``congestion_threshold`` when possible."""
    if not math.isfinite(congestion_threshold) or congestion_threshold < 0:
        raise ValueError("congestion_threshold must be finite and >= 0")
    excluded = {
        link
        for link, utilisation in congestion.utilisation.items()
        if utilisation > congestion_threshold
    }
    path = _bfs_path(topology, src_die, dst_die, excluded)
    return path if path is not None else _bfs_path(topology, src_die, dst_die, set())


def bandwidth_aware_route(
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    required_gbps: float,
) -> list[int] | None:
    """Find a path whose every link meets ``required_gbps``."""
    if not math.isfinite(required_gbps) or required_gbps < 0:
        raise ValueError("required_gbps must be finite and >= 0")
    if src_die == dst_die:
        return [src_die]
    visited = {src_die: [src_die]}
    queue: deque[int] = deque([src_die])
    while queue:
        current = queue.popleft()
        for link in topology.get_links_from(current):
            next_die = link.dst_die
            if next_die in visited or link.bandwidth_gbps < required_gbps:
                continue
            visited[next_die] = [*visited[current], next_die]
            if next_die == dst_die:
                return visited[next_die]
            queue.append(next_die)
    return None


__all__ = [
    "CongestionReport",
    "PackageEnergyReport",
    "RoutingEntry",
    "RoutingTable",
    "TimingSimResult",
    "adaptive_route",
    "bandwidth_aware_route",
    "compute_decorrelation_seeds",
    "estimate_congestion",
    "estimate_package_energy",
    "find_disjoint_paths",
    "link_energy_pj",
    "simulate_timing",
]
