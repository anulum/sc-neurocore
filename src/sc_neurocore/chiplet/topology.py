# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet topology and interposer models

"""Die, interposer, planar-topology, and vertical-stack models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class InterposerTech(Enum):
    """Supported die-to-die interconnect technology presets."""

    UCIE = "UCIe"
    BOW = "BoW"
    EMIB = "EMIB"
    COWOS = "CoWoS"
    ORGANIC = "Organic"
    CUSTOM = "Custom"


_INTERPOSER_PRESETS: dict[InterposerTech, tuple[float, float, float, float]] = {
    InterposerTech.UCIE: (2.0, 0.05, 32.0, 1e-15),
    InterposerTech.BOW: (1.5, 0.03, 16.0, 1e-12),
    InterposerTech.EMIB: (1.0, 0.02, 64.0, 1e-15),
    InterposerTech.COWOS: (0.5, 0.01, 128.0, 1e-16),
    InterposerTech.ORGANIC: (5.0, 0.5, 8.0, 1e-12),
    InterposerTech.CUSTOM: (2.0, 0.1, 32.0, 1e-15),
}


@dataclass
class InterposerLink:
    """Describe one directed die-to-die link.

    Parameters
    ----------
    src_die, dst_die
        Non-negative source and destination die identifiers.
    technology
        Interconnect technology used by the link.
    latency_ns, jitter_ns
        Nominal latency and non-negative timing jitter in nanoseconds.
    bandwidth_gbps
        Positive link bandwidth in gigabits per second.
    bit_error_rate
        Per-bit error probability in the closed interval ``[0, 1]``.
    data_width
        Positive payload width in bits.
    is_bidirectional
        Whether package metadata treats the physical link as bidirectional.
    thermal_resistance_k_per_w
        Optional measured bond resistance in kelvin per watt.
    """

    src_die: int
    dst_die: int
    technology: InterposerTech = InterposerTech.UCIE
    latency_ns: float = 2.0
    jitter_ns: float = 0.1
    bandwidth_gbps: float = 32.0
    bit_error_rate: float = 1e-15
    data_width: int = 64
    is_bidirectional: bool = True
    thermal_resistance_k_per_w: float | None = None

    def __post_init__(self) -> None:
        """Validate endpoint identities and finite physical link properties."""
        if self.src_die < 0 or self.dst_die < 0:
            raise ValueError("src_die and dst_die must be >= 0")
        if not math.isfinite(self.latency_ns) or self.latency_ns < 0:
            raise ValueError("latency_ns must be finite and >= 0")
        if not math.isfinite(self.jitter_ns) or self.jitter_ns < 0:
            raise ValueError("jitter_ns must be finite and >= 0")
        if not math.isfinite(self.bandwidth_gbps) or self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be finite and > 0")
        if not math.isfinite(self.bit_error_rate) or not 0 <= self.bit_error_rate <= 1:
            raise ValueError("bit_error_rate must be finite and in [0, 1]")
        if self.data_width <= 0:
            raise ValueError("data_width must be > 0")
        if self.thermal_resistance_k_per_w is not None and (
            not math.isfinite(self.thermal_resistance_k_per_w)
            or self.thermal_resistance_k_per_w <= 0
        ):
            raise ValueError("thermal_resistance_k_per_w must be finite and > 0 when provided")

    @classmethod
    def from_tech(cls, src: int, dst: int, tech: InterposerTech) -> InterposerLink:
        """Construct a link from a technology preset.

        Parameters
        ----------
        src, dst
            Source and destination die identifiers.
        tech
            Technology whose timing, bandwidth, and BER defaults are used.

        Returns
        -------
        InterposerLink
            Link populated with the selected preset.
        """
        latency_ns, jitter_ns, bandwidth_gbps, bit_error_rate = _INTERPOSER_PRESETS[tech]
        return cls(
            src_die=src,
            dst_die=dst,
            technology=tech,
            latency_ns=latency_ns,
            jitter_ns=jitter_ns,
            bandwidth_gbps=bandwidth_gbps,
            bit_error_rate=bit_error_rate,
        )

    @property
    def latency_cycles(self) -> int:
        """Return rounded link latency at the historical 200 MHz reference clock."""
        return max(1, int(self.latency_ns / 5.0 + 0.5))

    @property
    def fifo_depth_log2(self) -> int:
        """Return the minimum asynchronous FIFO depth exponent for link jitter."""
        jitter_cycles = max(1, int(self.jitter_ns / 5.0 + 0.5))
        depth = 1
        while (1 << depth) < jitter_cycles * 4:
            depth += 1
        return max(depth, 3)


@dataclass
class ChipletDie:
    """Describe one die and its local AER configuration."""

    die_id: int
    clock_mhz: float = 200.0
    lfsr_seed: int = 0xACE1
    neuron_ids: list[int] = field(default_factory=list)
    n_neurons: int = 128
    aer_id_width: int = 10
    data_width: int = 16

    def __post_init__(self) -> None:
        """Validate die identity, clock, seed, and local interface widths."""
        if self.die_id < 0:
            raise ValueError("die_id must be >= 0")
        if not math.isfinite(self.clock_mhz) or self.clock_mhz <= 0:
            raise ValueError("clock_mhz must be finite and > 0")
        if not 1 <= self.lfsr_seed <= 0xFFFF:
            raise ValueError("lfsr_seed must be in [1, 65535]")
        if self.n_neurons <= 0 or self.aer_id_width <= 0 or self.data_width <= 0:
            raise ValueError("n_neurons, aer_id_width, and data_width must be > 0")

    @property
    def clock_period_ns(self) -> float:
        """Return the die clock period in nanoseconds."""
        return 1000.0 / self.clock_mhz


def _seed_for_die(die_id: int) -> int:
    seed = (0xACE1 + die_id * 7919) & 0xFFFF
    return seed or 1


@dataclass
class ChipletTopology:
    """Store the directed die and interposer graph for one package."""

    dies: list[ChipletDie] = field(default_factory=list)
    links: list[InterposerLink] = field(default_factory=list)

    def add_die(self, die: ChipletDie) -> None:
        """Append a die to the topology."""
        self.dies.append(die)

    def add_link(self, link: InterposerLink) -> None:
        """Append a directed interposer link to the topology."""
        self.links.append(link)

    @classmethod
    def mesh_2d(
        cls, rows: int, cols: int, tech: InterposerTech = InterposerTech.UCIE
    ) -> ChipletTopology:
        """Construct a rectangular mesh without wrap-around links."""
        _require_grid(rows, cols)
        topo = cls()
        for r in range(rows):
            for c in range(cols):
                die_id = r * cols + c
                topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=_seed_for_die(die_id)))
        for r in range(rows):
            for c in range(cols):
                src = r * cols + c
                if c + 1 < cols:
                    topo.add_link(InterposerLink.from_tech(src, src + 1, tech))
                if r + 1 < rows:
                    topo.add_link(InterposerLink.from_tech(src, src + cols, tech))
        return topo

    @classmethod
    def ring(cls, n_dies: int, tech: InterposerTech = InterposerTech.UCIE) -> ChipletTopology:
        """Construct a directed ring with one outgoing edge per die."""
        _require_die_count(n_dies)
        topo = cls()
        for die_id in range(n_dies):
            topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=_seed_for_die(die_id)))
        for die_id in range(n_dies):
            topo.add_link(InterposerLink.from_tech(die_id, (die_id + 1) % n_dies, tech))
        return topo

    @classmethod
    def star(cls, n_dies: int, tech: InterposerTech = InterposerTech.UCIE) -> ChipletTopology:
        """Construct a bidirectional star with die zero as the hub."""
        _require_die_count(n_dies)
        topo = cls()
        for die_id in range(n_dies):
            topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=_seed_for_die(die_id)))
        for die_id in range(1, n_dies):
            topo.add_link(InterposerLink.from_tech(0, die_id, tech))
            topo.add_link(InterposerLink.from_tech(die_id, 0, tech))
        return topo

    def get_links_from(self, die_id: int) -> list[InterposerLink]:
        """Return all directed links originating at ``die_id``."""
        return [link for link in self.links if link.src_die == die_id]

    def get_links_to(self, die_id: int) -> list[InterposerLink]:
        """Return all directed links terminating at ``die_id``."""
        return [link for link in self.links if link.dst_die == die_id]

    def get_die(self, die_id: int) -> ChipletDie | None:
        """Return the die with ``die_id``, or ``None`` when it is absent."""
        return next((die for die in self.dies if die.die_id == die_id), None)

    @property
    def num_dies(self) -> int:
        """Return the number of dies registered in the topology."""
        return len(self.dies)


def _require_grid(rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be > 0")


def _require_die_count(n_dies: int) -> None:
    if n_dies <= 0:
        raise ValueError("n_dies must be > 0")


def make_torus(
    rows: int,
    cols: int,
    tech: InterposerTech = InterposerTech.UCIE,
) -> ChipletTopology:
    """Construct a rectangular torus with right and downward wrap-around links."""
    _require_grid(rows, cols)
    topo = ChipletTopology()
    for r in range(rows):
        for c in range(cols):
            die_id = r * cols + c
            topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=_seed_for_die(die_id)))
    for r in range(rows):
        for c in range(cols):
            src = r * cols + c
            right = r * cols + (c + 1) % cols
            down = ((r + 1) % rows) * cols + c
            topo.add_link(InterposerLink.from_tech(src, right, tech))
            topo.add_link(InterposerLink.from_tech(src, down, tech))
    return topo


class StackingType(Enum):
    """Supported die-stacking geometries."""

    COPLANAR = "coplanar"
    TSV_3D = "tsv_3d"
    HYBRID_BONDING = "hybrid_bonding"


@dataclass
class TSVLink:
    """Describe the physical geometry of a through-silicon-via link."""

    src_die: int
    dst_die: int
    stacking: StackingType = StackingType.TSV_3D
    tsv_pitch_um: float = 10.0
    tsv_count: int = 1024
    latency_ps: float = 50.0

    @property
    def latency_ns(self) -> float:
        """Return TSV latency in nanoseconds."""
        return self.latency_ps / 1000.0

    @property
    def bandwidth_gbps(self) -> float:
        """Return aggregate bandwidth at one bit per TSV and 200 MHz."""
        return self.tsv_count * 200e6 / 1e9


_STACKING_PRESETS: dict[StackingType, tuple[float, float, float]] = {
    StackingType.TSV_3D: (0.05, 256.0, 1e-18),
    StackingType.HYBRID_BONDING: (0.01, 512.0, 1e-20),
    StackingType.COPLANAR: (2.0, 32.0, 1e-15),
}


def add_3d_stack(
    topology: ChipletTopology,
    bottom_die: int,
    top_die: int,
    stacking: StackingType = StackingType.TSV_3D,
) -> InterposerLink:
    """Add reciprocal links between vertically associated dies.

    Returns
    -------
    InterposerLink
        The bottom-to-top link. The reciprocal link is also added to ``topology``.
    """
    latency_ns, bandwidth_gbps, bit_error_rate = _STACKING_PRESETS[stacking]
    link = InterposerLink(
        src_die=bottom_die,
        dst_die=top_die,
        technology=InterposerTech.CUSTOM,
        latency_ns=latency_ns,
        bandwidth_gbps=bandwidth_gbps,
        bit_error_rate=bit_error_rate,
        is_bidirectional=True,
    )
    reverse = InterposerLink(
        src_die=top_die,
        dst_die=bottom_die,
        technology=InterposerTech.CUSTOM,
        latency_ns=latency_ns,
        bandwidth_gbps=bandwidth_gbps,
        bit_error_rate=bit_error_rate,
        is_bidirectional=True,
    )
    topology.add_link(link)
    topology.add_link(reverse)
    return link


__all__ = [
    "ChipletDie",
    "ChipletTopology",
    "InterposerLink",
    "InterposerTech",
    "StackingType",
    "TSVLink",
    "add_3d_stack",
    "make_torus",
]
