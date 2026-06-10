# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UCIe protocol mapper

"""UCIe die-to-die protocol mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UCIeMapping:
    """UCIe die-to-die protocol mapping result.

    Attributes
    ----------
    lanes : dict[str, int]
    protocol_version : str
    total_bandwidth_gbps : float
    """

    lanes: dict[str, int]
    protocol_version: str
    total_bandwidth_gbps: float


def map_ucie_protocol(
    blocks: dict[str, int],
    *,
    lane_bandwidth_gbps: float = 32.0,
    protocol_version: str = "UCIe 2.0",
) -> UCIeMapping:
    """Map neuron array blocks to UCIe die-to-die protocol lanes.

    Parameters
    ----------
    blocks : dict[str, int]
        Block name → data width in bits per cycle.
    lane_bandwidth_gbps : float
        Bandwidth per UCIe lane.
    protocol_version : str
        UCIe protocol version.

    Returns
    -------
    UCIeMapping
    """
    lanes = {}
    total_bw = 0.0
    for block, width_bits in blocks.items():
        # Each lane carries lane_bandwidth_gbps
        needed_lanes = max(1, (width_bits + 31) // 32)
        lanes[block] = needed_lanes
        total_bw += needed_lanes * lane_bandwidth_gbps

    return UCIeMapping(
        lanes=lanes,
        protocol_version=protocol_version,
        total_bandwidth_gbps=total_bw,
    )
