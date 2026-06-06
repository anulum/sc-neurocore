# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UCIe partitioning advisor

"""Neuron array partitioning for chiplet-based multi-die systems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UCIePartition:
    """Partitioning plan for a neuron array across chiplet tiles.

    Attributes
    ----------
    tile_count : int
        Number of chiplet tiles used.
    neurons_per_tile : int
        Neurons assigned per tile.
    inter_tile_spikes : int
        Estimated spikes crossing tile boundaries per timestep.
    die_to_die_bandwidth_gbps : float
        Required UCIe bandwidth (Gbps).
    latency_penalty_ns : float
        Additional latency from die-to-die communication.
    partition_map : dict[int, list[int]]
        Tile ID → list of neuron indices.
    """

    tile_count: int
    neurons_per_tile: int
    inter_tile_spikes: int
    die_to_die_bandwidth_gbps: float
    latency_penalty_ns: float
    partition_map: dict[int, list[int]]


def advise_ucie_partition(
    neuron_count: int,
    connectivity: float = 0.1,
    *,
    tile_count: int = 4,
    spike_rate_hz: float = 10.0,
    timestep_us: float = 1000.0,
    ucie_lane_gbps: float = 32.0,
    ucie_latency_ns: float = 2.0,
) -> UCIePartition:
    """Advise on neuron array partitioning across chiplet tiles.

    Analyses a neuron array's connectivity to estimate inter-tile
    spike traffic and UCIe bandwidth requirements.

    Parameters
    ----------
    neuron_count : int
        Total neurons in the network.
    connectivity : float
        Connection probability between any two neurons (0.0–1.0).
    tile_count : int
        Number of chiplet tiles.
    spike_rate_hz : float
        Average firing rate per neuron (Hz).
    timestep_us : float
        Simulation timestep (µs).
    ucie_lane_gbps : float
        UCIe lane bandwidth (Gbps per lane).
    ucie_latency_ns : float
        UCIe die-to-die latency (ns).

    Returns
    -------
    UCIePartition
    """
    neurons_per_tile = max(1, -(-neuron_count // tile_count))  # ceil

    # Estimate inter-tile spikes per timestep
    intra_tile_frac = 1.0 / tile_count
    inter_tile_frac = 1.0 - intra_tile_frac

    spikes_per_timestep = neuron_count * spike_rate_hz * (timestep_us / 1e6)
    inter_tile_spikes = int(spikes_per_timestep * inter_tile_frac)

    # Bandwidth: each spike = ~8 bytes (neuron ID + timestamp)
    bytes_per_spike = 8
    bytes_per_timestep = inter_tile_spikes * bytes_per_spike
    bits_per_second = bytes_per_timestep * 8 / (timestep_us * 1e-6)
    required_gbps = round(bits_per_second / 1e9, 4)

    # Latency penalty from die-to-die crossing
    latency_ns = ucie_latency_ns * (tile_count - 1)  # worst-case path

    # Simple round-robin partition
    partition_map = {}
    for t in range(tile_count):
        start = t * neurons_per_tile
        end = min(start + neurons_per_tile, neuron_count)
        partition_map[t] = list(range(start, end))

    return UCIePartition(
        tile_count=tile_count,
        neurons_per_tile=neurons_per_tile,
        inter_tile_spikes=inter_tile_spikes,
        die_to_die_bandwidth_gbps=required_gbps,
        latency_penalty_ns=round(latency_ns, 2),
        partition_map=partition_map,
    )
