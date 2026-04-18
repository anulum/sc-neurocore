# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware Resource Estimator

"""Estimate neuromorphic hardware resources needed for a given network.

Computes core count, utilization, power, and latency for mapping an
SC-NeuroCore network onto a specific ``DeviceSpec`` target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .device import DeviceSpec


@dataclass
class ResourceEstimate:
    """Hardware resource estimation result.

    Attributes:
        cores_needed: Minimum cores to host the network.
        neurons_mapped: Total neurons to place.
        synapses_mapped: Total synapses to route.
        utilization_pct: Average core utilization (%).
        power_mw: Estimated total power (mW).
        latency_us: Estimated single-tick latency (µs).
        fits: Whether the network fits on the target device.
    """
    cores_needed: int
    neurons_mapped: int
    synapses_mapped: int
    utilization_pct: float
    power_mw: float
    latency_us: float
    fits: bool


class ResourceEstimator:
    """Estimate hardware cost for deploying an SC-NeuroCore network."""

    def estimate(
        self,
        adjacency: np.ndarray[Any, Any],
        device: DeviceSpec,
    ) -> ResourceEstimate:
        """Estimate resources from an adjacency matrix.

        Parameters:
            adjacency: (N, N) weighted adjacency matrix.
            device: Target device specification.

        Returns:
            ResourceEstimate with core counts, power, etc.
        """
        n_neurons = adjacency.shape[0]
        n_synapses = int(np.count_nonzero(adjacency))

        # Core count from neuron packing
        cores_from_neurons = math.ceil(n_neurons / device.neurons_per_core)

        # Core count from synapse packing
        cores_from_synapses = math.ceil(n_synapses / device.synapses_per_core) if device.synapses_per_core > 0 else 1

        cores_needed = max(cores_from_neurons, cores_from_synapses, 1)

        # Utilization
        total_neuron_slots = cores_needed * device.neurons_per_core
        utilization = (n_neurons / total_neuron_slots * 100) if total_neuron_slots > 0 else 0.0

        # Power
        power = cores_needed * device.power_per_core_mw

        # Latency: one tick
        latency_us = device.tick_ns / 1000.0

        fits = cores_needed <= device.cores

        return ResourceEstimate(
            cores_needed=cores_needed,
            neurons_mapped=n_neurons,
            synapses_mapped=n_synapses,
            utilization_pct=round(utilization, 2),
            power_mw=round(power, 3),
            latency_us=latency_us,
            fits=fits,
        )

    def fits(
        self,
        adjacency: np.ndarray[Any, Any],
        device: DeviceSpec,
    ) -> bool:
        """Quick check: does the network fit on the device?"""
        return self.estimate(adjacency, device).fits

    def compare(
        self,
        adjacency: np.ndarray[Any, Any],
        devices: list[DeviceSpec],
    ) -> list[ResourceEstimate]:
        """Compare resource requirements across multiple devices."""
        return [self.estimate(adjacency, dev) for dev in devices]
