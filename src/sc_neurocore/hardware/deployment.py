# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware Deployment Package

"""Package a mapped network for deployment to neuromorphic hardware.

Generates a self-contained ``DeploymentPackage`` with configuration
blobs, placement tables, and validation metadata.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .device import DeviceSpec
from .mapping import NeuronPlacement


@dataclass
class DeploymentPackage:
    """Self-contained deployment artifact for neuromorphic hardware.

    Attributes:
        device: Target device specification.
        placements: Neuron-to-core mapping.
        config_blob: Binary configuration data for the target.
        metadata: Additional deployment metadata.
    """

    device: DeviceSpec
    placements: list[NeuronPlacement]
    config_blob: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


class Deployer:
    """Create and validate deployment packages."""

    def package(
        self,
        adjacency: np.ndarray[Any, Any],
        device: DeviceSpec,
        placements: list[NeuronPlacement],
        weights: np.ndarray[Any, Any] | None = None,
    ) -> DeploymentPackage:
        """Create a deployment package.

        Parameters:
            adjacency: (N, N) network connectivity matrix.
            device: Target device.
            placements: Neuron-to-core mapping.
            weights: Optional weight matrix (defaults to adjacency values).

        Returns:
            DeploymentPackage ready for deployment.
        """
        w = weights if weights is not None else adjacency
        config = self._build_config(w, placements, device)
        n_neurons = adjacency.shape[0]
        n_synapses = int(np.count_nonzero(adjacency))
        n_cores = max(p.core_id for p in placements) + 1

        metadata = {
            "n_neurons": n_neurons,
            "n_synapses": n_synapses,
            "n_cores_used": n_cores,
            "device_family": device.family.name,
            "weight_bits": device.weight_bits,
            "fits": n_cores <= device.cores,
        }

        return DeploymentPackage(
            device=device,
            placements=placements,
            config_blob=config,
            metadata=metadata,
        )

    def validate(self, package: DeploymentPackage) -> bool:
        """Validate a deployment package for consistency.

        Checks:
        - All neuron IDs are unique
        - No core_id exceeds device capacity
        - Config blob is non-empty
        - All local IDs are within neurons_per_core
        """
        if not package.config_blob:
            return False

        neuron_ids = [p.neuron_id for p in package.placements]
        if len(set(neuron_ids)) != len(neuron_ids):
            return False  # duplicate neuron placement

        for p in package.placements:
            if p.core_id >= package.device.cores:
                return False
            if p.local_id >= package.device.neurons_per_core:
                return False

        return True

    def summary(self, package: DeploymentPackage) -> str:
        """Human-readable deployment summary."""
        m = package.metadata
        lines = [
            f"=== Deployment Summary ===",
            f"Device:      {m.get('device_family', 'unknown')}",
            f"Neurons:     {m.get('n_neurons', 0)}",
            f"Synapses:    {m.get('n_synapses', 0)}",
            f"Cores used:  {m.get('n_cores_used', 0)} / {package.device.cores}",
            f"Fits:        {'Yes' if m.get('fits') else 'No'}",
            f"Config size: {len(package.config_blob)} bytes",
            f"Weight bits: {m.get('weight_bits', 0)}",
        ]
        return "\n".join(lines)

    def _build_config(
        self,
        weights: np.ndarray[Any, Any],
        placements: list[NeuronPlacement],
        device: DeviceSpec,
    ) -> bytes:
        """Build binary configuration blob.

        Format: header + per-core neuron count + per-synapse (src, tgt, weight).
        """
        n = weights.shape[0]
        # Quantize weights to target precision
        w_max = 2 ** (device.weight_bits - 1) - 1
        abs_max = np.max(np.abs(weights))
        if abs_max > 0:
            scale = w_max / abs_max
        else:
            scale = 1.0

        # Header: magic + n_neurons + n_synapses
        buf = bytearray()
        buf.extend(struct.pack(">4sII", b"SCNC", n, int(np.count_nonzero(weights))))

        # Placement table
        for p in placements:
            buf.extend(struct.pack(">IHH", p.neuron_id, p.core_id, p.local_id))

        # Synapse table
        rows, cols = np.nonzero(weights)
        for r, c in zip(rows, cols):
            qw = int(np.round(weights[r, c] * scale))
            qw = max(-w_max, min(w_max, qw))
            buf.extend(struct.pack(">IIh", int(r), int(c), qw))

        return bytes(buf)
