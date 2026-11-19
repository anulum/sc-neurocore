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

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .device import DeviceSpec
from .mapping import NeuronPlacement

#: Width of the signed weight code in each synapse entry of the configuration blob.
SYNAPSE_WEIGHT_FIELD_BITS = 16
#: Unsigned widths of the placement entry's fields: neuron id, core id, local id.
PLACEMENT_FIELD_BITS = {"neuron_id": 32, "core_id": 16, "local_id": 16}


@dataclass
class DeploymentPackage:
    """Self-contained deployment artifact for neuromorphic hardware.

    Attributes
    ----------
    device : DeviceSpec
        Target device specification.
    placements : list of NeuronPlacement
        Neuron-to-core mapping.
    config_blob : bytes
        Binary configuration data for the target.
    metadata : dict
        Additional deployment metadata. ``weight_scale`` turns the blob's integer weight codes back into weights:
        weight = code / weight_scale.
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

        Parameters
        ----------
        adjacency : ndarray of shape (N, N)
            Network connectivity matrix.
        device : DeviceSpec
            Target device.
        placements : list of NeuronPlacement
            Neuron-to-core mapping; an empty list uses no core.
        weights : ndarray, optional
            Weight matrix (defaults to adjacency values).

        Returns
        -------
        DeploymentPackage
            Package ready for deployment, its metadata holding the weight scale.

        Raises
        ------
        ValueError
            If the device's weight precision is below 1 bit or above the blob's 16-bit weight field, a weight is
            not finite, or a placement id is negative or too wide for its field (32-bit neuron id, 16-bit core
            and local ids).
        """
        w = weights if weights is not None else adjacency
        config, weight_scale = self._build_config(w, placements, device)
        n_neurons = adjacency.shape[0]
        n_synapses = int(np.count_nonzero(adjacency))
        n_cores = max((p.core_id for p in placements), default=-1) + 1

        metadata = {
            "n_neurons": n_neurons,
            "n_synapses": n_synapses,
            "n_cores_used": n_cores,
            "device_family": device.family.name,
            "weight_bits": device.weight_bits,
            "weight_scale": weight_scale,
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
        - All neuron IDs are unique and non-negative
        - Every core_id lies in [0, device cores)
        - Config blob is non-empty
        - Every local_id lies in [0, neurons_per_core)
        """
        if not package.config_blob:
            return False

        neuron_ids = [p.neuron_id for p in package.placements]
        if len(set(neuron_ids)) != len(neuron_ids):
            return False  # duplicate neuron placement

        for p in package.placements:
            if p.neuron_id < 0:
                return False
            if not 0 <= p.core_id < package.device.cores:
                return False
            if not 0 <= p.local_id < package.device.neurons_per_core:
                return False

        return True

    def summary(self, package: DeploymentPackage) -> str:
        """Human-readable deployment summary."""
        m = package.metadata
        lines = [
            "=== Deployment Summary ===",
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
    ) -> tuple[bytes, float]:
        """Build binary configuration blob and return it with its weight scale.

        Format: header + per-core neuron count + per-synapse (src, tgt, weight code), each code a signed 16-bit
        integer, ``round(weight * scale)``. With ``b >= 2`` bits the codes span ``2**(b-1) - 1`` levels a side and
        ``scale = (2**(b-1) - 1) / max|w|``. One bit codes only the sign: ``+1`` or ``-1``, with ``scale`` the
        inverse of the mean non-zero magnitude, the level that minimises the squared error for a fixed sign pattern.
        """
        bits = device.weight_bits
        if not 1 <= bits <= SYNAPSE_WEIGHT_FIELD_BITS:
            raise ValueError(
                f"weight_bits={bits} does not fit the blob's signed {SYNAPSE_WEIGHT_FIELD_BITS}-bit weight field; "
                f"it must lie in [1, {SYNAPSE_WEIGHT_FIELD_BITS}]"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite to be coded (NaN or infinity found)")
        for p in placements:
            for name, width in PLACEMENT_FIELD_BITS.items():
                value = getattr(p, name)
                if not 0 <= value < 2**width:
                    raise ValueError(
                        f"placement {name}={value} does not fit the blob's unsigned {width}-bit field"
                    )
        n = weights.shape[0]
        rows, cols = np.nonzero(weights)
        values = np.asarray(weights[rows, cols], dtype=float)
        magnitudes = np.abs(values)
        if bits == 1:
            level = float(magnitudes.mean()) if values.size else 0.0
            scale = 1.0 / level if level > 0 else 1.0
            codes = np.where(values >= 0, 1, -1)
        else:
            w_max = 2 ** (bits - 1) - 1
            abs_max = float(magnitudes.max()) if values.size else 0.0
            scale = w_max / abs_max if abs_max > 0 else 1.0
            codes = np.clip(np.round(values * scale), -w_max, w_max).astype(int)

        # Header: magic + n_neurons + n_synapses
        buf = bytearray()
        buf.extend(struct.pack(">4sII", b"SCNC", n, int(np.count_nonzero(weights))))

        # Placement table
        for p in placements:
            buf.extend(struct.pack(">IHH", p.neuron_id, p.core_id, p.local_id))

        # Synapse table
        for r, c, code in zip(rows, cols, codes):
            buf.extend(struct.pack(">IIh", int(r), int(c), int(code)))

        return bytes(buf), scale
