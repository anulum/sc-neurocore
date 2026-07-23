# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor tenant domain

"""Define tenant priority, QoS, checkpoint, and runtime state contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class TenantPriority(Enum):
    REALTIME = 0
    HIGH = 1
    NORMAL = 2
    BEST_EFFORT = 3


@dataclass
class QoSPolicy:
    """Quality-of-Service policy for a tenant."""

    max_bandwidth_mbps: float = 100.0
    max_latency_us: float = 1000.0
    min_compute_share: float = 0.1
    max_neurons: int = 1024
    max_synapses: int = 16384
    preemptible: bool = True


@dataclass
class TenantState:
    """Checkpointable state for live migration."""

    neuron_voltages: Optional[np.ndarray[Any, Any]] = None
    synapse_weights: Optional[np.ndarray[Any, Any]] = None
    spike_queues: Optional[np.ndarray[Any, Any]] = None
    lfsr_state: int = 0
    timestep: int = 0
    checksum: str = ""

    def compute_checksum(self) -> str:
        h = hashlib.sha256()
        if self.neuron_voltages is not None:
            h.update(self.neuron_voltages.tobytes())
        if self.synapse_weights is not None:
            h.update(self.synapse_weights.tobytes())
        if self.spike_queues is not None:
            h.update(self.spike_queues.tobytes())
        h.update(self.lfsr_state.to_bytes(4, "little"))
        h.update(self.timestep.to_bytes(4, "little"))
        self.checksum = h.hexdigest()[:16]
        return self.checksum


@dataclass
class Tenant:
    """One SC network tenant on the hypervisor."""

    tenant_id: str
    name: str
    priority: TenantPriority = TenantPriority.NORMAL
    qos: QoSPolicy = field(default_factory=QoSPolicy)
    region_id: Optional[int] = None
    state: Optional[TenantState] = None
    active: bool = False
    total_spikes: int = 0
    total_cycles: int = 0
    created_ns: int = 0
    last_scheduled_ns: int = 0
