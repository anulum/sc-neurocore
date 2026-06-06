# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CXL coherence advisor

"""CXL.mem Type-3 device mapping for large-scale SNN state expansion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CXLMapping:
    """CXL.mem Type-3 mapping for neuron state.

    Attributes
    ----------
    device_count : int
        Number of CXL memory devices.
    state_device_ids : list[int]
        Devices hosting neuron state.
    weight_device_ids : list[int]
        Devices hosting synaptic weights.
    total_capacity_gb : float
        Total CXL memory capacity.
    host_bandwidth_gbps : float
        Required host→CXL bandwidth.
    coherence_protocol : str
        CXL protocol used (``"CXL.mem"`` or ``"CXL.cache"``).
    """

    device_count: int
    state_device_ids: list[int]
    weight_device_ids: list[int]
    total_capacity_gb: float
    host_bandwidth_gbps: float
    coherence_protocol: str


def advise_cxl_mapping(
    neuron_count: int,
    synapse_count: int,
    *,
    data_width: int = 16,
    device_capacity_gb: float = 16.0,
    max_devices: int = 8,
    access_pattern: str = "streaming",
) -> CXLMapping:
    """Advise on CXL.mem Type-3 device mapping for neuron state.

    Plans the distribution of neuron state and synaptic weights
    across CXL 3.0 Type-3 memory expander devices.

    Parameters
    ----------
    neuron_count : int
        Total neurons.
    synapse_count : int
        Total synaptic connections.
    data_width : int
        Bits per value.
    device_capacity_gb : float
        Capacity per CXL device (GB).
    max_devices : int
        Maximum CXL devices available.
    access_pattern : str
        ``"streaming"`` (sequential) or ``"random"`` (scattered).

    Returns
    -------
    CXLMapping
    """
    bytes_per_val = max(1, data_width // 8)
    state_bytes = neuron_count * bytes_per_val * 4  # 4 state vars avg
    weight_bytes = synapse_count * bytes_per_val
    total_bytes = state_bytes + weight_bytes
    total_gb = total_bytes / (1024**3)

    devices_needed = max(1, int(-(-total_gb // device_capacity_gb)))
    devices_used = min(devices_needed, max_devices)

    # Split: state on first devices, weights on remaining
    state_devs = max(1, int(devices_used * state_bytes / total_bytes))
    weight_devs = max(1, devices_used - state_devs)

    state_device_ids = list(range(state_devs))
    weight_device_ids = list(range(state_devs, state_devs + weight_devs))

    # Bandwidth estimation
    bw_factor = 1.0 if access_pattern == "streaming" else 2.5
    update_rate_hz = 1000
    bytes_per_update = total_bytes * 0.1
    raw_bw = bytes_per_update * update_rate_hz * 8 / 1e9
    required_gbps = round(raw_bw * bw_factor, 4)

    # Protocol selection
    protocol = "CXL.cache" if access_pattern == "random" else "CXL.mem"

    return CXLMapping(
        device_count=devices_used,
        state_device_ids=state_device_ids,
        weight_device_ids=weight_device_ids,
        total_capacity_gb=round(devices_used * device_capacity_gb, 2),
        host_bandwidth_gbps=required_gbps,
        coherence_protocol=protocol,
    )
