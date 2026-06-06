# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PIM layout planner

"""Memory layout planning for Processing-in-Memory (PIM) targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIMLayout:
    """Memory layout plan for Processing-in-Memory targets.

    Attributes
    ----------
    bank_count : int
        Number of memory banks used.
    neurons_per_bank : int
        Neurons assigned per bank.
    weights_per_bank : int
        Weight entries per bank.
    bank_utilisation : float
        Fraction of bank capacity used (0.0–1.0).
    parallel_factor : int
        Number of banks that can compute in parallel.
    layout_map : dict[str, list[int]]
        Mapping of data regions to bank IDs.
    """

    bank_count: int
    neurons_per_bank: int
    weights_per_bank: int
    bank_utilisation: float
    parallel_factor: int
    layout_map: dict[str, list[int]]


def plan_pim_layout(
    neuron_count: int,
    synapse_count: int,
    *,
    data_width: int = 16,
    bank_size_kb: int = 64,
    num_banks: int = 16,
    target: str = "upmem_pim",
) -> PIMLayout:
    """Plan data placement across PIM memory banks.

    Distributes neuron state and synaptic weights across memory banks.

    Parameters
    ----------
    neuron_count : int
        Total neurons in the network.
    synapse_count : int
        Total synaptic connections.
    data_width : int
        Bits per value.
    bank_size_kb : int
        Capacity of each memory bank in KB.
    num_banks : int
        Number of available memory banks.
    target : str
        Target platform name.

    Returns
    -------
    PIMLayout
    """
    bytes_per_val = max(1, data_width // 8)
    neuron_bytes = neuron_count * bytes_per_val
    weight_bytes = synapse_count * bytes_per_val
    total_bytes = neuron_bytes + weight_bytes

    bank_bytes = bank_size_kb * 1024
    banks_needed = max(1, -(-total_bytes // bank_bytes))  # ceil div
    banks_used = min(banks_needed, num_banks)

    neurons_per_bank = max(1, -(-neuron_count // banks_used))
    weights_per_bank = max(1, -(-synapse_count // banks_used))

    used_bytes_per_bank = neurons_per_bank * bytes_per_val + weights_per_bank * bytes_per_val
    utilisation = min(1.0, used_bytes_per_bank / bank_bytes)

    # Layout: first half for neuron state, second half for weights
    state_banks = list(range(0, banks_used // 2 or 1))
    weight_banks = list(range(banks_used // 2 or 1, banks_used))
    if not weight_banks:
        weight_banks = state_banks  # Small networks share banks

    return PIMLayout(
        bank_count=banks_used,
        neurons_per_bank=neurons_per_bank,
        weights_per_bank=weights_per_bank,
        bank_utilisation=round(utilisation, 4),
        parallel_factor=banks_used,
        layout_map={
            "neuron_state": state_banks,
            "synaptic_weights": weight_banks,
        },
    )
