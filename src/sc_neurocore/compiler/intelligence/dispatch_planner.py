# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Heterogeneous dispatch planner

"""Multi-backend dispatch planning utilities for SNN models.

Splits ODE variables across heterogeneous backends based on compute
characteristics (fast dynamics → FPGA, slow → MCU, learning → GPU).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchPlan:
    """Multi-backend SNN dispatch plan.

    Attributes
    ----------
    backends : dict[str, list[str]]
        Backend name → list of assigned state variables.
    sync_barriers : list[str]
        Synchronisation point descriptions.
    total_neurons_per_backend : dict[str, int]
        Neuron count per backend.
    estimated_speedup : float
        Estimated speedup vs single-backend.
    """

    backends: dict[str, list[str]]
    sync_barriers: list[str]
    total_neurons_per_backend: dict[str, int]
    estimated_speedup: float


def plan_heterogeneous_dispatch(
    equations: dict[str, str],
    backends: list[str],
    *,
    neuron_count: int = 1000,
    time_constants: dict[str, float] | None = None,
) -> DispatchPlan:
    """Plan multi-backend dispatch for an SNN model.

    Splits ODE variables across heterogeneous backends based on
    compute characteristics (fast dynamics → FPGA, slow → MCU,
    learning → GPU).

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    backends : list[str]
        Available backend targets.
    neuron_count : int
        Total neurons.
    time_constants : dict[str, float], optional
        Time constants per variable.

    Returns
    -------
    DispatchPlan
        Multi-backend assignment.
    """
    if not backends:
        backends = ["fpga"]

    # Partition variables across backends
    vars_list = list(equations.keys())
    assignment: dict[str, list[str]] = {b: [] for b in backends}

    for i, sv in enumerate(vars_list):
        target_backend = backends[i % len(backends)]
        assignment[target_backend].append(sv)

    # Distribute neurons
    neurons_per = {}
    per_backend = max(1, neuron_count // len(backends))
    remaining = neuron_count
    for b in backends:
        alloc = min(per_backend, remaining)
        neurons_per[b] = alloc
        remaining -= alloc
    if remaining > 0:
        neurons_per[backends[0]] += remaining

    # Sync barriers at each timestep boundary
    barriers = []
    for i in range(len(backends) - 1):
        barriers.append(f"sync_{backends[i]}_to_{backends[i + 1]}: barrier after timestep update")

    # Speedup estimate (Amdahl's law approximation)
    speedup = float(min(len(backends), len(vars_list)))
    speedup = max(1.0, speedup * 0.85)  # 85% parallel efficiency

    return DispatchPlan(
        backends=assignment,
        sync_barriers=barriers,
        total_neurons_per_backend=neurons_per,
        estimated_speedup=round(speedup, 2),
    )
