# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Declarative Network Simulation Engine

"""Declarative network simulation engine for SC-NeuroCore."""

from __future__ import annotations

from .population import Population
from .projection import Projection
from .monitor import SpikeMonitor, StateMonitor, RateMonitor
from .network import Network
from .stimulus import TimedArray, PoissonInput, StepCurrent
from .topology import (
    random_connectivity,
    small_world,
    scale_free,
    ring_topology,
    grid_topology,
    all_to_all,
)
from .export import export_verilog
from .mpi_runner import HAS_MPI, MPIRunner
from .cortical_column import CorticalColumn
from .gamma_oscillation import PINGCircuit
from .sc_compte_wm import (
    SCCompteCellSpec,
    SCCompteProtocolSpec,
    SCCompteWMActivityStatistics,
    SCCompteWMNetworkSpec,
    circular_displacement_deg,
    circular_distance_deg,
    summarize_activity,
)
from .sc_compte_wm_drive import CounterPoissonDrive, CounterPoissonReceipt
from .sc_compte_wm_network import (
    SCCompteWMNetwork,
    SCCompteWMNetworkState,
    SCCompteWMRunReceipt,
    SCCompteWMStepReceipt,
    SCCompteWMStimulus,
    SCCompteWMWindowReceipt,
)

__all__ = [
    "Population",
    "Projection",
    "SpikeMonitor",
    "StateMonitor",
    "RateMonitor",
    "Network",
    "TimedArray",
    "PoissonInput",
    "StepCurrent",
    "random_connectivity",
    "small_world",
    "scale_free",
    "ring_topology",
    "grid_topology",
    "all_to_all",
    "export_verilog",
    "MPIRunner",
    "HAS_MPI",
    "CorticalColumn",
    "PINGCircuit",
    "SCCompteCellSpec",
    "SCCompteProtocolSpec",
    "SCCompteWMActivityStatistics",
    "SCCompteWMNetworkSpec",
    "circular_displacement_deg",
    "circular_distance_deg",
    "summarize_activity",
    "CounterPoissonDrive",
    "CounterPoissonReceipt",
    "SCCompteWMNetwork",
    "SCCompteWMNetworkState",
    "SCCompteWMRunReceipt",
    "SCCompteWMStepReceipt",
    "SCCompteWMStimulus",
    "SCCompteWMWindowReceipt",
]
