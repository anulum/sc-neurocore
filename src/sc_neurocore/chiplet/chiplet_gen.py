# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical chiplet generator compatibility facade

"""Preserve the original chiplet-generator API over responsibility modules.

New code may import from ``sc_neurocore.chiplet`` or the focused modules. This
facade keeps historical imports and pickle-qualified names stable.
"""

from __future__ import annotations

from sc_neurocore.chiplet.link_protocols import (
    CDCConfig,
    CreditConfig,
    LinkProtection,
    compute_cdc_configs,
    emit_crc32_sv,
    emit_credit_controller_sv,
)
from sc_neurocore.chiplet.partition import PartitionAssignment
from sc_neurocore.chiplet.power import PowerDomain, PowerDomainMap, emit_power_gating_sv
from sc_neurocore.chiplet.routing import (
    CongestionReport,
    PackageEnergyReport,
    RoutingEntry,
    RoutingTable,
    TimingSimResult,
    adaptive_route,
    bandwidth_aware_route,
    compute_decorrelation_seeds,
    estimate_congestion,
    estimate_package_energy,
    find_disjoint_paths,
    link_energy_pj,
    simulate_timing,
)
from sc_neurocore.chiplet.rtl import ChipletGenerator, ChipletOutput
from sc_neurocore.chiplet.thermal import DieThermal, PackageThermalReport, simulate_thermal
from sc_neurocore.chiplet.topology import (
    ChipletDie,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    StackingType,
    TSVLink,
    add_3d_stack,
    make_torus,
)


__all__ = [
    "CDCConfig",
    "ChipletDie",
    "ChipletGenerator",
    "ChipletOutput",
    "ChipletTopology",
    "CongestionReport",
    "CreditConfig",
    "DieThermal",
    "InterposerLink",
    "InterposerTech",
    "LinkProtection",
    "PackageEnergyReport",
    "PackageThermalReport",
    "PartitionAssignment",
    "PowerDomain",
    "PowerDomainMap",
    "RoutingEntry",
    "RoutingTable",
    "StackingType",
    "TimingSimResult",
    "TSVLink",
    "adaptive_route",
    "add_3d_stack",
    "bandwidth_aware_route",
    "compute_cdc_configs",
    "compute_decorrelation_seeds",
    "emit_credit_controller_sv",
    "emit_crc32_sv",
    "emit_power_gating_sv",
    "estimate_congestion",
    "estimate_package_energy",
    "find_disjoint_paths",
    "link_energy_pj",
    "make_torus",
    "simulate_thermal",
    "simulate_timing",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__

del _public_name
