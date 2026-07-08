# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.memristor -- Memristor crossbar mapping and aging simulation

"""sc_neurocore.memristor -- Memristor crossbar mapping and aging simulation.

Tier: research.
"""

from .memristor_mapper import (
    AgingReport,
    AgingSimulator,
    CompensationLUT,
    CompensationStrategy,
    ConductanceModel,
    CrossbarArray,
    CrossbarEstimator,
    CrossbarMapping,
    CrossbarPowerEstimate,
    CrossbarTopology,
    IRDropModel,
    MappingResult,
    MemristorMapper,
    MemristorTechnology,
    MonteCarloReport,
    MonteCarloSimulator,
    SCAbsorbEncoder,
    SneakPathModel,
    StuckFaultMap,
    VariabilityInjector,
    VerilogEmitter,
    WriteVerifyProtocol,
    WriteVerifyResult,
)

__tier__ = "research"

__all__ = [
    "AgingReport",
    "AgingSimulator",
    "CompensationLUT",
    "CompensationStrategy",
    "ConductanceModel",
    "CrossbarArray",
    "CrossbarEstimator",
    "CrossbarMapping",
    "CrossbarPowerEstimate",
    "CrossbarTopology",
    "IRDropModel",
    "MappingResult",
    "MemristorMapper",
    "MemristorTechnology",
    "MonteCarloReport",
    "MonteCarloSimulator",
    "SCAbsorbEncoder",
    "SneakPathModel",
    "StuckFaultMap",
    "VariabilityInjector",
    "VerilogEmitter",
    "WriteVerifyProtocol",
    "WriteVerifyResult",
]
