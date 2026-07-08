# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.spintronic -- Magnetic-domain stochastic computing mapper

"""sc_neurocore.spintronic -- Magnetic-domain stochastic computing mapper.

Tier: research.
"""

from .spintronic_mapper import (
    AgingModel,
    DefectEntry,
    DefectMap,
    MLCConfig,
    MappingResult,
    MaterialParams,
    MuMax3OutputParser,
    MuMax3Result,
    MuMax3ScriptGenerator,
    RacetrackShiftRegister,
    RadiationModel,
    SkyrmionHallCorrector,
    SpintronicArray,
    SpintronicCell,
    SpintronicDeviceConfig,
    SpintronicMapper,
    SpintronicTech,
    SpintronicVerilogGenerator,
    VariabilityModel,
    WriteVerifyResult,
    retention_failure_probability,
    switching_current_vs_temperature,
    switching_time_vs_temperature,
    write_verify,
)

__tier__ = "research"

__all__ = [
    "AgingModel",
    "DefectEntry",
    "DefectMap",
    "MLCConfig",
    "MappingResult",
    "MaterialParams",
    "MuMax3OutputParser",
    "MuMax3Result",
    "MuMax3ScriptGenerator",
    "RacetrackShiftRegister",
    "RadiationModel",
    "SkyrmionHallCorrector",
    "SpintronicArray",
    "SpintronicCell",
    "SpintronicDeviceConfig",
    "SpintronicMapper",
    "SpintronicTech",
    "SpintronicVerilogGenerator",
    "VariabilityModel",
    "WriteVerifyResult",
    "retention_failure_probability",
    "switching_current_vs_temperature",
    "switching_time_vs_temperature",
    "write_verify",
]
