# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spintronic_mapper.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.spintronic.spintronic_mapper import (
    AgingModel,
    DefectMap,
    MLCConfig,
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
    retention_failure_probability,
    switching_current_vs_temperature,
    switching_time_vs_temperature,
    write_verify,
)

__all__ = [
    "np",
    "pytest",
    "AgingModel",
    "DefectMap",
    "MLCConfig",
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
    "retention_failure_probability",
    "switching_current_vs_temperature",
    "switching_time_vs_temperature",
    "write_verify",
]
