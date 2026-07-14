# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical UVM/SystemVerilog verification-IP generator facade

"""Expose the stable UVM generator API over focused responsibility modules.

RTL parsing, configuration, generated-artifact contracts, UVM component
emission, simulator/formal harness emission, and orchestration live in private
modules. Historical imports and pickle-qualified names remain stable here.
"""

from __future__ import annotations

from sc_neurocore.uvm_gen._benchmark import UVMBenchmark as UVMBenchmark
from sc_neurocore.uvm_gen._config import (
    CoverageSpec as CoverageSpec,
    FormalLink as FormalLink,
    ScoreboardConfig as ScoreboardConfig,
    SIM_TARGETS as SIM_TARGETS,
    SimTarget as SimTarget,
    StimulusConfig as StimulusConfig,
)
from sc_neurocore.uvm_gen._generator import UVMGenerator as UVMGenerator
from sc_neurocore.uvm_gen._rtl import (
    ModuleParam as ModuleParam,
    ModulePort as ModulePort,
    PortDirection as PortDirection,
    PortType as PortType,
    RTLModule as RTLModule,
)

_HISTORICAL_DEFINITIONS = (
    CoverageSpec,
    FormalLink,
    ModuleParam,
    ModulePort,
    PortDirection,
    PortType,
    RTLModule,
    ScoreboardConfig,
    SimTarget,
    StimulusConfig,
    UVMBenchmark,
    UVMGenerator,
)

for _definition in _HISTORICAL_DEFINITIONS:
    _definition.__module__ = __name__

del _definition
del _HISTORICAL_DEFINITIONS
