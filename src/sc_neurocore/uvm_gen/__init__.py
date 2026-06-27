# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.uvm_gen -- UVM testbench generator for SC neuromorphic IP

"""Package facade for the UVM/SystemVerilog verification-IP generator.

Tier: industrial.
"""

from sc_neurocore.uvm_gen.uvm_gen import (
    CoverageSpec,
    FormalLink,
    ModuleParam,
    ModulePort,
    PortDirection,
    PortType,
    RTLModule,
    SIM_TARGETS,
    ScoreboardConfig,
    SimTarget,
    StimulusConfig,
    UVMBenchmark,
    UVMGenerator,
)

__tier__ = "industrial"

__all__ = [
    "CoverageSpec",
    "FormalLink",
    "ModuleParam",
    "ModulePort",
    "PortDirection",
    "PortType",
    "RTLModule",
    "SIM_TARGETS",
    "ScoreboardConfig",
    "SimTarget",
    "StimulusConfig",
    "UVMBenchmark",
    "UVMGenerator",
]
